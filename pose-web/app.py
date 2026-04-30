import logging
import os
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from functools import wraps
from threading import Lock

import cv2
from flask import Flask, flash, g, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix
from pose_engine import (
    create_landmarker,
    extract_joint_angles,
    extract_joint_angles_from_landmarks,
    compare_joint_scores,
    weighted_overall_score,
    build_feedback_message,
    draw_pose_overlay,
    draw_pose_overlay_from_landmarks,
    encode_png_frame,
    get_landmarks,
    normalize_landmarks_payload,
    to_mp_image,
)
from data.exercises import EXERCISES, BODY_PART_OPTIONS

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


APP_ENV = os.environ.get("APP_ENV", os.environ.get("FLASK_ENV", "development")).strip().lower()
IS_PRODUCTION = APP_ENV == "production"

if IS_PRODUCTION and not os.environ.get("SECRET_KEY"):
    raise RuntimeError("SECRET_KEY must be set when APP_ENV=production.")

app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=env_flag("SESSION_COOKIE_SECURE", IS_PRODUCTION),
    PERMANENT_SESSION_LIFETIME=timedelta(days=int(os.environ.get("SESSION_DAYS", "14"))),
    PREFERRED_URL_SCHEME="https" if IS_PRODUCTION else "http",
    SEND_FILE_MAX_AGE_DEFAULT=timedelta(days=30) if IS_PRODUCTION else timedelta(seconds=0),
)

STATIC_ASSET_VERSION = os.environ.get("STATIC_ASSET_VERSION") or os.environ.get("RENDER_GIT_COMMIT")
if not STATIC_ASSET_VERSION:
    training_core_path = os.path.join(app.root_path, "static", "js", "training-core.js")
    STATIC_ASSET_VERSION = str(int(os.path.getmtime(training_core_path))) if os.path.exists(training_core_path) else "1"

trusted_hosts = [host.strip() for host in os.environ.get("TRUSTED_HOSTS", "").split(",") if host.strip()]
if trusted_hosts:
    app.config["TRUSTED_HOSTS"] = trusted_hosts

if env_flag("USE_PROXY_FIX", IS_PRODUCTION):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(app.root_path, "pose_landmarker.task"))
DATABASE_PATH = os.environ.get("DATABASE_PATH", os.path.join(app.root_path, "instance", "app.db"))
database_dir = os.path.dirname(os.path.abspath(DATABASE_PATH))
if database_dir:
    os.makedirs(database_dir, exist_ok=True)

cap_video = None
landmarker = None
current_video_path = None
video_lock = Lock()


def safe_file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def looks_like_lfs_pointer(path):
    try:
        with open(path, "rb") as file_obj:
            header = file_obj.read(256)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def build_reference_debug(exercise, reference_time_sec):
    rel_video_path = exercise["video"]
    video_path = os.path.join(app.root_path, rel_video_path)
    model_exists = os.path.exists(MODEL_PATH)
    model_size = safe_file_size(MODEL_PATH)

    return {
        "exercise_id": exercise.get("id"),
        "exercise_name": exercise.get("name"),
        "reference_time": round(reference_time_sec, 3),
        "app_root": app.root_path,
        "cwd": os.getcwd(),
        "video_relative_path": rel_video_path,
        "video_path": video_path,
        "video_exists": os.path.exists(video_path),
        "video_size_bytes": safe_file_size(video_path),
        "video_is_lfs_pointer": looks_like_lfs_pointer(video_path) if os.path.exists(video_path) else False,
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "model_size_bytes": model_size,
        "model_is_lfs_pointer": looks_like_lfs_pointer(MODEL_PATH) if model_exists else False,
        "capture_opened": False,
        "capture_backend": None,
        "capture_fps": 0.0,
        "capture_frame_count": 0,
        "database_path": DATABASE_PATH,
        "render_service": os.environ.get("RENDER_SERVICE_NAME"),
        "render_git_commit": os.environ.get("RENDER_GIT_COMMIT"),
    }


def summarize_exception(exc, limit=8):
    return " | ".join(line.strip() for line in traceback.format_exception(type(exc), exc, exc.__traceback__, limit=limit))


def enrich_capture_debug(debug_info, video_capture):
    debug_info = dict(debug_info)
    if video_capture is None:
        return debug_info

    try:
        capture_opened = bool(video_capture.isOpened())
    except Exception as exc:
        debug_info["capture_state_exception"] = f"{type(exc).__name__}: {exc}"
        capture_opened = False

    debug_info["capture_opened"] = capture_opened

    if not capture_opened:
        return debug_info

    try:
        debug_info["capture_backend"] = video_capture.getBackendName() if hasattr(video_capture, "getBackendName") else None
    except Exception as exc:
        debug_info["capture_backend_exception"] = f"{type(exc).__name__}: {exc}"
        debug_info["capture_backend"] = None

    try:
        debug_info["capture_fps"] = float(video_capture.get(cv2.CAP_PROP_FPS) or 0)
    except Exception as exc:
        debug_info["capture_fps_exception"] = f"{type(exc).__name__}: {exc}"
        debug_info["capture_fps"] = 0.0

    try:
        debug_info["capture_frame_count"] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    except Exception as exc:
        debug_info["capture_frame_count_exception"] = f"{type(exc).__name__}: {exc}"
        debug_info["capture_frame_count"] = 0

    return debug_info


def reference_error(message, debug_info, status_code=500, exc=None):
    debug_info = dict(debug_info)
    debug_info["error"] = message
    if exc is not None:
        debug_info["traceback_summary"] = summarize_exception(exc)
    app.logger.error("[reference-frame] error: %s | debug=%s", message, debug_info)
    return jsonify({
        "error": message,
        "traceback_summary": debug_info.get("traceback_summary"),
        "video_path": debug_info.get("video_path"),
        "video_exists": debug_info.get("video_exists"),
        "video_size_bytes": debug_info.get("video_size_bytes"),
        "video_is_lfs_pointer": debug_info.get("video_is_lfs_pointer"),
        "capture_opened": debug_info.get("capture_opened"),
        "capture_frame_count": debug_info.get("capture_frame_count"),
        "model_path": debug_info.get("model_path"),
        "model_exists": debug_info.get("model_exists"),
        "model_size_bytes": debug_info.get("model_size_bytes"),
        "model_is_lfs_pointer": debug_info.get("model_is_lfs_pointer"),
        "reference_debug": debug_info,
    }), status_code

TRANSLATIONS = {
    "nav_home": {"en": "Home", "zh": "首页"},
    "nav_exercises": {"en": "Exercises", "zh": "动作库"},
    "nav_records": {"en": "Training Records", "zh": "训练记录"},
    "nav_about": {"en": "About", "zh": "关于"},
    "nav_admin": {"en": "Admin", "zh": "后台管理"},
    "nav_analytics": {"en": "Analytics", "zh": "数据统计"},
    "nav_login": {"en": "Log In", "zh": "登录"},
    "nav_register": {"en": "Register", "zh": "注册"},
    "nav_logout": {"en": "Log Out", "zh": "退出登录"},
    "nav_start": {"en": "Start Practice", "zh": "开始练习"},
    "lang_en": {"en": "EN", "zh": "英文"},
    "lang_zh": {"en": "中文", "zh": "中文"},
    "auth_welcome": {"en": "Welcome back", "zh": "欢迎回来"},
    "auth_login_title": {"en": "Log in to continue practicing", "zh": "登录后继续练习"},
    "auth_register_title": {"en": "Create your training account", "zh": "创建你的训练账号"},
    "auth_email": {"en": "Email", "zh": "邮箱"},
    "auth_password": {"en": "Password", "zh": "密码"},
    "auth_name": {"en": "Display Name", "zh": "昵称"},
    "auth_submit_login": {"en": "Log In", "zh": "登录"},
    "auth_submit_register": {"en": "Create Account", "zh": "创建账号"},
    "auth_switch_login": {"en": "Already have an account? Log in", "zh": "已有账号？去登录"},
    "auth_switch_register": {"en": "Need an account? Register", "zh": "还没有账号？去注册"},
    "admin_title": {"en": "Operations Dashboard", "zh": "运营后台"},
    "analytics_title": {"en": "Analytics Center", "zh": "数据中心"},
    "records_title": {"en": "Your Practice Records", "zh": "你的训练记录"},
    "records_empty": {
        "en": "No completed sessions yet. Start a training session to build your history.",
        "zh": "你还没有完成过训练，开始一次练习来积累记录吧。",
    },
    "records_total_sessions": {"en": "Total Sessions", "zh": "累计训练次数"},
    "records_avg_score": {"en": "Average Score", "zh": "平均分"},
    "records_total_minutes": {"en": "Practice Minutes", "zh": "训练分钟数"},
    "records_recent": {"en": "Recent Sessions", "zh": "最近训练"},
    "admin_users": {"en": "Registered Users", "zh": "注册用户"},
    "admin_sessions": {"en": "Completed Sessions", "zh": "完成训练"},
    "admin_best_score": {"en": "Best Score", "zh": "最高分"},
    "analytics_mode_split": {"en": "Mode Split", "zh": "模式占比"},
    "analytics_top_users": {"en": "Top Active Users", "zh": "活跃用户排行"},
}


def get_lang():
    lang = session.get("lang", "en")
    return lang if lang in {"en", "zh"} else "en"


def t(key):
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    return entry.get(get_lang(), entry.get("en", key))


def tt(en_text, zh_text):
    return zh_text if get_lang() == "zh" else en_text


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            preferred_lang TEXT NOT NULL DEFAULT 'en',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            exercise_id TEXT NOT NULL,
            exercise_name TEXT NOT NULL,
            focus TEXT NOT NULL,
            mode TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0,
            duration_sec INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


def query_one(query, params=()):
    conn = get_db_connection()
    row = conn.execute(query, params).fetchone()
    conn.close()
    return row


def execute_write(query, params=()):
    conn = get_db_connection()
    cursor = conn.execute(query, params)
    conn.commit()
    lastrowid = cursor.lastrowid
    conn.close()
    return lastrowid


def fetch_all(query, params=()):
    conn = get_db_connection()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return rows


def get_user_by_id(user_id):
    if not user_id:
        return None
    return query_one("SELECT * FROM users WHERE id = ?", (user_id,))


def get_user_by_email(email):
    return query_one("SELECT * FROM users WHERE email = ?", (email.lower().strip(),))


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapper


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        if not g.user["is_admin"]:
            flash("Admin access is required.", "error")
            return redirect(url_for("home"))
        return view_func(*args, **kwargs)
    return wrapper


def build_user_stats(user_id):
    summary = query_one(
        """
        SELECT
            COUNT(*) AS total_sessions,
            COALESCE(ROUND(AVG(score), 1), 0) AS avg_score,
            COALESCE(SUM(duration_sec), 0) AS total_duration_sec
        FROM training_sessions
        WHERE user_id = ?
        """,
        (user_id,),
    )
    recent = fetch_all(
        """
        SELECT exercise_name, focus, mode, score, duration_sec, created_at
        FROM training_sessions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 12
        """,
        (user_id,),
    )
    return summary, recent


def build_admin_stats():
    overview = query_one(
        """
        SELECT
            (SELECT COUNT(*) FROM users) AS total_users,
            (SELECT COUNT(*) FROM training_sessions) AS total_sessions,
            (SELECT COALESCE(MAX(score), 0) FROM training_sessions) AS best_score
        """
    )
    mode_rows = fetch_all(
        """
        SELECT mode, COUNT(*) AS total
        FROM training_sessions
        GROUP BY mode
        ORDER BY total DESC
        """
    )
    top_users = fetch_all(
        """
        SELECT u.display_name, u.email, COUNT(ts.id) AS total_sessions, COALESCE(ROUND(AVG(ts.score), 1), 0) AS avg_score
        FROM users u
        LEFT JOIN training_sessions ts ON ts.user_id = u.id
        GROUP BY u.id
        ORDER BY total_sessions DESC, avg_score DESC
        LIMIT 8
        """
    )
    recent_sessions = fetch_all(
        """
        SELECT u.display_name, ts.exercise_name, ts.mode, ts.score, ts.created_at
        FROM training_sessions ts
        JOIN users u ON u.id = ts.user_id
        ORDER BY ts.created_at DESC
        LIMIT 10
        """
    )
    return overview, mode_rows, top_users, recent_sessions


@app.before_request
def load_globals():
    g.user = get_user_by_id(session.get("user_id"))
    if g.user and g.user["preferred_lang"] != get_lang():
        session["lang"] = g.user["preferred_lang"]


@app.after_request
def apply_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Permissions-Policy", "camera=(self)")
    return response


@app.context_processor
def inject_template_helpers():
    return {
        "t": t,
        "tt": tt,
        "static_asset_version": STATIC_ASSET_VERSION,
        "current_lang": get_lang(),
        "current_user": g.get("user"),
    }


def get_exercise_by_id(exercise_id):
    for ex in EXERCISES:
        if ex["id"] == exercise_id:
            return ex
    return EXERCISES[0]


def format_duration_label(duration_sec):
    total = max(int(duration_sec or 0), 0)
    minutes, seconds = divmod(total, 60)
    if minutes and seconds:
        return f"{minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def build_score_grade(score):
    value = max(min(float(score or 0), 100.0), 0.0)
    if value >= 95:
        return "A+"
    if value >= 90:
        return "A"
    if value >= 85:
        return "B+"
    if value >= 80:
        return "B"
    if value >= 75:
        return "C+"
    if value >= 70:
        return "C"
    return "D"


def build_result_dimensions(score, focus):
    base = max(min(int(round(float(score or 0))), 100), 0)
    emphasis = {
        "upper_body": "Shoulders",
        "lower_body": "Stability",
        "arms": "Breathing",
        "legs": "Hips",
        "full_body": "Spine",
    }.get((focus or "full_body").lower(), "Spine")

    dimensions = [
        {"label": "Spine", "value": max(min(base + 6, 100), 0)},
        {"label": "Breathing", "value": max(min(base - 3, 100), 0)},
        {"label": "Hips", "value": max(min(base + 1, 100), 0)},
        {"label": "Stability", "value": max(min(base - 5, 100), 0)},
        {"label": "Shoulders", "value": max(min(base - 8, 100), 0)},
    ]

    for item in dimensions:
        if item["label"] == emphasis:
            item["value"] = max(min(item["value"] + 6, 100), 0)

    return dimensions


def build_session_result(result_row=None, feedback_message=None):
    if result_row is None:
        return None

    score = float(result_row["score"] or 0)
    duration_sec = int(result_row["duration_sec"] or 0)
    focus = result_row["focus"] or "full_body"
    mode = (result_row["mode"] or "a").lower()
    dimensions = build_result_dimensions(score, focus)
    weakest_dimension = min(dimensions, key=lambda item: item["value"])

    if score >= 90:
        summary = "Excellent alignment and control throughout the session."
        summary_zh = "本次训练整体控制和姿态对齐表现非常出色。"
    elif score >= 80:
        summary = "A stable practice round with a few details worth refining."
        summary_zh = "这次练习整体比较稳定，但还有一些细节可以继续优化。"
    else:
        summary = "You completed the session well. A little more consistency will improve the next round."
        summary_zh = "你已经顺利完成本次训练，再提升一点稳定性，下一轮会更好。"

    return {
        "exercise_name": result_row["exercise_name"],
        "score": round(score, 1),
        "duration_label": format_duration_label(duration_sec),
        "duration_sec": duration_sec,
        "focus": focus,
        "focus_label": focus.replace("_", " ").title(),
        "mode": mode if mode in {"a", "b"} else "a",
        "grade": build_score_grade(score),
        "feedback_message": feedback_message or "Keep your breathing steady and release extra shoulder tension.",
        "summary": summary,
        "summary_zh": summary_zh,
        "dimensions": dimensions,
        "main_issue": weakest_dimension["label"],
    }


def init_resources(video_path):
    global cap_video, landmarker, current_video_path

    with video_lock:
        if landmarker is None:
            landmarker = create_landmarker(MODEL_PATH)

        if current_video_path != video_path:
            if cap_video is not None:
                cap_video.release()
            cap_video = cv2.VideoCapture(video_path)
            current_video_path = video_path
            cap_video.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def get_video_frame_for_time(video_capture, reference_time_sec):
    fps = video_capture.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0 or frame_count <= 0:
        ok, frame = video_capture.read()
        if not ok:
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = video_capture.read()
        return ok, frame

    target_frame = int(max(reference_time_sec, 0.0) * fps) % frame_count
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    ok, frame = video_capture.read()
    if not ok:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ok, frame = video_capture.read()
    return ok, frame


def read_reference_pose(exercise, reference_time_sec):
    debug_info = build_reference_debug(exercise, reference_time_sec)
    video_path = debug_info["video_path"]
    app.logger.info("[reference-frame] begin debug=%s", debug_info)

    if not debug_info["video_exists"]:
        return None, None, None, reference_error(f"Video not found: {video_path}", debug_info), debug_info
    if debug_info["video_is_lfs_pointer"]:
        return None, None, None, reference_error(f"Reference video is a Git LFS pointer, not a playable MP4: {video_path}", debug_info), debug_info
    if not debug_info["model_exists"]:
        return None, None, None, reference_error(f"Pose landmarker model not found: {MODEL_PATH}", debug_info), debug_info
    if debug_info["model_is_lfs_pointer"]:
        return None, None, None, reference_error(f"Pose landmarker model is a Git LFS pointer, not a real task file: {MODEL_PATH}", debug_info), debug_info

    try:
        init_resources(video_path)
    except Exception as exc:
        debug_info["init_resources_exception"] = f"{type(exc).__name__}: {exc}"
        debug_info["traceback_summary"] = summarize_exception(exc)
        app.logger.exception("[reference-frame] init_resources failed")
        return None, None, None, reference_error("Failed to initialize MediaPipe/OpenCV resources", debug_info, exc=exc), debug_info

    with video_lock:
        debug_info = enrich_capture_debug(debug_info, cap_video)
        capture_opened = debug_info["capture_opened"]

        if not capture_opened:
            return video_path, None, None, reference_error("Cannot open reference video", debug_info), debug_info

        try:
            ok_v, frame_v = get_video_frame_for_time(cap_video, reference_time_sec)
        except Exception as exc:
            debug_info["frame_read_exception"] = f"{type(exc).__name__}: {exc}"
            debug_info["traceback_summary"] = summarize_exception(exc)
            app.logger.exception("[reference-frame] get_video_frame_for_time failed")
            return video_path, None, None, reference_error("Failed while seeking/reading the reference video", debug_info, exc=exc), debug_info

        debug_info["frame_read_ok"] = ok_v
        debug_info["frame_shape"] = list(frame_v.shape) if ok_v and frame_v is not None else None

        if not ok_v:
            return video_path, None, None, reference_error("Failed to read reference video", debug_info), debug_info
        if frame_v is None:
            return video_path, None, None, reference_error("Reference video read returned an empty frame", debug_info), debug_info

        try:
            result_v = landmarker.detect(to_mp_image(frame_v))
        except Exception as exc:
            debug_info["landmarker_exception"] = f"{type(exc).__name__}: {exc}"
            debug_info["traceback_summary"] = summarize_exception(exc)
            app.logger.exception("[reference-frame] landmarker.detect failed")
            return video_path, frame_v, None, reference_error("MediaPipe failed while processing the reference frame", debug_info, exc=exc), debug_info

    pose_landmarks = getattr(result_v, "pose_landmarks", None)
    reference_landmarks = get_landmarks(result_v)
    debug_info["pose_landmarks_sets"] = len(pose_landmarks) if pose_landmarks else 0
    debug_info["has_reference_pose"] = reference_landmarks is not None
    debug_info["reference_landmark_count"] = len(reference_landmarks) if reference_landmarks else 0

    app.logger.info("[reference-frame] processed debug=%s", debug_info)

    return video_path, frame_v, result_v, None, debug_info


def serialize_landmarks(landmarks):
    if landmarks is None:
        return []

    return [
        {
            "x": float(getattr(point, "x", 0.0)),
            "y": float(getattr(point, "y", 0.0)),
            "z": float(getattr(point, "z", 0.0)),
            "visibility": float(getattr(point, "visibility", 0.0)),
        }
        for point in landmarks
    ]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/set-language/<lang>")
def set_language(lang):
    if lang not in {"en", "zh"}:
        lang = "en"
    session["lang"] = lang
    if g.user:
        execute_write("UPDATE users SET preferred_lang = ? WHERE id = ?", (lang, g.user["id"]))
    return redirect(request.args.get("next") or request.referrer or url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("home"))

    if request.method == "POST":
        display_name = request.form.get("display_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not display_name or not email or not password:
            flash("Please complete every field.", "error")
        elif get_user_by_email(email):
            flash("That email is already registered.", "error")
        else:
            is_first_user = query_one("SELECT COUNT(*) AS total FROM users")["total"] == 0
            user_id = execute_write(
                """
                INSERT INTO users (email, password_hash, display_name, is_admin, preferred_lang, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    email,
                    generate_password_hash(password),
                    display_name,
                    1 if is_first_user else 0,
                    get_lang(),
                    datetime.utcnow().isoformat(timespec="seconds"),
                ),
            )
            session["user_id"] = user_id
            flash("Account created successfully.", "success")
            return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = get_user_by_email(email)

        if not user or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
        else:
            session["user_id"] = user["id"]
            session["lang"] = user["preferred_lang"]
            flash("Logged in successfully.", "success")
            return redirect(request.args.get("next") or url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/selection")
@login_required
def selection():
    return render_template(
        "selection.html",
        exercises=EXERCISES,
        body_parts=BODY_PART_OPTIONS
    )


@app.route("/exercises")
@login_required
def exercises():
    return redirect(url_for("selection"))


@app.route("/exercise/<exercise_id>")
@login_required
def exercise_detail(exercise_id):
    exercise = get_exercise_by_id(exercise_id)
    return render_template("exercise_detail.html", exercise=exercise)


@app.route("/start-training")
@login_required
def start_training():
    exercise_id = request.args.get("exercise_id", EXERCISES[0]["id"])
    focus = request.args.get("focus", "full_body")
    mode = request.args.get("mode", session.get("mode", "a")).lower()
    if mode not in {"a", "b"}:
        mode = "a"

    session["exercise_id"] = exercise_id
    session["focus"] = focus
    session["mode"] = mode
    return redirect(url_for("training"))


@app.route("/training")
@login_required
def training():
    exercise_id = session.get("exercise_id", EXERCISES[0]["id"])
    focus = session.get("focus", "full_body")
    mode = request.args.get("mode", session.get("mode", "a")).lower()
    if mode not in {"a", "b"}:
        mode = "a"
    session["mode"] = mode
    exercise = get_exercise_by_id(exercise_id)
    template_name = "training_b.html" if mode == "b" else "training_a.html"

    return render_template(
        template_name,
        exercise=exercise,
        focus=focus,
        mode=mode
    )


@app.route("/practice-score")
@login_required
def practice_score():
    latest_result = session.get("last_training_result")

    if latest_result:
        result_payload = latest_result
    else:
        latest_session = query_one(
            """
            SELECT exercise_name, focus, mode, score, duration_sec, created_at
            FROM training_sessions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (g.user["id"],),
        )
        result_payload = build_session_result(latest_session) if latest_session else None

    return render_template("practice_result_score.html", result=result_payload)


@app.route("/result")
@login_required
def result():
    exercise_id = session.get("exercise_id", EXERCISES[0]["id"])
    exercise = get_exercise_by_id(exercise_id)
    mock_result = {
        "exercise_name": exercise["name"],
        "duration": "6 分 20 秒",
        "score": 81,
        "completion": "良好",
        "main_issue": "肩部放松不够",
        "next_suggestion": "继续做肩颈专项入门练习",
        "related_exercise": "双手托天理三焦",
    }
    return render_template("result.html", result=mock_result)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/records")
@login_required
def records():
    summary, recent = build_user_stats(g.user["id"])
    return render_template("records.html", summary=summary, recent_sessions=recent)


@app.route("/admin")
@admin_required
def admin_dashboard():
    overview, mode_rows, top_users, recent_sessions = build_admin_stats()
    return render_template(
        "admin_dashboard.html",
        overview=overview,
        mode_rows=mode_rows,
        top_users=top_users,
        recent_sessions=recent_sessions,
    )


@app.route("/analytics")
@admin_required
def analytics():
    overview, mode_rows, top_users, recent_sessions = build_admin_stats()
    return render_template(
        "analytics.html",
        overview=overview,
        mode_rows=mode_rows,
        top_users=top_users,
        recent_sessions=recent_sessions,
    )


@app.route("/api/training-session", methods=["POST"])
@login_required
def save_training_session():
    payload = request.get_json(silent=True) or {}
    exercise_id = payload.get("exercise_id") or session.get("exercise_id", EXERCISES[0]["id"])
    exercise = get_exercise_by_id(exercise_id)

    score = float(payload.get("score", 0) or 0)
    duration_sec = int(payload.get("duration_sec", 0) or 0)
    mode = (payload.get("mode") or session.get("mode") or "a").lower()
    focus = payload.get("focus") or session.get("focus", "full_body")

    created_at = datetime.utcnow().isoformat(timespec="seconds")

    execute_write(
        """
        INSERT INTO training_sessions (user_id, exercise_id, exercise_name, focus, mode, score, duration_sec, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            g.user["id"],
            exercise_id,
            exercise["name"],
            focus,
            mode if mode in {"a", "b"} else "a",
            score,
            max(duration_sec, 0),
            created_at,
        ),
    )

    result_row = {
        "exercise_name": exercise["name"],
        "focus": focus,
        "mode": mode,
        "score": score,
        "duration_sec": max(duration_sec, 0),
        "created_at": created_at,
    }
    session["last_training_result"] = build_session_result(
        result_row,
        payload.get("feedback_message"),
    )

    return jsonify({"ok": True})


@app.route("/assets/pose-landmarker.task")
def pose_model_asset():
    model_exists = os.path.exists(MODEL_PATH)
    debug_info = {
        "model_path": MODEL_PATH,
        "model_exists": model_exists,
        "model_size_bytes": safe_file_size(MODEL_PATH),
        "model_is_lfs_pointer": looks_like_lfs_pointer(MODEL_PATH) if model_exists else False,
    }
    app.logger.info("[pose-model] request debug=%s", debug_info)
    if not model_exists:
        return jsonify({"error": "Pose landmarker model not found", "model_debug": debug_info}), 404
    return send_file(MODEL_PATH, mimetype="application/octet-stream")


@app.route("/api/score-frame", methods=["POST"])
@login_required
def api_score_frame():
    reference_debug = {}
    try:
        started_at = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        exercise_id = session.get("exercise_id", EXERCISES[0]["id"])
        focus = session.get("focus", "full_body")
        exercise = get_exercise_by_id(exercise_id)
        reference_time_sec = float(payload.get("reference_time", 0) or 0)
        frame_width = max(int(payload.get("frame_width", 720) or 720), 1)
        frame_height = max(int(payload.get("frame_height", 1280) or 1280), 1)
        user_landmarks = normalize_landmarks_payload(payload.get("user_landmarks"))
        reference_debug = build_reference_debug(exercise, reference_time_sec)
        reference_debug["endpoint"] = "score-frame"
        reference_debug["stage"] = "validate_request"

        if user_landmarks is None:
            return jsonify({"error": "No pose detected from browser camera."}), 400

        reference_debug["stage"] = "read_reference_pose"
        _, frame_v, result_v, error_response, reference_debug = read_reference_pose(
            exercise,
            reference_time_sec,
        )
        if error_response is not None:
            return error_response

        reference_debug["stage"] = "extract_joint_angles"
        ref_angles = extract_joint_angles(result_v)
        cam_angles = extract_joint_angles_from_landmarks(user_landmarks)

        reference_debug["stage"] = "score_pose"
        joint_scores = compare_joint_scores(ref_angles, cam_angles)
        overall_score = weighted_overall_score(joint_scores, focus=focus)
        message = build_feedback_message(joint_scores, focus=focus)

        reference_debug["stage"] = "render_overlays"
        user_pose_overlay = draw_pose_overlay_from_landmarks(
            frame_width,
            frame_height,
            user_landmarks,
            line_color=(60, 220, 120, 255),
            point_color=(0, 255, 120, 255)
        )

        ref_h, ref_w = frame_v.shape[:2]
        reference_landmarks = get_landmarks(result_v)
        reference_overlay_left = draw_pose_overlay(
            ref_w,
            ref_h,
            result_v,
            line_color=(230, 192, 79, 255),
            point_color=(95, 150, 95, 255)
        )

        reference_overlay_right = draw_pose_overlay(
            frame_width,
            frame_height,
            result_v,
            line_color=(255, 210, 90, 255),
            point_color=(255, 170, 90, 255)
        )
        reference_debug["stage"] = "encode_response"

        return jsonify({
            "score": overall_score,
            "joint_scores": joint_scores,
            "message": message,
            "focus": focus,
            "exercise_name": exercise["name"],
            "reference_time": round(reference_time_sec, 3),
            "processing_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "camera_pose_overlay_image": encode_png_frame(user_pose_overlay),
            "reference_image": encode_png_frame(reference_overlay_left),
            "reference_overlay_image": encode_png_frame(reference_overlay_right),
            "has_reference_pose": reference_landmarks is not None,
            "reference_landmark_count": len(reference_landmarks) if reference_landmarks else 0,
            "reference_landmarks": serialize_landmarks(reference_landmarks),
            "reference_debug": reference_debug,
        })
    except Exception as exc:
        app.logger.exception("Failed to build frame payload")
        reference_debug = dict(reference_debug or {})
        reference_debug["endpoint"] = "score-frame"
        reference_debug["unhandled_exception"] = f"{type(exc).__name__}: {exc}"
        reference_debug["traceback_summary"] = summarize_exception(exc)
        return reference_error(f"{type(exc).__name__}: {exc}", reference_debug, exc=exc)


@app.route("/api/reference-frame", methods=["POST"])
@login_required
def api_reference_frame():
    reference_debug = {}
    try:
        started_at = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        exercise_id = session.get("exercise_id", EXERCISES[0]["id"])
        exercise = get_exercise_by_id(exercise_id)
        reference_time_sec = float(payload.get("reference_time", 0) or 0)
        reference_debug = build_reference_debug(exercise, reference_time_sec)
        reference_debug["endpoint"] = "reference-frame"
        reference_debug["stage"] = "read_reference_pose"

        _, frame_v, result_v, error_response, reference_debug = read_reference_pose(
            exercise,
            reference_time_sec,
        )
        if error_response is not None:
            return error_response

        reference_debug["stage"] = "render_overlay"
        ref_h, ref_w = frame_v.shape[:2]
        reference_landmarks = get_landmarks(result_v)
        reference_overlay_left = draw_pose_overlay(
            ref_w,
            ref_h,
            result_v,
            line_color=(230, 192, 79, 255),
            point_color=(95, 150, 95, 255)
        )
        reference_debug["stage"] = "encode_response"

        return jsonify({
            "exercise_name": exercise["name"],
            "reference_time": round(reference_time_sec, 3),
            "processing_ms": round((time.perf_counter() - started_at) * 1000, 1),
            "has_reference_pose": reference_landmarks is not None,
            "reference_landmark_count": len(reference_landmarks) if reference_landmarks else 0,
            "reference_image": encode_png_frame(reference_overlay_left),
            "reference_landmarks": serialize_landmarks(reference_landmarks),
            "reference_debug": reference_debug,
        })
    except Exception as exc:
        app.logger.exception("Failed to build reference frame payload")
        reference_debug = dict(reference_debug or {})
        reference_debug["endpoint"] = "reference-frame"
        reference_debug["unhandled_exception"] = f"{type(exc).__name__}: {exc}"
        reference_debug["traceback_summary"] = summarize_exception(exc)
        return reference_error(f"{type(exc).__name__}: {exc}", reference_debug, exc=exc)


init_db()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("FLASK_RUN_HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=env_flag("FLASK_DEBUG", not IS_PRODUCTION),
    )
