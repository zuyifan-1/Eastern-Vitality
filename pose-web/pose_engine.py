import base64
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

CONNECTIONS = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 31),
    (28, 32),
]

ANGLE_TRIPLETS = {
    "left_elbow": (11, 13, 15),
    "right_elbow": (12, 14, 16),
    "left_shoulder": (13, 11, 23),
    "right_shoulder": (14, 12, 24),
    "left_hip": (11, 23, 25),
    "right_hip": (12, 24, 26),
    "left_knee": (23, 25, 27),
    "right_knee": (24, 26, 28),
}

FOCUS_WEIGHTS = {
    "full_body": {
        "left_elbow": 1.0, "right_elbow": 1.0,
        "left_shoulder": 1.0, "right_shoulder": 1.0,
        "left_hip": 1.0, "right_hip": 1.0,
        "left_knee": 1.0, "right_knee": 1.0,
    },
    "shoulders": {
        "left_shoulder": 2.0, "right_shoulder": 2.0,
        "left_elbow": 1.2, "right_elbow": 1.2,
    },
    "arms": {
        "left_elbow": 2.0, "right_elbow": 2.0,
        "left_shoulder": 1.2, "right_shoulder": 1.2,
    },
    "torso": {
        "left_shoulder": 1.5, "right_shoulder": 1.5,
        "left_hip": 1.5, "right_hip": 1.5,
    },
    "legs": {
        "left_knee": 2.0, "right_knee": 2.0,
        "left_hip": 1.3, "right_hip": 1.3,
    },
}


def create_landmarker(model_path: str):
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


def to_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def calc_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    c = np.dot(v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def get_landmarks(result):
    if not result or not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]


def _extract_joint_angles_from_landmarks(lm):
    if lm is None:
        return None

    angles = {}
    for name, (a, b, c) in ANGLE_TRIPLETS.items():
        if (
            getattr(lm[a], "visibility", 1.0) < 0.5 or
            getattr(lm[b], "visibility", 1.0) < 0.5 or
            getattr(lm[c], "visibility", 1.0) < 0.5
        ):
            angles[name] = np.nan
            continue

        pa = np.array([lm[a].x, lm[a].y, lm[a].z], dtype=np.float32)
        pb = np.array([lm[b].x, lm[b].y, lm[b].z], dtype=np.float32)
        pc = np.array([lm[c].x, lm[c].y, lm[c].z], dtype=np.float32)
        angles[name] = calc_angle(pa, pb, pc)

    return angles


def extract_joint_angles(result):
    return _extract_joint_angles_from_landmarks(get_landmarks(result))


def extract_joint_angles_from_landmarks(landmarks):
    return _extract_joint_angles_from_landmarks(landmarks)


def compare_joint_scores(ref_angles, cam_angles):
    if ref_angles is None or cam_angles is None:
        return {}

    joint_scores = {}
    for key in ANGLE_TRIPLETS.keys():
        a = ref_angles.get(key, np.nan)
        b = cam_angles.get(key, np.nan)
        if np.isnan(a) or np.isnan(b):
            joint_scores[key] = 0.0
        else:
            diff = abs(a - b)
            joint_scores[key] = float(np.clip(100.0 * (1.0 - diff / 180.0), 0.0, 100.0))
    return joint_scores


def weighted_overall_score(joint_scores, focus="full_body"):
    weights = FOCUS_WEIGHTS.get(focus, FOCUS_WEIGHTS["full_body"])
    total_w = 0.0
    total_score = 0.0

    for k, w in weights.items():
        if k in joint_scores:
            total_score += joint_scores[k] * w
            total_w += w

    if total_w <= 0:
        return 0.0
    return round(total_score / total_w, 1)


def build_feedback_message(joint_scores, focus="full_body"):
    if not joint_scores:
        return "Move into camera view for posture analysis."

    weakest = min(joint_scores.items(), key=lambda x: x[1])
    joint, score = weakest

    name_map = {
        "left_elbow": "left elbow",
        "right_elbow": "right elbow",
        "left_shoulder": "left shoulder",
        "right_shoulder": "right shoulder",
        "left_hip": "left hip",
        "right_hip": "right hip",
        "left_knee": "left knee",
        "right_knee": "right knee",
    }

    if score >= 85:
        return f"Good alignment overall. Maintain controlled movement in the {focus.replace('_', ' ')}."
    elif score >= 60:
        return f"Your weakest area is the {name_map.get(joint, joint)}. Adjust posture slightly for better alignment."
    else:
        return f"Large deviation detected at the {name_map.get(joint, joint)}. Slow down and match the reference more closely."


def make_portrait_crop(frame):
    h, w = frame.shape[:2]
    target_ratio = 9 / 16
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = max((w - new_w) // 2, 0)
        frame = frame[:, x1:x1 + new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = max((h - new_h) // 2, 0)
        frame = frame[y1:y1 + new_h, :]
    return frame


def _draw_pose_on_image(image, result, line_color, point_color, draw_title=None, title_color=(60, 90, 60)):
    h, w = image.shape[:2]
    base = min(h, w)
    line_thickness = max(4, int(base / 160))
    point_radius = max(5, int(base / 110))
    text_scale = max(0.9, base / 700)
    text_thickness = max(2, int(base / 300))

    lm = get_landmarks(result)
    if lm is None:
        if draw_title:
            cv2.putText(
                image, draw_title, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, text_scale, title_color, text_thickness
            )
        return image

    for a, b in CONNECTIONS:
        la = lm[a]
        lb = lm[b]
        if getattr(la, "visibility", 1.0) < 0.5 or getattr(lb, "visibility", 1.0) < 0.5:
            continue

        x1, y1 = int(la.x * w), int(la.y * h)
        x2, y2 = int(lb.x * w), int(lb.y * h)
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.line(image, (x1, y1), (x2, y2), line_color, line_thickness, lineType=cv2.LINE_AA)

    for p in lm:
        if getattr(p, "visibility", 1.0) < 0.5:
            continue
        x, y = int(p.x * w), int(p.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), point_radius, point_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(image, (x, y), point_radius + 2, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    if draw_title:
        cv2.putText(
            image, draw_title, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, title_color, text_thickness
        )

    return image


def draw_pose(frame, result, title="", line_color=(230, 192, 79), point_color=(95, 150, 95)):
    return _draw_pose_on_image(
        frame,
        result,
        line_color=line_color,
        point_color=point_color,
        draw_title=title if title else None,
    )


def create_blank_overlay(width, height):
    return np.zeros((height, width, 4), dtype=np.uint8)


def _draw_pose_on_rgba_overlay(overlay, result, line_color=(230, 192, 79, 255), point_color=(95, 150, 95, 255)):
    h, w = overlay.shape[:2]
    base = min(h, w)
    line_thickness = max(4, int(base / 160))
    point_radius = max(5, int(base / 110))

    lm = get_landmarks(result)
    if lm is None:
        return overlay

    bgr_line = tuple(int(c) for c in line_color[:3])
    bgr_point = tuple(int(c) for c in point_color[:3])

    for a, b in CONNECTIONS:
        la = lm[a]
        lb = lm[b]
        if getattr(la, "visibility", 1.0) < 0.5 or getattr(lb, "visibility", 1.0) < 0.5:
            continue

        x1, y1 = int(la.x * w), int(la.y * h)
        x2, y2 = int(lb.x * w), int(lb.y * h)
        if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
            cv2.line(overlay, (x1, y1), (x2, y2), (*bgr_line, 255), line_thickness, lineType=cv2.LINE_AA)

    for p in lm:
        if getattr(p, "visibility", 1.0) < 0.5:
            continue
        x, y = int(p.x * w), int(p.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(overlay, (x, y), point_radius, (*bgr_point, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (x, y), point_radius + 2, (255, 255, 255, 255), 2, lineType=cv2.LINE_AA)

    return overlay


def draw_pose_overlay(width, height, result, line_color=(230, 192, 79, 255), point_color=(95, 150, 95, 255)):
    overlay = create_blank_overlay(width, height)
    return _draw_pose_on_rgba_overlay(
        overlay,
        result,
        line_color=line_color,
        point_color=point_color,
    )


def draw_pose_overlay_from_landmarks(width, height, landmarks, line_color=(230, 192, 79, 255), point_color=(95, 150, 95, 255)):
    overlay = create_blank_overlay(width, height)
    if landmarks is None:
        return overlay
    return _draw_pose_on_rgba_overlay(
        overlay,
        type("PoseResult", (), {"pose_landmarks": [landmarks]}),
        line_color=line_color,
        point_color=point_color,
    )


def resize_result_landmarks_to_target(result, target_width, target_height, source_width, source_height):
    """
    当前先保持 landmarks 的归一化坐标不变。
    因为 MediaPipe landmarks 本来就是 0~1 归一化坐标，
    在同样使用 object-cover 的前提下，这里可以先直接复用。
    后面如果你要做更严格的 letterbox / crop 对齐，再单独细调。
    """
    return result


def encode_frame(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def encode_png_frame(frame):
    ok, buf = cv2.imencode(".png", frame)
    if not ok:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def normalize_landmarks_payload(payload):
    if not isinstance(payload, list):
        return None

    landmarks = []
    for item in payload:
        if not isinstance(item, dict):
            return None
        try:
            landmarks.append(
                type(
                    "NormalizedLandmark",
                    (),
                    {
                        "x": float(item.get("x", 0.0)),
                        "y": float(item.get("y", 0.0)),
                        "z": float(item.get("z", 0.0)),
                        "visibility": float(item.get("visibility", 0.0)),
                    },
                )()
            )
        except (TypeError, ValueError):
            return None

    return landmarks if landmarks else None
