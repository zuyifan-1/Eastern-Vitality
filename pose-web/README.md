# Pose Web

Flask demo for browser-based pose training with MediaPipe, reference videos, login, and SQLite-backed practice records.

## Local Run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Gunicorn Check

Render starts the app through Gunicorn:

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

On Windows, run this check in WSL/Linux or rely on Render, because Gunicorn is Unix-oriented.

## Render Deploy

This repo includes `render.yaml` in the repository root. On Render:

1. Connect the GitHub repository.
2. Use the Blueprint or create a Web Service with root directory `pose-web`.
3. Set `SECRET_KEY` to a long random value.
4. Set `TRUSTED_HOSTS` to your Render host, for example `your-app-name.onrender.com`.
5. Keep `APP_ENV=production`, `SESSION_COOKIE_SECURE=1`, and `USE_PROXY_FIX=1`.

Render will run:

```bash
pip install -r requirements.txt
gunicorn -c gunicorn.conf.py wsgi:app
```

The deployment is pinned to Python 3.11.11 through `PYTHON_VERSION` in `render.yaml`.
The repository also includes `.python-version` for tools and manual Render services that read it.

## Environment Variables

See `.env.example`.

Important defaults:

- `DATABASE_PATH=/opt/render/project/src/pose-web/instance/app.db`
- `MODEL_PATH=/opt/render/project/src/pose-web/pose_landmarker.task`
- `PORT=10000`

SQLite is fine for a demo. For real multi-user usage, move to PostgreSQL.

## Reference Videos

Compressed reference videos live in `static/videos/` and are served by Flask at `/static/videos/<filename>.mp4`.

Before deploying publicly:

- Keep each demo video below GitHub's 100MB per-file limit.
- Make sure `.gitignore` does not ignore `*.mp4` or `pose-web/static/videos/`.
- Reference videos through Flask static URLs, for example `url_for('static', filename='videos/example.mp4')`.
