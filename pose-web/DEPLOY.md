# Pose Web Deployment Guide

## What changed in code

- `SECRET_KEY` is now read from environment variables.
- Session cookies are configured for safer production defaults.
- The app can trust reverse proxy headers when `USE_PROXY_FIX=1`.
- Gunicorn startup files are included: `wsgi.py` and `gunicorn.conf.py`.

## What you need to prepare

1. A Linux server such as Ubuntu 24.04.
2. A domain name pointing to that server.
3. HTTPS enabled before public launch, because browser camera access requires HTTPS.
4. Enough disk space for large training videos.

## Why HTTPS is mandatory

The browser only allows `getUserMedia()` camera access on:

- `http://localhost` during local development
- `https://...` in production

If you deploy without HTTPS, the live training page will load but the user's camera will be blocked by the browser.

## Why the MediaPipe CDN matters

The frontend currently loads the MediaPipe JavaScript runtime from jsDelivr.

That means:

- Your server does not host that runtime itself.
- The user's browser must be able to reach `cdn.jsdelivr.net`.
- If your target users are in a network environment where jsDelivr is blocked or unstable, the pose model will fail to load.

If that becomes a problem, the next step is to self-host the MediaPipe runtime files under your own domain.

## Recommended first production setup

For your current stage, use:

- Flask app code
- Gunicorn app server
- Nginx reverse proxy
- SQLite for small private testing or pilot rollout
- CDN or object storage for large videos when traffic grows

Do not switch everything at once unless you need it right now.

## Environment variables

Copy `.env.example` and set real values in your server environment:

```bash
APP_ENV=production
SECRET_KEY=replace-with-a-long-random-secret
SESSION_COOKIE_SECURE=1
USE_PROXY_FIX=1
TRUSTED_HOSTS=your-domain.com,www.your-domain.com
DATABASE_PATH=/var/www/pose-web/shared/app.db
MODEL_PATH=/var/www/pose-web/current/pose_landmarker.task
GUNICORN_BIND=127.0.0.1:8000
```

## Server steps

### 1. Install system packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nginx
```

### 2. Upload project

Put the project on the server, for example:

```bash
/var/www/pose-web/current
```

Create a writable shared directory for the database:

```bash
sudo mkdir -p /var/www/pose-web/shared
sudo chown -R $USER:$USER /var/www/pose-web
```

### 3. Create virtual environment

```bash
cd /var/www/pose-web/current/pose-web
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start Gunicorn manually first

```bash
cd /var/www/pose-web/current/pose-web
export APP_ENV=production
export SECRET_KEY='replace-this'
export SESSION_COOKIE_SECURE=1
export USE_PROXY_FIX=1
export TRUSTED_HOSTS='your-domain.com,www.your-domain.com'
export DATABASE_PATH='/var/www/pose-web/shared/app.db'
.venv/bin/gunicorn -c gunicorn.conf.py wsgi:app
```

If this works, your app should listen on `127.0.0.1:8000`.

### 5. Configure Nginx

Example site config:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;

    client_max_body_size 10m;

    location /static/ {
        alias /var/www/pose-web/current/pose-web/static/;
        expires 30d;
        add_header Cache-Control "public, max-age=2592000";
    }

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Port $server_port;
    }
}
```

Then enable and reload:

```bash
sudo ln -s /etc/nginx/sites-available/pose-web /etc/nginx/sites-enabled/pose-web
sudo nginx -t
sudo systemctl reload nginx
```

### 6. Enable HTTPS

Use Certbot with Nginx after the domain resolves to your server:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

After HTTPS is active, camera access will work for real users.

## What to do next after first launch

### Move off SQLite when:

- You have multiple Gunicorn workers writing frequently
- You want safer concurrency
- You need backups and managed reliability

At that point, migrate to PostgreSQL.

### Move videos to CDN/object storage when:

- The server bandwidth becomes expensive
- Page loads are slow in other regions
- Video files keep growing

Good pattern:

- App HTML/API from your server
- MP4 files from object storage or CDN

### Self-host MediaPipe runtime when:

- Your users cannot reliably access jsDelivr
- You want fewer third-party runtime dependencies

## Minimal launch checklist

- Domain resolves correctly
- HTTPS works
- `SECRET_KEY` is set
- `APP_ENV=production`
- `SESSION_COOKIE_SECURE=1`
- Gunicorn runs behind Nginx
- Camera works on your public domain
- Database file is outside the release folder
- Large videos are monitored for bandwidth and disk usage
