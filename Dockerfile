FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖（关键）
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制代码
COPY pose-web /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 启动
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000"]
