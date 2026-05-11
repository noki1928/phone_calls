FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config.yaml ./config.yaml

RUN mkdir -p /cache \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app /cache

USER appuser

EXPOSE 8096

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8096"]
