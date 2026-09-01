# syntax=docker/dockerfile:1.7

FROM node:24.14.0-alpine AS web-builder
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY web/index.html ./index.html
COPY web/src ./src
RUN npm run build

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS runtime

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/requirements.txt /tmp/deploy-requirements.txt
COPY pyproject.toml LICENSE.txt ./
COPY lingbot_map ./lingbot_map
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/deploy-requirements.txt \
    && python -m pip install --no-cache-dir ".[vis]"

COPY demo.py ./demo.py
COPY webapp ./webapp
COPY --from=web-builder /build/web/dist ./web_dist

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/.cache/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    LINGBOT_WEB_DIST=/app/web_dist \
    LINGBOT_MODEL_DIR=/models \
    LINGBOT_JOBS_DIR=/data/jobs

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=4 \
    CMD curl -fsS http://127.0.0.1:8080/readyz >/dev/null || exit 1

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "*"]
