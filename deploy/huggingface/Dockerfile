# Single-container image for free-tier hosts that only run one container per
# app (Hugging Face Spaces' Docker SDK, Render's free web service, etc).
# It is the same FastAPI backend as backend/Dockerfile, with
# SERVE_FRONTEND=true so it also mounts the static frontend at "/" - see
# app/main.py. Build context must be the repository root:
#   docker build -f deploy/huggingface/Dockerfile -t lifeverse-combined .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVE_FRONTEND=true \
    STATIC_DIR=static \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY frontend ./static
# Combined mode serves frontend and API from the same origin, so the
# frontend must call relative URLs - overwrite the standalone-mode default.
COPY deploy/huggingface/config.js ./static/config.js
RUN rm -f ./static/Dockerfile ./static/docker-entrypoint.sh ./static/config.template.js

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
