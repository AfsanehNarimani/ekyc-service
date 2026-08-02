FROM python:3.11-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install \
    --prefix=/install \
    --requirement requirements.txt


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system ekyc \
    && useradd \
        --system \
        --gid ekyc \
        --home-dir /app \
        --no-create-home \
        ekyc \
    && mkdir -p \
        /app/data/temp \
        /app/data/references \
        /app/data/embeddings \
        /models \
    && chown -R ekyc:ekyc \
        /app \
        /models

COPY --from=builder /install /usr/local
COPY --chown=ekyc:ekyc . /app

USER ekyc

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=60s \
    --retries=3 \
    CMD python -c \
        "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); data = json.load(response); raise SystemExit(0 if response.status == 200 and data.get('ok') else 1)"

    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]