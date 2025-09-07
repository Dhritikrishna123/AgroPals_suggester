# =========================
# 1. Builder stage
# =========================
FROM python:3.11-bullseye AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =========================
# 2. Runtime stage
# =========================
FROM python:3.11-bullseye

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p data trained_models logs \
    && useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

USER app

RUN chmod +x scripts/*.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["sh", "-c", "python scripts/setup.py && uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
