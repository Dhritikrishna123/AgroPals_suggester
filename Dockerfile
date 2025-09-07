# =========================
# 1. Builder stage
# =========================
FROM python:3.11-bullseye AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Upgrade pip and install minimal build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install into /install
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

# Copy installed dependencies
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Directories & permissions
RUN mkdir -p data trained_models logs \
    && useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app

USER app

# Scripts executable
RUN chmod +x scripts/*.py

EXPOSE 8080

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run setup -> train -> serve
CMD ["sh", "-c", "python scripts/setup.py && python scripts/train_models.py && uvicorn app.main:app --host 0.0.0.0 --port 8080"]
