# ── Stage 1: build dependencies ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only what's needed to compile native extensions (xgboost, scipy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Copy installed packages from builder — keeps the final image lean.
COPY --from=builder /install /usr/local

WORKDIR /app

# Create a non-root user so the process cannot write outside /app.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy application source
COPY app.py        ./app.py
COPY src/          ./src/
COPY templates/    ./templates/
COPY static/       ./static/
COPY data/         ./data/
COPY models/       ./models/

# Hand ownership to the non-root user.
RUN chown -R appuser:appgroup /app

USER appuser

# Gunicorn settings — override PORT at runtime (Render sets it automatically).
ENV PORT=8000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE $PORT

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app"]
