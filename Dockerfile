# TenSafe Production Dockerfile
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml ./
COPY src/ ./src/

# Create virtual environment and install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir gunicorn uvicorn[standard] && \
    pip install --no-cache-dir -e .

# Production stage
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r tensafe && useradd -r -g tensafe tensafe

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=tensafe:tensafe src/ ./src/
COPY --chown=tensafe:tensafe he_lora_microkernel/ ./he_lora_microkernel/

# Create necessary directories
RUN mkdir -p /app/.cache /tmp && \
    chown -R tensafe:tensafe /app/.cache /tmp

# Environment defaults (override in K8s deployment)
ENV TG_ENVIRONMENT=production \
    PORT=8000 \
    WORKERS=4 \
    TIMEOUT=120 \
    KEEPALIVE=5 \
    MAX_REQUESTS=10000 \
    MAX_REQUESTS_JITTER=1000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Switch to non-root user
USER tensafe

# Production entry point using gunicorn with uvicorn workers
CMD ["gunicorn", "tensorguard.platform.main:app", \
     "-c", "src/tensorguard/platform/gunicorn_config.py", \
     "-k", "uvicorn.workers.UvicornWorker"]
