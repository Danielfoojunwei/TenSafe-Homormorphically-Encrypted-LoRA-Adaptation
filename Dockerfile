# TenSafe Production Dockerfile
# Multi-stage build for optimized image size
# Uses wheel installation (not editable) for proper package resolution

# =============================================================================
# Build stage: Build wheel and install dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel build

# Copy only what's needed for building the wheel
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Build the wheel
RUN python -m build --wheel --outdir /wheels

# Install the wheel and runtime dependencies
RUN pip install --no-cache-dir /wheels/*.whl && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# =============================================================================
# Production stage: Runtime image
# =============================================================================
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r tensafe && useradd -r -g tensafe tensafe

WORKDIR /app

# Install runtime dependencies (curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder (includes installed wheel)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy additional modules not in the wheel
# he_lora_microkernel is used at runtime for HE operations
COPY --chown=tensafe:tensafe he_lora_microkernel/ ./he_lora_microkernel/

# Copy gunicorn config (needed for CMD)
COPY --chown=tensafe:tensafe src/tensorguard/platform/gunicorn_config.py ./gunicorn_config.py

# Create necessary directories
RUN mkdir -p /app/.cache /app/data /tmp && \
    chown -R tensafe:tensafe /app/.cache /app/data /tmp

# Environment defaults (override in K8s deployment)
ENV TG_ENVIRONMENT=production \
    PORT=8000 \
    WORKERS=4 \
    TIMEOUT=120 \
    KEEPALIVE=5 \
    MAX_REQUESTS=10000 \
    MAX_REQUESTS_JITTER=1000 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Add he_lora_microkernel to Python path
    PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose ports
# 8000: API server
# 9090: Prometheus metrics
EXPOSE 8000 9090

# Switch to non-root user
USER tensafe

# Production entry point using gunicorn with uvicorn workers
CMD ["gunicorn", "tensorguard.platform.main:app", \
     "-c", "/app/gunicorn_config.py", \
     "-k", "uvicorn.workers.UvicornWorker"]
