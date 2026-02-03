"""
TG-Tinker Platform Server.

Privacy-first ML training API server built on FastAPI.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlmodel import SQLModel
from starlette.middleware.base import BaseHTTPMiddleware

from ..security.csp import ContentSecurityPolicy, CSPMiddleware

# Security modules
from ..security.rate_limiter import RateLimitConfig, RateLimitMiddleware
from ..security.sanitization import ValidationMiddleware
from .database import check_db_health, engine
from .tg_tinker_api import router as tinker_router

logger = logging.getLogger(__name__)

# Environment configuration
TG_ENVIRONMENT = os.getenv("TG_ENVIRONMENT", "development")
_raw_origins = os.getenv("TG_ALLOWED_ORIGINS", "")

if _raw_origins:
    TG_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
elif TG_ENVIRONMENT == "production":
    TG_ALLOWED_ORIGINS = []
    logger.warning("SECURITY: No TG_ALLOWED_ORIGINS configured for production.")
else:
    TG_ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]

TG_ENABLE_SECURITY_HEADERS = os.getenv("TG_ENABLE_SECURITY_HEADERS", "true").lower() == "true"
TG_ALLOW_CREDENTIALS = os.getenv("TG_ALLOW_CREDENTIALS", "false").lower() == "true"
TG_ENABLE_RATE_LIMITING = os.getenv("TG_ENABLE_RATE_LIMITING", "true").lower() == "true"
TG_ENABLE_CSP = os.getenv("TG_ENABLE_CSP", "true").lower() == "true"
TG_ENABLE_INPUT_VALIDATION = os.getenv("TG_ENABLE_INPUT_VALIDATION", "true").lower() == "true"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        if TG_ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting TG-Tinker Platform...")

    # Initialize database in development/demo mode
    if TG_ENVIRONMENT != "production" or os.getenv("TG_DEMO_MODE") == "true":
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables initialized.")

    yield
    logger.info("Shutting down TG-Tinker Platform...")


app = FastAPI(
    title="TG-Tinker",
    description="Privacy-First ML Training API",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security headers middleware
if TG_ENABLE_SECURITY_HEADERS:
    app.add_middleware(SecurityHeadersMiddleware)

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=TG_ALLOWED_ORIGINS,
    allow_credentials=TG_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID"],
)

# Rate limiting middleware
if TG_ENABLE_RATE_LIMITING:
    rate_limit_config = RateLimitConfig.from_env()
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    logger.info("Rate limiting enabled")

# Content Security Policy middleware
if TG_ENABLE_CSP:
    csp_policy = ContentSecurityPolicy.for_api()
    app.add_middleware(CSPMiddleware, policy=csp_policy)
    logger.info("Content Security Policy enabled")

# Input validation middleware
if TG_ENABLE_INPUT_VALIDATION:
    app.add_middleware(
        ValidationMiddleware,
        check_sql_injection=True,
        check_command_injection=True,
        check_xss=True,
        exclude_paths=["/health", "/ready", "/live", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )
    logger.info("Input validation middleware enabled")


# Health endpoints
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    db_health = check_db_health()
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "3.0.0",
        "environment": TG_ENVIRONMENT,
        "checks": {"database": db_health},
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Kubernetes readiness probe."""
    db_health = check_db_health()
    if db_health["status"] != "healthy":
        return Response(
            content='{"ready": false, "reason": "database unavailable"}', status_code=503, media_type="application/json"
        )
    return {"ready": True}


@app.get("/live", tags=["health"])
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"alive": True}


@app.get("/version", tags=["health"])
async def version_info():
    """Version information endpoint."""
    return {
        "service": "TG-Tinker",
        "version": "3.0.0",
        "api_version": "v1",
        "python_version": "3.9+",
        "environment": TG_ENVIRONMENT,
        "security_features": {
            "rate_limiting": TG_ENABLE_RATE_LIMITING,
            "csp": TG_ENABLE_CSP,
            "input_validation": TG_ENABLE_INPUT_VALIDATION,
            "security_headers": TG_ENABLE_SECURITY_HEADERS,
        },
    }


# TG-Tinker API routes
app.include_router(tinker_router, prefix="/api")


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root - service information."""
    return {
        "service": "TG-Tinker",
        "version": "3.0.0",
        "description": "Privacy-First ML Training API",
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1/training_clients",
    }


def run_development():
    """Run server in development mode (single-process)."""
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    log_level = os.getenv("TG_LOG_LEVEL", "info")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level=log_level,
        access_log=True,
    )


def run_production():
    """Run server in production mode (multi-worker via gunicorn)."""
    import subprocess
    import sys

    workers = os.getenv("WORKERS", "4")
    port = os.getenv("PORT", "8000")
    timeout = os.getenv("TIMEOUT", "120")
    keepalive = os.getenv("KEEPALIVE", "5")

    cmd = [
        sys.executable,
        "-m",
        "gunicorn",
        "tensorguard.platform.main:app",
        "-c",
        "tensorguard/platform/gunicorn_config.py",
        "--workers",
        workers,
        "--bind",
        f"0.0.0.0:{port}",
        "--timeout",
        timeout,
        "--keep-alive",
        keepalive,
    ]

    logger.info(f"Starting production server with {workers} workers on port {port}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    if TG_ENVIRONMENT == "production":
        run_production()
    else:
        run_development()
