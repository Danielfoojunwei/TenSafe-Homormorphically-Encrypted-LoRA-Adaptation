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

from .database import check_db_health, engine
from .tg_tinker_api import router as tinker_router
from .playground import router as playground_router
from .auth_routes import router as auth_router
from .sso.routes import router as sso_router

# Security modules
from ..security.rate_limiter import RateLimitMiddleware, RateLimitConfig
from ..security.csp import CSPMiddleware, ContentSecurityPolicy
from ..security.sanitization import ValidationMiddleware

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


API_VERSION = "4.0.0"
API_TITLE = "TenSafe API"
API_DESCRIPTION = """
# TenSafe - Privacy-Preserving ML Platform

TenSafe is the only ML platform combining:
- **Homomorphic Encryption (HE-LoRA)** for encrypted inference
- **Differential Privacy (DP-SGD)** for training data protection
- **Post-Quantum Cryptography** for quantum-resistant security

## Quick Start

```python
from tensafe import TenSafeClient

client = TenSafeClient(api_key="your-api-key")

# Create a training client with differential privacy
tc = client.create_training_client(
    model_ref="meta-llama/Llama-3-8B",
    dp_config={"epsilon": 8.0, "delta": 1e-5}
)

# Train with privacy guarantees
tc.train(dataset="your-dataset")
```

## Authentication

All API requests require a Bearer token in the Authorization header:

```
Authorization: Bearer <your-api-key>
```

## Rate Limits

| Tier | Requests/min | Requests/hour |
|------|-------------|---------------|
| Free | 60 | 1,000 |
| Pro | 300 | 10,000 |
| Business | 1,000 | 50,000 |
| Enterprise | Custom | Custom |

## Support

- **Documentation**: https://docs.tensafe.io
- **Status**: https://status.tensafe.io
- **Support**: support@tensafe.io
"""

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "auth",
            "description": "User authentication - signup, login, password reset",
        },
        {
            "name": "sso",
            "description": "Enterprise SSO - OIDC and SAML authentication",
        },
        {
            "name": "training",
            "description": "Training client management and operations",
        },
        {
            "name": "inference",
            "description": "Model inference endpoints",
        },
        {
            "name": "privacy",
            "description": "Differential privacy and HE operations",
        },
        {
            "name": "tgsp",
            "description": "TenSafe Secure Package (TGSP) adapter management",
        },
        {
            "name": "audit",
            "description": "Audit logging and compliance",
        },
        {
            "name": "health",
            "description": "Health check and system status",
        },
        {
            "name": "admin",
            "description": "Administrative operations (requires admin role)",
        },
    ],
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0",
    },
    contact={
        "name": "TenSafe Support",
        "url": "https://tensafe.io/support",
        "email": "support@tensafe.io",
    },
    servers=[
        {"url": "https://api.tensafe.io", "description": "Production"},
        {"url": "https://api.staging.tensafe.io", "description": "Staging"},
        {"url": "http://localhost:8000", "description": "Local development"},
    ],
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
        "version": "4.0.0",
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


# Authentication routes
app.include_router(auth_router)
app.include_router(sso_router)

# TG-Tinker API routes
app.include_router(tinker_router, prefix="/api")

# Playground routes
app.include_router(playground_router)


# Root endpoint
@app.get("/", tags=["health"])
async def root():
    """
    API root - service information.

    Returns basic information about the TenSafe API including version,
    available endpoints, and links to documentation.
    """
    return {
        "service": "TenSafe",
        "version": API_VERSION,
        "description": "Privacy-Preserving ML Platform with HE-LoRA, DP-SGD, and PQC",
        "links": {
            "documentation": "/docs",
            "openapi_spec": "/openapi.json",
            "health": "/health",
            "status": "/status",
        },
        "api_versions": {
            "current": "v1",
            "supported": ["v1"],
            "deprecated": [],
        },
        "auth": {
            "signup": "/auth/signup",
            "login": "/auth/token",
            "sso_providers": "/auth/sso/providers",
        },
        "endpoints": {
            "training": "/api/v1/training_clients",
            "inference": "/api/v1/inference",
            "tgsp": "/api/v1/tgsp",
            "audit": "/api/v1/audit_logs",
        },
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
