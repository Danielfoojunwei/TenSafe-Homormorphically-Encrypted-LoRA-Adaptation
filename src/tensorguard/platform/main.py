"""
TG-Tinker Platform Server.

Privacy-first ML training API server built on FastAPI.
"""

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlmodel import SQLModel
from starlette.middleware.base import BaseHTTPMiddleware

from .database import check_db_health, engine
from .tg_tinker_api import router as tinker_router
from .auth_routes import router as auth_router
from .sso.routes import router as sso_router

# Security modules
from ..security.rate_limiter import RateLimitMiddleware, RateLimitConfig
from ..security.csp import CSPMiddleware, ContentSecurityPolicy
from ..security.sanitization import ValidationMiddleware

# Unified environment resolver
from ..config.runtime import (
    ENVIRONMENT,
    is_production,
    is_local_or_dev,
    validate_no_demo_mode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment configuration (uses unified resolver)
# ---------------------------------------------------------------------------
TG_ENVIRONMENT = ENVIRONMENT.value

_raw_origins = os.getenv("TG_ALLOWED_ORIGINS", "")

if _raw_origins:
    TG_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
elif is_production():
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

        if is_production():
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting TG-Tinker Platform (env=%s)...", ENVIRONMENT.value)

    # Block demo mode in production/staging unconditionally
    validate_no_demo_mode()

    # Auto-create tables ONLY in local/dev (never in staging/production).
    if is_local_or_dev():
        SQLModel.metadata.create_all(engine)
        logger.info("Database tables auto-created (env=%s).", ENVIRONMENT.value)
    else:
        logger.info(
            "Skipping auto table creation (env=%s). Use Alembic migrations.",
            ENVIRONMENT.value,
        )

    yield
    logger.info("Shutting down TG-Tinker Platform...")


from tensorguard.version import tensafe_version
API_VERSION = tensafe_version()
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
    docs_url="/docs" if is_local_or_dev() else None,
    redoc_url="/redoc" if is_local_or_dev() else None,
    openapi_url="/openapi.json" if is_local_or_dev() else None,
    openapi_tags=[
        {"name": "auth", "description": "User authentication - signup, login, password reset"},
        {"name": "sso", "description": "Enterprise SSO - OIDC and SAML authentication"},
        {"name": "training", "description": "Training client management and operations"},
        {"name": "inference", "description": "Model inference endpoints"},
        {"name": "privacy", "description": "Differential privacy and HE operations"},
        {"name": "tgsp", "description": "TenSafe Secure Package (TGSP) adapter management"},
        {"name": "audit", "description": "Audit logging and compliance"},
        {"name": "health", "description": "Health check and system status"},
        {"name": "admin", "description": "Administrative operations (requires admin role)"},
    ],
    license_info={"name": "Apache 2.0", "url": "https://www.apache.org/licenses/LICENSE-2.0"},
    contact={"name": "TenSafe Support", "url": "https://tensafe.io/support", "email": "support@tensafe.io"},
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
    allow_methods=["GET", "POST", "PUT", "OPTIONS"],
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
        exclude_paths=["/healthz", "/readyz", "/health", "/ready", "/live", "/metrics", "/docs", "/redoc", "/openapi.json"],
    )
    logger.info("Input validation middleware enabled")


# ---------------------------------------------------------------------------
# Health check caching
# ---------------------------------------------------------------------------
_health_cache: dict = {"result": None, "timestamp": 0.0}
_HEALTH_CACHE_TTL = 5.0  # seconds


def _get_cached_db_health() -> dict:
    """Get DB health with caching to reduce probe overhead."""
    import time as _time
    now = _time.time()
    if _health_cache["result"] is None or (now - _health_cache["timestamp"]) > _HEALTH_CACHE_TTL:
        _health_cache["result"] = check_db_health()
        _health_cache["timestamp"] = now
    return _health_cache["result"]


# ---------------------------------------------------------------------------
# Health endpoints — split per K8s convention
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["health"])
async def healthz():
    """Minimal liveness probe — 200 if process is alive."""
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
async def readyz():
    """Readiness probe with dependency checks (DB connectivity)."""
    db_health = _get_cached_db_health()
    if db_health["status"] != "healthy":
        return Response(
            content='{"ready": false, "reason": "database unavailable"}',
            status_code=503,
            media_type="application/json",
        )
    return {"ready": True, "checks": {"database": db_health}}


# Legacy endpoints kept for backward compatibility (hidden from OpenAPI)
@app.get("/health", tags=["health"], include_in_schema=False)
async def health_check():
    """Legacy health check — full status."""
    db_health = _get_cached_db_health()
    return {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": API_VERSION,
        "environment": TG_ENVIRONMENT,
        "checks": {"database": db_health},
    }


@app.get("/ready", tags=["health"], include_in_schema=False)
async def readiness_check():
    """Legacy readiness probe."""
    return await readyz()


@app.get("/live", tags=["health"], include_in_schema=False)
async def liveness_check():
    """Legacy liveness probe."""
    return await healthz()


@app.get("/version", tags=["health"])
async def version_info():
    """Version information endpoint."""
    return {
        "service": "TenSafe",
        "version": API_VERSION,
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


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------
app.include_router(auth_router)
app.include_router(sso_router)
app.include_router(tinker_router, prefix="/api")

# Playground — only in local/dev
if is_local_or_dev():
    from .playground import router as playground_router
    app.include_router(playground_router)
    logger.info("Playground router mounted (env=%s)", ENVIRONMENT.value)
else:
    logger.info("Playground router disabled (env=%s)", ENVIRONMENT.value)


@app.get("/", tags=["health"])
async def root():
    """API root - service information."""
    return {
        "service": "TenSafe",
        "version": API_VERSION,
        "description": "Privacy-Preserving ML Platform with HE-LoRA, DP-SGD, and PQC",
        "links": {
            "documentation": "/docs",
            "openapi_spec": "/openapi.json",
            "health": "/healthz",
            "readiness": "/readyz",
        },
        "api_versions": {"current": "v1", "supported": ["v1"], "deprecated": []},
        "auth": {"signup": "/auth/signup", "login": "/auth/token", "sso_providers": "/auth/sso/providers"},
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
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=log_level, access_log=True)


if __name__ == "__main__":
    run_development()
