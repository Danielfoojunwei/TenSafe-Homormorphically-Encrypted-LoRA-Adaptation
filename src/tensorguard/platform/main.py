"""
TG-Tinker Platform Server.

Privacy-first ML training API server built on FastAPI.
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from contextlib import asynccontextmanager
import os
import logging
from datetime import datetime

from .database import check_db_health, engine
from .tg_tinker_api import router as tinker_router
from sqlmodel import SQLModel

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
        "checks": {"database": db_health}
    }


@app.get("/ready", tags=["health"])
async def readiness_check():
    """Kubernetes readiness probe."""
    db_health = check_db_health()
    if db_health["status"] != "healthy":
        return Response(
            content='{"ready": false, "reason": "database unavailable"}',
            status_code=503,
            media_type="application/json"
        )
    return {"ready": True}


@app.get("/live", tags=["health"])
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"alive": True}


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
        "api": "/api/v1/training_clients"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
