from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from .database import check_db_health, SessionLocal
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Environment configuration
TG_ENVIRONMENT = os.getenv("TG_ENVIRONMENT", "development")
_raw_origins = os.getenv("TG_ALLOWED_ORIGINS", "")
# SECURITY: Default to restrictive CORS in production, permissive only in dev
if _raw_origins:
    TG_ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
elif TG_ENVIRONMENT == "production":
    TG_ALLOWED_ORIGINS = []  # No origins allowed by default in production
    logger.warning("SECURITY: No TG_ALLOWED_ORIGINS configured for production. CORS will reject all origins.")
else:
    TG_ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"]
TG_ENABLE_SECURITY_HEADERS = os.getenv("TG_ENABLE_SECURITY_HEADERS", "true").lower() == "true"

# SECURITY: Credentials require explicit origin list (not wildcard)
TG_ALLOW_CREDENTIALS = os.getenv("TG_ALLOW_CREDENTIALS", "false").lower() == "true"
if TG_ALLOW_CREDENTIALS and "*" in TG_ALLOWED_ORIGINS:
    logger.critical("SECURITY ERROR: Cannot use allow_credentials=true with wildcard origin '*'")
    TG_ALLOW_CREDENTIALS = False

# Database schema managed via Alembic migrations (no auto-init)

from .api import endpoints

import asyncio
from contextlib import asynccontextmanager
from .api.identity_endpoints import get_session
from ..identity.scheduler import RenewalScheduler
from .models.identity_models import IdentityRenewalJob, RenewalJobStatus
from .models.core import AuditLog, Fleet
from .models.telemetry_models import TelemetryStageEvent, TelemetrySystemEvent
from sqlmodel import select, func
from ..utils.startup_validation import validate_startup_config


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses for production hardening."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Security headers (OWASP recommendations)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS for production (only over HTTPS)
        if TG_ENVIRONMENT == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy (relaxed for SPA)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https:;"
        )

        return response

from ..integrations.peft_hub.catalog import discover_connectors

@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_startup_config(
        "platform",
        require_database=True,
        require_secret_key=True,
        require_key_master=True,
        required_dependencies=[("cryptography", "Install cryptography: pip install cryptography>=41.0")],
    )
    # Startup logic (e.g., connector discovery)
    discover_connectors()
    
    # Auto-initialize database for Demo Mode
    if os.getenv("TG_DEMO_MODE") == "true":
        from .database import engine, SQLModel
        logger.info("DEMO_MODE: Initializing database tables...")
        SQLModel.metadata.create_all(engine)
        
    logger.info("Starting TensorGuard Management Platform...")
    yield
    # Shutdown logic
    logger.info("Shutting down...")

app = FastAPI(
    title="TensorGuard Management Platform",
    description="White-label backend for TensorGuard fleets",
    version="2.3.0",
    lifespan=lifespan
)

# Security headers middleware (first in chain)
if TG_ENABLE_SECURITY_HEADERS:
    app.add_middleware(SecurityHeadersMiddleware)

# GZip compression for responses > 1KB (60-70% bandwidth savings)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS - configurable via environment with secure defaults
app.add_middleware(
    CORSMiddleware,
    allow_origins=TG_ALLOWED_ORIGINS,
    allow_credentials=TG_ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=[
        "Authorization", "Content-Type", "X-Request-ID", "X-Tenant-ID",
        # HMAC auth headers for edge agent telemetry
        "X-TG-Fleet-Id", "X-TG-Timestamp", "X-TG-Nonce", "X-TG-Signature",
    ],
)

# Output structure for dev convenience
os.makedirs("public", exist_ok=True)


# --- Health Check Endpoints ---

@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for load balancers and monitoring.
    Returns system health including database connectivity.
    """
    db_health = check_db_health()

    health_status = {
        "status": "healthy" if db_health["status"] == "healthy" else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.3.0",
        "environment": TG_ENVIRONMENT,
        "checks": {
            "database": db_health
        }
    }

    return health_status


@app.get("/ready", tags=["health"])
async def readiness_check():
    """
    Kubernetes readiness probe.
    Returns 200 if the service can handle requests.
    """
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
    """
    Kubernetes liveness probe.
    Returns 200 if the process is alive.
    """
    return {"alive": True}


@app.get("/metrics", tags=["observability"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns a Prometheus-formatted response derived from live database data.
    """
    try:
        with SessionLocal() as session:
            audit_total = session.exec(select(func.count()).select_from(AuditLog)).one()
            fleet_total = session.exec(select(func.count()).select_from(Fleet)).one()
            stage_events_total = session.exec(
                select(func.count()).select_from(TelemetryStageEvent)
            ).one()
            system_events_total = session.exec(
                select(func.count()).select_from(TelemetrySystemEvent)
            ).one()

        lines = [
            "# TensorGuard Platform Metrics",
            "# TYPE tensorguard_info gauge",
            "tensorguard_info{version=\"2.3.0\"} 1",
            "# TYPE tensorguard_audit_logs_total counter",
            f"tensorguard_audit_logs_total {audit_total}",
            "# TYPE tensorguard_fleets_total gauge",
            f"tensorguard_fleets_total {fleet_total}",
            "# TYPE tensorguard_telemetry_stage_events_total counter",
            f"tensorguard_telemetry_stage_events_total {stage_events_total}",
            "# TYPE tensorguard_telemetry_system_events_total counter",
            f"tensorguard_telemetry_system_events_total {system_events_total}",
        ]
        content = "\n".join(lines) + "\n"
    except Exception as exc:
        logger.error(f"Metrics collection failed: {exc}")
        return Response(
            content='{"error": "metrics collection failed"}',
            status_code=503,
            media_type="application/json",
        )
    return Response(
        content=content,
        media_type="text/plain; charset=utf-8"
    )


# Routes
app.include_router(endpoints.router, prefix="/api/v1")

# Identity routes
from .api import identity_endpoints
app.include_router(identity_endpoints.router, prefix="/api/v1/identity", tags=["identity"])

# Unified Config routes
from .api import config_endpoints
app.include_router(config_endpoints.router, prefix="/api/v1/config", tags=["config"])

# Enablement routes (Trust Console)
# from .api import enablement_endpoints
# app.include_router(enablement_endpoints.router, prefix="/api/v1/enablement", tags=["enablement"])

# Runs & Evidence
from .api import runs_endpoints
app.include_router(runs_endpoints.router, prefix="/api/v1", tags=["runs"])

# Community TGSP
# from .api import community_tgsp
# app.include_router(community_tgsp.router, prefix="/api/community/tgsp", tags=["community-tgsp"])

# PEFT Studio
# from .api import peft_endpoints
# app.include_router(peft_endpoints.router, prefix="/api/v1/peft", tags=["peft"])

# Continuous Learning (TGFlow Core)
from .api import continuous_endpoints, metrics_endpoints
app.include_router(continuous_endpoints.router, prefix="/api/v1", tags=["continuous-learning"])
app.include_router(metrics_endpoints.router, prefix="/api/v1")

# FedMoE Experts & Skills Library
# from .api import fedmoe_endpoints
# app.include_router(fedmoe_endpoints.router, prefix="/api/v1/fedmoe", tags=["fedmoe"])

# System Settings
from .api import settings_endpoints
app.include_router(settings_endpoints.router, prefix="/api/v1", tags=["settings"])

# Pipeline Configuration
# from .api import pipeline_config_endpoints
# app.include_router(pipeline_config_endpoints.router, prefix="/api/v1", tags=["pipeline-config"])

# KMS (Key Management Service)
# from .api import kms_endpoints
# app.include_router(kms_endpoints.router, prefix="/api/v1", tags=["kms"])

# Advanced 3-Tier Gating & Forensics
# from .api import edge_gating_endpoints
# app.include_router(edge_gating_endpoints.router, prefix="/api/v1", tags=["edge-gating"])

# from .api import skills_library_endpoints
# app.include_router(skills_library_endpoints.router, prefix="/api/v1", tags=["skills-library"])

# from .api import bayesian_policy_endpoints
# app.include_router(bayesian_policy_endpoints.router, prefix="/api/v1", tags=["bayesian-policy"])

# from .api import forensics_endpoints
# app.include_router(forensics_endpoints.router, prefix="/api/v1", tags=["forensics"])

# from .api import integrations_endpoints
# app.include_router(integrations_endpoints.router, prefix="/api/v1", tags=["integrations"])

# Model Lineage (Version Control)
# from .api import lineage_endpoints
# app.include_router(lineage_endpoints.router, prefix="/api/v1", tags=["lineage"])

# VLA (Vision-Language-Action) for Robotics
# from .api import vla_endpoints
# app.include_router(vla_endpoints.router, prefix="/api/v1", tags=["vla"])

# Production Telemetry Ingestion & Query
# from .api import telemetry_endpoints
# app.include_router(telemetry_endpoints.router, prefix="/api/v1", tags=["telemetry"])

# Deployment Management (Canary, A/B, Shadow, Rollback)
# from .api import deployment_endpoints
# app.include_router(deployment_endpoints.router, prefix="/api/v1", tags=["deployments"])

# Enterprise Stubs (Proprietary Boundary)
try:
    from .enterprise import check_entitlement, log_audit_event
    print("Enterprise extensions found.")
except ImportError:
    def check_entitlement(user, feat): return True
    def log_audit_event(ev): pass

# Serve UI
# Single Page Application (SPA) catch-all
from fastapi.responses import FileResponse

# Use absolute path for public directory

# Use absolute path for public directory (Vue Build)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PUBLIC_DIR = os.path.join(BASE_DIR, "frontend", "dist")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    # Skip API routes (though definition order should handle this)
    if full_path.startswith("api/v1"):
        return None 
        
    file_path = os.path.join(PUBLIC_DIR, full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    # Default to index.html for SPA
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))

# StaticFiles mounting for structured assets if needed
app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
