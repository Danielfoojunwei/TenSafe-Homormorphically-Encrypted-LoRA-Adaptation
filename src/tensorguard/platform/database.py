"""
TensorGuard Platform Database Configuration.

Production-ready database configuration with connection pooling,
health checks, and environment-based configuration.
"""

import logging
import os

from sqlalchemy import event, text
from sqlalchemy.pool import NullPool, QueuePool
from sqlmodel import Session, create_engine

logger = logging.getLogger(__name__)

# Import all models to register them with SQLModel
from .models.continuous_models import AdapterLifecycleState, CandidateEvent, Feed, Policy, Route  # noqa: F401
from .models.core import AuditLog, Fleet, Job, Tenant, User  # noqa: F401
from .models.enablement_models import *  # noqa: F401
from .models.evidence_models import *  # noqa: F401
from .models.fedmoe_models import FedMoEExpert, SkillEvidence  # noqa: F401
from .models.identity_models import (  # noqa: F401
    IdentityAgent,
    IdentityAuditLog,
    IdentityCertificate,
    IdentityEndpoint,
    IdentityPolicy,
    IdentityRenewalJob,
)
from .models.metrics_models import AdapterMetricSnapshot, RouteMetricSeries, RunStepMetrics  # noqa: F401
from .models.peft_models import IntegrationConfig, PeftRun, PeftWizardDraft  # noqa: F401
from .models.settings_models import KMSKey, KMSRotationLog, SystemSetting  # noqa: F401
from .models.tgflow_core_models import *  # noqa: F401
from .models.vla_models import VLABenchmarkResult, VLADeploymentLog, VLAModel, VLASafetyCheck  # noqa: F401

# Unified environment resolver
from ..config.runtime import is_production, is_local_or_dev

# Environment configuration
DATABASE_URL = os.getenv("DATABASE_URL")
TG_DB_POOL_SIZE = int(os.getenv("TG_DB_POOL_SIZE", "10"))
TG_DB_MAX_OVERFLOW = int(os.getenv("TG_DB_MAX_OVERFLOW", "20"))
TG_DB_POOL_TIMEOUT = int(os.getenv("TG_DB_POOL_TIMEOUT", "30"))
TG_DB_POOL_RECYCLE = int(os.getenv("TG_DB_POOL_RECYCLE", "3600"))
TG_DB_ECHO = os.getenv("TG_DB_ECHO", "false").lower() == "true"


def create_production_engine(url: str):
    """
    Create a database engine with production-ready configuration.

    Features:
    - Connection pooling for PostgreSQL/MySQL
    - Pool pre-ping to validate stale connections
    - Automatic connection recycling
    - Debug logging when enabled
    """
    if url.startswith("sqlite"):
        # SQLite: single-threaded, no pooling
        logger.warning("Using SQLite database - NOT RECOMMENDED FOR PRODUCTION")
        return create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=TG_DB_ECHO,
            poolclass=NullPool,  # SQLite doesn't benefit from pooling
        )

    # PostgreSQL/MySQL: full connection pooling
    logger.info(
        f"Creating database engine: pool_size={TG_DB_POOL_SIZE}, "
        f"max_overflow={TG_DB_MAX_OVERFLOW}, pool_recycle={TG_DB_POOL_RECYCLE}s"
    )

    engine = create_engine(
        url,
        poolclass=QueuePool,
        pool_size=TG_DB_POOL_SIZE,
        max_overflow=TG_DB_MAX_OVERFLOW,
        pool_timeout=TG_DB_POOL_TIMEOUT,
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=TG_DB_POOL_RECYCLE,  # Recycle connections after N seconds
        echo=TG_DB_ECHO,
        echo_pool="debug" if TG_DB_ECHO else False,
    )

    # Connection event listeners for monitoring
    @event.listens_for(engine, "connect")
    def on_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")

    @event.listens_for(engine, "checkout")
    def on_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")

    return engine


# Initialize database URL with validation
if not DATABASE_URL:
    if is_production():
        raise RuntimeError(
            "FATAL: DATABASE_URL must be set in production/staging. "
            "Set TENSAFE_ENV=local to use SQLite fallback."
        )
    logger.warning("DATABASE_URL not set, using local SQLite (NOT FOR PRODUCTION)")
    DATABASE_URL = "sqlite:///./tg_platform.db"

# Create the engine
engine = create_production_engine(DATABASE_URL)


# Session factory for background tasks and non-FastAPI contexts
def SessionLocal():
    return Session(engine)


def get_session():
    """Dependency injection for database sessions."""
    with Session(engine) as session:
        yield session


def check_db_health() -> dict:
    """
    Check database connectivity and pool status.
    Returns health check information for monitoring.
    """
    try:
        with Session(engine) as session:
            session.execute(text("SELECT 1"))

        pool = engine.pool
        return {
            "status": "healthy",
            "pool_size": pool.size() if hasattr(pool, "size") else "N/A",
            "checked_in": pool.checkedin() if hasattr(pool, "checkedin") else "N/A",
            "checked_out": pool.checkedout() if hasattr(pool, "checkedout") else "N/A",
            "overflow": pool.overflow() if hasattr(pool, "overflow") else "N/A",
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}
