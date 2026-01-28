"""
TG-Tinker API Server Module.

Provides FastAPI routes and supporting infrastructure for the
TG-Tinker training API.

Components:
    - routes: FastAPI routers for training API endpoints
    - models: SQLModel database models
    - queue: Job queue for async operations
    - worker: Background worker for job execution
    - storage: Encrypted artifact storage
    - audit: Hash-chained audit logging
    - dp: Differential privacy scaffolding
"""

from .audit import AuditLogger, get_audit_logger, set_audit_logger
from .dp import (
    DPConfig,
    DPMetrics,
    DPState,
    DPTrainer,
    PrivacyAccountant,
    RDPAccountant,
    MomentsAccountant,
    PRVAccountant,
    create_accountant,
    clip_gradients,
    add_noise,
)
from .models import (
    TinkerTrainingClient,
    TinkerFuture,
    TinkerArtifact,
    TinkerAuditLog,
    TinkerDataKey,
)
from .queue import (
    Job,
    JobStatus,
    JobQueue,
    JobQueueBackend,
    InMemoryJobQueue,
    get_job_queue,
    set_job_queue,
)
from .routes import router
from .storage import (
    StorageBackend,
    LocalStorageBackend,
    EncryptedArtifactStore,
    KeyManager,
)
from .worker import (
    Worker,
    MockMLBackend,
    get_worker,
    start_worker,
    stop_worker,
)

__all__ = [
    # Routes
    "router",
    # Models
    "TinkerTrainingClient",
    "TinkerFuture",
    "TinkerArtifact",
    "TinkerAuditLog",
    "TinkerDataKey",
    # Queue
    "Job",
    "JobStatus",
    "JobQueue",
    "JobQueueBackend",
    "InMemoryJobQueue",
    "get_job_queue",
    "set_job_queue",
    # Worker
    "Worker",
    "MockMLBackend",
    "get_worker",
    "start_worker",
    "stop_worker",
    # Storage
    "StorageBackend",
    "LocalStorageBackend",
    "EncryptedArtifactStore",
    "KeyManager",
    # Audit
    "AuditLogger",
    "get_audit_logger",
    "set_audit_logger",
    # DP
    "DPConfig",
    "DPMetrics",
    "DPState",
    "DPTrainer",
    "PrivacyAccountant",
    "RDPAccountant",
    "MomentsAccountant",
    "PRVAccountant",
    "create_accountant",
    "clip_gradients",
    "add_noise",
]
