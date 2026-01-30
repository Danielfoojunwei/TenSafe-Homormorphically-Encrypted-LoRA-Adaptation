"""
TG-Tinker API Server Module.

Provides FastAPI routes and supporting infrastructure for the
TG-Tinker training API.

Components:
    - routes: FastAPI routers for training API endpoints
    - models: SQLModel database models
    - queue: Job queue for async operations
    - worker: Background worker for job execution
    - storage: Encrypted artifact storage with identity integration
    - audit: Hash-chained audit logging
    - dp: Differential privacy scaffolding
    - tgsp_bridge: Integration with TGSP secure packaging
"""

from .audit import AuditLogger, get_audit_logger, set_audit_logger
from .dp import (
    DPConfig,
    DPMetrics,
    DPState,
    DPTrainer,
    MomentsAccountant,
    PrivacyAccountant,
    PRVAccountant,
    RDPAccountant,
    add_noise,
    clip_gradients,
    create_accountant,
)
from .models import (
    TinkerArtifact,
    TinkerAuditLog,
    TinkerDataKey,
    TinkerFuture,
    TinkerTrainingClient,
)
from .queue import (
    InMemoryJobQueue,
    Job,
    JobQueue,
    JobQueueBackend,
    JobStatus,
    get_job_queue,
    set_job_queue,
)
from .routes import router
from .storage import (
    EncryptedArtifactStore,
    IdentityKeyManager,
    KeyManager,
    LocalStorageBackend,
    SignedArtifactStore,
    StorageBackend,
    create_artifact_store,
)
from .tgsp_bridge import (
    TinkerTGSPBridge,
    create_dp_certificate,
)
from .worker import (
    MockMLBackend,
    Worker,
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
    "IdentityKeyManager",
    "SignedArtifactStore",
    "create_artifact_store",
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
    # TGSP Integration
    "TinkerTGSPBridge",
    "create_dp_certificate",
]
