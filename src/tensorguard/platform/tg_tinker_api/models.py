"""
TG-Tinker API database models.

SQLModel definitions for training clients, futures, artifacts, and audit logs.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Field, Relationship, SQLModel, Column, JSON


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def generate_tc_id() -> str:
    """Generate a training client ID."""
    return f"tc-{uuid.uuid4()}"


def generate_future_id() -> str:
    """Generate a future ID."""
    return f"fut-{uuid.uuid4()}"


def generate_artifact_id() -> str:
    """Generate an artifact ID."""
    return f"art-{uuid.uuid4()}"


def generate_audit_id() -> str:
    """Generate an audit log entry ID."""
    return f"aud-{uuid.uuid4()}"


# ==============================================================================
# Training Client
# ==============================================================================


class TinkerTrainingClient(SQLModel, table=True):
    """Training client state in the database."""

    __tablename__ = "tinker_training_clients"

    id: str = Field(default_factory=generate_tc_id, primary_key=True)
    tenant_id: str = Field(index=True)
    model_ref: str
    status: str = Field(default="ready")  # initializing, ready, busy, error, terminated
    step: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Configuration stored as JSON
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # DP state
    dp_enabled: bool = Field(default=False)
    dp_total_epsilon: float = Field(default=0.0)
    dp_total_delta: float = Field(default=1e-5)

    # Relationships
    futures: List["TinkerFuture"] = Relationship(back_populates="training_client")
    artifacts: List["TinkerArtifact"] = Relationship(back_populates="training_client")


# ==============================================================================
# Future
# ==============================================================================


class TinkerFuture(SQLModel, table=True):
    """Async operation future in the database."""

    __tablename__ = "tinker_futures"

    id: str = Field(default_factory=generate_future_id, primary_key=True)
    training_client_id: str = Field(foreign_key="tinker_training_clients.id", index=True)
    tenant_id: str = Field(index=True)
    operation: str  # forward_backward, optim_step, sample, save_state, load_state
    status: str = Field(default="pending")  # pending, running, completed, failed, cancelled

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Request data (hash only for privacy)
    request_hash: str
    request_size_bytes: int = Field(default=0)

    # Result data stored as JSON (only for completed futures)
    result_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    error_message: Optional[str] = None

    # Priority for queue ordering
    priority: int = Field(default=0)

    # Relationship
    training_client: Optional[TinkerTrainingClient] = Relationship(
        back_populates="futures"
    )


# ==============================================================================
# Artifact
# ==============================================================================


class TinkerArtifact(SQLModel, table=True):
    """Encrypted artifact metadata in the database."""

    __tablename__ = "tinker_artifacts"

    id: str = Field(default_factory=generate_artifact_id, primary_key=True)
    training_client_id: str = Field(foreign_key="tinker_training_clients.id", index=True)
    tenant_id: str = Field(index=True)
    artifact_type: str  # checkpoint, weights, optimizer_state

    # Storage location (path or key in storage backend)
    storage_key: str
    size_bytes: int

    # Encryption metadata
    encryption_algorithm: str = Field(default="AES-256-GCM")
    encryption_key_id: str  # DEK ID
    encryption_nonce: str  # Base64-encoded nonce

    # Content hash for integrity verification
    content_hash: str

    # Custom metadata
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationship
    training_client: Optional[TinkerTrainingClient] = Relationship(
        back_populates="artifacts"
    )


# ==============================================================================
# Audit Log
# ==============================================================================


class TinkerAuditLog(SQLModel, table=True):
    """Append-only audit log with hash chaining."""

    __tablename__ = "tinker_audit_log"

    id: str = Field(default_factory=generate_audit_id, primary_key=True)
    tenant_id: str = Field(index=True)
    training_client_id: str = Field(index=True)
    operation: str  # forward_backward, optim_step, sample, save_state, load_state

    # Request info (hashed for privacy)
    request_hash: str
    request_size_bytes: int = Field(default=0)

    # Artifacts involved
    artifact_ids_produced: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    artifact_ids_consumed: List[str] = Field(default_factory=list, sa_column=Column(JSON))

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Status
    success: bool = Field(default=True)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # Hash chaining for tamper detection
    prev_hash: str  # Hash of previous log entry (or genesis hash)
    record_hash: str  # Hash of this record

    # DP metrics if applicable
    dp_metrics_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Sequence number for ordering
    sequence: int = Field(default=0, index=True)


# ==============================================================================
# Data Encryption Key
# ==============================================================================


class TinkerDataKey(SQLModel, table=True):
    """Per-tenant data encryption key (DEK)."""

    __tablename__ = "tinker_data_keys"

    id: str = Field(default_factory=generate_uuid, primary_key=True)
    tenant_id: str = Field(index=True, unique=True)

    # Wrapped key (encrypted by KEK)
    wrapped_key: str  # Base64-encoded wrapped key
    key_algorithm: str = Field(default="AES-256-GCM")

    # KEK reference
    kek_id: str
    kek_version: int = Field(default=1)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    rotated_at: Optional[datetime] = None

    # Status
    active: bool = Field(default=True)
