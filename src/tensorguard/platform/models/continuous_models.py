"""Continuous learning/deployment database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class Route(SQLModel, table=True):
    """Model serving route."""
    __tablename__ = "routes"

    id: Optional[int] = Field(default=None, primary_key=True)
    route_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    path_pattern: str  # e.g., "/v1/completions"
    model_id: str = Field(index=True)
    adapter_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    config_json: Optional[str] = None


class Feed(SQLModel, table=True):
    """Data feed for continuous learning."""
    __tablename__ = "feeds"

    id: Optional[int] = Field(default=None, primary_key=True)
    feed_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    source_type: str  # s3, kafka, webhook
    source_config_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class Policy(SQLModel, table=True):
    """Continuous learning policy."""
    __tablename__ = "policies"

    id: Optional[int] = Field(default=None, primary_key=True)
    policy_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    policy_type: str  # retraining, promotion, rollback
    rules_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class CandidateEvent(SQLModel, table=True):
    """Candidate model event for A/B testing."""
    __tablename__ = "candidate_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    event_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    route_id: str = Field(index=True)
    candidate_id: str = Field(index=True)
    event_type: str  # request, feedback, metric
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details_json: Optional[str] = None


class AdapterLifecycleState(SQLModel, table=True):
    """Adapter lifecycle state tracking."""
    __tablename__ = "adapter_lifecycle_states"

    id: Optional[int] = Field(default=None, primary_key=True)
    state_id: str = Field(index=True, unique=True)
    adapter_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    state: str = Field(default="draft")  # draft, training, evaluating, staged, deployed, archived
    version: int = Field(default=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    promoted_at: Optional[datetime] = None
    retired_at: Optional[datetime] = None
