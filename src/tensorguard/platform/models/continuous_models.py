"""
Continuous Learning Domain Models

Core domain objects for the Continuous PEFT Control Plane:
- Route: Unit of continuous learning (one route = one evolving adapter family)
- Feed: Reference pointer to data sources (NOT a data lake)
- Policy: Stability vs plasticity controls
- CandidateEvent: Persistent timeline event
- AdapterLifecycleState: FAST/SLOW lane + stage tracking
"""

from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
from datetime import datetime
from enum import Enum
import uuid


# --- Enums ---

class FeedType(str, Enum):
    """Supported feed types (data source references)."""
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    HF_DATASET = "hf_dataset"
    LOCAL = "local"


class PrivacyMode(str, Enum):
    """Privacy mode for data handling."""
    OFF = "off"
    N2HE = "n2he"


class UpdateCadence(str, Enum):
    """How often to check for updates."""
    MANUAL = "manual"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class ConsolidationCadence(str, Enum):
    """How often to consolidate FAST lane into SLOW lane."""
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"


class AdapterLane(str, Enum):
    """Memory lane for adapter (CLS-inspired)."""
    FAST = "fast"   # Recent adaptations, more plastic
    SLOW = "slow"   # Consolidated knowledge, more stable


class AdapterStage(str, Enum):
    """Lifecycle stage of an adapter."""
    CANDIDATE = "candidate"   # Just trained, awaiting evaluation
    SHADOW = "shadow"         # Evaluated, running in background
    CANARY = "canary"         # Active for small traffic/testing
    STABLE = "stable"         # Production traffic
    ARCHIVED = "archived"     # No longer active, kept for rollback
    REVOKED = "revoked"       # Failed integrity/trust check


class EventType(str, Enum):
    """Types of events in the continuous learning loop."""
    FEED_INGESTED = "FEED_INGESTED"
    NOVELTY_LOW = "NOVELTY_LOW"
    UPDATE_PROPOSED = "UPDATE_PROPOSED"
    TRAIN_STARTED = "TRAIN_STARTED"
    TRAIN_DONE = "TRAIN_DONE"
    EVAL_DONE = "EVAL_DONE"
    PACKAGED = "PACKAGED"
    REGISTERED = "REGISTERED"
    PROMOTED = "PROMOTED"
    ROLLED_BACK = "ROLLED_BACK"
    CONSOLIDATED = "CONSOLIDATED"
    ARCHIVED = "ARCHIVED"
    FAILED = "FAILED"
    CONFIG_UPDATED = "CONFIG_UPDATED"
    INFO = "INFO"


# --- Core Domain Models ---

class Route(SQLModel, table=True):
    """
    A Route is the unit of continuous learning.
    
    One route = one evolving adapter family (e.g., "customer_support", "finance_qa").
    """
    __tablename__ = "cl_routes"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Identification
    route_key: str = Field(index=True, unique=True)
    description: Optional[str] = None
    
    # Base model
    base_model_ref: str = Field(index=True)
    
    # Active adapters
    active_adapter_id: Optional[str] = Field(default=None, index=True)
    fallback_adapter_id: Optional[str] = Field(default=None)
    canary_adapter_id: Optional[str] = Field(default=None)
    
    # Status
    enabled: bool = Field(default=True)
    last_loop_at: Optional[datetime] = None
    next_scheduled_at: Optional[datetime] = None
    
    # Metadata
    labels: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Feed(SQLModel, table=True):
    """
    A Feed is a reference pointer to data sources.
    """
    __tablename__ = "cl_feeds"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    route_key: str = Field(index=True)
    
    # Feed configuration
    feed_type: FeedType
    feed_uri: str
    
    # Schema tracking (for drift detection)
    schema_hash: Optional[str] = None
    last_ingest_hash: Optional[str] = None
    last_ingest_at: Optional[datetime] = None
    
    # Privacy
    privacy_mode: PrivacyMode = Field(default=PrivacyMode.OFF)
    
    # Status
    enabled: bool = Field(default=True)
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class Policy(SQLModel, table=True):
    """
    Policy controls continuous update decisions.
    """
    __tablename__ = "cl_policies"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    route_key: str = Field(index=True, unique=True)
    
    # Novelty threshold: below this, skip training
    novelty_threshold: float = Field(default=0.3)
    
    # Quality gates
    promotion_threshold: float = Field(default=0.9)   # Primary metric threshold
    forgetting_budget: float = Field(default=0.1)     # Max allowed forgetting
    regression_budget: float = Field(default=0.05)    # Max regression on held-out
    
    # Adapter caps (bloat control)
    max_fast_adapters: int = Field(default=5)         # Max in FAST lane
    max_total_adapters: int = Field(default=20)       # Total cap before archival
    
    # Cadence
    update_cadence: UpdateCadence = Field(default=UpdateCadence.DAILY)
    consolidation_cadence: ConsolidationCadence = Field(default=ConsolidationCadence.WEEKLY)
    
    # Auto-promotion
    auto_promote_to_canary: bool = Field(default=False)
    auto_promote_to_stable: bool = Field(default=False)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CandidateEvent(SQLModel, table=True):
    """
    Persistent timeline event for the continuous learning loop.
    Replaces ephemeral TimelineEvent for auditability.
    """
    __tablename__ = "cl_candidate_events"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    route_key: str = Field(index=True)
    
    # Event details
    event_type: EventType
    event_payload_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Context
    loop_id: Optional[str] = Field(index=True)
    adapter_id: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AdapterLifecycleState(SQLModel, table=True):
    """
    Tracks the lifecycle state of each adapter within continuous learning.
    """
    __tablename__ = "cl_adapter_states"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Adatper foreign key (linking to core AdapterArtifact)
    adapter_id: str = Field(index=True) 
    route_key: str = Field(index=True)
    
    # Lane (memory type)
    lane: AdapterLane = Field(default=AdapterLane.FAST)
    
    # Stage (lifecycle position)
    stage: AdapterStage = Field(default=AdapterStage.CANDIDATE)
    
    # Lineage
    parent_adapter_id: Optional[str] = None
    
    # Metrics snapshot at creation
    primary_metric: Optional[float] = None
    forgetting_score: Optional[float] = None
    regression_score: Optional[float] = None
    novelty_score: Optional[float] = None
    
    # Evidence references
    tgsp_hash: Optional[str] = None
    evidence_head_hash: Optional[str] = None
    
    # Stage transitions
    promoted_to_canary_at: Optional[datetime] = None
    promoted_to_stable_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoke_reason: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# --- Pydantic Schemas for API ---

from pydantic import BaseModel


class RouteCreate(BaseModel):
    """Schema for creating a route."""
    route_key: str
    base_model_ref: str
    description: Optional[str] = None
    labels: Optional[Dict[str, str]] = None


class FeedConnect(BaseModel):
    """Schema for connecting a feed to a route."""
    feed_type: FeedType
    feed_uri: str
    privacy_mode: PrivacyMode = PrivacyMode.OFF
    config: Optional[Dict[str, Any]] = None


class PolicyUpdate(BaseModel):
    """Schema for updating policy."""
    novelty_threshold: Optional[float] = None
    promotion_threshold: Optional[float] = None
    forgetting_budget: Optional[float] = None
    regression_budget: Optional[float] = None
    max_fast_adapters: Optional[int] = None
    max_total_adapters: Optional[int] = None
    update_cadence: Optional[UpdateCadence] = None
    consolidation_cadence: Optional[ConsolidationCadence] = None
    auto_promote_to_canary: Optional[bool] = None
    auto_promote_to_stable: Optional[bool] = None


class RouteStatus(BaseModel):
    """Human-readable route status."""
    route_key: str
    enabled: bool
    base_model_ref: str
    active_adapter_id: Optional[str]
    canary_adapter_id: Optional[str]
    fallback_adapter_id: Optional[str]
    last_loop_at: Optional[datetime]
    next_scheduled_at: Optional[datetime]
    adapter_count: int
    fast_lane_count: int
    slow_lane_count: int
    privacy_mode: str
