"""
TGFlow Core Models - Adapter Registry

Database models for adapter lifecycle management:
- AdapterArtifact: Adapter storage and versioning
- AdapterEvalReport: Evaluation results and quality metrics
- AdapterRoute: Routing configuration for adapter selection
- AdapterRelease: Release tracking and channel management
"""

from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
from datetime import datetime
import uuid
from enum import Enum


class AdapterStatus(str, Enum):
    """Adapter lifecycle status."""
    DRAFT = "draft"
    REGISTERED = "registered"
    PROMOTED = "promoted"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ReleaseChannel(str, Enum):
    """Release channel for adapter deployment."""
    CANARY = "canary"
    STAGING = "staging"
    STABLE = "stable"
    ROLLBACK = "rollback"


class AdapterArtifact(SQLModel, table=True):
    """
    Registered adapter artifact with provenance tracking.
    
    Stores adapter metadata, TGSP reference, and evidence chain link.
    """
    __tablename__ = "adapter_artifacts"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Identification
    name: str = Field(index=True)
    version: str = Field(default="1.0.0")
    base_model_id: str = Field(index=True)
    
    # Status
    status: AdapterStatus = Field(default=AdapterStatus.DRAFT)
    
    # Artifacts
    adapter_path: str  # Path to adapter weights
    tgsp_path: Optional[str] = None  # Path to TGSP package
    evidence_path: Optional[str] = None  # Path to evidence chain
    
    # Hashes for integrity
    adapter_hash: str  # SHA-256 of adapter weights
    tgsp_manifest_hash: Optional[str] = None
    evidence_head_hash: Optional[str] = None
    
    # Privacy
    privacy_mode: str = Field(default="off")  # "off" or "n2he"
    privacy_claims_hash: Optional[str] = None
    
    # Metadata
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    labels: Dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AdapterEvalReport(SQLModel, table=True):
    """
    Evaluation report for an adapter.
    
    Captures quality metrics, forgetting scores, and promotion eligibility.
    """
    __tablename__ = "adapter_eval_reports"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    adapter_id: str = Field(index=True, foreign_key="adapter_artifacts.id")
    tenant_id: str = Field(index=True)
    
    # Metrics
    primary_metric: float  # Main evaluation metric (accuracy, F1, etc.)
    forgetting_score: float  # Catastrophic forgetting measure
    regression_score: float = Field(default=0.0)  # Regression on held-out tasks
    
    # Breakdown
    metrics_detail: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Policy evaluation
    promotion_eligible: bool = Field(default=False)
    promotion_threshold: float = Field(default=0.9)
    forgetting_budget: float = Field(default=0.1)
    
    # Gate results
    gates_passed: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    gates_failed: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    
    # N2HE evaluation (if privacy.profile == router_plus_eval)
    n2he_eval_used: bool = Field(default=False)
    n2he_eval_bounds: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    # Timestamps
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


class AdapterRoute(SQLModel, table=True):
    """
    Routing configuration for adapter selection.
    
    Maps route_key to adapter with channel-based deployment.
    """
    __tablename__ = "adapter_routes"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # Routing key (e.g., "customer-support", "code-generation")
    route_key: str = Field(index=True)
    
    # Current adapter per channel
    canary_adapter_id: Optional[str] = Field(default=None, foreign_key="adapter_artifacts.id")
    staging_adapter_id: Optional[str] = Field(default=None, foreign_key="adapter_artifacts.id")
    stable_adapter_id: Optional[str] = Field(default=None, foreign_key="adapter_artifacts.id")
    
    # Rollback tracking
    rollback_adapter_id: Optional[str] = Field(default=None, foreign_key="adapter_artifacts.id")
    rollback_enabled: bool = Field(default=True)
    
    # Traffic split (percentage to canary)
    canary_traffic_percent: float = Field(default=0.0)
    
    # Privacy mode for this route
    privacy_mode: str = Field(default="off")
    
    # Metadata
    config: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AdapterRelease(SQLModel, table=True):
    """
    Release record for adapter deployments.
    
    Tracks promotion history, release decisions, and trust signatures.
    """
    __tablename__ = "adapter_releases"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    
    # What was released
    adapter_id: str = Field(index=True, foreign_key="adapter_artifacts.id")
    route_key: str = Field(index=True)
    channel: ReleaseChannel
    
    # From what
    previous_adapter_id: Optional[str] = Field(default=None)
    
    # Why (evaluation reference)
    eval_report_id: Optional[str] = Field(default=None, foreign_key="adapter_eval_reports.id")
    
    # Trust signature
    signed: bool = Field(default=False)
    signature: Optional[str] = None
    signer_key_id: Optional[str] = None
    
    # Privacy claims
    privacy_mode: str = Field(default="off")
    privacy_receipt_hash: Optional[str] = None
    
    # Decision metadata
    decision_reason: str = Field(default="manual")
    decision_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Timestamps
    released_at: datetime = Field(default_factory=datetime.utcnow)
    released_by: Optional[str] = None
