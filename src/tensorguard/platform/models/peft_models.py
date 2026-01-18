from typing import Optional, List, Dict, Any
from sqlmodel import SQLModel, Field, Relationship, Column, JSON
from datetime import datetime
import uuid
from enum import Enum

class ConnectorCategory(str, Enum):
    TRAINING = "training"
    DATA = "data"
    TRACKING = "tracking"
    STORE = "store"
    REGISTRY = "registry"
    ORCHESTRATOR = "orchestrator"
    SECURITY = "security"
    NOTIFY = "notify"

class PeftRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class IntegrationConfig(SQLModel, table=True):
    """Configuration for an MLOps integration connector."""
    __tablename__ = "peft_integration_configs"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    name: str = Field(index=True)
    connector_id: str = Field(index=True) # e.g. "mlflow", "s3"
    category: ConnectorCategory
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PeftWizardDraft(SQLModel, table=True):
    """Saved progress for the PEFT Studio wizard."""
    __tablename__ = "peft_wizard_drafts"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    user_id: str = Field(index=True)
    step: int = Field(default=1)
    draft_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class PeftRun(SQLModel, table=True):
    """A PEFT training run initiated from the Studio."""
    __tablename__ = "peft_runs"
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    created_by_user_id: str
    
    status: PeftRunStatus = Field(default=PeftRunStatus.PENDING)
    stage: str = Field(default="INIT") # e.g. "TRAIN", "PACK", "PUBLISH"
    progress: float = Field(default=0.0) # 0.0 to 100.0
    
    # Configuration
    config_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Metrics and Logs
    metrics_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    log_path: Optional[str] = None
    
    # Policy & Governance
    policy_verdict: Optional[str] = None # PASS/FAIL
    policy_details_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Artifacts
    adapter_path: Optional[str] = None
    tgsp_path: Optional[str] = None
    evidence_path: Optional[str] = None
    release_bundle_path: Optional[str] = None
    registry_ref: Optional[str] = None
    
    # Timeline
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
