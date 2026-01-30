"""PEFT (Parameter-Efficient Fine-Tuning) database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class IntegrationConfig(SQLModel, table=True):
    """External integration configuration."""
    __tablename__ = "integration_configs"

    id: Optional[int] = Field(default=None, primary_key=True)
    config_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    integration_type: str  # huggingface, wandb, mlflow, etc.
    name: str
    config_json: str
    credentials_encrypted: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class PeftWizardDraft(SQLModel, table=True):
    """PEFT configuration wizard draft."""
    __tablename__ = "peft_wizard_drafts"

    id: Optional[int] = Field(default=None, primary_key=True)
    draft_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    user_id: str = Field(index=True)
    name: str
    current_step: str = Field(default="model_selection")
    config_json: str = Field(default="{}")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class PeftRun(SQLModel, table=True):
    """PEFT training run."""
    __tablename__ = "peft_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    job_id: Optional[str] = Field(index=True)
    name: str
    model_ref: str
    peft_type: str = Field(default="lora")
    config_json: str
    status: str = Field(default="pending")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metrics_json: Optional[str] = None
    artifact_ids_json: Optional[str] = None
