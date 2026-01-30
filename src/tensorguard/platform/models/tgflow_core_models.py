"""TGFlow core database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class TGFlowPipeline(SQLModel, table=True):
    """TGFlow pipeline definition."""
    __tablename__ = "tgflow_pipelines"

    id: Optional[int] = Field(default=None, primary_key=True)
    pipeline_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    description: Optional[str] = None
    definition_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class TGFlowRun(SQLModel, table=True):
    """TGFlow pipeline execution."""
    __tablename__ = "tgflow_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True, unique=True)
    pipeline_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    params_json: Optional[str] = None
    results_json: Optional[str] = None


class TGFlowStep(SQLModel, table=True):
    """TGFlow pipeline step execution."""
    __tablename__ = "tgflow_steps"

    id: Optional[int] = Field(default=None, primary_key=True)
    step_id: str = Field(index=True, unique=True)
    run_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    step_name: str
    step_type: str
    status: str = Field(default="pending")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    inputs_json: Optional[str] = None
    outputs_json: Optional[str] = None
    error_message: Optional[str] = None
