"""VLA (Vision-Language-Action) database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class VLAModel(SQLModel, table=True):
    """Vision-Language-Action model registration."""
    __tablename__ = "vla_models"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_id: str = Field(index=True, unique=True)
    tenant_id: str = Field(index=True)
    name: str
    description: Optional[str] = None
    model_ref: str
    modalities_json: str = Field(default='["vision", "language", "action"]')
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)


class VLASafetyCheck(SQLModel, table=True):
    """VLA model safety check record."""
    __tablename__ = "vla_safety_checks"

    id: Optional[int] = Field(default=None, primary_key=True)
    check_id: str = Field(index=True, unique=True)
    model_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    check_type: str  # boundary, collision, force_limit
    status: str = Field(default="pending")
    passed: Optional[bool] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    details_json: Optional[str] = None


class VLADeploymentLog(SQLModel, table=True):
    """VLA model deployment log."""
    __tablename__ = "vla_deployment_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    deployment_id: str = Field(index=True, unique=True)
    model_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    target_environment: str
    version: str
    deployed_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_by: str
    status: str = Field(default="deployed")
    rollback_from: Optional[str] = None


class VLABenchmarkResult(SQLModel, table=True):
    """VLA model benchmark result."""
    __tablename__ = "vla_benchmark_results"

    id: Optional[int] = Field(default=None, primary_key=True)
    result_id: str = Field(index=True, unique=True)
    model_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    benchmark_name: str
    score: float
    metrics_json: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
