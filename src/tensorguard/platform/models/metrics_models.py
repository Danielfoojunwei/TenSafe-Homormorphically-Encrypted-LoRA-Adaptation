"""Metrics and monitoring database models."""

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class RouteMetricSeries(SQLModel, table=True):
    """Time-series metrics for routes."""
    __tablename__ = "route_metric_series"

    id: Optional[int] = Field(default=None, primary_key=True)
    route_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    metric_name: str = Field(index=True)
    timestamp: datetime = Field(index=True)
    value: float
    tags_json: Optional[str] = None


class AdapterMetricSnapshot(SQLModel, table=True):
    """Point-in-time adapter metrics."""
    __tablename__ = "adapter_metric_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    snapshot_id: str = Field(index=True, unique=True)
    adapter_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metrics_json: str
    tags_json: Optional[str] = None


class RunStepMetrics(SQLModel, table=True):
    """Metrics for training run steps."""
    __tablename__ = "run_step_metrics"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str = Field(index=True)
    tenant_id: str = Field(index=True)
    step: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    grad_norm: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_used_mb: Optional[float] = None
    extra_metrics_json: Optional[str] = None
