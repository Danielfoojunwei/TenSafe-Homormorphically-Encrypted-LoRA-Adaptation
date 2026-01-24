from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, Column, JSON
from datetime import datetime
import uuid

class RouteMetricSeries(SQLModel, table=True):
    """Time-series data for a route."""
    __tablename__ = "cl_route_metrics"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    route_key: str = Field(index=True)
    metric_name: str = Field(index=True)
    ts: datetime = Field(default_factory=datetime.utcnow, index=True)
    value: float
    unit: str
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

class AdapterMetricSnapshot(SQLModel, table=True):
    """Metric snapshot for a specific adapter artifact."""
    __tablename__ = "cl_adapter_metrics"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    adapter_id: str = Field(index=True)
    route_key: str = Field(index=True)
    metric_name: str = Field(index=True)
    ts: datetime = Field(default_factory=datetime.utcnow)
    value: float
    unit: str
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))

class RunStepMetrics(SQLModel, table=True):
    """Granular step timing and resource usage per loop run."""
    __tablename__ = "cl_run_metrics"
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    tenant_id: str = Field(index=True)
    run_id: str = Field(index=True)
    route_key: str = Field(index=True)
    step_name: str = Field(index=True)
    duration_ms: float
    peak_mem_mb: Optional[float] = None
    ts: datetime = Field(default_factory=datetime.utcnow)
    metadata_json: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
