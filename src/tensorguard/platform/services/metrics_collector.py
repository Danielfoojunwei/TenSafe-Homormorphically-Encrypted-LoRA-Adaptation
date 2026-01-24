import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from sqlmodel import Session
from ..models.metrics_models import RouteMetricSeries, AdapterMetricSnapshot, RunStepMetrics
from ...metrics.schemas import MetricData, MetricName, MetricCategory, MetricUnit

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Persistence service for all TensorGuardFlow metrics.
    """
    def __init__(self, session: Session):
        self.session = session

    def append_route_series(self, tenant_id: str, route_key: str, 
                            metrics: Dict[MetricName, float], unit_map: Dict[MetricName, str],
                            ts: Optional[datetime] = None):
        """Appends multiple points to the route's time series."""
        ts_val = ts or datetime.utcnow()
        for name, value in metrics.items():
            unit = unit_map.get(name, "none")
            metric = RouteMetricSeries(
                tenant_id=tenant_id,
                route_key=route_key,
                metric_name=name.value if hasattr(name, "value") else name,
                ts=ts_val,
                value=float(value),
                unit=unit,
                metadata_json={}
            )
            self.session.add(metric)
        self.session.commit()

    def write_adapter_snapshot(self, tenant_id: str, adapter_id: str, route_key: str,
                                 metrics: Dict[MetricName, float], unit_map: Dict[MetricName, str]):
        """Records a batch of metrics for a specific adapter."""
        ts = datetime.utcnow()
        for name, value in metrics.items():
            unit = unit_map.get(name, "none")
            snapshot = AdapterMetricSnapshot(
                tenant_id=tenant_id,
                adapter_id=adapter_id,
                route_key=route_key,
                metric_name=name.value if hasattr(name, "value") else name,
                ts=ts,
                value=float(value),
                unit=unit
            )
            self.session.add(snapshot)
        self.session.commit()

    def record_run_step(self, tenant_id: str, run_id: str, route_key: str,
                        step_name: str, duration_ms: float, peak_mem_mb: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Records granular step timing for a loop run."""
        run_metric = RunStepMetrics(
            tenant_id=tenant_id,
            run_id=run_id,
            route_key=route_key,
            step_name=step_name,
            duration_ms=duration_ms,
            peak_mem_mb=peak_mem_mb,
            ts=datetime.utcnow(),
            metadata_json=metadata or {}
        )
        self.session.add(run_metric)
        self.session.commit()

    def record_batch(self, points: List[MetricData]):
        """Handles a batch of MetricData objects."""
        for p in points:
            if p.adapter_id:
                # Store as snapshot
                self.record_adapter_snapshot(
                    p.tenant_id, p.route_key, p.adapter_id, 
                    {p.name: p.value}, {p.name: p.unit}
                )
            # Always store in series for routing trends
            self.record_route_metric(
                p.tenant_id, p.route_key, p.name, p.value, p.unit, p.timestamp, p.metadata
            )
