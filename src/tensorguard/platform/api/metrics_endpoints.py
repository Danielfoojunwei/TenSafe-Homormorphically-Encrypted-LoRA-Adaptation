from fastapi import APIRouter, Depends, HTTPException, Query
from sqlmodel import Session, select, func
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..database import get_session
from ..models.metrics_models import RouteMetricSeries, AdapterMetricSnapshot, RunStepMetrics
from ..models.continuous_models import Route, CandidateEvent, EventType
from ...metrics.schemas import MetricName, MetricUnit
from ...metrics.route_health import compute_route_health_score
from ...integrations.framework.manager import IntegrationManager

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.get("/routes/summary")
async def get_routes_summary(tenant_id: str, session: Session = Depends(get_session)):
    """
    Returns a list of routes with key KPIs and health scores.
    """
    routes = session.exec(select(Route).where(Route.tenant_id == tenant_id)).all()
    summary = []
    
    for route in routes:
        # Get latest metrics
        latest_metrics = {}
        for m_name in [MetricName.AVG_ACCURACY, MetricName.FORGETTING_MEAN, 
                       MetricName.ADAPTER_COUNT, MetricName.EVIDENCE_COMPLETENESS]:
            val = session.exec(
                select(RouteMetricSeries.value)
                .where(RouteMetricSeries.route_key == route.route_key)
                .where(RouteMetricSeries.metric_name == m_name.value)
                .order_by(RouteMetricSeries.ts.desc())
                .limit(1)
            ).first()
            latest_metrics[m_name.value] = val or 0.0
            
        health = compute_route_health_score(latest_metrics)
        
        summary.append({
            "route_key": route.route_key,
            "base_model": route.base_model_ref,
            "kpis": latest_metrics,
            "health_score": health["score"],
            "health_status": health["status"],
            "health_reasons": health["reasons"],
            "last_update": route.last_loop_at
        })
        
    return summary

@router.get("/routes/{route_key}/timeseries")
async def get_route_timeseries(
    route_key: str, 
    tenant_id: str,
    metric: str = Query(..., description="Metric name to fetch"),
    window_days: int = 30,
    session: Session = Depends(get_session)
):
    """
    Returns time-series data for a specific metric.
    """
    since = datetime.utcnow() - timedelta(days=window_days)
    series = session.exec(
        select(RouteMetricSeries)
        .where(RouteMetricSeries.route_key == route_key)
        .where(RouteMetricSeries.tenant_id == tenant_id)
        .where(RouteMetricSeries.metric_name == metric)
        .where(RouteMetricSeries.ts >= since)
        .order_by(RouteMetricSeries.ts.asc())
    ).all()
    
    return [{"ts": s.ts, "value": s.value, "unit": s.unit} for s in series]

@router.get("/routes/{route_key}/dashboard_bundle")
async def get_dashboard_bundle(
    route_key: str,
    tenant_id: str,
    window_days: int = 30,
    session: Session = Depends(get_session)
):
    """
    One-call bundle for route dashboard.
    """
    # 1. KPIs & Health
    summary = await get_routes_summary(tenant_id, session)
    route_summary = next((s for s in summary if s["route_key"] == route_key), None)
    
    # 2. Key Time Series
    metrics_to_fetch = [
        MetricName.AVG_ACCURACY.value,
        MetricName.FORGETTING_MEAN.value,
        MetricName.BWT.value,
        MetricName.ADAPTER_GROWTH_RATE.value
    ]
    bundle_series = {}
    for m in metrics_to_fetch:
        bundle_series[m] = await get_route_timeseries(route_key, tenant_id, m, window_days, session)
        
    # 3. Latest 20 Events
    events = session.exec(
        select(CandidateEvent)
        .where(CandidateEvent.route_key == route_key)
        .order_by(CandidateEvent.created_at.desc())
        .limit(20)
    ).all()
    
    # 4. Integration Topology mapping (Reuse logic from get_integrations_topology)
    int_manager = IntegrationManager(tenant_id)
    snapshot = await int_manager.get_compatibility_snapshot()
    
    nodes = []
    edges = []
    integrations = snapshot.get("integrations", {})
    for name, data in integrations.items():
        caps = data.get("capabilities", {})
        cat = caps.get("type", "unknown")
        nodes.append({
            "id": name,
            "label": name,
            "category": cat,
            "status": data.get("status"),
            "caps": caps
        })
    
    cat_order = {"data_source": 0, "training_executor": 1, "model_registry": 2, "serving_exporter": 3}
    sorted_nodes = sorted(nodes, key=lambda x: cat_order.get(x["category"], 99))
    for i in range(len(sorted_nodes) - 1):
        edges.append({
            "source": sorted_nodes[i]["id"],
            "target": sorted_nodes[i+1]["id"]
        })
        
    return {
        "summary": route_summary,
        "timeseries": bundle_series,
        "events": events,
        "topology": {"nodes": nodes, "edges": edges}
    }

@router.get("/integrations/topology")
async def get_integrations_topology(tenant_id: str):
    """
    Returns the full pipeline map and tool statuses.
    """
    int_manager = IntegrationManager(tenant_id)
    snapshot = await int_manager.get_compatibility_snapshot()
    
    # Map integrations to C/D/E/F categories for UI
    nodes = []
    edges = []
    
    integrations = snapshot.get("integrations", {})
    for name, data in integrations.items():
        caps = data.get("capabilities", {})
        cat = caps.get("type", "unknown")
        
        nodes.append({
            "id": name,
            "label": name,
            "category": cat,
            "status": data.get("status"),
            "caps": caps
        })
        
    # Logic for edges based on category flow C -> D -> E -> F
    # For now, simple linear or based on orchestrator logic
    cat_order = {"data_source": 0, "training_executor": 1, "model_registry": 2, "serving_exporter": 3}
    sorted_nodes = sorted(nodes, key=lambda x: cat_order.get(x["category"], 99))
    
    for i in range(len(sorted_nodes) - 1):
        edges.append({
            "source": sorted_nodes[i]["id"],
            "target": sorted_nodes[i+1]["id"]
        })
        
    return {"nodes": nodes, "edges": edges}

@router.get("/runs/{run_id}/ops_breakdown")
async def get_ops_breakdown(run_id: str, session: Session = Depends(get_session)):
    """
    Returns step-level performance for a specific run.
    """
    steps = session.exec(
        select(RunStepMetrics)
        .where(RunStepMetrics.run_id == run_id)
        .order_by(RunStepMetrics.ts.asc())
    ).all()
    
    return [s.dict() for s in steps]
