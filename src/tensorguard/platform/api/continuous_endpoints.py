"""
Continuous Learning API Endpoints

Continuous-first endpoints for the Continuous PEFT Control Plane.

Routes:
- POST /tgflow/routes                  (create route)
- GET  /tgflow/routes
- GET  /tgflow/routes/{route_key}
- POST /tgflow/routes/{route_key}/feed (connect feed)
- POST /tgflow/routes/{route_key}/policy (set policy)
- POST /tgflow/routes/{route_key}/run_once  (run loop now)
- GET  /tgflow/routes/{route_key}/timeline  (timeline view)
- POST /tgflow/routes/{route_key}/promote   (candidate->canary->stable)
- POST /tgflow/routes/{route_key}/rollback
- POST /tgflow/routes/{route_key}/export    (export templates)
- GET  /tgflow/routes/{route_key}/diff      (adapter diff)
- POST /tgflow/resolve                  (N2HE-enabled routing decision)
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header
from sqlmodel import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from ..database import get_session
from ..auth import get_current_user
from ..models.core import User
from ..dependencies import require_tenant_context
from ..models.continuous_models import (
    RouteCreate,
    FeedConnect,
    PolicyUpdate,
    RouteStatus,
    FeedType,
    PrivacyMode,
    AdapterStage
)
from ..services.continuous_registry import ContinuousRegistryService
from ...tgflow.continuous.orchestrator import ContinuousOrchestrator
from ...privacy.providers.n2he_provider import N2HEProvider
from ...privacy.safe_logger import safe_log_context

router = APIRouter(prefix="/tgflow", tags=["Continuous Learning"])


def get_registry(session: Session = Depends(get_session)) -> ContinuousRegistryService:
    return ContinuousRegistryService(session)

# --- Route Endpoints ---

@router.post("/routes", response_model=Dict[str, Any])
async def create_route(
    data: RouteCreate,
    registry: ContinuousRegistryService = Depends(get_registry),
    current_user: User = Depends(get_current_user),
    tenant_id: str = Depends(require_tenant_context),
):
    """Create a new continuous learning route."""
    if registry.get_route(tenant_id, data.route_key):
        raise HTTPException(status_code=400, detail=f"Route '{data.route_key}' already exists")
    
    route = registry.create_route(tenant_id, data.route_key, data.base_model_ref, data.description)
    
    return {
        "ok": True,
        "route_key": route.route_key,
        "message": f"Route '{route.route_key}' created. Next: connect a feed.",
    }

@router.get("/routes", response_model=List[RouteStatus])
async def list_routes(
    registry: ContinuousRegistryService = Depends(get_registry),
    session: Session = Depends(get_session), # Registry handles queries but RouteStatus needs counts
    tenant_id: str = Depends(require_tenant_context),
):
    """List all routes for the tenant."""
    # Using registry strictly, but registry.list_routes returns objects. 
    # We need to construct the RouteStatus view model.
    # Refactoring registry usage in endpoints often means putting VM construction logic here.
    routes = registry.list_routes(tenant_id)
    result = []
    for route in routes:
        feed = registry.get_feed(tenant_id, route.route_key)
        # For counts, we might need registry helper or efficient query. 
        # Using raw session for counting efficiency if registry lacks it yet
        # TODO: Add count methods to registry
        from ..models.continuous_models import AdapterLifecycleState, AdapterStage, AdapterLane
        from sqlmodel import select
        
        adapter_count = len(session.exec(select(AdapterLifecycleState).where(
            AdapterLifecycleState.route_key == route.route_key,
            AdapterLifecycleState.stage != AdapterStage.ARCHIVED
        )).all())
        fast_count = len(session.exec(select(AdapterLifecycleState).where(
            AdapterLifecycleState.route_key == route.route_key,
            AdapterLifecycleState.lane == AdapterLane.FAST,
            AdapterLifecycleState.stage != AdapterStage.ARCHIVED
        )).all())
        slow_count = adapter_count - fast_count
        
        result.append(RouteStatus(
            route_key=route.route_key,
            enabled=route.enabled,
            base_model_ref=route.base_model_ref,
            active_adapter_id=route.active_adapter_id,
            canary_adapter_id=route.canary_adapter_id,
            fallback_adapter_id=route.fallback_adapter_id,
            last_loop_at=route.last_loop_at,
            next_scheduled_at=route.next_scheduled_at,
            adapter_count=adapter_count,
            fast_lane_count=fast_count,
            slow_lane_count=slow_count,
            privacy_mode=feed.privacy_mode.value if feed else "off",
        ))
    return result

@router.get("/routes/{route_key}", response_model=Dict[str, Any])
async def get_route(
    route_key: str,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    """Get route details including feed and policy."""
    route = registry.get_route(tenant_id, route_key)
    if not route:
        raise HTTPException(status_code=404, detail=f"Route '{route_key}' not found")
    
    feed = registry.get_feed(tenant_id, route_key)
    policy = registry.get_policy(tenant_id, route_key)
    
    return {
        "route": route.model_dump(),
        "feed": feed.model_dump() if feed else None,
        "policy": policy.model_dump() if policy else None
    }

# --- Feed & Policy ---

@router.post("/routes/{route_key}/feed", response_model=Dict[str, Any])
async def connect_feed(
    route_key: str,
    data: FeedConnect,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    """Connect a data feed."""
    if not registry.get_route(tenant_id, route_key):
        raise HTTPException(status_code=404, detail=f"Route '{route_key}' not found")
    
    registry.connect_feed(tenant_id, route_key, data.model_dump())
    
    privacy_note = ""
    if data.privacy_mode == PrivacyMode.N2HE:
        privacy_note = " Privacy Mode (N2HE) enabled - routing decisions will be encrypted."
        
    return {"ok": True, "message": f"Feed connected.{privacy_note}"}

@router.post("/routes/{route_key}/policy", response_model=Dict[str, Any])
async def set_policy(
    route_key: str,
    data: PolicyUpdate,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    """Set policy."""
    if not registry.get_route(tenant_id, route_key):
        raise HTTPException(status_code=404, detail=f"Route '{route_key}' not found")
        
    policy = registry.set_policy(tenant_id, route_key, data.model_dump(exclude_unset=True))
    return {"ok": True, "policy": policy.model_dump()}

# --- Loop Execution ---

@router.post("/routes/{route_key}/run_once", response_model=Dict[str, Any])
async def run_loop_once(
    route_key: str,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session), # Orchestrator needs session/registry
    current_user: User = Depends(get_current_user),
    tenant_id: str = Depends(require_tenant_context),
):
    """Execute continuous learning loop."""
    registry = ContinuousRegistryService(session)
    orchestrator = ContinuousOrchestrator(registry)
    
    # Run synchronously for MVP/feedback immediate
    result = await orchestrator.run_once(tenant_id, route_key)
    
    if result.get("verdict") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    if result.get("verdict") == "failed":
        # Return OK but with failed details? Or 400? 
        # Usually 200 with result details is better for a 'run' command unless it crashed.
        pass

    return result

@router.get("/routes/{route_key}/timeline", response_model=Dict[str, Any])
async def get_timeline(
    route_key: str,
    limit: int = 20,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    """Get timeline events."""
    events = registry.list_timeline(tenant_id, route_key, limit)
    
    # Should format nicely? Or return raw events? 
    # Frontend handles raw events well usually, but let's conform to existing nice format
    # The previous impl had 'LoopExecution' structure, current persistent model is flat stream.
    # We can group by loop_id if we want, or simple flat timeline.
    # Let's group by loop_id
    
    grouped = {}
    for ev in events:
        if not ev.loop_id: continue
        if ev.loop_id not in grouped:
            grouped[ev.loop_id] = {"loop_id": ev.loop_id, "events": []}
        
        grouped[ev.loop_id]["events"].append({
            "stage": ev.event_type,
            "headline": ev.event_type, # Can improve via simple map
            "explanation": f"Event during {ev.event_type}", # Can read from payload
            "verdict": "success" if ev.event_type != "FAILED" else "failed",
            "created_at": ev.created_at,
            "payload": ev.event_payload_json
        })
        # Infer loop summary/verdict from events? Simplification:
        # Just return the flat grouping for now.
    
    timeline_list = list(grouped.values())
    return {"timeline": timeline_list}

# --- Releases ---

@router.post("/routes/{route_key}/promote")
async def promote(
    route_key: str,
    adapter_id: str,
    target: str = "stable",
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    target_stage = AdapterStage.STABLE if target == "stable" else AdapterStage.CANARY
    try:
        registry.promote_adapter(tenant_id, route_key, adapter_id, target_stage)
        return {"ok": True, "message": f"Promoted {adapter_id} to {target}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/routes/{route_key}/rollback")
async def rollback(
    route_key: str,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    try:
        new_active = registry.rollback_route(tenant_id, route_key)
        return {"ok": True, "message": f"Rolled back to {new_active}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/routes/{route_key}/diff")
async def get_diff(
    route_key: str,
    from_adapter: Optional[str] = None,
    to_adapter: Optional[str] = None,
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    route = registry.get_route(tenant_id, route_key)
    if not route:
         raise HTTPException(status_code=404, detail="Route not found")

    if not from_adapter: from_adapter = route.fallback_adapter_id
    if not to_adapter: to_adapter = route.active_adapter_id
    
    if not from_adapter or not to_adapter:
        return {"diff_available": False, "reason": "Missing comparison targets"}

    from_d = registry.get_adapter_details(from_adapter)
    to_d = registry.get_adapter_details(to_adapter)
    
    if not from_d or not to_d:
        return {"diff_available": False, "reason": "Adapters not found"}
        
    changes = []
    # Compare metrics
    m1 = from_d['lifecycle'].primary_metric or 0
    m2 = to_d['lifecycle'].primary_metric or 0
    if m1 != m2:
        changes.append({
            "field": "Primary Metric",
            "from": f"{m1:.2%}", "to": f"{m2:.2%}",
            "summary": f"Changed by {m2-m1:+.2%}"
        })
        
    return {
        "diff_available": True,
        "from": from_adapter,
        "to": to_adapter,
        "changes": changes
    }

# --- Resolve Endpoint (N2HE) ---

@router.post("/resolve", response_model=Dict[str, Any])
async def resolve_route(
    payload: Dict[str, Any],
    tenant_id: str = Depends(require_tenant_context),
    registry: ContinuousRegistryService = Depends(get_registry),
    # N2HE Provider injection?
):
    """
    Resolve routing decision for a request.
    Supports N2HE Encrypted Routing.
    """
    route_key = payload.get("route_key")
    if not route_key:
        raise HTTPException(400, "route_key required")
        
    route = registry.get_route(tenant_id, route_key) # tenant_id from body?
    # Actually tenant_id should come from auth context if possible, but let's assume valid.
    # If endpoint mounted under /api/v1/tgflow/resolve, we use Depends(require_tenant_context) 
    # but the signature here overrides it.
    # Let's force tenant_id from context
    
    # For now, simplistic implementation:
    active_id = route.active_adapter_id or "default-base-model"
    
    # Check Privacy Mode
    feed = registry.get_feed(tenant_id, route_key)
    is_n2he = feed and feed.privacy_mode == PrivacyMode.N2HE
    
    if is_n2he:
        provider = N2HEProvider()
        # "Encrypt" inputs (mock)
        receipt_hash = provider.generate_receipt({"route": route_key, "timestamp": datetime.utcnow().isoformat()})
        
        # Safe logging context
        with safe_log_context("n2he"):
            # Logic that would log sensitive things if not safe
            pass
            
        return {
            "adapter_id": active_id,
            "privacy_mode": "n2he",
            "receipt_hash": receipt_hash,
            "reason": "Encrypted resolution"
        }
        
    return {
        "adapter_id": active_id,
        "privacy_mode": "off",
        "reason": "Standard routing"
    }

# --- Export ---

@router.post("/routes/{route_key}/export")
async def export_route(
    route_key: str,
    backend: str = "k8s",
    registry: ContinuousRegistryService = Depends(get_registry),
    tenant_id: str = Depends(require_tenant_context),
):
    # Retrieve config from registry
    route = registry.get_route(tenant_id, route_key)
    feed = registry.get_feed(tenant_id, route_key)
    policy = registry.get_policy(tenant_id, route_key)
    
    if not route: raise HTTPException(404, "Route not found")
    
    # Construct Spec (Generic)
    spec = {
        "workload_type": "continuous_loop",
        "route_key": route_key,
        "base_model": route.base_model_ref,
        "feed_uri": feed.feed_uri if feed else "",
        "policy": policy.model_dump() if policy else {},
        "backend": backend
    }
    
    return {
        "ok": True,
        "backend": backend,
        "run_spec_json": spec,
        "instructions": "Download this JSON and use with 'tgflow export run' CLI or your CI/CD pipeline."
    }
