"""Status Page API Routes for TenSafe.

FastAPI routes for the public status page at status.tensafe.io.
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel

from .models import (
    ComponentState,
    ComponentStatus,
    CreateIncidentRequest,
    CreateMaintenanceRequest,
    Incident,
    IncidentSeverity,
    IncidentStatus,
    MaintenanceWindow,
    StatusSummaryResponse,
    SubscribeRequest,
    UpdateIncidentRequest,
    UptimeMetrics,
)
from .status import StatusService, get_status_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/status", tags=["status"])


# ==============================================================================
# Dependency
# ==============================================================================


def get_service() -> StatusService:
    """Get status service instance."""
    return get_status_service()


# ==============================================================================
# Response Models
# ==============================================================================


class SubscriptionResponse(BaseModel):
    """Subscription response."""

    id: str
    message: str
    verification_required: bool = True


class OperationResponse(BaseModel):
    """Generic operation response."""

    success: bool
    message: str


class IncidentListResponse(BaseModel):
    """Paginated incident list response."""

    incidents: List[Incident]
    total: int
    has_more: bool


class MaintenanceListResponse(BaseModel):
    """Paginated maintenance list response."""

    maintenance_windows: List[MaintenanceWindow]
    total: int


class UptimeResponse(BaseModel):
    """Uptime metrics response."""

    system: UptimeMetrics
    components: dict[str, UptimeMetrics]


class HistoricalUptimeResponse(BaseModel):
    """Historical uptime response for charting."""

    component_id: str
    period_days: int
    history: List[dict]


# ==============================================================================
# Public Status Endpoints (No Auth Required)
# ==============================================================================


@router.get(
    "",
    response_model=StatusSummaryResponse,
    summary="Current Status Summary",
    description="Get a summary of the current system status. Suitable for status badges and quick checks.",
)
async def get_status_summary(
    service: StatusService = Depends(get_service),
):
    """Get current status summary for status.tensafe.io homepage."""
    system_status = service.get_system_status()

    # Find last incident timestamp
    incidents = service.get_incidents(limit=1)
    last_incident = incidents[0].started_at if incidents else None

    # Calculate 30-day uptime
    uptime_30d = None
    try:
        metrics = service.calculate_uptime("system", period="monthly")
        uptime_30d = metrics.uptime_percentage
    except Exception as e:
        logger.warning(f"Failed to calculate uptime: {e}")

    return StatusSummaryResponse(
        status=system_status.status,
        message=system_status.status_message,
        components={c.id: c.state for c in system_status.components},
        active_incidents_count=len(system_status.active_incidents),
        active_maintenance_count=len(system_status.active_maintenance),
        last_incident=last_incident,
        uptime_30d=uptime_30d,
        timestamp=datetime.utcnow(),
    )


@router.get(
    "/components",
    response_model=List[ComponentStatus],
    summary="Component Status",
    description="Get detailed status for all system components.",
)
async def get_component_status(
    service: StatusService = Depends(get_service),
):
    """Get status for all components."""
    return service.get_all_components()


@router.get(
    "/components/{component_id}",
    response_model=ComponentStatus,
    summary="Single Component Status",
    description="Get status for a specific component.",
)
async def get_single_component_status(
    component_id: str,
    service: StatusService = Depends(get_service),
):
    """Get status for a specific component."""
    component = service.get_component_status(component_id)
    if not component:
        raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")
    return component


@router.get(
    "/incidents",
    response_model=IncidentListResponse,
    summary="Incidents",
    description="Get active and past incidents with optional filtering.",
)
async def get_incidents(
    limit: int = Query(default=20, le=100, description="Maximum incidents to return"),
    include_resolved: bool = Query(default=True, description="Include resolved incidents"),
    severity: Optional[IncidentSeverity] = Query(default=None, description="Filter by severity"),
    component: Optional[str] = Query(default=None, description="Filter by affected component"),
    service: StatusService = Depends(get_service),
):
    """Get incidents with optional filters."""
    incidents = service.get_incidents(
        limit=limit + 1,  # Fetch one extra to check if there's more
        include_resolved=include_resolved,
        severity=severity,
        component=component,
    )

    has_more = len(incidents) > limit
    if has_more:
        incidents = incidents[:limit]

    return IncidentListResponse(
        incidents=incidents,
        total=len(incidents),
        has_more=has_more,
    )


@router.get(
    "/incidents/active",
    response_model=List[Incident],
    summary="Active Incidents",
    description="Get all currently active (unresolved) incidents.",
)
async def get_active_incidents(
    service: StatusService = Depends(get_service),
):
    """Get all active incidents."""
    return service.get_active_incidents()


@router.get(
    "/incidents/{incident_id}",
    response_model=Incident,
    summary="Incident Details",
    description="Get detailed information about a specific incident including all updates.",
)
async def get_incident_details(
    incident_id: str,
    service: StatusService = Depends(get_service),
):
    """Get details for a specific incident."""
    incident = service.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")
    return incident


@router.get(
    "/uptime",
    response_model=UptimeResponse,
    summary="Uptime Metrics",
    description="Get uptime metrics for the system and all components.",
)
async def get_uptime_metrics(
    period: str = Query(default="monthly", regex="^(daily|monthly|yearly)$"),
    service: StatusService = Depends(get_service),
):
    """Get uptime metrics for the specified period."""
    system_metrics = service.calculate_uptime("system", period=period)

    component_metrics = {}
    for component in service.get_all_components():
        component_metrics[component.id] = service.calculate_uptime(
            component.id, period=period
        )

    return UptimeResponse(
        system=system_metrics,
        components=component_metrics,
    )


@router.get(
    "/uptime/{component_id}",
    response_model=UptimeMetrics,
    summary="Component Uptime",
    description="Get uptime metrics for a specific component.",
)
async def get_component_uptime(
    component_id: str,
    period: str = Query(default="monthly", regex="^(daily|monthly|yearly)$"),
    service: StatusService = Depends(get_service),
):
    """Get uptime metrics for a specific component."""
    if component_id != "system":
        component = service.get_component_status(component_id)
        if not component:
            raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")

    return service.calculate_uptime(component_id, period=period)


@router.get(
    "/uptime/{component_id}/history",
    response_model=HistoricalUptimeResponse,
    summary="Historical Uptime",
    description="Get historical daily uptime for charting.",
)
async def get_historical_uptime(
    component_id: str,
    days: int = Query(default=90, le=365, description="Number of days of history"),
    service: StatusService = Depends(get_service),
):
    """Get historical uptime for charting."""
    history = service.get_historical_uptime(component_id, periods=days)

    return HistoricalUptimeResponse(
        component_id=component_id,
        period_days=days,
        history=history,
    )


@router.get(
    "/maintenance",
    response_model=MaintenanceListResponse,
    summary="Scheduled Maintenance",
    description="Get scheduled and active maintenance windows.",
)
async def get_maintenance(
    include_past: bool = Query(default=False, description="Include completed maintenance"),
    limit: int = Query(default=10, le=50),
    service: StatusService = Depends(get_service),
):
    """Get scheduled maintenance windows."""
    windows = service.get_scheduled_maintenance(include_past=include_past, limit=limit)

    return MaintenanceListResponse(
        maintenance_windows=windows,
        total=len(windows),
    )


@router.get(
    "/maintenance/{maintenance_id}",
    response_model=MaintenanceWindow,
    summary="Maintenance Details",
    description="Get details for a specific maintenance window.",
)
async def get_maintenance_details(
    maintenance_id: str,
    service: StatusService = Depends(get_service),
):
    """Get details for a specific maintenance window."""
    maintenance = service.get_maintenance(maintenance_id)
    if not maintenance:
        raise HTTPException(status_code=404, detail=f"Maintenance not found: {maintenance_id}")
    return maintenance


@router.get(
    "/maintenance/upcoming",
    response_model=List[MaintenanceWindow],
    summary="Upcoming Maintenance",
    description="Get upcoming scheduled maintenance windows.",
)
async def get_upcoming_maintenance(
    days: int = Query(default=30, le=90, description="Days to look ahead"),
    service: StatusService = Depends(get_service),
):
    """Get upcoming maintenance windows."""
    return service.get_upcoming_maintenance(days=days)


# ==============================================================================
# Subscription Endpoints (Public)
# ==============================================================================


@router.post(
    "/subscribe",
    response_model=SubscriptionResponse,
    summary="Subscribe to Updates",
    description="Subscribe to status update notifications via email.",
)
async def subscribe_to_updates(
    request: SubscribeRequest,
    background_tasks: BackgroundTasks,
    service: StatusService = Depends(get_service),
):
    """Subscribe to status update emails."""
    subscriber = service.subscribe(request)

    # In production, send verification email in background
    background_tasks.add_task(
        _send_verification_email,
        subscriber.email,
        subscriber.verification_token,
    )

    return SubscriptionResponse(
        id=subscriber.id,
        message="Please check your email to verify your subscription.",
        verification_required=True,
    )


@router.get(
    "/subscribe/verify",
    response_model=OperationResponse,
    summary="Verify Subscription",
    description="Verify email subscription using token from verification email.",
)
async def verify_subscription(
    token: str = Query(..., description="Verification token from email"),
    service: StatusService = Depends(get_service),
):
    """Verify subscription with token."""
    if service.verify_subscription(token):
        return OperationResponse(
            success=True,
            message="Subscription verified successfully.",
        )
    raise HTTPException(status_code=400, detail="Invalid or expired verification token")


@router.get(
    "/unsubscribe",
    response_model=OperationResponse,
    summary="Unsubscribe",
    description="Unsubscribe from status updates using token from email.",
)
async def unsubscribe(
    token: str = Query(..., description="Unsubscribe token from email"),
    service: StatusService = Depends(get_service),
):
    """Unsubscribe from status updates."""
    if service.unsubscribe(token):
        return OperationResponse(
            success=True,
            message="Successfully unsubscribed from status updates.",
        )
    raise HTTPException(status_code=400, detail="Invalid unsubscribe token")


# ==============================================================================
# Admin Endpoints (Require Authentication)
# ==============================================================================


@router.post(
    "/admin/incidents",
    response_model=Incident,
    summary="Create Incident",
    description="Create a new incident. Requires admin authentication.",
    tags=["status-admin"],
)
async def create_incident(
    request: CreateIncidentRequest,
    service: StatusService = Depends(get_service),
    # In production, add: current_user: User = Depends(get_current_admin_user)
):
    """Create a new incident (admin only)."""
    return service.create_incident(request, created_by="admin")


@router.put(
    "/admin/incidents/{incident_id}",
    response_model=Incident,
    summary="Update Incident",
    description="Update an existing incident. Requires admin authentication.",
    tags=["status-admin"],
)
async def update_incident(
    incident_id: str,
    request: UpdateIncidentRequest,
    service: StatusService = Depends(get_service),
):
    """Update an incident (admin only)."""
    incident = service.update_incident(incident_id, request, updated_by="admin")
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")
    return incident


@router.post(
    "/admin/incidents/{incident_id}/resolve",
    response_model=Incident,
    summary="Resolve Incident",
    description="Mark an incident as resolved. Requires admin authentication.",
    tags=["status-admin"],
)
async def resolve_incident(
    incident_id: str,
    message: str = Query(..., min_length=10, description="Resolution message"),
    service: StatusService = Depends(get_service),
):
    """Resolve an incident (admin only)."""
    request = UpdateIncidentRequest(
        status=IncidentStatus.RESOLVED,
        message=message,
    )
    incident = service.update_incident(incident_id, request, updated_by="admin")
    if not incident:
        raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")
    return incident


@router.post(
    "/admin/maintenance",
    response_model=MaintenanceWindow,
    summary="Schedule Maintenance",
    description="Schedule a new maintenance window. Requires admin authentication.",
    tags=["status-admin"],
)
async def schedule_maintenance(
    request: CreateMaintenanceRequest,
    service: StatusService = Depends(get_service),
):
    """Schedule a maintenance window (admin only)."""
    return service.schedule_maintenance(request, created_by="admin")


@router.post(
    "/admin/maintenance/{maintenance_id}/start",
    response_model=MaintenanceWindow,
    summary="Start Maintenance",
    description="Start a scheduled maintenance window. Requires admin authentication.",
    tags=["status-admin"],
)
async def start_maintenance(
    maintenance_id: str,
    service: StatusService = Depends(get_service),
):
    """Start a maintenance window (admin only)."""
    maintenance = service.start_maintenance(maintenance_id)
    if not maintenance:
        raise HTTPException(status_code=404, detail=f"Maintenance not found: {maintenance_id}")
    return maintenance


@router.post(
    "/admin/maintenance/{maintenance_id}/complete",
    response_model=MaintenanceWindow,
    summary="Complete Maintenance",
    description="Mark a maintenance window as completed. Requires admin authentication.",
    tags=["status-admin"],
)
async def complete_maintenance(
    maintenance_id: str,
    service: StatusService = Depends(get_service),
):
    """Complete a maintenance window (admin only)."""
    maintenance = service.complete_maintenance(maintenance_id)
    if not maintenance:
        raise HTTPException(status_code=404, detail=f"Maintenance not found: {maintenance_id}")
    return maintenance


@router.post(
    "/admin/maintenance/{maintenance_id}/cancel",
    response_model=MaintenanceWindow,
    summary="Cancel Maintenance",
    description="Cancel a scheduled maintenance window. Requires admin authentication.",
    tags=["status-admin"],
)
async def cancel_maintenance(
    maintenance_id: str,
    service: StatusService = Depends(get_service),
):
    """Cancel a maintenance window (admin only)."""
    maintenance = service.cancel_maintenance(maintenance_id)
    if not maintenance:
        raise HTTPException(status_code=404, detail=f"Maintenance not found: {maintenance_id}")
    return maintenance


@router.put(
    "/admin/components/{component_id}",
    response_model=ComponentStatus,
    summary="Update Component Status",
    description="Manually update component status. Requires admin authentication.",
    tags=["status-admin"],
)
async def update_component(
    component_id: str,
    state: ComponentState = Query(..., description="New component state"),
    service: StatusService = Depends(get_service),
):
    """Manually update component status (admin only)."""
    component = service.update_component_status(component_id, state)
    if not component:
        raise HTTPException(status_code=404, detail=f"Component not found: {component_id}")
    return component


# ==============================================================================
# Helper Functions
# ==============================================================================


async def _send_verification_email(email: str, token: str):
    """Send verification email (placeholder implementation)."""
    # In production, integrate with email service (SendGrid, SES, etc.)
    verification_url = f"https://status.tensafe.io/api/status/subscribe/verify?token={token}"
    logger.info(f"Verification email would be sent to {email}: {verification_url}")


# ==============================================================================
# OpenAPI Customization
# ==============================================================================


def get_status_openapi_tags():
    """Get OpenAPI tags for status endpoints."""
    return [
        {
            "name": "status",
            "description": "Public status page endpoints for status.tensafe.io",
        },
        {
            "name": "status-admin",
            "description": "Administrative endpoints for managing incidents and maintenance",
        },
    ]
