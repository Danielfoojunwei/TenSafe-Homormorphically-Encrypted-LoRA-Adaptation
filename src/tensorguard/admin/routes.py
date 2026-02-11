"""
Admin Console FastAPI Routes.

Provides HTTP endpoints for administrative operations:
- System health and metrics
- Tenant management (CRUD, suspension, activation)
- Dashboard analytics
- Data export
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from .dashboard import DashboardService, get_dashboard_service
from .models import (
    SystemHealth,
    SystemMetrics,
    Tenant,
    TenantCreate,
    TenantList,
    TenantPlan,
    TenantQuota,
    TenantStatus,
    TenantUpdate,
    TenantUsage,
    TopTenantMetric,
)
from .permissions import AdminUserContext, require_org_admin, require_super_admin
from .tenant_management import TenantManager, get_tenant_manager

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/admin", tags=["admin"])


# ==============================================================================
# Response Models
# ==============================================================================


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str


class TenantExportResponse(BaseModel):
    """Response for tenant data export."""

    success: bool = True
    export_path: str
    tenant_id: str


class ActiveJobsResponse(BaseModel):
    """Response for active training jobs."""

    jobs: List[Dict[str, Any]]
    total: int


class ErrorAnalyticsResponse(BaseModel):
    """Response for error analytics."""

    total_errors: int
    error_rate_percent: float
    errors_by_type: Dict[str, int]
    errors_by_endpoint: Dict[str, int]
    errors_by_tenant: Dict[str, int]


# ==============================================================================
# System Health & Metrics
# ==============================================================================


@router.get("/health", response_model=SystemHealth)
async def get_system_health(
    admin_user: AdminUserContext = Depends(require_org_admin),
    dashboard: DashboardService = Depends(get_dashboard_service),
) -> SystemHealth:
    """
    Get comprehensive system health status.

    Returns health information for all system components including
    database, cache, message queue, KMS, HE service, and training workers.

    Requires: org_admin or higher
    """
    return await dashboard.get_system_health()


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    period: str = Query(default="24h", pattern="^(1h|24h|7d|30d)$"),
    include_trends: bool = Query(default=True),
    admin_user: AdminUserContext = Depends(require_org_admin),
    dashboard: DashboardService = Depends(get_dashboard_service),
) -> SystemMetrics:
    """
    Get system-wide metrics and analytics.

    Args:
        period: Time period for metrics (1h, 24h, 7d, 30d)
        include_trends: Whether to include trend data

    Requires: org_admin or higher
    """
    return await dashboard.get_system_metrics(period=period, include_trends=include_trends)


@router.get("/top-tenants", response_model=List[TopTenantMetric])
async def get_top_tenants(
    metric: str = Query(default="requests", pattern="^(requests|compute_hours|storage|users)$"),
    limit: int = Query(default=10, ge=1, le=100),
    period: str = Query(default="30d", pattern="^(7d|30d|90d)$"),
    admin_user: AdminUserContext = Depends(require_super_admin),
    dashboard: DashboardService = Depends(get_dashboard_service),
) -> List[TopTenantMetric]:
    """
    Get top tenants by a specific metric.

    Args:
        metric: Metric to rank by (requests, compute_hours, storage, users)
        limit: Maximum number of tenants to return
        period: Time period for the metric

    Requires: super_admin
    """
    return await dashboard.get_top_tenants_by_usage(metric=metric, limit=limit, period=period)


@router.get("/jobs/active", response_model=ActiveJobsResponse)
async def get_active_jobs(
    tenant_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    admin_user: AdminUserContext = Depends(require_org_admin),
    dashboard: DashboardService = Depends(get_dashboard_service),
) -> ActiveJobsResponse:
    """
    Get list of active training jobs.

    Args:
        tenant_id: Filter by tenant (None for all tenants)
        limit: Maximum number of jobs to return

    Requires: org_admin or higher
    """
    jobs = await dashboard.get_active_training_jobs(tenant_id=tenant_id, limit=limit)
    return ActiveJobsResponse(jobs=jobs, total=len(jobs))


@router.get("/errors", response_model=ErrorAnalyticsResponse)
async def get_error_analytics(
    period: str = Query(default="24h", pattern="^(1h|24h|7d|30d)$"),
    tenant_id: Optional[str] = Query(default=None),
    admin_user: AdminUserContext = Depends(require_org_admin),
    dashboard: DashboardService = Depends(get_dashboard_service),
) -> ErrorAnalyticsResponse:
    """
    Get error analytics and breakdown.

    Args:
        period: Time period for analysis
        tenant_id: Filter by tenant (None for all tenants)

    Requires: org_admin or higher
    """
    analytics = await dashboard.get_error_analytics(period=period, tenant_id=tenant_id)
    return ErrorAnalyticsResponse(
        total_errors=analytics["total_errors"],
        error_rate_percent=analytics["error_rate_percent"],
        errors_by_type=analytics["errors_by_type"],
        errors_by_endpoint=analytics["errors_by_endpoint"],
        errors_by_tenant=analytics["errors_by_tenant"],
    )


# ==============================================================================
# Tenant Management
# ==============================================================================


@router.get("/tenants", response_model=TenantList)
async def list_tenants(
    status: Optional[TenantStatus] = Query(default=None),
    plan: Optional[TenantPlan] = Query(default=None),
    search: Optional[str] = Query(default=None, max_length=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> TenantList:
    """
    List all tenants with filtering and pagination.

    Args:
        status: Filter by tenant status
        plan: Filter by subscription plan
        search: Search in tenant name and slug
        page: Page number (1-indexed)
        page_size: Items per page

    Requires: super_admin
    """
    tenants, total = await manager.list_tenants(
        status=status,
        plan=plan,
        search=search,
        page=page,
        page_size=page_size,
    )
    return TenantList(
        items=tenants,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.post("/tenants", response_model=Tenant, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    request: TenantCreate,
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Create and provision a new tenant.

    This includes:
    - Generating unique tenant ID
    - Creating admin user
    - Setting up resource quotas
    - Provisioning isolated resources

    Requires: super_admin
    """
    try:
        return await manager.create_tenant(request=request, admin_user=admin_user)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/tenants/{tenant_id}", response_model=Tenant)
async def get_tenant(
    tenant_id: str,
    admin_user: AdminUserContext = Depends(require_org_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Get tenant by ID.

    Requires: org_admin or higher
    """
    tenant = await manager.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{tenant_id}' not found",
        )
    return tenant


@router.put("/tenants/{tenant_id}", response_model=Tenant)
async def update_tenant(
    tenant_id: str,
    request: TenantUpdate,
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Update tenant properties.

    Requires: super_admin
    """
    try:
        return await manager.update_tenant(
            tenant_id=tenant_id,
            request=request,
            admin_user=admin_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.delete("/tenants/{tenant_id}", response_model=SuccessResponse)
async def delete_tenant(
    tenant_id: str,
    hard_delete: bool = Query(default=False),
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> SuccessResponse:
    """
    Delete a tenant.

    Args:
        tenant_id: Tenant identifier
        hard_delete: If True, permanently delete all data. If False, soft delete.

    Requires: super_admin
    """
    try:
        await manager.delete_tenant(
            tenant_id=tenant_id,
            admin_user=admin_user,
            hard_delete=hard_delete,
        )
        delete_type = "permanently deleted" if hard_delete else "marked as deleted"
        return SuccessResponse(message=f"Tenant '{tenant_id}' {delete_type}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/tenants/{tenant_id}/suspend", response_model=Tenant)
async def suspend_tenant(
    tenant_id: str,
    reason: str = Query(..., min_length=10, max_length=500),
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Suspend a tenant.

    Suspended tenants cannot access the API but retain their data.

    Args:
        tenant_id: Tenant identifier
        reason: Reason for suspension (min 10 characters)

    Requires: super_admin
    """
    try:
        return await manager.suspend_tenant(
            tenant_id=tenant_id,
            admin_user=admin_user,
            reason=reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/tenants/{tenant_id}/activate", response_model=Tenant)
async def activate_tenant(
    tenant_id: str,
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Activate a suspended or pending tenant.

    Requires: super_admin
    """
    try:
        return await manager.activate_tenant(
            tenant_id=tenant_id,
            admin_user=admin_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.put("/tenants/{tenant_id}/quota", response_model=Tenant)
async def set_tenant_quota(
    tenant_id: str,
    quota: TenantQuota,
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> Tenant:
    """
    Set custom quota for a tenant.

    Requires: super_admin
    """
    try:
        return await manager.set_tenant_quota(
            tenant_id=tenant_id,
            quota=quota,
            admin_user=admin_user,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.get("/tenants/{tenant_id}/usage", response_model=TenantUsage)
async def get_tenant_usage(
    tenant_id: str,
    admin_user: AdminUserContext = Depends(require_org_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> TenantUsage:
    """
    Get current resource usage for a tenant.

    Requires: org_admin or higher
    """
    try:
        return await manager.get_tenant_usage(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/tenants/{tenant_id}/export", response_model=TenantExportResponse)
async def export_tenant_data(
    tenant_id: str,
    include_artifacts: bool = Query(default=True),
    include_audit_logs: bool = Query(default=True),
    admin_user: AdminUserContext = Depends(require_super_admin),
    manager: TenantManager = Depends(get_tenant_manager),
) -> TenantExportResponse:
    """
    Export all tenant data for compliance/portability.

    Creates a ZIP archive containing tenant configuration, user data,
    training jobs, artifacts, and audit logs.

    Args:
        tenant_id: Tenant identifier
        include_artifacts: Whether to include artifacts
        include_audit_logs: Whether to include audit logs

    Requires: super_admin
    """
    try:
        export_path = await manager.export_tenant_data(
            tenant_id=tenant_id,
            admin_user=admin_user,
            include_artifacts=include_artifacts,
            include_audit_logs=include_audit_logs,
        )
        return TenantExportResponse(
            export_path=export_path,
            tenant_id=tenant_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
