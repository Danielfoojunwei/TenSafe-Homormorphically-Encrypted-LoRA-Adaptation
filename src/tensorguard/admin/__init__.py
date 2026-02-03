"""
TensorGuard Admin Console API.

Provides backend APIs for administrative operations including:
- Tenant management (CRUD, provisioning, deletion)
- User management (CRUD, role assignment)
- System health and metrics
- Dashboard data aggregation
- Maintenance mode control

All operations require admin authentication and are logged to the audit trail.
"""

from .dashboard import DashboardService
from .models import (
    AdminUser,
    AdminUserCreate,
    AdminUserUpdate,
    SystemConfig,
    SystemHealth,
    SystemMetrics,
    Tenant,
    TenantCreate,
    TenantUpdate,
    TenantUsage,
    UsageTrend,
)
from .permissions import (
    AdminRole,
    AuditAction,
    audit_admin_action,
    get_admin_user,
    require_org_admin,
    require_super_admin,
)
from .routes import router as admin_router
from .tenant_management import TenantManager

__all__ = [
    # Router
    "admin_router",
    # Models
    "Tenant",
    "TenantCreate",
    "TenantUpdate",
    "TenantUsage",
    "AdminUser",
    "AdminUserCreate",
    "AdminUserUpdate",
    "SystemHealth",
    "SystemMetrics",
    "SystemConfig",
    "UsageTrend",
    # Permissions
    "AdminRole",
    "AuditAction",
    "require_super_admin",
    "require_org_admin",
    "get_admin_user",
    "audit_admin_action",
    # Services
    "DashboardService",
    "TenantManager",
]
