"""
Admin Console Models.

Pydantic models for admin API requests and responses.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ==============================================================================
# Enums
# ==============================================================================


class TenantStatus(str, Enum):
    """Tenant status values."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class TenantPlan(str, Enum):
    """Tenant subscription plan."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class UserStatus(str, Enum):
    """User status values."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"


class ServiceStatus(str, Enum):
    """Service health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# ==============================================================================
# Tenant Models
# ==============================================================================


class TenantQuota(BaseModel):
    """Tenant resource quota configuration."""

    max_users: int = Field(default=10, ge=1, description="Maximum number of users")
    max_training_jobs: int = Field(default=5, ge=0, description="Maximum concurrent training jobs")
    max_storage_gb: float = Field(default=10.0, ge=0, description="Maximum storage in GB")
    max_compute_hours: float = Field(default=100.0, ge=0, description="Maximum compute hours per month")
    max_api_requests_per_day: int = Field(default=10000, ge=0, description="Maximum API requests per day")
    dp_budget_epsilon: float = Field(default=10.0, ge=0, description="Maximum differential privacy budget")
    he_operations_per_day: int = Field(default=1000, ge=0, description="Maximum HE operations per day")


class TenantCreate(BaseModel):
    """Request model for creating a tenant."""

    name: str = Field(..., min_length=2, max_length=100, description="Tenant display name")
    slug: str = Field(..., min_length=2, max_length=50, pattern=r"^[a-z0-9][a-z0-9-]*[a-z0-9]$", description="URL-safe tenant identifier")
    plan: TenantPlan = Field(default=TenantPlan.FREE, description="Subscription plan")
    admin_email: EmailStr = Field(..., description="Primary admin email")
    admin_name: str = Field(..., min_length=2, max_length=100, description="Primary admin name")
    quota: Optional[TenantQuota] = Field(default=None, description="Custom quota (uses plan defaults if not provided)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional tenant metadata")

    @field_validator("slug")
    @classmethod
    def validate_slug(cls, v: str) -> str:
        """Validate slug format."""
        reserved = {"admin", "api", "system", "internal", "public", "private"}
        if v.lower() in reserved:
            raise ValueError(f"Slug '{v}' is reserved")
        return v.lower()


class TenantUpdate(BaseModel):
    """Request model for updating a tenant."""

    name: Optional[str] = Field(None, min_length=2, max_length=100)
    plan: Optional[TenantPlan] = None
    status: Optional[TenantStatus] = None
    quota: Optional[TenantQuota] = None
    metadata: Optional[Dict[str, Any]] = None


class Tenant(BaseModel):
    """Tenant response model."""

    id: str = Field(..., description="Unique tenant identifier")
    name: str = Field(..., description="Tenant display name")
    slug: str = Field(..., description="URL-safe tenant identifier")
    plan: TenantPlan = Field(..., description="Subscription plan")
    status: TenantStatus = Field(..., description="Current status")
    quota: TenantQuota = Field(..., description="Resource quota")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    user_count: int = Field(default=0, description="Number of users")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        from_attributes = True


class TenantUsage(BaseModel):
    """Tenant resource usage metrics."""

    tenant_id: str
    period_start: datetime
    period_end: datetime

    # User metrics
    total_users: int = 0
    active_users: int = 0

    # Compute metrics
    training_jobs_count: int = 0
    training_jobs_completed: int = 0
    compute_hours_used: float = 0.0

    # Storage metrics
    storage_used_gb: float = 0.0
    artifacts_count: int = 0

    # API metrics
    api_requests_count: int = 0
    api_errors_count: int = 0

    # Privacy metrics
    dp_budget_consumed: float = 0.0
    he_operations_count: int = 0

    # Cost metrics (for billing)
    estimated_cost_usd: float = 0.0


class TenantList(BaseModel):
    """Paginated tenant list response."""

    items: List[Tenant]
    total: int
    page: int
    page_size: int
    has_more: bool


# ==============================================================================
# User Models
# ==============================================================================


class AdminUserRole(str, Enum):
    """Admin user roles."""

    SUPER_ADMIN = "super_admin"
    ORG_ADMIN = "org_admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class AdminUserCreate(BaseModel):
    """Request model for creating an admin user."""

    email: EmailStr = Field(..., description="User email address")
    name: str = Field(..., min_length=2, max_length=100, description="User display name")
    role: AdminUserRole = Field(default=AdminUserRole.VIEWER, description="Admin role")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (required for org_admin, optional for super_admin)")
    password: Optional[str] = Field(None, min_length=12, description="Initial password (generated if not provided)")
    send_invite: bool = Field(default=True, description="Send invitation email")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional user metadata")


class AdminUserUpdate(BaseModel):
    """Request model for updating an admin user."""

    name: Optional[str] = Field(None, min_length=2, max_length=100)
    role: Optional[AdminUserRole] = None
    status: Optional[UserStatus] = None
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AdminUser(BaseModel):
    """Admin user response model."""

    id: str = Field(..., description="Unique user identifier")
    email: str = Field(..., description="User email address")
    name: str = Field(..., description="User display name")
    role: AdminUserRole = Field(..., description="Admin role")
    tenant_id: Optional[str] = Field(None, description="Associated tenant ID")
    status: UserStatus = Field(..., description="User status")
    created_at: datetime = Field(..., description="Creation timestamp")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    mfa_enabled: bool = Field(default=False, description="MFA enabled status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        from_attributes = True


class AdminUserList(BaseModel):
    """Paginated admin user list response."""

    items: List[AdminUser]
    total: int
    page: int
    page_size: int
    has_more: bool


# ==============================================================================
# System Health Models
# ==============================================================================


class ServiceHealth(BaseModel):
    """Individual service health status."""

    name: str
    status: ServiceStatus
    latency_ms: Optional[float] = None
    last_check: datetime
    details: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class SystemHealth(BaseModel):
    """Overall system health status."""

    status: ServiceStatus = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

    # Component health
    services: List[ServiceHealth] = Field(default_factory=list, description="Individual service health")

    # Resource utilization
    cpu_percent: float = Field(default=0.0, description="CPU utilization percentage")
    memory_percent: float = Field(default=0.0, description="Memory utilization percentage")
    disk_percent: float = Field(default=0.0, description="Disk utilization percentage")

    # Additional details
    active_connections: int = Field(default=0, description="Active connections")
    pending_jobs: int = Field(default=0, description="Pending jobs in queue")


# ==============================================================================
# System Metrics Models
# ==============================================================================


class MetricDataPoint(BaseModel):
    """Single metric data point."""

    timestamp: datetime
    value: float
    labels: Dict[str, str] = Field(default_factory=dict)


class UsageTrend(BaseModel):
    """Usage trend data for a metric."""

    metric_name: str
    period: str  # "7d", "30d", "90d"
    data_points: List[MetricDataPoint]
    total: float
    average: float
    min_value: float
    max_value: float
    trend_percent: float  # Percentage change from previous period


class SystemMetrics(BaseModel):
    """System-wide metrics."""

    timestamp: datetime
    period: str  # "1h", "24h", "7d", "30d"

    # Tenant metrics
    total_tenants: int = 0
    active_tenants: int = 0
    new_tenants_period: int = 0

    # User metrics
    total_users: int = 0
    active_users: int = 0
    new_users_period: int = 0

    # Request metrics
    total_requests: int = 0
    requests_per_second: float = 0.0
    error_rate_percent: float = 0.0
    avg_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Training metrics
    active_training_jobs: int = 0
    completed_jobs_period: int = 0
    failed_jobs_period: int = 0
    total_compute_hours: float = 0.0

    # Storage metrics
    total_storage_used_gb: float = 0.0
    total_artifacts: int = 0

    # Privacy metrics
    total_dp_budget_consumed: float = 0.0
    total_he_operations: int = 0

    # Usage trends
    trends: List[UsageTrend] = Field(default_factory=list)


class TopTenantMetric(BaseModel):
    """Top tenant metric for dashboard."""

    tenant_id: str
    tenant_name: str
    metric_value: float
    metric_unit: str


# ==============================================================================
# System Configuration Models
# ==============================================================================


class MaintenanceMode(BaseModel):
    """Maintenance mode configuration."""

    enabled: bool = False
    message: str = "System is under maintenance. Please try again later."
    allowed_ips: List[str] = Field(default_factory=list, description="IPs allowed during maintenance")
    estimated_end: Optional[datetime] = None
    started_at: Optional[datetime] = None
    started_by: Optional[str] = None


class FeatureFlags(BaseModel):
    """Feature flag configuration."""

    enable_new_users: bool = True
    enable_new_tenants: bool = True
    enable_training: bool = True
    enable_inference: bool = True
    enable_he_operations: bool = True
    enable_dp_training: bool = True
    enable_webhooks: bool = True
    enable_billing: bool = True


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""

    default_requests_per_minute: int = 60
    default_requests_per_hour: int = 1000
    default_requests_per_day: int = 10000
    burst_multiplier: float = 2.0


class SystemConfig(BaseModel):
    """System configuration response."""

    maintenance_mode: MaintenanceMode
    feature_flags: FeatureFlags
    rate_limits: RateLimitConfig
    version: str
    environment: str
    region: str
    last_updated: datetime
    last_updated_by: Optional[str] = None


class MaintenanceModeRequest(BaseModel):
    """Request to enable/disable maintenance mode."""

    enabled: bool
    message: Optional[str] = None
    allowed_ips: Optional[List[str]] = None
    estimated_duration_minutes: Optional[int] = None


# ==============================================================================
# Audit Models
# ==============================================================================


class AdminAuditLog(BaseModel):
    """Admin action audit log entry."""

    id: str
    timestamp: datetime
    admin_user_id: str
    admin_user_email: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    target_tenant_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None


class AdminAuditLogList(BaseModel):
    """Paginated admin audit log list."""

    items: List[AdminAuditLog]
    total: int
    page: int
    page_size: int
    has_more: bool


# ==============================================================================
# Response Models
# ==============================================================================


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = True
    message: str


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: ErrorDetail
