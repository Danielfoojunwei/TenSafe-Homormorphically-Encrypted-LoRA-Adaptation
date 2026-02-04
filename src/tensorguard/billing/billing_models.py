"""
TensorGuard Billing Models.

Pydantic models for billing entities including:
- UsageEvent: Individual usage records
- UsageSummary: Aggregated usage statistics
- PricingTier: Pricing tier definitions
- TenantQuota: Tenant quota state
- Invoice: Billing invoice with line items

Security:
- All models validate input data
- Sensitive fields are excluded from serialization where appropriate
- Timestamps use timezone-aware UTC by default
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class OperationType(str, Enum):
    """Types of billable operations."""

    # API operations
    API_REQUEST = "api_request"
    API_INFERENCE = "api_inference"
    API_EMBEDDING = "api_embedding"

    # Token processing
    TOKENS_INPUT = "tokens_input"
    TOKENS_OUTPUT = "tokens_output"

    # Training operations
    TRAINING_STEP = "training_step"
    TRAINING_EPOCH = "training_epoch"
    FINE_TUNING_JOB = "fine_tuning_job"

    # Compute resources
    GPU_SECONDS = "gpu_seconds"
    GPU_HOURS = "gpu_hours"
    CPU_HOURS = "cpu_hours"

    # Storage
    STORAGE_GB_HOURS = "storage_gb_hours"
    MODEL_STORAGE_GB = "model_storage_gb"

    # Data transfer
    DATA_EGRESS_GB = "data_egress_gb"
    DATA_INGRESS_GB = "data_ingress_gb"

    # Encryption operations
    HE_ENCRYPTION = "he_encryption"
    HE_DECRYPTION = "he_decryption"
    HE_COMPUTATION = "he_computation"


class TierType(str, Enum):
    """Pricing tier types."""

    FREE = "free"
    PRO = "pro"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class QuotaStatus(str, Enum):
    """Quota status levels."""

    OK = "ok"  # Under soft limit
    WARNING = "warning"  # Between soft and hard limit
    EXCEEDED = "exceeded"  # At or over hard limit
    BLOCKED = "blocked"  # Access blocked due to quota


class InvoiceStatus(str, Enum):
    """Invoice status."""

    DRAFT = "draft"
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class UsageEvent(BaseModel):
    """
    Individual usage event record.

    Captures a single billable event with all relevant metadata
    for accurate billing and audit trails.
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event identifier")
    tenant_id: str = Field(..., description="Tenant/organization identifier")
    user_id: Optional[str] = Field(default=None, description="User who triggered the event")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp in UTC",
    )

    # Operation details
    operation_type: OperationType = Field(..., description="Type of billable operation")
    quantity: float = Field(..., ge=0, description="Quantity consumed (e.g., tokens, seconds)")
    unit: str = Field(default="units", description="Unit of measurement")

    # Context
    resource_id: Optional[str] = Field(default=None, description="Related resource ID (job, model, etc.)")
    endpoint: Optional[str] = Field(default=None, description="API endpoint if applicable")
    model_id: Optional[str] = Field(default=None, description="Model identifier if applicable")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional event metadata")

    # Billing
    unit_price: Optional[Decimal] = Field(default=None, description="Price per unit at time of event")
    total_cost: Optional[Decimal] = Field(default=None, description="Total cost for this event")
    currency: str = Field(default="USD", description="Currency code")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            Decimal: str,
        }

    @field_validator("timestamp", mode="before")
    @classmethod
    def ensure_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp has timezone information."""
        if isinstance(v, datetime):
            if v.tzinfo is None:
                return v.replace(tzinfo=timezone.utc)
        return v


class UsageSummary(BaseModel):
    """
    Aggregated usage summary for a tenant.

    Provides rollup statistics for billing periods with
    breakdowns by operation type.
    """

    tenant_id: str = Field(..., description="Tenant identifier")
    period_start: datetime = Field(..., description="Summary period start")
    period_end: datetime = Field(..., description="Summary period end")
    aggregation_level: str = Field(
        default="daily",
        description="Aggregation level (hourly, daily, monthly)",
    )

    # Token usage
    total_tokens_input: int = Field(default=0, description="Total input tokens processed")
    total_tokens_output: int = Field(default=0, description="Total output tokens generated")

    # API usage
    total_api_requests: int = Field(default=0, description="Total API requests made")
    total_inference_requests: int = Field(default=0, description="Total inference requests")
    total_embedding_requests: int = Field(default=0, description="Total embedding requests")

    # Training usage
    total_training_steps: int = Field(default=0, description="Total training steps executed")
    total_training_jobs: int = Field(default=0, description="Total training jobs completed")

    # Compute usage
    total_gpu_seconds: float = Field(default=0.0, description="Total GPU seconds consumed")
    total_gpu_hours: float = Field(default=0.0, description="Total GPU hours consumed")
    total_cpu_hours: float = Field(default=0.0, description="Total CPU hours consumed")

    # Storage
    avg_storage_gb: float = Field(default=0.0, description="Average storage in GB")
    peak_storage_gb: float = Field(default=0.0, description="Peak storage in GB")

    # HE operations
    total_he_operations: int = Field(default=0, description="Total HE operations")

    # Breakdown by operation type
    usage_by_type: Dict[str, float] = Field(
        default_factory=dict,
        description="Usage breakdown by operation type",
    )

    # Cost summary
    total_cost: Decimal = Field(default=Decimal("0.00"), description="Total cost for period")
    cost_by_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Cost breakdown by operation type",
    )
    currency: str = Field(default="USD", description="Currency code")

    # Event counts
    total_events: int = Field(default=0, description="Total events in this period")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }


class TierLimits(BaseModel):
    """Rate and quota limits for a pricing tier."""

    # Token limits
    tokens_per_month: int = Field(..., description="Monthly token limit")
    tokens_per_minute: int = Field(default=0, description="Per-minute token rate limit")

    # Training limits
    training_steps_per_day: int = Field(..., description="Daily training step limit")
    training_steps_per_month: int = Field(default=0, description="Monthly training step limit")
    concurrent_training_jobs: int = Field(default=1, description="Max concurrent training jobs")

    # Compute limits
    gpu_hours_per_month: float = Field(default=0.0, description="Monthly GPU hours limit")
    max_gpu_per_job: int = Field(default=1, description="Max GPUs per training job")

    # API rate limits
    requests_per_minute: int = Field(default=60, description="API requests per minute")
    requests_per_hour: int = Field(default=1000, description="API requests per hour")

    # Storage limits
    storage_gb: float = Field(default=10.0, description="Storage limit in GB")
    model_count: int = Field(default=5, description="Max stored models")

    # Unlimited flag
    unlimited_training: bool = Field(default=False, description="Unlimited training flag")
    unlimited_tokens: bool = Field(default=False, description="Unlimited tokens flag")


class TierFeatures(BaseModel):
    """Feature flags for a pricing tier."""

    # Support
    community_support: bool = Field(default=True, description="Community support access")
    email_support: bool = Field(default=False, description="Email support access")
    priority_support: bool = Field(default=False, description="Priority support access")
    dedicated_support: bool = Field(default=False, description="Dedicated support manager")
    sla: bool = Field(default=False, description="SLA guarantee")

    # Features
    sso: bool = Field(default=False, description="Single Sign-On")
    audit_logs: bool = Field(default=False, description="Audit logging")
    advanced_analytics: bool = Field(default=False, description="Advanced analytics")
    custom_models: bool = Field(default=False, description="Custom model training")
    fine_tuning: bool = Field(default=False, description="Model fine-tuning")
    he_encryption: bool = Field(default=False, description="Homomorphic encryption")
    api_access: bool = Field(default=True, description="API access")
    webhooks: bool = Field(default=False, description="Webhook integrations")
    custom_retention: bool = Field(default=False, description="Custom data retention")

    # Security
    ip_allowlist: bool = Field(default=False, description="IP allowlist feature")
    mfa_required: bool = Field(default=False, description="MFA requirement")
    data_residency: bool = Field(default=False, description="Data residency options")


class PricingTier(BaseModel):
    """
    Complete pricing tier definition.

    Includes tier metadata, limits, features, and pricing information.
    """

    tier_type: TierType = Field(..., description="Tier type identifier")
    name: str = Field(..., description="Display name")
    description: str = Field(default="", description="Tier description")

    # Pricing
    base_price_monthly: Decimal = Field(default=Decimal("0.00"), description="Monthly base price")
    base_price_yearly: Decimal = Field(default=Decimal("0.00"), description="Yearly base price")
    currency: str = Field(default="USD", description="Currency code")

    # Usage-based pricing
    price_per_1k_tokens: Decimal = Field(
        default=Decimal("0.00"),
        description="Price per 1000 tokens",
    )
    price_per_training_step: Decimal = Field(
        default=Decimal("0.00"),
        description="Price per training step",
    )
    price_per_gpu_hour: Decimal = Field(
        default=Decimal("0.00"),
        description="Price per GPU hour",
    )
    price_per_storage_gb_month: Decimal = Field(
        default=Decimal("0.00"),
        description="Price per GB storage per month",
    )

    # Limits and features
    limits: TierLimits = Field(..., description="Tier limits")
    features: TierFeatures = Field(..., description="Tier features")

    # Custom pricing flag
    custom_pricing: bool = Field(default=False, description="Custom pricing negotiated")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            Decimal: str,
        }


class TenantQuota(BaseModel):
    """
    Tenant quota state and tracking.

    Tracks current usage against quota limits with
    soft/hard limit enforcement.
    """

    tenant_id: str = Field(..., description="Tenant identifier")
    tier_type: TierType = Field(..., description="Current pricing tier")

    # Period tracking
    period_start: datetime = Field(..., description="Current billing period start")
    period_end: datetime = Field(..., description="Current billing period end")
    last_reset: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last quota reset timestamp",
    )

    # Token usage
    tokens_used: int = Field(default=0, description="Tokens used this period")
    tokens_limit: int = Field(..., description="Token limit for period")
    tokens_soft_limit: int = Field(default=0, description="Token soft limit (warning)")

    # Training usage
    training_steps_today: int = Field(default=0, description="Training steps used today")
    training_steps_daily_limit: int = Field(..., description="Daily training step limit")
    training_steps_month: int = Field(default=0, description="Training steps this month")

    # Compute usage
    gpu_hours_used: float = Field(default=0.0, description="GPU hours used this period")
    gpu_hours_limit: float = Field(default=0.0, description="GPU hours limit")

    # API usage
    api_requests_minute: int = Field(default=0, description="API requests this minute")
    api_requests_hour: int = Field(default=0, description="API requests this hour")

    # Status
    status: QuotaStatus = Field(default=QuotaStatus.OK, description="Current quota status")
    blocked_until: Optional[datetime] = Field(default=None, description="Blocked until timestamp")
    overage_allowed: bool = Field(default=False, description="Allow overage (with charges)")

    # Warnings
    warning_sent: bool = Field(default=False, description="Warning notification sent")
    warning_sent_at: Optional[datetime] = Field(default=None, description="Warning timestamp")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @model_validator(mode="after")
    def calculate_soft_limit(self) -> "TenantQuota":
        """Calculate soft limit as 80% of hard limit if not set."""
        if self.tokens_soft_limit == 0:
            self.tokens_soft_limit = int(self.tokens_limit * 0.8)
        return self


class InvoiceLineItem(BaseModel):
    """Individual line item on an invoice."""

    line_id: UUID = Field(default_factory=uuid4, description="Line item identifier")
    description: str = Field(..., description="Line item description")
    operation_type: Optional[OperationType] = Field(
        default=None,
        description="Operation type if applicable",
    )

    quantity: float = Field(..., ge=0, description="Quantity")
    unit: str = Field(default="units", description="Unit of measurement")
    unit_price: Decimal = Field(..., description="Price per unit")
    total: Decimal = Field(..., description="Line item total")

    # Period
    period_start: Optional[datetime] = Field(default=None, description="Usage period start")
    period_end: Optional[datetime] = Field(default=None, description="Usage period end")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            Decimal: str,
        }


class Invoice(BaseModel):
    """
    Billing invoice for a tenant.

    Contains all charges for a billing period with
    detailed line items and payment status.
    """

    invoice_id: UUID = Field(default_factory=uuid4, description="Invoice identifier")
    invoice_number: str = Field(..., description="Human-readable invoice number")
    tenant_id: str = Field(..., description="Tenant identifier")

    # Period
    period_start: datetime = Field(..., description="Billing period start")
    period_end: datetime = Field(..., description="Billing period end")
    issued_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Invoice issue date",
    )
    due_date: datetime = Field(..., description="Payment due date")

    # Status
    status: InvoiceStatus = Field(default=InvoiceStatus.DRAFT, description="Invoice status")

    # Amounts
    subtotal: Decimal = Field(default=Decimal("0.00"), description="Subtotal before tax")
    tax_amount: Decimal = Field(default=Decimal("0.00"), description="Tax amount")
    discount_amount: Decimal = Field(default=Decimal("0.00"), description="Discount applied")
    total: Decimal = Field(default=Decimal("0.00"), description="Total amount due")
    currency: str = Field(default="USD", description="Currency code")

    # Line items
    line_items: List[InvoiceLineItem] = Field(
        default_factory=list,
        description="Invoice line items",
    )

    # Tier information
    tier_type: TierType = Field(..., description="Pricing tier for this period")
    base_fee: Decimal = Field(default=Decimal("0.00"), description="Base subscription fee")

    # Payment
    payment_method: Optional[str] = Field(default=None, description="Payment method used")
    paid_at: Optional[datetime] = Field(default=None, description="Payment timestamp")
    payment_reference: Optional[str] = Field(default=None, description="Payment reference ID")

    # Notes
    notes: Optional[str] = Field(default=None, description="Invoice notes")

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: str,
            Decimal: str,
        }

    @model_validator(mode="after")
    def calculate_totals(self) -> "Invoice":
        """Calculate invoice totals from line items."""
        if self.line_items:
            self.subtotal = sum(item.total for item in self.line_items) + self.base_fee
            self.total = self.subtotal + self.tax_amount - self.discount_amount
        return self
