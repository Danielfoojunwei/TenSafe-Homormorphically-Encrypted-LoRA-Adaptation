"""
TensorGuard Billing Module.

Provides comprehensive usage metering, pricing, and quota management for TenSafe.

Components:
- billing_models: Pydantic models for billing entities
- pricing: Pricing tier definitions and feature flags
- metering: Usage tracking and aggregation
- quota: Quota management and enforcement
- middleware: FastAPI middleware for automatic usage tracking
"""

from .billing_models import (
    Invoice,
    InvoiceLineItem,
    InvoiceStatus,
    OperationType,
    PricingTier,
    QuotaStatus,
    TenantQuota,
    TierType,
    UsageEvent,
    UsageSummary,
)
from .metering import MeteringConfig, MeteringService
from .middleware import UsageMeteringMiddleware
from .pricing import (
    PRICING_TIERS,
    PricingManager,
    get_tier,
    get_tier_features,
    get_tier_limits,
)
from .quota import QuotaConfig, QuotaManager, QuotaViolation

__all__ = [
    # Models
    "UsageEvent",
    "UsageSummary",
    "PricingTier",
    "TenantQuota",
    "Invoice",
    "InvoiceLineItem",
    "InvoiceStatus",
    "OperationType",
    "TierType",
    "QuotaStatus",
    # Metering
    "MeteringService",
    "MeteringConfig",
    # Pricing
    "PRICING_TIERS",
    "PricingManager",
    "get_tier",
    "get_tier_limits",
    "get_tier_features",
    # Quota
    "QuotaManager",
    "QuotaConfig",
    "QuotaViolation",
    # Middleware
    "UsageMeteringMiddleware",
]
