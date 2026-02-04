"""
TensorGuard Pricing Tier Definitions.

Defines pricing tiers for TenSafe with associated limits, features, and pricing.

Tiers:
- Free: Entry-level with basic limits
- Pro: Professional tier for individuals/small teams
- Business: Business tier with advanced features
- Enterprise: Custom enterprise agreements

Usage:
    from tensorguard.billing.pricing import PRICING_TIERS, get_tier

    tier = get_tier(TierType.PRO)
    limits = get_tier_limits(TierType.PRO)
    features = get_tier_features(TierType.PRO)
"""

import logging
from decimal import Decimal
from typing import Dict, Optional

from .billing_models import (
    PricingTier,
    TierFeatures,
    TierLimits,
    TierType,
)

logger = logging.getLogger(__name__)

# =============================================================================
# TIER DEFINITIONS
# =============================================================================

FREE_TIER = PricingTier(
    tier_type=TierType.FREE,
    name="Free",
    description="Get started with TenSafe for free. Perfect for exploration and small projects.",
    base_price_monthly=Decimal("0.00"),
    base_price_yearly=Decimal("0.00"),
    price_per_1k_tokens=Decimal("0.00"),
    price_per_training_step=Decimal("0.00"),
    price_per_gpu_hour=Decimal("0.00"),
    price_per_storage_gb_month=Decimal("0.00"),
    limits=TierLimits(
        # 100K tokens per month
        tokens_per_month=100_000,
        tokens_per_minute=1_000,
        # 10 training steps per day
        training_steps_per_day=10,
        training_steps_per_month=300,
        concurrent_training_jobs=1,
        # No GPU hours included
        gpu_hours_per_month=0.0,
        max_gpu_per_job=1,
        # API limits
        requests_per_minute=20,
        requests_per_hour=200,
        # Storage
        storage_gb=1.0,
        model_count=2,
        unlimited_training=False,
        unlimited_tokens=False,
    ),
    features=TierFeatures(
        community_support=True,
        email_support=False,
        priority_support=False,
        dedicated_support=False,
        sla=False,
        sso=False,
        audit_logs=False,
        advanced_analytics=False,
        custom_models=False,
        fine_tuning=False,
        he_encryption=False,
        api_access=True,
        webhooks=False,
        custom_retention=False,
        ip_allowlist=False,
        mfa_required=False,
        data_residency=False,
    ),
)

PRO_TIER = PricingTier(
    tier_type=TierType.PRO,
    name="Pro",
    description="For professionals and growing teams. Includes email support and more resources.",
    base_price_monthly=Decimal("49.00"),
    base_price_yearly=Decimal("490.00"),  # ~2 months free
    price_per_1k_tokens=Decimal("0.002"),
    price_per_training_step=Decimal("0.01"),
    price_per_gpu_hour=Decimal("2.50"),
    price_per_storage_gb_month=Decimal("0.10"),
    limits=TierLimits(
        # 10M tokens per month
        tokens_per_month=10_000_000,
        tokens_per_minute=10_000,
        # 1000 training steps per day
        training_steps_per_day=1_000,
        training_steps_per_month=30_000,
        concurrent_training_jobs=3,
        # 10 GPU hours included
        gpu_hours_per_month=10.0,
        max_gpu_per_job=2,
        # API limits
        requests_per_minute=100,
        requests_per_hour=2_000,
        # Storage
        storage_gb=50.0,
        model_count=20,
        unlimited_training=False,
        unlimited_tokens=False,
    ),
    features=TierFeatures(
        community_support=True,
        email_support=True,
        priority_support=False,
        dedicated_support=False,
        sla=False,
        sso=False,
        audit_logs=True,
        advanced_analytics=False,
        custom_models=True,
        fine_tuning=True,
        he_encryption=True,
        api_access=True,
        webhooks=True,
        custom_retention=False,
        ip_allowlist=False,
        mfa_required=False,
        data_residency=False,
    ),
)

BUSINESS_TIER = PricingTier(
    tier_type=TierType.BUSINESS,
    name="Business",
    description="For organizations needing advanced features, SSO, and priority support.",
    base_price_monthly=Decimal("299.00"),
    base_price_yearly=Decimal("2990.00"),  # ~2 months free
    price_per_1k_tokens=Decimal("0.0015"),
    price_per_training_step=Decimal("0.008"),
    price_per_gpu_hour=Decimal("2.00"),
    price_per_storage_gb_month=Decimal("0.08"),
    limits=TierLimits(
        # 100M tokens per month
        tokens_per_month=100_000_000,
        tokens_per_minute=50_000,
        # Unlimited training
        training_steps_per_day=0,  # 0 = unlimited when unlimited_training=True
        training_steps_per_month=0,
        concurrent_training_jobs=10,
        # 100 GPU hours included
        gpu_hours_per_month=100.0,
        max_gpu_per_job=8,
        # API limits
        requests_per_minute=500,
        requests_per_hour=10_000,
        # Storage
        storage_gb=500.0,
        model_count=100,
        unlimited_training=True,
        unlimited_tokens=False,
    ),
    features=TierFeatures(
        community_support=True,
        email_support=True,
        priority_support=True,
        dedicated_support=False,
        sla=True,
        sso=True,
        audit_logs=True,
        advanced_analytics=True,
        custom_models=True,
        fine_tuning=True,
        he_encryption=True,
        api_access=True,
        webhooks=True,
        custom_retention=True,
        ip_allowlist=True,
        mfa_required=True,
        data_residency=False,
    ),
)

ENTERPRISE_TIER = PricingTier(
    tier_type=TierType.ENTERPRISE,
    name="Enterprise",
    description="Custom enterprise solutions with dedicated support, SLA guarantees, and data residency options.",
    base_price_monthly=Decimal("0.00"),  # Custom pricing
    base_price_yearly=Decimal("0.00"),
    price_per_1k_tokens=Decimal("0.00"),  # Negotiated
    price_per_training_step=Decimal("0.00"),
    price_per_gpu_hour=Decimal("0.00"),
    price_per_storage_gb_month=Decimal("0.00"),
    custom_pricing=True,
    limits=TierLimits(
        # Custom limits - these are defaults that can be overridden
        tokens_per_month=0,  # Unlimited
        tokens_per_minute=100_000,
        training_steps_per_day=0,
        training_steps_per_month=0,
        concurrent_training_jobs=0,  # Unlimited
        gpu_hours_per_month=0.0,  # Unlimited
        max_gpu_per_job=0,  # Unlimited
        requests_per_minute=1_000,
        requests_per_hour=50_000,
        storage_gb=0.0,  # Unlimited
        model_count=0,  # Unlimited
        unlimited_training=True,
        unlimited_tokens=True,
    ),
    features=TierFeatures(
        community_support=True,
        email_support=True,
        priority_support=True,
        dedicated_support=True,
        sla=True,
        sso=True,
        audit_logs=True,
        advanced_analytics=True,
        custom_models=True,
        fine_tuning=True,
        he_encryption=True,
        api_access=True,
        webhooks=True,
        custom_retention=True,
        ip_allowlist=True,
        mfa_required=True,
        data_residency=True,
    ),
)

# =============================================================================
# PRICING TIERS REGISTRY
# =============================================================================

PRICING_TIERS: Dict[TierType, PricingTier] = {
    TierType.FREE: FREE_TIER,
    TierType.PRO: PRO_TIER,
    TierType.BUSINESS: BUSINESS_TIER,
    TierType.ENTERPRISE: ENTERPRISE_TIER,
}


# =============================================================================
# PRICING UTILITIES
# =============================================================================


def get_tier(tier_type: TierType) -> PricingTier:
    """
    Get pricing tier definition.

    Args:
        tier_type: The tier type to retrieve

    Returns:
        PricingTier definition

    Raises:
        ValueError: If tier type is not found
    """
    if tier_type not in PRICING_TIERS:
        raise ValueError(f"Unknown tier type: {tier_type}")
    return PRICING_TIERS[tier_type]


def get_tier_limits(tier_type: TierType) -> TierLimits:
    """
    Get limits for a pricing tier.

    Args:
        tier_type: The tier type

    Returns:
        TierLimits for the tier
    """
    return get_tier(tier_type).limits


def get_tier_features(tier_type: TierType) -> TierFeatures:
    """
    Get features for a pricing tier.

    Args:
        tier_type: The tier type

    Returns:
        TierFeatures for the tier
    """
    return get_tier(tier_type).features


class PricingManager:
    """
    Manages pricing calculations and tier comparisons.

    Provides utilities for:
    - Calculating usage costs
    - Comparing tier features
    - Suggesting tier upgrades
    - Handling custom enterprise pricing
    """

    def __init__(self, custom_tiers: Optional[Dict[str, PricingTier]] = None):
        """
        Initialize pricing manager.

        Args:
            custom_tiers: Optional custom tier overrides for enterprise tenants
        """
        self._tiers = PRICING_TIERS.copy()
        self._custom_tiers: Dict[str, PricingTier] = custom_tiers or {}

    def get_tier_for_tenant(
        self,
        tenant_id: str,
        tier_type: TierType,
    ) -> PricingTier:
        """
        Get pricing tier for a specific tenant.

        Checks for custom enterprise pricing first.

        Args:
            tenant_id: Tenant identifier
            tier_type: Base tier type

        Returns:
            PricingTier (custom or standard)
        """
        # Check for custom tier
        custom_key = f"{tenant_id}:{tier_type.value}"
        if custom_key in self._custom_tiers:
            return self._custom_tiers[custom_key]

        return get_tier(tier_type)

    def set_custom_tier(
        self,
        tenant_id: str,
        tier: PricingTier,
    ) -> None:
        """
        Set custom pricing tier for a tenant.

        Args:
            tenant_id: Tenant identifier
            tier: Custom PricingTier
        """
        custom_key = f"{tenant_id}:{tier.tier_type.value}"
        self._custom_tiers[custom_key] = tier
        logger.info(f"Set custom tier for tenant {tenant_id}: {tier.name}")

    def calculate_token_cost(
        self,
        tier_type: TierType,
        token_count: int,
        tenant_id: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate cost for token usage.

        Args:
            tier_type: Pricing tier
            token_count: Number of tokens
            tenant_id: Optional tenant ID for custom pricing

        Returns:
            Cost in tier currency
        """
        if tenant_id:
            tier = self.get_tier_for_tenant(tenant_id, tier_type)
        else:
            tier = get_tier(tier_type)

        # Cost per 1K tokens
        return (Decimal(token_count) / Decimal("1000")) * tier.price_per_1k_tokens

    def calculate_training_cost(
        self,
        tier_type: TierType,
        training_steps: int,
        tenant_id: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate cost for training steps.

        Args:
            tier_type: Pricing tier
            training_steps: Number of training steps
            tenant_id: Optional tenant ID for custom pricing

        Returns:
            Cost in tier currency
        """
        if tenant_id:
            tier = self.get_tier_for_tenant(tenant_id, tier_type)
        else:
            tier = get_tier(tier_type)

        return Decimal(training_steps) * tier.price_per_training_step

    def calculate_gpu_cost(
        self,
        tier_type: TierType,
        gpu_hours: float,
        tenant_id: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate cost for GPU usage.

        Args:
            tier_type: Pricing tier
            gpu_hours: GPU hours consumed
            tenant_id: Optional tenant ID for custom pricing

        Returns:
            Cost in tier currency
        """
        if tenant_id:
            tier = self.get_tier_for_tenant(tenant_id, tier_type)
        else:
            tier = get_tier(tier_type)

        return Decimal(str(gpu_hours)) * tier.price_per_gpu_hour

    def calculate_storage_cost(
        self,
        tier_type: TierType,
        storage_gb: float,
        tenant_id: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate monthly cost for storage.

        Args:
            tier_type: Pricing tier
            storage_gb: Storage in GB
            tenant_id: Optional tenant ID for custom pricing

        Returns:
            Monthly cost in tier currency
        """
        if tenant_id:
            tier = self.get_tier_for_tenant(tenant_id, tier_type)
        else:
            tier = get_tier(tier_type)

        return Decimal(str(storage_gb)) * tier.price_per_storage_gb_month

    def suggest_upgrade(
        self,
        current_tier: TierType,
        monthly_tokens: int,
        monthly_training_steps: int,
        required_features: Optional[list] = None,
    ) -> Optional[TierType]:
        """
        Suggest tier upgrade based on usage patterns.

        Args:
            current_tier: Current tier type
            monthly_tokens: Projected monthly token usage
            monthly_training_steps: Projected monthly training steps
            required_features: List of required feature names

        Returns:
            Suggested tier type or None if current is sufficient
        """
        current = get_tier(current_tier)

        # Check if current tier is sufficient
        needs_upgrade = False
        reason = []

        # Check token limits
        if not current.limits.unlimited_tokens:
            if monthly_tokens > current.limits.tokens_per_month:
                needs_upgrade = True
                reason.append("token_limit")

        # Check training limits
        if not current.limits.unlimited_training:
            if monthly_training_steps > current.limits.training_steps_per_month:
                needs_upgrade = True
                reason.append("training_limit")

        # Check features
        if required_features:
            for feature in required_features:
                if hasattr(current.features, feature):
                    if not getattr(current.features, feature):
                        needs_upgrade = True
                        reason.append(f"feature:{feature}")

        if not needs_upgrade:
            return None

        # Find the best tier
        tier_order = [TierType.FREE, TierType.PRO, TierType.BUSINESS, TierType.ENTERPRISE]
        current_index = tier_order.index(current_tier)

        for tier_type in tier_order[current_index + 1 :]:
            tier = get_tier(tier_type)

            # Check if this tier meets requirements
            meets_tokens = tier.limits.unlimited_tokens or monthly_tokens <= tier.limits.tokens_per_month
            meets_training = (
                tier.limits.unlimited_training or monthly_training_steps <= tier.limits.training_steps_per_month
            )

            meets_features = True
            if required_features:
                for feature in required_features:
                    if hasattr(tier.features, feature):
                        if not getattr(tier.features, feature):
                            meets_features = False
                            break

            if meets_tokens and meets_training and meets_features:
                logger.info(f"Suggesting upgrade from {current_tier} to {tier_type}: {reason}")
                return tier_type

        # Enterprise is always the fallback
        return TierType.ENTERPRISE

    def compare_tiers(
        self,
        tier_a: TierType,
        tier_b: TierType,
    ) -> Dict[str, Dict[str, any]]:
        """
        Compare two pricing tiers.

        Args:
            tier_a: First tier
            tier_b: Second tier

        Returns:
            Comparison dictionary with differences
        """
        a = get_tier(tier_a)
        b = get_tier(tier_b)

        comparison = {
            "pricing": {
                "monthly_difference": float(b.base_price_monthly - a.base_price_monthly),
                "yearly_difference": float(b.base_price_yearly - a.base_price_yearly),
            },
            "limits": {},
            "features": {},
        }

        # Compare limits
        a_limits = a.limits.model_dump()
        b_limits = b.limits.model_dump()

        for key in a_limits:
            if a_limits[key] != b_limits[key]:
                comparison["limits"][key] = {
                    tier_a.value: a_limits[key],
                    tier_b.value: b_limits[key],
                }

        # Compare features
        a_features = a.features.model_dump()
        b_features = b.features.model_dump()

        for key in a_features:
            if a_features[key] != b_features[key]:
                comparison["features"][key] = {
                    tier_a.value: a_features[key],
                    tier_b.value: b_features[key],
                }

        return comparison


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FREE_TIER",
    "PRO_TIER",
    "BUSINESS_TIER",
    "ENTERPRISE_TIER",
    "PRICING_TIERS",
    "get_tier",
    "get_tier_limits",
    "get_tier_features",
    "PricingManager",
]
