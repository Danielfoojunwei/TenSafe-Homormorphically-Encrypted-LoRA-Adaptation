"""
TensorGuard Quota Management.

Provides quota enforcement and tracking with:
- Soft limits (warning) and hard limits (blocking)
- Monthly quota reset logic
- Real-time quota checking
- Over-usage handling
- Quota state persistence

Usage:
    from tensorguard.billing.quota import QuotaManager

    quota_manager = QuotaManager()

    # Check quota before processing
    result = await quota_manager.check_quota(
        tenant_id="tenant-123",
        operation_type=OperationType.TOKENS_INPUT,
        quantity=1000,
    )

    if result.allowed:
        # Process request
        await quota_manager.consume_quota(...)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from .billing_models import (
    OperationType,
    QuotaStatus,
    TenantQuota,
    TierType,
)
from .pricing import get_tier, get_tier_limits

logger = logging.getLogger(__name__)


class QuotaViolation(Exception):
    """
    Raised when a quota limit is exceeded.

    Attributes:
        tenant_id: Tenant that violated the quota
        operation_type: Operation type that was blocked
        limit_type: Type of limit exceeded (soft, hard, rate)
        current_usage: Current usage amount
        limit: The limit that was exceeded
        reset_time: When the quota will reset
    """

    def __init__(
        self,
        tenant_id: str,
        operation_type: OperationType,
        limit_type: str,
        current_usage: float,
        limit: float,
        reset_time: Optional[datetime] = None,
        message: Optional[str] = None,
    ):
        self.tenant_id = tenant_id
        self.operation_type = operation_type
        self.limit_type = limit_type
        self.current_usage = current_usage
        self.limit = limit
        self.reset_time = reset_time

        default_message = (
            f"Quota exceeded for {operation_type.value}: "
            f"{current_usage}/{limit} ({limit_type} limit)"
        )
        super().__init__(message or default_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "error": "quota_exceeded",
            "tenant_id": self.tenant_id,
            "operation_type": self.operation_type.value,
            "limit_type": self.limit_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "reset_time": self.reset_time.isoformat() if self.reset_time else None,
            "message": str(self),
        }


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    status: QuotaStatus
    remaining: float
    limit: float
    reset_time: Optional[datetime] = None
    warning_message: Optional[str] = None
    overage_cost: Optional[Decimal] = None


@dataclass
class QuotaConfig:
    """Configuration for quota management."""

    # Redis configuration
    redis_url: Optional[str] = None
    redis_prefix: str = "tg:quota:"

    # Soft limit percentage (warning threshold)
    soft_limit_percentage: float = 0.8  # Warn at 80%

    # Grace period after hard limit before blocking
    grace_period_minutes: int = 30

    # Allow overage with charges
    allow_overage: bool = True
    overage_multiplier: float = 1.5  # 1.5x price for overage

    # Rate limiting
    rate_limit_window_seconds: int = 60

    # Cache settings
    cache_ttl_seconds: int = 60  # Cache quota state for 1 minute

    # Notification callbacks
    notify_on_warning: bool = True
    notify_on_exceeded: bool = True

    @classmethod
    def from_env(cls) -> "QuotaConfig":
        """Create config from environment variables."""
        return cls(
            redis_url=os.getenv("TG_REDIS_URL"),
            redis_prefix=os.getenv("TG_QUOTA_PREFIX", "tg:quota:"),
            soft_limit_percentage=float(os.getenv("TG_QUOTA_SOFT_LIMIT_PCT", "0.8")),
            grace_period_minutes=int(os.getenv("TG_QUOTA_GRACE_MINUTES", "30")),
            allow_overage=os.getenv("TG_QUOTA_ALLOW_OVERAGE", "true").lower() == "true",
            overage_multiplier=float(os.getenv("TG_QUOTA_OVERAGE_MULTIPLIER", "1.5")),
        )


class QuotaManager:
    """
    Manages tenant quotas with enforcement and tracking.

    Features:
    - Real-time quota checking
    - Soft/hard limit enforcement
    - Monthly quota reset
    - Rate limiting
    - Overage handling
    - Quota state persistence

    The quota manager works with the metering service to track
    actual usage against tier limits.
    """

    def __init__(
        self,
        config: Optional[QuotaConfig] = None,
        warning_callback: Optional[Callable[[str, str], None]] = None,
        exceeded_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize quota manager.

        Args:
            config: Quota configuration
            warning_callback: Callback for quota warnings (tenant_id, message)
            exceeded_callback: Callback for quota exceeded (tenant_id, message)
        """
        self.config = config or QuotaConfig.from_env()
        self._warning_callback = warning_callback
        self._exceeded_callback = exceeded_callback

        # Redis client
        self._redis_client = None
        self._redis_available = False

        # In-memory quota state
        self._quota_cache: Dict[str, TenantQuota] = {}
        self._rate_limit_cache: Dict[str, Dict[str, int]] = {}
        self._lock = asyncio.Lock()

        # Initialize Redis
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not self.config.redis_url:
            logger.info("Quota manager using in-memory storage")
            return

        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._redis_available = True
            logger.info("Quota manager connected to Redis")
        except ImportError:
            logger.warning("redis package not installed")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")

    # =========================================================================
    # QUOTA CHECKING
    # =========================================================================

    async def check_quota(
        self,
        tenant_id: str,
        operation_type: OperationType,
        quantity: float,
        tier_type: Optional[TierType] = None,
    ) -> QuotaCheckResult:
        """
        Check if a tenant is within their quota for an operation.

        Args:
            tenant_id: Tenant identifier
            operation_type: Type of operation
            quantity: Quantity to consume
            tier_type: Optional tier override (fetched if not provided)

        Returns:
            QuotaCheckResult with allowed status and details
        """
        # Get tenant quota state
        quota = await self._get_quota_state(tenant_id, tier_type)

        # Check if quota is currently blocked
        if quota.status == QuotaStatus.BLOCKED:
            if quota.blocked_until and datetime.now(timezone.utc) < quota.blocked_until:
                return QuotaCheckResult(
                    allowed=False,
                    status=QuotaStatus.BLOCKED,
                    remaining=0,
                    limit=quota.tokens_limit,
                    reset_time=quota.blocked_until,
                    warning_message="Access blocked due to quota violations",
                )

        # Get appropriate limits based on operation type
        result = await self._check_operation_quota(quota, operation_type, quantity)

        # Handle warnings and notifications
        if result.status == QuotaStatus.WARNING and self.config.notify_on_warning:
            if not quota.warning_sent:
                await self._send_warning(tenant_id, operation_type, result)
                quota.warning_sent = True
                quota.warning_sent_at = datetime.now(timezone.utc)
                await self._save_quota_state(quota)

        if result.status == QuotaStatus.EXCEEDED and self.config.notify_on_exceeded:
            await self._send_exceeded(tenant_id, operation_type, result)

        return result

    async def _check_operation_quota(
        self,
        quota: TenantQuota,
        operation_type: OperationType,
        quantity: float,
    ) -> QuotaCheckResult:
        """Check quota for specific operation type."""
        tier_limits = get_tier_limits(quota.tier_type)

        # Token operations
        if operation_type in (
            OperationType.TOKENS_INPUT,
            OperationType.TOKENS_OUTPUT,
        ):
            return self._check_token_quota(quota, tier_limits, quantity)

        # Training operations
        if operation_type in (
            OperationType.TRAINING_STEP,
            OperationType.TRAINING_EPOCH,
        ):
            return self._check_training_quota(quota, tier_limits, quantity)

        # GPU operations
        if operation_type in (
            OperationType.GPU_SECONDS,
            OperationType.GPU_HOURS,
        ):
            return self._check_gpu_quota(quota, tier_limits, quantity, operation_type)

        # API rate limiting
        if operation_type in (
            OperationType.API_REQUEST,
            OperationType.API_INFERENCE,
            OperationType.API_EMBEDDING,
        ):
            return await self._check_rate_limit(quota, tier_limits, quantity)

        # Default: allow
        return QuotaCheckResult(
            allowed=True,
            status=QuotaStatus.OK,
            remaining=float("inf"),
            limit=float("inf"),
        )

    def _check_token_quota(
        self,
        quota: TenantQuota,
        tier_limits,
        quantity: float,
    ) -> QuotaCheckResult:
        """Check token quota."""
        if tier_limits.unlimited_tokens:
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.OK,
                remaining=float("inf"),
                limit=float("inf"),
            )

        limit = quota.tokens_limit
        soft_limit = quota.tokens_soft_limit
        current = quota.tokens_used
        projected = current + quantity

        if projected > limit:
            if self.config.allow_overage and quota.overage_allowed:
                return QuotaCheckResult(
                    allowed=True,
                    status=QuotaStatus.EXCEEDED,
                    remaining=0,
                    limit=limit,
                    reset_time=quota.period_end,
                    warning_message=f"Token quota exceeded. Overage charges apply.",
                    overage_cost=self._calculate_overage_cost(
                        quota.tier_type,
                        projected - limit,
                        "tokens",
                    ),
                )
            return QuotaCheckResult(
                allowed=False,
                status=QuotaStatus.EXCEEDED,
                remaining=max(0, limit - current),
                limit=limit,
                reset_time=quota.period_end,
                warning_message=f"Token quota exceeded: {current}/{limit}",
            )

        if projected > soft_limit:
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.WARNING,
                remaining=limit - current,
                limit=limit,
                reset_time=quota.period_end,
                warning_message=f"Approaching token quota: {int(current / limit * 100)}% used",
            )

        return QuotaCheckResult(
            allowed=True,
            status=QuotaStatus.OK,
            remaining=limit - current,
            limit=limit,
            reset_time=quota.period_end,
        )

    def _check_training_quota(
        self,
        quota: TenantQuota,
        tier_limits,
        quantity: float,
    ) -> QuotaCheckResult:
        """Check training quota."""
        if tier_limits.unlimited_training:
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.OK,
                remaining=float("inf"),
                limit=float("inf"),
            )

        limit = quota.training_steps_daily_limit
        current = quota.training_steps_today
        projected = current + quantity

        # Calculate reset time (midnight UTC)
        now = datetime.now(timezone.utc)
        reset_time = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        if projected > limit:
            return QuotaCheckResult(
                allowed=False,
                status=QuotaStatus.EXCEEDED,
                remaining=max(0, limit - current),
                limit=limit,
                reset_time=reset_time,
                warning_message=f"Daily training step quota exceeded: {current}/{limit}",
            )

        soft_limit = int(limit * self.config.soft_limit_percentage)
        if projected > soft_limit:
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.WARNING,
                remaining=limit - current,
                limit=limit,
                reset_time=reset_time,
                warning_message=f"Approaching daily training limit: {int(current / limit * 100)}% used",
            )

        return QuotaCheckResult(
            allowed=True,
            status=QuotaStatus.OK,
            remaining=limit - current,
            limit=limit,
            reset_time=reset_time,
        )

    def _check_gpu_quota(
        self,
        quota: TenantQuota,
        tier_limits,
        quantity: float,
        operation_type: OperationType,
    ) -> QuotaCheckResult:
        """Check GPU quota."""
        if tier_limits.gpu_hours_per_month == 0:
            # Unlimited
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.OK,
                remaining=float("inf"),
                limit=float("inf"),
            )

        # Convert to hours if needed
        if operation_type == OperationType.GPU_SECONDS:
            quantity_hours = quantity / 3600
        else:
            quantity_hours = quantity

        limit = quota.gpu_hours_limit
        current = quota.gpu_hours_used
        projected = current + quantity_hours

        if projected > limit:
            if self.config.allow_overage and quota.overage_allowed:
                return QuotaCheckResult(
                    allowed=True,
                    status=QuotaStatus.EXCEEDED,
                    remaining=0,
                    limit=limit,
                    reset_time=quota.period_end,
                    warning_message="GPU quota exceeded. Overage charges apply.",
                    overage_cost=self._calculate_overage_cost(
                        quota.tier_type,
                        projected - limit,
                        "gpu_hours",
                    ),
                )
            return QuotaCheckResult(
                allowed=False,
                status=QuotaStatus.EXCEEDED,
                remaining=max(0, limit - current),
                limit=limit,
                reset_time=quota.period_end,
                warning_message=f"GPU hours quota exceeded: {current:.2f}/{limit}",
            )

        soft_limit = limit * self.config.soft_limit_percentage
        if projected > soft_limit:
            return QuotaCheckResult(
                allowed=True,
                status=QuotaStatus.WARNING,
                remaining=limit - current,
                limit=limit,
                reset_time=quota.period_end,
                warning_message=f"Approaching GPU quota: {int(current / limit * 100)}% used",
            )

        return QuotaCheckResult(
            allowed=True,
            status=QuotaStatus.OK,
            remaining=limit - current,
            limit=limit,
            reset_time=quota.period_end,
        )

    async def _check_rate_limit(
        self,
        quota: TenantQuota,
        tier_limits,
        quantity: float,
    ) -> QuotaCheckResult:
        """Check API rate limits."""
        tenant_id = quota.tenant_id
        now = datetime.now(timezone.utc)

        # Get current minute and hour buckets
        minute_bucket = now.strftime("%Y%m%d%H%M")
        hour_bucket = now.strftime("%Y%m%d%H")

        # Get current counts
        minute_count = await self._get_rate_count(tenant_id, "minute", minute_bucket)
        hour_count = await self._get_rate_count(tenant_id, "hour", hour_bucket)

        # Check per-minute limit
        if minute_count + quantity > tier_limits.requests_per_minute:
            reset_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            return QuotaCheckResult(
                allowed=False,
                status=QuotaStatus.EXCEEDED,
                remaining=max(0, tier_limits.requests_per_minute - minute_count),
                limit=tier_limits.requests_per_minute,
                reset_time=reset_time,
                warning_message="Rate limit exceeded (per minute)",
            )

        # Check per-hour limit
        if hour_count + quantity > tier_limits.requests_per_hour:
            reset_time = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            return QuotaCheckResult(
                allowed=False,
                status=QuotaStatus.EXCEEDED,
                remaining=max(0, tier_limits.requests_per_hour - hour_count),
                limit=tier_limits.requests_per_hour,
                reset_time=reset_time,
                warning_message="Rate limit exceeded (per hour)",
            )

        return QuotaCheckResult(
            allowed=True,
            status=QuotaStatus.OK,
            remaining=min(
                tier_limits.requests_per_minute - minute_count,
                tier_limits.requests_per_hour - hour_count,
            ),
            limit=tier_limits.requests_per_minute,
        )

    async def _get_rate_count(
        self,
        tenant_id: str,
        window: str,
        bucket: str,
    ) -> int:
        """Get rate limit count for a bucket."""
        if self._redis_available and self._redis_client:
            key = f"{self.config.redis_prefix}rate:{tenant_id}:{window}:{bucket}"
            count = await self._redis_client.get(key)
            return int(count) if count else 0
        else:
            cache_key = f"{tenant_id}:{window}:{bucket}"
            return self._rate_limit_cache.get(cache_key, {}).get("count", 0)

    async def _increment_rate_count(
        self,
        tenant_id: str,
        window: str,
        bucket: str,
        amount: int = 1,
    ) -> None:
        """Increment rate limit count."""
        if self._redis_available and self._redis_client:
            key = f"{self.config.redis_prefix}rate:{tenant_id}:{window}:{bucket}"
            await self._redis_client.incr(key, amount)

            # Set TTL based on window
            ttl = 120 if window == "minute" else 7200
            await self._redis_client.expire(key, ttl)
        else:
            cache_key = f"{tenant_id}:{window}:{bucket}"
            if cache_key not in self._rate_limit_cache:
                self._rate_limit_cache[cache_key] = {"count": 0}
            self._rate_limit_cache[cache_key]["count"] += amount

    # =========================================================================
    # QUOTA CONSUMPTION
    # =========================================================================

    async def consume_quota(
        self,
        tenant_id: str,
        operation_type: OperationType,
        quantity: float,
        tier_type: Optional[TierType] = None,
        check_first: bool = True,
    ) -> QuotaCheckResult:
        """
        Consume quota for an operation.

        Args:
            tenant_id: Tenant identifier
            operation_type: Type of operation
            quantity: Quantity to consume
            tier_type: Optional tier override
            check_first: Whether to check quota before consuming

        Returns:
            QuotaCheckResult with status

        Raises:
            QuotaViolation: If quota exceeded and check_first is True
        """
        if check_first:
            result = await self.check_quota(tenant_id, operation_type, quantity, tier_type)
            if not result.allowed:
                raise QuotaViolation(
                    tenant_id=tenant_id,
                    operation_type=operation_type,
                    limit_type="hard",
                    current_usage=result.limit - result.remaining,
                    limit=result.limit,
                    reset_time=result.reset_time,
                )

        # Get quota state
        quota = await self._get_quota_state(tenant_id, tier_type)

        # Update usage based on operation type
        if operation_type in (OperationType.TOKENS_INPUT, OperationType.TOKENS_OUTPUT):
            quota.tokens_used += int(quantity)
        elif operation_type in (OperationType.TRAINING_STEP, OperationType.TRAINING_EPOCH):
            quota.training_steps_today += int(quantity)
            quota.training_steps_month += int(quantity)
        elif operation_type == OperationType.GPU_SECONDS:
            quota.gpu_hours_used += quantity / 3600
        elif operation_type == OperationType.GPU_HOURS:
            quota.gpu_hours_used += quantity

        # Update rate limits
        if operation_type in (
            OperationType.API_REQUEST,
            OperationType.API_INFERENCE,
            OperationType.API_EMBEDDING,
        ):
            now = datetime.now(timezone.utc)
            await self._increment_rate_count(
                tenant_id, "minute", now.strftime("%Y%m%d%H%M"), int(quantity)
            )
            await self._increment_rate_count(
                tenant_id, "hour", now.strftime("%Y%m%d%H"), int(quantity)
            )

        # Update status
        await self._update_quota_status(quota)

        # Save state
        await self._save_quota_state(quota)

        return QuotaCheckResult(
            allowed=True,
            status=quota.status,
            remaining=self._get_remaining(quota, operation_type),
            limit=self._get_limit(quota, operation_type),
            reset_time=quota.period_end,
        )

    def _get_remaining(
        self,
        quota: TenantQuota,
        operation_type: OperationType,
    ) -> float:
        """Get remaining quota for operation type."""
        if operation_type in (OperationType.TOKENS_INPUT, OperationType.TOKENS_OUTPUT):
            return max(0, quota.tokens_limit - quota.tokens_used)
        elif operation_type in (OperationType.TRAINING_STEP, OperationType.TRAINING_EPOCH):
            return max(0, quota.training_steps_daily_limit - quota.training_steps_today)
        elif operation_type in (OperationType.GPU_SECONDS, OperationType.GPU_HOURS):
            return max(0, quota.gpu_hours_limit - quota.gpu_hours_used)
        return float("inf")

    def _get_limit(
        self,
        quota: TenantQuota,
        operation_type: OperationType,
    ) -> float:
        """Get limit for operation type."""
        if operation_type in (OperationType.TOKENS_INPUT, OperationType.TOKENS_OUTPUT):
            return quota.tokens_limit
        elif operation_type in (OperationType.TRAINING_STEP, OperationType.TRAINING_EPOCH):
            return quota.training_steps_daily_limit
        elif operation_type in (OperationType.GPU_SECONDS, OperationType.GPU_HOURS):
            return quota.gpu_hours_limit
        return float("inf")

    # =========================================================================
    # QUOTA STATE MANAGEMENT
    # =========================================================================

    async def _get_quota_state(
        self,
        tenant_id: str,
        tier_type: Optional[TierType] = None,
    ) -> TenantQuota:
        """Get or create quota state for tenant."""
        # Check cache first
        if tenant_id in self._quota_cache:
            quota = self._quota_cache[tenant_id]
            # Check if reset is needed
            if datetime.now(timezone.utc) >= quota.period_end:
                quota = await self._reset_quota(quota)
            return quota

        # Try Redis
        if self._redis_available and self._redis_client:
            quota = await self._load_quota_redis(tenant_id)
            if quota:
                # Check for reset
                if datetime.now(timezone.utc) >= quota.period_end:
                    quota = await self._reset_quota(quota)
                self._quota_cache[tenant_id] = quota
                return quota

        # Create new quota state
        tier = tier_type or TierType.FREE
        limits = get_tier_limits(tier)

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Calculate period end (first day of next month)
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)

        quota = TenantQuota(
            tenant_id=tenant_id,
            tier_type=tier,
            period_start=period_start,
            period_end=period_end,
            tokens_limit=limits.tokens_per_month,
            tokens_soft_limit=int(limits.tokens_per_month * self.config.soft_limit_percentage),
            training_steps_daily_limit=limits.training_steps_per_day,
            gpu_hours_limit=limits.gpu_hours_per_month,
            overage_allowed=self.config.allow_overage,
        )

        await self._save_quota_state(quota)
        self._quota_cache[tenant_id] = quota
        return quota

    async def _load_quota_redis(self, tenant_id: str) -> Optional[TenantQuota]:
        """Load quota state from Redis."""
        try:
            key = f"{self.config.redis_prefix}state:{tenant_id}"
            data = await self._redis_client.get(key)
            if data:
                return TenantQuota.model_validate_json(data)
        except Exception as e:
            logger.error(f"Failed to load quota from Redis: {e}")
        return None

    async def _save_quota_state(self, quota: TenantQuota) -> None:
        """Save quota state."""
        self._quota_cache[quota.tenant_id] = quota

        if self._redis_available and self._redis_client:
            try:
                key = f"{self.config.redis_prefix}state:{quota.tenant_id}"
                data = quota.model_dump_json()
                await self._redis_client.set(key, data)
            except Exception as e:
                logger.error(f"Failed to save quota to Redis: {e}")

    async def _update_quota_status(self, quota: TenantQuota) -> None:
        """Update quota status based on current usage."""
        # Check token usage
        token_pct = quota.tokens_used / quota.tokens_limit if quota.tokens_limit > 0 else 0

        if token_pct >= 1.0:
            if not self.config.allow_overage or not quota.overage_allowed:
                quota.status = QuotaStatus.EXCEEDED
            else:
                quota.status = QuotaStatus.WARNING
        elif token_pct >= self.config.soft_limit_percentage:
            quota.status = QuotaStatus.WARNING
        else:
            quota.status = QuotaStatus.OK

    async def _reset_quota(self, quota: TenantQuota) -> TenantQuota:
        """Reset quota for new period."""
        logger.info(f"Resetting quota for tenant {quota.tenant_id}")

        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if period_start.month == 12:
            period_end = period_start.replace(year=period_start.year + 1, month=1)
        else:
            period_end = period_start.replace(month=period_start.month + 1)

        # Reset counters
        quota.period_start = period_start
        quota.period_end = period_end
        quota.last_reset = now
        quota.tokens_used = 0
        quota.training_steps_month = 0
        quota.gpu_hours_used = 0
        quota.status = QuotaStatus.OK
        quota.blocked_until = None
        quota.warning_sent = False
        quota.warning_sent_at = None

        await self._save_quota_state(quota)
        return quota

    async def reset_daily_quota(self, tenant_id: str) -> None:
        """Reset daily quotas (training steps)."""
        quota = await self._get_quota_state(tenant_id)
        quota.training_steps_today = 0
        await self._save_quota_state(quota)
        logger.debug(f"Reset daily quota for tenant {tenant_id}")

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def _calculate_overage_cost(
        self,
        tier_type: TierType,
        overage_amount: float,
        resource_type: str,
    ) -> Decimal:
        """Calculate overage cost."""
        from .pricing import get_tier

        tier = get_tier(tier_type)
        multiplier = Decimal(str(self.config.overage_multiplier))

        if resource_type == "tokens":
            base_price = tier.price_per_1k_tokens
            return (Decimal(str(overage_amount)) / Decimal("1000")) * base_price * multiplier
        elif resource_type == "gpu_hours":
            base_price = tier.price_per_gpu_hour
            return Decimal(str(overage_amount)) * base_price * multiplier
        elif resource_type == "training_steps":
            base_price = tier.price_per_training_step
            return Decimal(str(overage_amount)) * base_price * multiplier

        return Decimal("0.00")

    async def _send_warning(
        self,
        tenant_id: str,
        operation_type: OperationType,
        result: QuotaCheckResult,
    ) -> None:
        """Send quota warning notification."""
        if self._warning_callback:
            try:
                self._warning_callback(tenant_id, result.warning_message or "Quota warning")
            except Exception as e:
                logger.error(f"Warning callback failed: {e}")

        logger.warning(f"Quota warning for tenant {tenant_id}: {result.warning_message}")

    async def _send_exceeded(
        self,
        tenant_id: str,
        operation_type: OperationType,
        result: QuotaCheckResult,
    ) -> None:
        """Send quota exceeded notification."""
        if self._exceeded_callback:
            try:
                self._exceeded_callback(tenant_id, result.warning_message or "Quota exceeded")
            except Exception as e:
                logger.error(f"Exceeded callback failed: {e}")

        logger.warning(f"Quota exceeded for tenant {tenant_id}: {result.warning_message}")

    async def get_quota_summary(self, tenant_id: str) -> Dict[str, Any]:
        """
        Get quota summary for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Quota summary dictionary
        """
        quota = await self._get_quota_state(tenant_id)

        return {
            "tenant_id": tenant_id,
            "tier": quota.tier_type.value,
            "status": quota.status.value,
            "period": {
                "start": quota.period_start.isoformat(),
                "end": quota.period_end.isoformat(),
            },
            "tokens": {
                "used": quota.tokens_used,
                "limit": quota.tokens_limit,
                "remaining": max(0, quota.tokens_limit - quota.tokens_used),
                "percentage": round(quota.tokens_used / quota.tokens_limit * 100, 2) if quota.tokens_limit > 0 else 0,
            },
            "training": {
                "today": quota.training_steps_today,
                "daily_limit": quota.training_steps_daily_limit,
                "month": quota.training_steps_month,
            },
            "gpu": {
                "used_hours": round(quota.gpu_hours_used, 2),
                "limit_hours": quota.gpu_hours_limit,
            },
            "overage_allowed": quota.overage_allowed,
        }

    async def update_tier(
        self,
        tenant_id: str,
        new_tier: TierType,
    ) -> TenantQuota:
        """
        Update tenant's pricing tier.

        Args:
            tenant_id: Tenant identifier
            new_tier: New tier type

        Returns:
            Updated TenantQuota
        """
        quota = await self._get_quota_state(tenant_id)
        limits = get_tier_limits(new_tier)

        quota.tier_type = new_tier
        quota.tokens_limit = limits.tokens_per_month
        quota.tokens_soft_limit = int(limits.tokens_per_month * self.config.soft_limit_percentage)
        quota.training_steps_daily_limit = limits.training_steps_per_day
        quota.gpu_hours_limit = limits.gpu_hours_per_month

        await self._update_quota_status(quota)
        await self._save_quota_state(quota)

        logger.info(f"Updated tier for tenant {tenant_id} to {new_tier.value}")
        return quota


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "QuotaManager",
    "QuotaConfig",
    "QuotaCheckResult",
    "QuotaViolation",
]
