"""
TensorGuard Usage Metering Service.

Production-grade usage metering with:
- Real-time event recording (sync and async)
- Redis backend for high-performance metering
- In-memory fallback for single-instance deployments
- Batch persistence for durability
- Usage aggregation (hourly, daily, monthly)
- Automatic cleanup of old data

Architecture:
- Events are first written to Redis for real-time access
- Batch job persists events to durable storage periodically
- Aggregation jobs roll up events into summaries

Usage:
    from tensorguard.billing.metering import MeteringService

    metering = MeteringService()

    # Record usage
    await metering.record_event(
        tenant_id="tenant-123",
        operation_type=OperationType.TOKENS_INPUT,
        quantity=1000,
    )

    # Get aggregated usage
    summary = await metering.get_usage_summary(
        tenant_id="tenant-123",
        period_start=datetime.now() - timedelta(days=30),
        period_end=datetime.now(),
    )
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .billing_models import (
    OperationType,
    UsageEvent,
    UsageSummary,
)

logger = logging.getLogger(__name__)


@dataclass
class MeteringConfig:
    """Configuration for the metering service."""

    # Redis configuration
    redis_url: Optional[str] = None
    redis_prefix: str = "tg:metering:"

    # Batch settings
    batch_size: int = 100  # Events per batch
    batch_interval_seconds: int = 60  # Flush interval
    max_queue_size: int = 10000  # Max events before force flush

    # Retention settings
    event_ttl_hours: int = 168  # 7 days for raw events in Redis
    summary_ttl_days: int = 365  # 1 year for summaries

    # Aggregation settings
    aggregation_interval_minutes: int = 5  # Real-time aggregation window

    # Performance settings
    enable_async_recording: bool = True
    enable_compression: bool = True

    # Fallback
    enable_memory_fallback: bool = True

    @classmethod
    def from_env(cls) -> "MeteringConfig":
        """Create config from environment variables."""
        return cls(
            redis_url=os.getenv("TG_REDIS_URL"),
            redis_prefix=os.getenv("TG_METERING_PREFIX", "tg:metering:"),
            batch_size=int(os.getenv("TG_METERING_BATCH_SIZE", "100")),
            batch_interval_seconds=int(os.getenv("TG_METERING_BATCH_INTERVAL", "60")),
            max_queue_size=int(os.getenv("TG_METERING_MAX_QUEUE", "10000")),
            event_ttl_hours=int(os.getenv("TG_METERING_EVENT_TTL_HOURS", "168")),
            enable_async_recording=os.getenv("TG_METERING_ASYNC", "true").lower() == "true",
        )


class MeteringService:
    """
    Production-grade usage metering service.

    Features:
    - High-performance event recording with Redis backend
    - In-memory fallback for single-instance deployments
    - Batch processing for durability
    - Real-time and historical aggregation
    - Thread-safe async operations

    The service uses a two-tier storage model:
    1. Redis: For real-time metering and quick lookups
    2. Batch persistence: For durable storage and historical analysis
    """

    def __init__(
        self,
        config: Optional[MeteringConfig] = None,
        persistence_callback: Optional[Callable[[List[UsageEvent]], None]] = None,
    ):
        """
        Initialize metering service.

        Args:
            config: Metering configuration
            persistence_callback: Optional callback for batch persistence
        """
        self.config = config or MeteringConfig.from_env()
        self._persistence_callback = persistence_callback

        # Redis client (lazy initialized)
        self._redis_client = None
        self._redis_available = False

        # In-memory fallback storage
        self._event_queue: List[UsageEvent] = []
        self._memory_store: Dict[str, List[UsageEvent]] = {}
        self._aggregation_cache: Dict[str, UsageSummary] = {}

        # Locks
        self._queue_lock = asyncio.Lock()
        self._store_lock = asyncio.Lock()

        # Background tasks
        self._batch_task: Optional[asyncio.Task] = None
        self._running = False

        # Initialize Redis if available
        self._init_redis()

    def _init_redis(self) -> None:
        """Initialize Redis connection."""
        if not self.config.redis_url:
            if self.config.enable_memory_fallback:
                logger.info("Metering service using in-memory storage (no Redis URL)")
            else:
                logger.warning("No Redis URL configured and memory fallback disabled")
            return

        try:
            import redis.asyncio as redis

            self._redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._redis_available = True
            logger.info("Metering service connected to Redis")
        except ImportError:
            logger.warning("redis package not installed, using in-memory storage")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            if not self.config.enable_memory_fallback:
                raise

    async def start(self) -> None:
        """Start background tasks for batch processing."""
        if self._running:
            return

        self._running = True
        self._batch_task = asyncio.create_task(self._batch_processor())
        logger.info("Metering service started")

    async def stop(self) -> None:
        """Stop background tasks and flush pending events."""
        self._running = False

        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass

        # Flush remaining events
        await self._flush_batch()

        if self._redis_client:
            await self._redis_client.close()

        logger.info("Metering service stopped")

    # =========================================================================
    # EVENT RECORDING
    # =========================================================================

    async def record_event(
        self,
        tenant_id: str,
        operation_type: OperationType,
        quantity: float,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        unit: str = "units",
        unit_price: Optional[Decimal] = None,
    ) -> UsageEvent:
        """
        Record a usage event asynchronously.

        Args:
            tenant_id: Tenant identifier
            operation_type: Type of operation
            quantity: Quantity consumed
            user_id: Optional user identifier
            resource_id: Optional resource identifier
            endpoint: Optional API endpoint
            model_id: Optional model identifier
            request_id: Optional request correlation ID
            metadata: Optional additional metadata
            unit: Unit of measurement
            unit_price: Optional unit price at time of event

        Returns:
            Created UsageEvent
        """
        event = UsageEvent(
            tenant_id=tenant_id,
            user_id=user_id,
            operation_type=operation_type,
            quantity=quantity,
            unit=unit,
            resource_id=resource_id,
            endpoint=endpoint,
            model_id=model_id,
            request_id=request_id,
            metadata=metadata or {},
            unit_price=unit_price,
        )

        # Calculate total cost if unit price provided
        if unit_price:
            event.total_cost = Decimal(str(quantity)) * unit_price

        if self.config.enable_async_recording:
            asyncio.create_task(self._store_event(event))
        else:
            await self._store_event(event)

        return event

    def record_event_sync(
        self,
        tenant_id: str,
        operation_type: OperationType,
        quantity: float,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        unit: str = "units",
        unit_price: Optional[Decimal] = None,
    ) -> UsageEvent:
        """
        Record a usage event synchronously.

        This is a blocking version for use in synchronous contexts.
        Events are queued for async processing.

        Args:
            Same as record_event

        Returns:
            Created UsageEvent
        """
        event = UsageEvent(
            tenant_id=tenant_id,
            user_id=user_id,
            operation_type=operation_type,
            quantity=quantity,
            unit=unit,
            resource_id=resource_id,
            endpoint=endpoint,
            model_id=model_id,
            request_id=request_id,
            metadata=metadata or {},
            unit_price=unit_price,
        )

        if unit_price:
            event.total_cost = Decimal(str(quantity)) * unit_price

        # Queue for async processing
        self._event_queue.append(event)

        # Force flush if queue is too large
        if len(self._event_queue) >= self.config.max_queue_size:
            logger.warning("Event queue at max size, forcing sync flush")
            asyncio.run(self._flush_batch())

        return event

    async def _store_event(self, event: UsageEvent) -> None:
        """Store event in backend."""
        try:
            if self._redis_available and self._redis_client:
                await self._store_event_redis(event)
            else:
                await self._store_event_memory(event)

            # Add to batch queue for persistence
            async with self._queue_lock:
                self._event_queue.append(event)

        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            # Fallback to memory
            await self._store_event_memory(event)

    async def _store_event_redis(self, event: UsageEvent) -> None:
        """Store event in Redis."""
        # Event key
        event_key = f"{self.config.redis_prefix}events:{event.tenant_id}:{event.event_id}"

        # Serialize event
        event_data = event.model_dump_json()

        # Store with TTL
        await self._redis_client.setex(
            event_key,
            self.config.event_ttl_hours * 3600,
            event_data,
        )

        # Add to sorted set for time-range queries
        score = event.timestamp.timestamp()
        await self._redis_client.zadd(
            f"{self.config.redis_prefix}events:timeline:{event.tenant_id}",
            {str(event.event_id): score},
        )

        # Update real-time counters
        await self._update_realtime_counters(event)

    async def _update_realtime_counters(self, event: UsageEvent) -> None:
        """Update real-time counters in Redis."""
        timestamp = event.timestamp
        minute_bucket = timestamp.strftime("%Y%m%d%H%M")
        hour_bucket = timestamp.strftime("%Y%m%d%H")
        day_bucket = timestamp.strftime("%Y%m%d")
        month_bucket = timestamp.strftime("%Y%m")

        # Counter keys
        base_key = f"{self.config.redis_prefix}counters:{event.tenant_id}"

        # Increment counters with appropriate TTLs
        pipe = self._redis_client.pipeline()

        # Minute counter (5 min TTL)
        minute_key = f"{base_key}:minute:{minute_bucket}:{event.operation_type.value}"
        pipe.incrbyfloat(minute_key, event.quantity)
        pipe.expire(minute_key, 300)

        # Hour counter (2 hour TTL)
        hour_key = f"{base_key}:hour:{hour_bucket}:{event.operation_type.value}"
        pipe.incrbyfloat(hour_key, event.quantity)
        pipe.expire(hour_key, 7200)

        # Day counter (2 day TTL)
        day_key = f"{base_key}:day:{day_bucket}:{event.operation_type.value}"
        pipe.incrbyfloat(day_key, event.quantity)
        pipe.expire(day_key, 172800)

        # Month counter (35 day TTL)
        month_key = f"{base_key}:month:{month_bucket}:{event.operation_type.value}"
        pipe.incrbyfloat(month_key, event.quantity)
        pipe.expire(month_key, 3024000)

        await pipe.execute()

    async def _store_event_memory(self, event: UsageEvent) -> None:
        """Store event in memory (fallback)."""
        async with self._store_lock:
            if event.tenant_id not in self._memory_store:
                self._memory_store[event.tenant_id] = []
            self._memory_store[event.tenant_id].append(event)

            # Limit memory usage - keep last 10000 events per tenant
            if len(self._memory_store[event.tenant_id]) > 10000:
                self._memory_store[event.tenant_id] = self._memory_store[event.tenant_id][-10000:]

    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================

    async def _batch_processor(self) -> None:
        """Background task for batch processing."""
        while self._running:
            try:
                await asyncio.sleep(self.config.batch_interval_seconds)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processor error: {e}")

    async def _flush_batch(self) -> None:
        """Flush queued events to persistence layer."""
        async with self._queue_lock:
            if not self._event_queue:
                return

            batch = self._event_queue[: self.config.batch_size]
            self._event_queue = self._event_queue[self.config.batch_size :]

        if not batch:
            return

        try:
            if self._persistence_callback:
                self._persistence_callback(batch)
            logger.debug(f"Flushed {len(batch)} events to persistence")
        except Exception as e:
            logger.error(f"Failed to persist batch: {e}")
            # Re-queue events on failure
            async with self._queue_lock:
                self._event_queue = batch + self._event_queue

    # =========================================================================
    # USAGE QUERIES
    # =========================================================================

    async def get_usage_summary(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        aggregation_level: str = "daily",
    ) -> UsageSummary:
        """
        Get aggregated usage summary for a period.

        Args:
            tenant_id: Tenant identifier
            period_start: Period start datetime
            period_end: Period end datetime
            aggregation_level: Aggregation level (hourly, daily, monthly)

        Returns:
            UsageSummary with aggregated data
        """
        # Ensure timezone awareness
        if period_start.tzinfo is None:
            period_start = period_start.replace(tzinfo=timezone.utc)
        if period_end.tzinfo is None:
            period_end = period_end.replace(tzinfo=timezone.utc)

        if self._redis_available and self._redis_client:
            return await self._get_summary_redis(tenant_id, period_start, period_end, aggregation_level)
        else:
            return await self._get_summary_memory(tenant_id, period_start, period_end, aggregation_level)

    async def _get_summary_redis(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        aggregation_level: str,
    ) -> UsageSummary:
        """Get usage summary from Redis."""
        summary = UsageSummary(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            aggregation_level=aggregation_level,
        )

        # Get events from timeline
        timeline_key = f"{self.config.redis_prefix}events:timeline:{tenant_id}"
        start_score = period_start.timestamp()
        end_score = period_end.timestamp()

        event_ids = await self._redis_client.zrangebyscore(timeline_key, start_score, end_score)

        usage_by_type: Dict[str, float] = {}

        for event_id in event_ids:
            event_key = f"{self.config.redis_prefix}events:{tenant_id}:{event_id}"
            event_data = await self._redis_client.get(event_key)

            if event_data:
                event = UsageEvent.model_validate_json(event_data)
                self._aggregate_event(summary, event, usage_by_type)

        summary.usage_by_type = usage_by_type
        summary.total_events = len(event_ids)

        return summary

    async def _get_summary_memory(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        aggregation_level: str,
    ) -> UsageSummary:
        """Get usage summary from memory storage."""
        summary = UsageSummary(
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            aggregation_level=aggregation_level,
        )

        async with self._store_lock:
            events = self._memory_store.get(tenant_id, [])

        usage_by_type: Dict[str, float] = {}
        event_count = 0

        for event in events:
            if period_start <= event.timestamp <= period_end:
                self._aggregate_event(summary, event, usage_by_type)
                event_count += 1

        summary.usage_by_type = usage_by_type
        summary.total_events = event_count

        return summary

    def _aggregate_event(
        self,
        summary: UsageSummary,
        event: UsageEvent,
        usage_by_type: Dict[str, float],
    ) -> None:
        """Aggregate a single event into summary."""
        op_type = event.operation_type

        # Update type breakdown
        type_key = op_type.value
        usage_by_type[type_key] = usage_by_type.get(type_key, 0.0) + event.quantity

        # Update specific counters based on operation type
        if op_type == OperationType.TOKENS_INPUT:
            summary.total_tokens_input += int(event.quantity)
        elif op_type == OperationType.TOKENS_OUTPUT:
            summary.total_tokens_output += int(event.quantity)
        elif op_type == OperationType.API_REQUEST:
            summary.total_api_requests += int(event.quantity)
        elif op_type == OperationType.API_INFERENCE:
            summary.total_inference_requests += int(event.quantity)
        elif op_type == OperationType.API_EMBEDDING:
            summary.total_embedding_requests += int(event.quantity)
        elif op_type == OperationType.TRAINING_STEP:
            summary.total_training_steps += int(event.quantity)
        elif op_type == OperationType.FINE_TUNING_JOB:
            summary.total_training_jobs += int(event.quantity)
        elif op_type == OperationType.GPU_SECONDS:
            summary.total_gpu_seconds += event.quantity
            summary.total_gpu_hours = summary.total_gpu_seconds / 3600
        elif op_type == OperationType.GPU_HOURS:
            summary.total_gpu_hours += event.quantity
            summary.total_gpu_seconds = summary.total_gpu_hours * 3600
        elif op_type == OperationType.CPU_HOURS:
            summary.total_cpu_hours += event.quantity
        elif op_type in (
            OperationType.HE_ENCRYPTION,
            OperationType.HE_DECRYPTION,
            OperationType.HE_COMPUTATION,
        ):
            summary.total_he_operations += int(event.quantity)

        # Aggregate cost
        if event.total_cost:
            summary.total_cost += event.total_cost
            cost_key = op_type.value
            summary.cost_by_type[cost_key] = summary.cost_by_type.get(cost_key, Decimal("0.00")) + event.total_cost

    # =========================================================================
    # REAL-TIME QUERIES
    # =========================================================================

    async def get_realtime_usage(
        self,
        tenant_id: str,
        operation_type: Optional[OperationType] = None,
        window_minutes: int = 5,
    ) -> Dict[str, float]:
        """
        Get real-time usage for the last N minutes.

        Args:
            tenant_id: Tenant identifier
            operation_type: Optional filter by operation type
            window_minutes: Time window in minutes

        Returns:
            Dictionary of operation types to quantities
        """
        if not self._redis_available or not self._redis_client:
            return await self._get_realtime_memory(tenant_id, operation_type, window_minutes)

        now = datetime.now(timezone.utc)
        base_key = f"{self.config.redis_prefix}counters:{tenant_id}"
        usage: Dict[str, float] = {}

        # Get minute buckets for the window
        for i in range(window_minutes):
            bucket_time = now - timedelta(minutes=i)
            minute_bucket = bucket_time.strftime("%Y%m%d%H%M")

            if operation_type:
                key = f"{base_key}:minute:{minute_bucket}:{operation_type.value}"
                value = await self._redis_client.get(key)
                if value:
                    usage[operation_type.value] = usage.get(operation_type.value, 0.0) + float(value)
            else:
                # Get all operation types
                pattern = f"{base_key}:minute:{minute_bucket}:*"
                async for key in self._redis_client.scan_iter(match=pattern):
                    op_type = key.split(":")[-1]
                    value = await self._redis_client.get(key)
                    if value:
                        usage[op_type] = usage.get(op_type, 0.0) + float(value)

        return usage

    async def _get_realtime_memory(
        self,
        tenant_id: str,
        operation_type: Optional[OperationType],
        window_minutes: int,
    ) -> Dict[str, float]:
        """Get real-time usage from memory."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_minutes)
        usage: Dict[str, float] = {}

        async with self._store_lock:
            events = self._memory_store.get(tenant_id, [])

        for event in events:
            if event.timestamp >= cutoff:
                if operation_type is None or event.operation_type == operation_type:
                    key = event.operation_type.value
                    usage[key] = usage.get(key, 0.0) + event.quantity

        return usage

    async def get_hourly_usage(
        self,
        tenant_id: str,
        operation_type: OperationType,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get hourly usage breakdown.

        Args:
            tenant_id: Tenant identifier
            operation_type: Operation type to query
            hours: Number of hours to retrieve

        Returns:
            List of hourly usage records
        """
        if not self._redis_available or not self._redis_client:
            return []

        now = datetime.now(timezone.utc)
        base_key = f"{self.config.redis_prefix}counters:{tenant_id}"
        hourly_data = []

        for i in range(hours):
            bucket_time = now - timedelta(hours=i)
            hour_bucket = bucket_time.strftime("%Y%m%d%H")
            key = f"{base_key}:hour:{hour_bucket}:{operation_type.value}"

            value = await self._redis_client.get(key)
            hourly_data.append(
                {
                    "hour": bucket_time.replace(minute=0, second=0, microsecond=0).isoformat(),
                    "quantity": float(value) if value else 0.0,
                    "operation_type": operation_type.value,
                }
            )

        return list(reversed(hourly_data))

    async def get_daily_usage(
        self,
        tenant_id: str,
        operation_type: OperationType,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get daily usage breakdown.

        Args:
            tenant_id: Tenant identifier
            operation_type: Operation type to query
            days: Number of days to retrieve

        Returns:
            List of daily usage records
        """
        if not self._redis_available or not self._redis_client:
            return []

        now = datetime.now(timezone.utc)
        base_key = f"{self.config.redis_prefix}counters:{tenant_id}"
        daily_data = []

        for i in range(days):
            bucket_time = now - timedelta(days=i)
            day_bucket = bucket_time.strftime("%Y%m%d")
            key = f"{base_key}:day:{day_bucket}:{operation_type.value}"

            value = await self._redis_client.get(key)
            daily_data.append(
                {
                    "date": bucket_time.strftime("%Y-%m-%d"),
                    "quantity": float(value) if value else 0.0,
                    "operation_type": operation_type.value,
                }
            )

        return list(reversed(daily_data))

    async def get_monthly_usage(
        self,
        tenant_id: str,
        operation_type: OperationType,
        months: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Get monthly usage breakdown.

        Args:
            tenant_id: Tenant identifier
            operation_type: Operation type to query
            months: Number of months to retrieve

        Returns:
            List of monthly usage records
        """
        if not self._redis_available or not self._redis_client:
            return []

        now = datetime.now(timezone.utc)
        base_key = f"{self.config.redis_prefix}counters:{tenant_id}"
        monthly_data = []

        for i in range(months):
            # Calculate month offset
            month = now.month - i
            year = now.year
            while month <= 0:
                month += 12
                year -= 1

            month_bucket = f"{year}{month:02d}"
            key = f"{base_key}:month:{month_bucket}:{operation_type.value}"

            value = await self._redis_client.get(key)
            monthly_data.append(
                {
                    "month": f"{year}-{month:02d}",
                    "quantity": float(value) if value else 0.0,
                    "operation_type": operation_type.value,
                }
            )

        return list(reversed(monthly_data))

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def get_event_count(
        self,
        tenant_id: str,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
    ) -> int:
        """Get count of events for a tenant in a period."""
        if self._redis_available and self._redis_client:
            timeline_key = f"{self.config.redis_prefix}events:timeline:{tenant_id}"

            if period_start and period_end:
                start_score = period_start.timestamp()
                end_score = period_end.timestamp()
                return await self._redis_client.zcount(timeline_key, start_score, end_score)
            else:
                return await self._redis_client.zcard(timeline_key)
        else:
            async with self._store_lock:
                events = self._memory_store.get(tenant_id, [])

            if period_start and period_end:
                return len([e for e in events if period_start <= e.timestamp <= period_end])
            return len(events)

    async def delete_tenant_data(self, tenant_id: str) -> int:
        """
        Delete all metering data for a tenant.

        For GDPR/compliance data deletion requests.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of records deleted
        """
        deleted = 0

        if self._redis_available and self._redis_client:
            # Delete timeline
            timeline_key = f"{self.config.redis_prefix}events:timeline:{tenant_id}"
            event_ids = await self._redis_client.zrange(timeline_key, 0, -1)

            # Delete events
            for event_id in event_ids:
                event_key = f"{self.config.redis_prefix}events:{tenant_id}:{event_id}"
                await self._redis_client.delete(event_key)
                deleted += 1

            # Delete timeline
            await self._redis_client.delete(timeline_key)

            # Delete counters
            pattern = f"{self.config.redis_prefix}counters:{tenant_id}:*"
            async for key in self._redis_client.scan_iter(match=pattern):
                await self._redis_client.delete(key)
                deleted += 1

        # Clear memory store
        async with self._store_lock:
            if tenant_id in self._memory_store:
                deleted += len(self._memory_store[tenant_id])
                del self._memory_store[tenant_id]

        logger.info(f"Deleted {deleted} metering records for tenant {tenant_id}")
        return deleted

    async def health_check(self) -> Dict[str, Any]:
        """
        Check metering service health.

        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "redis_available": self._redis_available,
            "queue_size": len(self._event_queue),
            "memory_tenants": len(self._memory_store),
            "running": self._running,
        }

        if self._redis_available and self._redis_client:
            try:
                await self._redis_client.ping()
                health["redis_connected"] = True
            except Exception as e:
                health["redis_connected"] = False
                health["redis_error"] = str(e)
                health["status"] = "degraded"

        if health["queue_size"] > self.config.max_queue_size * 0.8:
            health["status"] = "warning"
            health["warning"] = "Event queue approaching limit"

        return health


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "MeteringService",
    "MeteringConfig",
]
