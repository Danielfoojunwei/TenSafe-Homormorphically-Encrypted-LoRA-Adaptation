"""
Resource Pool Management.

Provides generic resource pooling:
- Connection pooling for databases, HTTP clients, etc.
- Bounded resource allocation
- Health checking and eviction
- Automatic resource lifecycle management

Usage:
    pool = ResourcePool(
        factory=create_connection,
        max_size=10,
        min_size=2,
    )

    async with pool.acquire() as conn:
        result = await conn.execute(query)
"""

import asyncio
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Generic, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PoolExhaustedError(Exception):
    """Raised when resource pool is exhausted."""

    def __init__(
        self,
        message: str,
        pool_name: str,
        max_size: int,
        active: int,
    ):
        self.pool_name = pool_name
        self.max_size = max_size
        self.active = active
        super().__init__(message)


@dataclass
class PooledResource(Generic[T]):
    """Wrapper for a pooled resource with metadata."""

    resource: T
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    healthy: bool = True

    def mark_used(self) -> None:
        """Mark resource as used."""
        self.last_used = datetime.now(timezone.utc)
        self.use_count += 1


@dataclass
class PoolStats:
    """Statistics for resource pool."""

    total_created: int = 0
    total_destroyed: int = 0
    total_acquired: int = 0
    total_released: int = 0
    total_timeouts: int = 0
    total_evicted: int = 0
    current_size: int = 0
    current_available: int = 0
    current_in_use: int = 0

    def to_dict(self) -> dict:
        """Export as dictionary."""
        return {
            "total_created": self.total_created,
            "total_destroyed": self.total_destroyed,
            "total_acquired": self.total_acquired,
            "total_released": self.total_released,
            "total_timeouts": self.total_timeouts,
            "total_evicted": self.total_evicted,
            "current_size": self.current_size,
            "current_available": self.current_available,
            "current_in_use": self.current_in_use,
        }


class ResourcePool(Generic[T]):
    """
    Generic resource pool with lifecycle management.

    Features:
    - Bounded pool size with min/max
    - Async and sync resource acquisition
    - Health checking before use
    - Automatic eviction of stale resources
    - Resource recycling
    """

    def __init__(
        self,
        factory: Union[Callable[[], T], Callable[[], Coroutine[Any, Any, T]]],
        max_size: int = 10,
        min_size: int = 0,
        acquire_timeout: float = 30.0,
        max_idle_time: float = 300.0,
        max_lifetime: float = 3600.0,
        health_check: Optional[Callable[[T], bool]] = None,
        cleanup: Optional[Callable[[T], None]] = None,
        name: str = "pool",
    ):
        """
        Initialize resource pool.

        Args:
            factory: Function to create new resources
            max_size: Maximum pool size
            min_size: Minimum pool size
            acquire_timeout: Timeout for acquiring a resource
            max_idle_time: Max time a resource can sit idle
            max_lifetime: Max lifetime of a resource
            health_check: Function to verify resource health
            cleanup: Function to cleanup a resource
            name: Pool name for logging
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.acquire_timeout = acquire_timeout
        self.max_idle_time = max_idle_time
        self.max_lifetime = max_lifetime
        self.health_check = health_check
        self.cleanup = cleanup
        self.name = name

        self._available: deque[PooledResource[T]] = deque()
        self._in_use: set[PooledResource[T]] = set()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._async_semaphore: Optional[asyncio.Semaphore] = None
        self._stats = PoolStats()
        self._closed = False
        self._maintenance_task: Optional[asyncio.Task] = None

    async def _ensure_async_semaphore(self) -> asyncio.Semaphore:
        """Ensure async semaphore is created."""
        if self._async_semaphore is None:
            self._async_semaphore = asyncio.Semaphore(self.max_size)
        return self._async_semaphore

    def _create_resource(self) -> PooledResource[T]:
        """Create a new pooled resource."""
        now = datetime.now(timezone.utc)

        if asyncio.iscoroutinefunction(self.factory):
            raise RuntimeError("Async factory requires async context")

        resource = self.factory()
        pooled = PooledResource(
            resource=resource,
            created_at=now,
            last_used=now,
        )

        self._stats.total_created += 1
        self._stats.current_size += 1

        logger.debug(f"Pool '{self.name}': created new resource")
        return pooled

    async def _create_resource_async(self) -> PooledResource[T]:
        """Create a new pooled resource asynchronously."""
        now = datetime.now(timezone.utc)

        if asyncio.iscoroutinefunction(self.factory):
            resource = await self.factory()
        else:
            loop = asyncio.get_event_loop()
            resource = await loop.run_in_executor(None, self.factory)

        pooled = PooledResource(
            resource=resource,
            created_at=now,
            last_used=now,
        )

        with self._lock:
            self._stats.total_created += 1
            self._stats.current_size += 1

        logger.debug(f"Pool '{self.name}': created new resource")
        return pooled

    def _destroy_resource(self, pooled: PooledResource[T]) -> None:
        """Destroy a pooled resource."""
        try:
            if self.cleanup:
                self.cleanup(pooled.resource)
        except Exception as e:
            logger.warning(f"Pool '{self.name}': cleanup error: {e}")

        self._stats.total_destroyed += 1
        self._stats.current_size -= 1

        logger.debug(f"Pool '{self.name}': destroyed resource")

    def _is_healthy(self, pooled: PooledResource[T]) -> bool:
        """Check if a resource is healthy."""
        if not pooled.healthy:
            return False

        # Check lifetime
        now = datetime.now(timezone.utc)
        age = (now - pooled.created_at).total_seconds()
        if age > self.max_lifetime:
            return False

        # Check idle time
        idle = (now - pooled.last_used).total_seconds()
        if idle > self.max_idle_time:
            return False

        # Run health check
        if self.health_check:
            try:
                return self.health_check(pooled.resource)
            except Exception:
                return False

        return True

    def _get_available(self) -> Optional[PooledResource[T]]:
        """Get an available healthy resource (must hold lock)."""
        while self._available:
            pooled = self._available.popleft()

            if self._is_healthy(pooled):
                return pooled

            # Resource is unhealthy, destroy it
            self._destroy_resource(pooled)
            self._stats.total_evicted += 1

        return None

    @contextmanager
    def acquire_sync(self, timeout: Optional[float] = None):
        """
        Acquire a resource synchronously.

        Args:
            timeout: Timeout for acquisition

        Yields:
            Resource

        Raises:
            PoolExhaustedError: If pool is exhausted
        """
        timeout = timeout or self.acquire_timeout
        deadline = time.time() + timeout

        with self._condition:
            while True:
                if self._closed:
                    raise RuntimeError(f"Pool '{self.name}' is closed")

                # Try to get available resource
                pooled = self._get_available()
                if pooled:
                    break

                # Try to create new resource
                current_size = len(self._available) + len(self._in_use)
                if current_size < self.max_size:
                    pooled = self._create_resource()
                    break

                # Wait for resource to become available
                remaining = deadline - time.time()
                if remaining <= 0:
                    self._stats.total_timeouts += 1
                    raise PoolExhaustedError(
                        f"Pool '{self.name}' exhausted after {timeout}s",
                        pool_name=self.name,
                        max_size=self.max_size,
                        active=len(self._in_use),
                    )

                self._condition.wait(timeout=remaining)

            # Mark as in use
            self._in_use.add(pooled)
            pooled.mark_used()
            self._stats.total_acquired += 1
            self._stats.current_available = len(self._available)
            self._stats.current_in_use = len(self._in_use)

        try:
            yield pooled.resource
        except Exception:
            # Mark as unhealthy on exception
            pooled.healthy = False
            raise
        finally:
            with self._condition:
                self._in_use.discard(pooled)

                if pooled.healthy and self._is_healthy(pooled):
                    self._available.append(pooled)
                else:
                    self._destroy_resource(pooled)

                self._stats.total_released += 1
                self._stats.current_available = len(self._available)
                self._stats.current_in_use = len(self._in_use)
                self._condition.notify()

    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a resource asynchronously.

        Args:
            timeout: Timeout for acquisition

        Yields:
            Resource

        Raises:
            PoolExhaustedError: If pool is exhausted
        """
        timeout = timeout or self.acquire_timeout
        semaphore = await self._ensure_async_semaphore()

        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            with self._lock:
                self._stats.total_timeouts += 1
            raise PoolExhaustedError(
                f"Pool '{self.name}' exhausted after {timeout}s",
                pool_name=self.name,
                max_size=self.max_size,
                active=len(self._in_use),
            )

        pooled: Optional[PooledResource[T]] = None

        try:
            with self._lock:
                if self._closed:
                    raise RuntimeError(f"Pool '{self.name}' is closed")

                pooled = self._get_available()

                if pooled is None:
                    # Create new resource
                    pass  # Will create outside lock

            if pooled is None:
                pooled = await self._create_resource_async()

            with self._lock:
                self._in_use.add(pooled)
                pooled.mark_used()
                self._stats.total_acquired += 1
                self._stats.current_available = len(self._available)
                self._stats.current_in_use = len(self._in_use)

            try:
                yield pooled.resource
            except Exception:
                pooled.healthy = False
                raise

        finally:
            if pooled:
                with self._lock:
                    self._in_use.discard(pooled)

                    if pooled.healthy and self._is_healthy(pooled):
                        self._available.append(pooled)
                    else:
                        self._destroy_resource(pooled)

                    self._stats.total_released += 1
                    self._stats.current_available = len(self._available)
                    self._stats.current_in_use = len(self._in_use)

            semaphore.release()

    async def start_maintenance(self, interval: float = 60.0) -> None:
        """Start background maintenance task."""
        if self._maintenance_task is not None:
            return

        async def maintenance_loop():
            while not self._closed:
                try:
                    await asyncio.sleep(interval)
                    await self._run_maintenance()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Pool '{self.name}' maintenance error: {e}")

        self._maintenance_task = asyncio.create_task(maintenance_loop())

    async def stop_maintenance(self) -> None:
        """Stop background maintenance task."""
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None

    async def _run_maintenance(self) -> None:
        """Run pool maintenance."""
        with self._lock:
            # Evict unhealthy/stale resources
            healthy = deque()
            while self._available:
                pooled = self._available.popleft()
                if self._is_healthy(pooled):
                    healthy.append(pooled)
                else:
                    self._destroy_resource(pooled)
                    self._stats.total_evicted += 1

            self._available = healthy

            # Ensure minimum pool size
            current = len(self._available) + len(self._in_use)
            needed = self.min_size - current

        # Create resources to meet minimum
        for _ in range(max(0, needed)):
            try:
                pooled = await self._create_resource_async()
                with self._lock:
                    self._available.append(pooled)
            except Exception as e:
                logger.warning(f"Pool '{self.name}': failed to create resource: {e}")
                break

    async def close(self) -> None:
        """Close the pool and cleanup all resources."""
        self._closed = True

        await self.stop_maintenance()

        with self._lock:
            # Destroy all available resources
            while self._available:
                pooled = self._available.popleft()
                self._destroy_resource(pooled)

            # Mark in-use resources for cleanup when returned
            for pooled in self._in_use:
                pooled.healthy = False

        logger.info(f"Pool '{self.name}' closed")

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self._lock:
            return self._stats.to_dict()
