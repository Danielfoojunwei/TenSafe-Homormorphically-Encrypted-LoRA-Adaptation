"""
Health Check Aggregation System.

Provides comprehensive health monitoring:
- Component-level health checks
- Aggregated system health status
- Degraded state detection
- Dependency health tracking
- Periodic background checks

Usage:
    health = HealthAggregator()

    # Register components
    health.register("database", db_health_check)
    health.register("kms", kms_health_check)
    health.register("cache", cache_health_check, critical=False)

    # Get aggregated health
    status = await health.check_all()
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status for a single component."""

    name: str
    status: HealthStatus
    message: Optional[str] = None
    latency_ms: Optional[float] = None
    last_check: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)
    critical: bool = True
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "details": self.details,
            "critical": self.critical,
        }


@dataclass
class SystemHealth:
    """Aggregated system health status."""

    status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "status": self.status.value,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "components": {
                name: comp.to_dict() for name, comp in self.components.items()
            },
        }


HealthCheckFunc = Union[
    Callable[[], Dict[str, Any]],
    Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
]


@dataclass
class HealthCheck:
    """Configuration for a health check."""

    name: str
    check_func: HealthCheckFunc
    timeout: float = 5.0
    critical: bool = True
    interval: float = 30.0
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True


class HealthAggregator:
    """
    Aggregates health checks from multiple components.

    Features:
    - Register multiple health check functions
    - Parallel health check execution
    - Degraded state detection
    - Caching of health results
    - Background periodic checking
    """

    def __init__(
        self,
        cache_ttl: float = 5.0,
        default_timeout: float = 5.0,
    ):
        """
        Initialize health aggregator.

        Args:
            cache_ttl: How long to cache health results
            default_timeout: Default timeout for health checks
        """
        self.cache_ttl = cache_ttl
        self.default_timeout = default_timeout

        self._checks: Dict[str, HealthCheck] = {}
        self._results: Dict[str, ComponentHealth] = {}
        self._last_check: Optional[float] = None
        self._lock = threading.RLock()
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

    def register(
        self,
        name: str,
        check_func: HealthCheckFunc,
        timeout: Optional[float] = None,
        critical: bool = True,
        interval: float = 30.0,
        failure_threshold: int = 3,
        success_threshold: int = 2,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Component name
            check_func: Function returning health status dict
            timeout: Timeout for this check
            critical: Whether this component is critical
            interval: Check interval for background checking
            failure_threshold: Failures before marking unhealthy
            success_threshold: Successes before marking healthy
        """
        with self._lock:
            self._checks[name] = HealthCheck(
                name=name,
                check_func=check_func,
                timeout=timeout or self.default_timeout,
                critical=critical,
                interval=interval,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
            )
            self._results[name] = ComponentHealth(
                name=name,
                status=HealthStatus.UNKNOWN,
                critical=critical,
            )

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._results.pop(name, None)

    async def check_component(
        self,
        name: str,
        force: bool = False,
    ) -> ComponentHealth:
        """
        Check health of a single component.

        Args:
            name: Component name
            force: Force check even if cached

        Returns:
            Component health status
        """
        with self._lock:
            check = self._checks.get(name)
            if check is None:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message="Component not registered",
                )

            # Check cache
            result = self._results.get(name)
            if (
                result
                and result.last_check
                and not force
                and (datetime.now(timezone.utc) - result.last_check).total_seconds()
                < self.cache_ttl
            ):
                return result

        # Execute check
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(check.check_func):
                check_result = await asyncio.wait_for(
                    check.check_func(),
                    timeout=check.timeout,
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                check_result = await asyncio.wait_for(
                    loop.run_in_executor(None, check.check_func),
                    timeout=check.timeout,
                )

            latency_ms = (time.time() - start_time) * 1000

            # Parse result
            status_str = check_result.get("status", "healthy")
            if status_str == "healthy":
                status = HealthStatus.HEALTHY
            elif status_str == "degraded":
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.UNHEALTHY

            with self._lock:
                existing = self._results.get(name)
                consecutive_failures = 0
                consecutive_successes = (existing.consecutive_successes if existing else 0) + 1

                # Apply success threshold
                if status == HealthStatus.HEALTHY and consecutive_successes < check.success_threshold:
                    if existing and existing.status == HealthStatus.UNHEALTHY:
                        status = HealthStatus.DEGRADED

                result = ComponentHealth(
                    name=name,
                    status=status,
                    message=check_result.get("message"),
                    latency_ms=latency_ms,
                    last_check=datetime.now(timezone.utc),
                    details=check_result,
                    critical=check.critical,
                    consecutive_failures=consecutive_failures,
                    consecutive_successes=consecutive_successes,
                )
                self._results[name] = result
                return result

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000

            with self._lock:
                existing = self._results.get(name)
                consecutive_failures = (existing.consecutive_failures if existing else 0) + 1

                # Apply failure threshold
                if consecutive_failures >= check.failure_threshold:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED

                result = ComponentHealth(
                    name=name,
                    status=status,
                    message=f"Health check timed out after {check.timeout}s",
                    latency_ms=latency_ms,
                    last_check=datetime.now(timezone.utc),
                    critical=check.critical,
                    consecutive_failures=consecutive_failures,
                    consecutive_successes=0,
                )
                self._results[name] = result
                return result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            with self._lock:
                existing = self._results.get(name)
                consecutive_failures = (existing.consecutive_failures if existing else 0) + 1

                # Apply failure threshold
                if consecutive_failures >= check.failure_threshold:
                    status = HealthStatus.UNHEALTHY
                else:
                    status = HealthStatus.DEGRADED

                result = ComponentHealth(
                    name=name,
                    status=status,
                    message=f"Health check failed: {e}",
                    latency_ms=latency_ms,
                    last_check=datetime.now(timezone.utc),
                    critical=check.critical,
                    consecutive_failures=consecutive_failures,
                    consecutive_successes=0,
                )
                self._results[name] = result
                return result

    async def check_all(
        self,
        force: bool = False,
        parallel: bool = True,
    ) -> SystemHealth:
        """
        Check health of all registered components.

        Args:
            force: Force check even if cached
            parallel: Run checks in parallel

        Returns:
            Aggregated system health
        """
        with self._lock:
            component_names = list(self._checks.keys())

        if not component_names:
            return SystemHealth(
                status=HealthStatus.HEALTHY,
                components={},
                timestamp=datetime.now(timezone.utc),
                summary="No components registered",
            )

        # Execute checks
        if parallel:
            tasks = [
                self.check_component(name, force=force) for name in component_names
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            components = {}
            for name, result in zip(component_names, results):
                if isinstance(result, Exception):
                    components[name] = ComponentHealth(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check error: {result}",
                        last_check=datetime.now(timezone.utc),
                    )
                else:
                    components[name] = result
        else:
            components = {}
            for name in component_names:
                components[name] = await self.check_component(name, force=force)

        # Calculate aggregate status
        status = self._calculate_aggregate_status(components)
        summary = self._generate_summary(components, status)

        return SystemHealth(
            status=status,
            components=components,
            timestamp=datetime.now(timezone.utc),
            summary=summary,
        )

    def _calculate_aggregate_status(
        self,
        components: Dict[str, ComponentHealth],
    ) -> HealthStatus:
        """Calculate aggregate status from component statuses."""
        if not components:
            return HealthStatus.HEALTHY

        has_critical_unhealthy = False
        has_critical_degraded = False
        has_any_unhealthy = False
        has_any_degraded = False

        for comp in components.values():
            if comp.status == HealthStatus.UNHEALTHY:
                has_any_unhealthy = True
                if comp.critical:
                    has_critical_unhealthy = True
            elif comp.status == HealthStatus.DEGRADED:
                has_any_degraded = True
                if comp.critical:
                    has_critical_degraded = True

        if has_critical_unhealthy:
            return HealthStatus.UNHEALTHY
        if has_critical_degraded or has_any_unhealthy:
            return HealthStatus.DEGRADED
        if has_any_degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _generate_summary(
        self,
        components: Dict[str, ComponentHealth],
        status: HealthStatus,
    ) -> str:
        """Generate summary message."""
        total = len(components)
        healthy = sum(1 for c in components.values() if c.status == HealthStatus.HEALTHY)
        degraded = sum(1 for c in components.values() if c.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for c in components.values() if c.status == HealthStatus.UNHEALTHY)

        if status == HealthStatus.HEALTHY:
            return f"All {total} components healthy"
        elif status == HealthStatus.DEGRADED:
            issues = []
            if unhealthy > 0:
                issues.append(f"{unhealthy} unhealthy")
            if degraded > 0:
                issues.append(f"{degraded} degraded")
            return f"System degraded: {', '.join(issues)} of {total} components"
        else:
            critical_down = [
                c.name for c in components.values()
                if c.status == HealthStatus.UNHEALTHY and c.critical
            ]
            return f"System unhealthy: critical components down: {', '.join(critical_down)}"

    async def start_background_checks(self) -> None:
        """Start background health check loop."""
        if self._running:
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info("Health check background loop started")

    async def stop_background_checks(self) -> None:
        """Stop background health check loop."""
        self._running = False

        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None

        logger.info("Health check background loop stopped")

    async def _background_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                # Find the next check due
                now = time.time()
                min_wait = 1.0

                with self._lock:
                    for name, check in self._checks.items():
                        if not check.enabled:
                            continue

                        result = self._results.get(name)
                        if result and result.last_check:
                            elapsed = (
                                datetime.now(timezone.utc) - result.last_check
                            ).total_seconds()
                            if elapsed >= check.interval:
                                # Check is due
                                asyncio.create_task(self.check_component(name, force=True))
                            else:
                                # Calculate wait time
                                wait = check.interval - elapsed
                                min_wait = min(min_wait, wait)
                        else:
                            # Never checked, check now
                            asyncio.create_task(self.check_component(name, force=True))

                await asyncio.sleep(min_wait)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check background loop error: {e}")
                await asyncio.sleep(5.0)

    def get_cached_status(self) -> SystemHealth:
        """Get cached health status without running new checks."""
        with self._lock:
            components = dict(self._results)

        if not components:
            return SystemHealth(
                status=HealthStatus.UNKNOWN,
                components={},
                timestamp=datetime.now(timezone.utc),
                summary="No health data available",
            )

        status = self._calculate_aggregate_status(components)
        summary = self._generate_summary(components, status)

        return SystemHealth(
            status=status,
            components=components,
            timestamp=datetime.now(timezone.utc),
            summary=summary,
        )

    def is_ready(self) -> bool:
        """Check if system is ready (all critical components healthy)."""
        with self._lock:
            for name, check in self._checks.items():
                if not check.critical:
                    continue

                result = self._results.get(name)
                if result is None or result.status == HealthStatus.UNHEALTHY:
                    return False

        return True

    def is_live(self) -> bool:
        """Check if system is live (not completely failed)."""
        with self._lock:
            for name, check in self._checks.items():
                if not check.critical:
                    continue

                result = self._results.get(name)
                if result and result.status != HealthStatus.UNHEALTHY:
                    return True

        return True  # Live by default if no critical components registered
