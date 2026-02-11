"""Background Health Checker for TenSafe Status Page.

Performs periodic health checks on system components and automatically
creates incidents on failures.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import httpx

from .models import (
    ComponentState,
    CreateIncidentRequest,
    IncidentSeverity,
    IncidentStatus,
    UpdateIncidentRequest,
)
from .status import StatusService, get_status_service

logger = logging.getLogger(__name__)


class CheckResult(Enum):
    """Health check result."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    component_id: str
    result: CheckResult
    response_time_ms: float
    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_component_state(self) -> ComponentState:
        """Convert result to component state."""
        mapping = {
            CheckResult.HEALTHY: ComponentState.OPERATIONAL,
            CheckResult.DEGRADED: ComponentState.DEGRADED,
            CheckResult.UNHEALTHY: ComponentState.MAJOR_OUTAGE,
            CheckResult.TIMEOUT: ComponentState.PARTIAL_OUTAGE,
            CheckResult.ERROR: ComponentState.PARTIAL_OUTAGE,
        }
        return mapping.get(self.result, ComponentState.UNKNOWN)


@dataclass
class HealthCheckConfig:
    """Configuration for health checker."""

    # Check intervals
    check_interval_seconds: int = 30
    initial_delay_seconds: int = 5

    # Timeouts
    http_timeout_seconds: float = 10.0
    tcp_timeout_seconds: float = 5.0

    # Thresholds
    degraded_latency_ms: float = 1000.0  # Consider degraded above this
    consecutive_failures_for_incident: int = 3

    # Auto-recovery
    auto_recovery_checks: int = 2  # Successful checks before auto-recovery

    # Retry configuration
    retry_count: int = 2
    retry_delay_seconds: float = 1.0


class ComponentHealthChecker(ABC):
    """Abstract base class for component health checkers."""

    def __init__(self, component_id: str, config: Optional[HealthCheckConfig] = None):
        """Initialize health checker.

        Args:
            component_id: Component identifier
            config: Health check configuration
        """
        self.component_id = component_id
        self.config = config or HealthCheckConfig()

    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check.

        Returns:
            HealthCheckResult indicating component health
        """
        pass


class HTTPHealthChecker(ComponentHealthChecker):
    """HTTP endpoint health checker."""

    def __init__(
        self,
        component_id: str,
        endpoint: str,
        config: Optional[HealthCheckConfig] = None,
        expected_status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        """Initialize HTTP health checker.

        Args:
            component_id: Component identifier
            endpoint: HTTP endpoint to check
            config: Health check configuration
            expected_status: Expected HTTP status code
            headers: Optional headers to include
            verify_ssl: Whether to verify SSL certificates
        """
        super().__init__(component_id, config)
        self.endpoint = endpoint
        self.expected_status = expected_status
        self.headers = headers or {}
        self.verify_ssl = verify_ssl

    async def check_health(self) -> HealthCheckResult:
        """Check HTTP endpoint health."""
        start_time = time.monotonic()

        try:
            async with httpx.AsyncClient(
                timeout=self.config.http_timeout_seconds,
                verify=self.verify_ssl,
            ) as client:
                response = await client.get(
                    self.endpoint,
                    headers=self.headers,
                )

            elapsed_ms = (time.monotonic() - start_time) * 1000

            if response.status_code == self.expected_status:
                result = CheckResult.HEALTHY
                if elapsed_ms > self.config.degraded_latency_ms:
                    result = CheckResult.DEGRADED

                return HealthCheckResult(
                    component_id=self.component_id,
                    result=result,
                    response_time_ms=elapsed_ms,
                    message=f"HTTP {response.status_code}",
                    metadata={"status_code": response.status_code},
                )
            else:
                return HealthCheckResult(
                    component_id=self.component_id,
                    result=CheckResult.UNHEALTHY,
                    response_time_ms=elapsed_ms,
                    message=f"Unexpected status: {response.status_code}",
                    metadata={"status_code": response.status_code},
                )

        except httpx.TimeoutException:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.TIMEOUT,
                response_time_ms=elapsed_ms,
                message=f"Timeout after {self.config.http_timeout_seconds}s",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.ERROR,
                response_time_ms=elapsed_ms,
                message=str(e),
            )


class TCPHealthChecker(ComponentHealthChecker):
    """TCP port health checker."""

    def __init__(
        self,
        component_id: str,
        host: str,
        port: int,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize TCP health checker.

        Args:
            component_id: Component identifier
            host: Host to check
            port: Port to check
            config: Health check configuration
        """
        super().__init__(component_id, config)
        self.host = host
        self.port = port

    async def check_health(self) -> HealthCheckResult:
        """Check TCP port connectivity."""
        start_time = time.monotonic()

        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port),
                timeout=self.config.tcp_timeout_seconds,
            )
            writer.close()
            await writer.wait_closed()

            elapsed_ms = (time.monotonic() - start_time) * 1000

            result = CheckResult.HEALTHY
            if elapsed_ms > self.config.degraded_latency_ms:
                result = CheckResult.DEGRADED

            return HealthCheckResult(
                component_id=self.component_id,
                result=result,
                response_time_ms=elapsed_ms,
                message="TCP connection successful",
                metadata={"host": self.host, "port": self.port},
            )

        except asyncio.TimeoutError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.TIMEOUT,
                response_time_ms=elapsed_ms,
                message=f"Connection timeout to {self.host}:{self.port}",
            )
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.ERROR,
                response_time_ms=elapsed_ms,
                message=str(e),
            )


class DatabaseHealthChecker(ComponentHealthChecker):
    """Database health checker with query test."""

    def __init__(
        self,
        component_id: str,
        connection_func: Callable,
        query: str = "SELECT 1",
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize database health checker.

        Args:
            component_id: Component identifier
            connection_func: Function to get database connection
            query: Health check query
            config: Health check configuration
        """
        super().__init__(component_id, config)
        self.connection_func = connection_func
        self.query = query

    async def check_health(self) -> HealthCheckResult:
        """Check database health."""
        start_time = time.monotonic()

        try:
            # Get connection and execute query
            conn = self.connection_func()

            # Handle both sync and async connections
            if asyncio.iscoroutinefunction(conn.execute):
                await conn.execute(self.query)
            else:
                conn.execute(self.query)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            result = CheckResult.HEALTHY
            if elapsed_ms > self.config.degraded_latency_ms:
                result = CheckResult.DEGRADED

            return HealthCheckResult(
                component_id=self.component_id,
                result=result,
                response_time_ms=elapsed_ms,
                message="Database query successful",
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.ERROR,
                response_time_ms=elapsed_ms,
                message=f"Database error: {str(e)}",
            )


class CustomHealthChecker(ComponentHealthChecker):
    """Custom health checker with user-provided check function."""

    def __init__(
        self,
        component_id: str,
        check_func: Callable[[], Tuple[bool, Optional[str]]],
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize custom health checker.

        Args:
            component_id: Component identifier
            check_func: Function returning (is_healthy, message)
            config: Health check configuration
        """
        super().__init__(component_id, config)
        self.check_func = check_func

    async def check_health(self) -> HealthCheckResult:
        """Execute custom health check."""
        start_time = time.monotonic()

        try:
            # Support both sync and async check functions
            if asyncio.iscoroutinefunction(self.check_func):
                is_healthy, message = await self.check_func()
            else:
                is_healthy, message = self.check_func()

            elapsed_ms = (time.monotonic() - start_time) * 1000

            result = CheckResult.HEALTHY if is_healthy else CheckResult.UNHEALTHY

            return HealthCheckResult(
                component_id=self.component_id,
                result=result,
                response_time_ms=elapsed_ms,
                message=message,
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return HealthCheckResult(
                component_id=self.component_id,
                result=CheckResult.ERROR,
                response_time_ms=elapsed_ms,
                message=str(e),
            )


class HealthChecker:
    """Background health checker service.

    Manages multiple component health checkers and integrates with
    the status service for automatic incident creation.

    Example:
        >>> checker = HealthChecker()
        >>> checker.register_http_check("api", "https://api.tensafe.io/health")
        >>> checker.register_tcp_check("database", "localhost", 5432)
        >>> await checker.start()
    """

    def __init__(
        self,
        status_service: Optional[StatusService] = None,
        config: Optional[HealthCheckConfig] = None,
    ):
        """Initialize health checker.

        Args:
            status_service: Status service for updates
            config: Health check configuration
        """
        self._status_service = status_service or get_status_service()
        self._config = config or HealthCheckConfig()
        self._checkers: Dict[str, ComponentHealthChecker] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()

        # Track failures for auto-incident creation
        self._consecutive_failures: Dict[str, int] = {}
        self._consecutive_successes: Dict[str, int] = {}
        self._active_auto_incidents: Dict[str, str] = {}  # component_id -> incident_id

        # Metrics
        self._check_count = 0
        self._failure_count = 0
        self._last_check_time: Optional[datetime] = None

    def register_checker(self, checker: ComponentHealthChecker):
        """Register a component health checker.

        Args:
            checker: Health checker instance
        """
        with self._lock:
            self._checkers[checker.component_id] = checker
            self._consecutive_failures[checker.component_id] = 0
            self._consecutive_successes[checker.component_id] = 0
            logger.info(f"Registered health checker for: {checker.component_id}")

    def register_http_check(
        self,
        component_id: str,
        endpoint: str,
        expected_status: int = 200,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True,
    ):
        """Register an HTTP health check.

        Args:
            component_id: Component identifier
            endpoint: HTTP endpoint to check
            expected_status: Expected HTTP status code
            headers: Optional headers
            verify_ssl: Whether to verify SSL
        """
        checker = HTTPHealthChecker(
            component_id=component_id,
            endpoint=endpoint,
            config=self._config,
            expected_status=expected_status,
            headers=headers,
            verify_ssl=verify_ssl,
        )
        self.register_checker(checker)

    def register_tcp_check(
        self,
        component_id: str,
        host: str,
        port: int,
    ):
        """Register a TCP health check.

        Args:
            component_id: Component identifier
            host: Host to check
            port: Port to check
        """
        checker = TCPHealthChecker(
            component_id=component_id,
            host=host,
            port=port,
            config=self._config,
        )
        self.register_checker(checker)

    def register_custom_check(
        self,
        component_id: str,
        check_func: Callable[[], Tuple[bool, Optional[str]]],
    ):
        """Register a custom health check.

        Args:
            component_id: Component identifier
            check_func: Function returning (is_healthy, message)
        """
        checker = CustomHealthChecker(
            component_id=component_id,
            check_func=check_func,
            config=self._config,
        )
        self.register_checker(checker)

    async def start(self):
        """Start background health checking."""
        if self._running:
            logger.warning("Health checker already running")
            return

        self._running = True
        logger.info("Starting health checker")

        # Initial delay
        await asyncio.sleep(self._config.initial_delay_seconds)

        self._task = asyncio.create_task(self._check_loop())

    async def stop(self):
        """Stop background health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health checker stopped")

    async def _check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                await self._run_all_checks()
                self._last_check_time = datetime.utcnow()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            await asyncio.sleep(self._config.check_interval_seconds)

    async def _run_all_checks(self):
        """Run all registered health checks concurrently."""
        if not self._checkers:
            return

        # Run all checks concurrently
        tasks = [
            self._run_single_check(checker)
            for checker in self._checkers.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Health check error: {result}")

    async def _run_single_check(self, checker: ComponentHealthChecker) -> HealthCheckResult:
        """Run a single health check with retry logic.

        Args:
            checker: Health checker to run

        Returns:
            Health check result
        """
        component_id = checker.component_id
        last_result = None

        for attempt in range(self._config.retry_count + 1):
            result = await checker.check_health()
            last_result = result
            self._check_count += 1

            if result.result == CheckResult.HEALTHY:
                break

            if attempt < self._config.retry_count:
                await asyncio.sleep(self._config.retry_delay_seconds)

        # Update status service
        self._status_service.update_component_status(
            component_id,
            last_result.to_component_state(),
            response_time_ms=last_result.response_time_ms,
            metadata=last_result.metadata,
        )

        # Handle failure tracking
        if last_result.result in (CheckResult.UNHEALTHY, CheckResult.TIMEOUT, CheckResult.ERROR):
            self._handle_failure(component_id, last_result)
        else:
            self._handle_success(component_id, last_result)

        return last_result

    def _handle_failure(self, component_id: str, result: HealthCheckResult):
        """Handle a failed health check.

        Args:
            component_id: Component that failed
            result: Check result
        """
        with self._lock:
            self._consecutive_failures[component_id] = self._consecutive_failures.get(component_id, 0) + 1
            self._consecutive_successes[component_id] = 0
            self._failure_count += 1

            failures = self._consecutive_failures[component_id]

            # Create auto-incident after consecutive failures
            if (
                failures >= self._config.consecutive_failures_for_incident
                and component_id not in self._active_auto_incidents
            ):
                self._create_auto_incident(component_id, result)

    def _handle_success(self, component_id: str, result: HealthCheckResult):
        """Handle a successful health check.

        Args:
            component_id: Component that succeeded
            result: Check result
        """
        with self._lock:
            self._consecutive_failures[component_id] = 0
            self._consecutive_successes[component_id] = self._consecutive_successes.get(component_id, 0) + 1

            successes = self._consecutive_successes[component_id]

            # Auto-resolve incident after consecutive successes
            if (
                component_id in self._active_auto_incidents
                and successes >= self._config.auto_recovery_checks
            ):
                self._resolve_auto_incident(component_id)

    def _create_auto_incident(self, component_id: str, result: HealthCheckResult):
        """Create an automatic incident for a failing component.

        Args:
            component_id: Failing component
            result: Check result
        """
        try:
            request = CreateIncidentRequest(
                title=f"Automated: {component_id} health check failures",
                severity=IncidentSeverity.P2_MAJOR,
                affected_components=[component_id],
                initial_message=(
                    f"Automated incident created after {self._config.consecutive_failures_for_incident} "
                    f"consecutive health check failures. "
                    f"Last error: {result.message or 'Unknown'}"
                ),
            )

            incident = self._status_service.create_incident(request, created_by="health_checker")
            self._active_auto_incidents[component_id] = incident.id

            logger.warning(f"Auto-incident created for {component_id}: {incident.id}")

        except Exception as e:
            logger.error(f"Failed to create auto-incident: {e}")

    def _resolve_auto_incident(self, component_id: str):
        """Resolve an automatic incident after recovery.

        Args:
            component_id: Recovered component
        """
        try:
            incident_id = self._active_auto_incidents.pop(component_id, None)
            if not incident_id:
                return

            request = UpdateIncidentRequest(
                status=IncidentStatus.RESOLVED,
                message=(
                    f"Automated resolution: {component_id} has passed "
                    f"{self._config.auto_recovery_checks} consecutive health checks."
                ),
            )

            self._status_service.update_incident(incident_id, request, updated_by="health_checker")

            logger.info(f"Auto-incident resolved for {component_id}: {incident_id}")

        except Exception as e:
            logger.error(f"Failed to resolve auto-incident: {e}")

    async def run_check(self, component_id: str) -> Optional[HealthCheckResult]:
        """Run a single health check on demand.

        Args:
            component_id: Component to check

        Returns:
            Health check result or None if component not found
        """
        checker = self._checkers.get(component_id)
        if not checker:
            logger.warning(f"No health checker registered for: {component_id}")
            return None

        return await self._run_single_check(checker)

    def get_stats(self) -> Dict[str, Any]:
        """Get health checker statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "running": self._running,
                "registered_checkers": len(self._checkers),
                "check_count": self._check_count,
                "failure_count": self._failure_count,
                "last_check_time": self._last_check_time.isoformat() if self._last_check_time else None,
                "active_auto_incidents": len(self._active_auto_incidents),
                "consecutive_failures": dict(self._consecutive_failures),
            }


# ==============================================================================
# Factory Functions
# ==============================================================================


def create_default_health_checker(
    status_service: Optional[StatusService] = None,
    base_url: str = "http://localhost:8000",
) -> HealthChecker:
    """Create health checker with default TenSafe component checks.

    Args:
        status_service: Optional status service
        base_url: Base URL for HTTP checks

    Returns:
        Configured HealthChecker
    """
    checker = HealthChecker(status_service=status_service)

    # Register default checks
    checker.register_http_check("api", f"{base_url}/health")
    checker.register_http_check("training", f"{base_url}/api/v1/training/health")
    checker.register_http_check("inference", f"{base_url}/api/v1/inference/health")

    # These would be configured with actual database/service endpoints
    # checker.register_tcp_check("database", "localhost", 5432)
    # checker.register_tcp_check("queue", "localhost", 6379)

    return checker


async def start_background_health_checker(
    status_service: Optional[StatusService] = None,
    config: Optional[HealthCheckConfig] = None,
) -> HealthChecker:
    """Start background health checker.

    Args:
        status_service: Optional status service
        config: Optional health check configuration

    Returns:
        Running HealthChecker
    """
    checker = HealthChecker(status_service=status_service, config=config)
    await checker.start()
    return checker
