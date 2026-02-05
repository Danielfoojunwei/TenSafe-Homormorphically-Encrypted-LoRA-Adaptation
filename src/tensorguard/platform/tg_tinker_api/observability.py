"""
TG-Tinker Observability Module.

Provides:
- Structured logging with correlation IDs
- Prometheus metrics for SLO monitoring
- Request tracing context
- Health check endpoints
"""

import logging
import os
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ==============================================================================
# Correlation ID Context
# ==============================================================================

# Context variable for request correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_operation_id: ContextVar[Optional[str]] = ContextVar("operation_id", default=None)


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def get_tenant_id() -> Optional[str]:
    """Get the current tenant ID."""
    return _tenant_id.get()


def set_tenant_id(tid: str) -> None:
    """Set the tenant ID for the current context."""
    _tenant_id.set(tid)


def get_operation_id() -> Optional[str]:
    """Get the current operation ID."""
    return _operation_id.get()


def set_operation_id(oid: str) -> None:
    """Set the operation ID for the current context."""
    _operation_id.set(oid)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


# ==============================================================================
# Structured Logging
# ==============================================================================


class StructuredLogFormatter(logging.Formatter):
    """
    Structured log formatter that outputs JSON-like logs.

    Includes correlation ID, tenant ID, and other context.
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add context
        if cid := get_correlation_id():
            log_data["correlation_id"] = cid
        if tid := get_tenant_id():
            log_data["tenant_id"] = tid
        if oid := get_operation_id():
            log_data["operation_id"] = oid

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "asctime",
            ):
                log_data[key] = value

        # Format as pseudo-JSON (for readability in console)
        parts = [f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}" for k, v in log_data.items()]
        return " ".join(parts)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
) -> None:
    """
    Configure structured logging.

    Args:
        level: Log level
        json_format: If True, output JSON logs (for production)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add structured handler
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredLogFormatter())
    root_logger.addHandler(handler)


# ==============================================================================
# Metrics (Prometheus-compatible)
# ==============================================================================


class MetricType(str, Enum):
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """A single metric."""
    name: str
    type: MetricType
    help: str
    labels: List[str] = field(default_factory=list)
    values: Dict[tuple, float] = field(default_factory=dict)
    buckets: List[float] = field(default_factory=lambda: [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])

    def inc(self, labels: Dict[str, str] = None, value: float = 1.0) -> None:
        """Increment a counter or gauge."""
        key = self._labels_to_key(labels)
        self.values[key] = self.values.get(key, 0.0) + value

    def set(self, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        key = self._labels_to_key(labels)
        self.values[key] = value

    def observe(self, value: float, labels: Dict[str, str] = None) -> None:
        """Observe a histogram value."""
        key = self._labels_to_key(labels)
        # Store sum and count for histogram
        sum_key = key + ("_sum",)
        count_key = key + ("_count",)
        self.values[sum_key] = self.values.get(sum_key, 0.0) + value
        self.values[count_key] = self.values.get(count_key, 0.0) + 1

        # Update bucket counts
        for bucket in self.buckets:
            bucket_key = key + (f"_bucket_le_{bucket}",)
            if value <= bucket:
                self.values[bucket_key] = self.values.get(bucket_key, 0.0) + 1

        # Inf bucket
        inf_key = key + ("_bucket_le_+Inf",)
        self.values[inf_key] = self.values.get(inf_key, 0.0) + 1

    def _labels_to_key(self, labels: Dict[str, str] = None) -> tuple:
        """Convert labels dict to hashable key."""
        if not labels:
            return ()
        return tuple(sorted(labels.items()))


class MetricsRegistry:
    """Registry for all metrics."""

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}

    def counter(
        self,
        name: str,
        help: str,
        labels: List[str] = None,
    ) -> Metric:
        """Create or get a counter metric."""
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.COUNTER,
                help=help,
                labels=labels or [],
            )
        return self._metrics[name]

    def gauge(
        self,
        name: str,
        help: str,
        labels: List[str] = None,
    ) -> Metric:
        """Create or get a gauge metric."""
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.GAUGE,
                help=help,
                labels=labels or [],
            )
        return self._metrics[name]

    def histogram(
        self,
        name: str,
        help: str,
        labels: List[str] = None,
        buckets: List[float] = None,
    ) -> Metric:
        """Create or get a histogram metric."""
        if name not in self._metrics:
            self._metrics[name] = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                help=help,
                labels=labels or [],
                buckets=buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            )
        return self._metrics[name]

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        for name, metric in self._metrics.items():
            # Help line
            lines.append(f"# HELP {name} {metric.help}")
            lines.append(f"# TYPE {name} {metric.type.value}")

            # Metric values
            for labels_key, value in metric.values.items():
                if labels_key:
                    # Parse labels
                    label_str = ",".join(f'{k}="{v}"' for k, v in labels_key if not k.startswith("_"))
                    suffix = "".join(k for k in labels_key if isinstance(k, str) and k.startswith("_"))

                    if suffix:
                        # Histogram bucket or sum/count
                        if "_bucket_le_" in suffix:
                            bucket = suffix.replace("_bucket_le_", "")
                            if label_str:
                                lines.append(f'{name}_bucket{{{label_str},le="{bucket}"}} {value}')
                            else:
                                lines.append(f'{name}_bucket{{le="{bucket}"}} {value}')
                        elif suffix == "_sum":
                            if label_str:
                                lines.append(f"{name}_sum{{{label_str}}} {value}")
                            else:
                                lines.append(f"{name}_sum {value}")
                        elif suffix == "_count":
                            if label_str:
                                lines.append(f"{name}_count{{{label_str}}} {value}")
                            else:
                                lines.append(f"{name}_count {value}")
                    else:
                        if label_str:
                            lines.append(f"{name}{{{label_str}}} {value}")
                        else:
                            lines.append(f"{name} {value}")
                else:
                    lines.append(f"{name} {value}")

        return "\n".join(lines)


# Global metrics registry
metrics = MetricsRegistry()

# Pre-defined metrics
request_count = metrics.counter(
    "tg_tinker_requests_total",
    "Total number of API requests",
    labels=["method", "endpoint", "status"],
)

request_duration = metrics.histogram(
    "tg_tinker_request_duration_seconds",
    "Request duration in seconds",
    labels=["method", "endpoint"],
)

training_clients_active = metrics.gauge(
    "tg_tinker_training_clients_active",
    "Number of active training clients",
    labels=["tenant_id"],
)

queue_depth = metrics.gauge(
    "tg_tinker_queue_depth",
    "Number of jobs in queue",
    labels=["status"],
)

dp_epsilon_total = metrics.counter(
    "tg_tinker_dp_epsilon_total",
    "Total differential privacy epsilon spent",
    labels=["tenant_id", "training_client_id"],
)

he_operations_total = metrics.counter(
    "tg_tinker_he_operations_total",
    "Total homomorphic encryption operations",
    labels=["operation", "mode"],
)

artifact_storage_bytes = metrics.gauge(
    "tg_tinker_artifact_storage_bytes",
    "Total bytes stored in artifact storage",
    labels=["tenant_id"],
)


# ==============================================================================
# Middleware
# ==============================================================================


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracing and metrics.

    Adds correlation ID, captures metrics, and logs requests.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate or extract correlation ID
        correlation_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        set_correlation_id(correlation_id)

        # Extract tenant from auth header if present
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            # Set a placeholder tenant ID for metrics
            # Real tenant ID is set by the auth middleware
            pass

        # Start timing
        start_time = time.time()

        # Log request
        logger = logging.getLogger("tg_tinker.http")
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "http_method": request.method,
                "http_path": request.url.path,
            },
        )

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            labels = {
                "method": request.method,
                "endpoint": request.url.path,
                "status": str(response.status_code),
            }
            request_count.inc(labels)
            request_duration.observe(duration, {
                "method": request.method,
                "endpoint": request.url.path,
            })

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} {response.status_code}",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": response.status_code,
                    "duration_ms": int(duration * 1000),
                },
            )

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            return response

        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time

            # Record error metrics
            labels = {
                "method": request.method,
                "endpoint": request.url.path,
                "status": "500",
            }
            request_count.inc(labels)

            # Log error
            logger.exception(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": int(duration * 1000),
                },
            )

            raise


# ==============================================================================
# Health Check
# ==============================================================================


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    healthy: bool
    message: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HealthChecker:
    """
    Health checker for liveness and readiness probes.

    Checks:
    - Database connectivity
    - Queue health
    - Storage backend
    """

    def __init__(self):
        self._checks: List[Callable[[], HealthCheckResult]] = []

    def register(self, check: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        self._checks.append(check)

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = []
        overall_healthy = True

        for check in self._checks:
            try:
                start = time.time()
                result = check()
                result.latency_ms = (time.time() - start) * 1000
                results.append(result)

                if not result.healthy:
                    overall_healthy = False

            except Exception as e:
                results.append(HealthCheckResult(
                    name=check.__name__,
                    healthy=False,
                    message=str(e),
                ))
                overall_healthy = False

        return {
            "healthy": overall_healthy,
            "checks": [
                {
                    "name": r.name,
                    "healthy": r.healthy,
                    "message": r.message,
                    "latency_ms": r.latency_ms,
                    "details": r.details,
                }
                for r in results
            ],
        }


# Global health checker
health_checker = HealthChecker()


# ==============================================================================
# FastAPI Integration
# ==============================================================================


def add_observability_routes(app):
    """Add observability routes to FastAPI app."""
    from fastapi import APIRouter

    router = APIRouter(prefix="/internal", tags=["observability"])

    @router.get("/health")
    async def health():
        """Health check endpoint for liveness probes."""
        return {"status": "ok"}

    @router.get("/ready")
    async def ready():
        """Readiness check endpoint."""
        result = health_checker.check_all()
        if result["healthy"]:
            return result
        else:
            from fastapi import HTTPException
            raise HTTPException(status_code=503, detail=result)

    @router.get("/metrics")
    async def prometheus_metrics():
        """Prometheus metrics endpoint."""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=metrics.export(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    app.include_router(router)


# ==============================================================================
# Decorators
# ==============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def traced(operation: str) -> Callable[[F], F]:
    """
    Decorator to trace function execution.

    Adds logging and timing for the decorated function.
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation_id = f"op-{uuid.uuid4().hex[:8]}"
            set_operation_id(operation_id)

            logger = logging.getLogger(func.__module__)
            logger.info(f"Starting operation: {operation}")

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"Completed operation: {operation}",
                    extra={"duration_ms": int(duration * 1000)},
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.exception(
                    f"Failed operation: {operation}",
                    extra={"duration_ms": int(duration * 1000)},
                )
                raise

        return wrapper  # type: ignore
    return decorator


async def traced_async(operation: str) -> Callable[[F], F]:
    """Async version of traced decorator."""
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation_id = f"op-{uuid.uuid4().hex[:8]}"
            set_operation_id(operation_id)

            logger = logging.getLogger(func.__module__)
            logger.info(f"Starting operation: {operation}")

            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                logger.info(
                    f"Completed operation: {operation}",
                    extra={"duration_ms": int(duration * 1000)},
                )
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.exception(
                    f"Failed operation: {operation}",
                    extra={"duration_ms": int(duration * 1000)},
                )
                raise

        return wrapper  # type: ignore
    return decorator
