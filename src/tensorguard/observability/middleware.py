"""Tracing Middleware for TenSafe.

Provides automatic request tracing with privacy-aware attribute handling.
"""

import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Conditional OpenTelemetry import
OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    pass


class TenSafeTracingMiddleware(BaseHTTPMiddleware):
    """Middleware for automatic request tracing with privacy awareness.

    Features:
    - Automatic span creation for each request
    - Privacy-aware attribute redaction
    - Request/response metadata capture
    - Error tracking and status codes
    - Correlation ID propagation
    """

    # Fields to redact from traces
    SENSITIVE_PATTERNS = [
        re.compile(r'password', re.IGNORECASE),
        re.compile(r'token', re.IGNORECASE),
        re.compile(r'api[_-]?key', re.IGNORECASE),
        re.compile(r'secret', re.IGNORECASE),
        re.compile(r'credential', re.IGNORECASE),
        re.compile(r'authorization', re.IGNORECASE),
    ]

    def __init__(
        self,
        app,
        service_name: str = "tensafe",
        redact_sensitive: bool = True,
        exclude_paths: Optional[list] = None,
    ):
        """Initialize tracing middleware.

        Args:
            app: ASGI application
            service_name: Service name for spans
            redact_sensitive: Whether to redact sensitive data
            exclude_paths: Paths to exclude from tracing
        """
        super().__init__(app)
        self.service_name = service_name
        self.redact_sensitive = redact_sensitive
        self.exclude_paths = exclude_paths or ["/health", "/readiness", "/metrics"]

        if OTEL_AVAILABLE:
            self._tracer = trace.get_tracer(service_name)
        else:
            self._tracer = None

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        start_time = time.perf_counter()

        # Extract or generate correlation ID
        correlation_id = request.headers.get("X-Correlation-ID", str(time.time()))

        if self._tracer is not None and OTEL_AVAILABLE:
            return await self._dispatch_with_tracing(
                request, call_next, start_time, correlation_id
            )
        else:
            return await self._dispatch_without_tracing(
                request, call_next, start_time, correlation_id
            )

    async def _dispatch_with_tracing(
        self,
        request: Request,
        call_next: Callable,
        start_time: float,
        correlation_id: str,
    ) -> Response:
        """Dispatch with OpenTelemetry tracing."""
        span_name = f"{request.method} {request.url.path}"

        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.SERVER,
        ) as span:
            # Set span attributes
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.url", str(request.url))
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("http.host", request.url.hostname or "")
            span.set_attribute("correlation_id", correlation_id)

            # Add safe headers
            self._add_safe_headers(span, request.headers)

            try:
                response = await call_next(request)

                # Record response
                span.set_attribute("http.status_code", response.status_code)

                duration_ms = (time.perf_counter() - start_time) * 1000
                span.set_attribute("http.duration_ms", duration_ms)

                if response.status_code >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))

                # Add correlation ID to response
                response.headers["X-Correlation-ID"] = correlation_id

                return response

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    async def _dispatch_without_tracing(
        self,
        request: Request,
        call_next: Callable,
        start_time: float,
        correlation_id: str,
    ) -> Response:
        """Dispatch without tracing (fallback)."""
        try:
            response = await call_next(request)
            response.headers["X-Correlation-ID"] = correlation_id

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"{request.method} {request.url.path} "
                f"completed in {duration_ms:.2f}ms "
                f"status={response.status_code}"
            )

            return response

        except Exception as e:
            logger.exception(f"Request failed: {e}")
            raise

    def _add_safe_headers(self, span, headers: Dict[str, str]):
        """Add headers to span, redacting sensitive values."""
        for key, value in headers.items():
            if self._is_sensitive(key):
                span.set_attribute(f"http.header.{key}", "[REDACTED]")
            else:
                span.set_attribute(f"http.header.{key}", value)

    def _is_sensitive(self, key: str) -> bool:
        """Check if a key matches sensitive patterns."""
        if not self.redact_sensitive:
            return False

        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.search(key):
                return True

        return False


def create_span_decorator(tracer_name: str = "tensafe"):
    """Create a decorator for tracing functions.

    Example:
        ```python
        @trace_function("my_operation")
        def my_function(x, y):
            return x + y
        ```
    """
    if not OTEL_AVAILABLE:
        def noop_decorator(name: str):
            def decorator(func):
                return func
            return decorator
        return noop_decorator

    tracer = trace.get_tracer(tracer_name)

    def trace_function(name: str, attributes: Optional[Dict[str, Any]] = None):
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(name) as span:
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)

                    try:
                        result = func(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    return trace_function
