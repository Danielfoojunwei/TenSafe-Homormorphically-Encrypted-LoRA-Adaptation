"""
TenSafe Security Hardening Module.

This module provides production hardening utilities:
- Input validation with type-safe constraints
- Secure error handling (no information leakage)
- Rate limiting and resource controls
- Secrets sanitization for logging
- Security audit logging

Usage:
    from tensafe.core.security import (
        validate_input,
        sanitize_error,
        RateLimiter,
        SecureLogger,
        SecurityContext,
    )

    # Validate input
    validated = validate_input(user_input, InputConstraints(
        max_length=1000,
        pattern=r'^[a-zA-Z0-9_-]+$',
    ))

    # Sanitize errors for external responses
    safe_error = sanitize_error(exception)

    # Rate limiting
    limiter = RateLimiter(max_requests=100, window_seconds=60)
    if not limiter.allow(tenant_id):
        raise RateLimitExceeded()

    # Secure logging
    logger = SecureLogger(__name__)
    logger.info("Processing request", extra={"api_key": "ts-xxx"})  # Automatically redacted
"""

from __future__ import annotations

import functools
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# ==============================================================================
# Input Validation
# ==============================================================================


class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation failed for '{field}': {message}")


@dataclass
class InputConstraints:
    """Constraints for input validation."""

    # String constraints
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_chars: Optional[str] = None
    forbidden_chars: Optional[str] = None

    # Numeric constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    must_be_positive: bool = False
    must_be_integer: bool = False

    # List constraints
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: bool = False

    # Type constraints
    required: bool = True
    nullable: bool = False
    allowed_types: Optional[Tuple[type, ...]] = None

    # Custom validator
    custom_validator: Optional[Callable[[Any], bool]] = None

    # Security constraints
    no_null_bytes: bool = True
    no_control_chars: bool = True
    sanitize_html: bool = False


# Pre-compiled patterns for common validations
PATTERNS = {
    "alphanumeric": re.compile(r'^[a-zA-Z0-9]+$'),
    "alphanumeric_dash": re.compile(r'^[a-zA-Z0-9_-]+$'),
    "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
    "uuid": re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I),
    "api_key": re.compile(r'^ts-[a-zA-Z0-9]{32,64}$'),
    "safe_filename": re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$'),
    "no_traversal": re.compile(r'^(?!.*\.\.)(?!.*[/\\]).*$'),
}


def validate_input(
    value: Any,
    constraints: InputConstraints,
    field_name: str = "input",
) -> Any:
    """
    Validate input against constraints.

    Args:
        value: Value to validate
        constraints: Validation constraints
        field_name: Name of field for error messages

    Returns:
        Validated (and possibly sanitized) value

    Raises:
        ValidationError: If validation fails
    """
    # Handle None
    if value is None:
        if constraints.nullable:
            return None
        if constraints.required:
            raise ValidationError(field_name, "Value is required")
        return None

    # Type validation
    if constraints.allowed_types:
        if not isinstance(value, constraints.allowed_types):
            raise ValidationError(
                field_name,
                f"Invalid type: expected {constraints.allowed_types}, got {type(value).__name__}"
            )

    # String validations
    if isinstance(value, str):
        value = _validate_string(value, constraints, field_name)

    # Numeric validations
    elif isinstance(value, (int, float)):
        value = _validate_numeric(value, constraints, field_name)

    # List validations
    elif isinstance(value, (list, tuple)):
        value = _validate_list(value, constraints, field_name)

    # Custom validator
    if constraints.custom_validator:
        if not constraints.custom_validator(value):
            raise ValidationError(field_name, "Custom validation failed")

    return value


def _validate_string(
    value: str,
    constraints: InputConstraints,
    field_name: str,
) -> str:
    """Validate string value."""
    # Security: null bytes
    if constraints.no_null_bytes and '\x00' in value:
        raise ValidationError(field_name, "Null bytes not allowed")

    # Security: control characters
    if constraints.no_control_chars:
        if any(ord(c) < 32 and c not in '\n\r\t' for c in value):
            raise ValidationError(field_name, "Control characters not allowed")

    # Length constraints
    if constraints.min_length is not None and len(value) < constraints.min_length:
        raise ValidationError(field_name, f"Too short: minimum {constraints.min_length} characters")

    if constraints.max_length is not None and len(value) > constraints.max_length:
        raise ValidationError(field_name, f"Too long: maximum {constraints.max_length} characters")

    # Pattern matching
    if constraints.pattern:
        pattern = PATTERNS.get(constraints.pattern) or re.compile(constraints.pattern)
        if not pattern.match(value):
            raise ValidationError(field_name, "Does not match required pattern")

    # Allowed characters
    if constraints.allowed_chars:
        invalid = set(value) - set(constraints.allowed_chars)
        if invalid:
            raise ValidationError(field_name, f"Invalid characters: {invalid}")

    # Forbidden characters
    if constraints.forbidden_chars:
        forbidden = set(value) & set(constraints.forbidden_chars)
        if forbidden:
            raise ValidationError(field_name, f"Forbidden characters: {forbidden}")

    # HTML sanitization
    if constraints.sanitize_html:
        value = _sanitize_html(value)

    return value


def _validate_numeric(
    value: Union[int, float],
    constraints: InputConstraints,
    field_name: str,
) -> Union[int, float]:
    """Validate numeric value."""
    if constraints.must_be_integer and not isinstance(value, int):
        if isinstance(value, float) and not value.is_integer():
            raise ValidationError(field_name, "Must be an integer")
        value = int(value)

    if constraints.must_be_positive and value <= 0:
        raise ValidationError(field_name, "Must be positive")

    if constraints.min_value is not None and value < constraints.min_value:
        raise ValidationError(field_name, f"Below minimum: {constraints.min_value}")

    if constraints.max_value is not None and value > constraints.max_value:
        raise ValidationError(field_name, f"Above maximum: {constraints.max_value}")

    return value


def _validate_list(
    value: Union[list, tuple],
    constraints: InputConstraints,
    field_name: str,
) -> list:
    """Validate list value."""
    if constraints.min_items is not None and len(value) < constraints.min_items:
        raise ValidationError(field_name, f"Too few items: minimum {constraints.min_items}")

    if constraints.max_items is not None and len(value) > constraints.max_items:
        raise ValidationError(field_name, f"Too many items: maximum {constraints.max_items}")

    if constraints.unique_items and len(value) != len(set(value)):
        raise ValidationError(field_name, "Items must be unique")

    return list(value)


def _sanitize_html(value: str) -> str:
    """Remove potentially dangerous HTML."""
    # Simple HTML tag removal
    return re.sub(r'<[^>]+>', '', value)


# Common validation presets
class Validators:
    """Pre-configured validators for common use cases."""

    TENANT_ID = InputConstraints(
        pattern="uuid",
        max_length=36,
    )

    API_KEY = InputConstraints(
        pattern="api_key",
        min_length=35,
        max_length=70,
    )

    MODEL_NAME = InputConstraints(
        max_length=256,
        pattern="alphanumeric_dash",
    )

    SAFE_FILENAME = InputConstraints(
        max_length=255,
        pattern="safe_filename",
    )

    POSITIVE_INT = InputConstraints(
        must_be_integer=True,
        must_be_positive=True,
    )

    EPSILON = InputConstraints(
        min_value=0.0,
        max_value=100.0,
    )

    NOISE_MULTIPLIER = InputConstraints(
        min_value=0.0,
        max_value=100.0,
    )

    LEARNING_RATE = InputConstraints(
        min_value=0.0,
        max_value=1.0,
    )


# ==============================================================================
# Secure Error Handling
# ==============================================================================


class ErrorCategory(Enum):
    """Categories of errors for safe responses."""
    VALIDATION = "validation_error"
    AUTHENTICATION = "authentication_error"
    AUTHORIZATION = "authorization_error"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    INTERNAL = "internal_error"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTED = "resource_exhausted"


@dataclass
class SafeError:
    """A sanitized error safe for external responses."""
    category: ErrorCategory
    message: str
    code: str
    details: Optional[Dict[str, Any]] = None
    retry_after: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "error": {
                "category": self.category.value,
                "message": self.message,
                "code": self.code,
            }
        }
        if self.details:
            result["error"]["details"] = self.details
        if self.retry_after:
            result["error"]["retry_after"] = self.retry_after
        return result


# Error message mappings (internal -> external)
SAFE_ERROR_MESSAGES = {
    "validation": "Invalid request parameters",
    "authentication": "Authentication failed",
    "authorization": "Access denied",
    "not_found": "Resource not found",
    "rate_limited": "Too many requests",
    "internal": "An internal error occurred",
    "timeout": "Request timed out",
    "resource_exhausted": "Resource limit exceeded",
}


def sanitize_error(
    exception: Exception,
    include_type: bool = False,
    log_internal: bool = True,
) -> SafeError:
    """
    Convert an exception to a safe error for external responses.

    Never exposes internal details, stack traces, or sensitive information.

    Args:
        exception: The exception to sanitize
        include_type: Include exception type name (for debugging)
        log_internal: Log the full exception internally

    Returns:
        SafeError safe for external response
    """
    # Log the full exception internally
    if log_internal:
        logger.exception(f"Internal error: {type(exception).__name__}")

    # Map to safe error
    if isinstance(exception, ValidationError):
        return SafeError(
            category=ErrorCategory.VALIDATION,
            message=f"Validation failed for '{exception.field}'",
            code="VALIDATION_ERROR",
            details={"field": exception.field},
        )

    if isinstance(exception, PermissionError):
        return SafeError(
            category=ErrorCategory.AUTHORIZATION,
            message=SAFE_ERROR_MESSAGES["authorization"],
            code="ACCESS_DENIED",
        )

    if isinstance(exception, FileNotFoundError):
        return SafeError(
            category=ErrorCategory.NOT_FOUND,
            message=SAFE_ERROR_MESSAGES["not_found"],
            code="NOT_FOUND",
        )

    if isinstance(exception, TimeoutError):
        return SafeError(
            category=ErrorCategory.TIMEOUT,
            message=SAFE_ERROR_MESSAGES["timeout"],
            code="TIMEOUT",
        )

    if isinstance(exception, MemoryError):
        return SafeError(
            category=ErrorCategory.RESOURCE_EXHAUSTED,
            message=SAFE_ERROR_MESSAGES["resource_exhausted"],
            code="RESOURCE_EXHAUSTED",
        )

    # Default: internal error (no details exposed)
    return SafeError(
        category=ErrorCategory.INTERNAL,
        message=SAFE_ERROR_MESSAGES["internal"],
        code="INTERNAL_ERROR",
    )


def safe_error_handler(func: F) -> F:
    """
    Decorator that catches exceptions and returns safe errors.

    Usage:
        @safe_error_handler
        def my_api_endpoint():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            safe_error = sanitize_error(e)
            return safe_error.to_dict(), _error_status_code(safe_error.category)

    return wrapper  # type: ignore


def _error_status_code(category: ErrorCategory) -> int:
    """Map error category to HTTP status code."""
    mapping = {
        ErrorCategory.VALIDATION: 400,
        ErrorCategory.AUTHENTICATION: 401,
        ErrorCategory.AUTHORIZATION: 403,
        ErrorCategory.NOT_FOUND: 404,
        ErrorCategory.RATE_LIMITED: 429,
        ErrorCategory.TIMEOUT: 504,
        ErrorCategory.RESOURCE_EXHAUSTED: 503,
        ErrorCategory.INTERNAL: 500,
    }
    return mapping.get(category, 500)


# ==============================================================================
# Rate Limiting
# ==============================================================================


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, limit: int, window: int, retry_after: int):
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded: {limit} requests per {window}s")


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 100
    window_seconds: int = 60
    burst_multiplier: float = 1.5  # Allow bursts up to 1.5x
    per_endpoint: bool = False  # Separate limits per endpoint


class RateLimiter:
    """
    Token bucket rate limiter with sliding window.

    Thread-safe implementation suitable for multi-tenant environments.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(
        self,
        key: str,
        cost: int = 1,
        endpoint: Optional[str] = None,
    ) -> Tuple[bool, int]:
        """
        Check if a request should be allowed.

        Args:
            key: Rate limit key (e.g., tenant_id, IP address)
            cost: Request cost (for weighted limiting)
            endpoint: Optional endpoint for per-endpoint limiting

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        if self.config.per_endpoint and endpoint:
            key = f"{key}:{endpoint}"

        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            # Get bucket and prune old entries
            bucket = self._buckets[key]
            bucket[:] = [t for t in bucket if t > window_start]

            # Check limit
            current_count = len(bucket)
            max_allowed = int(self.config.max_requests * self.config.burst_multiplier)

            if current_count + cost > max_allowed:
                # Calculate retry after
                if bucket:
                    oldest = min(bucket)
                    retry_after = int(oldest + self.config.window_seconds - now) + 1
                else:
                    retry_after = self.config.window_seconds
                return False, max(1, retry_after)

            # Add request timestamps
            for _ in range(cost):
                bucket.append(now)

            return True, 0

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.config.window_seconds

        with self._lock:
            bucket = self._buckets.get(key, [])
            current_count = sum(1 for t in bucket if t > window_start)
            return max(0, self.config.max_requests - current_count)

    def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        with self._lock:
            self._buckets.pop(key, None)

    def cleanup(self) -> int:
        """Remove expired entries. Returns number of keys cleaned."""
        now = time.time()
        window_start = now - self.config.window_seconds
        cleaned = 0

        with self._lock:
            keys_to_remove = []
            for key, bucket in self._buckets.items():
                bucket[:] = [t for t in bucket if t > window_start]
                if not bucket:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._buckets[key]
                cleaned += 1

        return cleaned


# Global rate limiters
_rate_limiters: Dict[str, RateLimiter] = {}


def get_rate_limiter(name: str = "default") -> RateLimiter:
    """Get or create a named rate limiter."""
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter()
    return _rate_limiters[name]


def rate_limit(
    key_func: Callable[..., str],
    limiter_name: str = "default",
) -> Callable[[F], F]:
    """
    Decorator for rate limiting.

    Args:
        key_func: Function to extract rate limit key from arguments
        limiter_name: Name of rate limiter to use
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            limiter = get_rate_limiter(limiter_name)
            allowed, retry_after = limiter.allow(key)

            if not allowed:
                raise RateLimitExceeded(
                    limiter.config.max_requests,
                    limiter.config.window_seconds,
                    retry_after,
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore
    return decorator


# ==============================================================================
# Secrets Sanitization for Logging
# ==============================================================================


# Patterns for sensitive data
SENSITIVE_PATTERNS = [
    (re.compile(r'(api[_-]?key|apikey)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})', re.I), r'\1=***REDACTED***'),
    (re.compile(r'(password|passwd|pwd)["\']?\s*[:=]\s*["\']?([^\s"\']+)', re.I), r'\1=***REDACTED***'),
    (re.compile(r'(secret|token)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_-]{10,})', re.I), r'\1=***REDACTED***'),
    (re.compile(r'(bearer|authorization)["\']?\s*[:=]\s*["\']?([a-zA-Z0-9_.-]+)', re.I), r'\1=***REDACTED***'),
    (re.compile(r'ts-[a-zA-Z0-9]{32,64}'), '***API_KEY_REDACTED***'),  # TenSafe API keys
    (re.compile(r'-----BEGIN[^-]+-----[^-]+-----END[^-]+-----', re.S), '***CERTIFICATE_REDACTED***'),
]

# Keys to redact in dictionaries
SENSITIVE_KEYS = {
    'api_key', 'apikey', 'api-key',
    'password', 'passwd', 'pwd',
    'secret', 'secret_key', 'secretkey',
    'token', 'access_token', 'refresh_token',
    'authorization', 'auth', 'bearer',
    'private_key', 'privatekey',
    'credential', 'credentials',
    'master_key', 'dek', 'kek',
}


def sanitize_for_logging(value: Any, depth: int = 0) -> Any:
    """
    Recursively sanitize a value for safe logging.

    Redacts sensitive data like API keys, passwords, tokens.

    Args:
        value: Value to sanitize
        depth: Current recursion depth (to prevent infinite loops)

    Returns:
        Sanitized value safe for logging
    """
    if depth > 10:
        return "***MAX_DEPTH***"

    if isinstance(value, str):
        result = value
        for pattern, replacement in SENSITIVE_PATTERNS:
            result = pattern.sub(replacement, result)
        return result

    if isinstance(value, dict):
        return {
            k: "***REDACTED***" if k.lower() in SENSITIVE_KEYS
            else sanitize_for_logging(v, depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [sanitize_for_logging(v, depth + 1) for v in value]

    return value


class SecureLogFilter(logging.Filter):
    """Log filter that redacts sensitive information."""

    def filter(self, record: logging.LogRecord) -> bool:
        # Sanitize message
        if isinstance(record.msg, str):
            record.msg = sanitize_for_logging(record.msg)

        # Sanitize args
        if record.args:
            record.args = tuple(sanitize_for_logging(arg) for arg in record.args)

        # Sanitize extra fields
        for key in list(vars(record).keys()):
            if key not in logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__:
                setattr(record, key, sanitize_for_logging(getattr(record, key)))

        return True


class SecureLogger:
    """
    A logger that automatically sanitizes sensitive data.

    Usage:
        logger = SecureLogger(__name__)
        logger.info("API call", extra={"api_key": "ts-xxx"})
        # Logs: "API call" with api_key=***REDACTED***
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._filter = SecureLogFilter()
        self._logger.addFilter(self._filter)

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(sanitize_for_logging(msg), *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(sanitize_for_logging(msg), *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(sanitize_for_logging(msg), *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(sanitize_for_logging(msg), *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(sanitize_for_logging(msg), *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._logger.exception(sanitize_for_logging(msg), *args, **kwargs)


# ==============================================================================
# Resource Controls
# ==============================================================================


@dataclass
class ResourceLimits:
    """Resource limits for operations."""
    max_memory_mb: int = 1024
    max_time_seconds: int = 300
    max_file_size_mb: int = 100
    max_batch_size: int = 1000
    max_sequence_length: int = 8192
    max_concurrent_operations: int = 10


class ResourceLimitExceeded(Exception):
    """Raised when a resource limit is exceeded."""

    def __init__(self, resource: str, limit: Any, actual: Any):
        self.resource = resource
        self.limit = limit
        self.actual = actual
        super().__init__(f"Resource limit exceeded: {resource} (limit={limit}, actual={actual})")


def check_resource_limits(
    limits: ResourceLimits,
    memory_mb: Optional[int] = None,
    file_size_mb: Optional[int] = None,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
) -> None:
    """
    Check if resources are within limits.

    Raises:
        ResourceLimitExceeded: If any limit is exceeded
    """
    if memory_mb is not None and memory_mb > limits.max_memory_mb:
        raise ResourceLimitExceeded("memory", limits.max_memory_mb, memory_mb)

    if file_size_mb is not None and file_size_mb > limits.max_file_size_mb:
        raise ResourceLimitExceeded("file_size", limits.max_file_size_mb, file_size_mb)

    if batch_size is not None and batch_size > limits.max_batch_size:
        raise ResourceLimitExceeded("batch_size", limits.max_batch_size, batch_size)

    if sequence_length is not None and sequence_length > limits.max_sequence_length:
        raise ResourceLimitExceeded("sequence_length", limits.max_sequence_length, sequence_length)


class TimeoutContext:
    """
    Context manager for operation timeouts.

    Note: This uses threading and may not interrupt all operations.
    For true timeout guarantees, use multiprocessing or async.
    """

    def __init__(self, seconds: int, operation_name: str = "operation"):
        self.seconds = seconds
        self.operation_name = operation_name
        self._start_time: Optional[float] = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def check(self) -> None:
        """Check if timeout has been exceeded. Call periodically in long operations."""
        if self._start_time is None:
            return
        elapsed = time.time() - self._start_time
        if elapsed > self.seconds:
            raise TimeoutError(f"{self.operation_name} timed out after {self.seconds}s")

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def remaining(self) -> float:
        """Get remaining time in seconds."""
        return max(0.0, self.seconds - self.elapsed)


# ==============================================================================
# Security Context
# ==============================================================================


@dataclass
class SecurityContext:
    """
    Security context for requests.

    Tracks security-relevant information for audit logging.
    """
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    authenticated: bool = False
    permissions: Set[str] = field(default_factory=set)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Rate limit state
    rate_limit_key: Optional[str] = None
    rate_limit_remaining: Optional[int] = None

    def has_permission(self, permission: str) -> bool:
        """Check if context has a specific permission."""
        return permission in self.permissions or "admin" in self.permissions

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit logging."""
        return {
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "request_id": self.request_id,
            "client_ip": self.client_ip,
            "authenticated": self.authenticated,
            "timestamp": self.timestamp.isoformat(),
        }


# Thread-local storage for security context
_security_context = threading.local()


def get_security_context() -> Optional[SecurityContext]:
    """Get the current security context."""
    return getattr(_security_context, 'context', None)


def set_security_context(context: SecurityContext) -> None:
    """Set the current security context."""
    _security_context.context = context


def clear_security_context() -> None:
    """Clear the current security context."""
    _security_context.context = None
