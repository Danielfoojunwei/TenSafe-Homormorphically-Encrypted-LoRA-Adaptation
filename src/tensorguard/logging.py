"""
TensorGuard Structured Logging Configuration.

Provides consistent logging across all TensorGuard components with:
- Structured JSON output for production
- Human-readable output for development
- Request ID correlation
- Sensitive data filtering

Usage:
    from tensorguard.logging import get_logger, configure_logging

    # At application startup
    configure_logging(level="INFO", json_format=True)

    # In modules
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"request_id": "abc123"})
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Sensitive field patterns to filter from logs
SENSITIVE_PATTERNS = frozenset(
    {
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "private_key",
        "privatekey",
        "secret_key",
        "secretkey",
        "access_key",
        "accesskey",
        "auth",
        "credential",
        "ssn",
        "credit_card",
    }
)


def _is_sensitive_key(key: str) -> bool:
    """Check if a key name suggests sensitive data."""
    key_lower = key.lower()
    return any(pattern in key_lower for pattern in SENSITIVE_PATTERNS)


def _filter_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively filter sensitive values from a dictionary."""
    if not isinstance(data, dict):
        return data

    filtered = {}
    for key, value in data.items():
        if _is_sensitive_key(key):
            filtered[key] = "[REDACTED]"
        elif isinstance(value, dict):
            filtered[key] = _filter_sensitive(value)
        elif isinstance(value, list):
            filtered[key] = [_filter_sensitive(item) if isinstance(item, dict) else item for item in value]
        else:
            filtered[key] = value
    return filtered


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.pathname:
            log_data["location"] = {
                "file": record.pathname.split("/")[-1],
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add extra fields (filtered for sensitive data)
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            }:
                if isinstance(value, dict):
                    extra_fields[key] = _filter_sensitive(value)
                elif not _is_sensitive_key(key):
                    extra_fields[key] = value
                else:
                    extra_fields[key] = "[REDACTED]"

        if extra_fields:
            log_data["extra"] = extra_fields

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class DevelopmentFormatter(logging.Formatter):
    """Human-readable formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Color support for terminal
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        # Basic format
        timestamp = datetime.utcnow().strftime("%H:%M:%S.%f")[:-3]
        level = f"{color}{record.levelname:8}{reset}"
        name = record.name.split(".")[-1][:15].ljust(15)
        message = record.getMessage()

        # Add request_id if present
        request_id = getattr(record, "request_id", None)
        if request_id:
            message = f"[{request_id[:8]}] {message}"

        formatted = f"{timestamp} {level} {name} {message}"

        # Add exception info
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def configure_logging(
    level: str = "INFO",
    json_format: Optional[bool] = None,
    stream: Any = None,
) -> None:
    """
    Configure logging for TensorGuard components.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON output. Default: True in production, False otherwise
        stream: Output stream. Default: sys.stderr
    """
    # Determine format based on environment if not specified
    if json_format is None:
        env = os.environ.get("TG_ENVIRONMENT", "development")
        json_format = env == "production"

    # Get root logger for tensorguard
    root_logger = logging.getLogger("tensorguard")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create handler
    handler = logging.StreamHandler(stream or sys.stderr)

    # Set formatter
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(DevelopmentFormatter())

    root_logger.addHandler(handler)

    # Don't propagate to root logger
    root_logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a TensorGuard module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Message", extra={"request_id": "123"})
    """
    # Ensure it's under tensorguard namespace
    if not name.startswith("tensorguard"):
        name = f"tensorguard.{name}"
    return logging.getLogger(name)


class LogContext:
    """
    Context manager for adding correlation IDs to logs.

    Usage:
        with LogContext(request_id="abc123") as ctx:
            logger.info("Processing")  # Automatically includes request_id
    """

    _current: Optional["LogContext"] = None

    def __init__(self, **kwargs: Any):
        self.context = kwargs
        self._previous: Optional[LogContext] = None

    def __enter__(self) -> "LogContext":
        self._previous = LogContext._current
        LogContext._current = self
        return self

    def __exit__(self, *args: Any) -> None:
        LogContext._current = self._previous

    @classmethod
    def get_current(cls) -> Dict[str, Any]:
        """Get the current logging context."""
        if cls._current:
            return cls._current.context.copy()
        return {}


# Initialize with default config when module is imported
# (can be reconfigured later with configure_logging())
if not logging.getLogger("tensorguard").handlers:
    configure_logging()


__all__ = [
    "configure_logging",
    "get_logger",
    "LogContext",
    "StructuredFormatter",
    "DevelopmentFormatter",
]
