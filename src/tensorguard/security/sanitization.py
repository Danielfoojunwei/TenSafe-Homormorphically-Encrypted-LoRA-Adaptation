"""
Input Sanitization and Validation Module.

Provides comprehensive input validation:
- Path traversal prevention
- HTML sanitization
- SQL injection prevention patterns
- Command injection prevention
- Email validation
- URL validation
"""

import html
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Union
from urllib.parse import urlparse

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


# ============================================================================
# PATTERNS FOR VALIDATION
# ============================================================================

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS = [
    re.compile(r"\.\./"),  # ../
    re.compile(r"\.\.\\"),  # ..\
    re.compile(r"%2e%2e[/\\]", re.IGNORECASE),  # URL encoded
    re.compile(r"\.\.%2f", re.IGNORECASE),
    re.compile(r"%2e%2e/", re.IGNORECASE),
    re.compile(r"\.\./"),
    re.compile(r"^/etc/"),  # Absolute paths
    re.compile(r"^/var/"),
    re.compile(r"^/usr/"),
    re.compile(r"^/home/"),
    re.compile(r"^/root/"),
    re.compile(r"^[A-Za-z]:\\"),  # Windows paths
]

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r"(\s|^)(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER|CREATE|TRUNCATE)\s", re.IGNORECASE),
    re.compile(r"--\s*$"),  # SQL comment
    re.compile(r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP)", re.IGNORECASE),
    re.compile(r"'\s*OR\s*'", re.IGNORECASE),
    re.compile(r"'\s*AND\s*'", re.IGNORECASE),
    re.compile(r"1\s*=\s*1"),
    re.compile(r"'\s*=\s*'"),
]

# Command injection patterns
COMMAND_INJECTION_PATTERNS = [
    re.compile(r"[;&|`$]"),  # Shell metacharacters
    re.compile(r"\$\("),  # Command substitution
    re.compile(r"`.*`"),  # Backtick substitution
    re.compile(r"\|\|"),  # OR
    re.compile(r"&&"),  # AND
    re.compile(r">\s*/dev/"),  # Redirect to device
    re.compile(r"<\s*/dev/"),
]

# XSS patterns
XSS_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"on\w+\s*=", re.IGNORECASE),  # Event handlers
    re.compile(r"<iframe", re.IGNORECASE),
    re.compile(r"<object", re.IGNORECASE),
    re.compile(r"<embed", re.IGNORECASE),
    re.compile(r"<svg.*onload", re.IGNORECASE),
    re.compile(r"expression\s*\(", re.IGNORECASE),  # CSS expression
]

# Email pattern (RFC 5322 simplified)
EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)

# Safe filename pattern
SAFE_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


# ============================================================================
# SANITIZATION FUNCTIONS
# ============================================================================


def sanitize_input(
    value: str,
    max_length: int = 1000,
    strip_whitespace: bool = True,
    normalize_unicode: bool = True,
    remove_null_bytes: bool = True,
) -> str:
    """
    Sanitize a string input.

    Args:
        value: Input string
        max_length: Maximum allowed length
        strip_whitespace: Strip leading/trailing whitespace
        normalize_unicode: Normalize Unicode characters
        remove_null_bytes: Remove null bytes

    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        value = str(value)

    # Remove null bytes
    if remove_null_bytes:
        value = value.replace("\x00", "")

    # Normalize Unicode
    if normalize_unicode:
        value = unicodedata.normalize("NFKC", value)

    # Strip whitespace
    if strip_whitespace:
        value = value.strip()

    # Truncate to max length
    if len(value) > max_length:
        value = value[:max_length]

    return value


def sanitize_path(
    path: str,
    base_path: Optional[str] = None,
    allow_absolute: bool = False,
) -> str:
    """
    Sanitize a file path to prevent path traversal.

    Args:
        path: Input path
        base_path: Base path that result must be under
        allow_absolute: Allow absolute paths

    Returns:
        Sanitized path

    Raises:
        ValueError: If path is invalid or contains traversal
    """
    # Check for path traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern.search(path):
            raise ValueError(f"Path traversal detected in: {path!r}")

    # Normalize the path
    normalized = os.path.normpath(path)

    # Check for absolute paths
    if os.path.isabs(normalized) and not allow_absolute:
        raise ValueError(f"Absolute paths not allowed: {path!r}")

    # If base_path provided, ensure result is under it
    if base_path:
        base_path = os.path.abspath(base_path)
        full_path = os.path.abspath(os.path.join(base_path, normalized))

        # Ensure the resolved path is under base_path
        if not full_path.startswith(base_path + os.sep) and full_path != base_path:
            raise ValueError(f"Path escapes base directory: {path!r}")

        return full_path

    return normalized


def sanitize_filename(
    filename: str,
    max_length: int = 255,
    replacement: str = "_",
) -> str:
    """
    Sanitize a filename.

    Args:
        filename: Input filename
        max_length: Maximum filename length
        replacement: Replacement for invalid characters

    Returns:
        Sanitized filename

    Raises:
        ValueError: If filename is empty after sanitization
    """
    # Remove path separators
    filename = os.path.basename(filename)

    # Remove null bytes
    filename = filename.replace("\x00", "")

    # Replace unsafe characters
    unsafe_chars = '<>:"/\\|?*\x00'
    for char in unsafe_chars:
        filename = filename.replace(char, replacement)

    # Remove leading/trailing dots and spaces
    filename = filename.strip(". ")

    # Collapse multiple replacements
    while replacement + replacement in filename:
        filename = filename.replace(replacement + replacement, replacement)

    # Truncate
    if len(filename) > max_length:
        # Preserve extension
        name, ext = os.path.splitext(filename)
        if ext:
            filename = name[: max_length - len(ext)] + ext
        else:
            filename = filename[:max_length]

    if not filename:
        raise ValueError("Filename is empty after sanitization")

    return filename


def sanitize_html(
    html_content: str,
    allowed_tags: Optional[Set[str]] = None,
    allowed_attributes: Optional[Dict[str, Set[str]]] = None,
) -> str:
    """
    Sanitize HTML content by escaping dangerous elements.

    For simple cases, use html.escape(). For complex HTML sanitization,
    consider using a dedicated library like bleach.

    Args:
        html_content: HTML content to sanitize
        allowed_tags: Tags to allow (not yet implemented)
        allowed_attributes: Attributes to allow per tag (not yet implemented)

    Returns:
        Sanitized HTML
    """
    # For API use, we typically just escape all HTML
    # For complex use cases, use bleach or similar
    return html.escape(html_content)


def sanitize_sql_identifier(identifier: str) -> str:
    """
    Sanitize a SQL identifier (table/column name).

    Args:
        identifier: SQL identifier

    Returns:
        Sanitized identifier

    Raises:
        ValueError: If identifier contains invalid characters
    """
    # Only allow alphanumeric and underscore
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", identifier):
        raise ValueError(f"Invalid SQL identifier: {identifier!r}")

    # Check for SQL keywords (basic list)
    sql_keywords = {
        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "TRUNCATE", "UNION", "JOIN", "WHERE", "FROM",
        "TABLE", "DATABASE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE",
    }

    if identifier.upper() in sql_keywords:
        raise ValueError(f"SQL keyword not allowed as identifier: {identifier}")

    return identifier


def validate_email(email: str) -> bool:
    """
    Validate an email address.

    Args:
        email: Email address to validate

    Returns:
        True if valid
    """
    if not email or len(email) > 254:
        return False

    return bool(EMAIL_PATTERN.match(email))


def validate_url(
    url: str,
    allowed_schemes: Optional[Set[str]] = None,
    require_tld: bool = True,
) -> bool:
    """
    Validate a URL.

    Args:
        url: URL to validate
        allowed_schemes: Allowed URL schemes
        require_tld: Require a TLD in the hostname

    Returns:
        True if valid
    """
    allowed_schemes = allowed_schemes or {"http", "https"}

    try:
        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in allowed_schemes:
            return False

        # Check hostname
        if not parsed.netloc:
            return False

        # Check for TLD
        if require_tld:
            hostname = parsed.hostname or ""
            if "." not in hostname:
                return False

        return True

    except Exception:
        return False


def check_sql_injection(value: str) -> bool:
    """
    Check if a value contains potential SQL injection.

    Args:
        value: Value to check

    Returns:
        True if potential SQL injection detected
    """
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(value):
            return True
    return False


def check_command_injection(value: str) -> bool:
    """
    Check if a value contains potential command injection.

    Args:
        value: Value to check

    Returns:
        True if potential command injection detected
    """
    for pattern in COMMAND_INJECTION_PATTERNS:
        if pattern.search(value):
            return True
    return False


def check_xss(value: str) -> bool:
    """
    Check if a value contains potential XSS.

    Args:
        value: Value to check

    Returns:
        True if potential XSS detected
    """
    for pattern in XSS_PATTERNS:
        if pattern.search(value):
            return True
    return False


# ============================================================================
# INPUT VALIDATOR
# ============================================================================


@dataclass
class ValidationRule:
    """A validation rule for input fields."""

    field_name: str
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    pattern: Optional[Pattern] = None
    validator: Optional[Callable[[Any], bool]] = None
    sanitizer: Optional[Callable[[Any], Any]] = None
    error_message: str = "Invalid input"


class InputValidator:
    """
    Input validator with customizable rules.

    Provides centralized input validation with support for
    custom rules and sanitizers.
    """

    def __init__(self):
        """Initialize input validator."""
        self._rules: Dict[str, List[ValidationRule]] = {}
        self._global_sanitizers: List[Callable[[str], str]] = []

    def add_rule(
        self,
        field_name: str,
        rule: ValidationRule,
    ) -> "InputValidator":
        """
        Add a validation rule.

        Args:
            field_name: Field to validate
            rule: Validation rule

        Returns:
            Self for chaining
        """
        if field_name not in self._rules:
            self._rules[field_name] = []
        self._rules[field_name].append(rule)
        return self

    def add_global_sanitizer(
        self,
        sanitizer: Callable[[str], str],
    ) -> "InputValidator":
        """
        Add a global sanitizer applied to all string inputs.

        Args:
            sanitizer: Sanitization function

        Returns:
            Self for chaining
        """
        self._global_sanitizers.append(sanitizer)
        return self

    def validate(
        self,
        data: Dict[str, Any],
        strict: bool = False,
    ) -> tuple[Dict[str, Any], List[str]]:
        """
        Validate and sanitize input data.

        Args:
            data: Input data dictionary
            strict: Raise exception on validation failure

        Returns:
            Tuple of (sanitized_data, errors)
        """
        sanitized = {}
        errors = []

        for field_name, value in data.items():
            # Apply global sanitizers to strings
            if isinstance(value, str):
                for sanitizer in self._global_sanitizers:
                    value = sanitizer(value)

            # Apply field-specific rules
            rules = self._rules.get(field_name, [])
            field_valid = True

            for rule in rules:
                # Apply sanitizer
                if rule.sanitizer:
                    value = rule.sanitizer(value)

                # Check length constraints
                if isinstance(value, str):
                    if rule.max_length and len(value) > rule.max_length:
                        errors.append(f"{field_name}: exceeds max length {rule.max_length}")
                        field_valid = False

                    if rule.min_length and len(value) < rule.min_length:
                        errors.append(f"{field_name}: below min length {rule.min_length}")
                        field_valid = False

                # Check pattern
                if rule.pattern and isinstance(value, str):
                    if not rule.pattern.match(value):
                        errors.append(f"{field_name}: {rule.error_message}")
                        field_valid = False

                # Run custom validator
                if rule.validator:
                    try:
                        if not rule.validator(value):
                            errors.append(f"{field_name}: {rule.error_message}")
                            field_valid = False
                    except Exception as e:
                        errors.append(f"{field_name}: validation error - {e}")
                        field_valid = False

            if field_valid:
                sanitized[field_name] = value

        if strict and errors:
            raise ValueError(f"Validation failed: {'; '.join(errors)}")

        return sanitized, errors


class ValidationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic input validation.

    Validates request parameters and body against security patterns.
    """

    def __init__(
        self,
        app,
        check_sql_injection: bool = True,
        check_command_injection: bool = True,
        check_xss: bool = True,
        max_body_size: int = 10 * 1024 * 1024,  # 10 MB
        exclude_paths: Optional[List[str]] = None,
    ):
        """
        Initialize validation middleware.

        Args:
            app: FastAPI application
            check_sql_injection: Check for SQL injection
            check_command_injection: Check for command injection
            check_xss: Check for XSS
            max_body_size: Maximum request body size
            exclude_paths: Paths to exclude from validation
        """
        super().__init__(app)
        self.check_sql_injection = check_sql_injection
        self.check_command_injection = check_command_injection
        self.check_xss = check_xss
        self.max_body_size = max_body_size
        self.exclude_paths = set(exclude_paths or [])

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Validate request inputs."""
        path = request.url.path

        # Check exclusions
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.max_body_size:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Request body too large",
                    )
            except ValueError:
                pass

        # Check query parameters
        for key, value in request.query_params.items():
            self._validate_value(f"query.{key}", value)

        # Check path parameters
        if hasattr(request, "path_params"):
            for key, value in request.path_params.items():
                self._validate_value(f"path.{key}", str(value))

        return await call_next(request)

    def _validate_value(self, field: str, value: str) -> None:
        """Validate a single value."""
        if self.check_sql_injection and check_sql_injection(value):
            logger.warning(
                f"Potential SQL injection blocked: {field}",
                extra={"field": field, "value_preview": value[:50]},
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected",
            )

        if self.check_command_injection and check_command_injection(value):
            logger.warning(
                f"Potential command injection blocked: {field}",
                extra={"field": field, "value_preview": value[:50]},
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected",
            )

        if self.check_xss and check_xss(value):
            logger.warning(
                f"Potential XSS blocked: {field}",
                extra={"field": field, "value_preview": value[:50]},
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input detected",
            )
