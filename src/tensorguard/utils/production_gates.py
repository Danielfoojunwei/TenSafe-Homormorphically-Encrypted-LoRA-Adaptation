"""
Production Gates Module.

Provides utilities for enforcing production environment requirements
and validating critical configuration.
"""

import os
import sys
from typing import Optional


def is_production() -> bool:
    """
    Check if running in production environment.

    Returns True if TG_ENVIRONMENT is set to 'production'.
    """
    return os.getenv("TG_ENVIRONMENT", "").lower() == "production"


def require_env(
    var_name: str,
    remediation: Optional[str] = None,
    min_length: Optional[int] = None,
) -> str:
    """
    Require an environment variable to be set.

    In production, raises SystemExit if the variable is not set or doesn't
    meet requirements. In development, logs a warning and returns empty string.

    Args:
        var_name: Name of the environment variable
        remediation: Message explaining how to fix the issue
        min_length: Minimum required length for the value

    Returns:
        The environment variable value

    Raises:
        SystemExit: If in production and variable is missing/invalid
    """
    value = os.getenv(var_name, "")

    if not value:
        message = f"Required environment variable {var_name} is not set."
        if remediation:
            message += f"\n{remediation}"

        if is_production():
            print(f"FATAL: {message}", file=sys.stderr)
            sys.exit(1)
        else:
            return ""

    if min_length and len(value) < min_length:
        message = f"Environment variable {var_name} must be at least {min_length} characters."
        if remediation:
            message += f"\n{remediation}"

        if is_production():
            print(f"FATAL: {message}", file=sys.stderr)
            sys.exit(1)

    return value


def require_database_url() -> str:
    """Require DATABASE_URL to be set in production."""
    return require_env(
        "DATABASE_URL",
        remediation="Set DATABASE_URL to a PostgreSQL connection string: postgresql://user:pass@host:5432/db",
    )


def require_secret_key() -> str:
    """Require TG_SECRET_KEY to be set in production."""
    return require_env(
        "TG_SECRET_KEY",
        remediation='Generate with: python -c "import secrets; print(secrets.token_hex(32))"',
        min_length=32,
    )
