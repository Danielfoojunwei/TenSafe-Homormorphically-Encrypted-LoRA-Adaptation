"""
Unified runtime environment resolver.

Single source of truth for determining the deployment environment.
All environment-sensitive code should use this module instead of
reading TG_ENVIRONMENT directly.

Accepted values for TENSAFE_ENV: local, dev, staging, production.
Unknown or missing values are treated as production (fail-closed).
"""

import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_VALID_ENVS = {"local", "dev", "staging", "production"}

# Legacy mapping: translate old TG_ENVIRONMENT values
_LEGACY_MAP = {
    "development": "dev",
    "test": "dev",
    "testing": "dev",
}


class Environment(str, Enum):
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


def _resolve_env() -> Environment:
    """Resolve environment from TENSAFE_ENV (preferred) or TG_ENVIRONMENT (legacy)."""
    raw = os.getenv("TENSAFE_ENV") or os.getenv("TG_ENVIRONMENT") or ""
    raw = raw.strip().lower()

    # Apply legacy mapping
    raw = _LEGACY_MAP.get(raw, raw)

    if raw in _VALID_ENVS:
        return Environment(raw)

    # Fail-closed: unknown/missing => production
    if raw:
        logger.warning(
            "TENSAFE_ENV=%r is not a recognized value (%s). "
            "Defaulting to production (fail-closed).",
            raw,
            ", ".join(sorted(_VALID_ENVS)),
        )
    else:
        logger.warning(
            "TENSAFE_ENV is not set. Defaulting to production (fail-closed). "
            "Set TENSAFE_ENV=local for development."
        )
    return Environment.PRODUCTION


# Module-level singleton, computed once at import time
ENVIRONMENT: Environment = _resolve_env()


def is_production() -> bool:
    """Return True if running in production or staging."""
    return ENVIRONMENT in (Environment.PRODUCTION, Environment.STAGING)


def is_local_or_dev() -> bool:
    """Return True if running in local or dev."""
    return ENVIRONMENT in (Environment.LOCAL, Environment.DEV)


def require_env_var(name: str, *, allow_missing_in_dev: bool = False) -> Optional[str]:
    """
    Get an environment variable, failing hard in production if missing.

    Args:
        name: The environment variable name.
        allow_missing_in_dev: If True, return None in local/dev instead of raising.

    Returns:
        The variable value, or None if allow_missing_in_dev and in dev.

    Raises:
        RuntimeError: If the variable is missing in production/staging.
    """
    value = os.getenv(name)
    if value:
        return value

    if is_production():
        raise RuntimeError(
            f"FATAL: {name} must be set in {ENVIRONMENT.value} environment. "
            f"Set TENSAFE_ENV=local to use development defaults."
        )

    if allow_missing_in_dev:
        return None

    logger.warning("%s is not set (env=%s)", name, ENVIRONMENT.value)
    return None


def validate_no_demo_mode() -> None:
    """
    Hard-fail if demo mode is enabled outside local/dev.

    This prevents accidentally running with demo/insecure defaults
    in staging or production.
    """
    demo_flags = [
        ("TG_DEMO_MODE", os.getenv("TG_DEMO_MODE", "").lower()),
        ("TG_ADMIN_DEMO_MODE", os.getenv("TG_ADMIN_DEMO_MODE", "").lower()),
    ]

    for flag_name, flag_value in demo_flags:
        if flag_value in ("true", "1", "yes"):
            if is_production():
                raise RuntimeError(
                    f"FATAL: {flag_name}=true is not allowed in {ENVIRONMENT.value}. "
                    f"Demo mode can only be used with TENSAFE_ENV=local or TENSAFE_ENV=dev."
                )
            logger.warning(
                "SECURITY: %s is enabled (env=%s). This is NOT safe for production.",
                flag_name,
                ENVIRONMENT.value,
            )
