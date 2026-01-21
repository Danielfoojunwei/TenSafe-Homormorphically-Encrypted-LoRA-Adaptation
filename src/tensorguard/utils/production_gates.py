"""
Production Gates - Fail-Closed Startup Validation

This module provides utilities to enforce production invariants at startup.
In production mode (TG_ENVIRONMENT=production), missing required configuration
or dependencies will cause immediate startup failure with clear error messages.

Usage:
    from tensorguard.utils.production_gates import assert_production_invariants

    # Call at application startup
    assert_production_invariants()
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class ProductionGateError(RuntimeError):
    """Raised when a production gate check fails."""

    def __init__(self, gate_name: str, message: str, remediation: str):
        self.gate_name = gate_name
        self.remediation = remediation
        super().__init__(
            f"[PRODUCTION GATE FAILED: {gate_name}] {message}\n"
            f"REMEDIATION: {remediation}"
        )


@lru_cache(maxsize=1)
def is_production() -> bool:
    """Check if running in production mode."""
    return os.getenv("TG_ENVIRONMENT", "development").lower() == "production"


def is_demo_mode() -> bool:
    """
    Check if demo mode is allowed.

    Demo mode is ONLY allowed when:
    - TG_DEMO_MODE=true AND
    - TG_ENVIRONMENT != production

    This ensures that demo/mock data endpoints are never accessible
    in production environments.

    Returns:
        True if demo mode is active and allowed, False otherwise
    """
    demo_enabled = os.getenv("TG_DEMO_MODE", "false").lower() == "true"
    return demo_enabled and not is_production()


def require_env(
    var_name: str,
    remediation: Optional[str] = None,
    allow_empty: bool = False,
    min_length: Optional[int] = None,
) -> str:
    """
    Require an environment variable to be set in production.

    Args:
        var_name: Name of the environment variable
        remediation: Custom remediation message
        allow_empty: If False, empty string also fails
        min_length: Minimum required length (for secrets)

    Returns:
        The environment variable value

    Raises:
        ProductionGateError: If variable not set/invalid in production
    """
    value = os.getenv(var_name)

    if value is None:
        if is_production():
            raise ProductionGateError(
                gate_name=f"ENV_{var_name}",
                message=f"Required environment variable '{var_name}' is not set.",
                remediation=remediation or f"Set the {var_name} environment variable before starting."
            )
        else:
            logger.warning(f"[DEV MODE] Missing env var '{var_name}' - would fail in production")
            return ""

    if not allow_empty and value == "":
        if is_production():
            raise ProductionGateError(
                gate_name=f"ENV_{var_name}",
                message=f"Environment variable '{var_name}' is set but empty.",
                remediation=remediation or f"Provide a non-empty value for {var_name}."
            )
        else:
            logger.warning(f"[DEV MODE] Empty env var '{var_name}' - would fail in production")
            return ""

    if min_length is not None and len(value) < min_length:
        if is_production():
            raise ProductionGateError(
                gate_name=f"ENV_{var_name}_LENGTH",
                message=f"Environment variable '{var_name}' must be at least {min_length} characters.",
                remediation=f"Provide a value of at least {min_length} characters for {var_name}."
            )
        else:
            logger.warning(
                f"[DEV MODE] Env var '{var_name}' too short ({len(value)} < {min_length}) - would fail in production"
            )

    return value


def require_dependency(
    module_name: str,
    package_name: Optional[str] = None,
    remediation: Optional[str] = None,
) -> Any:
    """
    Require a Python dependency to be importable in production.

    Args:
        module_name: Name of the module to import
        package_name: Display name for error messages (defaults to module_name)
        remediation: Custom remediation message

    Returns:
        The imported module

    Raises:
        ProductionGateError: If module not importable in production
    """
    import importlib

    display_name = package_name or module_name

    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        if is_production():
            raise ProductionGateError(
                gate_name=f"DEP_{module_name.upper().replace('.', '_')}",
                message=f"Required dependency '{display_name}' is not installed.",
                remediation=remediation or f"Install {display_name}: pip install {display_name}"
            ) from e
        else:
            logger.warning(f"[DEV MODE] Missing dependency '{display_name}' - would fail in production")
            return None


def require_file(
    file_path: str,
    description: str = "file",
    remediation: Optional[str] = None,
) -> str:
    """
    Require a file to exist in production.

    Args:
        file_path: Path to the required file
        description: Human-readable description of the file
        remediation: Custom remediation message

    Returns:
        The file path if it exists

    Raises:
        ProductionGateError: If file doesn't exist in production
    """
    if not os.path.isfile(file_path):
        if is_production():
            raise ProductionGateError(
                gate_name=f"FILE_{os.path.basename(file_path).upper().replace('.', '_')}",
                message=f"Required {description} not found: {file_path}",
                remediation=remediation or f"Create or provide the {description} at: {file_path}"
            )
        else:
            logger.warning(f"[DEV MODE] Missing {description} '{file_path}' - would fail in production")
            return ""

    return file_path


def require_directory(
    dir_path: str,
    description: str = "directory",
    create_if_missing: bool = False,
) -> str:
    """
    Require a directory to exist in production.

    Args:
        dir_path: Path to the required directory
        description: Human-readable description
        create_if_missing: If True, create the directory if it doesn't exist

    Returns:
        The directory path

    Raises:
        ProductionGateError: If directory doesn't exist and can't be created
    """
    if not os.path.isdir(dir_path):
        if create_if_missing:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.info(f"Created {description}: {dir_path}")
                return dir_path
            except OSError as e:
                if is_production():
                    raise ProductionGateError(
                        gate_name=f"DIR_{os.path.basename(dir_path).upper()}",
                        message=f"Cannot create {description}: {dir_path}",
                        remediation=f"Create the directory manually or fix permissions: {e}"
                    ) from e

        if is_production():
            raise ProductionGateError(
                gate_name=f"DIR_{os.path.basename(dir_path).upper()}",
                message=f"Required {description} not found: {dir_path}",
                remediation=f"Create the {description} at: {dir_path}"
            )
        else:
            logger.warning(f"[DEV MODE] Missing {description} '{dir_path}' - would fail in production")
            return ""

    return dir_path


def block_demo_mode() -> None:
    """
    Block DEMO_MODE in production.

    Raises:
        ProductionGateError: If TG_DEMO_MODE=true in production
    """
    demo_mode = os.getenv("TG_DEMO_MODE", "false").lower() == "true"

    if demo_mode and is_production():
        raise ProductionGateError(
            gate_name="DEMO_MODE",
            message="TG_DEMO_MODE=true is not allowed in production.",
            remediation="Set TG_DEMO_MODE=false or remove it entirely for production deployment."
        )
    elif demo_mode:
        logger.warning("[DEV MODE] DEMO_MODE is enabled - this would be blocked in production")


def assert_production_invariants() -> Dict[str, Any]:
    """
    Assert all production invariants at startup.

    This function should be called at application startup to verify
    that all required configuration and dependencies are present.

    Returns:
        Dict with validation results

    Raises:
        ProductionGateError: If any invariant fails in production
    """
    results = {
        "environment": os.getenv("TG_ENVIRONMENT", "development"),
        "is_production": is_production(),
        "gates_checked": [],
        "warnings": [],
    }

    if is_production():
        logger.info("=" * 60)
        logger.info("PRODUCTION MODE: Running startup gate checks...")
        logger.info("=" * 60)

    # Gate 1: Block demo mode
    try:
        block_demo_mode()
        results["gates_checked"].append("demo_mode_blocked")
    except ProductionGateError:
        raise

    # Gate 2: Require JWT secret key
    try:
        secret_key = require_env(
            "TG_SECRET_KEY",
            remediation="Generate a secure key: python -c \"import secrets; print(secrets.token_hex(32))\"",
            min_length=32,
        )
        if secret_key:
            results["gates_checked"].append("secret_key_set")
    except ProductionGateError:
        raise

    # Gate 3: Database URL (required for platform)
    try:
        db_url = require_env(
            "DATABASE_URL",
            remediation="Set DATABASE_URL to your PostgreSQL connection string.",
        )
        if db_url:
            results["gates_checked"].append("database_url_set")
    except ProductionGateError:
        raise

    # Gate 4: Check cryptography dependency
    try:
        crypto = require_dependency(
            "cryptography",
            remediation="pip install cryptography>=41.0"
        )
        if crypto:
            results["gates_checked"].append("cryptography_available")
    except ProductionGateError:
        raise

    # Gate 5: Optional but recommended - PQC support
    if os.getenv("TG_PQC_REQUIRED", "false").lower() == "true":
        try:
            require_dependency(
                "oqs",
                package_name="liboqs-python",
                remediation="pip install liboqs-python>=0.10"
            )
            results["gates_checked"].append("pqc_available")
        except ProductionGateError:
            raise

    # Gate 6: Key master (for encrypted key storage)
    try:
        key_master = require_env(
            "TG_KEY_MASTER",
            remediation="Generate a 32-byte hex key: python -c \"import os; print(os.urandom(32).hex())\"",
            min_length=64,  # 32 bytes as hex = 64 chars
        )
        if key_master:
            results["gates_checked"].append("key_master_set")
    except ProductionGateError:
        raise

    if is_production():
        logger.info(f"All {len(results['gates_checked'])} production gates passed.")
        logger.info("=" * 60)
    else:
        if results["warnings"]:
            logger.warning(f"Development mode: {len(results['warnings'])} production warnings (see above)")

    return results


def warn_incomplete_feature(
    feature_name: str,
    description: str,
    issue_url: Optional[str] = None,
) -> None:
    """
    Warn about an incomplete/simplified feature implementation.

    In production mode, logs a CRITICAL warning and may optionally
    require an explicit acknowledgment via environment variable.

    Args:
        feature_name: Name of the feature
        description: Description of what's incomplete
        issue_url: Optional link to tracking issue
    """
    ack_var = f"TG_ACK_{feature_name.upper().replace(' ', '_').replace('-', '_')}"
    acknowledged = os.getenv(ack_var, "false").lower() == "true"

    msg = f"INCOMPLETE FEATURE: {feature_name} - {description}"
    if issue_url:
        msg += f" See: {issue_url}"

    if is_production():
        if not acknowledged:
            logger.critical(
                f"{msg}\n"
                f"To acknowledge this limitation in production, set {ack_var}=true"
            )
        else:
            logger.warning(f"{msg} (acknowledged via {ack_var})")
    else:
        logger.info(f"[DEV MODE] {msg}")


def require_production_gate(
    gate_name: str,
    message: str,
    remediation: Optional[str] = None,
    allow_env_var: Optional[str] = None,
) -> None:
    """
    Require a production gate to pass, or raise ProductionGateError.

    This is a general-purpose gate that blocks an operation in production
    unless explicitly allowed via environment variable.

    Args:
        gate_name: Name of the gate for error messages
        message: Description of why this is blocked
        remediation: How to fix (defaults to generic message)
        allow_env_var: If set, check this env var to allow the operation

    Raises:
        ProductionGateError: If in production and not allowed
    """
    if not is_production():
        logger.warning(f"[DEV MODE] Gate {gate_name}: {message}")
        return

    # Check if explicitly allowed via env var
    if allow_env_var and os.getenv(allow_env_var, "").lower() == "true":
        logger.warning(f"[PRODUCTION] Gate {gate_name} bypassed via {allow_env_var}")
        return

    raise ProductionGateError(
        gate_name=gate_name,
        message=message,
        remediation=remediation or f"This operation is not allowed in production."
    )


def assert_feature_available(
    feature_name: str,
    required_modules: List[str],
    required_env_vars: Optional[List[str]] = None,
) -> bool:
    """
    Check if a feature is available based on dependencies and configuration.

    Args:
        feature_name: Name of the feature for error messages
        required_modules: List of required Python modules
        required_env_vars: Optional list of required environment variables

    Returns:
        True if feature is available, False otherwise (in dev)

    Raises:
        ProductionGateError: If feature unavailable in production and called
    """
    import importlib

    missing_modules = []
    missing_vars = []

    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            missing_modules.append(module)

    if required_env_vars:
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

    if missing_modules or missing_vars:
        msg_parts = []
        if missing_modules:
            msg_parts.append(f"Missing modules: {', '.join(missing_modules)}")
        if missing_vars:
            msg_parts.append(f"Missing env vars: {', '.join(missing_vars)}")

        if is_production():
            raise ProductionGateError(
                gate_name=f"FEATURE_{feature_name.upper().replace(' ', '_')}",
                message=f"Feature '{feature_name}' is not available. {'; '.join(msg_parts)}",
                remediation=f"Install required dependencies and set environment variables to enable {feature_name}."
            )
        else:
            logger.warning(f"[DEV MODE] Feature '{feature_name}' unavailable: {'; '.join(msg_parts)}")
            return False

    return True
