"""
TenSafe Production Gating and Feature Flags.

This module provides production hardening through:
- Feature gates that control access to features
- Environment-based gating for staged rollouts
- Validation gates for production safety
- Audit trail for gate checks

Feature gates help ensure:
1. New features can be gradually rolled out
2. Dangerous operations require explicit opt-in
3. Production vs development environments are distinguished
4. Compliance requirements are enforced

Usage:
    from tensafe.core.gates import ProductionGates, require_gate, GateStatus

    # Check a gate
    if ProductionGates.HE_ENABLED.is_allowed():
        # Use HE features
        pass

    # Require a gate (raises if not allowed)
    @require_gate(ProductionGates.DP_BYPASS)
    def train_without_dp():
        pass

    # Check gate status
    status = ProductionGates.TOY_HE.check()
    if status == GateStatus.DENIED:
        raise RuntimeError("Toy HE not allowed in production")
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class GateStatus(Enum):
    """Status of a feature gate check."""
    ALLOWED = "allowed"  # Gate passed, feature allowed
    DENIED = "denied"  # Gate blocked, feature denied
    WARN = "warn"  # Gate passed with warning
    AUDIT = "audit"  # Gate passed but logged for audit


class GateDeniedError(Exception):
    """Raised when a feature gate check fails."""

    def __init__(self, gate: "FeatureGate", reason: str):
        self.gate = gate
        self.reason = reason
        super().__init__(f"Gate '{gate.name}' denied: {reason}")


@dataclass
class GateCheckResult:
    """Result of a gate check."""
    gate_name: str
    status: GateStatus
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to audit log format."""
        return {
            "gate": self.gate_name,
            "status": self.status.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
        }


class FeatureGate:
    """
    A feature gate that controls access to a feature.

    Gates can be controlled by:
    - Environment variables (TENSAFE_GATE_<NAME>=1)
    - Explicit allow/deny lists
    - Custom validation functions
    - Environment detection (production vs dev)

    Example:
        gate = FeatureGate(
            name="toy_he",
            description="Allow toy/simulation HE mode",
            default_allowed=False,
            env_var="TENSAFE_TOY_HE",
            production_allowed=False,
        )

        if gate.is_allowed():
            # Use toy HE
            pass
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        default_allowed: bool = False,
        env_var: Optional[str] = None,
        production_allowed: bool = True,
        requires_audit: bool = False,
        validator: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize a feature gate.

        Args:
            name: Gate identifier
            description: Human-readable description
            default_allowed: Default state if no override
            env_var: Environment variable to check (1/true = allow)
            production_allowed: Whether gate can be opened in production
            requires_audit: Log all gate checks for audit trail
            validator: Custom validation function
        """
        self.name = name
        self.description = description
        self.default_allowed = default_allowed
        self.env_var = env_var or f"TENSAFE_GATE_{name.upper()}"
        self.production_allowed = production_allowed
        self.requires_audit = requires_audit
        self.validator = validator

        # Tracking
        self._check_count = 0
        self._allow_count = 0
        self._deny_count = 0
        self._last_check: Optional[GateCheckResult] = None
        self._lock = threading.Lock()

    def check(self, context: Optional[Dict[str, Any]] = None) -> GateStatus:
        """
        Check the gate status.

        Args:
            context: Optional context for audit logging

        Returns:
            GateStatus indicating if feature is allowed
        """
        context = context or {}
        reason = ""

        with self._lock:
            self._check_count += 1

            # Check if in production
            is_production = _is_production_environment()

            # Check environment variable
            env_value = os.environ.get(self.env_var, "").lower()
            env_enabled = env_value in ("1", "true", "yes", "enabled")
            env_disabled = env_value in ("0", "false", "no", "disabled")

            # Determine status
            if env_disabled:
                status = GateStatus.DENIED
                reason = f"Explicitly disabled via {self.env_var}"
            elif is_production and not self.production_allowed:
                status = GateStatus.DENIED
                reason = "Not allowed in production environment"
            elif env_enabled:
                if is_production and not self.production_allowed:
                    status = GateStatus.DENIED
                    reason = f"Cannot enable {self.env_var} in production"
                else:
                    status = GateStatus.ALLOWED if not self.requires_audit else GateStatus.AUDIT
                    reason = f"Enabled via {self.env_var}"
            elif self.validator is not None:
                try:
                    if self.validator():
                        status = GateStatus.ALLOWED if not self.requires_audit else GateStatus.AUDIT
                        reason = "Validator approved"
                    else:
                        status = GateStatus.DENIED
                        reason = "Validator rejected"
                except Exception as e:
                    status = GateStatus.DENIED
                    reason = f"Validator error: {e}"
            elif self.default_allowed:
                status = GateStatus.ALLOWED if not self.requires_audit else GateStatus.AUDIT
                reason = "Default allowed"
            else:
                status = GateStatus.DENIED
                reason = "Default denied"

            # Update counters
            if status in (GateStatus.ALLOWED, GateStatus.AUDIT, GateStatus.WARN):
                self._allow_count += 1
            else:
                self._deny_count += 1

            # Create result
            result = GateCheckResult(
                gate_name=self.name,
                status=status,
                reason=reason,
                context={
                    **context,
                    "is_production": is_production,
                    "env_var": self.env_var,
                    "env_value": env_value or None,
                },
            )
            self._last_check = result

            # Log if audit required or denied
            if self.requires_audit or status == GateStatus.DENIED:
                self._log_check(result)

            return status

    def is_allowed(self, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if the gate allows the feature.

        Args:
            context: Optional context for logging

        Returns:
            True if feature is allowed
        """
        status = self.check(context)
        return status in (GateStatus.ALLOWED, GateStatus.AUDIT, GateStatus.WARN)

    def require(self, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Require the gate to pass, raising if denied.

        Args:
            context: Optional context for logging

        Raises:
            GateDeniedError: If gate check fails
        """
        status = self.check(context)
        if status == GateStatus.DENIED:
            raise GateDeniedError(self, self._last_check.reason if self._last_check else "Denied")

    def _log_check(self, result: GateCheckResult) -> None:
        """Log a gate check for audit."""
        if result.status == GateStatus.DENIED:
            logger.warning(
                f"Gate '{self.name}' DENIED: {result.reason}",
                extra={"gate_check": result.to_audit_dict()},
            )
        elif result.status == GateStatus.AUDIT:
            logger.info(
                f"Gate '{self.name}' AUDIT: {result.reason}",
                extra={"gate_check": result.to_audit_dict()},
            )
        elif result.status == GateStatus.WARN:
            logger.warning(
                f"Gate '{self.name}' WARN: {result.reason}",
                extra={"gate_check": result.to_audit_dict()},
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get gate check statistics."""
        with self._lock:
            return {
                "name": self.name,
                "check_count": self._check_count,
                "allow_count": self._allow_count,
                "deny_count": self._deny_count,
                "last_check": self._last_check.to_audit_dict() if self._last_check else None,
            }


class ProductionGates:
    """
    Collection of production gates for TenSafe.

    These gates control access to sensitive features and ensure
    production safety.
    """

    # HE Gates
    HE_ENABLED = FeatureGate(
        name="he_enabled",
        description="Enable homomorphic encryption features",
        default_allowed=True,
        production_allowed=True,
    )

    TOY_HE = FeatureGate(
        name="toy_he",
        description="Allow toy/simulation HE mode (NOT SECURE)",
        default_allowed=False,
        env_var="TENSAFE_TOY_HE",
        production_allowed=False,  # Never allow in production
        requires_audit=True,
    )

    # DP Gates
    DP_ENABLED = FeatureGate(
        name="dp_enabled",
        description="Enable differential privacy",
        default_allowed=True,
        production_allowed=True,
    )

    DP_BYPASS = FeatureGate(
        name="dp_bypass",
        description="Allow training without differential privacy",
        default_allowed=False,
        env_var="TENSAFE_DP_BYPASS",
        production_allowed=False,  # Must have DP in production
        requires_audit=True,
    )

    # Training Gates
    RLVR_ENABLED = FeatureGate(
        name="rlvr_enabled",
        description="Enable RLVR training mode",
        default_allowed=True,
        production_allowed=True,
    )

    EXPERIMENTAL_FEATURES = FeatureGate(
        name="experimental",
        description="Enable experimental features",
        default_allowed=False,
        env_var="TENSAFE_EXPERIMENTAL",
        production_allowed=False,
        requires_audit=True,
    )

    # Security Gates
    UNSAFE_DESERIALIZATION = FeatureGate(
        name="unsafe_deserialization",
        description="Allow unsafe pickle/torch.load operations",
        default_allowed=False,
        env_var="TENSAFE_UNSAFE_DESER",
        production_allowed=False,
        requires_audit=True,
    )

    REMOTE_CODE_EXECUTION = FeatureGate(
        name="remote_code",
        description="Allow trust_remote_code for models",
        default_allowed=False,
        env_var="TENSAFE_TRUST_REMOTE_CODE",
        production_allowed=False,
        requires_audit=True,
    )

    # Debug Gates
    DEBUG_MODE = FeatureGate(
        name="debug_mode",
        description="Enable debug mode with extra logging",
        default_allowed=False,
        env_var="TENSAFE_DEBUG",
        production_allowed=True,  # Allow but audit
        requires_audit=True,
    )

    PROFILING = FeatureGate(
        name="profiling",
        description="Enable performance profiling",
        default_allowed=False,
        env_var="TENSAFE_PROFILING",
        production_allowed=True,
        requires_audit=True,
    )

    # Model Gates
    LARGE_MODEL = FeatureGate(
        name="large_model",
        description="Allow models > 13B parameters",
        default_allowed=True,
        production_allowed=True,
    )

    # All gates for iteration
    _ALL_GATES: List[FeatureGate] = []

    @classmethod
    def get_all_gates(cls) -> List[FeatureGate]:
        """Get all defined gates."""
        if not cls._ALL_GATES:
            cls._ALL_GATES = [
                value for name, value in vars(cls).items()
                if isinstance(value, FeatureGate)
            ]
        return cls._ALL_GATES

    @classmethod
    def get_gate_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all gates."""
        return {
            gate.name: gate.get_stats()
            for gate in cls.get_all_gates()
        }

    @classmethod
    def check_all(cls) -> Dict[str, GateStatus]:
        """Check status of all gates."""
        return {
            gate.name: gate.check()
            for gate in cls.get_all_gates()
        }


def require_gate(gate: FeatureGate, context: Optional[Dict[str, Any]] = None) -> Callable[[F], F]:
    """
    Decorator that requires a gate to pass before executing function.

    Args:
        gate: The gate to check
        context: Optional context for audit logging

    Returns:
        Decorated function

    Example:
        @require_gate(ProductionGates.EXPERIMENTAL_FEATURES)
        def new_experimental_feature():
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            gate.require(context={
                **(context or {}),
                "function": func.__name__,
                "module": func.__module__,
            })
            return func(*args, **kwargs)
        return wrapper  # type: ignore
    return decorator


def check_gate(gate: FeatureGate, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if a gate allows a feature.

    Args:
        gate: The gate to check
        context: Optional context for logging

    Returns:
        True if feature is allowed
    """
    return gate.is_allowed(context)


def _is_production_environment() -> bool:
    """
    Detect if running in production environment.

    Checks:
    - TENSAFE_ENV or ENV environment variable
    - Common production indicators
    """
    env = os.environ.get("TENSAFE_ENV") or os.environ.get("ENV") or ""
    env = env.lower()

    # Explicit production
    if env in ("production", "prod"):
        return True

    # Explicit non-production
    if env in ("development", "dev", "test", "testing", "local", "ci"):
        return False

    # Check other indicators
    if os.environ.get("CI"):
        return False  # CI environment
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True  # K8s production
    if os.environ.get("AWS_EXECUTION_ENV"):
        return True  # AWS Lambda/ECS

    # Default to production (safe default)
    return True


@dataclass
class ValidationResult:
    """Result of a validation check."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


class ProductionValidator:
    """
    Validator for production readiness checks.

    Ensures configuration and environment are suitable for production.
    """

    @staticmethod
    def validate_config(config: Any) -> ValidationResult:
        """
        Validate configuration for production use.

        Args:
            config: TenSafeConfig to validate

        Returns:
            ValidationResult with any errors/warnings
        """
        errors = []
        warnings = []

        # Check HE mode
        if hasattr(config, 'he'):
            from tensafe.core.config import HEMode
            if config.he.mode == HEMode.TOY:
                if _is_production_environment():
                    errors.append("Toy HE mode not allowed in production")
                else:
                    warnings.append("Toy HE mode is not cryptographically secure")

        # Check DP
        if hasattr(config, 'dp'):
            if not config.dp.enabled and _is_production_environment():
                if not ProductionGates.DP_BYPASS.is_allowed():
                    errors.append("DP must be enabled in production")

        # Check model trust
        if hasattr(config, 'model'):
            if config.model.trust_remote_code:
                if not ProductionGates.REMOTE_CODE_EXECUTION.is_allowed():
                    errors.append("trust_remote_code not allowed without explicit gate")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def validate_environment() -> ValidationResult:
        """
        Validate environment for production use.

        Returns:
            ValidationResult with any errors/warnings
        """
        errors = []
        warnings = []

        is_prod = _is_production_environment()

        # Check for debug mode in production
        if is_prod and os.environ.get("TENSAFE_DEBUG"):
            warnings.append("Debug mode enabled in production")

        # Check for toy HE in production
        if is_prod and os.environ.get("TENSAFE_TOY_HE"):
            errors.append("Toy HE cannot be enabled in production")

        # Check for DP bypass in production
        if is_prod and os.environ.get("TENSAFE_DP_BYPASS"):
            errors.append("DP bypass cannot be enabled in production")

        # Check SSL verification
        if os.environ.get("TENSAFE_VERIFY_SSL", "").lower() == "false":
            if is_prod:
                errors.append("SSL verification cannot be disabled in production")
            else:
                warnings.append("SSL verification is disabled")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def production_check(config: Optional[Any] = None) -> ValidationResult:
    """
    Perform comprehensive production readiness check.

    Args:
        config: Optional TenSafeConfig to validate

    Returns:
        Combined ValidationResult
    """
    results = []

    # Environment check
    results.append(ProductionValidator.validate_environment())

    # Config check if provided
    if config is not None:
        results.append(ProductionValidator.validate_config(config))

    # Combine results
    all_errors = []
    all_warnings = []
    for r in results:
        all_errors.extend(r.errors)
        all_warnings.extend(r.warnings)

    return ValidationResult(
        valid=len(all_errors) == 0,
        errors=all_errors,
        warnings=all_warnings,
    )
