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

    # === Additional Security Gates ===

    # Network Security Gates
    INSECURE_SSL = FeatureGate(
        name="insecure_ssl",
        description="Allow connections without SSL verification",
        default_allowed=False,
        env_var="TENSAFE_INSECURE_SSL",
        production_allowed=False,
        requires_audit=True,
    )

    ALLOW_HTTP = FeatureGate(
        name="allow_http",
        description="Allow non-HTTPS connections",
        default_allowed=False,
        env_var="TENSAFE_ALLOW_HTTP",
        production_allowed=False,
        requires_audit=True,
    )

    # Crypto Security Gates
    WEAK_CRYPTO = FeatureGate(
        name="weak_crypto",
        description="Allow weak cryptographic parameters",
        default_allowed=False,
        env_var="TENSAFE_WEAK_CRYPTO",
        production_allowed=False,
        requires_audit=True,
    )

    SKIP_SIGNATURE_VERIFICATION = FeatureGate(
        name="skip_signature_verification",
        description="Skip artifact signature verification",
        default_allowed=False,
        env_var="TENSAFE_SKIP_SIG_VERIFY",
        production_allowed=False,
        requires_audit=True,
    )

    # Rate Limiting Gates
    RATE_LIMIT_BYPASS = FeatureGate(
        name="rate_limit_bypass",
        description="Bypass rate limiting",
        default_allowed=False,
        env_var="TENSAFE_NO_RATE_LIMIT",
        production_allowed=False,
        requires_audit=True,
    )

    # Audit Gates
    AUDIT_BYPASS = FeatureGate(
        name="audit_bypass",
        description="Bypass audit logging",
        default_allowed=False,
        env_var="TENSAFE_NO_AUDIT",
        production_allowed=False,
        requires_audit=True,
    )

    # Demo/Test Gates
    DEMO_MODE = FeatureGate(
        name="demo_mode",
        description="Enable demo mode with reduced security",
        default_allowed=False,
        env_var="TENSAFE_DEMO_MODE",
        production_allowed=False,
        requires_audit=True,
    )

    MOCK_SERVICES = FeatureGate(
        name="mock_services",
        description="Use mock services instead of real backends",
        default_allowed=False,
        env_var="TENSAFE_MOCK_SERVICES",
        production_allowed=False,
        requires_audit=True,
    )

    # Data Export Gates
    ALLOW_DATA_EXPORT = FeatureGate(
        name="allow_data_export",
        description="Allow exporting training data and artifacts",
        default_allowed=False,
        env_var="TENSAFE_ALLOW_EXPORT",
        production_allowed=True,
        requires_audit=True,
    )

    ALLOW_MODEL_DOWNLOAD = FeatureGate(
        name="allow_model_download",
        description="Allow downloading trained models",
        default_allowed=True,
        production_allowed=True,
        requires_audit=True,
    )

    # Resource Control Gates
    UNLIMITED_RESOURCES = FeatureGate(
        name="unlimited_resources",
        description="Bypass resource limits (memory, time, etc.)",
        default_allowed=False,
        env_var="TENSAFE_UNLIMITED_RESOURCES",
        production_allowed=False,
        requires_audit=True,
    )

    # TGSP Format Enforcement Gates
    TGSP_ENFORCEMENT = FeatureGate(
        name="tgsp_enforcement",
        description="Enforce TGSP format for encrypted inference (lock-in)",
        default_allowed=True,  # Enabled by default for security
        env_var="TENSAFE_TGSP_ENFORCEMENT",
        production_allowed=True,
        requires_audit=True,
    )

    TGSP_BYPASS = FeatureGate(
        name="tgsp_bypass",
        description="Bypass TGSP format requirement for encrypted inference (DANGEROUS)",
        default_allowed=False,  # Must be explicitly enabled
        env_var="TENSAFE_TGSP_BYPASS",
        production_allowed=False,  # NEVER allow in production
        requires_audit=True,
    )

    TGSP_SIGNATURE_SKIP = FeatureGate(
        name="tgsp_signature_skip",
        description="Skip TGSP signature verification (DANGEROUS)",
        default_allowed=False,
        env_var="TENSAFE_TGSP_SKIP_SIG",
        production_allowed=False,  # NEVER allow in production
        requires_audit=True,
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
        is_prod = _is_production_environment()

        # Check HE mode
        if hasattr(config, 'he'):
            from tensafe.core.config import HEMode
            if config.he.mode == HEMode.TOY:
                if is_prod:
                    errors.append("Toy HE mode not allowed in production")
                else:
                    warnings.append("Toy HE mode is not cryptographically secure")

            # Check HE security parameters
            if config.he.security_level < 128:
                if is_prod:
                    errors.append(f"HE security level {config.he.security_level} below 128-bit minimum")
                else:
                    warnings.append(f"HE security level {config.he.security_level} is weak")

        # Check DP
        if hasattr(config, 'dp'):
            if not config.dp.enabled and is_prod:
                if not ProductionGates.DP_BYPASS.is_allowed():
                    errors.append("DP must be enabled in production")

            if config.dp.enabled:
                # Check DP parameters
                if config.dp.noise_multiplier <= 0:
                    errors.append("DP noise_multiplier must be positive when DP is enabled")
                if config.dp.target_epsilon <= 0:
                    errors.append("DP target_epsilon must be positive")
                if config.dp.target_epsilon < 1.0:
                    warnings.append(f"Very tight epsilon={config.dp.target_epsilon} may severely impact utility")

        # Check model trust
        if hasattr(config, 'model'):
            if config.model.trust_remote_code:
                if not ProductionGates.REMOTE_CODE_EXECUTION.is_allowed():
                    errors.append("trust_remote_code not allowed without explicit gate")

        # Check API configuration
        if hasattr(config, 'base_url'):
            if config.base_url and config.base_url.startswith('http://'):
                if is_prod and not ProductionGates.ALLOW_HTTP.is_allowed():
                    errors.append("HTTP connections not allowed in production, use HTTPS")
                else:
                    warnings.append("Using HTTP instead of HTTPS")

        # Check training configuration
        if hasattr(config, 'training'):
            if config.training.total_steps < 1:
                errors.append("total_steps must be at least 1")
            if config.training.learning_rate <= 0:
                errors.append("learning_rate must be positive")
            if config.training.learning_rate > 0.1:
                warnings.append(f"Learning rate {config.training.learning_rate} is unusually high")

        # Check LoRA configuration
        if hasattr(config, 'lora') and config.lora.enabled:
            if config.lora.rank <= 0:
                errors.append("LoRA rank must be positive")
            if config.lora.rank > 256:
                warnings.append(f"LoRA rank {config.lora.rank} is unusually high")
            if not config.lora.target_modules:
                errors.append("LoRA enabled but no target_modules specified")

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

        # Security-critical environment variables that should never be set in production
        FORBIDDEN_IN_PROD = [
            ("TENSAFE_TOY_HE", "Toy HE cannot be enabled in production"),
            ("TENSAFE_DP_BYPASS", "DP bypass cannot be enabled in production"),
            ("TENSAFE_UNSAFE_DESER", "Unsafe deserialization cannot be enabled in production"),
            ("TENSAFE_NO_RATE_LIMIT", "Rate limiting cannot be disabled in production"),
            ("TENSAFE_NO_AUDIT", "Audit logging cannot be disabled in production"),
            ("TENSAFE_DEMO_MODE", "Demo mode cannot be enabled in production"),
            ("TENSAFE_MOCK_SERVICES", "Mock services cannot be used in production"),
            ("TENSAFE_UNLIMITED_RESOURCES", "Unlimited resources cannot be enabled in production"),
            ("TENSAFE_INSECURE_SSL", "Insecure SSL cannot be enabled in production"),
            ("TENSAFE_ALLOW_HTTP", "HTTP cannot be allowed in production"),
            ("TENSAFE_WEAK_CRYPTO", "Weak crypto cannot be enabled in production"),
            ("TENSAFE_TGSP_BYPASS", "TGSP format bypass cannot be enabled in production"),
            ("TENSAFE_TGSP_SKIP_SIG", "TGSP signature skip cannot be enabled in production"),
        ]

        # Warning-level environment variables in production
        WARN_IN_PROD = [
            ("TENSAFE_DEBUG", "Debug mode enabled in production"),
            ("TENSAFE_PROFILING", "Profiling enabled in production"),
            ("TENSAFE_EXPERIMENTAL", "Experimental features enabled in production"),
            ("TENSAFE_TRUST_REMOTE_CODE", "Remote code execution enabled"),
        ]

        # Check forbidden variables
        for env_var, message in FORBIDDEN_IN_PROD:
            if is_prod and os.environ.get(env_var):
                errors.append(message)

        # Check warning-level variables
        for env_var, message in WARN_IN_PROD:
            if is_prod and os.environ.get(env_var):
                warnings.append(message)

        # Check SSL verification
        if os.environ.get("TENSAFE_VERIFY_SSL", "").lower() == "false":
            if is_prod:
                errors.append("SSL verification cannot be disabled in production")
            else:
                warnings.append("SSL verification is disabled")

        # Check for common misconfigurations
        if not os.environ.get("TENSAFE_ENV") and is_prod:
            warnings.append("TENSAFE_ENV not explicitly set, defaulting to production")

        # Check for secrets in environment (should use secret manager)
        sensitive_env_patterns = ["_KEY", "_SECRET", "_PASSWORD", "_TOKEN"]
        for key in os.environ:
            if key.startswith("TENSAFE_") and any(p in key for p in sensitive_env_patterns):
                if is_prod:
                    warnings.append(f"Sensitive value in environment variable: {key} (consider using secret manager)")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def validate_security() -> ValidationResult:
        """
        Perform security-focused validation checks.

        Returns:
            ValidationResult with security findings
        """
        errors = []
        warnings = []
        is_prod = _is_production_environment()

        # Check if running as root (not recommended)
        try:
            if os.getuid() == 0:
                warnings.append("Running as root user is not recommended")
        except AttributeError:
            pass  # Windows doesn't have getuid

        # Check file permissions on sensitive directories
        sensitive_paths = [
            "/tmp/tg_tinker_artifacts",
            "keys/",
            ".env",
        ]
        for path in sensitive_paths:
            if os.path.exists(path):
                try:
                    mode = os.stat(path).st_mode
                    # Check if world-readable
                    if mode & 0o004:
                        warnings.append(f"Sensitive path {path} is world-readable")
                except OSError:
                    pass

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


def production_check(
    config: Optional[Any] = None,
    include_security: bool = True,
) -> ValidationResult:
    """
    Perform comprehensive production readiness check.

    Args:
        config: Optional TenSafeConfig to validate
        include_security: Include security-focused checks

    Returns:
        Combined ValidationResult
    """
    results = []

    # Environment check
    results.append(ProductionValidator.validate_environment())

    # Security check
    if include_security:
        results.append(ProductionValidator.validate_security())

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


def security_audit() -> Dict[str, Any]:
    """
    Perform a comprehensive security audit.

    Returns:
        Dict with audit results including:
        - environment: Environment validation results
        - security: Security validation results
        - gates: Status of all security-related gates
        - recommendations: List of security recommendations
    """
    env_result = ProductionValidator.validate_environment()
    sec_result = ProductionValidator.validate_security()

    # Check all security-related gates
    security_gates = [
        ProductionGates.TOY_HE,
        ProductionGates.DP_BYPASS,
        ProductionGates.UNSAFE_DESERIALIZATION,
        ProductionGates.REMOTE_CODE_EXECUTION,
        ProductionGates.INSECURE_SSL,
        ProductionGates.ALLOW_HTTP,
        ProductionGates.WEAK_CRYPTO,
        ProductionGates.RATE_LIMIT_BYPASS,
        ProductionGates.AUDIT_BYPASS,
        ProductionGates.DEMO_MODE,
        ProductionGates.TGSP_ENFORCEMENT,
        ProductionGates.TGSP_BYPASS,
        ProductionGates.TGSP_SIGNATURE_SKIP,
    ]

    gate_status = {}
    for gate in security_gates:
        status = gate.check()
        gate_status[gate.name] = {
            "status": status.value,
            "production_allowed": gate.production_allowed,
            "requires_audit": gate.requires_audit,
        }

    # Generate recommendations
    recommendations = []
    is_prod = _is_production_environment()

    if is_prod:
        recommendations.append("Ensure all sensitive data is encrypted at rest")
        recommendations.append("Use a secrets manager for API keys and credentials")
        recommendations.append("Enable audit logging for compliance")
        recommendations.append("Implement regular key rotation")
        recommendations.append("Configure network security groups/firewalls")

    if env_result.warnings or sec_result.warnings:
        recommendations.append("Review and address all warnings before production deployment")

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "is_production": is_prod,
        "environment": {
            "valid": env_result.valid,
            "errors": env_result.errors,
            "warnings": env_result.warnings,
        },
        "security": {
            "valid": sec_result.valid,
            "errors": sec_result.errors,
            "warnings": sec_result.warnings,
        },
        "gates": gate_status,
        "recommendations": recommendations,
        "overall_valid": env_result.valid and sec_result.valid,
    }
