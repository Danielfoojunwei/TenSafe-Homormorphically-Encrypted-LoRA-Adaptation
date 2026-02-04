"""
Execution Policy and HE Enforcement Layer.

Provides fail-closed privacy modes to ensure HE is active when required.
Prevents silent fallback to plaintext computation.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    """Actual execution mode of a computation."""
    HE = "HE"  # Homomorphic Encryption active
    PLAINTEXT = "PLAINTEXT"  # Plaintext computation (no HE)
    SIMULATED = "SIMULATED"  # Simulated HE (for testing only)


class ExecutionPolicy(str, Enum):
    """
    Execution policy for privacy-sensitive operations.

    Determines how the system handles HE availability.
    """
    HE_REQUIRED = "he_required"  # Fail if HE not available (default for production)
    HE_PREFERRED = "he_preferred"  # Allow fallback but signal it clearly
    PLAINTEXT_ONLY = "plaintext_only"  # Explicitly allow plaintext (non-privacy path)


@dataclass
class ExecutionAttestation:
    """
    Attestation of how a request was executed.

    Included in every response to provide transparency about
    the execution mode used.
    """
    execution_mode: ExecutionMode
    he_backend: Optional[str] = None  # e.g., "tenseal-0.3.14", "he-lora-moai-1.0"
    he_backend_version: Optional[str] = None
    fallback_reason: Optional[str] = None  # Why HE wasn't used (if applicable)
    policy_applied: Optional[ExecutionPolicy] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "execution_mode": self.execution_mode.value,
        }
        if self.he_backend:
            result["he_backend"] = self.he_backend
        if self.he_backend_version:
            result["he_backend_version"] = self.he_backend_version
        if self.fallback_reason:
            result["fallback_reason"] = self.fallback_reason
        if self.policy_applied:
            result["policy_applied"] = self.policy_applied.value
        return result


class HENotAvailableError(Exception):
    """Raised when HE is required but not available."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


class ExecutionPolicyEnforcer:
    """
    Enforces execution policy for HE operations.

    This is the central point for checking HE availability
    and enforcing fail-closed behavior.
    """

    def __init__(self):
        self._he_available = False
        self._he_backend_name: Optional[str] = None
        self._he_backend_version: Optional[str] = None
        self._initialization_error: Optional[str] = None
        self._check_he_availability()

    def _check_he_availability(self) -> None:
        """Check if HE backend is available and working."""
        # Try to import and initialize HE backend
        try:
            # Check for TenSEAL (used for CKKS)
            import tenseal
            self._he_available = True
            self._he_backend_name = "tenseal"
            self._he_backend_version = tenseal.__version__
            logger.info(f"HE backend available: tenseal {tenseal.__version__}")
            return
        except ImportError:
            pass

        # Check for HE-LoRA MOAI backend
        try:
            from crypto_backend.ckks_moai import MOAIContext
            # Verify we can create a context
            ctx = MOAIContext()
            self._he_available = True
            self._he_backend_name = "ckks-moai"
            self._he_backend_version = "1.0.0"
            logger.info("HE backend available: ckks-moai")
            return
        except (ImportError, Exception) as e:
            self._initialization_error = str(e)

        # No HE backend available
        self._he_available = False
        logger.warning(
            "No HE backend available. Install tenseal or configure MOAI backend. "
            f"Last error: {self._initialization_error}"
        )

    @property
    def is_he_available(self) -> bool:
        """Check if HE backend is available."""
        return self._he_available

    @property
    def he_backend_info(self) -> Dict[str, str]:
        """Get HE backend information."""
        return {
            "name": self._he_backend_name or "none",
            "version": self._he_backend_version or "n/a",
            "available": self._he_available,
        }

    def get_default_policy(self) -> ExecutionPolicy:
        """
        Get default execution policy based on environment.

        Production defaults to HE_REQUIRED.
        Development defaults to HE_PREFERRED.
        """
        environment = os.getenv("TG_ENVIRONMENT", "development").lower()
        if environment in ("production", "prod"):
            return ExecutionPolicy.HE_REQUIRED
        return ExecutionPolicy.HE_PREFERRED

    def enforce(
        self,
        policy: Optional[ExecutionPolicy] = None,
        operation: str = "unknown",
    ) -> ExecutionAttestation:
        """
        Enforce execution policy and return attestation.

        Args:
            policy: The execution policy to enforce. If None, uses default.
            operation: Name of the operation being performed (for logging).

        Returns:
            ExecutionAttestation with the execution mode

        Raises:
            HENotAvailableError: If policy is HE_REQUIRED and HE is unavailable
        """
        if policy is None:
            policy = self.get_default_policy()

        # Check HE availability against policy
        if self._he_available:
            return ExecutionAttestation(
                execution_mode=ExecutionMode.HE,
                he_backend=self._he_backend_name,
                he_backend_version=self._he_backend_version,
                policy_applied=policy,
            )

        # HE not available - check policy
        if policy == ExecutionPolicy.HE_REQUIRED:
            reason = self._initialization_error or "HE backend not installed"
            logger.error(
                f"HE_REQUIRED policy violated for operation '{operation}': {reason}"
            )
            raise HENotAvailableError(
                f"HE is required but not available for operation '{operation}'. "
                f"Reason: {reason}. "
                "Install TenSEAL (pip install tenseal) or configure MOAI backend.",
                reason=reason,
            )

        if policy == ExecutionPolicy.HE_PREFERRED:
            reason = self._initialization_error or "HE backend not installed"
            logger.warning(
                f"HE not available for operation '{operation}', "
                f"falling back to plaintext. Reason: {reason}"
            )
            return ExecutionAttestation(
                execution_mode=ExecutionMode.PLAINTEXT,
                fallback_reason=reason,
                policy_applied=policy,
            )

        # PLAINTEXT_ONLY policy
        return ExecutionAttestation(
            execution_mode=ExecutionMode.PLAINTEXT,
            fallback_reason="Explicit plaintext policy",
            policy_applied=policy,
        )


# Global enforcer instance
_enforcer: Optional[ExecutionPolicyEnforcer] = None


def get_execution_enforcer() -> ExecutionPolicyEnforcer:
    """Get or create the global execution policy enforcer."""
    global _enforcer
    if _enforcer is None:
        _enforcer = ExecutionPolicyEnforcer()
    return _enforcer


def reset_enforcer() -> None:
    """Reset the global enforcer (for testing)."""
    global _enforcer
    _enforcer = None
