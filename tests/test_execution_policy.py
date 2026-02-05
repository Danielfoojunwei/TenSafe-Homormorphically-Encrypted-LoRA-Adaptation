"""
Execution Policy Tests

Verifies that:
1. Fail-closed behavior works when HE is required
2. Attestation is properly included in responses
3. Policy enforcement matches environment settings
"""

import os
import sys

import pytest

# Import directly to avoid tensorguard's crypto imports
sys.path.insert(0, "src")


def import_execution_policy_module():
    """Import execution policy module avoiding crypto conflicts."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "execution_policy",
        "src/tensorguard/platform/tg_tinker_api/execution_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["execution_policy"] = module
    spec.loader.exec_module(module)
    return module


policy = import_execution_policy_module()


class TestExecutionPolicy:
    """Test ExecutionPolicy enum and enforcement."""

    def test_execution_modes_defined(self):
        """Test that all execution modes are defined."""
        assert policy.ExecutionMode.HE.value == "HE"
        assert policy.ExecutionMode.PLAINTEXT.value == "PLAINTEXT"
        assert policy.ExecutionMode.SIMULATED.value == "SIMULATED"

    def test_execution_policies_defined(self):
        """Test that all execution policies are defined."""
        assert policy.ExecutionPolicy.HE_REQUIRED.value == "he_required"
        assert policy.ExecutionPolicy.HE_PREFERRED.value == "he_preferred"
        assert policy.ExecutionPolicy.PLAINTEXT_ONLY.value == "plaintext_only"


class TestExecutionAttestation:
    """Test ExecutionAttestation data class."""

    def test_attestation_to_dict(self):
        """Test attestation serialization."""
        attestation = policy.ExecutionAttestation(
            execution_mode=policy.ExecutionMode.HE,
            he_backend="tenseal",
            he_backend_version="0.3.14",
            policy_applied=policy.ExecutionPolicy.HE_REQUIRED,
        )

        result = attestation.to_dict()
        assert result["execution_mode"] == "HE"
        assert result["he_backend"] == "tenseal"
        assert result["he_backend_version"] == "0.3.14"
        assert result["policy_applied"] == "he_required"
        assert "fallback_reason" not in result

    def test_attestation_with_fallback(self):
        """Test attestation with fallback reason."""
        attestation = policy.ExecutionAttestation(
            execution_mode=policy.ExecutionMode.PLAINTEXT,
            fallback_reason="HE backend not installed",
            policy_applied=policy.ExecutionPolicy.HE_PREFERRED,
        )

        result = attestation.to_dict()
        assert result["execution_mode"] == "PLAINTEXT"
        assert result["fallback_reason"] == "HE backend not installed"
        assert result["policy_applied"] == "he_preferred"


class TestExecutionPolicyEnforcer:
    """Test ExecutionPolicyEnforcer class."""

    def test_default_policy_development(self):
        """Test default policy in development mode."""
        policy.reset_enforcer()
        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "development"
            enforcer = policy.ExecutionPolicyEnforcer()
            assert enforcer.get_default_policy() == policy.ExecutionPolicy.HE_PREFERRED
        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]
            policy.reset_enforcer()

    def test_default_policy_production(self):
        """Test default policy in production mode."""
        policy.reset_enforcer()
        old_env = os.environ.get("TG_ENVIRONMENT")
        try:
            os.environ["TG_ENVIRONMENT"] = "production"
            enforcer = policy.ExecutionPolicyEnforcer()
            assert enforcer.get_default_policy() == policy.ExecutionPolicy.HE_REQUIRED
        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]
            policy.reset_enforcer()

    def test_enforce_he_required_without_he(self):
        """Test that HE_REQUIRED fails when HE is unavailable."""
        policy.reset_enforcer()
        enforcer = policy.ExecutionPolicyEnforcer()

        # If HE is not available, this should raise
        if not enforcer.is_he_available:
            with pytest.raises(policy.HENotAvailableError) as exc_info:
                enforcer.enforce(policy.ExecutionPolicy.HE_REQUIRED, "test_operation")

            assert "HE is required but not available" in str(exc_info.value)
            assert exc_info.value.reason is not None

    def test_enforce_he_preferred_without_he(self):
        """Test that HE_PREFERRED allows fallback with attestation."""
        policy.reset_enforcer()
        enforcer = policy.ExecutionPolicyEnforcer()

        # If HE is not available, should return plaintext attestation
        if not enforcer.is_he_available:
            attestation = enforcer.enforce(policy.ExecutionPolicy.HE_PREFERRED, "test_operation")
            assert attestation.execution_mode == policy.ExecutionMode.PLAINTEXT
            assert attestation.fallback_reason is not None
            assert attestation.policy_applied == policy.ExecutionPolicy.HE_PREFERRED

    def test_enforce_plaintext_only(self):
        """Test that PLAINTEXT_ONLY returns plaintext attestation."""
        policy.reset_enforcer()
        enforcer = policy.ExecutionPolicyEnforcer()

        attestation = enforcer.enforce(policy.ExecutionPolicy.PLAINTEXT_ONLY, "test_operation")
        # PLAINTEXT_ONLY should always return plaintext mode
        # (even if HE is available, it's explicitly requesting plaintext)
        if not enforcer.is_he_available:
            assert attestation.execution_mode == policy.ExecutionMode.PLAINTEXT
            assert attestation.policy_applied == policy.ExecutionPolicy.PLAINTEXT_ONLY

    def test_he_backend_info(self):
        """Test HE backend information retrieval."""
        policy.reset_enforcer()
        enforcer = policy.ExecutionPolicyEnforcer()

        info = enforcer.he_backend_info
        assert "name" in info
        assert "version" in info
        assert "available" in info
        assert isinstance(info["available"], bool)


class TestHENotAvailableError:
    """Test HENotAvailableError exception."""

    def test_error_attributes(self):
        """Test error attributes are preserved."""
        error = policy.HENotAvailableError("Test message", reason="test reason")
        assert str(error) == "Test message"
        assert error.reason == "test reason"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
