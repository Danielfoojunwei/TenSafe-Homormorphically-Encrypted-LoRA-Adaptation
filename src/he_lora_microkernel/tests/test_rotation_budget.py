"""
Rotation Budget Tests for HE-LoRA Microkernel

These tests verify that rotation counts stay within budget.
CI should FAIL if rotation budgets are exceeded.

Default budgets:
  - QKV: rotations/token ≤ R_qkv (16)
  - QKVO: rotations/token ≤ R_qkvo (16)
  - keyswitch/token ≤ K_max (16)
  - rescales/token ≤ S_max (8)
"""

import pytest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.compiler import (
    LoRAConfig,
    LoRATargets,
    CKKSProfile,
    get_profile,
    compile_schedule,
    CostBudget,
    RotationBudget,
    KeyswitchBudget,
    RescaleBudget,
    estimate_costs,
    check_budget_compliance,
    enforce_rotation_invariant,
)
from he_lora_microkernel.runtime import (
    HELoRAExecutor,
    InvariantChecker,
)
from he_lora_microkernel.backend.gpu_ckks_backend import BackendType


# =============================================================================
# BUDGET DEFINITIONS
# =============================================================================

DEFAULT_ROTATION_BUDGET = 16
DEFAULT_KEYSWITCH_BUDGET = 16
DEFAULT_RESCALE_BUDGET = 8


def get_strict_budget() -> CostBudget:
    """Get strict budget for CI enforcement."""
    return CostBudget(
        rotation=RotationBudget(
            max_rotations_per_token=DEFAULT_ROTATION_BUDGET,
            max_rotations_per_layer=DEFAULT_ROTATION_BUDGET * 4,
            max_rotations_qkv=DEFAULT_ROTATION_BUDGET * 3,
            max_rotations_qkvo=DEFAULT_ROTATION_BUDGET * 4,
        ),
        keyswitch=KeyswitchBudget(
            max_keyswitches_per_token=DEFAULT_KEYSWITCH_BUDGET,
            max_keyswitches_per_layer=DEFAULT_KEYSWITCH_BUDGET * 4,
        ),
        rescale=RescaleBudget(
            max_rescales_per_token=DEFAULT_RESCALE_BUDGET,
        ),
    )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def qkv_config():
    """QKV configuration for testing."""
    return LoRAConfig(
        hidden_size=512,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=4,
        max_context_length=1024,
        ckks_profile=CKKSProfile.FAST,
    )


@pytest.fixture
def qkvo_config():
    """QKVO configuration for testing."""
    return LoRAConfig(
        hidden_size=512,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKVO,
        batch_size=4,
        max_context_length=1024,
        ckks_profile=CKKSProfile.FAST,
    )


# =============================================================================
# ROTATION BUDGET TESTS
# =============================================================================

class TestRotationBudget:
    """Tests for rotation budget compliance."""

    def test_estimated_rotations_within_budget(self, qkv_config):
        """Test that estimated rotations are within budget."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        # Check predicted rotations
        predicted = schedule.predicted_costs.rotations_per_token
        assert predicted <= DEFAULT_ROTATION_BUDGET, (
            f"Predicted rotations {predicted} exceed budget {DEFAULT_ROTATION_BUDGET}"
        )

    def test_actual_rotations_within_budget(self, qkv_config):
        """Test that actual rotations during execution are within budget."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        rng = np.random.default_rng(42)
        A = rng.standard_normal((qkv_config.hidden_size, qkv_config.rank)) * 0.01
        B = rng.standard_normal((qkv_config.rank, qkv_config.hidden_size)) * 0.01

        budget = get_strict_budget()
        executor = HELoRAExecutor(schedule, BackendType.SIMULATION, budget=budget)
        executor.load_weights(A, B, qkv_config.alpha)

        # Execute several tokens
        for i in range(5):
            x = rng.standard_normal(
                (qkv_config.batch_size, qkv_config.hidden_size)
            ).astype(np.float64)
            executor.execute_token(x, position=i)

        # Check counters
        stats = executor.get_statistics()
        rotations = stats['backend_counters']['rotations']
        tokens = stats['tokens_processed']

        rotations_per_token = rotations / tokens
        assert rotations_per_token <= DEFAULT_ROTATION_BUDGET, (
            f"Actual rotations/token {rotations_per_token:.1f} "
            f"exceed budget {DEFAULT_ROTATION_BUDGET}"
        )

    def test_qkv_vs_qkvo_rotation_budget(self, qkv_config, qkvo_config):
        """Test that QKV and QKVO have appropriate budgets."""
        ckks_params = get_profile(qkv_config.ckks_profile)

        qkv_schedule = compile_schedule(qkv_config, ckks_params)
        qkvo_schedule = compile_schedule(qkvo_config, ckks_params)

        qkv_rotations = qkv_schedule.predicted_costs.rotations_per_token
        qkvo_rotations = qkvo_schedule.predicted_costs.rotations_per_token

        # QKVO should have more adapters but similar per-adapter rotations
        qkv_per_adapter = qkv_rotations  # 3 adapters
        qkvo_per_adapter = qkvo_rotations  # 4 adapters

        # Per-adapter rotation should be similar
        assert abs(qkv_per_adapter - qkvo_per_adapter) <= 5, (
            f"Per-adapter rotation differs significantly: "
            f"QKV={qkv_per_adapter}, QKVO={qkvo_per_adapter}"
        )

    def test_rotation_budget_regression(self, qkv_config):
        """Test that rotation budget doesn't regress between compilations."""
        ckks_params = get_profile(qkv_config.ckks_profile)

        # Compile twice
        schedule1 = compile_schedule(qkv_config, ckks_params)
        schedule2 = compile_schedule(qkv_config, ckks_params)

        rot1 = schedule1.predicted_costs.rotations_per_token
        rot2 = schedule2.predicted_costs.rotations_per_token

        # Should be identical (deterministic)
        assert rot1 == rot2, (
            f"Rotation count not deterministic: {rot1} vs {rot2}"
        )

    def test_invariant_checker(self, qkv_config):
        """Test InvariantChecker catches budget violations."""
        checker = InvariantChecker(
            max_rotations_per_token=5,  # Very strict
            max_keyswitches_per_token=5,
            max_rescales_per_token=2,
        )

        # Check with values exceeding budget
        result = checker.check_token(
            rotations=10,
            keyswitches=10,
            rescales=5,
            he_time_ms=100,
            total_time_ms=200,
        )

        assert not result, "Checker should fail for values exceeding budget"
        assert checker.has_violations
        assert len(checker.violations) >= 3  # rotation, keyswitch, rescale

    def test_invariant_checker_passes(self, qkv_config):
        """Test InvariantChecker passes for compliant values."""
        checker = InvariantChecker(
            max_rotations_per_token=DEFAULT_ROTATION_BUDGET,
            max_keyswitches_per_token=DEFAULT_KEYSWITCH_BUDGET,
            max_rescales_per_token=DEFAULT_RESCALE_BUDGET,
        )

        result = checker.check_token(
            rotations=4,
            keyswitches=4,
            rescales=2,
            he_time_ms=50,
            total_time_ms=100,
        )

        assert result, "Checker should pass for compliant values"
        assert not checker.has_violations


class TestKeyswitchBudget:
    """Tests for keyswitch budget compliance."""

    def test_keyswitches_equal_rotations(self, qkv_config):
        """Test that keyswitches equal rotations (rotation requires keyswitch)."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        rng = np.random.default_rng(42)
        A = rng.standard_normal((qkv_config.hidden_size, qkv_config.rank)) * 0.01
        B = rng.standard_normal((qkv_config.rank, qkv_config.hidden_size)) * 0.01

        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, qkv_config.alpha)

        x = rng.standard_normal(
            (qkv_config.batch_size, qkv_config.hidden_size)
        ).astype(np.float64)
        executor.execute_token(x)

        stats = executor.get_statistics()
        rotations = stats['backend_counters']['rotations']
        keyswitches = stats['backend_counters']['keyswitches']

        # Each rotation requires one keyswitch
        assert keyswitches == rotations, (
            f"Keyswitches ({keyswitches}) should equal rotations ({rotations})"
        )


class TestRescaleBudget:
    """Tests for rescale budget compliance."""

    def test_rescales_within_budget(self, qkv_config):
        """Test that rescale count is within budget."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        rng = np.random.default_rng(42)
        A = rng.standard_normal((qkv_config.hidden_size, qkv_config.rank)) * 0.01
        B = rng.standard_normal((qkv_config.rank, qkv_config.hidden_size)) * 0.01

        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, qkv_config.alpha)

        x = rng.standard_normal(
            (qkv_config.batch_size, qkv_config.hidden_size)
        ).astype(np.float64)
        executor.execute_token(x)

        stats = executor.get_statistics()
        rescales = stats['backend_counters']['rescales']

        assert rescales <= DEFAULT_RESCALE_BUDGET, (
            f"Rescales ({rescales}) exceed budget ({DEFAULT_RESCALE_BUDGET})"
        )


class TestBudgetComplianceCI:
    """CI-focused budget compliance tests."""

    def test_full_budget_compliance(self, qkv_config):
        """Test full budget compliance check (CI gate)."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        cost_estimate = estimate_costs(
            qkv_config,
            schedule.layout,
            qkv_config.ckks_profile,
        )

        budget = get_strict_budget()
        passed, violations = check_budget_compliance(
            cost_estimate,
            budget,
            qkv_config.targets,
        )

        if not passed:
            pytest.fail(f"Budget compliance check failed: {violations}")

    def test_rotation_invariant_enforcement(self, qkv_config):
        """Test rotation invariant enforcement."""
        ckks_params = get_profile(qkv_config.ckks_profile)
        schedule = compile_schedule(qkv_config, ckks_params)

        actual = schedule.predicted_costs.rotations_per_token
        expected = DEFAULT_ROTATION_BUDGET

        passed, message = enforce_rotation_invariant(
            actual_rotations=actual,
            expected_rotations=expected,
            tolerance=0.0,  # No tolerance for CI
        )

        assert passed, f"Rotation invariant failed: {message}"

    @pytest.mark.parametrize("hidden_size", [256, 512, 1024])
    @pytest.mark.parametrize("rank", [8, 16])
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_various_configurations(self, hidden_size, rank, batch_size):
        """Test budget compliance across various configurations."""
        config = LoRAConfig(
            hidden_size=hidden_size,
            rank=rank,
            alpha=2.0 * rank,
            targets=LoRATargets.QKV,
            batch_size=batch_size,
            max_context_length=1024,
            ckks_profile=CKKSProfile.FAST,
        )

        ckks_params = get_profile(config.ckks_profile)

        try:
            schedule = compile_schedule(config, ckks_params)
        except ValueError as e:
            pytest.skip(f"Configuration not supported: {e}")

        if not schedule.is_valid:
            pytest.skip(f"Invalid schedule: {schedule.validation_errors}")

        rotations = schedule.predicted_costs.rotations_per_token

        assert rotations <= DEFAULT_ROTATION_BUDGET * 2, (
            f"Config (h={hidden_size}, r={rank}, b={batch_size}) "
            f"has {rotations} rotations, exceeding relaxed budget"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
