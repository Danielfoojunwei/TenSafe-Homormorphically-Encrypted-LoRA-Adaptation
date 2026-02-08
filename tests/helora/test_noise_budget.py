"""
Noise budget tests for HE-LoRA.

These tests verify that:
1. Noise budget is properly tracked
2. Levels remaining after LoRA delta >= configured minimum
3. Operations fail explicitly when budget exhausted

Test requirements:
1. Assert levels_remaining >= min_levels after B(Ax)
2. Test must FAIL if noise budget exhausted unexpectedly
3. Test must FAIL if N2HE-HEXL backend not available
"""

import numpy as np
import pytest


def backend_available():
    """Check if N2HE-HEXL backend is available."""
    try:
        from crypto_backend.n2he_hexl import N2HEHEXLBackend
        backend = N2HEHEXLBackend()
        return backend.is_available()
    except ImportError:
        return False


# Minimum levels required after LoRA delta computation
MIN_LEVELS_AFTER_LORA = 1


class TestNoiseBudget:
    """Test noise budget tracking and management."""

    def test_backend_required(self):
        """Test that backend is available - fail if not."""
        if not backend_available():
            pytest.fail(
                "N2HE-HEXL backend is REQUIRED for noise budget tests.\n"
                "Build with: ./scripts/build_n2he_hexl.sh"
            )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_levels_after_lora_delta(self):
        """Test that sufficient levels remain after LoRA delta."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        config = HELoRAConfig(rank=8, alpha=16.0)
        adapter = HELoRAAdapter(config)

        # Register weights
        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        adapter.register_weights("test", lora_a, lora_b, rank=8, alpha=16.0)

        # Run forward
        x = np.random.randn(64).astype(np.float64)
        _ = adapter.forward(x, "test")

        # Check levels remaining
        metrics = adapter.get_last_metrics()
        assert metrics is not None, "No metrics from forward pass"

        levels_remaining = metrics.levels_remaining
        assert levels_remaining >= MIN_LEVELS_AFTER_LORA, (
            f"Only {levels_remaining} levels remaining after LoRA delta, "
            f"need at least {MIN_LEVELS_AFTER_LORA}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_levels_logged_correctly(self):
        """Test that level information is in structured log."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        config = HELoRAConfig(rank=8, alpha=16.0)
        adapter = HELoRAAdapter(config)

        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        adapter.register_weights("test", lora_a, lora_b, rank=8, alpha=16.0)

        x = np.random.randn(64).astype(np.float64)
        _ = adapter.forward(x, "test")

        metrics = adapter.get_last_metrics()
        log_dict = metrics.to_log_dict()

        assert "levels_remaining" in log_dict, "levels_remaining not in log output"
        assert isinstance(log_dict["levels_remaining"], int), "levels_remaining not integer"
        assert log_dict["levels_remaining"] >= 0, "levels_remaining cannot be negative"

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_scale_bits_tracked(self):
        """Test that scale is properly tracked."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        scale_bits = 40
        config = HELoRAConfig(rank=8, alpha=16.0, scale_bits=scale_bits)
        adapter = HELoRAAdapter(config)

        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        adapter.register_weights("test", lora_a, lora_b)

        x = np.random.randn(64).astype(np.float64)
        _ = adapter.forward(x, "test")

        metrics = adapter.get_last_metrics()
        log_dict = metrics.to_log_dict()

        assert "scale_bits" in log_dict, "scale_bits not in log output"
        assert log_dict["scale_bits"] == scale_bits, (
            f"Scale bits mismatch: expected {scale_bits}, got {log_dict['scale_bits']}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_noise_tracker_estimates_levels(self):
        """Test noise tracker can estimate required levels."""
        from tensafe.he_lora.noise_tracker import NoiseTracker

        tracker = NoiseTracker(initial_levels=4, scale_bits=40)

        # Estimate for typical LoRA
        levels_needed = tracker.estimate_lora_levels_needed(
            hidden_dim=64,
            rank=8,
            out_dim=64,
            scaling=True
        )

        # Should need 2-3 levels for matmuls + 1 for scaling + 1 reserved
        assert levels_needed >= 3, f"Estimate too low: {levels_needed}"
        assert levels_needed <= 5, f"Estimate too high: {levels_needed}"


class TestNoiseBudgetExhaustion:
    """Test noise budget exhaustion handling."""

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_explicit_failure_on_exhaustion(self):
        """Test that exhaustion causes explicit failure, not silent corruption."""
        from tensafe.he_lora.noise_tracker import NoiseBudgetExhaustedError, NoiseTracker

        # Create tracker with very limited budget
        tracker = NoiseTracker(
            initial_levels=2,  # Very limited
            scale_bits=40,
            min_levels_required=1
        )

        # Simulate operations that would exhaust budget
        ct_id = 1
        tracker.create_state(ct_id, level=1, scale=2**40)

        # This should raise when we try to consume more levels
        with pytest.raises(NoiseBudgetExhaustedError):
            # Try to consume levels we don't have
            for _ in range(5):
                tracker.update_state(ct_id, "rescale")

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_check_can_operate(self):
        """Test the can_operate check works correctly."""
        from tensafe.he_lora.noise_tracker import NoiseTracker

        tracker = NoiseTracker(
            initial_levels=4,
            scale_bits=40,
            min_levels_required=1
        )

        ct_id = 1
        tracker.create_state(ct_id, level=3, scale=2**40)

        # Should be able to do 2 rescales (consume 2 levels)
        assert tracker.check_can_operate(ct_id, ["rescale", "rescale"]), (
            "Should have budget for 2 rescales"
        )

        # Should NOT be able to do 4 rescales (would go below min)
        assert not tracker.check_can_operate(ct_id, ["rescale"] * 4), (
            "Should not have budget for 4 rescales"
        )


class TestContextParameters:
    """Test that context parameters are correct for noise budget."""

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_coeff_modulus_chain_sufficient(self):
        """Test coefficient modulus chain has enough levels."""
        from tensafe.he_lora import get_backend

        backend = get_backend()
        params = backend.get_context_params()

        chain_length = params.get("coeff_modulus_chain_length", 0)

        # Need at least 4 levels for LoRA:
        # 1 for encrypt, 2 for matmuls, 1 reserved for decrypt
        MIN_CHAIN_LENGTH = 4

        assert chain_length >= MIN_CHAIN_LENGTH, (
            f"Coefficient modulus chain too short: {chain_length}, "
            f"need at least {MIN_CHAIN_LENGTH}"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_scale_appropriate(self):
        """Test scale is appropriate for precision."""
        from tensafe.he_lora import get_backend

        backend = get_backend()
        params = backend.get_context_params()

        scale = params.get("scale", 0)
        scale_bits = params.get("scale_bits", 0)

        # Scale should be 2^scale_bits
        expected_scale = 2.0 ** scale_bits

        assert abs(scale - expected_scale) < 1, (
            f"Scale mismatch: {scale} vs expected {expected_scale}"
        )

        # Scale bits should be reasonable (30-50 for CKKS)
        assert 30 <= scale_bits <= 50, (
            f"Scale bits {scale_bits} outside reasonable range"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
