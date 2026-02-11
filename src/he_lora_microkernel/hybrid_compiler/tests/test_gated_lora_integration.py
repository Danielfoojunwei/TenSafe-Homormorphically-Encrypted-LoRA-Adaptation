"""
Integration Tests for Gated LoRA Adapter

Validates numerical correctness between:
1. Plaintext gated LoRA reference implementation
2. Hybrid CKKS-TFHE encrypted gated LoRA

Tests gate behavior, error bounds, and edge cases.
"""

import pytest
import numpy as np
from typing import Dict, Any, Tuple

from ..backend import (
    HybridHEBackend,
    HybridHEConfig,
    BridgeMode,
)
from ..adapters import (
    HEGatedLoRAAdapter,
    GatedLoRAAdapterConfig,
    AdapterWeights,
    AdapterMetrics,
    plaintext_gated_lora,
)
from ..tfhe_lut import LUTLibrary, step_lut, sign_lut


# =============================================================================
# Test Fixtures
# =============================================================================

class TestFixtures:
    """Shared test fixtures."""

    @staticmethod
    def create_random_weights(
        hidden_size: int = 64,
        rank: int = 4,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create random LoRA and gate weights."""
        rng = np.random.default_rng(seed)

        lora_A = rng.normal(0, 0.1, (rank, hidden_size)).astype(np.float64)
        lora_B = rng.normal(0, 0.1, (hidden_size, rank)).astype(np.float64)
        w_gate = rng.normal(0, 0.1, (hidden_size,)).astype(np.float64)
        b_gate = np.array([0.0])  # Default zero bias

        return lora_A, lora_B, w_gate, b_gate

    @staticmethod
    def create_adapter_weights(
        hidden_size: int = 64,
        rank: int = 4,
        seed: int = 42,
    ) -> AdapterWeights:
        """Create AdapterWeights object."""
        lora_A, lora_B, w_gate, b_gate = TestFixtures.create_random_weights(
            hidden_size, rank, seed
        )
        return AdapterWeights(
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=b_gate,
        )


# =============================================================================
# Plaintext vs Encrypted Comparison Tests
# =============================================================================

class TestPlaintextVsEncrypted:
    """Compare plaintext and encrypted gated LoRA results."""

    @pytest.mark.parametrize("hidden_size,rank,batch", [
        (64, 4, 1),
        (64, 4, 2),
        (128, 8, 1),
        (256, 16, 1),
    ])
    def test_step_gate_numerical_match(
        self,
        hidden_size: int,
        rank: int,
        batch: int,
    ):
        """Test step gate produces matching results."""
        rng = np.random.default_rng(42)

        # Create inputs and weights
        x = rng.normal(0, 1, (batch, hidden_size)).astype(np.float64)
        base_output = rng.normal(0, 1, (batch, hidden_size)).astype(np.float64)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        # Configure adapter
        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            lora_alpha=32.0,
            gate_type="step",
        )

        # Create backend and adapter
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        # Run plaintext forward
        plaintext_outputs = []
        for i in range(batch):
            out, _ = adapter.forward(x[i], base_output[i], weights)
            plaintext_outputs.append(out)
        plaintext_result = np.stack(plaintext_outputs)

        # Run encrypted forward (simulation)
        encrypted_outputs = []
        for i in range(batch):
            ct_x = backend.ckks_encrypt(x[i])
            ct_base = backend.ckks_encrypt(base_output[i])
            ct_out, _ = adapter.forward_encrypted(
                ct_x, ct_base, weights, f"req_{i}"
            )
            encrypted_outputs.append(backend.ckks_decrypt(ct_out))
        encrypted_result = np.stack(encrypted_outputs)

        # Compare results
        np.testing.assert_allclose(
            plaintext_result, encrypted_result,
            rtol=1e-5, atol=1e-5,
            err_msg="Step gate: plaintext vs encrypted mismatch"
        )

    @pytest.mark.parametrize("hidden_size,rank,batch", [
        (64, 4, 1),
        (64, 4, 2),
    ])
    def test_sign_gate_numerical_match(
        self,
        hidden_size: int,
        rank: int,
        batch: int,
    ):
        """Test sign gate produces matching results."""
        rng = np.random.default_rng(42)

        x = rng.normal(0, 1, (batch, hidden_size)).astype(np.float64)
        base_output = rng.normal(0, 1, (batch, hidden_size)).astype(np.float64)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            lora_alpha=32.0,
            gate_type="sign",
        )

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        # Run plaintext
        plaintext_outputs = []
        for i in range(batch):
            out, _ = adapter.forward(x[i], base_output[i], weights)
            plaintext_outputs.append(out)
        plaintext_result = np.stack(plaintext_outputs)

        # Run encrypted
        encrypted_outputs = []
        for i in range(batch):
            ct_x = backend.ckks_encrypt(x[i])
            ct_base = backend.ckks_encrypt(base_output[i])
            ct_out, _ = adapter.forward_encrypted(
                ct_x, ct_base, weights, f"req_{i}"
            )
            encrypted_outputs.append(backend.ckks_decrypt(ct_out))
        encrypted_result = np.stack(encrypted_outputs)

        np.testing.assert_allclose(
            plaintext_result, encrypted_result,
            rtol=1e-5, atol=1e-5,
            err_msg="Sign gate: plaintext vs encrypted mismatch"
        )


# =============================================================================
# Gate Behavior Tests
# =============================================================================

class TestGateBehavior:
    """Test gate classification behavior."""

    def test_step_gate_on_positive_preactivation(self):
        """Step gate should be ON for positive pre-activation."""
        hidden_size, rank = 64, 4
        rng = np.random.default_rng(42)

        # Create weights where gate will be positive
        x = np.ones(hidden_size) * 0.1  # Positive input
        base = rng.normal(0, 0.1, hidden_size)

        weights = TestFixtures.create_adapter_weights(hidden_size, rank)
        # Set gate weights to produce positive pre-activation
        weights.w_gate = np.ones(hidden_size) * 0.1
        weights.b_gate = np.array([1.0])  # Positive bias

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            gate_type="step",
        )
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        output, metrics = adapter.forward(x, base, weights)

        assert metrics.gate_value == 1.0, f"Expected gate ON (1.0), got {metrics.gate_value}"

        # Output should differ from base by the full LoRA delta
        u = weights.lora_A @ x
        delta = weights.lora_B @ u
        expected = base + config.scaling_factor * delta
        np.testing.assert_allclose(output, expected, rtol=1e-5)

    def test_step_gate_off_negative_preactivation(self):
        """Step gate should be OFF for negative pre-activation."""
        hidden_size, rank = 64, 4
        rng = np.random.default_rng(42)

        x = rng.normal(0, 0.1, hidden_size)
        base = rng.normal(0, 0.1, hidden_size)

        weights = TestFixtures.create_adapter_weights(hidden_size, rank)
        # Set gate weights to produce negative pre-activation
        weights.w_gate = np.ones(hidden_size) * -0.1
        weights.b_gate = np.array([-1.0])  # Negative bias

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            gate_type="step",
        )
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        output, metrics = adapter.forward(x, base, weights)

        assert metrics.gate_value == 0.0, f"Expected gate OFF (0.0), got {metrics.gate_value}"

        # Output should equal base (no LoRA contribution)
        np.testing.assert_allclose(output, base, rtol=1e-5)

    def test_sign_gate_positive(self):
        """Sign gate should be +1 for positive pre-activation."""
        hidden_size, rank = 64, 4

        x = np.ones(hidden_size) * 0.1
        base = np.zeros(hidden_size)

        weights = TestFixtures.create_adapter_weights(hidden_size, rank)
        weights.w_gate = np.ones(hidden_size) * 0.1
        weights.b_gate = np.array([1.0])

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            gate_type="sign",
        )
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        _, metrics = adapter.forward(x, base, weights)
        assert metrics.gate_value == 1.0, f"Expected sign +1, got {metrics.gate_value}"

    def test_sign_gate_negative(self):
        """Sign gate should be -1 for negative pre-activation."""
        hidden_size, rank = 64, 4

        x = np.ones(hidden_size) * 0.1
        base = np.zeros(hidden_size)

        weights = TestFixtures.create_adapter_weights(hidden_size, rank)
        weights.w_gate = np.ones(hidden_size) * -0.1
        weights.b_gate = np.array([-1.0])

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            gate_type="sign",
        )
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        _, metrics = adapter.forward(x, base, weights)
        assert metrics.gate_value == -1.0, f"Expected sign -1, got {metrics.gate_value}"

    def test_sign_gate_zero(self):
        """Sign gate should be 0 at exactly zero pre-activation."""
        hidden_size, rank = 64, 4

        x = np.zeros(hidden_size)
        base = np.zeros(hidden_size)

        weights = TestFixtures.create_adapter_weights(hidden_size, rank)
        weights.w_gate = np.ones(hidden_size)
        weights.b_gate = np.array([0.0])

        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            gate_type="sign",
        )
        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)

        _, metrics = adapter.forward(x, base, weights)
        assert metrics.gate_value == 0.0, f"Expected sign 0, got {metrics.gate_value}"


# =============================================================================
# Error Bound Tests
# =============================================================================

class TestErrorBounds:
    """Test numerical error bounds."""

    def test_quantization_error_within_bounds(self):
        """Quantization error should be within theoretical bounds."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            quantization_bits=8,
            quantization_clip_min=-10.0,
            quantization_clip_max=10.0,
        )

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        # Run multiple samples and check error
        rng = np.random.default_rng(42)
        max_observed_error = 0.0

        for _ in range(100):
            x = rng.normal(0, 1, hidden_size)
            base = rng.normal(0, 1, hidden_size)

            _, metrics = adapter.forward(x, base, weights)
            max_observed_error = max(max_observed_error, metrics.quantization_error)

        # Theoretical max error = clip_range / (2^(bits-1) - 1) / 2
        # For 8-bit, [-10, 10]: step = 20 / 127 ≈ 0.157, max error ≈ 0.079
        theoretical_max = 20.0 / (2**7 - 1) / 2 + 0.01  # Small margin

        assert max_observed_error <= theoretical_max, \
            f"Quantization error {max_observed_error} exceeds bound {theoretical_max}"

    def test_output_error_bounded_relative(self):
        """Output error should be bounded relative to signal magnitude."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
        )

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        rng = np.random.default_rng(42)

        for _ in range(50):
            x = rng.normal(0, 1, hidden_size)
            base = rng.normal(0, 1, hidden_size)

            # Plaintext
            out_plain, _ = adapter.forward(x, base, weights)

            # Encrypted
            ct_x = backend.ckks_encrypt(x)
            ct_base = backend.ckks_encrypt(base)
            ct_out, _ = adapter.forward_encrypted(ct_x, ct_base, weights, "req")
            out_enc = backend.ckks_decrypt(ct_out)

            # Relative error
            rel_error = np.max(np.abs(out_plain - out_enc) / (np.abs(out_plain) + 1e-10))
            assert rel_error < 1e-4, f"Relative error {rel_error} too large"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_input(self):
        """Test with zero input."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(hidden_size=hidden_size, lora_rank=rank)

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        x = np.zeros(hidden_size)
        base = np.zeros(hidden_size)

        output, metrics = adapter.forward(x, base, weights)

        # Output should be base (all zeros)
        np.testing.assert_allclose(output, base, atol=1e-10)

    def test_large_input(self):
        """Test with large input values (should clip)."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(
            hidden_size=hidden_size,
            lora_rank=rank,
            quantization_clip_min=-10.0,
            quantization_clip_max=10.0,
        )

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        # Large positive input
        x = np.ones(hidden_size) * 100.0
        base = np.zeros(hidden_size)

        output, metrics = adapter.forward(x, base, weights)

        # Should not crash, gate should be determined by clipped value
        assert not np.any(np.isnan(output))
        assert not np.any(np.isinf(output))

    def test_very_small_rank(self):
        """Test with rank=1."""
        hidden_size, rank = 64, 1
        config = GatedLoRAAdapterConfig(hidden_size=hidden_size, lora_rank=rank)

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, hidden_size)
        base = rng.normal(0, 1, hidden_size)

        output, metrics = adapter.forward(x, base, weights)
        assert not np.any(np.isnan(output))


# =============================================================================
# Metrics Collection Tests
# =============================================================================

class TestMetricsCollection:
    """Test that metrics are correctly collected."""

    def test_timing_metrics_positive(self):
        """All timing metrics should be non-negative."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(hidden_size=hidden_size, lora_rank=rank)

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, hidden_size)
        base = rng.normal(0, 1, hidden_size)

        _, metrics = adapter.forward(x, base, weights)

        assert metrics.total_time_ms >= 0
        assert metrics.ckks_lora_time_ms >= 0
        assert metrics.ckks_gate_pre_time_ms >= 0
        assert metrics.tfhe_lut_time_ms >= 0

    def test_operation_counts(self):
        """Operation counts should be correct."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(hidden_size=hidden_size, lora_rank=rank)

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, hidden_size)
        base = rng.normal(0, 1, hidden_size)

        _, metrics = adapter.forward(x, base, weights)

        # At minimum: 2 CKKS ops (LoRA), 1 TFHE op (LUT), 1 bootstrap
        assert metrics.ckks_ops >= 2
        assert metrics.tfhe_ops >= 1
        assert metrics.bootstrap_count >= 1

    def test_metrics_to_dict(self):
        """Metrics should be serializable to dict."""
        hidden_size, rank = 64, 4
        config = GatedLoRAAdapterConfig(hidden_size=hidden_size, lora_rank=rank)

        backend = HybridHEBackend.create_simulated()
        adapter = HEGatedLoRAAdapter(config, backend)
        weights = TestFixtures.create_adapter_weights(hidden_size, rank)

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, hidden_size)
        base = rng.normal(0, 1, hidden_size)

        _, metrics = adapter.forward(x, base, weights)

        d = metrics.to_dict()
        assert isinstance(d, dict)
        assert "total_time_ms" in d
        assert "gate_value" in d


# =============================================================================
# Test Runner
# =============================================================================

def run_integration_tests():
    """Run all integration tests."""
    test_classes = [
        TestPlaintextVsEncrypted,
        TestGateBehavior,
        TestErrorBounds,
        TestEdgeCases,
        TestMetricsCollection,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                except Exception as e:
                    failed += 1
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")

    print(f"\nIntegration Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
