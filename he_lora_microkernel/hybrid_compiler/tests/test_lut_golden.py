"""
Golden Test Suite for LUTs and Bridge Operations

Tests correctness of:
1. LUT entries across full integer domain
2. Quantization/dequantization error bounds and monotonicity
3. Bridge operations round-trip consistency
4. Sign encoding/decoding correctness
"""

import pytest
import numpy as np
from typing import List, Tuple

from ..tfhe_lut import (
    LUTEntry,
    LUTLibrary,
    step_lut,
    sign_lut,
    clip_lut,
    relu_lut,
    argmax_2_lut,
    create_custom_lut,
)
from ..bridge import CKKSTFHEBridge, BridgeConfig, QuantizationParams
from ..backend import (
    HybridHEBackend,
    HybridHEConfig,
    SimulatedBridgeService,
)


# =============================================================================
# LUT Correctness Tests
# =============================================================================

class TestStepLUT:
    """Golden tests for step function LUT."""

    @pytest.mark.parametrize("bits", [4, 6, 8, 10])
    def test_step_correctness_full_domain(self, bits: int):
        """Test step function across full input domain."""
        lut = step_lut(bits)
        size = 1 << bits
        half = size // 2

        # Verify all entries
        for i in range(size):
            signed_val = i - half
            expected = 1 if signed_val >= 0 else 0
            actual = lut.lookup(signed_val)
            assert actual == expected, f"step({signed_val}) = {actual}, expected {expected}"

    def test_step_output_bits(self):
        """Test step LUT uses correct output bits."""
        lut = step_lut(8)
        assert lut.output_bits == 1
        assert all(e in [0, 1] for e in lut.entries)

    def test_step_boundary(self):
        """Test step function at boundary (x=0)."""
        lut = step_lut(8)
        assert lut.lookup(0) == 1  # step(0) = 1
        assert lut.lookup(-1) == 0  # step(-1) = 0
        assert lut.lookup(1) == 1  # step(1) = 1

    def test_step_validation(self):
        """Test step LUT passes validation."""
        lut = step_lut(8)
        errors = lut.validate_entries()
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestSignLUT:
    """Golden tests for sign function LUT."""

    @pytest.mark.parametrize("bits", [4, 6, 8, 10])
    def test_sign_correctness_full_domain(self, bits: int):
        """Test sign function across full input domain."""
        lut = sign_lut(bits)
        size = 1 << bits
        half = size // 2

        for i in range(size):
            signed_val = i - half
            if signed_val > 0:
                expected = 1
            elif signed_val < 0:
                expected = -1
            else:
                expected = 0

            # Get decoded result
            actual = lut.lookup_decoded(signed_val)
            assert actual == expected, f"sign({signed_val}) = {actual}, expected {expected}"

    def test_sign_encoding_valid_range(self):
        """Test that all encoded entries are in valid range."""
        lut = sign_lut(8)
        max_val = (1 << lut.output_bits) - 1

        for entry in lut.entries:
            assert 0 <= entry <= max_val, f"Entry {entry} out of range [0, {max_val}]"

    def test_sign_encoding_decode_roundtrip(self):
        """Test encode/decode round-trip for sign values."""
        lut = sign_lut(8)

        # Test all logical values
        for logical in [-1, 0, 1]:
            encoded = lut.encode_output(logical)
            decoded = lut.decode_output(encoded)
            assert decoded == logical, f"Round-trip failed: {logical} -> {encoded} -> {decoded}"

    def test_sign_offset_correct(self):
        """Test sign LUT uses correct offset."""
        lut = sign_lut(8)
        assert lut.output_offset == 1, f"Expected offset 1, got {lut.output_offset}"

        # Verify encoding: -1 -> 0, 0 -> 1, 1 -> 2
        assert lut.encode_output(-1) == 0
        assert lut.encode_output(0) == 1
        assert lut.encode_output(1) == 2

    def test_sign_validation(self):
        """Test sign LUT passes validation."""
        lut = sign_lut(8)
        errors = lut.validate_entries()
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestClipLUT:
    """Golden tests for clip function LUT."""

    @pytest.mark.parametrize("lo,hi", [(-64, 64), (-32, 32), (-100, 100)])
    def test_clip_correctness(self, lo: int, hi: int):
        """Test clip function with various bounds."""
        lut = clip_lut(8, lo, hi)
        half = 128  # For 8-bit

        for i in range(256):
            signed_val = i - half
            expected = max(lo, min(hi, signed_val))

            # Note: clip uses offset encoding for signed output
            encoded = lut.lookup(signed_val)
            actual = encoded - half  # Decode

            assert actual == expected, f"clip({signed_val}, {lo}, {hi}) = {actual}, expected {expected}"

    def test_clip_boundary_values(self):
        """Test clip at boundary values."""
        lut = clip_lut(8, -64, 64)
        half = 128

        # At lower bound
        assert lut.lookup(-128) - half == -64  # Clipped to lo
        assert lut.lookup(-64) - half == -64   # At boundary

        # At upper bound
        assert lut.lookup(127) - half == 64    # Clipped to hi
        assert lut.lookup(64) - half == 64     # At boundary

        # Inside range
        assert lut.lookup(0) - half == 0
        assert lut.lookup(-10) - half == -10

    def test_clip_validation(self):
        """Test clip LUT passes validation."""
        lut = clip_lut(8)
        errors = lut.validate_entries()
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestReLULUT:
    """Golden tests for ReLU function LUT."""

    def test_relu_correctness(self):
        """Test ReLU across full domain."""
        lut = relu_lut(8)
        half = 128

        for i in range(256):
            signed_val = i - half
            expected = max(0, signed_val)
            actual = lut.lookup(signed_val)
            assert actual == expected, f"relu({signed_val}) = {actual}, expected {expected}"

    def test_relu_output_non_negative(self):
        """Test all ReLU outputs are non-negative."""
        lut = relu_lut(8)
        assert all(e >= 0 for e in lut.entries)

    def test_relu_validation(self):
        """Test ReLU LUT passes validation."""
        lut = relu_lut(8)
        errors = lut.validate_entries()
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestArgmax2LUT:
    """Golden tests for binary argmax LUT."""

    def test_argmax2_correctness(self):
        """Test argmax_2 for all packed inputs."""
        lut = argmax_2_lut(8)

        for i in range(256):
            a = (i >> 4) & 0xF  # High 4 bits
            b = i & 0xF         # Low 4 bits
            expected = 0 if a > b else 1
            actual = lut.lookup(i - 128)  # Signed lookup
            # Actually for argmax_2, input is unsigned
            actual = lut.entries[i]
            assert actual == expected, f"argmax_2({a}, {b}) = {actual}, expected {expected}"


class TestCustomLUT:
    """Tests for custom LUT creation."""

    def test_custom_square(self):
        """Test custom square function LUT."""
        def square(x: int) -> int:
            return min(x * x, 127)  # Clamp to fit output range

        lut = create_custom_lut("square", square, bits=4, is_signed=True)

        # Verify some values
        assert lut.lookup(0) == 0
        assert lut.lookup(1) == 1
        assert lut.lookup(2) == 4
        assert lut.lookup(-2) == 4  # (-2)^2 = 4

    def test_custom_abs(self):
        """Test custom absolute value LUT."""
        def abs_func(x: int) -> int:
            return abs(x)

        lut = create_custom_lut("abs", abs_func, bits=8, is_signed=True)

        for val in [-127, -50, -1, 0, 1, 50, 127]:
            assert lut.lookup(val) == abs(val)


# =============================================================================
# Quantization Tests
# =============================================================================

class TestQuantization:
    """Tests for quantization error bounds and monotonicity."""

    def test_quantize_dequantize_roundtrip(self):
        """Test quantization round-trip error is bounded."""
        params = QuantizationParams(bits=8, clip_min=-10.0, clip_max=10.0)

        test_values = np.linspace(-10, 10, 1000)
        max_error = 0.0

        for val in test_values:
            q = params.quantize(val)
            dq = params.dequantize(q)
            error = abs(val - dq)
            max_error = max(max_error, error)

        # Maximum error should be at most half the quantization step
        step = 20.0 / (2 ** 7 - 1)  # Range / (max_int - 1) for symmetric
        assert max_error <= step, f"Max error {max_error} exceeds step {step}"

    def test_quantization_monotonicity(self):
        """Test quantization preserves monotonicity."""
        params = QuantizationParams(bits=8, clip_min=-10.0, clip_max=10.0)

        values = np.linspace(-9, 9, 100)
        quantized = [params.quantize(v) for v in values]

        # Should be monotonically non-decreasing
        for i in range(1, len(quantized)):
            assert quantized[i] >= quantized[i-1], \
                f"Monotonicity violated at {values[i-1]} -> {values[i]}"

    def test_quantization_symmetric(self):
        """Test symmetric quantization property."""
        params = QuantizationParams(bits=8, symmetric=True)

        # Symmetric around zero
        for val in [1.0, 2.0, 5.0, 9.0]:
            q_pos = params.quantize(val)
            q_neg = params.quantize(-val)
            assert q_pos == -q_neg, f"Asymmetric: q({val})={q_pos}, q({-val})={q_neg}"

    def test_quantization_clipping(self):
        """Test values outside clip range are clipped."""
        params = QuantizationParams(bits=8, clip_min=-10.0, clip_max=10.0)

        # Values outside range should be clipped
        q_over = params.quantize(100.0)
        q_under = params.quantize(-100.0)

        max_int = (1 << 7) - 1
        assert q_over == max_int
        assert q_under == -max_int


# =============================================================================
# Bridge Tests
# =============================================================================

class TestBridge:
    """Tests for CKKS-TFHE bridge operations."""

    def test_bridge_roundtrip_consistency(self):
        """Test bridge round-trip produces consistent results."""
        bridge = CKKSTFHEBridge()

        test_values = np.array([0.0, 1.0, -1.0, 5.5, -7.3])

        # CKKS -> TFHE -> CKKS round-trip
        tfhe_ct = bridge.ckks_to_tfhe(test_values)
        ckks_values = bridge.tfhe_to_ckks(tfhe_ct)

        # Should be close after quantization
        np.testing.assert_allclose(
            test_values, ckks_values,
            rtol=0.1, atol=0.1,
            err_msg="Bridge round-trip error too large"
        )

    def test_bridge_preserves_sign(self):
        """Test bridge preserves sign of values."""
        bridge = CKKSTFHEBridge()

        pos_values = np.array([1.0, 2.0, 3.0])
        neg_values = np.array([-1.0, -2.0, -3.0])

        tfhe_pos = bridge.ckks_to_tfhe(pos_values)
        tfhe_neg = bridge.ckks_to_tfhe(neg_values)

        # All positive values should have positive quantized values
        assert all(v > 0 for v in tfhe_pos['values']), "Sign not preserved for positive"
        # All negative values should have negative quantized values
        assert all(v < 0 for v in tfhe_neg['values']), "Sign not preserved for negative"

    def test_bridge_lut_application(self):
        """Test LUT application through bridge."""
        bridge = CKKSTFHEBridge()
        lut = step_lut(8)

        # Test various pre-activation values
        test_cases = [
            (1.0, 1),   # Positive -> gate ON
            (-1.0, 0),  # Negative -> gate OFF
            (0.0, 1),   # Zero -> gate ON (step(0) = 1)
            (5.0, 1),   # Large positive -> ON
            (-5.0, 0),  # Large negative -> OFF
        ]

        for ckks_val, expected_gate in test_cases:
            tfhe_ct = bridge.ckks_to_tfhe(np.array([ckks_val]))
            result = bridge.apply_lut_simulation(tfhe_ct, lut.entries)
            actual_gate = result['values'][0]
            assert actual_gate == expected_gate, \
                f"step({ckks_val}) = {actual_gate}, expected {expected_gate}"

    def test_bridge_stats_tracking(self):
        """Test bridge statistics are tracked."""
        bridge = CKKSTFHEBridge()
        bridge.reset_stats()

        values = np.array([1.0, 2.0, 3.0])
        bridge.ckks_to_tfhe(values)
        bridge.ckks_to_tfhe(values)

        stats = bridge.get_stats()
        assert stats['ckks_to_tfhe_count'] == 2


class TestSimulatedBridgeService:
    """Tests for simulated bridge service."""

    def test_quantization_error_tracking(self):
        """Test bridge service tracks quantization error."""
        service = SimulatedBridgeService(quant_bits=8)

        from ..backend import CKKSCiphertext
        ct = CKKSCiphertext(
            data=None,
            shape=(3,),
            scale=2**40,
            _plaintext=np.array([1.5, 2.7, -3.3]),
        )

        service.ckks_to_tfhe(ct, "req_1", 8)

        stats = service.get_stats()
        assert stats['conversions'] >= 1
        # Error tracking should be present
        assert 'total_error' in stats


# =============================================================================
# Integration Tests
# =============================================================================

class TestHybridBackendIntegration:
    """Integration tests for hybrid backend."""

    def test_gated_lora_step_gate_on(self):
        """Test gated LoRA with step gate ON."""
        backend = HybridHEBackend.create_simulated()

        # Input that triggers positive gate
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct_x = backend.ckks_encrypt(x)

        # Create gate weight that produces positive pre-activation
        w_gate = np.ones(4) * 0.5  # Positive gate pre-activation

        ct_z = backend.ckks_matvec(ct_x, w_gate.reshape(1, -1))

        # Bridge and apply step LUT
        ct_z_tfhe = backend.bridge_ckks_to_tfhe(ct_z, "test_req")
        ct_g_tfhe = backend.tfhe_lut_apply(ct_z_tfhe, "step")

        # Verify gate is ON
        g = backend.tfhe_decrypt(ct_g_tfhe)
        assert g[0] == 1, f"Expected gate ON (1), got {g[0]}"

    def test_gated_lora_step_gate_off(self):
        """Test gated LoRA with step gate OFF."""
        backend = HybridHEBackend.create_simulated()

        # Input that triggers negative gate
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct_x = backend.ckks_encrypt(x)

        # Create gate weight that produces negative pre-activation
        w_gate = np.ones(4) * -0.5  # Negative gate pre-activation

        ct_z = backend.ckks_matvec(ct_x, w_gate.reshape(1, -1))

        # Bridge and apply step LUT
        ct_z_tfhe = backend.bridge_ckks_to_tfhe(ct_z, "test_req")
        ct_g_tfhe = backend.tfhe_lut_apply(ct_z_tfhe, "step")

        # Verify gate is OFF
        g = backend.tfhe_decrypt(ct_g_tfhe)
        assert g[0] == 0, f"Expected gate OFF (0), got {g[0]}"

    def test_operation_stats_collection(self):
        """Test operation statistics are collected."""
        backend = HybridHEBackend.create_simulated()
        backend.reset_stats()

        x = np.array([1.0, 2.0])
        ct_x = backend.ckks_encrypt(x)
        ct_y = backend.ckks_add(ct_x, ct_x)

        ct_tfhe = backend.bridge_ckks_to_tfhe(ct_y, "req")
        ct_lut = backend.tfhe_lut_apply(ct_tfhe, "step")
        ct_back = backend.bridge_tfhe_to_ckks(ct_lut, "req")

        stats = backend.get_operation_stats()
        assert stats.ckks_ops >= 2, "Should have at least 2 CKKS ops"
        assert stats.bridge_to_tfhe_count == 1
        assert stats.bridge_to_ckks_count == 1
        assert stats.bootstrap_count == 1


# =============================================================================
# LUT Library Tests
# =============================================================================

class TestLUTLibrary:
    """Tests for LUT library management."""

    def test_library_has_standard_luts(self):
        """Test library contains all standard LUTs."""
        library = LUTLibrary(8)

        required_luts = ["step", "sign", "clip_-64_64", "argmax_2", "relu"]
        for name in required_luts:
            lut = library.get(name)
            assert lut is not None, f"Missing standard LUT: {name}"

    def test_library_register_custom(self):
        """Test registering custom LUTs."""
        library = LUTLibrary(8)

        def threshold_3(x: int) -> int:
            return 1 if x >= 3 else 0

        custom_lut = create_custom_lut("threshold_3", threshold_3, bits=8)
        library.register(custom_lut)

        retrieved = library.get("threshold_3")
        assert retrieved is not None
        assert retrieved.lookup(5) == 1
        assert retrieved.lookup(0) == 0


# =============================================================================
# Test Runner
# =============================================================================

def run_golden_tests():
    """Run all golden tests and report results."""
    test_classes = [
        TestStepLUT,
        TestSignLUT,
        TestClipLUT,
        TestReLULUT,
        TestArgmax2LUT,
        TestCustomLUT,
        TestQuantization,
        TestBridge,
        TestSimulatedBridgeService,
        TestHybridBackendIntegration,
        TestLUTLibrary,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    # Handle parametrized tests
                    if hasattr(method, 'pytestmark'):
                        # Skip parametrized for simple runner
                        method()
                    else:
                        method()
                    passed += 1
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                except AssertionError as e:
                    failed += 1
                    errors.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                except Exception as e:
                    failed += 1
                    errors.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"  ✗ {test_class.__name__}.{method_name}: ERROR - {e}")

    print(f"\nGolden Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_golden_tests()
    exit(0 if success else 1)
