"""
Tests for Optimized CKKS-TFHE Bridge

Tests cover:
1. 8-bit quantization correctness
2. Batched LUT evaluation
3. SIMD packing operations
4. Lazy evaluation queue
5. Step function accuracy
"""

import numpy as np
import pytest

from he_lora_microkernel.hybrid_compiler.bridge.optimized_bridge import (
    BatchedBootstrapConfig,
    BatchedLUT,
    LazyGateBridge,
    OptimizedCKKSTFHEBridge,
    QuantizationParams,
    SIMDPackedTFHE,
)


class TestQuantization:
    """Test 8-bit quantization."""

    def test_quantization_range(self):
        """8-bit quantization should produce values in [-127, 127]."""
        params = QuantizationParams()

        values = np.array([-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0])
        quantized = params.quantize_batch(values)

        assert np.all(quantized >= -127)
        assert np.all(quantized <= 127)

    def test_sign_preservation(self):
        """Quantization should preserve sign for non-tiny values."""
        params = QuantizationParams()

        values = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        quantized = params.quantize_batch(values)

        assert quantized[0] < 0  # -5.0 -> negative
        assert quantized[1] < 0  # -1.0 -> negative
        assert quantized[2] == 0  # 0.0 -> 0
        assert quantized[3] > 0  # 1.0 -> positive
        assert quantized[4] > 0  # 5.0 -> positive

    def test_clipping_behavior(self):
        """Values outside clip range should be clamped."""
        params = QuantizationParams(clip_min=-10.0, clip_max=10.0)

        values = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        quantized = params.quantize_batch(values)

        # Extreme values should be clipped
        assert quantized[0] == quantized[1]  # Both clipped to min
        assert quantized[4] == quantized[3]  # Both clipped to max

    def test_dequantize_roundtrip(self):
        """Quantize-dequantize should approximately preserve values."""
        params = QuantizationParams()

        original = 5.0
        quantized = params.quantize(original)
        recovered = params.dequantize(quantized)

        # Should be close (within quantization error)
        assert abs(recovered - original) < 0.1


class TestBatchedLUT:
    """Test batched LUT evaluation."""

    def test_step_lut_correctness(self):
        """Step LUT should return 0 for negative, 1 for non-negative."""
        lut = BatchedLUT.step_lut(bits=8)

        # Test range for 8-bit: -127 to 127
        test_values = np.array([-127, -64, -1, 0, 1, 64, 127], dtype=np.int32)
        results = lut.evaluate_batch(test_values)

        expected = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(results, expected)

    def test_sign_lut_correctness(self):
        """Sign LUT should return -1, 0, or 1."""
        lut = BatchedLUT.sign_lut(bits=8)

        test_values = np.array([-127, -1, 0, 1, 127], dtype=np.int32)
        results = lut.evaluate_batch(test_values)

        expected = np.array([-1, -1, 0, 1, 1], dtype=np.int32)
        np.testing.assert_array_equal(results, expected)

    def test_vectorized_evaluation(self):
        """Batch evaluation should be consistent with sequential."""
        lut = BatchedLUT.step_lut(bits=8)

        # Large batch
        values = np.random.randint(-127, 128, size=1000, dtype=np.int32)

        # Batch evaluation
        batch_results = lut.evaluate_batch(values)

        # Sequential evaluation
        seq_results = np.array([lut.entries[v + 128] for v in values])

        np.testing.assert_array_equal(batch_results, seq_results)


class TestOptimizedBridge:
    """Test the optimized bridge end-to-end."""

    def test_gate_evaluation_correctness(self):
        """Batched gate evaluation should produce correct results."""
        bridge = OptimizedCKKSTFHEBridge()

        # Test values with clear sign
        preactivations = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        gates = bridge.evaluate_gates_batched(preactivations)

        # Step function: 1 if >= 0, else 0
        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(gates, expected)

    def test_batch_size_independence(self):
        """Results should be same regardless of batch size."""
        bridge = OptimizedCKKSTFHEBridge()

        np.random.seed(42)
        values = np.random.randn(64) * 3

        # Single batch
        full_result = bridge.evaluate_gates_batched(values)

        # Multiple smaller batches
        partial_results = []
        for i in range(0, 64, 8):
            partial = bridge.evaluate_gates_batched(values[i:i+8])
            partial_results.append(partial)
        combined_result = np.concatenate(partial_results)

        np.testing.assert_array_equal(full_result, combined_result)

    def test_statistics_tracking(self):
        """Bridge should track statistics correctly."""
        bridge = OptimizedCKKSTFHEBridge()
        bridge.reset_stats()

        # Process some batches
        for batch_size in [8, 16, 32]:
            values = np.random.randn(batch_size)
            bridge.evaluate_gates_batched(values)

        stats = bridge.get_stats()
        assert stats['batched_bootstraps'] == 3
        assert stats['total_values_processed'] == 8 + 16 + 32

    def test_empty_batch(self):
        """Empty batch should not crash."""
        bridge = OptimizedCKKSTFHEBridge()
        result = bridge.evaluate_gates_batched(np.array([]))
        assert len(result) == 0


class TestSIMDPackedTFHE:
    """Test SIMD-style packing for TFHE operations."""

    def test_pack_unpack_roundtrip(self):
        """Packing and unpacking should preserve values."""
        simd = SIMDPackedTFHE(pack_size=8)

        # Test various sizes
        for size in [1, 7, 8, 15, 16, 63, 64]:
            values = np.random.randn(size)
            packed = simd.pack_signs(values)
            unpacked = simd.unpack_gates(packed, size)

            # Check sign preservation
            expected_signs = (values >= 0).astype(np.float64)
            np.testing.assert_array_equal(unpacked, expected_signs)

    def test_step_function_via_packing(self):
        """SIMD-packed step evaluation should match direct evaluation."""
        simd = SIMDPackedTFHE(pack_size=8)

        values = np.array([-3.0, -0.1, 0.0, 0.1, 3.0, -2.0, 1.0, -1.0])
        result = simd.evaluate_packed_step(values)

        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_pack_efficiency(self):
        """8 values should pack into 1 byte."""
        simd = SIMDPackedTFHE(pack_size=8)

        values = np.random.randn(64)
        packed = simd.pack_signs(values)

        # 64 values should pack into 8 bytes
        assert len(packed) == 8


class TestLazyGateBridge:
    """Test lazy evaluation with batching."""

    def test_queue_and_flush(self):
        """Queued evaluations should be processed on flush."""
        lazy = LazyGateBridge(batch_threshold=4)

        # Queue some evaluations
        preacts = [np.array([1.0]), np.array([-1.0]), np.array([0.5])]
        deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

        for p, d in zip(preacts, deltas):
            lazy.queue_gate_evaluation(p, d)

        assert lazy.pending_count == 3

        # Flush manually
        results = lazy.flush()

        assert lazy.pending_count == 0
        assert len(results) == 3

        # Check gating is applied correctly
        # preact[0] = 1.0 >= 0 -> gate = 1 -> delta unchanged
        np.testing.assert_array_equal(results[0], deltas[0])
        # preact[1] = -1.0 < 0 -> gate = 0 -> delta zeroed
        np.testing.assert_array_equal(results[1], np.zeros_like(deltas[1]))
        # preact[2] = 0.5 >= 0 -> gate = 1 -> delta unchanged
        np.testing.assert_array_equal(results[2], deltas[2])

    def test_auto_flush_at_threshold(self):
        """Should auto-flush when threshold reached."""
        lazy = LazyGateBridge(batch_threshold=2)

        results_collected = []

        def callback(gated):
            results_collected.append(gated)

        lazy.queue_gate_evaluation(np.array([1.0]), np.array([1.0]), callback)
        assert lazy.pending_count == 1
        assert len(results_collected) == 0

        lazy.queue_gate_evaluation(np.array([1.0]), np.array([2.0]), callback)
        # Should have auto-flushed
        assert lazy.pending_count == 0
        assert len(results_collected) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_value(self):
        """Single value should work correctly."""
        bridge = OptimizedCKKSTFHEBridge()

        for val in [-1.0, 0.0, 1.0]:
            result = bridge.evaluate_gates_batched(np.array([val]))
            expected = 1.0 if val >= 0 else 0.0
            assert result[0] == expected

    def test_extreme_values(self):
        """Very large values should be handled (clipped to range)."""
        bridge = OptimizedCKKSTFHEBridge()

        # Test very large values - these get clipped but sign is preserved
        values = np.array([-1e10, -1.0, 0.0, 1.0, 1e10])
        result = bridge.evaluate_gates_batched(values)

        expected = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_nan_handling(self):
        """NaN values should be handled gracefully."""
        bridge = OptimizedCKKSTFHEBridge()

        values = np.array([1.0, np.nan, -1.0])
        # Should not crash
        result = bridge.evaluate_gates_batched(values)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
