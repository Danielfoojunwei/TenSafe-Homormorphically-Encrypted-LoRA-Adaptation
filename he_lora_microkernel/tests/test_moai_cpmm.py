"""
Tests for MOAI CPMM (Column-Packed Matrix Multiplication) kernel.

Verifies that CPMM produces correct results and achieves rotation reduction.
"""

import pytest
import numpy as np
import math

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.compiler.moai_cpmm import (
    CPMMKernel,
    CPMMConfig,
    LoRACPMMKernel,
    compare_approaches,
)


class TestCPMMKernel:
    """Tests for the base CPMM kernel."""

    def test_basic_initialization(self):
        """Test kernel initialization."""
        kernel = CPMMKernel(
            out_size=64,
            in_size=128,
            slot_count=8192,
            batch_size=4,
        )

        assert kernel.config.out_size == 64
        assert kernel.config.in_size == 128
        assert kernel.config.batch_size == 4
        assert kernel.config.num_blocks >= 1

    def test_input_packing_roundtrip(self):
        """Test that pack/unpack preserves values for input."""
        kernel = CPMMKernel(
            out_size=32,
            in_size=64,
            slot_count=4096,
            batch_size=2,
        )

        # Random input
        x = np.random.randn(2, 64)

        # Pack
        packed = kernel.pack_input(x)

        # Verify correct slots are filled
        for (batch_idx, elem_idx), slot_idx in kernel.layout.input_slot_map.items():
            assert abs(packed[slot_idx] - x[batch_idx, elem_idx]) < 1e-10

    def test_weight_packing(self):
        """Test weight matrix packing."""
        kernel = CPMMKernel(
            out_size=32,
            in_size=64,
            slot_count=4096,
            batch_size=2,
        )

        W = np.random.randn(32, 64)
        packed_rows = kernel.pack_weights(W)

        assert len(packed_rows) == 32  # One per output row

        # Verify alignment with input slots
        for out_idx, packed in enumerate(packed_rows):
            for (batch_idx, elem_idx), slot_idx in kernel.layout.input_slot_map.items():
                if batch_idx == 0:  # Same for all batches
                    assert abs(packed[slot_idx] - W[out_idx, elem_idx]) < 1e-10

    def test_simulated_execution_correctness(self):
        """Test that simulated execution produces correct results."""
        kernel = CPMMKernel(
            out_size=64,
            in_size=128,
            slot_count=8192,
            batch_size=4,
        )

        # Random input and weights
        x = np.random.randn(4, 128)
        W = np.random.randn(64, 128)

        # Pack weights
        kernel.pack_weights(W)

        # Execute
        result = kernel.execute_simulated(x)

        # Compare to reference
        expected = x @ W.T

        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_rotation_count_single_block(self):
        """Test that single-block config has zero rotations."""
        kernel = CPMMKernel(
            out_size=32,
            in_size=64,
            slot_count=4096,
            batch_size=4,  # 64 * 4 = 256 slots, fits in one block
        )

        # Should have 1 block, 0 accumulation rotations
        assert kernel.config.num_blocks == 1
        assert kernel.layout.rotations_per_matmul == 0
        assert len(kernel.layout.accumulation_rotations) == 0

    def test_rotation_count_multi_block(self):
        """Test rotation count for multi-block config."""
        kernel = CPMMKernel(
            out_size=32,
            in_size=512,
            slot_count=1024,
            batch_size=2,  # 512 * 2 = 1024 slots, forces multiple blocks
            block_size=128,  # Force small blocks
        )

        # Should have multiple blocks
        assert kernel.config.num_blocks > 1

        # Rotations only for accumulation
        expected_rotations = len(kernel.layout.accumulation_rotations)
        assert kernel.layout.rotations_per_matmul == expected_rotations

        # Should be logarithmic
        assert expected_rotations <= math.ceil(math.log2(kernel.config.num_blocks)) + 1


class TestLoRACPMMKernel:
    """Tests for the LoRA-specialized CPMM kernel."""

    def test_lora_initialization(self):
        """Test LoRA kernel initialization."""
        kernel = LoRACPMMKernel(
            hidden_size=256,
            rank=16,
            slot_count=8192,
            batch_size=4,
        )

        assert kernel.hidden_size == 256
        assert kernel.rank == 16

    def test_lora_weight_loading(self):
        """Test LoRA weight loading and pre-computation."""
        kernel = LoRACPMMKernel(
            hidden_size=128,
            rank=8,
            slot_count=4096,
            batch_size=2,
        )

        A = np.random.randn(128, 8) * 0.01
        B = np.random.randn(8, 128) * 0.01
        alpha = 16.0

        kernel.load_weights(A, B, alpha)

        # Verify AB is pre-computed
        expected_AB = (alpha / 8) * (A @ B)
        np.testing.assert_allclose(kernel._AB, expected_AB, rtol=1e-10)

    def test_lora_delta_correctness(self):
        """Test LoRA delta computation correctness."""
        kernel = LoRACPMMKernel(
            hidden_size=128,
            rank=8,
            slot_count=4096,
            batch_size=4,
        )

        A = np.random.randn(128, 8) * 0.01
        B = np.random.randn(8, 128) * 0.01
        alpha = 16.0

        kernel.load_weights(A, B, alpha)

        # Random input
        x = np.random.randn(4, 128)

        # Compute delta
        delta = kernel.compute_delta(x)

        # Reference: Δy = (α/r) * A @ B @ x
        expected = (alpha / 8) * (x @ B.T @ A.T)

        np.testing.assert_allclose(delta, expected, rtol=1e-10)

    def test_lora_various_ranks(self):
        """Test LoRA kernel with various ranks."""
        for rank in [4, 8, 16, 32]:
            kernel = LoRACPMMKernel(
                hidden_size=256,
                rank=rank,
                slot_count=8192,
                batch_size=2,
            )

            A = np.random.randn(256, rank) * 0.01
            B = np.random.randn(rank, 256) * 0.01
            alpha = 2.0 * rank

            kernel.load_weights(A, B, alpha)

            x = np.random.randn(2, 256)
            delta = kernel.compute_delta(x)

            expected = (alpha / rank) * (x @ B.T @ A.T)
            np.testing.assert_allclose(delta, expected, rtol=1e-9)


class TestCompareApproaches:
    """Tests for approach comparison."""

    def test_comparison_output_structure(self):
        """Test that comparison returns expected structure."""
        result = compare_approaches(
            hidden_size=256,
            rank=8,
            batch_size=4,
            slot_count=4096,
        )

        assert 'configuration' in result
        assert 'diagonal_method' in result
        assert 'speculative_batching' in result
        assert 'moai_cpmm' in result
        assert 'speedup' in result

    def test_cpmm_has_fewer_rotations(self):
        """Test that CPMM has fewer rotations than diagonal method."""
        result = compare_approaches(
            hidden_size=256,
            rank=8,
            batch_size=4,
            slot_count=4096,
        )

        diagonal_rots = result['diagonal_method']['rotations_per_matmul']
        cpmm_rots = result['moai_cpmm']['rotations_per_matmul']

        # CPMM should have far fewer rotations
        assert cpmm_rots < diagonal_rots
        assert result['speedup']['cpmm_vs_diagonal'] > 1


class TestCPMMVsSpeculativeBatching:
    """
    Tests comparing CPMM to the existing speculative batching system.

    These tests verify the KEY DIFFERENCES between the approaches.
    """

    def test_packing_layout_difference(self):
        """
        Test the fundamental packing layout difference.

        Speculative Batching: batch-first layout
        CPMM: column-major blocks with weight alignment
        """
        # CPMM kernel
        cpmm = CPMMKernel(
            out_size=32,
            in_size=64,
            slot_count=2048,
            batch_size=4,
        )

        # In CPMM, input slots are organized by blocks
        # Check that consecutive elements within a block are adjacent
        block_size = cpmm.config.block_size

        for block_id in range(cpmm.config.num_blocks):
            start_elem = block_id * block_size
            slots_in_block = []

            for local_idx in range(block_size):
                elem_idx = start_elem + local_idx
                if elem_idx >= cpmm.config.in_size:
                    break
                # Check slot positions for batch 0
                if (0, elem_idx) in cpmm.layout.input_slot_map:
                    slots_in_block.append(cpmm.layout.input_slot_map[(0, elem_idx)])

            # Slots within a block should be grouped
            if len(slots_in_block) > 1:
                # Verify slots are nearby (not scattered)
                slot_range = max(slots_in_block) - min(slots_in_block)
                expected_range = len(slots_in_block) * cpmm.config.batch_size - 1
                assert slot_range <= expected_range * 2  # Allow some flexibility

    def test_zero_intra_block_rotations(self):
        """
        Verify CPMM achieves zero intra-block rotations.

        This is the KEY advantage of CPMM.
        """
        # Config that fits in single block
        cpmm = CPMMKernel(
            out_size=64,
            in_size=128,
            slot_count=4096,
            batch_size=4,  # 128 * 4 = 512 slots
        )

        # Single block = zero rotations total
        if cpmm.config.num_blocks == 1:
            assert cpmm.layout.rotations_per_matmul == 0

        # Multi-block config
        cpmm_multi = CPMMKernel(
            out_size=64,
            in_size=512,
            slot_count=1024,
            batch_size=1,
            block_size=64,  # Force multiple blocks
        )

        # Even with multiple blocks, rotations are only for accumulation
        # NOT for element gathering within blocks
        rotations = cpmm_multi.layout.rotations_per_matmul
        num_blocks = cpmm_multi.config.num_blocks

        # Rotations should be O(log(blocks)), not O(hidden_size)
        assert rotations <= math.ceil(math.log2(num_blocks)) + 1

    def test_element_wise_multiply_alignment(self):
        """
        Test that weight packing enables direct element-wise multiply.

        In CPMM, W[i,j] is placed at the same slot as x[j],
        so Ct×Pt directly computes partial dot products.
        """
        cpmm = CPMMKernel(
            out_size=16,
            in_size=32,
            slot_count=1024,
            batch_size=2,
        )

        W = np.random.randn(16, 32)
        cpmm.pack_weights(W)

        # For each output row i
        for out_idx, packed_row in enumerate(cpmm._weight_plaintexts):
            # W[i, j] should be at the same slot as x[j]
            for (batch_idx, elem_idx), slot_idx in cpmm.layout.input_slot_map.items():
                expected_weight = W[out_idx, elem_idx]
                actual_weight = packed_row[slot_idx]
                assert abs(expected_weight - actual_weight) < 1e-10, \
                    f"Weight alignment failed for W[{out_idx},{elem_idx}]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
