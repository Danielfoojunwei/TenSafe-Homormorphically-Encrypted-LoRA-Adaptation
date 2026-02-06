"""
Rotation budget tests for HE-LoRA.

These tests verify that MOAI-style column packing achieves
ZERO rotations for plaintext-ciphertext matrix multiplication.

This is a key optimization from MOAI - rotations are expensive
in HE, and column packing removes them for pt-ct matmul.

Test requirements:
1. Assert rotations_used == 0 for column-packed LoRA delta
2. Test must FAIL if rotations exceed threshold
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


# MOAI target: zero rotations for column-packed pt-ct matmul
MAX_ROTATIONS_COLUMN_PACKED = 0


class TestRotationBudget:
    """Test rotation count meets MOAI targets."""

    def test_backend_required(self):
        """Test that backend is available - fail if not."""
        if not backend_available():
            pytest.fail(
                "N2HE-HEXL backend is REQUIRED for rotation tests.\n"
                "Build with: ./scripts/build_n2he_hexl.sh"
            )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_column_packed_zero_rotations(self):
        """Test that column packing achieves zero rotations."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        hidden_dim = 64
        rank = 8
        alpha = 16.0

        # Create LoRA weights
        lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01

        # Create adapter with column packing
        config = HELoRAConfig(rank=rank, alpha=alpha)
        adapter = HELoRAAdapter(config)
        adapter.register_weights("test", lora_a, lora_b, rank=rank, alpha=alpha)

        # Run forward
        x = np.random.randn(hidden_dim).astype(np.float64)
        _ = adapter.forward(x, "test")

        # Check rotation count
        metrics = adapter.get_last_metrics()
        assert metrics is not None, "No metrics from forward pass"

        rotations = metrics.rotations_used
        assert rotations <= MAX_ROTATIONS_COLUMN_PACKED, (
            f"Column-packed LoRA used {rotations} rotations, "
            f"expected <= {MAX_ROTATIONS_COLUMN_PACKED}\n"
            "MOAI column packing should eliminate rotations for pt-ct matmul"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_rotation_count_logged(self):
        """Verify rotation count is properly tracked and logged."""
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

        # Check metrics contain rotation info
        metrics = adapter.get_last_metrics()
        log_dict = metrics.to_log_dict()

        assert "rotations_used" in log_dict, "Rotation count not in log output"
        assert isinstance(log_dict["rotations_used"], int), "Rotation count not integer"

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_rotation_verification_method(self):
        """Test the verify_rotation_count helper method."""
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

        # Verify rotation count
        assert adapter.verify_rotation_count(expected_max=0), (
            "verify_rotation_count(0) should pass for column-packed operations"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_multiple_forwards_consistent(self):
        """Test rotation count is consistent across multiple forwards."""
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig

        config = HELoRAConfig(rank=8, alpha=16.0)
        adapter = HELoRAAdapter(config)

        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        adapter.register_weights("test", lora_a, lora_b, rank=8, alpha=16.0)

        rotation_counts = []
        for _ in range(5):
            x = np.random.randn(64).astype(np.float64)
            _ = adapter.forward(x, "test")
            metrics = adapter.get_last_metrics()
            rotation_counts.append(metrics.rotations_used)

        # All should be zero
        assert all(r == 0 for r in rotation_counts), (
            f"Inconsistent rotation counts: {rotation_counts}"
        )


class TestPackingStrategyComparison:
    """Compare rotation costs across packing strategies."""

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_column_vs_estimated_row(self):
        """Compare column packing rotations vs estimated row packing."""
        from tensafe.he_lora.packing import PackingStrategy, estimate_rotation_count

        hidden_dim = 64
        rank = 8

        # Estimate rotations for row packing (would require rotations)
        row_rotations = estimate_rotation_count(
            PackingStrategy.ROW,
            matrix_shape=(rank, hidden_dim),
        )

        # Estimate rotations for column packing (should be zero)
        col_rotations = estimate_rotation_count(
            PackingStrategy.COLUMN,
            matrix_shape=(rank, hidden_dim),
        )

        assert col_rotations == 0, "Column packing should have zero rotations"
        assert row_rotations > col_rotations, (
            "Row packing should require more rotations than column packing"
        )

    @pytest.mark.skipif(not backend_available(), reason="Backend required")
    def test_rotation_savings_documented(self):
        """Verify the rotation savings are significant."""
        from tensafe.he_lora.packing import PackingStrategy, estimate_rotation_count

        # Typical LoRA dimensions
        test_cases = [
            (8, 64),    # Small
            (16, 256),  # Medium
            (32, 512),  # Large
        ]

        for rank, hidden_dim in test_cases:
            row_rotations = estimate_rotation_count(
                PackingStrategy.ROW,
                matrix_shape=(rank, hidden_dim),
            )
            col_rotations = estimate_rotation_count(
                PackingStrategy.COLUMN,
                matrix_shape=(rank, hidden_dim),
            )

            # Column packing should eliminate all rotations
            assert col_rotations == 0, f"Column packing failed for ({rank}, {hidden_dim})"

            # Row packing would have significant overhead
            # log2(hidden_dim) rotations per row * rank rows
            if row_rotations > 0:
                savings = row_rotations - col_rotations
                assert savings > 0, (
                    f"No rotation savings for ({rank}, {hidden_dim})"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
