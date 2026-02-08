"""
Fidelity Tests for HE-LoRA Microkernel

These tests verify that HE-LoRA produces results matching
FP16 PyTorch LoRA within acceptable error bounds.

Error bounds:
  - abs error ≤ 1e-2
  - rel error ≤ 1e-2
"""

import os
import sys
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.backend.gpu_ckks_backend import BackendType
from he_lora_microkernel.compiler import (
    CKKSProfile,
    LoRAConfig,
    LoRATargets,
    compile_schedule,
    get_profile,
    pack_activations,
    unpack_activations,
)
from he_lora_microkernel.runtime import HELoRAExecutor

# =============================================================================
# REFERENCE IMPLEMENTATION
# =============================================================================

def reference_lora_forward(
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Reference LoRA forward pass in FP64.

    Computes: Δy = (alpha / rank) * A @ B @ x

    Args:
        x: Input activations (batch_size, hidden_size)
        A: Up-projection (hidden_size, rank)
        B: Down-projection (rank, hidden_size)
        alpha: Scaling factor

    Returns:
        LoRA delta
    """
    rank = A.shape[1]
    scaling = alpha / rank

    # Two-step: B @ x^T, then A @ result
    # x is (batch, hidden), B is (rank, hidden)
    # B @ x^T gives (rank, batch), transpose gives (batch, rank)
    intermediate = (B @ x.T).T  # (batch, rank)
    delta = (A @ intermediate.T).T  # (batch, hidden)

    return scaling * delta


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def small_config():
    """Small configuration for quick tests."""
    return LoRAConfig(
        hidden_size=256,
        rank=8,
        alpha=16.0,
        targets=LoRATargets.QKV,
        batch_size=4,
        max_context_length=512,
        ckks_profile=CKKSProfile.FAST,
    )


@pytest.fixture
def medium_config():
    """Medium configuration for thorough tests."""
    return LoRAConfig(
        hidden_size=1024,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=8,
        max_context_length=2048,
        ckks_profile=CKKSProfile.FAST,
    )


@pytest.fixture
def random_weights(small_config) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random LoRA weights."""
    rng = np.random.default_rng(42)

    A = rng.standard_normal(
        (small_config.hidden_size, small_config.rank)
    ).astype(np.float64) * 0.01

    B = rng.standard_normal(
        (small_config.rank, small_config.hidden_size)
    ).astype(np.float64) * 0.01

    return A, B


@pytest.fixture
def random_activations(small_config) -> np.ndarray:
    """Generate random activations."""
    rng = np.random.default_rng(123)
    return rng.standard_normal(
        (small_config.batch_size, small_config.hidden_size)
    ).astype(np.float64)


# =============================================================================
# FIDELITY TESTS
# =============================================================================

class TestFidelity:
    """Fidelity tests comparing HE-LoRA to reference implementation."""

    def test_reference_implementation(self, small_config, random_weights, random_activations):
        """Test reference implementation produces expected shape."""
        A, B = random_weights
        x = random_activations

        delta = reference_lora_forward(x, A, B, small_config.alpha)

        assert delta.shape == x.shape
        assert np.isfinite(delta).all()

    def test_packing_roundtrip(self, small_config):
        """Test that packing/unpacking preserves values."""
        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        rng = np.random.default_rng(42)
        activations = rng.standard_normal(
            (small_config.batch_size, small_config.hidden_size)
        )

        packed = pack_activations(activations, schedule.layout)
        unpacked = unpack_activations(packed, schedule.layout)

        np.testing.assert_allclose(
            activations, unpacked,
            rtol=1e-10,
            err_msg="Packing roundtrip failed"
        )

    def test_simulation_fidelity(self, small_config, random_weights, random_activations):
        """Test HE-LoRA simulation matches reference within bounds."""
        A, B = random_weights
        x = random_activations
        alpha = small_config.alpha

        # Reference computation
        expected = reference_lora_forward(x, A, B, alpha)

        # HE-LoRA computation (simulation)
        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        executor = HELoRAExecutor(
            schedule,
            BackendType.SIMULATION,
        )
        executor.load_weights(A, B, alpha)

        actual = executor.execute_token(x)

        # Check error bounds
        abs_error = np.abs(expected - actual)
        rel_error = abs_error / (np.abs(expected) + 1e-10)

        max_abs_error = np.max(abs_error)
        max_rel_error = np.max(rel_error)

        assert max_abs_error <= 1e-2, f"Absolute error too high: {max_abs_error}"
        assert max_rel_error <= 1e-2, f"Relative error too high: {max_rel_error}"

    def test_multiple_tokens_fidelity(self, small_config, random_weights):
        """Test fidelity across multiple tokens."""
        A, B = random_weights
        alpha = small_config.alpha

        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        executor = HELoRAExecutor(
            schedule,
            BackendType.SIMULATION,
        )
        executor.load_weights(A, B, alpha)

        rng = np.random.default_rng(42)

        max_abs_errors = []
        max_rel_errors = []

        for token_idx in range(10):
            x = rng.standard_normal(
                (small_config.batch_size, small_config.hidden_size)
            ).astype(np.float64)

            expected = reference_lora_forward(x, A, B, alpha)
            actual = executor.execute_token(x, position=token_idx)

            abs_error = np.abs(expected - actual)
            rel_error = abs_error / (np.abs(expected) + 1e-10)

            max_abs_errors.append(np.max(abs_error))
            max_rel_errors.append(np.max(rel_error))

        avg_abs = np.mean(max_abs_errors)
        avg_rel = np.mean(max_rel_errors)

        assert avg_abs <= 1e-2, f"Average absolute error too high: {avg_abs}"
        assert avg_rel <= 1e-2, f"Average relative error too high: {avg_rel}"

    def test_batch_size_invariance(self, random_weights):
        """Test that results are consistent across batch sizes."""
        A, B = random_weights
        alpha = 16.0
        hidden_size = A.shape[0]
        rank = A.shape[1]

        rng = np.random.default_rng(42)

        # Single sample
        x_single = rng.standard_normal((1, hidden_size)).astype(np.float64)
        expected_single = reference_lora_forward(x_single, A, B, alpha)

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            config = LoRAConfig(
                hidden_size=hidden_size,
                rank=rank,
                alpha=alpha,
                targets=LoRATargets.QKV,
                batch_size=batch_size,
                max_context_length=512,
                ckks_profile=CKKSProfile.FAST,
            )

            ckks_params = get_profile(config.ckks_profile)
            schedule = compile_schedule(config, ckks_params)

            executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
            executor.load_weights(A, B, alpha)

            # Create batch with same first sample
            x_batch = np.zeros((batch_size, hidden_size))
            x_batch[0] = x_single[0]

            actual = executor.execute_token(x_batch)

            # First sample should match
            np.testing.assert_allclose(
                expected_single[0], actual[0],
                rtol=1e-2, atol=1e-2,
                err_msg=f"Batch size {batch_size} produced different result"
            )

    def test_rank_variations(self, small_config):
        """Test fidelity across different ranks."""
        rng = np.random.default_rng(42)

        for rank in [4, 8, 16]:
            A = rng.standard_normal(
                (small_config.hidden_size, rank)
            ).astype(np.float64) * 0.01

            B = rng.standard_normal(
                (rank, small_config.hidden_size)
            ).astype(np.float64) * 0.01

            alpha = 2.0 * rank

            x = rng.standard_normal(
                (small_config.batch_size, small_config.hidden_size)
            ).astype(np.float64)

            expected = reference_lora_forward(x, A, B, alpha)

            config = LoRAConfig(
                hidden_size=small_config.hidden_size,
                rank=rank,
                alpha=alpha,
                targets=LoRATargets.QKV,
                batch_size=small_config.batch_size,
                max_context_length=512,
                ckks_profile=CKKSProfile.FAST,
            )

            ckks_params = get_profile(config.ckks_profile)
            schedule = compile_schedule(config, ckks_params)

            executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
            executor.load_weights(A, B, alpha)

            actual = executor.execute_token(x)

            max_abs = np.max(np.abs(expected - actual))
            assert max_abs <= 1e-2, f"Rank {rank}: abs error {max_abs}"


class TestEdgeCases:
    """Edge case fidelity tests."""

    def test_zero_weights(self, small_config, random_activations):
        """Test with zero weights produces zero output."""
        A = np.zeros((small_config.hidden_size, small_config.rank))
        B = np.zeros((small_config.rank, small_config.hidden_size))

        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, small_config.alpha)

        delta = executor.execute_token(random_activations)

        # Should be close to zero
        assert np.max(np.abs(delta)) < 1e-10

    def test_identity_like_weights(self, small_config, random_activations):
        """Test with identity-like weights."""
        # A = I[:, :rank], B = I[:rank, :]
        rank = small_config.rank
        hidden = small_config.hidden_size

        A = np.zeros((hidden, rank))
        B = np.zeros((rank, hidden))
        for i in range(min(rank, hidden)):
            A[i, i] = 1.0
            B[i, i] = 1.0

        x = random_activations
        expected = reference_lora_forward(x, A, B, small_config.alpha)

        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, small_config.alpha)

        actual = executor.execute_token(x)

        np.testing.assert_allclose(expected, actual, rtol=1e-2, atol=1e-2)

    def test_large_values(self, small_config):
        """Test with large activation values."""
        rng = np.random.default_rng(42)

        A = rng.standard_normal(
            (small_config.hidden_size, small_config.rank)
        ).astype(np.float64) * 0.01

        B = rng.standard_normal(
            (small_config.rank, small_config.hidden_size)
        ).astype(np.float64) * 0.01

        # Large activations
        x = rng.standard_normal(
            (small_config.batch_size, small_config.hidden_size)
        ).astype(np.float64) * 100

        expected = reference_lora_forward(x, A, B, small_config.alpha)

        ckks_params = get_profile(small_config.ckks_profile)
        schedule = compile_schedule(small_config, ckks_params)

        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, small_config.alpha)

        actual = executor.execute_token(x)

        # Use relative error for large values
        rel_error = np.abs(expected - actual) / (np.abs(expected) + 1e-10)
        assert np.max(rel_error) <= 1e-2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
