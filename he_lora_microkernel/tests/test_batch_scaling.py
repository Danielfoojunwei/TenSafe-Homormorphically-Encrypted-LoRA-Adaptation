"""
Batch Scaling Tests for HE-LoRA Microkernel

These tests verify batch adjustability and scaling behavior:
  1. Batch size changes trigger recompilation
  2. Performance scales appropriately with batch size
  3. Results are consistent across batch sizes
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.backend.gpu_ckks_backend import BackendType
from he_lora_microkernel.compiler import (
    CKKSProfile,
    LoRAConfig,
    LoRATargets,
    compile_schedule,
    estimate_costs,
    get_profile,
)
from he_lora_microkernel.runtime import (
    BatchManager,
    DynamicBatchExecutor,
    HELoRAExecutor,
    pad_activations,
    unpad_activations,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def base_config():
    """Base configuration for batch scaling tests."""
    return LoRAConfig(
        hidden_size=512,
        rank=16,
        alpha=32.0,
        targets=LoRATargets.QKV,
        batch_size=8,  # Default
        max_context_length=1024,
        ckks_profile=CKKSProfile.FAST,
    )


@pytest.fixture
def random_weights():
    """Generate random LoRA weights."""
    hidden_size = 512
    rank = 16
    rng = np.random.default_rng(42)

    A = rng.standard_normal((hidden_size, rank)).astype(np.float64) * 0.01
    B = rng.standard_normal((rank, hidden_size)).astype(np.float64) * 0.01

    return A, B


# =============================================================================
# BATCH SIZE ADJUSTMENT TESTS
# =============================================================================

class TestBatchSizeAdjustment:
    """Tests for batch size adjustment and recompilation."""

    def test_recompile_on_batch_change(self, base_config):
        """Test that batch size change triggers recompilation."""
        ckks_params = get_profile(base_config.ckks_profile)

        # Compile for different batch sizes
        schedules = {}
        for batch_size in [1, 4, 8, 16]:
            config = LoRAConfig(
                hidden_size=base_config.hidden_size,
                rank=base_config.rank,
                alpha=base_config.alpha,
                targets=base_config.targets,
                batch_size=batch_size,
                max_context_length=base_config.max_context_length,
                ckks_profile=base_config.ckks_profile,
            )
            schedules[batch_size] = compile_schedule(config, ckks_params)

        # Verify different schedules are generated
        hashes = [s.schedule_hash for s in schedules.values()]
        assert len(set(hashes)) == len(hashes), (
            "Different batch sizes should produce different schedules"
        )

    def test_layout_changes_with_batch(self, base_config):
        """Test that packing layout changes with batch size."""
        ckks_params = get_profile(base_config.ckks_profile)

        layouts = {}
        for batch_size in [1, 4, 8]:
            config = LoRAConfig(
                hidden_size=base_config.hidden_size,
                rank=base_config.rank,
                alpha=base_config.alpha,
                targets=base_config.targets,
                batch_size=batch_size,
                max_context_length=base_config.max_context_length,
                ckks_profile=base_config.ckks_profile,
            )
            schedule = compile_schedule(config, ckks_params)
            layouts[batch_size] = schedule.layout

        # Slots used should scale with batch size
        assert layouts[4].total_slots_used > layouts[1].total_slots_used
        assert layouts[8].total_slots_used > layouts[4].total_slots_used

    def test_batch_manager_caching(self, base_config):
        """Test that BatchManager caches compiled schedules."""
        ckks_params = get_profile(base_config.ckks_profile)

        manager = BatchManager(base_config, ckks_params, precompile_sizes=[1, 4, 8])

        # First access - may compile or use cache
        schedule1 = manager.set_batch_size(4)

        # Second access - should use cache
        schedule2 = manager.set_batch_size(4)

        # Should be same object
        assert schedule1.schedule_hash == schedule2.schedule_hash

    def test_batch_manager_auto_select(self, base_config):
        """Test BatchManager automatic batch size selection."""
        ckks_params = get_profile(base_config.ckks_profile)

        manager = BatchManager(base_config, ckks_params, precompile_sizes=[1, 4, 8, 16])

        # Find optimal for throughput
        optimal = manager.get_optimal_batch_size(optimize_for='throughput')
        assert optimal in [1, 4, 8, 16]

        # Find optimal for latency
        optimal_latency = manager.get_optimal_batch_size(optimize_for='latency')
        assert optimal_latency in [1, 4, 8, 16]


class TestBatchPerformanceScaling:
    """Tests for performance scaling with batch size."""

    def test_aggregate_throughput_scaling(self, base_config):
        """Test that aggregate throughput increases with batch size."""
        ckks_params = get_profile(base_config.ckks_profile)

        costs = {}
        for batch_size in [1, 4, 8]:
            config = LoRAConfig(
                hidden_size=base_config.hidden_size,
                rank=base_config.rank,
                alpha=base_config.alpha,
                targets=base_config.targets,
                batch_size=batch_size,
                max_context_length=base_config.max_context_length,
                ckks_profile=base_config.ckks_profile,
            )
            schedule = compile_schedule(config, ckks_params)
            cost = estimate_costs(config, schedule.layout, config.ckks_profile)

            # Aggregate throughput = batch_size / time_per_batch
            costs[batch_size] = {
                'per_batch_us': cost.total_us,
                'aggregate_throughput': batch_size * 1_000_000 / cost.total_us,
            }

        # Larger batches should have better aggregate throughput
        # (not strictly linear due to overhead)
        assert costs[4]['aggregate_throughput'] > costs[1]['aggregate_throughput']

    def test_rotation_count_vs_batch(self, base_config):
        """Test rotation count relationship with batch size."""
        ckks_params = get_profile(base_config.ckks_profile)

        rotations = {}
        for batch_size in [1, 4, 8]:
            config = LoRAConfig(
                hidden_size=base_config.hidden_size,
                rank=base_config.rank,
                alpha=base_config.alpha,
                targets=base_config.targets,
                batch_size=batch_size,
                max_context_length=base_config.max_context_length,
                ckks_profile=base_config.ckks_profile,
            )
            schedule = compile_schedule(config, ckks_params)
            rotations[batch_size] = schedule.predicted_costs.rotations_per_token

        # Rotations per token should be similar across batch sizes
        # (MOAI packing processes batch in parallel in same slots)
        rot_values = list(rotations.values())
        max_diff = max(rot_values) - min(rot_values)
        assert max_diff <= 5, (
            f"Rotation count varies too much with batch size: {rotations}"
        )


class TestBatchPadding:
    """Tests for batch padding and unpadding."""

    def test_pad_activations(self):
        """Test activation padding."""
        actual_batch = 3
        target_batch = 8
        hidden_size = 256

        rng = np.random.default_rng(42)
        activations = rng.standard_normal((actual_batch, hidden_size))

        padded, actual = pad_activations(activations, target_batch)

        assert padded.shape == (target_batch, hidden_size)
        assert actual == actual_batch
        np.testing.assert_array_equal(padded[:actual_batch], activations)
        np.testing.assert_array_equal(padded[actual_batch:], 0)

    def test_unpad_activations(self):
        """Test activation unpadding."""
        target_batch = 8
        actual_batch = 3
        hidden_size = 256

        rng = np.random.default_rng(42)
        padded = rng.standard_normal((target_batch, hidden_size))

        unpadded = unpad_activations(padded, actual_batch)

        assert unpadded.shape == (actual_batch, hidden_size)
        np.testing.assert_array_equal(unpadded, padded[:actual_batch])

    def test_pad_unpad_roundtrip(self):
        """Test padding roundtrip preserves values."""
        actual_batch = 5
        target_batch = 16
        hidden_size = 128

        rng = np.random.default_rng(42)
        original = rng.standard_normal((actual_batch, hidden_size))

        padded, _ = pad_activations(original, target_batch)
        unpadded = unpad_activations(padded, actual_batch)

        np.testing.assert_array_equal(original, unpadded)


class TestBatchConsistency:
    """Tests for result consistency across batch sizes."""

    def test_single_vs_batch_consistency(self, base_config, random_weights):
        """Test that single sample gives same result in batch."""
        A, B = random_weights
        alpha = base_config.alpha
        hidden_size = base_config.hidden_size

        rng = np.random.default_rng(42)
        single_input = rng.standard_normal((1, hidden_size)).astype(np.float64)

        # Reference: single sample
        ckks_params = get_profile(base_config.ckks_profile)
        single_config = LoRAConfig(
            hidden_size=hidden_size,
            rank=base_config.rank,
            alpha=alpha,
            targets=base_config.targets,
            batch_size=1,
            max_context_length=base_config.max_context_length,
            ckks_profile=base_config.ckks_profile,
        )
        single_schedule = compile_schedule(single_config, ckks_params)
        single_executor = HELoRAExecutor(single_schedule, BackendType.SIMULATION)
        single_executor.load_weights(A, B, alpha)
        single_result = single_executor.execute_token(single_input)

        # Batch with same first sample
        batch_size = 4
        batch_config = LoRAConfig(
            hidden_size=hidden_size,
            rank=base_config.rank,
            alpha=alpha,
            targets=base_config.targets,
            batch_size=batch_size,
            max_context_length=base_config.max_context_length,
            ckks_profile=base_config.ckks_profile,
        )
        batch_schedule = compile_schedule(batch_config, ckks_params)
        batch_executor = HELoRAExecutor(batch_schedule, BackendType.SIMULATION)
        batch_executor.load_weights(A, B, alpha)

        batch_input = np.zeros((batch_size, hidden_size))
        batch_input[0] = single_input[0]
        batch_result = batch_executor.execute_token(batch_input)

        # First sample should match
        np.testing.assert_allclose(
            single_result[0], batch_result[0],
            rtol=1e-2, atol=1e-2,
            err_msg="Single vs batch results differ"
        )

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_size_result_stability(self, base_config, random_weights, batch_size):
        """Test result stability across batch sizes."""
        A, B = random_weights
        alpha = base_config.alpha
        hidden_size = base_config.hidden_size

        rng = np.random.default_rng(42)

        # Create config for this batch size
        config = LoRAConfig(
            hidden_size=hidden_size,
            rank=base_config.rank,
            alpha=alpha,
            targets=base_config.targets,
            batch_size=batch_size,
            max_context_length=base_config.max_context_length,
            ckks_profile=base_config.ckks_profile,
        )

        ckks_params = get_profile(config.ckks_profile)
        schedule = compile_schedule(config, ckks_params)
        executor = HELoRAExecutor(schedule, BackendType.SIMULATION)
        executor.load_weights(A, B, alpha)

        # Execute multiple times with same input
        x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)

        results = []
        for _ in range(3):
            result = executor.execute_token(x.copy())
            results.append(result.copy())

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0], results[i],
                rtol=1e-10,
                err_msg=f"Results differ between executions (batch_size={batch_size})"
            )


class TestDynamicBatchExecutor:
    """Tests for DynamicBatchExecutor."""

    def test_dynamic_batch_creation(self, base_config):
        """Test DynamicBatchExecutor creation."""
        from he_lora_microkernel.python.helora.config import HELoRAConfig

        config = HELoRAConfig(
            hidden_size=base_config.hidden_size,
            lora_rank=base_config.rank,
            batch_size=base_config.batch_size,
        )

        executor = DynamicBatchExecutor(base_config, backend_type='SIMULATION')
        assert executor is not None

    def test_performance_report(self, base_config):
        """Test performance report generation."""
        executor = DynamicBatchExecutor(base_config, backend_type='SIMULATION')

        report = executor.get_performance_report()

        assert 'batch_comparison' in report
        assert 'optimal_for_throughput' in report
        assert 'optimal_for_latency' in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
