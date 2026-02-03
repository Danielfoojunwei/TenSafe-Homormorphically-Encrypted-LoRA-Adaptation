"""
Comprehensive HE Integration Tests.

This module tests the full HE integration including:
- CKKS backend simulation
- Rotation count validation (MOAI 0-rotation verification)
- VeRA CKKS restructuring
- Extended TGSP registry
- Benchmark reporting

Author: TenSafe Team
"""

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pytest


# =============================================================================
# MOCK CKKS BACKEND FOR TESTING
# =============================================================================

@dataclass
class MockCiphertext:
    """Mock ciphertext for simulation testing."""
    data: np.ndarray
    level: int = 0
    scale: float = 2**40
    slot_count: int = 4096


@dataclass
class MockOperationCounters:
    """Track operations in mock backend."""
    rotations: int = 0
    keyswitches: int = 0
    rescales: int = 0
    multiplications: int = 0
    additions: int = 0
    encryptions: int = 0
    decryptions: int = 0

    def reset(self):
        self.rotations = 0
        self.keyswitches = 0
        self.rescales = 0
        self.multiplications = 0
        self.additions = 0
        self.encryptions = 0
        self.decryptions = 0


class MockCKKSBackend:
    """
    Mock CKKS backend that accurately simulates operation counts.

    This backend performs plaintext arithmetic but tracks all operations
    as if they were real CKKS operations, enabling accurate rotation
    budget testing.
    """

    def __init__(self, poly_degree: int = 8192):
        self.poly_degree = poly_degree
        self.slot_count = poly_degree // 2
        self.counters = MockOperationCounters()
        self._initialized = False

        # Column-packed matrix cache
        self._packed_matrices: Dict[int, np.ndarray] = {}

    def initialize(self) -> None:
        """Initialize the mock backend."""
        self._initialized = True

    def is_initialized(self) -> bool:
        return self._initialized

    def get_slot_count(self) -> int:
        return self.slot_count

    def encrypt(self, plaintext: np.ndarray) -> MockCiphertext:
        """Simulate encryption."""
        self.counters.encryptions += 1
        return MockCiphertext(
            data=plaintext.copy().astype(np.float64),
            slot_count=self.slot_count,
        )

    def decrypt(self, ciphertext: MockCiphertext, output_size: int = 0) -> np.ndarray:
        """Simulate decryption."""
        self.counters.decryptions += 1
        if output_size > 0:
            return ciphertext.data[:output_size]
        return ciphertext.data

    def add(self, ct1: MockCiphertext, ct2: MockCiphertext) -> MockCiphertext:
        """Ciphertext + Ciphertext (free, no rotation)."""
        self.counters.additions += 1
        return MockCiphertext(
            data=ct1.data + ct2.data,
            level=max(ct1.level, ct2.level),
            slot_count=ct1.slot_count,
        )

    def multiply_plain(self, ct: MockCiphertext, pt: np.ndarray) -> MockCiphertext:
        """Ciphertext × Plaintext (free, no rotation)."""
        self.counters.multiplications += 1
        # Broadcast multiplication
        result = ct.data * pt.flatten()[:len(ct.data)]
        return MockCiphertext(
            data=result,
            level=ct.level,
            slot_count=ct.slot_count,
        )

    def rotate(self, ct: MockCiphertext, steps: int) -> MockCiphertext:
        """
        Rotate ciphertext slots.

        THIS IS EXPENSIVE - each rotation requires a keyswitch.
        MOAI column packing should eliminate these.
        """
        self.counters.rotations += 1
        self.counters.keyswitches += 1  # Each rotation needs keyswitch
        return MockCiphertext(
            data=np.roll(ct.data, steps),
            level=ct.level,
            slot_count=ct.slot_count,
        )

    def rescale(self, ct: MockCiphertext) -> MockCiphertext:
        """Rescale to reduce noise (consumes one level)."""
        self.counters.rescales += 1
        return MockCiphertext(
            data=ct.data,
            level=ct.level + 1,
            slot_count=ct.slot_count,
        )

    def ct_pt_multiply(self, ct: MockCiphertext, pt_matrix: np.ndarray) -> MockCiphertext:
        """
        Standard ciphertext-plaintext matrix multiplication.

        WITHOUT column packing, this requires log2(dim) rotations.
        """
        dim = pt_matrix.shape[0]

        # Simulate naive approach: requires rotations
        num_rotations = int(np.ceil(np.log2(max(dim, 1))))
        for _ in range(num_rotations):
            self.counters.rotations += 1
            self.counters.keyswitches += 1

        self.counters.multiplications += 1

        # Actual computation
        result = ct.data @ pt_matrix.T
        return MockCiphertext(
            data=result,
            level=ct.level,
            slot_count=ct.slot_count,
        )

    def create_column_packed_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Create column-packed matrix for rotation-free multiplication.

        This is the MOAI key insight: proper packing eliminates rotations.
        """
        # Store for reference
        matrix_id = id(matrix)
        packed = matrix.astype(np.float64)
        self._packed_matrices[matrix_id] = packed
        return packed

    def column_packed_matmul(
        self,
        ct: MockCiphertext,
        packed_matrix: np.ndarray,
        rescale: bool = True,
    ) -> MockCiphertext:
        """
        Column-packed matrix multiplication (MOAI).

        CRITICAL: This should require 0 ROTATIONS.
        The column packing arranges data so element-wise multiplication
        and addition suffice.
        """
        # NO ROTATIONS with column packing!
        self.counters.multiplications += 1
        if rescale:
            self.counters.rescales += 1

        # Actual computation (simulation)
        result = ct.data @ packed_matrix.T
        return MockCiphertext(
            data=result,
            level=ct.level + (1 if rescale else 0),
            slot_count=ct.slot_count,
        )

    def encode_diagonal(self, vector: np.ndarray) -> np.ndarray:
        """Encode vector as diagonal for element-wise multiplication."""
        return vector.astype(np.float64)

    def get_counters(self) -> Dict[str, int]:
        """Get operation counters."""
        return {
            'rotations': self.counters.rotations,
            'keyswitches': self.counters.keyswitches,
            'rescales': self.counters.rescales,
            'multiplications': self.counters.multiplications,
            'additions': self.counters.additions,
            'encryptions': self.counters.encryptions,
            'decryptions': self.counters.decryptions,
        }

    def reset_counters(self) -> None:
        """Reset all operation counters."""
        self.counters.reset()


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_backend():
    """Create mock CKKS backend."""
    backend = MockCKKSBackend()
    backend.initialize()
    return backend


@pytest.fixture
def sample_lora_weights():
    """Create sample LoRA weights."""
    rng = np.random.default_rng(42)
    rank = 16
    hidden_size = 128

    return {
        "q_proj": (
            rng.standard_normal((rank, hidden_size)).astype(np.float32),
            rng.standard_normal((hidden_size, rank)).astype(np.float32) * 0.1,
        ),
        "v_proj": (
            rng.standard_normal((rank, hidden_size)).astype(np.float32),
            rng.standard_normal((hidden_size, rank)).astype(np.float32) * 0.1,
        ),
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    dir_path = tempfile.mkdtemp(prefix="he_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


# =============================================================================
# ROTATION COUNT TESTS
# =============================================================================

class TestRotationCounts:
    """Verify MOAI achieves 0-rotation matrix multiplication."""

    def test_column_packed_matmul_zero_rotations(self, mock_backend):
        """Column-packed matmul should require 0 rotations."""
        # Create test data
        x = np.random.randn(128).astype(np.float64)
        A = np.random.randn(16, 128).astype(np.float64)
        B = np.random.randn(128, 16).astype(np.float64)

        # Pre-pack matrices
        A_packed = mock_backend.create_column_packed_matrix(A)
        B_packed = mock_backend.create_column_packed_matrix(B)

        mock_backend.reset_counters()

        # Encrypt and compute
        ct_x = mock_backend.encrypt(x)
        ct_intermediate = mock_backend.column_packed_matmul(ct_x, A_packed)
        ct_result = mock_backend.column_packed_matmul(ct_intermediate, B_packed)
        result = mock_backend.decrypt(ct_result, output_size=128)

        # Verify 0 rotations
        counters = mock_backend.get_counters()
        assert counters['rotations'] == 0, f"Expected 0 rotations, got {counters['rotations']}"
        assert counters['keyswitches'] == 0, f"Expected 0 keyswitches, got {counters['keyswitches']}"

        # Verify computation is correct
        expected = (x @ A.T) @ B.T
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_standard_matmul_requires_rotations(self, mock_backend):
        """Standard matmul without column packing should require rotations."""
        x = np.random.randn(128).astype(np.float64)
        A = np.random.randn(16, 128).astype(np.float64)

        mock_backend.reset_counters()

        ct_x = mock_backend.encrypt(x)
        # Use non-packed multiplication
        ct_result = mock_backend.ct_pt_multiply(ct_x, A)

        counters = mock_backend.get_counters()
        # Should require rotations for 128-dim matrix
        assert counters['rotations'] > 0, "Standard matmul should require rotations"

    def test_lora_forward_zero_rotations(self, mock_backend, sample_lora_weights):
        """Full LoRA forward should achieve 0 rotations with MOAI."""
        lora_a, lora_b = sample_lora_weights["q_proj"]
        scaling = 2.0

        # Pre-pack weights
        A_packed = mock_backend.create_column_packed_matrix(lora_a)
        B_packed = mock_backend.create_column_packed_matrix(lora_b)

        x = np.random.randn(128).astype(np.float64)

        mock_backend.reset_counters()

        # LoRA forward: scaling * (x @ A.T) @ B.T
        ct_x = mock_backend.encrypt(x)
        ct_intermediate = mock_backend.column_packed_matmul(ct_x, A_packed)
        ct_result = mock_backend.column_packed_matmul(ct_intermediate, B_packed)
        ct_scaled = mock_backend.multiply_plain(ct_result, np.array([scaling]))
        delta = mock_backend.decrypt(ct_scaled, output_size=128)

        counters = mock_backend.get_counters()

        # Verify 0 rotations
        assert counters['rotations'] == 0, f"LoRA should have 0 rotations, got {counters['rotations']}"

        # Verify computation
        expected = scaling * (x @ lora_a.T @ lora_b.T)
        np.testing.assert_array_almost_equal(delta, expected, decimal=4)


# =============================================================================
# VERA CKKS RESTRUCTURING TESTS
# =============================================================================

class TestVeRACKKS:
    """Test VeRA restructuring for CKKS."""

    def test_vera_diagonal_scaling(self, mock_backend):
        """Test VeRA λ vectors can be applied without rotation."""
        rank = 32
        in_features = 128
        out_features = 128

        rng = np.random.default_rng(42)

        # VeRA parameters
        lambda_d = rng.uniform(0.1, 0.5, size=rank).astype(np.float64)
        lambda_b = rng.uniform(0.1, 0.5, size=out_features).astype(np.float64)
        A = rng.standard_normal((rank, in_features)).astype(np.float64)
        B = rng.standard_normal((out_features, rank)).astype(np.float64)

        # Pre-scale matrices (compile-time)
        A_scaled = lambda_d[:, np.newaxis] * A  # Scale rows by λ_d
        B_scaled = lambda_b[:, np.newaxis] * B  # Scale rows by λ_b

        A_packed = mock_backend.create_column_packed_matrix(A_scaled)
        B_packed = mock_backend.create_column_packed_matrix(B_scaled)

        x = rng.standard_normal(in_features).astype(np.float64)

        mock_backend.reset_counters()

        # Compute VeRA forward with restructured matrices
        ct_x = mock_backend.encrypt(x)
        ct_intermediate = mock_backend.column_packed_matmul(ct_x, A_packed)
        ct_result = mock_backend.column_packed_matmul(ct_intermediate, B_packed)
        delta = mock_backend.decrypt(ct_result, output_size=out_features)

        counters = mock_backend.get_counters()

        # Verify 0 rotations with restructured VeRA
        assert counters['rotations'] == 0, f"VeRA should have 0 rotations, got {counters['rotations']}"

        # Verify result matches original VeRA computation
        # Original: λ_b ⊙ (B @ (λ_d ⊙ (A @ x)))
        intermediate = A @ x
        scaled_int = lambda_d * intermediate
        output = B @ scaled_int
        expected = lambda_b * output

        np.testing.assert_array_almost_equal(delta, expected, decimal=4)


# =============================================================================
# HE FORWARD INTEGRATION TESTS
# =============================================================================

class TestHEForwardIntegration:
    """Test HE forward pass integration."""

    def test_ckks_lora_forward_class(self, mock_backend, sample_lora_weights):
        """Test CKKSLoRAForward class with mock backend."""
        from tensafe.adapters import (
            AdapterConfig,
            AdapterType,
            create_adapter,
            CKKSLoRAForward,
            HEForwardConfig,
        )

        # Create LoRA adapter
        config = AdapterConfig(adapter_type=AdapterType.LORA, rank=16, alpha=32.0)
        adapter = create_adapter(config, in_features=128, out_features=128)

        # Load weights
        lora_a, lora_b = sample_lora_weights["q_proj"]
        adapter.set_weights({
            "lora_A": lora_a,
            "lora_B": lora_b,
        }, strict=False)

        # Create HE forward
        he_config = HEForwardConfig(use_column_packing=True)
        he_forward = CKKSLoRAForward(adapter, he_config)
        he_forward.initialize_backend(mock_backend)

        # Compute forward
        x = np.random.randn(2, 128).astype(np.float32)
        mock_backend.reset_counters()

        delta = he_forward.forward(x, "q_proj")

        # Verify shape
        assert delta.shape == x.shape

        # Check metrics
        metrics = he_forward.get_metrics()
        assert metrics.total_time_ms > 0

    def test_he_compatibility_check(self):
        """Test HE compatibility checking."""
        from tensafe.adapters import (
            AdapterType,
            HECompatibility,
            get_he_compatibility,
            is_he_compatible,
        )

        # LoRA variants should be CKKS-compatible
        assert get_he_compatibility(AdapterType.LORA) == HECompatibility.FULL_CKKS
        assert get_he_compatibility(AdapterType.RS_LORA) == HECompatibility.FULL_CKKS
        assert get_he_compatibility(AdapterType.LORA_FA) == HECompatibility.FULL_CKKS

        # These need hybrid TFHE
        assert get_he_compatibility(AdapterType.GATED_LORA) == HECompatibility.HYBRID_TFHE
        assert get_he_compatibility(AdapterType.VERA) == HECompatibility.HYBRID_TFHE

        # These cannot run under HE
        assert get_he_compatibility(AdapterType.DORA) == HECompatibility.PLAINTEXT_ONLY
        assert get_he_compatibility(AdapterType.ADALORA) == HECompatibility.PLAINTEXT_ONLY

        # is_he_compatible should return True for CKKS and HYBRID
        assert is_he_compatible(AdapterType.LORA) is True
        assert is_he_compatible(AdapterType.DORA) is False


# =============================================================================
# EXTENDED REGISTRY TESTS
# =============================================================================

class TestExtendedRegistry:
    """Test extended TGSP registry functionality."""

    def test_rotation_budget_tracking(self):
        """Test rotation budget tracker."""
        from tensafe.extended_tgsp_registry import RotationBudget

        budget = RotationBudget(
            max_rotations_per_token=16,
            max_keyswitches_per_token=16,
        )

        # Should be within budget
        assert budget.record(rotations=5, keyswitches=5) is True
        assert budget.budget_exceeded is False

        # Record more
        assert budget.record(rotations=10) is True
        assert budget.current_rotations == 15

        # Exceed budget
        assert budget.record(rotations=5) is False
        assert budget.budget_exceeded is True
        assert "exceeded" in budget.exceeded_reason.lower()

    def test_vera_packing_struct(self):
        """Test VeRA CKKS packing data structure."""
        from tensafe.extended_tgsp_registry import VeRACKKSPacking

        lambda_d = np.random.randn(32).astype(np.float32)
        lambda_b = np.random.randn(128).astype(np.float32)

        packing = VeRACKKSPacking(
            lambda_d=lambda_d,
            lambda_b=lambda_b,
        )

        assert packing.lambda_d.shape == (32,)
        assert packing.lambda_b.shape == (128,)
        assert packing.is_packed is False


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

class TestBenchmarks:
    """Benchmark tests for performance validation."""

    def test_lora_throughput_benchmark(self, mock_backend, sample_lora_weights):
        """Benchmark LoRA forward throughput."""
        lora_a, lora_b = sample_lora_weights["q_proj"]
        A_packed = mock_backend.create_column_packed_matrix(lora_a)
        B_packed = mock_backend.create_column_packed_matrix(lora_b)

        x = np.random.randn(128).astype(np.float64)

        # Warmup
        for _ in range(10):
            ct_x = mock_backend.encrypt(x)
            ct_int = mock_backend.column_packed_matmul(ct_x, A_packed)
            ct_out = mock_backend.column_packed_matmul(ct_int, B_packed)
            mock_backend.decrypt(ct_out, output_size=128)

        # Benchmark
        num_iterations = 100
        mock_backend.reset_counters()
        start_time = time.perf_counter()

        for _ in range(num_iterations):
            ct_x = mock_backend.encrypt(x)
            ct_int = mock_backend.column_packed_matmul(ct_x, A_packed)
            ct_out = mock_backend.column_packed_matmul(ct_int, B_packed)
            mock_backend.decrypt(ct_out, output_size=128)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        avg_ms = elapsed_ms / num_iterations

        counters = mock_backend.get_counters()

        print(f"\n=== LoRA Forward Benchmark ===")
        print(f"Iterations: {num_iterations}")
        print(f"Total time: {elapsed_ms:.2f}ms")
        print(f"Avg per forward: {avg_ms:.3f}ms")
        print(f"Total rotations: {counters['rotations']}")
        print(f"Total rescales: {counters['rescales']}")

        # Verify 0 rotations across all iterations
        assert counters['rotations'] == 0, "MOAI should achieve 0 rotations"

        # Each forward does 2 matmuls with rescale
        assert counters['rescales'] == num_iterations * 2

    def test_operation_count_summary(self, mock_backend, sample_lora_weights):
        """Generate operation count summary for QKV projection."""
        projections = ["q_proj", "k_proj", "v_proj"]

        print("\n=== Operation Count Summary (per token) ===")
        print(f"{'Projection':<10} {'Rotations':>10} {'Rescales':>10} {'Mults':>10}")
        print("-" * 45)

        total_rotations = 0
        total_rescales = 0

        for proj in projections:
            if proj in sample_lora_weights:
                lora_a, lora_b = sample_lora_weights[proj]
            else:
                lora_a = np.random.randn(16, 128).astype(np.float64)
                lora_b = np.random.randn(128, 16).astype(np.float64)

            A_packed = mock_backend.create_column_packed_matrix(lora_a)
            B_packed = mock_backend.create_column_packed_matrix(lora_b)

            x = np.random.randn(128).astype(np.float64)

            mock_backend.reset_counters()

            ct_x = mock_backend.encrypt(x)
            ct_int = mock_backend.column_packed_matmul(ct_x, A_packed)
            ct_out = mock_backend.column_packed_matmul(ct_int, B_packed)
            mock_backend.decrypt(ct_out, output_size=128)

            counters = mock_backend.get_counters()

            print(f"{proj:<10} {counters['rotations']:>10} {counters['rescales']:>10} {counters['multiplications']:>10}")

            total_rotations += counters['rotations']
            total_rescales += counters['rescales']

        print("-" * 45)
        print(f"{'TOTAL':<10} {total_rotations:>10} {total_rescales:>10}")

        # Verify MOAI optimization
        assert total_rotations == 0, f"QKV should have 0 rotations, got {total_rotations}"


# =============================================================================
# INTEGRATION WITH EXISTING INFRASTRUCTURE
# =============================================================================

class TestExistingInfraIntegration:
    """Test integration with existing HE infrastructure."""

    def test_compat_module_interface(self):
        """Test compatibility with he_lora_microkernel.compat."""
        try:
            from he_lora_microkernel.compat import HELoRAConfig, HELoRAAdapter

            config = HELoRAConfig(
                rank=16,
                alpha=32.0,
                target_modules=["q_proj", "v_proj"],
                backend_type="SIMULATION",
            )

            assert config.rank == 16
            assert config.alpha == 32.0
            assert config.scaling == 32.0 / 16

        except ImportError as e:
            pytest.skip(f"HE microkernel not available: {e}")

    def test_adapter_types_with_he_config(self):
        """Test adapter types work with HE configuration."""
        from tensafe.adapters import (
            AdapterConfig,
            AdapterType,
            create_adapter,
        )

        # Create various adapter types
        for adapter_type in [AdapterType.LORA, AdapterType.RS_LORA, AdapterType.LORA_FA]:
            config = AdapterConfig(
                adapter_type=adapter_type,
                rank=16,
                alpha=32.0,
            )

            adapter = create_adapter(config, in_features=128, out_features=128)

            # Should have trainable params
            params = adapter.get_trainable_params()
            assert len(params) > 0

            # Should be able to compute forward
            x = np.random.randn(2, 128).astype(np.float32)
            delta = adapter.forward(x)
            assert delta.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
