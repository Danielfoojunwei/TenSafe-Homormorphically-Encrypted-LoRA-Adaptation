"""
End-to-End Tests for HintSight N2HE Integration.

These tests verify the complete flow of the HintSight N2HE backend:
1. Backend initialization and key generation
2. Encrypt/decrypt roundtrip accuracy
3. Homomorphic operations (add, multiply, matmul)
4. LoRA delta computation
5. Integration with the unified HE interface
6. Performance characteristics

Run with:
    pytest tests/e2e/test_hintsight_e2e.py -v
    pytest tests/e2e/test_hintsight_e2e.py -v -k "lora"  # Just LoRA tests
"""

import os
import sys
import time
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _hintsight_available() -> bool:
    """Check if HintSight backend is available."""
    try:
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend, _NATIVE_AVAILABLE
        return _NATIVE_AVAILABLE
    except ImportError:
        return False


# Skip all tests if HintSight backend is not available
pytestmark = pytest.mark.skipif(
    not _hintsight_available(),
    reason="HintSight N2HE backend not built. Run: ./scripts/build_n2he_hintsight.sh"
)


@pytest.fixture
def backend():
    """Create and initialize a HintSight backend."""
    from crypto_backend.n2he_hintsight import N2HEHintSightBackend, N2HEParams

    params = N2HEParams.default_lora_params()
    backend = N2HEHintSightBackend(params)
    backend.setup_context()
    backend.generate_keys()
    return backend


@pytest.fixture
def he_interface_backend():
    """Create backend via the unified HE interface."""
    from tensafe.core.he_interface import get_backend, HEBackendType

    return get_backend(HEBackendType.HINTSIGHT)


class TestHintSightBackendBasics:
    """Test basic HintSight backend functionality."""

    def test_backend_available(self):
        """Test that the backend is available."""
        from crypto_backend.n2he_hintsight import _NATIVE_AVAILABLE, _NATIVE_MODULE

        assert _NATIVE_AVAILABLE, "Native module should be available"
        assert _NATIVE_MODULE is not None, "Native module should be loaded"
        assert hasattr(_NATIVE_MODULE, "__version__"), "Module should have version"
        assert hasattr(_NATIVE_MODULE, "BACKEND_NAME"), "Module should have backend name"

    def test_context_creation(self):
        """Test context creation and parameters."""
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend, N2HEParams

        params = N2HEParams(
            n=1024,
            q=2**32,
            t=2**16,
            std_dev=3.2,
            security_level=128
        )

        backend = N2HEHintSightBackend(params)
        assert backend.get_backend_name() == "HintSight-N2HE"

        backend.setup_context()
        ctx_params = backend.get_context_params()

        assert ctx_params["n"] == 1024
        assert ctx_params["security_level"] == 128

    def test_key_generation(self, backend):
        """Test key generation returns valid keys."""
        from crypto_backend.n2he_hintsight import N2HEHintSightBackend

        # Keys are already generated in fixture, but verify they exist
        assert backend._native_ctx.keys_generated

        # Generate fresh keys and verify
        backend2 = N2HEHintSightBackend()
        backend2.setup_context()
        sk, pk, ek = backend2.generate_keys()

        assert isinstance(sk, bytes)
        assert isinstance(pk, bytes)
        assert isinstance(ek, bytes)
        assert len(sk) > 0
        assert len(pk) > 0
        assert len(ek) > 0


class TestHintSightEncryptDecrypt:
    """Test encrypt/decrypt operations."""

    @pytest.mark.parametrize("data", [
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([0.5, -0.5, 0.25, -0.25]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.random.randn(8).astype(np.float64),
        np.random.randn(16).astype(np.float64),
    ])
    def test_encrypt_decrypt_roundtrip(self, backend, data):
        """Test encrypt/decrypt roundtrip accuracy."""
        ct = backend.encrypt(data)
        decrypted = backend.decrypt(ct, len(data))

        # LWE has higher error than CKKS
        error = np.max(np.abs(data - decrypted))
        assert error < 0.1, f"Error too high: {error}"

    def test_ciphertext_properties(self, backend):
        """Test ciphertext metadata."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(data)

        # Check noise budget
        noise_budget = ct.noise_budget
        assert noise_budget > 0, "Noise budget should be positive"
        assert noise_budget <= 100, "Noise budget should be reasonable"

        # Check level
        level = ct.level
        assert level >= 0, "Level should be non-negative"

    def test_ciphertext_serialization(self, backend):
        """Test ciphertext serialization."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(data)

        # Serialize
        serialized = ct.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

        # Deserialize
        from crypto_backend.n2he_hintsight import N2HECiphertext
        ct_restored = N2HECiphertext.deserialize(serialized, backend)

        # Decrypt and verify
        decrypted = backend.decrypt(ct_restored, len(data))
        error = np.max(np.abs(data - decrypted))
        assert error < 0.1, f"Error after deserialization: {error}"


class TestHintSightHomomorphicOps:
    """Test homomorphic operations."""

    def test_addition(self, backend):
        """Test homomorphic addition."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.0, 1.5, 2.0])

        ct_a = backend.encrypt(a)
        ct_b = backend.encrypt(b)

        ct_sum = backend.add(ct_a, ct_b)
        decrypted = backend.decrypt(ct_sum, len(a))

        expected = a + b
        error = np.max(np.abs(expected - decrypted))
        assert error < 0.15, f"Addition error too high: {error}"

    def test_scalar_multiplication(self, backend):
        """Test multiplication by scalar."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        scalar = 2.0

        ct_a = backend.encrypt(a)
        ct_scaled = backend.multiply_plain(ct_a, np.array([scalar]))
        decrypted = backend.decrypt(ct_scaled, len(a))

        expected = a * scalar
        error = np.max(np.abs(expected - decrypted))
        assert error < 0.2, f"Scalar multiplication error too high: {error}"

    def test_element_wise_multiplication(self, backend):
        """Test element-wise multiplication."""
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([0.5, 1.0, 1.5, 2.0])

        ct_a = backend.encrypt(a)
        ct_mult = backend.multiply_plain(ct_a, b)
        decrypted = backend.decrypt(ct_mult, len(a))

        expected = a * b
        error = np.max(np.abs(expected - decrypted))
        assert error < 0.25, f"Element-wise multiplication error too high: {error}"


class TestHintSightMatMul:
    """Test matrix multiplication."""

    @pytest.mark.parametrize("in_dim,out_dim", [
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8),
    ])
    def test_matmul(self, backend, in_dim, out_dim):
        """Test encrypted matrix multiplication."""
        x = np.random.randn(in_dim).astype(np.float64)
        W = np.random.randn(out_dim, in_dim).astype(np.float64) * 0.1

        ct_x = backend.encrypt(x)
        ct_result = backend.matmul(ct_x, W)
        decrypted = backend.decrypt(ct_result, out_dim)

        expected = x @ W.T
        error = np.max(np.abs(expected - decrypted))
        assert error < 0.5, f"Matmul error too high: {error}"


class TestHintSightLoRADelta:
    """Test LoRA delta computation."""

    @pytest.mark.parametrize("in_dim,rank,out_dim", [
        (4, 8, 4),
        (8, 16, 8),
        (16, 8, 16),
    ])
    def test_lora_delta(self, backend, in_dim, rank, out_dim):
        """Test LoRA delta computation: scaling * (x @ A^T @ B^T)."""
        x = np.random.randn(in_dim).astype(np.float64)
        lora_a = np.random.randn(rank, in_dim).astype(np.float64) * 0.1
        lora_b = np.random.randn(out_dim, rank).astype(np.float64) * 0.1
        scaling = 0.5

        ct_x = backend.encrypt(x)
        ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, scaling)
        decrypted = backend.decrypt(ct_delta, out_dim)

        expected = scaling * (x @ lora_a.T @ lora_b.T)
        error = np.max(np.abs(expected - decrypted))
        assert error < 1.0, f"LoRA delta error too high: {error}"

    @pytest.mark.parametrize("scaling", [0.25, 0.5, 1.0, 2.0])
    def test_lora_delta_scaling(self, backend, scaling):
        """Test LoRA delta with different scaling factors."""
        x = np.random.randn(4).astype(np.float64)
        lora_a = np.random.randn(8, 4).astype(np.float64) * 0.1
        lora_b = np.random.randn(4, 8).astype(np.float64) * 0.1

        ct_x = backend.encrypt(x)
        ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, scaling)
        decrypted = backend.decrypt(ct_delta, 4)

        expected = scaling * (x @ lora_a.T @ lora_b.T)
        error = np.max(np.abs(expected - decrypted))
        assert error < 1.0, f"LoRA delta error with scaling={scaling}: {error}"


class TestHintSightHEInterface:
    """Test integration with unified HE interface."""

    def test_backend_type(self, he_interface_backend):
        """Test backend type is correct."""
        from tensafe.core.he_interface import HEBackendType

        assert he_interface_backend.backend_type == HEBackendType.HINTSIGHT

    def test_backend_name(self, he_interface_backend):
        """Test backend name."""
        assert he_interface_backend.backend_name == "HintSight-N2HE"

    def test_is_production_ready(self, he_interface_backend):
        """Test production readiness flag."""
        assert he_interface_backend.is_production_ready is True

    def test_encrypt_decrypt_via_interface(self, he_interface_backend):
        """Test encrypt/decrypt via unified interface."""
        data = np.array([1.0, 2.0, 3.0, 4.0])

        ct = he_interface_backend.encrypt(data)
        decrypted = he_interface_backend.decrypt(ct, len(data))

        error = np.max(np.abs(data - decrypted))
        assert error < 0.1, f"Error via interface: {error}"

    def test_lora_delta_via_interface(self, he_interface_backend):
        """Test LoRA delta via unified interface."""
        x = np.random.randn(4).astype(np.float64)
        lora_a = np.random.randn(8, 4).astype(np.float64) * 0.1
        lora_b = np.random.randn(4, 8).astype(np.float64) * 0.1
        scaling = 0.5

        ct_x = he_interface_backend.encrypt(x)
        ct_delta = he_interface_backend.lora_delta(ct_x, lora_a, lora_b, scaling)
        decrypted = he_interface_backend.decrypt(ct_delta, 4)

        expected = scaling * (x @ lora_a.T @ lora_b.T)
        error = np.max(np.abs(expected - decrypted))
        assert error < 1.0, f"LoRA delta error via interface: {error}"


class TestHintSightPerformance:
    """Test performance characteristics."""

    def test_encrypt_performance(self, backend):
        """Test encryption performance."""
        data = np.random.randn(64).astype(np.float64)
        iterations = 10

        start = time.time()
        for _ in range(iterations):
            ct = backend.encrypt(data)
        total_time = time.time() - start

        avg_time_ms = (total_time / iterations) * 1000
        # Should complete in reasonable time (< 1 second per encryption)
        assert avg_time_ms < 1000, f"Encryption too slow: {avg_time_ms:.1f} ms"

    def test_lora_delta_performance(self, backend):
        """Test LoRA delta performance."""
        x = np.random.randn(64).astype(np.float64)
        lora_a = np.random.randn(16, 64).astype(np.float64) * 0.1
        lora_b = np.random.randn(64, 16).astype(np.float64) * 0.1

        ct_x = backend.encrypt(x)
        iterations = 5

        start = time.time()
        for _ in range(iterations):
            ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, 0.5)
        total_time = time.time() - start

        avg_time_ms = (total_time / iterations) * 1000
        # LoRA delta should complete in reasonable time
        assert avg_time_ms < 10000, f"LoRA delta too slow: {avg_time_ms:.1f} ms"

    def test_operation_stats(self, backend):
        """Test operation statistics tracking."""
        backend.reset_stats()

        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(x)
        _ = backend.decrypt(ct, 4)

        stats = backend.get_operation_stats()
        assert "operations" in stats
        assert stats["operations"] > 0


class TestHintSightNoiseBudget:
    """Test noise budget tracking."""

    def test_noise_budget_decreases(self, backend):
        """Test that noise budget decreases after operations."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(x)

        initial_budget = ct.noise_budget

        # Perform operation
        ct_scaled = backend.multiply_plain(ct, np.array([2.0]))
        after_budget = ct_scaled.noise_budget

        # Noise budget should decrease (or stay same in some implementations)
        assert after_budget <= initial_budget, "Noise budget should not increase"

    def test_noise_budget_after_lora(self, backend):
        """Test noise budget after LoRA delta."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        lora_a = np.random.randn(8, 4).astype(np.float64) * 0.1
        lora_b = np.random.randn(4, 8).astype(np.float64) * 0.1

        ct_x = backend.encrypt(x)
        initial_budget = ct_x.noise_budget

        ct_delta = backend.lora_delta(ct_x, lora_a, lora_b, 0.5)
        final_budget = ct_delta.noise_budget

        # Budget should decrease after multiple operations
        assert final_budget < initial_budget, "Noise budget should decrease after LoRA"


class TestHintSightAutoSelection:
    """Test auto-selection of HintSight backend."""

    def test_auto_selects_hintsight(self):
        """Test that AUTO selects HintSight when available."""
        from tensafe.core.he_interface import get_backend, HEBackendType

        # Get backend with AUTO
        backend = get_backend(HEBackendType.AUTO)

        # Should select HintSight as first priority
        assert backend.backend_type == HEBackendType.HINTSIGHT

    def test_is_backend_available(self):
        """Test backend availability check."""
        from tensafe.core.he_interface import is_backend_available, HEBackendType

        assert is_backend_available(HEBackendType.HINTSIGHT)


# Benchmark tests (run with pytest --benchmark)
class TestHintSightBenchmarks:
    """Benchmark tests for HintSight backend."""

    @pytest.mark.benchmark
    def test_benchmark_encrypt(self, backend, benchmark):
        """Benchmark encryption."""
        data = np.random.randn(64).astype(np.float64)

        def encrypt():
            return backend.encrypt(data)

        benchmark(encrypt)

    @pytest.mark.benchmark
    def test_benchmark_decrypt(self, backend, benchmark):
        """Benchmark decryption."""
        data = np.random.randn(64).astype(np.float64)
        ct = backend.encrypt(data)

        def decrypt():
            return backend.decrypt(ct, 64)

        benchmark(decrypt)

    @pytest.mark.benchmark
    def test_benchmark_lora_delta(self, backend, benchmark):
        """Benchmark LoRA delta computation."""
        x = np.random.randn(64).astype(np.float64)
        lora_a = np.random.randn(16, 64).astype(np.float64) * 0.1
        lora_b = np.random.randn(64, 16).astype(np.float64) * 0.1
        ct_x = backend.encrypt(x)

        def lora_delta():
            return backend.lora_delta(ct_x, lora_a, lora_b, 0.5)

        benchmark(lora_delta)
