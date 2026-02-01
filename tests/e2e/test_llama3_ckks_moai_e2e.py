"""
E2E Test: Llama3 LoRA with CKKS MOAI Encrypted Inference.

This test verifies end-to-end functionality of the CKKS MOAI backend
(based on MOAI paper from Digital Trust Center NTU) with Llama3-style
LoRA computations, including:

1. Encrypted activation processing with CKKS approximate arithmetic
2. LoRA delta computation under HE with column packing optimization
3. Scaling and output generation
4. Performance benchmarking

The test simulates the encrypted inference path where:
- Client encrypts activations using CKKS
- Server computes LoRA delta under encryption (without rotations when possible)
- Client decrypts and applies delta

Key MOAI features tested:
- CKKS scheme for approximate float arithmetic
- Column packing for efficient matrix multiplication
- Consistent packing across layers

Run with:
    pytest tests/e2e/test_llama3_ckks_moai_e2e.py -v
    pytest tests/e2e/test_llama3_ckks_moai_e2e.py -v -k "benchmark" --benchmark
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _ckks_moai_available() -> bool:
    """Check if CKKS MOAI backend is available."""
    try:
        from crypto_backend.ckks_moai import CKKSMOAIBackend
        from Pyfhel import Pyfhel
        return True
    except ImportError:
        return False


# Skip all tests if CKKS MOAI backend is not available
pytestmark = pytest.mark.skipif(
    not _ckks_moai_available(),
    reason="CKKS MOAI backend not available. Install with: pip install Pyfhel"
)


@dataclass
class Llama3LoRAConfig:
    """Configuration matching Llama3-8B LoRA setup."""
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_scaling: float = 2.0  # alpha / rank

    # Reduced dimensions for testing (full size would be too slow)
    test_hidden_size: int = 64
    test_lora_rank: int = 8


@dataclass
class HEBenchmarkMetrics:
    """Metrics for HE operations."""
    keygen_time_ms: float = 0.0
    encrypt_times_ms: List[float] = field(default_factory=list)
    decrypt_times_ms: List[float] = field(default_factory=list)
    lora_delta_times_ms: List[float] = field(default_factory=list)
    matmul_times_ms: List[float] = field(default_factory=list)

    total_operations: int = 0
    total_multiplications: int = 0
    total_additions: int = 0
    total_rotations: int = 0
    total_rescales: int = 0

    max_error: float = 0.0
    avg_error: float = 0.0

    def to_report(self) -> Dict[str, Any]:
        """Generate metrics report."""
        def stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"mean": 0, "min": 0, "max": 0, "p95": 0}
            sorted_data = sorted(data)
            p95_idx = int(len(sorted_data) * 0.95)
            return {
                "mean": np.mean(data),
                "min": np.min(data),
                "max": np.max(data),
                "p95": sorted_data[p95_idx] if p95_idx < len(sorted_data) else sorted_data[-1],
            }

        return {
            "keygen_time_ms": self.keygen_time_ms,
            "encrypt_stats_ms": stats(self.encrypt_times_ms),
            "decrypt_stats_ms": stats(self.decrypt_times_ms),
            "lora_delta_stats_ms": stats(self.lora_delta_times_ms),
            "matmul_stats_ms": stats(self.matmul_times_ms),
            "total_operations": self.total_operations,
            "total_multiplications": self.total_multiplications,
            "total_additions": self.total_additions,
            "total_rotations": self.total_rotations,
            "total_rescales": self.total_rescales,
            "accuracy": {
                "max_error": self.max_error,
                "avg_error": self.avg_error,
            }
        }


class MockLlama3Layer:
    """
    Mock Llama3 transformer layer for testing HE-LoRA.

    Simulates the attention and MLP components with LoRA adapters.
    """

    def __init__(self, config: Llama3LoRAConfig):
        self.config = config

        # Use test dimensions
        d = config.test_hidden_size
        r = config.test_lora_rank

        # Initialize LoRA weights for q, k, v, o projections
        self.lora_weights = {
            "q_proj": {
                "A": np.random.randn(r, d).astype(np.float64) * 0.01,
                "B": np.random.randn(d, r).astype(np.float64) * 0.01,
            },
            "k_proj": {
                "A": np.random.randn(r, d).astype(np.float64) * 0.01,
                "B": np.random.randn(d, r).astype(np.float64) * 0.01,
            },
            "v_proj": {
                "A": np.random.randn(r, d).astype(np.float64) * 0.01,
                "B": np.random.randn(d, r).astype(np.float64) * 0.01,
            },
            "o_proj": {
                "A": np.random.randn(r, d).astype(np.float64) * 0.01,
                "B": np.random.randn(d, r).astype(np.float64) * 0.01,
            },
        }

    def compute_lora_delta_plaintext(
        self,
        x: np.ndarray,
        proj_name: str,
    ) -> np.ndarray:
        """Compute LoRA delta in plaintext (for comparison)."""
        A = self.lora_weights[proj_name]["A"]
        B = self.lora_weights[proj_name]["B"]
        scaling = self.config.lora_scaling

        # delta = scaling * (x @ A^T @ B^T)
        delta = scaling * (x @ A.T @ B.T)
        return delta


class HELoRAInferenceClient:
    """
    Client-side HE-LoRA inference handler.

    Handles encryption of activations and decryption of results.
    """

    def __init__(self, backend):
        self.backend = backend
        self.metrics = HEBenchmarkMetrics()

    def encrypt_activation(self, x: np.ndarray) -> Any:
        """Encrypt activation vector."""
        start = time.perf_counter()
        ct = self.backend.encrypt(x)
        self.metrics.encrypt_times_ms.append((time.perf_counter() - start) * 1000)
        return ct

    def decrypt_delta(self, ct_delta: Any, output_size: int) -> np.ndarray:
        """Decrypt LoRA delta."""
        start = time.perf_counter()
        result = self.backend.decrypt(ct_delta, output_size)
        self.metrics.decrypt_times_ms.append((time.perf_counter() - start) * 1000)
        return result


class HELoRAInferenceServer:
    """
    Server-side HE-LoRA inference handler.

    Computes LoRA delta under encryption without seeing plaintext.
    """

    def __init__(self, backend, layer: MockLlama3Layer):
        self.backend = backend
        self.layer = layer
        self.metrics = HEBenchmarkMetrics()

    def compute_encrypted_lora_delta(
        self,
        ct_x: Any,
        proj_name: str,
    ) -> Any:
        """
        Compute LoRA delta under encryption.

        Server computes: ct_delta = scaling * (ct_x @ A^T @ B^T)
        without decrypting ct_x.
        """
        A = self.layer.lora_weights[proj_name]["A"]
        B = self.layer.lora_weights[proj_name]["B"]
        scaling = self.layer.config.lora_scaling

        start = time.perf_counter()
        ct_delta = self.backend.lora_delta(ct_x, A, B, scaling)
        self.metrics.lora_delta_times_ms.append((time.perf_counter() - start) * 1000)

        return ct_delta


@pytest.fixture
def he_backend():
    """Create and initialize CKKS MOAI backend."""
    from crypto_backend.ckks_moai import CKKSMOAIBackend, CKKSParams

    params = CKKSParams.default_lora_params()
    backend = CKKSMOAIBackend(params)

    start = time.perf_counter()
    backend.setup_context()
    backend.generate_keys()
    keygen_time = (time.perf_counter() - start) * 1000

    # Store keygen time for metrics
    backend._keygen_time_ms = keygen_time

    return backend


@pytest.fixture
def llama3_config():
    """Llama3 LoRA configuration."""
    return Llama3LoRAConfig()


@pytest.fixture
def mock_layer(llama3_config):
    """Create mock Llama3 layer."""
    return MockLlama3Layer(llama3_config)


class TestLlama3HELoRABasics:
    """Basic tests for Llama3 HE-LoRA with CKKS MOAI."""

    def test_single_lora_delta(self, he_backend, mock_layer):
        """Test single LoRA delta computation."""
        d = mock_layer.config.test_hidden_size

        # Create test activation
        x = np.random.randn(d).astype(np.float64)

        # Compute plaintext delta
        expected = mock_layer.compute_lora_delta_plaintext(x, "q_proj")

        # Compute encrypted delta
        ct_x = he_backend.encrypt(x)
        A = mock_layer.lora_weights["q_proj"]["A"]
        B = mock_layer.lora_weights["q_proj"]["B"]
        scaling = mock_layer.config.lora_scaling

        ct_delta = he_backend.lora_delta(ct_x, A, B, scaling)
        decrypted = he_backend.decrypt(ct_delta, d)

        # Check accuracy - CKKS should have much lower error than LWE
        error = np.max(np.abs(expected - decrypted))
        assert error < 0.01, f"LoRA delta error too high: {error}"

    def test_all_projections(self, he_backend, mock_layer):
        """Test LoRA delta for all projection types."""
        d = mock_layer.config.test_hidden_size
        x = np.random.randn(d).astype(np.float64)

        for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            expected = mock_layer.compute_lora_delta_plaintext(x, proj_name)

            ct_x = he_backend.encrypt(x)
            A = mock_layer.lora_weights[proj_name]["A"]
            B = mock_layer.lora_weights[proj_name]["B"]
            scaling = mock_layer.config.lora_scaling

            ct_delta = he_backend.lora_delta(ct_x, A, B, scaling)
            decrypted = he_backend.decrypt(ct_delta, d)

            error = np.max(np.abs(expected - decrypted))
            assert error < 0.01, f"{proj_name} error too high: {error}"

    def test_ckks_precision(self, he_backend):
        """Test CKKS provides high precision for floats."""
        # Test various float values
        test_vectors = [
            np.array([0.123456789, 0.987654321, -0.555555, 0.111111]),
            np.array([1e-6, 1e-4, 1e-2, 1.0]),
            np.random.randn(16) * 0.01,
        ]

        for x in test_vectors:
            ct = he_backend.encrypt(x)
            decrypted = he_backend.decrypt(ct, len(x))

            error = np.max(np.abs(x - decrypted))
            assert error < 1e-6, f"CKKS precision error: {error}"


class TestLlama3HELoRAClientServer:
    """Test client-server HE-LoRA workflow."""

    def test_client_server_workflow(self, he_backend, mock_layer):
        """Test complete client-server workflow."""
        d = mock_layer.config.test_hidden_size

        # Initialize client and server
        client = HELoRAInferenceClient(he_backend)
        server = HELoRAInferenceServer(he_backend, mock_layer)

        # Client: encrypt activation
        x = np.random.randn(d).astype(np.float64)
        ct_x = client.encrypt_activation(x)

        # Server: compute encrypted delta
        ct_delta = server.compute_encrypted_lora_delta(ct_x, "q_proj")

        # Client: decrypt delta
        delta = client.decrypt_delta(ct_delta, d)

        # Verify
        expected = mock_layer.compute_lora_delta_plaintext(x, "q_proj")
        error = np.max(np.abs(expected - delta))
        assert error < 0.01, f"Client-server error: {error}"

    def test_batch_processing(self, he_backend, mock_layer):
        """Test processing multiple activations."""
        d = mock_layer.config.test_hidden_size
        batch_size = 4

        client = HELoRAInferenceClient(he_backend)
        server = HELoRAInferenceServer(he_backend, mock_layer)

        errors = []
        for i in range(batch_size):
            x = np.random.randn(d).astype(np.float64)
            ct_x = client.encrypt_activation(x)
            ct_delta = server.compute_encrypted_lora_delta(ct_x, "q_proj")
            delta = client.decrypt_delta(ct_delta, d)

            expected = mock_layer.compute_lora_delta_plaintext(x, "q_proj")
            errors.append(np.max(np.abs(expected - delta)))

        avg_error = np.mean(errors)
        max_error = np.max(errors)

        assert max_error < 0.01, f"Batch max error too high: {max_error}"
        assert avg_error < 0.005, f"Batch avg error too high: {avg_error}"


class TestLlama3HELoRABenchmark:
    """Benchmark tests for Llama3 HE-LoRA with CKKS MOAI."""

    @pytest.mark.benchmark
    def test_benchmark_full_layer(self, he_backend, mock_layer):
        """Benchmark full layer LoRA computation."""
        d = mock_layer.config.test_hidden_size
        iterations = 10

        metrics = HEBenchmarkMetrics()
        metrics.keygen_time_ms = he_backend._keygen_time_ms

        errors = []

        for i in range(iterations):
            x = np.random.randn(d).astype(np.float64)

            # Time full workflow
            total_start = time.perf_counter()

            # Encrypt
            enc_start = time.perf_counter()
            ct_x = he_backend.encrypt(x)
            metrics.encrypt_times_ms.append((time.perf_counter() - enc_start) * 1000)

            # Compute all projections
            deltas = {}
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                lora_start = time.perf_counter()

                A = mock_layer.lora_weights[proj_name]["A"]
                B = mock_layer.lora_weights[proj_name]["B"]
                scaling = mock_layer.config.lora_scaling

                ct_delta = he_backend.lora_delta(ct_x, A, B, scaling)
                metrics.lora_delta_times_ms.append((time.perf_counter() - lora_start) * 1000)

                # Decrypt
                dec_start = time.perf_counter()
                deltas[proj_name] = he_backend.decrypt(ct_delta, d)
                metrics.decrypt_times_ms.append((time.perf_counter() - dec_start) * 1000)

                # Check error
                expected = mock_layer.compute_lora_delta_plaintext(x, proj_name)
                errors.append(np.max(np.abs(expected - deltas[proj_name])))

        # Update metrics
        stats = he_backend.get_operation_stats()
        metrics.total_operations = stats.get("operations", 0)
        metrics.total_multiplications = stats.get("multiplications", 0)
        metrics.total_additions = stats.get("additions", 0)
        metrics.total_rotations = stats.get("rotations", 0)
        metrics.total_rescales = stats.get("rescales", 0)
        metrics.max_error = np.max(errors)
        metrics.avg_error = np.mean(errors)

        # Print report
        report = metrics.to_report()
        print("\n" + "=" * 60)
        print("CKKS MOAI Benchmark Results")
        print("=" * 60)
        print(f"Iterations: {iterations}")
        print(f"Hidden size: {d}")
        print(f"LoRA rank: {mock_layer.config.test_lora_rank}")
        print(f"\nKey Generation: {report['keygen_time_ms']:.2f} ms")
        print(f"\nEncrypt (per vector):")
        print(f"  Mean: {report['encrypt_stats_ms']['mean']:.2f} ms")
        print(f"  P95:  {report['encrypt_stats_ms']['p95']:.2f} ms")
        print(f"\nLoRA Delta (per projection):")
        print(f"  Mean: {report['lora_delta_stats_ms']['mean']:.2f} ms")
        print(f"  P95:  {report['lora_delta_stats_ms']['p95']:.2f} ms")
        print(f"\nDecrypt (per vector):")
        print(f"  Mean: {report['decrypt_stats_ms']['mean']:.2f} ms")
        print(f"  P95:  {report['decrypt_stats_ms']['p95']:.2f} ms")
        print(f"\nOperations:")
        print(f"  Total: {report['total_operations']}")
        print(f"  Multiplications: {report['total_multiplications']}")
        print(f"  Additions: {report['total_additions']}")
        print(f"  Rotations: {report['total_rotations']}")
        print(f"  Rescales: {report['total_rescales']}")
        print(f"\nAccuracy:")
        print(f"  Max Error: {report['accuracy']['max_error']:.8f}")
        print(f"  Avg Error: {report['accuracy']['avg_error']:.8f}")
        print("=" * 60)

        # Save report
        reports_dir = PROJECT_ROOT / "reports" / "bench"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"ckks_moai_llama3_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, "w") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "backend": "CKKS-MOAI",
                "config": {
                    "hidden_size": d,
                    "lora_rank": mock_layer.config.test_lora_rank,
                    "iterations": iterations,
                },
                "metrics": report,
            }, f, indent=2)

        print(f"\nReport saved to: {report_path}")

        # Assertions - CKKS should have much lower error than LWE
        assert metrics.max_error < 0.01, f"Max error too high: {metrics.max_error}"
        assert metrics.avg_error < 0.005, f"Avg error too high: {metrics.avg_error}"


class TestLlama3HELoRAIntegration:
    """Integration tests with unified HE interface."""

    def test_via_he_interface(self, mock_layer):
        """Test via unified HE interface."""
        from tensafe.core.he_interface import get_backend, HEBackendType

        backend = get_backend(HEBackendType.CKKS_MOAI)

        d = mock_layer.config.test_hidden_size
        x = np.random.randn(d).astype(np.float64)

        # Test through interface
        ct_x = backend.encrypt(x)
        A = mock_layer.lora_weights["q_proj"]["A"]
        B = mock_layer.lora_weights["q_proj"]["B"]
        scaling = mock_layer.config.lora_scaling

        ct_delta = backend.lora_delta(ct_x, A, B, scaling)
        delta = backend.decrypt(ct_delta, d)

        expected = mock_layer.compute_lora_delta_plaintext(x, "q_proj")
        error = np.max(np.abs(expected - delta))

        assert error < 0.01, f"Interface error: {error}"

    def test_auto_select_backend(self, mock_layer):
        """Test auto-selection chooses CKKS MOAI as first priority."""
        from tensafe.core.he_interface import get_backend, HEBackendType

        backend = get_backend(HEBackendType.AUTO)

        # Should select CKKS MOAI as first priority
        assert backend.backend_type == HEBackendType.CKKS_MOAI
        assert backend.is_production_ready


class TestLlama3HELoRAScaling:
    """Test scaling behavior and larger dimensions."""

    @pytest.mark.parametrize("hidden_size", [32, 64, 128])
    def test_varying_hidden_size(self, he_backend, hidden_size):
        """Test with different hidden sizes."""
        rank = 8
        scaling = 2.0

        x = np.random.randn(hidden_size).astype(np.float64)
        A = np.random.randn(rank, hidden_size).astype(np.float64) * 0.01
        B = np.random.randn(hidden_size, rank).astype(np.float64) * 0.01

        expected = scaling * (x @ A.T @ B.T)

        ct_x = he_backend.encrypt(x)
        ct_delta = he_backend.lora_delta(ct_x, A, B, scaling)
        decrypted = he_backend.decrypt(ct_delta, hidden_size)

        error = np.max(np.abs(expected - decrypted))
        assert error < 0.01, f"Error with hidden_size={hidden_size}: {error}"

    @pytest.mark.parametrize("rank", [4, 8, 16])
    def test_varying_rank(self, he_backend, rank):
        """Test with different LoRA ranks."""
        hidden_size = 64
        scaling = 2.0

        x = np.random.randn(hidden_size).astype(np.float64)
        A = np.random.randn(rank, hidden_size).astype(np.float64) * 0.01
        B = np.random.randn(hidden_size, rank).astype(np.float64) * 0.01

        expected = scaling * (x @ A.T @ B.T)

        ct_x = he_backend.encrypt(x)
        ct_delta = he_backend.lora_delta(ct_x, A, B, scaling)
        decrypted = he_backend.decrypt(ct_delta, hidden_size)

        error = np.max(np.abs(expected - decrypted))
        assert error < 0.01, f"Error with rank={rank}: {error}"


class TestCKKSMOAISpecificFeatures:
    """Test CKKS MOAI specific features."""

    def test_slot_count(self, he_backend):
        """Test slot count is correct."""
        from crypto_backend.ckks_moai import CKKSParams
        params = CKKSParams.default_lora_params()

        expected_slots = params.poly_modulus_degree // 2
        assert he_backend.get_slot_count() == expected_slots

    def test_matmul_accuracy(self, he_backend):
        """Test matrix multiplication accuracy."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        W = np.random.randn(4, 4).astype(np.float64) * 0.1

        expected = x @ W.T

        ct_x = he_backend.encrypt(x)
        ct_result = he_backend.matmul(ct_x, W)
        decrypted = he_backend.decrypt(ct_result, 4)

        error = np.max(np.abs(expected - decrypted))
        assert error < 1e-5, f"Matmul error: {error}"

    def test_operation_stats(self, he_backend):
        """Test operation statistics tracking."""
        he_backend.reset_stats()

        x = np.random.randn(64).astype(np.float64)
        A = np.random.randn(8, 64).astype(np.float64) * 0.01
        B = np.random.randn(64, 8).astype(np.float64) * 0.01

        ct_x = he_backend.encrypt(x)
        ct_delta = he_backend.lora_delta(ct_x, A, B, 1.0)

        stats = he_backend.get_operation_stats()
        assert stats["operations"] > 0
        assert stats["multiplications"] > 0
        assert stats["additions"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
