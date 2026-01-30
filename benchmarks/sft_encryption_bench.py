#!/usr/bin/env python3
"""
SFT + Encryption Pipeline E2E Benchmark

Benchmarks the complete pipeline integrating:
1. SFT (Supervised Fine-Tuning) simulation
2. N2HE (Neural-to-Homomorphic Encryption) - LWE-based encryption for gradients
3. MOAI (Model-Optimized Adaptive Inference) - CKKS FHE for inference
4. End-to-end latency and throughput measurements

This benchmark measures:
- N2HE encryption/decryption latency and throughput
- MOAI CKKS encryption/decryption latency
- Homomorphic operation performance
- Memory usage across encryption modes
- Complete E2E pipeline timing
"""

import json
import os
import sys
import time
import statistics
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

import numpy as np

# Suppress experimental crypto warnings for benchmarks
warnings.filterwarnings("ignore", category=UserWarning, module="tensorguard.core.crypto")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensorguard.core.crypto import (
    N2HEContext,
    N2HEParams,
    N2HEEncryptor,
    LWECiphertext,
    sample_skellam,
)
from tensorguard.utils.config import settings


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    name: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_ops_per_sec: float
    memory_bytes: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SFTEncryptionBenchmarkConfig:
    """Configuration for SFT encryption benchmarks."""
    # N2HE parameters
    n2he_lattice_dim: int = 256
    n2he_security_bits: int = 128

    # MOAI parameters (CKKS)
    moai_poly_modulus_degree: int = 8192
    moai_coeff_mod_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    moai_scale: float = 2.0 ** 40

    # Benchmark parameters
    warmup_iterations: int = 3
    benchmark_iterations: int = 50
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32, 128, 512])
    vector_dims: List[int] = field(default_factory=lambda: [64, 256, 1024, 4096])

    # SFT simulation parameters
    sft_batch_size: int = 4
    sft_seq_len: int = 128
    sft_gradient_dim: int = 1024
    sft_num_steps: int = 5

    # Output
    output_dir: str = "artifacts/benchmarks/sft_encryption"


class Timer:
    """High-resolution timer for benchmarks."""

    def __init__(self):
        self.times: List[float] = []
        self._start: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer not started")
        elapsed = (time.perf_counter() - self._start) * 1000  # ms
        self.times.append(elapsed)
        self._start = None
        return elapsed

    @contextmanager
    def measure(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

    def reset(self):
        self.times = []
        self._start = None

    def get_metrics(self, name: str, extra: Dict[str, Any] = None) -> BenchmarkMetrics:
        if not self.times:
            raise ValueError("No measurements recorded")

        sorted_times = sorted(self.times)
        n = len(sorted_times)

        return BenchmarkMetrics(
            name=name,
            iterations=n,
            total_time_ms=sum(self.times),
            mean_time_ms=statistics.mean(self.times),
            std_time_ms=statistics.stdev(self.times) if n > 1 else 0.0,
            min_time_ms=min(self.times),
            max_time_ms=max(self.times),
            p50_time_ms=sorted_times[n // 2],
            p95_time_ms=sorted_times[int(n * 0.95)] if n >= 20 else sorted_times[-1],
            p99_time_ms=sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
            throughput_ops_per_sec=1000.0 * n / sum(self.times) if sum(self.times) > 0 else 0.0,
            extra=extra or {},
        )


class N2HEBenchmark:
    """Benchmarks for N2HE (LWE-based) encryption."""

    def __init__(self, config: SFTEncryptionBenchmarkConfig):
        self.config = config
        self.params = N2HEParams(
            n=config.n2he_lattice_dim,
            security_bits=config.n2he_security_bits,
        )
        self.ctx = N2HEContext(self.params)
        self.ctx.generate_keys()

    def benchmark_encrypt_batch(self, batch_size: int) -> BenchmarkMetrics:
        """Benchmark batch encryption."""
        timer = Timer()

        # Warmup
        for _ in range(self.config.warmup_iterations):
            data = np.random.randint(0, self.params.t, size=batch_size, dtype=np.int64)
            self.ctx.encrypt_batch(data)

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            data = np.random.randint(0, self.params.t, size=batch_size, dtype=np.int64)
            with timer.measure():
                ct = self.ctx.encrypt_batch(data)

        return timer.get_metrics(
            f"n2he_encrypt_batch_{batch_size}",
            extra={
                "batch_size": batch_size,
                "lattice_dim": self.params.n,
                "ciphertext_size_bytes": len(ct.serialize()),
            }
        )

    def benchmark_decrypt_batch(self, batch_size: int) -> BenchmarkMetrics:
        """Benchmark batch decryption."""
        timer = Timer()

        # Prepare ciphertexts
        data = np.random.randint(0, self.params.t, size=batch_size, dtype=np.int64)
        ct = self.ctx.encrypt_batch(data)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            self.ctx.decrypt_batch(ct)

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = self.ctx.decrypt_batch(ct)

        return timer.get_metrics(
            f"n2he_decrypt_batch_{batch_size}",
            extra={"batch_size": batch_size}
        )

    def benchmark_homomorphic_add(self, batch_size: int) -> BenchmarkMetrics:
        """Benchmark homomorphic addition."""
        timer = Timer()

        # Prepare ciphertexts
        data1 = np.random.randint(0, self.params.t // 2, size=batch_size, dtype=np.int64)
        data2 = np.random.randint(0, self.params.t // 2, size=batch_size, dtype=np.int64)
        ct1 = self.ctx.encrypt_batch(data1)
        ct2 = self.ctx.encrypt_batch(data2)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = ct1 + ct2

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = ct1 + ct2

        # Verify correctness
        decrypted = self.ctx.decrypt_batch(result)
        expected = (data1 + data2) % self.params.t
        accuracy = np.mean(decrypted == expected)

        return timer.get_metrics(
            f"n2he_homomorphic_add_{batch_size}",
            extra={
                "batch_size": batch_size,
                "accuracy": float(accuracy),
            }
        )

    def benchmark_serialization(self, batch_size: int) -> BenchmarkMetrics:
        """Benchmark ciphertext serialization/deserialization."""
        timer = Timer()

        data = np.random.randint(0, self.params.t, size=batch_size, dtype=np.int64)
        ct = self.ctx.encrypt_batch(data)

        # Benchmark serialization
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                serialized = ct.serialize()
                _ = LWECiphertext.deserialize(serialized, self.params)

        return timer.get_metrics(
            f"n2he_serialization_{batch_size}",
            extra={
                "batch_size": batch_size,
                "serialized_size_bytes": len(ct.serialize()),
            }
        )

    def benchmark_skellam_noise(self, size: int) -> BenchmarkMetrics:
        """Benchmark Skellam noise generation (DP component)."""
        timer = Timer()
        mu = self.params.mu

        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                noise = sample_skellam(mu, size)

        return timer.get_metrics(
            f"n2he_skellam_noise_{size}",
            extra={
                "size": size,
                "mu": mu,
                "noise_variance": float(np.var(noise)),
            }
        )

    def run_all(self) -> List[BenchmarkMetrics]:
        """Run all N2HE benchmarks."""
        results = []

        print("\n" + "=" * 60)
        print("N2HE (LWE-based) Encryption Benchmarks")
        print("=" * 60)

        for batch_size in self.config.batch_sizes:
            print(f"\n  Batch size: {batch_size}")

            # Encryption
            metrics = self.benchmark_encrypt_batch(batch_size)
            print(f"    Encrypt: {metrics.mean_time_ms:.3f}ms (±{metrics.std_time_ms:.3f}ms)")
            results.append(metrics)

            # Decryption
            metrics = self.benchmark_decrypt_batch(batch_size)
            print(f"    Decrypt: {metrics.mean_time_ms:.3f}ms (±{metrics.std_time_ms:.3f}ms)")
            results.append(metrics)

            # Homomorphic add
            metrics = self.benchmark_homomorphic_add(batch_size)
            print(f"    HE Add:  {metrics.mean_time_ms:.3f}ms (accuracy: {metrics.extra['accuracy']:.2%})")
            results.append(metrics)

            # Serialization
            metrics = self.benchmark_serialization(batch_size)
            print(f"    Serde:   {metrics.mean_time_ms:.3f}ms ({metrics.extra['serialized_size_bytes']} bytes)")
            results.append(metrics)

        # Noise generation
        print("\n  Skellam Noise Generation:")
        for size in [256, 1024, 4096]:
            metrics = self.benchmark_skellam_noise(size)
            print(f"    Size {size}: {metrics.mean_time_ms:.3f}ms")
            results.append(metrics)

        return results


class MOAIBenchmark:
    """Benchmarks for MOAI (CKKS FHE) encryption via TenSEAL."""

    def __init__(self, config: SFTEncryptionBenchmarkConfig):
        self.config = config
        self.tenseal_available = False
        self.ctx = None

        try:
            import tenseal as ts
            self.ts = ts
            self.tenseal_available = True

            # Create CKKS context
            self.ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=config.moai_poly_modulus_degree,
                coeff_mod_bit_sizes=config.moai_coeff_mod_bits,
            )
            self.ctx.global_scale = config.moai_scale
            self.ctx.generate_galois_keys()
            self.ctx.generate_relin_keys()

        except ImportError:
            print("  [WARNING] TenSEAL not available - MOAI benchmarks will be skipped")

    def benchmark_ckks_encrypt(self, vector_dim: int) -> Optional[BenchmarkMetrics]:
        """Benchmark CKKS vector encryption."""
        if not self.tenseal_available:
            return None

        timer = Timer()

        # Warmup
        for _ in range(self.config.warmup_iterations):
            vec = np.random.randn(vector_dim).tolist()
            _ = self.ts.ckks_vector(self.ctx, vec)

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            vec = np.random.randn(vector_dim).tolist()
            with timer.measure():
                ct = self.ts.ckks_vector(self.ctx, vec)

        return timer.get_metrics(
            f"moai_ckks_encrypt_{vector_dim}",
            extra={
                "vector_dim": vector_dim,
                "ciphertext_size_bytes": len(ct.serialize()),
            }
        )

    def benchmark_ckks_decrypt(self, vector_dim: int) -> Optional[BenchmarkMetrics]:
        """Benchmark CKKS vector decryption."""
        if not self.tenseal_available:
            return None

        timer = Timer()

        vec = np.random.randn(vector_dim).tolist()
        ct = self.ts.ckks_vector(self.ctx, vec)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = ct.decrypt()

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = ct.decrypt()

        return timer.get_metrics(
            f"moai_ckks_decrypt_{vector_dim}",
            extra={"vector_dim": vector_dim}
        )

    def benchmark_ckks_add(self, vector_dim: int) -> Optional[BenchmarkMetrics]:
        """Benchmark CKKS homomorphic addition."""
        if not self.tenseal_available:
            return None

        timer = Timer()

        vec1 = np.random.randn(vector_dim).tolist()
        vec2 = np.random.randn(vector_dim).tolist()
        ct1 = self.ts.ckks_vector(self.ctx, vec1)
        ct2 = self.ts.ckks_vector(self.ctx, vec2)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = ct1 + ct2

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = ct1 + ct2

        # Verify correctness
        decrypted = np.array(result.decrypt())
        expected = np.array(vec1) + np.array(vec2)
        max_error = np.max(np.abs(decrypted[:len(expected)] - expected))

        return timer.get_metrics(
            f"moai_ckks_add_{vector_dim}",
            extra={
                "vector_dim": vector_dim,
                "max_error": float(max_error),
            }
        )

    def benchmark_ckks_mul_plain(self, vector_dim: int) -> Optional[BenchmarkMetrics]:
        """Benchmark CKKS multiplication with plaintext (for inference)."""
        if not self.tenseal_available:
            return None

        timer = Timer()

        vec = np.random.randn(vector_dim).tolist()
        plain_scalar = 2.5
        ct = self.ts.ckks_vector(self.ctx, vec)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = ct * plain_scalar

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = ct * plain_scalar

        return timer.get_metrics(
            f"moai_ckks_mul_plain_{vector_dim}",
            extra={"vector_dim": vector_dim}
        )

    def benchmark_ckks_dot_product(self, vector_dim: int) -> Optional[BenchmarkMetrics]:
        """Benchmark CKKS dot product (key operation for inference)."""
        if not self.tenseal_available:
            return None

        timer = Timer()

        vec = np.random.randn(vector_dim).tolist()
        weights = np.random.randn(vector_dim).tolist()
        ct = self.ts.ckks_vector(self.ctx, vec)

        # Warmup
        for _ in range(self.config.warmup_iterations):
            _ = ct.dot(weights)

        # Benchmark
        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                result = ct.dot(weights)

        return timer.get_metrics(
            f"moai_ckks_dot_product_{vector_dim}",
            extra={"vector_dim": vector_dim}
        )

    def run_all(self) -> List[BenchmarkMetrics]:
        """Run all MOAI benchmarks."""
        results = []

        print("\n" + "=" * 60)
        print("MOAI (CKKS FHE) Encryption Benchmarks")
        print("=" * 60)

        if not self.tenseal_available:
            print("  [SKIPPED] TenSEAL not installed")
            return results

        for vector_dim in self.config.vector_dims:
            print(f"\n  Vector dimension: {vector_dim}")

            # Encryption
            metrics = self.benchmark_ckks_encrypt(vector_dim)
            if metrics:
                print(f"    Encrypt:  {metrics.mean_time_ms:.3f}ms ({metrics.extra['ciphertext_size_bytes']} bytes)")
                results.append(metrics)

            # Decryption
            metrics = self.benchmark_ckks_decrypt(vector_dim)
            if metrics:
                print(f"    Decrypt:  {metrics.mean_time_ms:.3f}ms")
                results.append(metrics)

            # HE Addition
            metrics = self.benchmark_ckks_add(vector_dim)
            if metrics:
                print(f"    HE Add:   {metrics.mean_time_ms:.3f}ms (max_err: {metrics.extra['max_error']:.2e})")
                results.append(metrics)

            # Plaintext multiplication
            metrics = self.benchmark_ckks_mul_plain(vector_dim)
            if metrics:
                print(f"    Mul Plain: {metrics.mean_time_ms:.3f}ms")
                results.append(metrics)

            # Dot product
            metrics = self.benchmark_ckks_dot_product(vector_dim)
            if metrics:
                print(f"    Dot Prod: {metrics.mean_time_ms:.3f}ms")
                results.append(metrics)

        return results


class SFTEncryptionPipelineBenchmark:
    """
    E2E benchmark simulating SFT training with encryption.

    Simulates the complete pipeline:
    1. Generate gradient batch (simulating forward/backward)
    2. Encrypt gradients with N2HE (privacy-preserving aggregation)
    3. Homomorphic aggregation
    4. Decrypt aggregated result
    5. Apply optimizer step simulation
    """

    def __init__(self, config: SFTEncryptionBenchmarkConfig):
        self.config = config
        self.n2he_params = N2HEParams(n=config.n2he_lattice_dim)
        self.n2he_ctx = N2HEContext(self.n2he_params)
        self.n2he_ctx.generate_keys()

    def simulate_gradient_computation(self, batch_size: int, gradient_dim: int) -> np.ndarray:
        """Simulate gradient computation from forward/backward pass."""
        # Simulate gradients (quantized to fit N2HE plaintext modulus)
        return np.random.randint(
            -self.n2he_params.t // 4,
            self.n2he_params.t // 4,
            size=gradient_dim,
            dtype=np.int64
        )

    def benchmark_sft_step_with_n2he(self, num_clients: int = 4) -> BenchmarkMetrics:
        """
        Benchmark a single SFT step with N2HE-encrypted gradient aggregation.

        Simulates federated SFT where multiple clients contribute encrypted gradients.
        """
        timer = Timer()
        gradient_dim = self.config.sft_gradient_dim

        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                # Step 1: Each client computes and encrypts gradients
                encrypted_gradients = []
                for _ in range(num_clients):
                    grad = self.simulate_gradient_computation(
                        self.config.sft_batch_size,
                        gradient_dim
                    )
                    ct = self.n2he_ctx.encrypt_batch(grad)
                    encrypted_gradients.append(ct)

                # Step 2: Homomorphic aggregation (sum)
                aggregated = encrypted_gradients[0]
                for ct in encrypted_gradients[1:]:
                    aggregated = aggregated + ct

                # Step 3: Decrypt aggregated result
                decrypted_sum = self.n2he_ctx.decrypt_batch(aggregated)

                # Step 4: Apply averaging (simulated optimizer step)
                averaged_gradient = decrypted_sum // num_clients

        return timer.get_metrics(
            f"sft_step_n2he_{num_clients}_clients",
            extra={
                "num_clients": num_clients,
                "gradient_dim": gradient_dim,
                "privacy_mode": "n2he",
            }
        )

    def benchmark_sft_step_plaintext(self, num_clients: int = 4) -> BenchmarkMetrics:
        """Benchmark SFT step without encryption (baseline)."""
        timer = Timer()
        gradient_dim = self.config.sft_gradient_dim

        for _ in range(self.config.benchmark_iterations):
            with timer.measure():
                # Step 1: Each client computes gradients (no encryption)
                gradients = []
                for _ in range(num_clients):
                    grad = self.simulate_gradient_computation(
                        self.config.sft_batch_size,
                        gradient_dim
                    )
                    gradients.append(grad)

                # Step 2: Aggregate (plaintext sum)
                aggregated = np.sum(gradients, axis=0)

                # Step 3: Apply averaging
                averaged_gradient = aggregated // num_clients

        return timer.get_metrics(
            f"sft_step_plaintext_{num_clients}_clients",
            extra={
                "num_clients": num_clients,
                "gradient_dim": gradient_dim,
                "privacy_mode": "plaintext",
            }
        )

    def benchmark_full_sft_training(self, num_steps: int = None) -> BenchmarkMetrics:
        """Benchmark full SFT training loop with encryption."""
        timer = Timer()
        num_steps = num_steps or self.config.sft_num_steps
        num_clients = 4
        gradient_dim = self.config.sft_gradient_dim

        with timer.measure():
            for step in range(num_steps):
                # Simulate gradient computation and encryption for each client
                encrypted_gradients = []
                for _ in range(num_clients):
                    grad = self.simulate_gradient_computation(
                        self.config.sft_batch_size,
                        gradient_dim
                    )
                    ct = self.n2he_ctx.encrypt_batch(grad)
                    encrypted_gradients.append(ct)

                # Homomorphic aggregation
                aggregated = encrypted_gradients[0]
                for ct in encrypted_gradients[1:]:
                    aggregated = aggregated + ct

                # Decrypt and apply update
                decrypted_sum = self.n2he_ctx.decrypt_batch(aggregated)

        return timer.get_metrics(
            f"sft_full_training_{num_steps}_steps",
            extra={
                "num_steps": num_steps,
                "num_clients": num_clients,
                "gradient_dim": gradient_dim,
                "total_encrypted_ops": num_steps * num_clients,
            }
        )

    def run_all(self) -> List[BenchmarkMetrics]:
        """Run all SFT+encryption pipeline benchmarks."""
        results = []

        print("\n" + "=" * 60)
        print("SFT + Encryption Pipeline Benchmarks")
        print("=" * 60)

        # Compare encrypted vs plaintext for different client counts
        for num_clients in [2, 4, 8, 16]:
            print(f"\n  Federated SFT with {num_clients} clients:")

            # Plaintext baseline
            metrics_plain = self.benchmark_sft_step_plaintext(num_clients)
            print(f"    Plaintext: {metrics_plain.mean_time_ms:.3f}ms/step")
            results.append(metrics_plain)

            # N2HE encrypted
            metrics_n2he = self.benchmark_sft_step_with_n2he(num_clients)
            print(f"    N2HE:      {metrics_n2he.mean_time_ms:.3f}ms/step")
            results.append(metrics_n2he)

            # Overhead calculation
            overhead = (metrics_n2he.mean_time_ms / metrics_plain.mean_time_ms - 1) * 100
            print(f"    Overhead:  {overhead:.1f}%")

        # Full training benchmark
        print(f"\n  Full SFT Training ({self.config.sft_num_steps} steps):")
        metrics_full = self.benchmark_full_sft_training()
        print(f"    Total Time: {metrics_full.total_time_ms:.1f}ms")
        print(f"    Per Step:   {metrics_full.total_time_ms / self.config.sft_num_steps:.1f}ms")
        results.append(metrics_full)

        return results


class SFTEncryptionBenchmarkRunner:
    """Main benchmark runner orchestrating all benchmarks."""

    def __init__(self, config: SFTEncryptionBenchmarkConfig = None):
        self.config = config or SFTEncryptionBenchmarkConfig()
        self.results: Dict[str, Any] = {
            "metadata": {},
            "n2he_benchmarks": [],
            "moai_benchmarks": [],
            "pipeline_benchmarks": [],
            "summary": {},
        }

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect benchmark metadata."""
        import platform

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmark_version": "1.0.0",
            "config": {
                "n2he_lattice_dim": self.config.n2he_lattice_dim,
                "n2he_security_bits": self.config.n2he_security_bits,
                "moai_poly_modulus_degree": self.config.moai_poly_modulus_degree,
                "benchmark_iterations": self.config.benchmark_iterations,
                "batch_sizes": self.config.batch_sizes,
                "vector_dims": self.config.vector_dims,
            },
            "environment": {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "processor": platform.processor(),
            },
        }

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate benchmark summary."""
        summary = {
            "n2he": {},
            "moai": {},
            "pipeline": {},
        }

        # N2HE summary
        n2he_encrypt_times = [
            m["mean_time_ms"] for m in self.results["n2he_benchmarks"]
            if "encrypt_batch" in m["name"]
        ]
        if n2he_encrypt_times:
            summary["n2he"]["avg_encrypt_ms"] = statistics.mean(n2he_encrypt_times)

        # MOAI summary
        moai_encrypt_times = [
            m["mean_time_ms"] for m in self.results["moai_benchmarks"]
            if "encrypt" in m["name"]
        ]
        if moai_encrypt_times:
            summary["moai"]["avg_encrypt_ms"] = statistics.mean(moai_encrypt_times)

        # Pipeline summary
        pipeline_metrics = self.results["pipeline_benchmarks"]
        if pipeline_metrics:
            n2he_times = [m["mean_time_ms"] for m in pipeline_metrics if "n2he" in m["name"]]
            plain_times = [m["mean_time_ms"] for m in pipeline_metrics if "plaintext" in m["name"]]

            if n2he_times and plain_times:
                summary["pipeline"]["avg_n2he_step_ms"] = statistics.mean(n2he_times)
                summary["pipeline"]["avg_plaintext_step_ms"] = statistics.mean(plain_times)
                summary["pipeline"]["avg_overhead_percent"] = (
                    (statistics.mean(n2he_times) / statistics.mean(plain_times) - 1) * 100
                )

        return summary

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks and produce report."""
        print("\n" + "=" * 70)
        print("SFT + ENCRYPTION E2E PERFORMANCE BENCHMARK")
        print("=" * 70)
        print(f"Output: {self.config.output_dir}")

        start_time = time.time()

        # Collect metadata
        self.results["metadata"] = self._collect_metadata()

        # Run N2HE benchmarks
        n2he_bench = N2HEBenchmark(self.config)
        n2he_results = n2he_bench.run_all()
        self.results["n2he_benchmarks"] = [m.to_dict() for m in n2he_results]

        # Run MOAI benchmarks
        moai_bench = MOAIBenchmark(self.config)
        moai_results = moai_bench.run_all()
        self.results["moai_benchmarks"] = [m.to_dict() for m in moai_results]

        # Run pipeline benchmarks
        pipeline_bench = SFTEncryptionPipelineBenchmark(self.config)
        pipeline_results = pipeline_bench.run_all()
        self.results["pipeline_benchmarks"] = [m.to_dict() for m in pipeline_results]

        # Calculate summary
        self.results["summary"] = self._calculate_summary()
        self.results["metadata"]["total_duration_seconds"] = time.time() - start_time

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sft_encryption_bench_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")

        # Also save as latest
        latest_path = output_dir / "sft_encryption_bench_latest.json"
        with open(latest_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        summary = self.results.get("summary", {})

        # N2HE Summary
        if summary.get("n2he"):
            print("\n  N2HE (LWE-based) Encryption:")
            print(f"    Average Encrypt Time: {summary['n2he'].get('avg_encrypt_ms', 0):.3f}ms")

        # MOAI Summary
        if summary.get("moai"):
            print("\n  MOAI (CKKS FHE) Encryption:")
            print(f"    Average Encrypt Time: {summary['moai'].get('avg_encrypt_ms', 0):.3f}ms")

        # Pipeline Summary
        if summary.get("pipeline"):
            print("\n  SFT Pipeline Performance:")
            print(f"    Plaintext Step:  {summary['pipeline'].get('avg_plaintext_step_ms', 0):.3f}ms")
            print(f"    N2HE Step:       {summary['pipeline'].get('avg_n2he_step_ms', 0):.3f}ms")
            print(f"    Encryption Overhead: {summary['pipeline'].get('avg_overhead_percent', 0):.1f}%")

        duration = self.results.get("metadata", {}).get("total_duration_seconds", 0)
        print(f"\n  Total Benchmark Duration: {duration:.1f}s")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SFT + Encryption Pipeline E2E Benchmark"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of benchmark iterations (default: 50)"
    )
    parser.add_argument(
        "--output",
        default="artifacts/benchmarks/sft_encryption",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with fewer iterations"
    )

    args = parser.parse_args()

    config = SFTEncryptionBenchmarkConfig(
        benchmark_iterations=10 if args.quick else args.iterations,
        output_dir=args.output,
    )

    if args.quick:
        config.batch_sizes = [1, 32, 128]
        config.vector_dims = [64, 256]

    runner = SFTEncryptionBenchmarkRunner(config)
    results = runner.run_all()

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
