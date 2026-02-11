#!/usr/bin/env python3
"""
TensorGuard Unit Benchmarks - Offline Performance Measurement

This module provides REAL benchmarks that measure actual performance
of cryptographic and serialization operations without requiring
a running server.

All results are empirical measurements from actual execution.
"""

import time
import json
import statistics
import platform
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

import numpy as np

# Ensure we can import tensorguard
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class BenchmarkMeasurement:
    """A single benchmark measurement with statistical analysis."""
    name: str
    iterations: int
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def min_ms(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.95)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def p99_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = int(len(sorted_lat) * 0.99)
        return sorted_lat[min(idx, len(sorted_lat) - 1)]

    @property
    def throughput_ops_per_sec(self) -> float:
        if self.mean_ms <= 0:
            return 0.0
        return 1000.0 / self.mean_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "mean_ms": round(self.mean_ms, 4),
            "median_ms": round(self.median_ms, 4),
            "std_ms": round(self.std_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "throughput_ops_per_sec": round(self.throughput_ops_per_sec, 2),
        }


@dataclass
class UnitBenchmarkResults:
    """Complete unit benchmark results."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    platform_info: Dict[str, str] = field(default_factory=dict)
    benchmarks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "platform_info": self.platform_info,
            "benchmarks": self.benchmarks,
            "summary": self.summary,
        }


class UnitBenchmark:
    """
    Offline unit benchmarks for TensorGuard operations.

    Measures actual performance without requiring external services.
    All results are empirical measurements from real execution.
    """

    def __init__(self, iterations: int = 100, warmup: int = 5):
        self.iterations = iterations
        self.warmup = warmup
        self.results = UnitBenchmarkResults()
        self._collect_platform_info()

    def _collect_platform_info(self):
        """Collect system information for reproducibility."""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            cpu_count = "unknown"
            memory_gb = "unknown"

        self.results.platform_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": str(cpu_count),
            "memory_gb": f"{memory_gb:.1f}" if isinstance(memory_gb, float) else memory_gb,
            "numpy_version": np.__version__,
        }

    def _measure(self, name: str, func, *args, **kwargs) -> BenchmarkMeasurement:
        """Measure a function's execution time over multiple iterations."""
        measurement = BenchmarkMeasurement(name=name, iterations=self.iterations)

        # Warmup
        for _ in range(self.warmup):
            func(*args, **kwargs)

        # Actual measurement
        for _ in range(self.iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            measurement.latencies_ms.append(elapsed_ms)

        return measurement

    def benchmark_n2he_encrypt(self, tensor_sizes_kb: List[float] = None) -> Dict[str, Any]:
        """Benchmark N2HE encryption with various tensor sizes."""
        from tensorguard.core.crypto import N2HEEncryptor

        if tensor_sizes_kb is None:
            tensor_sizes_kb = [0.1, 1.0, 10.0, 100.0]

        results = {}
        encryptor = N2HEEncryptor(security_level=128)

        for size_kb in tensor_sizes_kb:
            # Create test data of specified size
            num_bytes = int(size_kb * 1024)
            test_data = np.random.bytes(num_bytes)

            measurement = self._measure(
                f"n2he_encrypt_{size_kb}kb",
                encryptor.encrypt,
                test_data
            )

            results[f"{size_kb}kb"] = {
                **measurement.to_dict(),
                "input_size_bytes": num_bytes,
                "throughput_kb_per_sec": round(size_kb * measurement.throughput_ops_per_sec, 2),
            }

            print(f"  N2HE Encrypt {size_kb}KB: {measurement.mean_ms:.2f}ms mean, "
                  f"{measurement.p95_ms:.2f}ms p95")

        return results

    def benchmark_n2he_decrypt(self, tensor_sizes_kb: List[float] = None) -> Dict[str, Any]:
        """Benchmark N2HE decryption with various tensor sizes."""
        from tensorguard.core.crypto import N2HEEncryptor

        if tensor_sizes_kb is None:
            tensor_sizes_kb = [0.1, 1.0, 10.0]  # Smaller sizes for decrypt

        results = {}
        encryptor = N2HEEncryptor(security_level=128)

        for size_kb in tensor_sizes_kb:
            num_bytes = int(size_kb * 1024)
            test_data = np.random.bytes(num_bytes)

            # Pre-encrypt for decryption benchmark
            encrypted = encryptor.encrypt(test_data)

            measurement = self._measure(
                f"n2he_decrypt_{size_kb}kb",
                encryptor.decrypt,
                encrypted
            )

            results[f"{size_kb}kb"] = {
                **measurement.to_dict(),
                "input_size_bytes": len(encrypted),
                "throughput_kb_per_sec": round(size_kb * measurement.throughput_ops_per_sec, 2),
            }

            print(f"  N2HE Decrypt {size_kb}KB: {measurement.mean_ms:.2f}ms mean, "
                  f"{measurement.p95_ms:.2f}ms p95")

        return results

    def benchmark_hybrid_signature(self) -> Dict[str, Any]:
        """Benchmark hybrid Ed25519+Dilithium3 signatures."""
        try:
            from tensorguard.crypto.sig import (
                generate_hybrid_sig_keypair,
                sign_hybrid,
                verify_hybrid,
            )
            # Test if PQC is available
            generate_hybrid_sig_keypair()
        except ImportError as e:
            print(f"  SKIPPED: {e}")
            return {"skipped": True, "reason": str(e)}

        results = {}
        message_sizes = [256, 1024, 4096, 16384]  # bytes

        # Key generation
        keygen_measurement = self._measure(
            "hybrid_sig_keygen",
            generate_hybrid_sig_keypair
        )
        results["keygen"] = keygen_measurement.to_dict()
        print(f"  Hybrid Keygen: {keygen_measurement.mean_ms:.2f}ms mean")

        # Generate a key pair for sign/verify
        pub, priv = generate_hybrid_sig_keypair()

        for msg_size in message_sizes:
            message = np.random.bytes(msg_size)

            # Sign
            sign_measurement = self._measure(
                f"hybrid_sig_sign_{msg_size}b",
                sign_hybrid,
                priv, message
            )
            results[f"sign_{msg_size}b"] = sign_measurement.to_dict()

            # Verify
            signature = sign_hybrid(priv, message)
            verify_measurement = self._measure(
                f"hybrid_sig_verify_{msg_size}b",
                verify_hybrid,
                pub, message, signature
            )
            results[f"verify_{msg_size}b"] = verify_measurement.to_dict()

            print(f"  Hybrid Sign {msg_size}B: {sign_measurement.mean_ms:.2f}ms, "
                  f"Verify: {verify_measurement.mean_ms:.2f}ms")

        return results

    def benchmark_ed25519_signature(self) -> Dict[str, Any]:
        """Benchmark Ed25519 signatures (classical crypto, always available)."""
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        results = {}
        message_sizes = [256, 1024, 4096, 16384]

        # Key generation
        keygen_measurement = self._measure(
            "ed25519_keygen",
            ed25519.Ed25519PrivateKey.generate
        )
        results["keygen"] = keygen_measurement.to_dict()
        print(f"  Ed25519 Keygen: {keygen_measurement.mean_ms:.3f}ms mean")

        # Generate key for sign/verify
        priv_key = ed25519.Ed25519PrivateKey.generate()
        pub_key = priv_key.public_key()

        for msg_size in message_sizes:
            message = np.random.bytes(msg_size)

            # Sign
            sign_measurement = self._measure(
                f"ed25519_sign_{msg_size}b",
                priv_key.sign,
                message
            )
            results[f"sign_{msg_size}b"] = sign_measurement.to_dict()

            # Verify
            signature = priv_key.sign(message)
            verify_measurement = self._measure(
                f"ed25519_verify_{msg_size}b",
                pub_key.verify,
                signature, message
            )
            results[f"verify_{msg_size}b"] = verify_measurement.to_dict()

            print(f"  Ed25519 Sign {msg_size}B: {sign_measurement.mean_ms:.3f}ms, "
                  f"Verify: {verify_measurement.mean_ms:.3f}ms")

        return results

    def benchmark_serialization(self) -> Dict[str, Any]:
        """Benchmark UpdatePackage serialization/deserialization."""
        from tensorguard.core.production import UpdatePackage

        results = {}
        payload_sizes_kb = [1, 10, 100, 1000]

        for size_kb in payload_sizes_kb:
            payload_bytes = np.random.bytes(size_kb * 1024)

            pkg = UpdatePackage(
                client_id="bench_client",
                delta_tensors={"grad": payload_bytes}
            )

            # Serialize
            serialize_measurement = self._measure(
                f"serialize_{size_kb}kb",
                pkg.serialize
            )

            serialized = pkg.serialize()

            # Deserialize
            deserialize_measurement = self._measure(
                f"deserialize_{size_kb}kb",
                UpdatePackage.deserialize,
                serialized
            )

            results[f"{size_kb}kb"] = {
                "serialize": serialize_measurement.to_dict(),
                "deserialize": deserialize_measurement.to_dict(),
                "payload_size_bytes": size_kb * 1024,
                "serialized_size_bytes": len(serialized),
                "overhead_ratio": round(len(serialized) / (size_kb * 1024), 4),
            }

            print(f"  Serialize {size_kb}KB: {serialize_measurement.mean_ms:.2f}ms, "
                  f"Deserialize: {deserialize_measurement.mean_ms:.2f}ms")

        return results

    def benchmark_lwe_ciphertext(self) -> Dict[str, Any]:
        """Benchmark LWE ciphertext serialization."""
        from tensorguard.core.crypto import LWECiphertext, N2HEParams

        results = {}
        batch_sizes = [1, 10, 100, 1000]

        params = N2HEParams()

        for batch_size in batch_sizes:
            # Create ciphertext
            import secrets
            seed = secrets.token_bytes(32)
            b = np.random.randint(0, params.q, size=batch_size, dtype=np.int64)
            ct = LWECiphertext(b=b, seed=seed, params=params)

            # Serialize
            serialize_measurement = self._measure(
                f"lwe_serialize_{batch_size}",
                ct.serialize
            )

            serialized = ct.serialize()

            # Deserialize
            deserialize_measurement = self._measure(
                f"lwe_deserialize_{batch_size}",
                LWECiphertext.deserialize,
                serialized, params
            )

            results[f"batch_{batch_size}"] = {
                "serialize": serialize_measurement.to_dict(),
                "deserialize": deserialize_measurement.to_dict(),
                "batch_size": batch_size,
                "serialized_bytes": len(serialized),
            }

            print(f"  LWE Batch {batch_size}: Serialize {serialize_measurement.mean_ms:.3f}ms, "
                  f"Deserialize {deserialize_measurement.mean_ms:.3f}ms")

        return results

    def benchmark_numpy_operations(self) -> Dict[str, Any]:
        """Benchmark numpy operations used in crypto (baseline)."""
        results = {}

        sizes = [1024, 4096, 16384]  # LWE dimensions

        for n in sizes:
            # Matrix-vector multiplication (core LWE operation)
            A = np.random.randint(0, 2**32, size=(100, n), dtype=np.int64)
            s = np.random.choice([-1, 0, 1], size=n).astype(np.int64)

            matmul_measurement = self._measure(
                f"matmul_{n}",
                lambda: np.dot(A, s)
            )

            results[f"matmul_{n}"] = matmul_measurement.to_dict()
            print(f"  NumPy Matmul [{100}x{n}]: {matmul_measurement.mean_ms:.3f}ms")

        return results

    def run_all(self) -> UnitBenchmarkResults:
        """Run all unit benchmarks and return results."""
        print("\n" + "=" * 70)
        print("TENSORGUARD UNIT BENCHMARKS")
        print("=" * 70)
        print(f"Iterations: {self.iterations}, Warmup: {self.warmup}")
        print(f"Platform: {self.results.platform_info.get('platform', 'unknown')}")
        print("=" * 70)

        print("\n[1/6] N2HE Encryption...")
        self.results.benchmarks["n2he_encrypt"] = self.benchmark_n2he_encrypt()

        print("\n[2/6] N2HE Decryption...")
        self.results.benchmarks["n2he_decrypt"] = self.benchmark_n2he_decrypt()

        print("\n[3/7] Ed25519 Signatures (Classical Crypto)...")
        self.results.benchmarks["ed25519_signature"] = self.benchmark_ed25519_signature()

        print("\n[4/7] Hybrid Signatures (Ed25519 + Dilithium3)...")
        self.results.benchmarks["hybrid_signature"] = self.benchmark_hybrid_signature()

        print("\n[5/7] UpdatePackage Serialization...")
        self.results.benchmarks["serialization"] = self.benchmark_serialization()

        print("\n[6/7] LWE Ciphertext Serialization...")
        self.results.benchmarks["lwe_ciphertext"] = self.benchmark_lwe_ciphertext()

        print("\n[7/7] NumPy Baseline Operations...")
        self.results.benchmarks["numpy_baseline"] = self.benchmark_numpy_operations()

        # Calculate summary
        self._calculate_summary()

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

        return self.results

    def _calculate_summary(self):
        """Calculate summary statistics across all benchmarks."""
        # Key metrics for regression testing
        enc = self.results.benchmarks.get("n2he_encrypt", {})
        dec = self.results.benchmarks.get("n2he_decrypt", {})
        ed25519 = self.results.benchmarks.get("ed25519_signature", {})
        hybrid = self.results.benchmarks.get("hybrid_signature", {})
        ser = self.results.benchmarks.get("serialization", {})
        lwe = self.results.benchmarks.get("lwe_ciphertext", {})

        self.results.summary = {
            # N2HE Encryption
            "n2he_encrypt_1kb_mean_ms": enc.get("1.0kb", {}).get("mean_ms", 0),
            "n2he_encrypt_1kb_p95_ms": enc.get("1.0kb", {}).get("p95_ms", 0),
            "n2he_encrypt_10kb_mean_ms": enc.get("10.0kb", {}).get("mean_ms", 0),
            "n2he_encrypt_10kb_p95_ms": enc.get("10.0kb", {}).get("p95_ms", 0),
            # N2HE Decryption
            "n2he_decrypt_1kb_mean_ms": dec.get("1.0kb", {}).get("mean_ms", 0),
            "n2he_decrypt_1kb_p95_ms": dec.get("1.0kb", {}).get("p95_ms", 0),
            # Ed25519 (always available)
            "ed25519_sign_1024b_mean_ms": ed25519.get("sign_1024b", {}).get("mean_ms", 0),
            "ed25519_sign_1024b_p95_ms": ed25519.get("sign_1024b", {}).get("p95_ms", 0),
            "ed25519_verify_1024b_mean_ms": ed25519.get("verify_1024b", {}).get("mean_ms", 0),
            "ed25519_verify_1024b_p95_ms": ed25519.get("verify_1024b", {}).get("p95_ms", 0),
            # Hybrid (if available)
            "hybrid_available": not hybrid.get("skipped", False),
            "hybrid_sign_1024b_p95_ms": hybrid.get("sign_1024b", {}).get("p95_ms", 0) if not hybrid.get("skipped") else None,
            # Serialization
            "serialize_100kb_mean_ms": ser.get("100kb", {}).get("serialize", {}).get("mean_ms", 0),
            "serialize_100kb_p95_ms": ser.get("100kb", {}).get("serialize", {}).get("p95_ms", 0),
            # LWE Ciphertext
            "lwe_serialize_100_mean_ms": lwe.get("batch_100", {}).get("serialize", {}).get("mean_ms", 0),
            # Metadata
            "total_benchmarks": len(self.results.benchmarks),
            "benchmarks_skipped": sum(1 for b in self.results.benchmarks.values() if b.get("skipped", False)),
        }

    def save_results(self, output_path: str = "artifacts/benchmarks/unit_benchmark_results.json"):
        """Save benchmark results to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Also save as latest
        latest_path = output_file.parent / "unit_benchmark_latest.json"
        with open(latest_path, "w") as f:
            json.dump(self.results.to_dict(), f, indent=2)

        return output_file


def run_unit_benchmarks(iterations: int = 50, warmup: int = 5) -> UnitBenchmarkResults:
    """Run all unit benchmarks and save results."""
    benchmark = UnitBenchmark(iterations=iterations, warmup=warmup)
    results = benchmark.run_all()
    benchmark.save_results()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TensorGuard Unit Benchmarks")
    parser.add_argument("--iterations", "-n", type=int, default=50,
                        help="Number of iterations per benchmark")
    parser.add_argument("--warmup", "-w", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--output", "-o", type=str,
                        default="artifacts/benchmarks/unit_benchmark_results.json",
                        help="Output file path")

    args = parser.parse_args()

    benchmark = UnitBenchmark(iterations=args.iterations, warmup=args.warmup)
    results = benchmark.run_all()
    benchmark.save_results(args.output)
