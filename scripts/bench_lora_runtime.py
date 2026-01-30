#!/usr/bin/env python3
"""
N2HE LoRA Runtime Benchmark Script.

Measures latency and throughput for key operations:
- Encryption
- Decryption
- Addition
- Multiplication
- Matrix multiplication

Usage:
    TENSAFE_TOY_HE=1 python scripts/bench_lora_runtime.py
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import List

import numpy as np

# Ensure toy mode is enabled
os.environ["TENSAFE_TOY_HE"] = "1"

from tensorguard.n2he.core import HESchemeParams, HESchemeType, ToyN2HEScheme


@dataclass
class BenchResult:
    """Benchmark result."""

    operation: str
    size: str
    iterations: int
    total_ms: float
    mean_ms: float
    std_ms: float
    ops_per_sec: float


def benchmark_operation(
    name: str, size: str, func, iterations: int = 100
) -> BenchResult:
    """Benchmark a single operation."""
    times = []

    # Warmup
    for _ in range(5):
        func()

    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)
    return BenchResult(
        operation=name,
        size=size,
        iterations=iterations,
        total_ms=times.sum(),
        mean_ms=times.mean(),
        std_ms=times.std(),
        ops_per_sec=1000 / times.mean() if times.mean() > 0 else 0,
    )


def run_benchmarks() -> List[BenchResult]:
    """Run all benchmarks."""
    results = []

    # Setup
    params = HESchemeParams(
        scheme_type=HESchemeType.LWE,
        n=512,
        q=2**40,
        std_dev=3.2,
        security_level=128,
    )
    scheme = ToyN2HEScheme(params=params)
    sk, pk = scheme.keygen()

    # Benchmark sizes
    sizes = [64, 256, 512, 768, 1024]

    print("=" * 70)
    print("N2HE LoRA Runtime Benchmark (Toy Mode)")
    print("=" * 70)
    print(f"Scheme: LWE, n={params.n}, q=2^40")
    print("WARNING: Toy mode is NOT cryptographically secure!")
    print("=" * 70)
    print()

    for size in sizes:
        print(f"Size: {size}")
        print("-" * 40)

        # Encrypt benchmark
        plaintext = np.random.randint(0, 1000, size=size, dtype=np.int64)
        result = benchmark_operation(
            "encrypt", str(size), lambda: scheme.encrypt(pk, plaintext)
        )
        results.append(result)
        print(f"  Encrypt:  {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ({result.ops_per_sec:.0f} ops/s)")

        # Decrypt benchmark
        ct = scheme.encrypt(pk, plaintext)
        result = benchmark_operation(
            "decrypt", str(size), lambda: scheme.decrypt(sk, ct)
        )
        results.append(result)
        print(f"  Decrypt:  {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ({result.ops_per_sec:.0f} ops/s)")

        # Add benchmark
        ct1 = scheme.encrypt(pk, plaintext)
        ct2 = scheme.encrypt(pk, plaintext)
        result = benchmark_operation(
            "add", str(size), lambda: scheme.add(ct1, ct2)
        )
        results.append(result)
        print(f"  Add:      {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ({result.ops_per_sec:.0f} ops/s)")

        # Multiply benchmark
        ct = scheme.encrypt(pk, plaintext)
        scalar = 5
        result = benchmark_operation(
            "multiply", str(size), lambda: scheme.multiply(ct, scalar)
        )
        results.append(result)
        print(f"  Multiply: {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ({result.ops_per_sec:.0f} ops/s)")

        print()

    # Matrix multiplication benchmark (separate)
    print("Matrix Operations")
    print("-" * 40)
    for hidden_dim, rank in [(256, 16), (512, 32), (768, 64)]:
        pt = np.random.randint(0, 100, size=hidden_dim, dtype=np.int64)
        weight = np.random.randn(rank, hidden_dim).astype(np.float32) * 0.02

        ct = scheme.encrypt(pk, pt)
        _, ek = scheme.generate_eval_keys(sk)

        result = benchmark_operation(
            "matmul",
            f"{hidden_dim}x{rank}",
            lambda: scheme.matmul(ct, weight, ek),
            iterations=50,
        )
        results.append(result)
        print(f"  MatMul {hidden_dim}x{rank}: {result.mean_ms:.3f} ms ± {result.std_ms:.3f} ({result.ops_per_sec:.0f} ops/s)")

    print()
    print("=" * 70)
    print("Benchmark complete.")
    print("=" * 70)

    return results


def main():
    """Main entry point."""
    if os.environ.get("TENSAFE_TOY_HE") != "1":
        print("ERROR: TENSAFE_TOY_HE=1 environment variable required")
        sys.exit(1)

    try:
        run_benchmarks()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
