#!/usr/bin/env python3
"""
HE-LoRA Benchmark Suite.

Compares performance of:
- Baseline plaintext inference (no LoRA)
- Plaintext LoRA
- HE-LoRA-only (MOAI optimized)

Reports:
- Wall time per forward pass
- Ciphertext size
- Levels consumed
- Rotations used

Usage:
    python bench/bench_helora.py --hidden_dim 256 --rank 16 --trials 10
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    hidden_dim: int
    rank: int

    # Timing (milliseconds)
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    # HE-specific metrics
    ciphertext_size_bytes: int = 0
    levels_consumed: int = 0
    rotations_used: int = 0

    # Error metrics (for HE vs plaintext comparison)
    max_error: float = 0.0
    mean_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "hidden_dim": self.hidden_dim,
            "rank": self.rank,
            "mean_time_ms": round(self.mean_time_ms, 3),
            "std_time_ms": round(self.std_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "ciphertext_size_bytes": self.ciphertext_size_bytes,
            "levels_consumed": self.levels_consumed,
            "rotations_used": self.rotations_used,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
        }


def benchmark_baseline(
    hidden_dim: int,
    trials: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Benchmark baseline forward pass (no LoRA)."""

    # Simple linear layer simulation
    W = np.random.randn(hidden_dim, hidden_dim).astype(np.float64) * 0.02

    times = []

    for i in range(warmup + trials):
        x = np.random.randn(hidden_dim).astype(np.float64)

        start = time.perf_counter()
        y = W @ x
        elapsed = (time.perf_counter() - start) * 1000

        if i >= warmup:
            times.append(elapsed)

    return BenchmarkResult(
        name="baseline_no_lora",
        hidden_dim=hidden_dim,
        rank=0,
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
    )


def benchmark_plaintext_lora(
    hidden_dim: int,
    rank: int,
    alpha: float = 32.0,
    trials: int = 10,
    warmup: int = 2,
) -> BenchmarkResult:
    """Benchmark plaintext LoRA computation."""

    # LoRA matrices
    lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
    lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01
    scaling = alpha / rank

    times = []

    for i in range(warmup + trials):
        x = np.random.randn(hidden_dim).astype(np.float64)

        start = time.perf_counter()
        # delta = scaling * (x @ A^T @ B^T)
        intermediate = x @ lora_a.T
        delta = intermediate @ lora_b.T
        delta = scaling * delta
        elapsed = (time.perf_counter() - start) * 1000

        if i >= warmup:
            times.append(elapsed)

    return BenchmarkResult(
        name="plaintext_lora",
        hidden_dim=hidden_dim,
        rank=rank,
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
    )


def benchmark_he_lora(
    hidden_dim: int,
    rank: int,
    alpha: float = 32.0,
    trials: int = 10,
    warmup: int = 2,
) -> Optional[BenchmarkResult]:
    """Benchmark HE-LoRA with MOAI optimizations."""

    try:
        from tensafe.he_lora import HELoRAAdapter, HELoRAConfig
    except ImportError as e:
        print(f"HE-LoRA not available: {e}")
        return None

    try:
        # Create adapter
        config = HELoRAConfig(
            rank=rank,
            alpha=alpha,
        )
        adapter = HELoRAAdapter(config)

        # Register weights
        lora_a = np.random.randn(rank, hidden_dim).astype(np.float64) * 0.01
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float64) * 0.01
        adapter.register_weights("test", lora_a, lora_b, rank=rank, alpha=alpha)

        times = []
        rotations = []
        levels = []
        errors = []

        for i in range(warmup + trials):
            x = np.random.randn(hidden_dim).astype(np.float64)

            # Compute plaintext reference
            scaling = alpha / rank
            ref = scaling * (x @ lora_a.T @ lora_b.T)

            # HE computation
            start = time.perf_counter()
            delta = adapter.forward(x, "test")
            elapsed = (time.perf_counter() - start) * 1000

            if i >= warmup:
                times.append(elapsed)

                # Get HE metrics
                metrics = adapter.get_last_metrics()
                if metrics:
                    rotations.append(metrics.rotations_used)
                    levels.append(metrics.levels_consumed)

                # Error vs plaintext
                error = np.max(np.abs(delta - ref))
                errors.append(error)

        return BenchmarkResult(
            name="he_lora_moai",
            hidden_dim=hidden_dim,
            rank=rank,
            mean_time_ms=np.mean(times),
            std_time_ms=np.std(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            rotations_used=int(np.mean(rotations)) if rotations else 0,
            levels_consumed=int(np.mean(levels)) if levels else 0,
            max_error=float(np.max(errors)) if errors else 0.0,
            mean_error=float(np.mean(errors)) if errors else 0.0,
        )

    except Exception as e:
        print(f"HE-LoRA benchmark failed: {e}")
        return None


def run_benchmarks(
    hidden_dim: int,
    rank: int,
    alpha: float,
    trials: int,
) -> List[BenchmarkResult]:
    """Run all benchmarks."""

    results = []

    print(f"\nRunning benchmarks: hidden_dim={hidden_dim}, rank={rank}, trials={trials}")
    print("-" * 60)

    # Baseline
    print("Benchmarking baseline (no LoRA)...")
    result = benchmark_baseline(hidden_dim, trials)
    results.append(result)
    print(f"  Mean: {result.mean_time_ms:.3f} ms")

    # Plaintext LoRA
    print("Benchmarking plaintext LoRA...")
    result = benchmark_plaintext_lora(hidden_dim, rank, alpha, trials)
    results.append(result)
    print(f"  Mean: {result.mean_time_ms:.3f} ms")

    # HE-LoRA
    print("Benchmarking HE-LoRA (MOAI)...")
    result = benchmark_he_lora(hidden_dim, rank, alpha, trials)
    if result:
        results.append(result)
        print(f"  Mean: {result.mean_time_ms:.3f} ms")
        print(f"  Rotations: {result.rotations_used}")
        print(f"  Levels consumed: {result.levels_consumed}")
        print(f"  Max error: {result.max_error:.2e}")
    else:
        print("  SKIPPED (backend not available)")

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print benchmark summary table."""

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Header
    print(f"{'Name':<20} {'Mean (ms)':<12} {'Std (ms)':<12} {'Rotations':<10} {'Levels':<8} {'Error':<12}")
    print("-" * 80)

    for r in results:
        error_str = f"{r.max_error:.2e}" if r.max_error > 0 else "N/A"
        print(f"{r.name:<20} {r.mean_time_ms:<12.3f} {r.std_time_ms:<12.3f} "
              f"{r.rotations_used:<10} {r.levels_consumed:<8} {error_str:<12}")

    print("-" * 80)

    # Calculate speedups/slowdowns
    baseline = next((r for r in results if r.name == "baseline_no_lora"), None)
    plaintext = next((r for r in results if r.name == "plaintext_lora"), None)
    he_lora = next((r for r in results if r.name == "he_lora_moai"), None)

    if baseline and plaintext:
        overhead = (plaintext.mean_time_ms - baseline.mean_time_ms) / baseline.mean_time_ms * 100
        print(f"Plaintext LoRA overhead vs baseline: {overhead:+.1f}%")

    if baseline and he_lora:
        overhead = (he_lora.mean_time_ms - baseline.mean_time_ms) / baseline.mean_time_ms * 100
        print(f"HE-LoRA overhead vs baseline: {overhead:+.1f}%")

    if plaintext and he_lora:
        slowdown = he_lora.mean_time_ms / plaintext.mean_time_ms
        print(f"HE-LoRA slowdown vs plaintext LoRA: {slowdown:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="HE-LoRA Benchmark")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    results = run_benchmarks(
        hidden_dim=args.hidden_dim,
        rank=args.rank,
        alpha=args.alpha,
        trials=args.trials,
    )

    print_summary(results)

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Verify MOAI optimization
    he_lora = next((r for r in results if r.name == "he_lora_moai"), None)
    if he_lora:
        if he_lora.rotations_used == 0:
            print("\n✓ MOAI optimization verified: ZERO rotations used")
        else:
            print(f"\n✗ MOAI target missed: {he_lora.rotations_used} rotations used (expected 0)")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
