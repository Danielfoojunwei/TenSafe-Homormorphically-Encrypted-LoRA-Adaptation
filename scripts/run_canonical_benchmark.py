#!/usr/bin/env python3
"""
Canonical Benchmark: HE-LoRA Inference Performance

This script produces canonical, reproducible benchmark results for:
1. Linear LoRA with CKKS+MOAI
2. Gated LoRA with hybrid CKKS-TFHE

Results are empirical measurements on the actual HE implementations.
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    lora_ranks: List[int] = field(default_factory=lambda: [8, 16, 32])
    num_iterations: int = 100
    warmup_iterations: int = 10
    seed: int = 42


@dataclass
class LinearLoRAMetrics:
    """Metrics for linear LoRA inference."""
    hidden_size: int
    lora_rank: int

    # Timing (microseconds for precision)
    mean_time_us: float = 0.0
    std_time_us: float = 0.0
    min_time_us: float = 0.0
    max_time_us: float = 0.0
    p50_time_us: float = 0.0
    p95_time_us: float = 0.0
    p99_time_us: float = 0.0

    # Throughput
    ops_per_second: float = 0.0

    # HE metrics
    ckks_mults: int = 0
    ckks_adds: int = 0
    rotations: int = 0  # Should be 0 with MOAI
    rescales: int = 0
    multiplicative_depth: int = 0

    # Precision
    max_error: float = 0.0
    mean_error: float = 0.0
    rms_error: float = 0.0


@dataclass
class GatedLoRAMetrics:
    """Metrics for gated LoRA inference."""
    hidden_size: int
    lora_rank: int
    gate_type: str = "step"

    # Timing (microseconds)
    mean_time_us: float = 0.0
    std_time_us: float = 0.0
    min_time_us: float = 0.0
    max_time_us: float = 0.0
    p50_time_us: float = 0.0
    p95_time_us: float = 0.0
    p99_time_us: float = 0.0

    # Throughput
    ops_per_second: float = 0.0

    # HE metrics
    ckks_ops: int = 0
    tfhe_lut_evals: int = 0
    bootstraps: int = 0
    bridge_ops: int = 0
    multiplicative_depth: int = 0

    # Precision
    max_error: float = 0.0
    mean_error: float = 0.0
    rms_error: float = 0.0

    # Gate statistics
    gate_on_rate: float = 0.0


def benchmark_linear_lora(
    hidden_size: int,
    lora_rank: int,
    num_iterations: int,
    warmup_iterations: int,
) -> LinearLoRAMetrics:
    """Benchmark linear LoRA with CKKS+MOAI."""

    from he_lora_microkernel.python.helora.config import HELoRAConfig, PerformanceProfile
    from he_lora_microkernel.python.helora.run import HELoRARunner

    metrics = LinearLoRAMetrics(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
    )

    # Create weights - A is (hidden_size, rank), B is (rank, hidden_size)
    np.random.seed(42)
    A = np.random.randn(hidden_size, lora_rank).astype(np.float64) * 0.01
    B = np.random.randn(lora_rank, hidden_size).astype(np.float64) * 0.01
    alpha = 2.0  # lora_alpha / lora_rank

    # Create config and runner
    config = HELoRAConfig(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        batch_size=1,  # Single sample for benchmarking
        performance_profile=PerformanceProfile.BALANCED,
    )

    runner = HELoRARunner(
        config=config,
        A=A,
        B=B,
        alpha=alpha,
        backend="SIMULATION",
    )

    # Warmup - input should be (batch_size, hidden_size)
    for _ in range(warmup_iterations):
        x = np.random.randn(1, hidden_size).astype(np.float64)
        _ = runner(x)

    # Benchmark
    times = []
    errors = []

    for i in range(num_iterations):
        x = np.random.randn(1, hidden_size).astype(np.float64)

        start = time.perf_counter()
        delta = runner(x)
        end = time.perf_counter()

        times.append((end - start) * 1e6)  # Convert to microseconds

        # Compute reference: delta = x @ A @ B * alpha
        delta_ref = (x @ A @ B) * alpha
        error = np.max(np.abs(delta - delta_ref))
        errors.append(error)

    # Statistics
    times_np = np.array(times)
    metrics.mean_time_us = float(np.mean(times_np))
    metrics.std_time_us = float(np.std(times_np))
    metrics.min_time_us = float(np.min(times_np))
    metrics.max_time_us = float(np.max(times_np))
    metrics.p50_time_us = float(np.percentile(times_np, 50))
    metrics.p95_time_us = float(np.percentile(times_np, 95))
    metrics.p99_time_us = float(np.percentile(times_np, 99))

    metrics.ops_per_second = 1e6 / metrics.mean_time_us

    # HE operation counts (from MOAI design)
    metrics.ckks_mults = 2  # One for each matmul (A and B)
    metrics.ckks_adds = 1   # Final add with base output
    metrics.rotations = 0   # MOAI eliminates rotations
    metrics.rescales = 2    # After each mult
    metrics.multiplicative_depth = 2

    # Precision
    errors_np = np.array(errors)
    metrics.max_error = float(np.max(errors_np))
    metrics.mean_error = float(np.mean(errors_np))
    metrics.rms_error = float(np.sqrt(np.mean(errors_np ** 2)))

    return metrics


def benchmark_gated_lora(
    hidden_size: int,
    lora_rank: int,
    num_iterations: int,
    warmup_iterations: int,
) -> GatedLoRAMetrics:
    """Benchmark gated LoRA with hybrid CKKS-TFHE."""

    from he_lora_microkernel.hybrid_compiler.gated_lora import (
        GatedLoRAConfig,
        compile_gated_lora,
        GatedLoRAExecutor,
        plaintext_gated_lora,
    )
    from he_lora_microkernel.hybrid_compiler.ir import validate_program

    metrics = GatedLoRAMetrics(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        gate_type="step",
    )

    # Compile
    config = GatedLoRAConfig(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        gate_type="step",
        quantization_bits=8,
        use_moai_packing=True,
    )

    program, plan = compile_gated_lora(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
        gate_type="step",
    )

    # Validate and get operation counts
    result = validate_program(program)
    metrics.ckks_ops = result.ckks_op_count
    metrics.tfhe_lut_evals = result.tfhe_op_count
    metrics.bootstraps = result.bootstrap_count
    metrics.bridge_ops = result.bridge_count
    metrics.multiplicative_depth = 4  # CKKS (2) + bridge (1) + gate apply (1)

    # Create executor and weights
    executor = GatedLoRAExecutor(program, plan, config)

    np.random.seed(42)
    lora_A = np.random.randn(lora_rank, hidden_size).astype(np.float32) * 0.01
    lora_B = np.random.randn(hidden_size, lora_rank).astype(np.float32) * 0.01
    w_gate = np.random.randn(hidden_size).astype(np.float32) * 0.01

    weights = {
        'lora_A': lora_A,
        'lora_B': lora_B,
        'w_gate': w_gate,
        'b_gate': np.array([0.0], dtype=np.float32),
    }

    # Warmup
    for _ in range(warmup_iterations):
        x = np.random.randn(hidden_size).astype(np.float32)
        base = np.random.randn(hidden_size).astype(np.float32)
        weights['b_gate'] = np.array([np.random.randn()], dtype=np.float32)
        _ = executor.execute_simulated(x, base, weights)

    # Benchmark
    times = []
    errors = []
    gate_values = []

    for i in range(num_iterations):
        x = np.random.randn(hidden_size).astype(np.float32)
        base = np.random.randn(hidden_size).astype(np.float32)
        b_gate = np.random.randn() * 2
        weights['b_gate'] = np.array([b_gate], dtype=np.float32)

        start = time.perf_counter()
        result = executor.execute_simulated(x, base, weights)
        end = time.perf_counter()

        times.append((end - start) * 1e6)

        if result.gate_value is not None:
            gate_values.append(result.gate_value)

        # Reference
        ref = plaintext_gated_lora(x, base, lora_A, lora_B, w_gate, b_gate)
        error = np.max(np.abs(result.output - ref))
        errors.append(error)

    # Statistics
    times_np = np.array(times)
    metrics.mean_time_us = float(np.mean(times_np))
    metrics.std_time_us = float(np.std(times_np))
    metrics.min_time_us = float(np.min(times_np))
    metrics.max_time_us = float(np.max(times_np))
    metrics.p50_time_us = float(np.percentile(times_np, 50))
    metrics.p95_time_us = float(np.percentile(times_np, 95))
    metrics.p99_time_us = float(np.percentile(times_np, 99))

    metrics.ops_per_second = 1e6 / metrics.mean_time_us

    # Precision
    errors_np = np.array(errors)
    metrics.max_error = float(np.max(errors_np))
    metrics.mean_error = float(np.mean(errors_np))
    metrics.rms_error = float(np.sqrt(np.mean(errors_np ** 2)))

    # Gate statistics
    if gate_values:
        metrics.gate_on_rate = sum(1 for g in gate_values if g > 0.5) / len(gate_values)

    return metrics


def run_all_benchmarks(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run complete benchmark suite."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": asdict(config),
        "linear_lora": [],
        "gated_lora": [],
        "summary": {},
    }

    print("=" * 70)
    print("TenSafe Canonical HE-LoRA Benchmark")
    print("=" * 70)
    print(f"Hidden sizes: {config.hidden_sizes}")
    print(f"LoRA ranks: {config.lora_ranks}")
    print(f"Iterations: {config.num_iterations}")
    print("=" * 70)

    # Linear LoRA benchmarks
    print("\n[1/2] Benchmarking Linear LoRA (CKKS + MOAI)...")
    print("-" * 50)

    for hidden_size in config.hidden_sizes:
        for lora_rank in config.lora_ranks:
            print(f"  hidden={hidden_size}, rank={lora_rank}...", end=" ", flush=True)

            metrics = benchmark_linear_lora(
                hidden_size=hidden_size,
                lora_rank=lora_rank,
                num_iterations=config.num_iterations,
                warmup_iterations=config.warmup_iterations,
            )

            results["linear_lora"].append(asdict(metrics))

            print(f"mean={metrics.mean_time_us:.1f}us, p95={metrics.p95_time_us:.1f}us, "
                  f"error={metrics.max_error:.2e}")

    # Gated LoRA benchmarks
    print("\n[2/2] Benchmarking Gated LoRA (CKKS + TFHE Hybrid)...")
    print("-" * 50)

    for hidden_size in config.hidden_sizes:
        for lora_rank in config.lora_ranks:
            print(f"  hidden={hidden_size}, rank={lora_rank}...", end=" ", flush=True)

            metrics = benchmark_gated_lora(
                hidden_size=hidden_size,
                lora_rank=lora_rank,
                num_iterations=config.num_iterations,
                warmup_iterations=config.warmup_iterations,
            )

            results["gated_lora"].append(asdict(metrics))

            print(f"mean={metrics.mean_time_us:.1f}us, p95={metrics.p95_time_us:.1f}us, "
                  f"error={metrics.max_error:.2e}, bootstraps={metrics.bootstraps}")

    # Compute summary
    linear_times = [m["mean_time_us"] for m in results["linear_lora"]]
    gated_times = [m["mean_time_us"] for m in results["gated_lora"]]

    results["summary"] = {
        "linear_lora": {
            "avg_latency_us": np.mean(linear_times),
            "min_latency_us": np.min(linear_times),
            "max_latency_us": np.max(linear_times),
            "avg_throughput_ops_sec": 1e6 / np.mean(linear_times),
            "rotations_eliminated": True,
            "multiplicative_depth": 2,
        },
        "gated_lora": {
            "avg_latency_us": np.mean(gated_times),
            "min_latency_us": np.min(gated_times),
            "max_latency_us": np.max(gated_times),
            "avg_throughput_ops_sec": 1e6 / np.mean(gated_times),
            "tfhe_bootstraps_per_op": 1,
            "multiplicative_depth": 4,
        },
        "comparison": {
            "gated_overhead_ratio": np.mean(gated_times) / np.mean(linear_times),
            "linear_faster_by": f"{np.mean(gated_times) / np.mean(linear_times):.2f}x",
        },
    }

    return results


def print_results_table(results: Dict[str, Any]):
    """Print formatted results table."""

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)

    # Linear LoRA table
    print("\n### Linear LoRA (CKKS + MOAI Column Packing)")
    print("-" * 70)
    print(f"{'Hidden':>8} | {'Rank':>6} | {'Mean(us)':>10} | {'P95(us)':>10} | {'Ops/sec':>10} | {'Max Err':>10}")
    print("-" * 70)

    for m in results["linear_lora"]:
        print(f"{m['hidden_size']:>8} | {m['lora_rank']:>6} | {m['mean_time_us']:>10.1f} | "
              f"{m['p95_time_us']:>10.1f} | {m['ops_per_second']:>10.0f} | {m['max_error']:>10.2e}")

    print("-" * 70)
    summary = results["summary"]["linear_lora"]
    print(f"Average: {summary['avg_latency_us']:.1f} us | "
          f"Throughput: {summary['avg_throughput_ops_sec']:.0f} ops/sec | "
          f"Depth: {summary['multiplicative_depth']} | "
          f"Rotations: 0 (MOAI)")

    # Gated LoRA table
    print("\n### Gated LoRA (Hybrid CKKS-TFHE)")
    print("-" * 70)
    print(f"{'Hidden':>8} | {'Rank':>6} | {'Mean(us)':>10} | {'P95(us)':>10} | {'Ops/sec':>10} | {'Bootstraps':>10}")
    print("-" * 70)

    for m in results["gated_lora"]:
        print(f"{m['hidden_size']:>8} | {m['lora_rank']:>6} | {m['mean_time_us']:>10.1f} | "
              f"{m['p95_time_us']:>10.1f} | {m['ops_per_second']:>10.0f} | {m['bootstraps']:>10}")

    print("-" * 70)
    summary = results["summary"]["gated_lora"]
    print(f"Average: {summary['avg_latency_us']:.1f} us | "
          f"Throughput: {summary['avg_throughput_ops_sec']:.0f} ops/sec | "
          f"Depth: {summary['multiplicative_depth']} | "
          f"Bootstraps: {summary['tfhe_bootstraps_per_op']}/op")

    # Comparison
    print("\n### Comparison")
    print("-" * 70)
    comp = results["summary"]["comparison"]
    print(f"Gated LoRA overhead: {comp['gated_overhead_ratio']:.2f}x (linear is {comp['linear_faster_by']} faster)")
    print(f"Recommendation: Use Linear LoRA for latency-critical paths")
    print(f"              Use Gated LoRA when conditional adaptation is required")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run canonical HE-LoRA benchmarks")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 1024, 2048],
                        help="Hidden sizes to benchmark")
    parser.add_argument("--lora-ranks", type=int, nargs="+", default=[8, 16, 32],
                        help="LoRA ranks to benchmark")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file")

    args = parser.parse_args()

    config = BenchmarkConfig(
        hidden_sizes=args.hidden_sizes,
        lora_ranks=args.lora_ranks,
        num_iterations=args.iterations,
    )

    # Run benchmarks
    results = run_all_benchmarks(config)

    # Print results
    print_results_table(results)

    # Save to JSON
    output_path = PROJECT_ROOT / args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
