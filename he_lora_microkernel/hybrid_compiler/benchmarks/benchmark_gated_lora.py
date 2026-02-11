#!/usr/bin/env python3
"""
Performance Benchmark for Hybrid CKKS-TFHE Gated LoRA

Measures:
- Per-token latency breakdown (CKKS linear, bridge, TFHE LUT)
- Operation counts (conversions, LUT applications, bootstraps)
- Accuracy metrics (quantization error, gate classification accuracy)
- Throughput (tokens/second)

Usage:
    python -m he_lora_microkernel.hybrid_compiler.benchmarks.benchmark_gated_lora
    python -m he_lora_microkernel.hybrid_compiler.benchmarks.benchmark_gated_lora --hidden_size 4096 --rank 16
"""

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

try:
    from ..backend import HybridHEBackend, HybridHEConfig, BridgeMode
    from ..adapters import (
        HEGatedLoRAAdapter,
        GatedLoRAAdapterConfig,
        AdapterWeights,
        AdapterMetrics,
    )
    from ..tfhe_lut import LUTLibrary
except ImportError:
    # Direct import for standalone execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from he_lora_microkernel.hybrid_compiler.backend import HybridHEBackend, HybridHEConfig, BridgeMode
    from he_lora_microkernel.hybrid_compiler.adapters import (
        HEGatedLoRAAdapter,
        GatedLoRAAdapterConfig,
        AdapterWeights,
        AdapterMetrics,
    )
    from he_lora_microkernel.hybrid_compiler.tfhe_lut import LUTLibrary


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class LatencyBreakdown:
    """Latency breakdown for a single operation."""
    total_ms: float = 0.0
    ckks_linear_ms: float = 0.0
    ckks_gate_pre_ms: float = 0.0
    bridge_to_tfhe_ms: float = 0.0
    tfhe_lut_ms: float = 0.0
    bridge_to_ckks_ms: float = 0.0
    ckks_apply_gate_ms: float = 0.0
    ckks_final_add_ms: float = 0.0

    @classmethod
    def from_metrics(cls, metrics: AdapterMetrics) -> 'LatencyBreakdown':
        return cls(
            total_ms=metrics.total_time_ms,
            ckks_linear_ms=metrics.ckks_lora_time_ms,
            ckks_gate_pre_ms=metrics.ckks_gate_pre_time_ms,
            bridge_to_tfhe_ms=metrics.bridge_to_tfhe_time_ms,
            tfhe_lut_ms=metrics.tfhe_lut_time_ms,
            bridge_to_ckks_ms=metrics.bridge_to_ckks_time_ms,
            ckks_apply_gate_ms=metrics.ckks_apply_gate_time_ms,
            ckks_final_add_ms=metrics.ckks_final_add_time_ms,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "ckks_linear_ms": self.ckks_linear_ms,
            "ckks_gate_pre_ms": self.ckks_gate_pre_ms,
            "bridge_to_tfhe_ms": self.bridge_to_tfhe_ms,
            "tfhe_lut_ms": self.tfhe_lut_ms,
            "bridge_to_ckks_ms": self.bridge_to_ckks_ms,
            "ckks_apply_gate_ms": self.ckks_apply_gate_ms,
            "ckks_final_add_ms": self.ckks_final_add_ms,
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    hidden_size: int = 64
    rank: int = 4
    alpha: float = 32.0
    batch_size: int = 1
    num_tokens: int = 100
    warmup_tokens: int = 10
    gate_type: str = "step"
    quantization_bits: int = 8
    seed: int = 42


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    config: BenchmarkConfig
    latency_samples: List[LatencyBreakdown] = field(default_factory=list)

    # Aggregated latencies (milliseconds)
    avg_total_ms: float = 0.0
    p50_total_ms: float = 0.0
    p90_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    min_total_ms: float = 0.0
    max_total_ms: float = 0.0

    # Breakdown averages
    avg_ckks_linear_ms: float = 0.0
    avg_bridge_ms: float = 0.0
    avg_lut_ms: float = 0.0

    # Operation counts
    total_ckks_ops: int = 0
    total_tfhe_ops: int = 0
    total_bootstraps: int = 0
    total_conversions: int = 0

    # Accuracy metrics
    avg_quantization_error: float = 0.0
    max_quantization_error: float = 0.0
    gate_on_count: int = 0
    gate_off_count: int = 0

    # Throughput
    tokens_per_second: float = 0.0

    def compute_aggregates(self):
        """Compute aggregate statistics from samples."""
        if not self.latency_samples:
            return

        totals = [s.total_ms for s in self.latency_samples]
        totals_sorted = sorted(totals)

        self.avg_total_ms = np.mean(totals)
        self.min_total_ms = min(totals)
        self.max_total_ms = max(totals)
        self.p50_total_ms = np.percentile(totals, 50)
        self.p90_total_ms = np.percentile(totals, 90)
        self.p99_total_ms = np.percentile(totals, 99)

        self.avg_ckks_linear_ms = np.mean([s.ckks_linear_ms for s in self.latency_samples])
        self.avg_bridge_ms = np.mean([s.bridge_to_tfhe_ms + s.bridge_to_ckks_ms for s in self.latency_samples])
        self.avg_lut_ms = np.mean([s.tfhe_lut_ms for s in self.latency_samples])

        # Throughput
        total_time_s = sum(totals) / 1000.0
        if total_time_s > 0:
            self.tokens_per_second = len(totals) / total_time_s

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "hidden_size": self.config.hidden_size,
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "batch_size": self.config.batch_size,
                "num_tokens": self.config.num_tokens,
                "gate_type": self.config.gate_type,
                "quantization_bits": self.config.quantization_bits,
            },
            "latency": {
                "avg_ms": self.avg_total_ms,
                "p50_ms": self.p50_total_ms,
                "p90_ms": self.p90_total_ms,
                "p99_ms": self.p99_total_ms,
                "min_ms": self.min_total_ms,
                "max_ms": self.max_total_ms,
            },
            "breakdown": {
                "avg_ckks_linear_ms": self.avg_ckks_linear_ms,
                "avg_bridge_ms": self.avg_bridge_ms,
                "avg_lut_ms": self.avg_lut_ms,
            },
            "operations": {
                "total_ckks_ops": self.total_ckks_ops,
                "total_tfhe_ops": self.total_tfhe_ops,
                "total_bootstraps": self.total_bootstraps,
                "total_conversions": self.total_conversions,
            },
            "accuracy": {
                "avg_quantization_error": self.avg_quantization_error,
                "max_quantization_error": self.max_quantization_error,
                "gate_on_count": self.gate_on_count,
                "gate_off_count": self.gate_off_count,
            },
            "throughput": {
                "tokens_per_second": self.tokens_per_second,
            },
        }

    def print_report(self):
        """Print formatted benchmark report."""
        print("\n" + "=" * 70)
        print("HYBRID GATED LORA BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\nConfiguration:")
        print(f"  Hidden Size:     {self.config.hidden_size}")
        print(f"  LoRA Rank:       {self.config.rank}")
        print(f"  Alpha:           {self.config.alpha}")
        print(f"  Gate Type:       {self.config.gate_type}")
        print(f"  Quant Bits:      {self.config.quantization_bits}")
        print(f"  Num Tokens:      {self.config.num_tokens}")

        print(f"\nLatency (milliseconds):")
        print(f"  Average:         {self.avg_total_ms:.3f} ms")
        print(f"  P50:             {self.p50_total_ms:.3f} ms")
        print(f"  P90:             {self.p90_total_ms:.3f} ms")
        print(f"  P99:             {self.p99_total_ms:.3f} ms")
        print(f"  Min:             {self.min_total_ms:.3f} ms")
        print(f"  Max:             {self.max_total_ms:.3f} ms")

        print(f"\nLatency Breakdown (average):")
        print(f"  CKKS Linear:     {self.avg_ckks_linear_ms:.3f} ms ({self.avg_ckks_linear_ms/self.avg_total_ms*100:.1f}%)")
        print(f"  Bridge:          {self.avg_bridge_ms:.3f} ms ({self.avg_bridge_ms/self.avg_total_ms*100:.1f}%)")
        print(f"  TFHE LUT:        {self.avg_lut_ms:.3f} ms ({self.avg_lut_ms/self.avg_total_ms*100:.1f}%)")

        print(f"\nOperation Counts:")
        print(f"  CKKS Operations: {self.total_ckks_ops}")
        print(f"  TFHE Operations: {self.total_tfhe_ops}")
        print(f"  Bootstraps:      {self.total_bootstraps}")
        print(f"  Conversions:     {self.total_conversions}")

        print(f"\nAccuracy Metrics:")
        print(f"  Avg Quant Error: {self.avg_quantization_error:.6f}")
        print(f"  Max Quant Error: {self.max_quantization_error:.6f}")
        print(f"  Gate ON Count:   {self.gate_on_count}")
        print(f"  Gate OFF Count:  {self.gate_off_count}")

        print(f"\nThroughput:")
        print(f"  Tokens/Second:   {self.tokens_per_second:.1f}")

        # North Star metric
        hybrid_overhead_ms = self.avg_bridge_ms + self.avg_lut_ms
        print(f"\nNORTH STAR METRIC (Hybrid Overhead):")
        print(f"  Additional latency per token: {hybrid_overhead_ms:.3f} ms")

        print("=" * 70)


# =============================================================================
# Benchmark Runner
# =============================================================================

def create_random_weights(config: BenchmarkConfig) -> AdapterWeights:
    """Create random weights for benchmarking."""
    rng = np.random.default_rng(config.seed)

    return AdapterWeights(
        lora_A=rng.normal(0, 0.1, (config.rank, config.hidden_size)).astype(np.float64),
        lora_B=rng.normal(0, 0.1, (config.hidden_size, config.rank)).astype(np.float64),
        w_gate=rng.normal(0, 0.1, (config.hidden_size,)).astype(np.float64),
        b_gate=np.array([0.0]),
    )


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """Run the benchmark with given configuration."""
    results = BenchmarkResults(config=config)
    rng = np.random.default_rng(config.seed)

    # Create adapter
    adapter_config = GatedLoRAAdapterConfig(
        hidden_size=config.hidden_size,
        lora_rank=config.rank,
        lora_alpha=config.alpha,
        gate_type=config.gate_type,
        quantization_bits=config.quantization_bits,
    )

    backend = HybridHEBackend.create_simulated()
    adapter = HEGatedLoRAAdapter(adapter_config, backend)
    weights = create_random_weights(config)

    # Warmup
    print(f"Warming up ({config.warmup_tokens} tokens)...")
    for _ in range(config.warmup_tokens):
        x = rng.normal(0, 1, (config.batch_size, config.hidden_size)).astype(np.float64)
        base = rng.normal(0, 1, (config.batch_size, config.hidden_size)).astype(np.float64)
        for i in range(config.batch_size):
            adapter.forward(x[i], base[i], weights)

    # Reset backend stats
    backend.reset_stats()

    # Run benchmark
    print(f"Running benchmark ({config.num_tokens} tokens)...")
    quantization_errors = []

    for token_idx in range(config.num_tokens):
        x = rng.normal(0, 1, (config.batch_size, config.hidden_size)).astype(np.float64)
        base = rng.normal(0, 1, (config.batch_size, config.hidden_size)).astype(np.float64)

        for i in range(config.batch_size):
            output, metrics = adapter.forward(x[i], base[i], weights)

            # Record latency
            results.latency_samples.append(LatencyBreakdown.from_metrics(metrics))

            # Track accuracy
            quantization_errors.append(metrics.quantization_error)
            if metrics.gate_value >= 0.5:  # For step gate
                results.gate_on_count += 1
            else:
                results.gate_off_count += 1

            # Accumulate operation counts
            results.total_ckks_ops += metrics.ckks_ops
            results.total_tfhe_ops += metrics.tfhe_ops
            results.total_bootstraps += metrics.bootstrap_count

    # Get conversion counts from backend
    stats = backend.get_operation_stats()
    results.total_conversions = stats.bridge_to_tfhe_count + stats.bridge_to_ckks_count

    # Compute accuracy metrics
    results.avg_quantization_error = np.mean(quantization_errors)
    results.max_quantization_error = max(quantization_errors)

    # Compute aggregates
    results.compute_aggregates()

    return results


def run_benchmark_suite(output_file: Optional[str] = None):
    """Run a suite of benchmarks with various configurations."""
    configs = [
        # Small model (testing)
        BenchmarkConfig(hidden_size=64, rank=4, num_tokens=100),
        BenchmarkConfig(hidden_size=64, rank=4, num_tokens=100, gate_type="sign"),

        # Medium model
        BenchmarkConfig(hidden_size=256, rank=8, num_tokens=50),

        # Large model (Llama-7B scale)
        BenchmarkConfig(hidden_size=4096, rank=16, num_tokens=20),
        BenchmarkConfig(hidden_size=4096, rank=16, num_tokens=20, gate_type="sign"),

        # Different quantization
        BenchmarkConfig(hidden_size=64, rank=4, num_tokens=50, quantization_bits=4),
        BenchmarkConfig(hidden_size=64, rank=4, num_tokens=50, quantization_bits=12),
    ]

    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running benchmark: hidden={config.hidden_size}, rank={config.rank}, gate={config.gate_type}")
        results = run_benchmark(config)
        results.print_report()
        all_results.append(results.to_dict())

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return all_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Hybrid CKKS-TFHE Gated LoRA"
    )
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=100)
    parser.add_argument("--warmup_tokens", type=int, default=10)
    parser.add_argument("--gate_type", choices=["step", "sign"], default="step")
    parser.add_argument("--quant_bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--output", type=str, help="Output file for JSON results")

    args = parser.parse_args()

    if args.suite:
        run_benchmark_suite(args.output)
    else:
        config = BenchmarkConfig(
            hidden_size=args.hidden_size,
            rank=args.rank,
            alpha=args.alpha,
            batch_size=args.batch_size,
            num_tokens=args.num_tokens,
            warmup_tokens=args.warmup_tokens,
            gate_type=args.gate_type,
            quantization_bits=args.quant_bits,
            seed=args.seed,
        )

        results = run_benchmark(config)
        results.print_report()

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
