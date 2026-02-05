#!/usr/bin/env python3
"""
Performance Comparison: Linear HE-LoRA vs Hybrid Gated LoRA

Compares:
- Standard CKKS-only linear LoRA (existing system)
- Hybrid CKKS-TFHE gated LoRA (new capability)

Metrics:
- Per-token latency (ms)
- Throughput (tokens/second)
- Operation counts (rotations, keyswitches, TFHE ops)
- Overhead analysis
"""

import time
import json
import random
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ComparisonConfig:
    """Configuration for comparison benchmark."""
    hidden_size: int = 64
    rank: int = 4
    alpha: float = 32.0
    batch_size: int = 1
    num_tokens: int = 100
    warmup_tokens: int = 10
    seed: int = 42

    # Gated LoRA specific
    gate_type: str = "step"
    quantization_bits: int = 8


# =============================================================================
# Simulated Backends (since we can't run real HE without numpy/dependencies)
# =============================================================================

def matvec(matrix: List[List[float]], vec: List[float]) -> List[float]:
    """Matrix-vector multiply."""
    result = []
    for row in matrix:
        val = sum(a * b for a, b in zip(row, vec))
        result.append(val)
    return result


def dot(a: List[float], b: List[float]) -> float:
    """Dot product."""
    return sum(x * y for x, y in zip(a, b))


def percentile(data: List[float], p: float) -> float:
    """Compute percentile."""
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


class SimulatedLinearLoRA:
    """Simulated linear LoRA (CKKS-only) for benchmarking."""

    def __init__(self, hidden_size: int, rank: int, alpha: float):
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Simulated operation costs (microseconds) based on real measurements
        # From docs/results/BENCHMARK_RESULTS.md
        self._encrypt_us = 50    # Encrypt input
        self._matvec_a_us = 200  # x @ A^T (CKKS ct-pt multiply)
        self._matvec_b_us = 200  # u @ B^T (CKKS ct-pt multiply)
        self._rescale_us = 50   # Rescale after multiply
        self._decrypt_us = 40   # Decrypt result

        # MOAI optimization eliminates rotations
        self._rotation_us = 0

        # Counters
        self.total_ops = 0
        self.total_rotations = 0
        self.total_keyswitches = 0
        self.total_rescales = 0

    def forward(
        self,
        x: List[float],
        base_output: List[float],
        lora_A: List[List[float]],
        lora_B: List[List[float]],
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Run forward pass and return output with timing."""
        timing = {
            'encrypt_us': 0,
            'compute_us': 0,
            'decrypt_us': 0,
            'total_us': 0,
        }

        t_start = time.perf_counter_ns()

        # Simulate encrypt
        time.sleep(self._encrypt_us / 1e6)
        timing['encrypt_us'] = self._encrypt_us

        # Simulate CKKS computation
        t0 = time.perf_counter_ns()

        # u = A @ x (ct-pt multiply + rescale)
        u = matvec(lora_A, x)
        time.sleep((self._matvec_a_us + self._rescale_us) / 1e6)

        # delta = B @ u (ct-pt multiply + rescale)
        delta = matvec(lora_B, u)
        time.sleep((self._matvec_b_us + self._rescale_us) / 1e6)

        timing['compute_us'] = (time.perf_counter_ns() - t0) // 1000

        # Scale and add
        output = [b + self.scaling * d for b, d in zip(base_output, delta)]

        # Simulate decrypt
        time.sleep(self._decrypt_us / 1e6)
        timing['decrypt_us'] = self._decrypt_us

        timing['total_us'] = (time.perf_counter_ns() - t_start) // 1000

        # Update counters
        self.total_ops += 1
        self.total_keyswitches += 2  # Two ct-pt multiplies
        self.total_rescales += 2

        return output, timing


class SimulatedGatedLoRA:
    """Simulated gated LoRA (hybrid CKKS-TFHE) for benchmarking."""

    def __init__(
        self,
        hidden_size: int,
        rank: int,
        alpha: float,
        gate_type: str = "step",
        quant_bits: int = 8,
    ):
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.gate_type = gate_type
        self.quant_bits = quant_bits

        # CKKS costs (same as linear)
        self._encrypt_us = 50
        self._matvec_a_us = 200
        self._matvec_b_us = 200
        self._rescale_us = 50
        self._decrypt_us = 40

        # Gate computation (CKKS)
        self._gate_pre_us = 100  # w_g @ x + b_g

        # Bridge costs (interactive round-trip)
        # Based on realistic network + crypto overhead
        self._bridge_ckks_to_tfhe_us = 15000  # ~15ms decrypt + quantize + encrypt
        self._bridge_tfhe_to_ckks_us = 15000  # ~15ms decrypt + dequantize + encrypt

        # TFHE costs
        # Programmable bootstrap is the dominant cost
        self._tfhe_bootstrap_us = 10000  # ~10ms for LUT evaluation

        # Counters
        self.total_ops = 0
        self.total_ckks_ops = 0
        self.total_tfhe_ops = 0
        self.total_bootstraps = 0
        self.total_conversions = 0
        self.gate_on_count = 0
        self.gate_off_count = 0

    def forward(
        self,
        x: List[float],
        base_output: List[float],
        lora_A: List[List[float]],
        lora_B: List[List[float]],
        w_gate: List[float],
        b_gate: float = 0.0,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Run forward pass and return output with timing."""
        timing = {
            'ckks_lora_us': 0,
            'ckks_gate_pre_us': 0,
            'bridge_to_tfhe_us': 0,
            'tfhe_lut_us': 0,
            'bridge_to_ckks_us': 0,
            'ckks_apply_gate_us': 0,
            'total_us': 0,
            'gate_value': 0.0,
        }

        t_start = time.perf_counter_ns()

        # Phase 1: CKKS LoRA delta
        t0 = time.perf_counter_ns()
        u = matvec(lora_A, x)
        delta = matvec(lora_B, u)
        time.sleep((self._matvec_a_us + self._matvec_b_us + 2 * self._rescale_us) / 1e6)
        timing['ckks_lora_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 2: CKKS gate pre-activation
        t0 = time.perf_counter_ns()
        z = dot(w_gate, x) + b_gate
        time.sleep(self._gate_pre_us / 1e6)
        timing['ckks_gate_pre_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 3: Bridge CKKS -> TFHE
        t0 = time.perf_counter_ns()
        time.sleep(self._bridge_ckks_to_tfhe_us / 1e6)
        timing['bridge_to_tfhe_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 4: TFHE LUT evaluation (programmable bootstrap)
        t0 = time.perf_counter_ns()
        if self.gate_type == "step":
            g = 1.0 if z >= 0 else 0.0
        else:  # sign
            g = 1.0 if z > 0 else (-1.0 if z < 0 else 0.0)
        time.sleep(self._tfhe_bootstrap_us / 1e6)
        timing['tfhe_lut_us'] = (time.perf_counter_ns() - t0) // 1000
        timing['gate_value'] = g

        # Phase 5: Bridge TFHE -> CKKS
        t0 = time.perf_counter_ns()
        time.sleep(self._bridge_tfhe_to_ckks_us / 1e6)
        timing['bridge_to_ckks_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 6: CKKS apply gate and final add
        t0 = time.perf_counter_ns()
        gated_delta = [g * self.scaling * d for d in delta]
        output = [b + gd for b, gd in zip(base_output, gated_delta)]
        time.sleep(self._rescale_us / 1e6)
        timing['ckks_apply_gate_us'] = (time.perf_counter_ns() - t0) // 1000

        timing['total_us'] = (time.perf_counter_ns() - t_start) // 1000

        # Update counters
        self.total_ops += 1
        self.total_ckks_ops += 4  # 2 matvec + gate_pre + apply_gate
        self.total_tfhe_ops += 1
        self.total_bootstraps += 1
        self.total_conversions += 2
        if g >= 0.5:
            self.gate_on_count += 1
        else:
            self.gate_off_count += 1

        return output, timing


# =============================================================================
# Benchmark Results
# =============================================================================

@dataclass
class LinearLoRAResults:
    """Results for linear LoRA benchmark."""
    config: ComparisonConfig
    latencies_us: List[float] = field(default_factory=list)

    # Aggregated stats
    avg_latency_us: float = 0.0
    p50_latency_us: float = 0.0
    p90_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0

    # Breakdown
    avg_encrypt_us: float = 0.0
    avg_compute_us: float = 0.0
    avg_decrypt_us: float = 0.0

    # Operations
    total_rotations: int = 0
    total_keyswitches: int = 0
    total_rescales: int = 0

    # Throughput
    tokens_per_second: float = 0.0

    def compute_stats(self):
        if not self.latencies_us:
            return
        self.avg_latency_us = sum(self.latencies_us) / len(self.latencies_us)
        self.p50_latency_us = percentile(self.latencies_us, 50)
        self.p90_latency_us = percentile(self.latencies_us, 90)
        self.p99_latency_us = percentile(self.latencies_us, 99)
        self.min_latency_us = min(self.latencies_us)
        self.max_latency_us = max(self.latencies_us)

        total_time_s = sum(self.latencies_us) / 1e6
        if total_time_s > 0:
            self.tokens_per_second = len(self.latencies_us) / total_time_s


@dataclass
class GatedLoRAResults:
    """Results for gated LoRA benchmark."""
    config: ComparisonConfig
    latencies_us: List[float] = field(default_factory=list)

    # Aggregated stats
    avg_latency_us: float = 0.0
    p50_latency_us: float = 0.0
    p90_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0

    # Breakdown
    avg_ckks_lora_us: float = 0.0
    avg_ckks_gate_pre_us: float = 0.0
    avg_bridge_us: float = 0.0
    avg_tfhe_lut_us: float = 0.0

    # Operations
    total_ckks_ops: int = 0
    total_tfhe_ops: int = 0
    total_bootstraps: int = 0
    total_conversions: int = 0

    # Gate stats
    gate_on_count: int = 0
    gate_off_count: int = 0

    # Throughput
    tokens_per_second: float = 0.0

    def compute_stats(self):
        if not self.latencies_us:
            return
        self.avg_latency_us = sum(self.latencies_us) / len(self.latencies_us)
        self.p50_latency_us = percentile(self.latencies_us, 50)
        self.p90_latency_us = percentile(self.latencies_us, 90)
        self.p99_latency_us = percentile(self.latencies_us, 99)
        self.min_latency_us = min(self.latencies_us)
        self.max_latency_us = max(self.latencies_us)

        total_time_s = sum(self.latencies_us) / 1e6
        if total_time_s > 0:
            self.tokens_per_second = len(self.latencies_us) / total_time_s


# =============================================================================
# Helper: Random Matrix/Vector Generation
# =============================================================================

def random_vector(size: int, mean: float = 0.0, std: float = 1.0) -> List[float]:
    """Generate a random vector with normal distribution (Box-Muller)."""
    result = []
    for _ in range((size + 1) // 2):
        u1 = random.random()
        u2 = random.random()
        while u1 == 0:
            u1 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
        z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(2 * math.pi * u2)
        result.append(mean + std * z0)
        if len(result) < size:
            result.append(mean + std * z1)
    return result[:size]


def random_matrix(rows: int, cols: int, mean: float = 0.0, std: float = 0.1) -> List[List[float]]:
    """Generate a random matrix with normal distribution."""
    return [random_vector(cols, mean, std) for _ in range(rows)]


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_linear_lora_benchmark(config: ComparisonConfig) -> LinearLoRAResults:
    """Run linear LoRA benchmark."""
    results = LinearLoRAResults(config=config)
    random.seed(config.seed)

    # Create adapter
    adapter = SimulatedLinearLoRA(
        hidden_size=config.hidden_size,
        rank=config.rank,
        alpha=config.alpha,
    )

    # Create weights
    lora_A = random_matrix(config.rank, config.hidden_size)
    lora_B = random_matrix(config.hidden_size, config.rank)

    # Warmup
    for _ in range(config.warmup_tokens):
        x = random_vector(config.hidden_size)
        base = random_vector(config.hidden_size)
        adapter.forward(x, base, lora_A, lora_B)

    # Reset counters after warmup
    adapter.total_ops = 0
    adapter.total_rotations = 0
    adapter.total_keyswitches = 0
    adapter.total_rescales = 0

    # Benchmark
    encrypt_times = []
    compute_times = []
    decrypt_times = []

    for _ in range(config.num_tokens):
        x = random_vector(config.hidden_size)
        base = random_vector(config.hidden_size)

        _, timing = adapter.forward(x, base, lora_A, lora_B)

        results.latencies_us.append(timing['total_us'])
        encrypt_times.append(timing['encrypt_us'])
        compute_times.append(timing['compute_us'])
        decrypt_times.append(timing['decrypt_us'])

    # Compute stats
    results.compute_stats()
    results.avg_encrypt_us = sum(encrypt_times) / len(encrypt_times) if encrypt_times else 0
    results.avg_compute_us = sum(compute_times) / len(compute_times) if compute_times else 0
    results.avg_decrypt_us = sum(decrypt_times) / len(decrypt_times) if decrypt_times else 0
    results.total_rotations = adapter.total_rotations
    results.total_keyswitches = adapter.total_keyswitches
    results.total_rescales = adapter.total_rescales

    return results


def run_gated_lora_benchmark(config: ComparisonConfig) -> GatedLoRAResults:
    """Run gated LoRA benchmark."""
    results = GatedLoRAResults(config=config)
    random.seed(config.seed + 1)  # Different seed to get different random state

    # Create adapter
    adapter = SimulatedGatedLoRA(
        hidden_size=config.hidden_size,
        rank=config.rank,
        alpha=config.alpha,
        gate_type=config.gate_type,
        quant_bits=config.quantization_bits,
    )

    # Create weights
    lora_A = random_matrix(config.rank, config.hidden_size)
    lora_B = random_matrix(config.hidden_size, config.rank)
    w_gate = random_vector(config.hidden_size, 0.0, 0.1)
    b_gate = 0.0

    # Warmup
    for _ in range(config.warmup_tokens):
        x = random_vector(config.hidden_size)
        base = random_vector(config.hidden_size)
        adapter.forward(x, base, lora_A, lora_B, w_gate, b_gate)

    # Reset counters
    adapter.total_ops = 0
    adapter.total_ckks_ops = 0
    adapter.total_tfhe_ops = 0
    adapter.total_bootstraps = 0
    adapter.total_conversions = 0
    adapter.gate_on_count = 0
    adapter.gate_off_count = 0

    # Benchmark
    ckks_lora_times = []
    ckks_gate_pre_times = []
    bridge_times = []
    tfhe_lut_times = []

    for _ in range(config.num_tokens):
        x = random_vector(config.hidden_size)
        base = random_vector(config.hidden_size)

        _, timing = adapter.forward(x, base, lora_A, lora_B, w_gate, b_gate)

        results.latencies_us.append(timing['total_us'])
        ckks_lora_times.append(timing['ckks_lora_us'])
        ckks_gate_pre_times.append(timing['ckks_gate_pre_us'])
        bridge_times.append(timing['bridge_to_tfhe_us'] + timing['bridge_to_ckks_us'])
        tfhe_lut_times.append(timing['tfhe_lut_us'])

    # Compute stats
    results.compute_stats()
    results.avg_ckks_lora_us = sum(ckks_lora_times) / len(ckks_lora_times) if ckks_lora_times else 0
    results.avg_ckks_gate_pre_us = sum(ckks_gate_pre_times) / len(ckks_gate_pre_times) if ckks_gate_pre_times else 0
    results.avg_bridge_us = sum(bridge_times) / len(bridge_times) if bridge_times else 0
    results.avg_tfhe_lut_us = sum(tfhe_lut_times) / len(tfhe_lut_times) if tfhe_lut_times else 0
    results.total_ckks_ops = adapter.total_ckks_ops
    results.total_tfhe_ops = adapter.total_tfhe_ops
    results.total_bootstraps = adapter.total_bootstraps
    results.total_conversions = adapter.total_conversions
    results.gate_on_count = adapter.gate_on_count
    results.gate_off_count = adapter.gate_off_count

    return results


def print_comparison_report(
    config: ComparisonConfig,
    linear_results: LinearLoRAResults,
    gated_results: GatedLoRAResults,
):
    """Print formatted comparison report."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: LINEAR HE-LoRA vs HYBRID GATED LoRA")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Hidden Size:        {config.hidden_size}")
    print(f"  LoRA Rank:          {config.rank}")
    print(f"  Alpha:              {config.alpha}")
    print(f"  Tokens Measured:    {config.num_tokens}")
    print(f"  Gate Type:          {config.gate_type}")

    print("\n" + "-" * 80)
    print("LATENCY COMPARISON (microseconds)")
    print("-" * 80)
    print(f"{'Metric':<25} {'Linear LoRA':>15} {'Gated LoRA':>15} {'Overhead':>15}")
    print("-" * 80)

    def overhead_str(linear, gated):
        diff = gated - linear
        pct = (diff / linear * 100) if linear > 0 else 0
        return f"+{diff:.0f} ({pct:+.1f}%)"

    print(f"{'Average':<25} {linear_results.avg_latency_us:>15.1f} {gated_results.avg_latency_us:>15.1f} {overhead_str(linear_results.avg_latency_us, gated_results.avg_latency_us):>15}")
    print(f"{'P50':<25} {linear_results.p50_latency_us:>15.1f} {gated_results.p50_latency_us:>15.1f} {overhead_str(linear_results.p50_latency_us, gated_results.p50_latency_us):>15}")
    print(f"{'P90':<25} {linear_results.p90_latency_us:>15.1f} {gated_results.p90_latency_us:>15.1f} {overhead_str(linear_results.p90_latency_us, gated_results.p90_latency_us):>15}")
    print(f"{'P99':<25} {linear_results.p99_latency_us:>15.1f} {gated_results.p99_latency_us:>15.1f} {overhead_str(linear_results.p99_latency_us, gated_results.p99_latency_us):>15}")
    print(f"{'Min':<25} {linear_results.min_latency_us:>15.1f} {gated_results.min_latency_us:>15.1f} {overhead_str(linear_results.min_latency_us, gated_results.min_latency_us):>15}")
    print(f"{'Max':<25} {linear_results.max_latency_us:>15.1f} {gated_results.max_latency_us:>15.1f} {overhead_str(linear_results.max_latency_us, gated_results.max_latency_us):>15}")

    print("\n" + "-" * 80)
    print("LATENCY BREAKDOWN (microseconds)")
    print("-" * 80)
    print(f"{'Component':<30} {'Linear LoRA':>20} {'Gated LoRA':>20}")
    print("-" * 80)

    # Linear breakdown
    print(f"{'Encrypt':<30} {linear_results.avg_encrypt_us:>20.1f} {'-':>20}")
    print(f"{'CKKS Compute':<30} {linear_results.avg_compute_us:>20.1f} {'-':>20}")
    print(f"{'Decrypt':<30} {linear_results.avg_decrypt_us:>20.1f} {'-':>20}")

    # Gated breakdown
    print(f"{'CKKS LoRA (A@x, B@u)':<30} {'-':>20} {gated_results.avg_ckks_lora_us:>20.1f}")
    print(f"{'CKKS Gate Pre (w_g@x)':<30} {'-':>20} {gated_results.avg_ckks_gate_pre_us:>20.1f}")
    print(f"{'Bridge (CKKS<->TFHE)':<30} {'-':>20} {gated_results.avg_bridge_us:>20.1f}")
    print(f"{'TFHE LUT (bootstrap)':<30} {'-':>20} {gated_results.avg_tfhe_lut_us:>20.1f}")

    # Overhead breakdown
    hybrid_overhead = gated_results.avg_latency_us - linear_results.avg_latency_us
    print(f"\n{'HYBRID OVERHEAD':<30} {'':>20} {hybrid_overhead:>20.1f}")
    print(f"{'  - Bridge contribution':<30} {'':>20} {gated_results.avg_bridge_us:>20.1f} ({gated_results.avg_bridge_us/hybrid_overhead*100:.1f}%)")
    print(f"{'  - TFHE contribution':<30} {'':>20} {gated_results.avg_tfhe_lut_us:>20.1f} ({gated_results.avg_tfhe_lut_us/hybrid_overhead*100:.1f}%)")

    print("\n" + "-" * 80)
    print("OPERATION COUNTS (per token)")
    print("-" * 80)
    print(f"{'Operation':<30} {'Linear LoRA':>20} {'Gated LoRA':>20}")
    print("-" * 80)

    n_tokens = config.num_tokens
    print(f"{'CKKS Rotations':<30} {linear_results.total_rotations/n_tokens:>20.1f} {0:>20.1f}")
    print(f"{'CKKS Keyswitches':<30} {linear_results.total_keyswitches/n_tokens:>20.1f} {gated_results.total_ckks_ops/n_tokens:>20.1f}")
    print(f"{'CKKS Rescales':<30} {linear_results.total_rescales/n_tokens:>20.1f} {'-':>20}")
    print(f"{'TFHE Operations':<30} {0:>20.1f} {gated_results.total_tfhe_ops/n_tokens:>20.1f}")
    print(f"{'TFHE Bootstraps':<30} {0:>20.1f} {gated_results.total_bootstraps/n_tokens:>20.1f}")
    print(f"{'Bridge Conversions':<30} {0:>20.1f} {gated_results.total_conversions/n_tokens:>20.1f}")

    print("\n" + "-" * 80)
    print("THROUGHPUT")
    print("-" * 80)
    print(f"{'Metric':<30} {'Linear LoRA':>20} {'Gated LoRA':>20}")
    print("-" * 80)
    print(f"{'Tokens/Second':<30} {linear_results.tokens_per_second:>20.1f} {gated_results.tokens_per_second:>20.1f}")
    print(f"{'ms/Token':<30} {linear_results.avg_latency_us/1000:>20.3f} {gated_results.avg_latency_us/1000:>20.3f}")

    # Gate statistics
    print("\n" + "-" * 80)
    print("GATE STATISTICS (Gated LoRA only)")
    print("-" * 80)
    gate_on_pct = gated_results.gate_on_count / (gated_results.gate_on_count + gated_results.gate_off_count) * 100
    print(f"  Gate ON:  {gated_results.gate_on_count:>6} ({gate_on_pct:.1f}%)")
    print(f"  Gate OFF: {gated_results.gate_off_count:>6} ({100-gate_on_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    overhead_ms = (gated_results.avg_latency_us - linear_results.avg_latency_us) / 1000
    overhead_pct = (gated_results.avg_latency_us / linear_results.avg_latency_us - 1) * 100
    print(f"\n  Hybrid Gated LoRA adds {overhead_ms:.2f} ms ({overhead_pct:.1f}%) overhead per token")
    print(f"  Primary cost: Interactive bridge ({gated_results.avg_bridge_us/1000:.2f} ms)")
    print(f"  Secondary cost: TFHE bootstrap ({gated_results.avg_tfhe_lut_us/1000:.2f} ms)")
    print(f"\n  Trade-off: Discrete gating enables conditional adapter application")
    print(f"  Use case: Input-dependent LoRA activation, mixture-of-adapters routing")

    print("=" * 80)


def run_comparison_suite():
    """Run full comparison suite across different configurations."""
    configs = [
        # Small model (testing)
        ComparisonConfig(hidden_size=64, rank=4, num_tokens=50),
        # Medium model
        ComparisonConfig(hidden_size=256, rank=8, num_tokens=50),
        # Large model (Llama-7B scale)
        ComparisonConfig(hidden_size=4096, rank=16, num_tokens=20),
    ]

    all_results = []

    for config in configs:
        print(f"\n\nRunning benchmark: hidden={config.hidden_size}, rank={config.rank}")
        print("-" * 60)

        print("  Running Linear LoRA...")
        linear_results = run_linear_lora_benchmark(config)

        print("  Running Gated LoRA...")
        gated_results = run_gated_lora_benchmark(config)

        print_comparison_report(config, linear_results, gated_results)

        all_results.append({
            "config": asdict(config),
            "linear": {
                "avg_latency_us": linear_results.avg_latency_us,
                "p50_latency_us": linear_results.p50_latency_us,
                "p90_latency_us": linear_results.p90_latency_us,
                "tokens_per_second": linear_results.tokens_per_second,
            },
            "gated": {
                "avg_latency_us": gated_results.avg_latency_us,
                "p50_latency_us": gated_results.p50_latency_us,
                "p90_latency_us": gated_results.p90_latency_us,
                "tokens_per_second": gated_results.tokens_per_second,
                "avg_bridge_us": gated_results.avg_bridge_us,
                "avg_tfhe_lut_us": gated_results.avg_tfhe_lut_us,
            },
        })

    return all_results


def main():
    """Main entry point."""
    print("=" * 80)
    print("HE-LoRA Performance Comparison Benchmark")
    print("Linear (CKKS-only) vs Hybrid (CKKS + TFHE) Gated LoRA")
    print("=" * 80)

    results = run_comparison_suite()

    print("\n\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
