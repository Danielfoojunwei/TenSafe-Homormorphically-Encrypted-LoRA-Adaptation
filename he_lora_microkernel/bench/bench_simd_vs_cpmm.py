"""
Direct Comparison: SIMD Speculative Batching vs MOAI CPMM

This benchmark directly compares the two approaches:
1. SIMD Speculative Batching (current system)
2. MOAI CPMM (new implementation)

Both optimize HE-LoRA, but use different packing strategies.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
import time

from he_lora_microkernel.compiler.moai_cpmm import (
    CPMMKernel,
    LoRACPMMKernel,
)
from he_lora_microkernel.compiler import (
    LoRAConfig,
    LoRATargets,
    CKKSProfile,
    get_profile,
    compile_schedule,
    pack_activations,
    unpack_activations,
)
from he_lora_microkernel.runtime import HELoRAExecutor
from he_lora_microkernel.backend.gpu_ckks_backend import BackendType


@dataclass
class ComparisonResult:
    """Result of comparing the two approaches."""
    hidden_size: int
    rank: int
    batch_size: int

    # SIMD Speculative Batching
    simd_rotations_per_token: float
    simd_ct_pt_muls: int
    simd_throughput: float  # tok/s
    simd_latency_ms: float

    # MOAI CPMM
    cpmm_rotations_per_token: float
    cpmm_ct_pt_muls: int
    cpmm_throughput: float  # tok/s
    cpmm_latency_ms: float

    # Comparison
    rotation_reduction: float  # percentage
    speedup: float


def benchmark_simd_batching(
    hidden_size: int,
    rank: int,
    batch_size: int,
    num_tokens: int = 50,
) -> Dict[str, Any]:
    """Benchmark SIMD Speculative Batching approach."""

    config = LoRAConfig(
        hidden_size=hidden_size,
        rank=rank,
        alpha=2.0 * rank,
        targets=LoRATargets.QKV,
        batch_size=batch_size,
        max_context_length=2048,
        ckks_profile=CKKSProfile.FAST,
    )

    try:
        ckks_params = get_profile(config.ckks_profile)
        schedule = compile_schedule(config, ckks_params)
    except ValueError as e:
        return {'error': str(e)}

    # Create executor
    executor = HELoRAExecutor(schedule, BackendType.SIMULATION)

    # Load weights
    rng = np.random.default_rng(42)
    A = rng.standard_normal((hidden_size, rank)).astype(np.float64) * 0.01
    B = rng.standard_normal((rank, hidden_size)).astype(np.float64) * 0.01
    executor.load_weights(A, B, config.alpha)

    # Warm up
    x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)
    for _ in range(5):
        executor.execute_token(x)

    # Benchmark
    executor.reset_statistics()
    start = time.perf_counter()

    for i in range(num_tokens):
        x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)
        executor.execute_token(x, position=i)

    elapsed = time.perf_counter() - start

    stats = executor.get_statistics()
    layout = schedule.layout

    return {
        'rotations_per_matmul': layout.total_rotations_per_matmul,
        'rotations_per_token': layout.total_rotations_per_matmul * 2,  # Two matmuls for LoRA
        'ct_pt_muls': layout.num_blocks * 2,  # Approximate
        'throughput': stats['tokens_per_second'],
        'latency_ms': stats['avg_time_per_token_ms'],
        'num_blocks': layout.num_blocks,
        'block_size': layout.block_size,
        'intra_block_rotations': layout.intra_block_rotations,
        'cross_block_rotations': layout.cross_block_rotations,
    }


def benchmark_moai_cpmm(
    hidden_size: int,
    rank: int,
    batch_size: int,
    slot_count: int = 8192,
    num_tokens: int = 50,
) -> Dict[str, Any]:
    """Benchmark MOAI CPMM approach."""

    try:
        kernel = LoRACPMMKernel(
            hidden_size=hidden_size,
            rank=rank,
            slot_count=slot_count,
            batch_size=batch_size,
        )
    except ValueError as e:
        return {'error': str(e)}

    # Load weights
    rng = np.random.default_rng(42)
    A = rng.standard_normal((hidden_size, rank)).astype(np.float64) * 0.01
    B = rng.standard_normal((rank, hidden_size)).astype(np.float64) * 0.01
    alpha = 2.0 * rank
    kernel.load_weights(A, B, alpha)

    # Warm up
    x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)
    for _ in range(5):
        kernel.compute_delta(x)

    # Benchmark
    start = time.perf_counter()

    for _ in range(num_tokens):
        x = rng.standard_normal((batch_size, hidden_size)).astype(np.float64)
        kernel.compute_delta(x)

    elapsed = time.perf_counter() - start

    layout = kernel.cpmm.layout

    return {
        'rotations_per_matmul': layout.rotations_per_matmul,
        'rotations_per_token': layout.rotations_per_matmul,  # Single AB@x
        'ct_pt_muls': layout.ct_pt_muls_per_matmul,
        'throughput': num_tokens / elapsed,
        'latency_ms': (elapsed / num_tokens) * 1000,
        'num_blocks': kernel.cpmm.config.num_blocks,
        'block_size': kernel.cpmm.config.block_size,
        'accumulation_rotations': layout.accumulation_rotations,
    }


def compare_approaches(
    hidden_size: int,
    rank: int,
    batch_size: int,
    slot_count: int = 8192,
) -> ComparisonResult:
    """Compare SIMD Batching vs MOAI CPMM."""

    print(f"\nBenchmarking h={hidden_size}, r={rank}, b={batch_size}...")

    # Benchmark SIMD
    simd = benchmark_simd_batching(hidden_size, rank, batch_size)
    if 'error' in simd:
        print(f"  SIMD: FAILED - {simd['error']}")
        simd_rot = float('inf')
        simd_tp = 0
        simd_lat = float('inf')
        simd_muls = 0
    else:
        simd_rot = simd['rotations_per_token']
        simd_tp = simd['throughput']
        simd_lat = simd['latency_ms']
        simd_muls = simd['ct_pt_muls']
        print(f"  SIMD Batching: {simd_rot} rot/tok, {simd_tp:.1f} tok/s")

    # Benchmark CPMM
    cpmm = benchmark_moai_cpmm(hidden_size, rank, batch_size, slot_count)
    if 'error' in cpmm:
        print(f"  CPMM: FAILED - {cpmm['error']}")
        cpmm_rot = float('inf')
        cpmm_tp = 0
        cpmm_lat = float('inf')
        cpmm_muls = 0
    else:
        cpmm_rot = cpmm['rotations_per_token']
        cpmm_tp = cpmm['throughput']
        cpmm_lat = cpmm['latency_ms']
        cpmm_muls = cpmm['ct_pt_muls']
        print(f"  MOAI CPMM:     {cpmm_rot} rot/tok, {cpmm_tp:.1f} tok/s")

    # Compute comparison metrics
    if simd_rot > 0 and cpmm_rot < float('inf'):
        rotation_reduction = (1 - cpmm_rot / simd_rot) * 100 if simd_rot > 0 else 0
    else:
        rotation_reduction = 0

    speedup = cpmm_tp / simd_tp if simd_tp > 0 else 0

    return ComparisonResult(
        hidden_size=hidden_size,
        rank=rank,
        batch_size=batch_size,
        simd_rotations_per_token=simd_rot,
        simd_ct_pt_muls=simd_muls,
        simd_throughput=simd_tp,
        simd_latency_ms=simd_lat,
        cpmm_rotations_per_token=cpmm_rot,
        cpmm_ct_pt_muls=cpmm_muls,
        cpmm_throughput=cpmm_tp,
        cpmm_latency_ms=cpmm_lat,
        rotation_reduction=rotation_reduction,
        speedup=speedup,
    )


def theoretical_comparison(
    hidden_size: int,
    rank: int,
    batch_size: int,
    slot_count: int = 8192,
) -> Dict[str, Any]:
    """
    Compare theoretical HE operation costs (not simulation throughput).

    This is what matters for real HE execution where rotations dominate cost.
    """
    # SIMD Speculative Batching theoretical costs
    # Uses batch-first packing with pre-computed AB
    simd_block_size = 512  # Typical block size
    simd_num_blocks = math.ceil(hidden_size / simd_block_size)

    # Cross-block rotations for accumulation
    simd_cross_block_rot = math.ceil(math.log2(simd_num_blocks)) + 1 if simd_num_blocks > 1 else 0

    # Total rotations per token (for full LoRA with QKV = 3 projections)
    simd_rot_per_projection = simd_cross_block_rot
    simd_rot_per_token = simd_rot_per_projection * 3  # Q, K, V

    # Ct×Pt multiplications
    simd_ctpt_per_projection = simd_num_blocks
    simd_ctpt_per_token = simd_ctpt_per_projection * 3

    # MOAI CPMM theoretical costs
    # Uses column-major blocks with weight-slot alignment
    try:
        cpmm = CPMMKernel(
            out_size=hidden_size,
            in_size=hidden_size,
            slot_count=slot_count,
            batch_size=batch_size,
        )
        cpmm_num_blocks = cpmm.config.num_blocks
        cpmm_rot_per_matmul = cpmm.layout.rotations_per_matmul
        cpmm_ctpt_per_matmul = cpmm.layout.ct_pt_muls_per_matmul
        cpmm_error = None
    except ValueError as e:
        cpmm_num_blocks = 0
        cpmm_rot_per_matmul = float('inf')
        cpmm_ctpt_per_matmul = 0
        cpmm_error = str(e)

    # For LoRA: single AB@x matmul per projection (pre-computed AB)
    cpmm_rot_per_projection = cpmm_rot_per_matmul
    cpmm_rot_per_token = cpmm_rot_per_projection * 3  # Q, K, V
    cpmm_ctpt_per_token = cpmm_ctpt_per_matmul * 3

    return {
        'hidden_size': hidden_size,
        'rank': rank,
        'batch_size': batch_size,
        'simd': {
            'num_blocks': simd_num_blocks,
            'rotations_per_projection': simd_rot_per_projection,
            'rotations_per_token': simd_rot_per_token,
            'ct_pt_muls_per_token': simd_ctpt_per_token,
            'description': 'Batch-first packing, tree reduction for accumulation',
        },
        'cpmm': {
            'num_blocks': cpmm_num_blocks,
            'rotations_per_projection': cpmm_rot_per_projection,
            'rotations_per_token': cpmm_rot_per_token,
            'ct_pt_muls_per_token': cpmm_ctpt_per_token,
            'error': cpmm_error,
            'description': 'Column-packed weights aligned with input slots',
        },
        'comparison': {
            'rotation_reduction': (1 - cpmm_rot_per_token / simd_rot_per_token) * 100 if simd_rot_per_token > 0 and cpmm_rot_per_token < float('inf') else 0,
            'rotation_speedup': simd_rot_per_token / cpmm_rot_per_token if cpmm_rot_per_token > 0 else float('inf'),
        }
    }


def run_full_comparison():
    """Run full comparison across multiple configurations."""

    print("=" * 80)
    print("SIMD SPECULATIVE BATCHING vs MOAI CPMM")
    print("Theoretical HE Operation Cost Comparison")
    print("=" * 80)

    # Test configurations
    configs = [
        # (hidden_size, rank, batch_size)
        (256, 16, 4),
        (512, 16, 4),
        (512, 16, 8),
        (1024, 16, 4),
        (1024, 16, 8),
        (2048, 16, 4),
        (4096, 16, 4),   # Llama 8B scale
        (7168, 16, 4),   # Kimi 2.5 scale
    ]

    results = []

    for h, r, b in configs:
        result = theoretical_comparison(h, r, b)
        results.append(result)

    # Print summary table
    print("\n" + "=" * 110)
    print("THEORETICAL HE OPERATION COST COMPARISON (QKV LoRA)")
    print("=" * 110)

    print(f"\n{'Config':<22} {'SIMD Blocks':<12} {'SIMD Rot/tok':<14} {'CPMM Blocks':<12} {'CPMM Rot/tok':<14} {'Reduction':<12} {'Speedup':<10}")
    print("-" * 110)

    for r in results:
        config = f"h={r['hidden_size']},r={r['rank']},b={r['batch_size']}"
        simd_blocks = r['simd']['num_blocks']
        simd_rot = r['simd']['rotations_per_token']
        cpmm_blocks = r['cpmm']['num_blocks'] if r['cpmm']['error'] is None else "N/A"
        cpmm_rot = r['cpmm']['rotations_per_token'] if r['cpmm']['error'] is None else "N/A"
        reduction = f"{r['comparison']['rotation_reduction']:.1f}%" if r['comparison']['rotation_reduction'] > 0 else "N/A"
        speedup = f"{r['comparison']['rotation_speedup']:.1f}x" if r['comparison']['rotation_speedup'] > 0 and r['comparison']['rotation_speedup'] < float('inf') else "∞"

        cpmm_rot_str = f"{cpmm_rot}" if isinstance(cpmm_rot, (int, float)) and cpmm_rot < float('inf') else "N/A"
        cpmm_blocks_str = f"{cpmm_blocks}" if isinstance(cpmm_blocks, int) else cpmm_blocks

        print(f"{config:<22} {simd_blocks:<12} {simd_rot:<14} {cpmm_blocks_str:<12} {cpmm_rot_str:<14} {reduction:<12} {speedup:<10}")

    print("-" * 110)

    # Show which configs CPMM can't handle (slot constraints)
    print("\nNOTE: CPMM requires h × b ≤ slot_count. Larger configs need N=32768+ CKKS profile.")

    # Summary
    valid = [r for r in results if r['cpmm']['error'] is None and r['simd']['rotations_per_token'] > 0]
    if valid:
        # Find best case
        best = max(valid, key=lambda x: x['comparison']['rotation_speedup'] if x['comparison']['rotation_speedup'] < float('inf') else 0)
        print(f"\nBest case: h={best['hidden_size']}, b={best['batch_size']}")
        print(f"  SIMD: {best['simd']['rotations_per_token']} rot/tok")
        print(f"  CPMM: {best['cpmm']['rotations_per_token']} rot/tok")
        if best['comparison']['rotation_speedup'] < float('inf'):
            print(f"  Speedup: {best['comparison']['rotation_speedup']:.1f}x fewer rotations")

    # Explain the key differences
    print("\n" + "=" * 100)
    print("KEY DIFFERENCES EXPLAINED")
    print("=" * 100)

    print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SIMD SPECULATIVE BATCHING vs MOAI CPMM                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SIMD SPECULATIVE BATCHING (Current):                                           │
│  ────────────────────────────────────                                           │
│  • Packing: Batch-first layout                                                  │
│    Slots: [b0_ch0, b1_ch0, ..., b0_ch1, b1_ch1, ...]                           │
│                                                                                 │
│  • Strategy: Pre-compute AB, pack in blocks, tree reduction                     │
│                                                                                 │
│  • Rotations: log2(blocks) for cross-block accumulation                         │
│    - Still needs to gather values from different positions                      │
│    - Rotations scale with hidden_size                                           │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MOAI CPMM (New):                                                               │
│  ────────────────                                                               │
│  • Packing: Column-major blocks with weight alignment                           │
│    Slots: [block0: ch0_b0, ch0_b1, ch1_b0, ch1_b1, ...]                         │
│                                                                                 │
│  • Key Innovation: W[i,j] packed at SAME slot as x[j]!                          │
│    → Element-wise Ct×Pt directly computes partial dot products                  │
│    → NO rotations needed within blocks!                                         │
│                                                                                 │
│  • Rotations: ZERO intra-block, only log2(K) for final accumulation            │
│    - When data fits in 1 block: 0 rotations total!                              │
│    - Rotation count independent of hidden_size (within block)                   │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  WHY CPMM IS FASTER:                                                            │
│  ───────────────────                                                            │
│  1. Weight-slot alignment eliminates need to rotate for gathering               │
│  2. Single-block configs achieve ZERO rotations                                 │
│  3. Multi-block configs only need accumulation rotations                        │
│  4. Rotation count is O(log(blocks)) vs O(hidden_size)                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
""")

    return results


def run_large_scale_comparison():
    """Compare at Llama 8B and Kimi 2.5 scale with larger CKKS profiles."""

    print("\n" + "=" * 110)
    print("LARGE SCALE COMPARISON (Llama 8B / Kimi 2.5)")
    print("With larger CKKS polynomial degree N=32768 (16384 slots) and N=65536 (32768 slots)")
    print("=" * 110)

    configs = [
        # (hidden_size, rank, batch_size, slot_count, model_name)
        (4096, 16, 1, 16384, "Llama 8B (N=32768)"),
        (4096, 16, 4, 16384, "Llama 8B (N=32768)"),
        (4096, 16, 8, 32768, "Llama 8B (N=65536)"),
        (7168, 16, 1, 16384, "Kimi 2.5 (N=32768)"),
        (7168, 16, 4, 32768, "Kimi 2.5 (N=65536)"),
    ]

    print(f"\n{'Model':<25} {'Config':<20} {'SIMD Rot/tok':<14} {'CPMM Rot/tok':<14} {'Reduction':<12}")
    print("-" * 110)

    for h, r, b, slots, name in configs:
        result = theoretical_comparison(h, r, b, slot_count=slots)

        simd_rot = result['simd']['rotations_per_token']
        cpmm_rot = result['cpmm']['rotations_per_token'] if result['cpmm']['error'] is None else "N/A"
        cpmm_rot_str = f"{cpmm_rot}" if isinstance(cpmm_rot, (int, float)) and cpmm_rot < float('inf') else "SLOTS"

        if result['cpmm']['error'] is None and simd_rot > 0 and cpmm_rot > 0:
            reduction = f"{(1 - cpmm_rot/simd_rot)*100:.0f}%"
        elif result['cpmm']['error'] is None and cpmm_rot == 0:
            reduction = "100%"
        else:
            reduction = "N/A"

        config = f"h={h},b={b}"
        print(f"{name:<25} {config:<20} {simd_rot:<14} {cpmm_rot_str:<14} {reduction:<12}")

    print("-" * 110)

    print("""
INTERPRETATION:
---------------
• SIMD Speculative Batching uses fixed block size (512), so rotations grow with h
• MOAI CPMM fits all data in single block when h×b ≤ slots → ZERO rotations
• For Llama 8B (h=4096, b=8): Need N=65536 profile for CPMM to achieve 0 rotations
• For Kimi 2.5 (h=7168, b=4): Need N=65536 profile, still achieves 0 rotations

ROTATION COST IMPACT (assuming 500μs per rotation):
• SIMD at h=4096, b=8: 12 rotations × 500μs = 6ms overhead
• CPMM at h=4096, b=8: 0 rotations = 0ms overhead
• Savings: 6ms per token = significant at scale!
""")


if __name__ == "__main__":
    results = run_full_comparison()
    run_large_scale_comparison()
