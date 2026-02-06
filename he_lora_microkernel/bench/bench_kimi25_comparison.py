"""
HE Benchmark Comparison for Kimi 2.5

Compares 4 HE approaches using LoRA without Regret parameters for Kimi 2.5.

Kimi 2.5 specifications (from Moonshot AI):
- hidden_size: 7168
- num_layers: 61
- num_attention_heads: 64
- MoE: 384 experts, 8 active per token
- MoE expert hidden_size: 2048
- 1T total parameters, 32B activated per token
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

from he_lora_microkernel.bench.bench_llama8b_comparison import (
    FullHEEstimator,
    HEOperationCosts,
    HybridNonLinearEstimator,
    NormalHELoRAEstimator,
    SIMDBatchingEstimator,
)


@dataclass(frozen=True)
class Kimi25Config:
    """Kimi 2.5 model specifications."""
    hidden_size: int = 7168
    intermediate_size: int = 2048 * 8  # MoE expert dim × active experts
    num_layers: int = 61
    num_attention_heads: int = 64
    head_dim: int = 112  # 7168 / 64
    vocab_size: int = 160000

    # MoE configuration
    num_experts: int = 384
    experts_per_token: int = 8
    expert_hidden_size: int = 2048

    # Projections
    qkv_proj_size: int = 7168 * 3
    o_proj_size: int = 7168


def run_kimi25_comparison(
    ranks: List[int] = None,
    batch_sizes: List[int] = None,
    targets: str = "all",
) -> Dict[str, Any]:
    """Run benchmark comparison for Kimi 2.5."""

    if batch_sizes is None:
        batch_sizes = [1, 4, 8]
    if ranks is None:
        ranks = [16, 32, 64]
    config = Kimi25Config()
    costs = HEOperationCosts()

    # Adjusted estimators for Kimi 2.5 dimensions
    class Kimi25FullHEEstimator(FullHEEstimator):
        def __init__(self):
            self.config = config
            self.costs = costs

    class Kimi25NormalHELoRAEstimator(NormalHELoRAEstimator):
        def __init__(self):
            self.config = config
            self.costs = costs

    class Kimi25HybridEstimator(HybridNonLinearEstimator):
        def __init__(self):
            self.config = config
            self.costs = costs

    class Kimi25SIMDEstimator(SIMDBatchingEstimator):
        def __init__(self):
            self.config = config
            self.costs = costs

    full_he = Kimi25FullHEEstimator()
    normal_helora = Kimi25NormalHELoRAEstimator()
    hybrid = Kimi25HybridEstimator()
    simd = Kimi25SIMDEstimator()

    results = {
        'model': 'Kimi-2.5',
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'intermediate_size': config.intermediate_size,
            'num_experts': config.num_experts,
            'experts_per_token': config.experts_per_token,
        },
        'lora_without_regret_params': {
            'ranks': ranks,
            'alpha': '2 * rank',
            'targets': targets,
        },
        'approaches': [],
    }

    print("=" * 80)
    print("KIMI 2.5 HE BENCHMARK COMPARISON")
    print("Using LoRA without Regret paper parameters")
    print("=" * 80)
    print(f"\nModel: Kimi 2.5 (h={config.hidden_size}, L={config.num_layers})")
    print(f"MoE: {config.num_experts} experts, {config.experts_per_token} active/token")
    print()

    # Full HE baseline
    print("1. Full HE (Entire Model) - BASELINE")
    print("-" * 40)
    for bs in batch_sizes[:1]:
        metrics = full_he.estimate(bs)
        results['approaches'].append(metrics.to_dict())
        print(f"   Batch={bs}: {metrics.tokens_per_second:.4f} tok/s | "
              f"{metrics.rotations_per_token:.0f} rot/tok | "
              f"Depth={metrics.multiplicative_depth}")
    print()

    # Normal HE-LoRA
    print("2. Normal HE-LoRA")
    print("-" * 40)
    for rank in ranks:
        for bs in batch_sizes:
            metrics = normal_helora.estimate(rank, bs, targets)
            results['approaches'].append(metrics.to_dict())
            print(f"   r={rank}, b={bs}: {metrics.tokens_per_second:.2f} tok/s | "
                  f"{metrics.rotations_per_token:.1f} rot/tok | "
                  f"Depth={metrics.multiplicative_depth}")
    print()

    # Hybrid Non-Linear
    print("3. HE-LoRA Non-Linear Hybrid")
    print("-" * 40)
    for rank in ranks:
        for bs in batch_sizes:
            metrics = hybrid.estimate(rank, bs, targets)
            results['approaches'].append(metrics.to_dict())
            print(f"   r={rank}, b={bs}: {metrics.tokens_per_second:.2f} tok/s | "
                  f"{metrics.rotations_per_token:.1f} rot/tok | "
                  f"Depth={metrics.multiplicative_depth}")
    print()

    # SIMD Batching
    print("4. HE-LoRA with Speculative SIMD Batching")
    print("-" * 40)
    for rank in ranks:
        for bs in batch_sizes:
            metrics = simd.estimate(rank, bs, targets)
            results['approaches'].append(metrics.to_dict())
            print(f"   r={rank}, b={bs}: {metrics.tokens_per_second:.2f} tok/s | "
                  f"{metrics.rotations_per_token:.1f} rot/tok | "
                  f"Depth={metrics.multiplicative_depth}")
    print()

    return results


def print_kimi_comparison_table(results: Dict[str, Any]) -> str:
    """Print formatted comparison table for Kimi 2.5."""

    approaches = {}
    for m in results['approaches']:
        approach = m['approach']
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append(m)

    lines = []
    lines.append("")
    lines.append("=" * 115)
    lines.append("BENCHMARK COMPARISON TABLE: Kimi 2.5 with LoRA without Regret Parameters")
    lines.append("=" * 115)
    lines.append(f"Model: hidden_size={results['config']['hidden_size']}, "
                 f"num_layers={results['config']['num_layers']}, "
                 f"MoE={results['config']['num_experts']} experts")
    lines.append("")

    header = f"{'Approach':<30} {'Rank':<6} {'Batch':<6} {'Tok/s':<12} {'Agg Tok/s':<12} {'ms/tok':<12} {'Rot/tok':<12} {'Depth':<6}"
    lines.append(header)
    lines.append("-" * 115)

    approach_names = {
        'full_he': 'Full HE (Baseline)',
        'normal_helora': 'Normal HE-LoRA',
        'hybrid_nonlinear': 'HE-LoRA Hybrid',
        'simd_batching': 'SIMD Batching (Ours)',
    }

    for approach_key in ['full_he', 'normal_helora', 'hybrid_nonlinear', 'simd_batching']:
        if approach_key in approaches:
            name = approach_names.get(approach_key, approach_key)
            first = True
            for m in approaches[approach_key]:
                if first:
                    approach_col = name
                    first = False
                else:
                    approach_col = ""

                line = f"{approach_col:<30} {m['rank']:<6} {m['batch_size']:<6} " \
                       f"{m['throughput']['tok_per_s']:<12.4f} " \
                       f"{m['throughput']['aggregate_tok_per_s']:<12.2f} " \
                       f"{m['throughput']['ms_per_token']:<12.1f} " \
                       f"{m['operations_per_token']['rotations']:<12.1f} " \
                       f"{m['depth']:<6}"
                lines.append(line)
            lines.append("-" * 115)

    lines.append("")

    # Key findings
    lines.append("KEY FINDINGS:")
    lines.append("-" * 50)

    simd_results = approaches.get('simd_batching', [])
    normal_results = approaches.get('normal_helora', [])

    if simd_results and normal_results:
        best_simd = max(simd_results, key=lambda x: x['throughput']['tok_per_s'])
        best_normal = max(normal_results, key=lambda x: x['throughput']['tok_per_s'])

        if best_normal['throughput']['tok_per_s'] > 0:
            speedup = best_simd['throughput']['tok_per_s'] / best_normal['throughput']['tok_per_s']
            lines.append(f"  • SIMD Batching Speedup over Normal HE-LoRA: {speedup:.2f}x")

        if best_normal['operations_per_token']['rotations'] > 0:
            rotation_reduction = (1 - best_simd['operations_per_token']['rotations'] /
                                 best_normal['operations_per_token']['rotations']) * 100
            lines.append(f"  • Rotation Reduction: {rotation_reduction:.1f}%")

        lines.append(f"  • Best SIMD Throughput: {best_simd['throughput']['tok_per_s']:.2f} tok/s "
                    f"(r={best_simd['rank']}, b={best_simd['batch_size']})")

    full_he_results = approaches.get('full_he', [])
    if full_he_results:
        full_he = full_he_results[0]
        if full_he['throughput']['tok_per_s'] > 0 and simd_results:
            best_simd = max(simd_results, key=lambda x: x['throughput']['tok_per_s'])
            improvement = best_simd['throughput']['tok_per_s'] / full_he['throughput']['tok_per_s']
            lines.append(f"  • Improvement over Full HE: {improvement:.0f}x")
        else:
            lines.append("  • Full HE is impractical for Kimi 2.5 (h=7168 requires massive rotation count)")

    lines.append("")
    lines.append("=" * 115)

    output = "\n".join(lines)
    print(output)
    return output


def compare_llama_vs_kimi():
    """Compare Llama 8B vs Kimi 2.5 HE performance."""
    print("\n" + "=" * 80)
    print("LLAMA 8B vs KIMI 2.5 COMPARISON")
    print("=" * 80)

    # Llama 8B
    llama_h = 4096
    llama_layers = 32

    # Kimi 2.5
    kimi_h = 7168
    kimi_layers = 61

    # LoRA projections per layer (all targets)
    projections_per_layer = 7

    # Compute rotation estimates for SIMD batching (b=8)
    batch_size = 8
    block_size = 512

    # Llama 8B
    llama_blocks = math.ceil(llama_h / block_size)
    llama_rotations = math.ceil(math.log2(llama_blocks)) if llama_blocks > 1 else 0
    llama_total_rot = llama_rotations * 2 * projections_per_layer * llama_layers / batch_size

    # Kimi 2.5
    kimi_blocks = math.ceil(kimi_h / block_size)
    kimi_rotations = math.ceil(math.log2(kimi_blocks)) if kimi_blocks > 1 else 0
    kimi_total_rot = kimi_rotations * 2 * projections_per_layer * kimi_layers / batch_size

    print(f"\n{'Metric':<30} {'Llama 8B':<20} {'Kimi 2.5':<20} {'Ratio':<10}")
    print("-" * 80)
    print(f"{'Hidden Size':<30} {llama_h:<20} {kimi_h:<20} {kimi_h/llama_h:.2f}x")
    print(f"{'Num Layers':<30} {llama_layers:<20} {kimi_layers:<20} {kimi_layers/llama_layers:.2f}x")
    print(f"{'Projections (all targets)':<30} {projections_per_layer * llama_layers:<20} {projections_per_layer * kimi_layers:<20} {kimi_layers/llama_layers:.2f}x")
    print(f"{'SIMD Blocks (b=8)':<30} {llama_blocks:<20} {kimi_blocks:<20} {kimi_blocks/llama_blocks:.2f}x")
    print(f"{'Est. Rotations/tok (SIMD)':<30} {llama_total_rot:<20.1f} {kimi_total_rot:<20.1f} {kimi_total_rot/llama_total_rot:.2f}x")
    print("-" * 80)

    # Compute time estimates (rotation cost = 500μs)
    rotation_cost_us = 500
    llama_time_ms = llama_total_rot * rotation_cost_us / 1000
    kimi_time_ms = kimi_total_rot * rotation_cost_us / 1000

    print(f"{'Est. Rotation Time (ms)':<30} {llama_time_ms:<20.1f} {kimi_time_ms:<20.1f} {kimi_time_ms/llama_time_ms:.2f}x")
    print(f"{'Est. Throughput (tok/s)':<30} {1000/llama_time_ms if llama_time_ms > 0 else 'N/A':<20.1f} {1000/kimi_time_ms if kimi_time_ms > 0 else 'N/A':<20.1f}")


if __name__ == "__main__":
    results = run_kimi25_comparison(
        ranks=[16, 32, 64],
        batch_sizes=[1, 4, 8],
        targets="all",
    )

    table = print_kimi_comparison_table(results)

    compare_llama_vs_kimi()

    # Save results
    with open('benchmark_kimi25_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to benchmark_kimi25_results.json")
