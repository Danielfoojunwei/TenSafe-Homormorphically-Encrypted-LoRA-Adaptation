"""
Comprehensive HE Benchmark Comparison for Llama 8B

Compares 4 HE approaches using LoRA without Regret paper parameters:
1. Full HE on entire Llama 8B (baseline)
2. Normal HE-LoRA
3. HE-LoRA Non-Linear Hybrid
4. HE-LoRA with Speculative SIMD Batching

LoRA without Regret paper parameters:
- Rank: 16, 32, 64 (commonly used)
- Alpha: 2 * rank
- Targets: All layers including MLP (attention + MLP projections)

Llama 8B specifications:
- hidden_size: 4096
- intermediate_size: 14336 (MLP)
- num_layers: 32
- num_attention_heads: 32
- vocab_size: 128256
"""

import time
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.compiler import (
    LoRAConfig,
    LoRATargets,
    CKKSProfile,
    get_profile,
    compile_schedule,
    select_optimal_profile,
)
from he_lora_microkernel.compiler.cost_model import (
    CostEstimate,
    estimate_costs,
    CostBudget,
)


# =============================================================================
# Llama 8B Configuration
# =============================================================================

@dataclass(frozen=True)
class Llama8BConfig:
    """Llama 8B model specifications."""
    hidden_size: int = 4096
    intermediate_size: int = 14336  # MLP intermediate
    num_layers: int = 32
    num_attention_heads: int = 32
    head_dim: int = 128  # hidden_size / num_attention_heads
    vocab_size: int = 128256

    # Projection sizes
    qkv_proj_size: int = 4096 * 3  # Q, K, V concatenated
    o_proj_size: int = 4096
    gate_up_proj_size: int = 14336 * 2  # gate and up projection
    down_proj_size: int = 4096


# =============================================================================
# HE Approach Types
# =============================================================================

class HEApproach(Enum):
    """Different HE implementation approaches."""
    FULL_HE = "full_he"           # Full HE on entire model
    NORMAL_HELORA = "normal_helora"  # Standard HE-LoRA
    HYBRID_NONLINEAR = "hybrid_nonlinear"  # HE-LoRA with plaintext non-linear
    SIMD_BATCHING = "simd_batching"  # HE-LoRA with speculative SIMD batching


# =============================================================================
# Cost Models for Different Approaches
# =============================================================================

@dataclass
class HEOperationCosts:
    """HE operation costs in microseconds (based on N=16384, FAST profile)."""
    # Base operation costs (us)
    encrypt_us: float = 200.0
    decrypt_us: float = 200.0
    mul_ct_ct_us: float = 800.0      # Ciphertext × Ciphertext
    mul_ct_pt_us: float = 100.0      # Ciphertext × Plaintext
    add_ct_ct_us: float = 20.0       # Ciphertext + Ciphertext
    rotate_us: float = 500.0         # Rotation (expensive!)
    rescale_us: float = 50.0         # Rescale after multiplication
    keyswitch_us: float = 500.0      # Key switching

    # Memory costs
    ciphertext_size_kb: float = 512.0  # Per ciphertext element

    # Slot utilization
    slot_count: int = 8192  # N/2 for N=16384


@dataclass
class ApproachMetrics:
    """Metrics for a single HE approach."""
    approach: HEApproach
    rank: int
    batch_size: int

    # Throughput
    tokens_per_second: float = 0.0
    aggregate_tokens_per_second: float = 0.0
    ms_per_token: float = 0.0

    # Latency breakdown (ms)
    total_latency_ms: float = 0.0
    encrypt_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    decrypt_time_ms: float = 0.0

    # HE operation counts per token
    rotations_per_token: float = 0.0
    keyswitches_per_token: float = 0.0
    rescales_per_token: float = 0.0
    ct_ct_muls_per_token: float = 0.0
    ct_pt_muls_per_token: float = 0.0

    # Memory
    memory_mb: float = 0.0

    # Model coverage
    layers_covered: int = 0
    projections_per_layer: int = 0

    # Depth
    multiplicative_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'approach': self.approach.value,
            'rank': self.rank,
            'batch_size': self.batch_size,
            'throughput': {
                'tok_per_s': round(self.tokens_per_second, 2),
                'aggregate_tok_per_s': round(self.aggregate_tokens_per_second, 2),
                'ms_per_token': round(self.ms_per_token, 3),
            },
            'latency_ms': {
                'total': round(self.total_latency_ms, 3),
                'encrypt': round(self.encrypt_time_ms, 3),
                'compute': round(self.compute_time_ms, 3),
                'decrypt': round(self.decrypt_time_ms, 3),
            },
            'operations_per_token': {
                'rotations': round(self.rotations_per_token, 1),
                'keyswitches': round(self.keyswitches_per_token, 1),
                'rescales': round(self.rescales_per_token, 1),
                'ct_ct_muls': round(self.ct_ct_muls_per_token, 1),
                'ct_pt_muls': round(self.ct_pt_muls_per_token, 1),
            },
            'memory_mb': round(self.memory_mb, 1),
            'coverage': {
                'layers': self.layers_covered,
                'projections_per_layer': self.projections_per_layer,
            },
            'depth': self.multiplicative_depth,
        }


# =============================================================================
# Approach Estimators
# =============================================================================

class FullHEEstimator:
    """
    Estimate costs for full HE on entire Llama 8B.

    This encrypts all model weights and computations.
    NOT practical - shown for comparison only.
    """

    def __init__(self, config: Llama8BConfig, costs: HEOperationCosts):
        self.config = config
        self.costs = costs

    def estimate(self, batch_size: int) -> ApproachMetrics:
        metrics = ApproachMetrics(
            approach=HEApproach.FULL_HE,
            rank=0,  # Not applicable
            batch_size=batch_size,
        )

        # Full HE requires encrypting all activations through all layers
        # For each layer: attention (QKV proj, attention, O proj) + MLP (gate, up, down)

        h = self.config.hidden_size  # 4096
        ffn = self.config.intermediate_size  # 14336

        # Matrix multiplication in HE using diagonal method requires O(d) rotations
        # For (d × d) @ (d × b): need d rotations for each diagonal
        rotations_per_matmul_attention = h  # 4096 rotations per matmul
        rotations_per_matmul_mlp = ffn  # 14336 for MLP layers

        # Per-layer operations
        attention_matmuls = 4  # Q, K, V, O
        mlp_matmuls = 3  # gate, up, down

        metrics.layers_covered = self.config.num_layers
        metrics.projections_per_layer = attention_matmuls + mlp_matmuls

        # Total rotations per layer:
        # - Q, K, V projections: 3 × 4096 rotations
        # - Attention scores: requires Ct×Ct and more rotations
        # - O projection: 4096 rotations
        # - MLP gate, up: 2 × 14336 rotations
        # - MLP down: 4096 rotations
        rotations_attention = attention_matmuls * rotations_per_matmul_attention
        rotations_mlp = 2 * rotations_per_matmul_mlp + rotations_per_matmul_attention
        rotations_per_layer = rotations_attention + rotations_mlp

        # Ct×Ct multiplications for attention scores (Q @ K^T)
        ct_ct_per_layer = batch_size * self.config.num_attention_heads * 128  # seq_len approx

        metrics.rotations_per_token = rotations_per_layer * self.config.num_layers
        metrics.keyswitches_per_token = metrics.rotations_per_token
        metrics.rescales_per_token = (attention_matmuls + mlp_matmuls) * self.config.num_layers * 2
        metrics.ct_ct_muls_per_token = ct_ct_per_layer * self.config.num_layers
        metrics.ct_pt_muls_per_token = 0  # All encrypted

        # Timing
        compute_us = (
            metrics.rotations_per_token * self.costs.rotate_us +
            metrics.keyswitches_per_token * self.costs.keyswitch_us +
            metrics.rescales_per_token * self.costs.rescale_us +
            metrics.ct_ct_muls_per_token * self.costs.mul_ct_ct_us
        )

        # Encrypt all activations
        ciphertexts_per_token = math.ceil(h / self.costs.slot_count) * batch_size
        metrics.encrypt_time_ms = ciphertexts_per_token * self.costs.encrypt_us / 1000
        metrics.decrypt_time_ms = ciphertexts_per_token * self.costs.decrypt_us / 1000
        metrics.compute_time_ms = compute_us / 1000

        metrics.total_latency_ms = (
            metrics.encrypt_time_ms + metrics.compute_time_ms + metrics.decrypt_time_ms
        )

        metrics.ms_per_token = metrics.total_latency_ms
        metrics.tokens_per_second = 1000 / metrics.ms_per_token if metrics.ms_per_token > 0 else 0
        metrics.aggregate_tokens_per_second = metrics.tokens_per_second * batch_size

        # Memory: all weights encrypted
        params = (
            self.config.hidden_size * self.config.hidden_size * 4 +  # Attention
            self.config.hidden_size * self.config.intermediate_size * 3  # MLP
        ) * self.config.num_layers
        metrics.memory_mb = params * self.costs.ciphertext_size_kb / 1024 / 1024

        # Depth: very deep - needs bootstrapping which we don't support
        metrics.multiplicative_depth = self.config.num_layers * 3

        return metrics


class NormalHELoRAEstimator:
    """
    Estimate costs for standard HE-LoRA.

    Only LoRA deltas are computed in HE.
    Delta = alpha/r * A @ B @ x
    """

    def __init__(self, config: Llama8BConfig, costs: HEOperationCosts):
        self.config = config
        self.costs = costs

    def estimate(self, rank: int, batch_size: int, targets: str = "all") -> ApproachMetrics:
        metrics = ApproachMetrics(
            approach=HEApproach.NORMAL_HELORA,
            rank=rank,
            batch_size=batch_size,
        )

        h = self.config.hidden_size
        r = rank

        # Number of projections with LoRA
        if targets == "all":
            # Q, K, V, O projections + gate, up, down projections
            projections_per_layer = 7
        elif targets == "qkvo":
            projections_per_layer = 4
        else:  # qkv
            projections_per_layer = 3

        metrics.layers_covered = self.config.num_layers
        metrics.projections_per_layer = projections_per_layer

        # Standard LoRA: two matrix multiplications
        # B @ x: (r × h) @ (h × b) → (r × b)
        # A @ intermediate: (h × r) @ (r × b) → (h × b)

        # For naive implementation, rotations proportional to dimensions
        rotations_per_lora = 2 * math.ceil(math.log2(max(h, r)))

        total_lora_ops = projections_per_layer * self.config.num_layers

        metrics.rotations_per_token = rotations_per_lora * total_lora_ops
        metrics.keyswitches_per_token = metrics.rotations_per_token
        metrics.rescales_per_token = 2 * total_lora_ops  # 2 muls per LoRA
        metrics.ct_ct_muls_per_token = 0  # LoRA uses Ct×Pt
        metrics.ct_pt_muls_per_token = 2 * total_lora_ops

        # Timing
        compute_us = (
            metrics.rotations_per_token * self.costs.rotate_us +
            metrics.keyswitches_per_token * self.costs.keyswitch_us +
            metrics.rescales_per_token * self.costs.rescale_us +
            metrics.ct_pt_muls_per_token * self.costs.mul_ct_pt_us
        )

        # Encrypt activations
        ciphertexts = math.ceil(h * batch_size / self.costs.slot_count)
        metrics.encrypt_time_ms = ciphertexts * self.costs.encrypt_us / 1000
        metrics.decrypt_time_ms = ciphertexts * self.costs.decrypt_us / 1000
        metrics.compute_time_ms = compute_us / 1000

        metrics.total_latency_ms = (
            metrics.encrypt_time_ms + metrics.compute_time_ms + metrics.decrypt_time_ms
        )

        metrics.ms_per_token = metrics.total_latency_ms
        metrics.tokens_per_second = 1000 / metrics.ms_per_token if metrics.ms_per_token > 0 else 0
        metrics.aggregate_tokens_per_second = metrics.tokens_per_second * batch_size

        # Memory: only LoRA weights encrypted
        lora_params = (h * r + r * h) * projections_per_layer * self.config.num_layers
        metrics.memory_mb = lora_params * self.costs.ciphertext_size_kb / 1024 / 1024

        # Depth: 2 (two matmuls)
        metrics.multiplicative_depth = 2

        return metrics


class HybridNonLinearEstimator:
    """
    Estimate costs for HE-LoRA with non-linear hybrid.

    Linear projections computed in HE, non-linear ops (softmax, GELU) in plaintext.
    Requires decrypt/re-encrypt at non-linear boundaries.
    """

    def __init__(self, config: Llama8BConfig, costs: HEOperationCosts):
        self.config = config
        self.costs = costs

    def estimate(self, rank: int, batch_size: int, targets: str = "all") -> ApproachMetrics:
        metrics = ApproachMetrics(
            approach=HEApproach.HYBRID_NONLINEAR,
            rank=rank,
            batch_size=batch_size,
        )

        h = self.config.hidden_size
        r = rank

        # Projections with LoRA
        if targets == "all":
            projections_per_layer = 7
        elif targets == "qkvo":
            projections_per_layer = 4
        else:
            projections_per_layer = 3

        metrics.layers_covered = self.config.num_layers
        metrics.projections_per_layer = projections_per_layer

        # Same LoRA computation as normal
        rotations_per_lora = 2 * math.ceil(math.log2(max(h, r)))
        total_lora_ops = projections_per_layer * self.config.num_layers

        metrics.rotations_per_token = rotations_per_lora * total_lora_ops
        metrics.keyswitches_per_token = metrics.rotations_per_token
        metrics.rescales_per_token = 2 * total_lora_ops
        metrics.ct_ct_muls_per_token = 0
        metrics.ct_pt_muls_per_token = 2 * total_lora_ops

        # Non-linear boundaries per layer:
        # - After attention scores (softmax)
        # - After MLP gate (activation)
        nonlinear_boundaries_per_layer = 2
        total_boundaries = nonlinear_boundaries_per_layer * self.config.num_layers

        # Extra decrypt/encrypt at boundaries
        ciphertexts_boundary = math.ceil(h * batch_size / self.costs.slot_count)

        # Timing
        compute_us = (
            metrics.rotations_per_token * self.costs.rotate_us +
            metrics.keyswitches_per_token * self.costs.keyswitch_us +
            metrics.rescales_per_token * self.costs.rescale_us +
            metrics.ct_pt_muls_per_token * self.costs.mul_ct_pt_us
        )

        # Base encrypt/decrypt
        base_ciphertexts = math.ceil(h * batch_size / self.costs.slot_count)
        base_encrypt_us = base_ciphertexts * self.costs.encrypt_us
        base_decrypt_us = base_ciphertexts * self.costs.decrypt_us

        # Boundary encrypt/decrypt
        boundary_encrypt_us = total_boundaries * ciphertexts_boundary * self.costs.encrypt_us
        boundary_decrypt_us = total_boundaries * ciphertexts_boundary * self.costs.decrypt_us

        metrics.encrypt_time_ms = (base_encrypt_us + boundary_encrypt_us) / 1000
        metrics.decrypt_time_ms = (base_decrypt_us + boundary_decrypt_us) / 1000
        metrics.compute_time_ms = compute_us / 1000

        metrics.total_latency_ms = (
            metrics.encrypt_time_ms + metrics.compute_time_ms + metrics.decrypt_time_ms
        )

        metrics.ms_per_token = metrics.total_latency_ms
        metrics.tokens_per_second = 1000 / metrics.ms_per_token if metrics.ms_per_token > 0 else 0
        metrics.aggregate_tokens_per_second = metrics.tokens_per_second * batch_size

        # Memory
        lora_params = (h * r + r * h) * projections_per_layer * self.config.num_layers
        metrics.memory_mb = lora_params * self.costs.ciphertext_size_kb / 1024 / 1024

        # Depth: 2 per segment, reset at boundaries
        metrics.multiplicative_depth = 2

        return metrics


class SIMDBatchingEstimator:
    """
    Estimate costs for HE-LoRA with speculative SIMD batching.

    Uses MOAI-style CPMM packing for rotation-minimal computation:
    - Column-packed matrices
    - Zero intra-block rotations for Ct×Pt
    - Batch-first SIMD layout
    """

    def __init__(self, config: Llama8BConfig, costs: HEOperationCosts):
        self.config = config
        self.costs = costs

    def estimate(self, rank: int, batch_size: int, targets: str = "all") -> ApproachMetrics:
        metrics = ApproachMetrics(
            approach=HEApproach.SIMD_BATCHING,
            rank=rank,
            batch_size=batch_size,
        )

        h = self.config.hidden_size
        r = rank
        block_size = 512  # MOAI block size

        # Projections with LoRA
        if targets == "all":
            projections_per_layer = 7
        elif targets == "qkvo":
            projections_per_layer = 4
        else:
            projections_per_layer = 3

        metrics.layers_covered = self.config.num_layers
        metrics.projections_per_layer = projections_per_layer

        # MOAI CPMM: Zero rotations within blocks!
        # Only rotations needed for cross-block accumulation
        num_blocks = math.ceil(h / block_size)

        # CPMM: rotations only for final accumulation across blocks
        # log2(num_blocks) rotations for tree reduction
        accumulation_rotations = math.ceil(math.log2(num_blocks)) if num_blocks > 1 else 0

        # Two matmuls (B@x, A@intermediate) per LoRA
        total_lora_ops = projections_per_layer * self.config.num_layers

        # SIMD batching: process batch_size tokens in parallel
        # Rotations shared across batch
        metrics.rotations_per_token = accumulation_rotations * 2 * total_lora_ops / batch_size
        metrics.keyswitches_per_token = metrics.rotations_per_token
        metrics.rescales_per_token = 2 * total_lora_ops / batch_size  # Shared
        metrics.ct_ct_muls_per_token = 0
        metrics.ct_pt_muls_per_token = num_blocks * 2 * total_lora_ops / batch_size

        # Timing
        compute_us = (
            metrics.rotations_per_token * self.costs.rotate_us +
            metrics.keyswitches_per_token * self.costs.keyswitch_us +
            metrics.rescales_per_token * self.costs.rescale_us +
            metrics.ct_pt_muls_per_token * self.costs.mul_ct_pt_us
        )

        # Encrypt with SIMD batching - fewer ciphertexts needed
        slots_used = batch_size * block_size
        ciphertexts = max(1, math.ceil(h * batch_size / self.costs.slot_count))

        # Amortized encrypt/decrypt across batch
        metrics.encrypt_time_ms = ciphertexts * self.costs.encrypt_us / 1000 / batch_size
        metrics.decrypt_time_ms = ciphertexts * self.costs.decrypt_us / 1000 / batch_size
        metrics.compute_time_ms = compute_us / 1000

        metrics.total_latency_ms = (
            metrics.encrypt_time_ms + metrics.compute_time_ms + metrics.decrypt_time_ms
        )

        metrics.ms_per_token = metrics.total_latency_ms
        metrics.tokens_per_second = 1000 / metrics.ms_per_token if metrics.ms_per_token > 0 else 0
        metrics.aggregate_tokens_per_second = metrics.tokens_per_second * batch_size

        # Memory: same as normal HE-LoRA but packed efficiently
        lora_params = (h * r + r * h) * projections_per_layer * self.config.num_layers
        # SIMD batching reduces effective memory through packing
        packing_efficiency = min(batch_size * block_size / self.costs.slot_count, 1.0)
        metrics.memory_mb = lora_params * self.costs.ciphertext_size_kb / 1024 / 1024 * (1 - packing_efficiency * 0.3)

        # Depth: still 2, but with TURBO profile support
        metrics.multiplicative_depth = 2

        return metrics


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_llama8b_comparison(
    ranks: List[int] = [16, 32, 64],
    batch_sizes: List[int] = [1, 4, 8],
    targets: str = "all",
) -> Dict[str, Any]:
    """
    Run comprehensive comparison benchmark for Llama 8B.

    Args:
        ranks: LoRA ranks to test (from LoRA without Regret)
        batch_sizes: Batch sizes to test
        targets: LoRA targets ("qkv", "qkvo", "all")

    Returns:
        Comprehensive benchmark results
    """
    config = Llama8BConfig()
    costs = HEOperationCosts()

    # Initialize estimators
    full_he = FullHEEstimator(config, costs)
    normal_helora = NormalHELoRAEstimator(config, costs)
    hybrid = HybridNonLinearEstimator(config, costs)
    simd = SIMDBatchingEstimator(config, costs)

    results = {
        'model': 'Llama-8B',
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'intermediate_size': config.intermediate_size,
        },
        'lora_without_regret_params': {
            'ranks': ranks,
            'alpha': '2 * rank',
            'targets': targets,
        },
        'approaches': [],
    }

    print("=" * 80)
    print("LLAMA 8B HE BENCHMARK COMPARISON")
    print("Using LoRA without Regret paper parameters")
    print("=" * 80)
    print()

    # Full HE baseline (single config for comparison)
    print("1. Full HE (Entire Model) - BASELINE")
    print("-" * 40)
    for bs in batch_sizes[:1]:  # Just one batch size for full HE
        metrics = full_he.estimate(bs)
        results['approaches'].append(metrics.to_dict())
        print(f"   Batch={bs}: {metrics.tokens_per_second:.2f} tok/s | "
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

    # SIMD Batching (Our System)
    print("4. HE-LoRA with Speculative SIMD Batching (THIS SYSTEM)")
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


def print_comparison_table(results: Dict[str, Any]) -> str:
    """Print formatted comparison table."""

    # Group by approach
    approaches = {}
    for m in results['approaches']:
        approach = m['approach']
        if approach not in approaches:
            approaches[approach] = []
        approaches[approach].append(m)

    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("BENCHMARK COMPARISON TABLE: Llama 8B with LoRA without Regret Parameters")
    lines.append("=" * 110)
    lines.append("")

    # Header
    header = f"{'Approach':<30} {'Rank':<6} {'Batch':<6} {'Tok/s':<12} {'Agg Tok/s':<12} {'ms/tok':<10} {'Rot/tok':<10} {'Depth':<6}"
    lines.append(header)
    lines.append("-" * 110)

    # Format approach names
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
                       f"{m['throughput']['tok_per_s']:<12.2f} " \
                       f"{m['throughput']['aggregate_tok_per_s']:<12.2f} " \
                       f"{m['throughput']['ms_per_token']:<10.3f} " \
                       f"{m['operations_per_token']['rotations']:<10.1f} " \
                       f"{m['depth']:<6}"
                lines.append(line)
            lines.append("-" * 110)

    lines.append("")

    # Summary statistics
    lines.append("KEY FINDINGS:")
    lines.append("-" * 50)

    # Find best SIMD result
    simd_results = approaches.get('simd_batching', [])
    normal_results = approaches.get('normal_helora', [])

    if simd_results and normal_results:
        best_simd = max(simd_results, key=lambda x: x['throughput']['tok_per_s'])
        best_normal = max(normal_results, key=lambda x: x['throughput']['tok_per_s'])

        speedup = best_simd['throughput']['tok_per_s'] / best_normal['throughput']['tok_per_s']
        rotation_reduction = (1 - best_simd['operations_per_token']['rotations'] /
                             best_normal['operations_per_token']['rotations']) * 100

        lines.append(f"  • SIMD Batching Speedup over Normal HE-LoRA: {speedup:.2f}x")
        lines.append(f"  • Rotation Reduction: {rotation_reduction:.1f}%")
        lines.append(f"  • Best SIMD Throughput: {best_simd['throughput']['tok_per_s']:.2f} tok/s (r={best_simd['rank']}, b={best_simd['batch_size']})")

    # Full HE comparison
    full_he_results = approaches.get('full_he', [])
    if full_he_results and simd_results:
        full_he = full_he_results[0]
        best_simd = max(simd_results, key=lambda x: x['throughput']['tok_per_s'])

        if full_he['throughput']['tok_per_s'] > 0:
            improvement = best_simd['throughput']['tok_per_s'] / full_he['throughput']['tok_per_s']
            lines.append(f"  • Improvement over Full HE: {improvement:.0f}x")
        else:
            # Full HE is impractically slow
            lines.append(f"  • Full HE is impractical (>1M rotations/token, requires bootstrapping)")
            lines.append(f"  • SIMD Batching makes HE-LoRA practical with {best_simd['operations_per_token']['rotations']:.0f} rot/tok")

    lines.append("")
    lines.append("=" * 110)

    output = "\n".join(lines)
    print(output)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Llama 8B HE Benchmark Comparison")
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64],
                       help="LoRA ranks to test")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8],
                       help="Batch sizes to test")
    parser.add_argument("--targets", default="all",
                       help="LoRA targets (qkv, qkvo, all)")
    parser.add_argument("--output", default=None, help="Output JSON file")

    args = parser.parse_args()

    results = run_llama8b_comparison(
        ranks=args.ranks,
        batch_sizes=args.batch_sizes,
        targets=args.targets,
    )

    table = print_comparison_table(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
