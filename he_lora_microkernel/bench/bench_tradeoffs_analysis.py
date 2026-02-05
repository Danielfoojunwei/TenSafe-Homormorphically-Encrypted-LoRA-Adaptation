"""
Honest Trade-off Analysis: SIMD Speculative Batching vs MOAI CPMM

The "0 rotations" claim for CPMM is MISLEADING. Let's analyze the REAL costs.
"""

import math
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HEOperationCosts:
    """Realistic HE operation costs in microseconds."""
    rotation_us: float = 500.0      # MOST EXPENSIVE
    ct_pt_mul_us: float = 100.0     # Ciphertext × Plaintext
    ct_ct_add_us: float = 20.0      # Ciphertext + Ciphertext
    rescale_us: float = 50.0        # Rescale after multiplication
    encrypt_us: float = 200.0
    decrypt_us: float = 200.0


def analyze_simd_batching(
    hidden_size: int,
    rank: int,
    batch_size: int,
    block_size: int = 512,
) -> Dict[str, Any]:
    """
    Analyze SIMD Speculative Batching costs.

    Strategy:
    1. Pre-compute AB = (α/r) * A @ B offline
    2. Pack activations in batch-first blocks
    3. For each block: Ct×Pt multiply
    4. Tree reduction across blocks
    """
    num_blocks = math.ceil(hidden_size / block_size)

    # Operations per token (single projection)
    # Each block does one Ct×Pt
    ct_pt_muls = num_blocks

    # Rescale after each Ct×Pt
    rescales = num_blocks

    # Tree reduction: log2(num_blocks) levels, each level has additions
    if num_blocks > 1:
        rotations = math.ceil(math.log2(num_blocks))
        additions = num_blocks - 1
    else:
        rotations = 0
        additions = 0

    return {
        'num_blocks': num_blocks,
        'ct_pt_muls': ct_pt_muls,
        'rescales': rescales,
        'rotations': rotations,
        'additions': additions,
        'description': 'Batch-first packing, block Ct×Pt, tree reduction',
    }


def analyze_moai_cpmm_honest(
    hidden_size: int,
    rank: int,
    batch_size: int,
    slot_count: int = 8192,
) -> Dict[str, Any]:
    """
    HONEST analysis of MOAI CPMM costs.

    What I CLAIMED:
    - W[i,j] at same slot as x[j] → element-wise multiply → no rotations!

    What I MISSED:
    - Element-wise multiply gives [x[0]*W[i,0], x[1]*W[i,1], ...]
    - We need SUM of these to get dot product y[i] = Σ_j x[j]*W[i,j]
    - This summation STILL REQUIRES ROTATIONS!

    The CPMM advantage is in HOW rotations are done, not eliminating them.
    """
    # Check if fits in slots
    if hidden_size * batch_size > slot_count:
        return {'error': f'Needs {hidden_size * batch_size} slots, have {slot_count}'}

    # For each output element y[i], we need:
    # 1. Ct×Pt to get [x[j]*W[i,j] for all j]
    # 2. Rotation-and-sum to accumulate into single value

    # The "column packing" advantage:
    # - Multiple output elements can SHARE the same input ciphertext
    # - But each output row needs its OWN set of rotations for summation

    # Rotations for summing d elements: log2(d) rotations
    rotations_per_output = math.ceil(math.log2(hidden_size))

    # HOWEVER, with clever packing (the REAL MOAI trick):
    # Pack multiple output computations in parallel across SIMD slots
    # This amortizes rotation cost across batch

    outputs_per_ciphertext = batch_size  # Can compute batch outputs in parallel

    # Total operations for all hidden_size outputs
    # But amortized across batch_size parallel computations

    # Ct×Pt: Need h separate weight rows × 1 ct_pt each
    # But if we pack cleverly, can batch these
    ct_pt_muls = hidden_size  # One per output row (this is HIGH!)

    # Rotations: log2(h) for summation, but SHARED across batch
    rotations = rotations_per_output  # Per output element

    # Total rescales
    rescales = hidden_size  # After each Ct×Pt

    # Additions for final assembly
    additions = hidden_size - 1 if hidden_size > 1 else 0

    return {
        'ct_pt_muls': ct_pt_muls,
        'rescales': rescales,
        'rotations': rotations,
        'additions': additions,
        'rotations_per_output': rotations_per_output,
        'description': 'Column-packed weights, BUT still needs rotation for dot product sum',
    }


def compute_total_time(ops: Dict[str, Any], costs: HEOperationCosts) -> float:
    """Compute total execution time in milliseconds."""
    if 'error' in ops:
        return float('inf')

    time_us = (
        ops.get('ct_pt_muls', 0) * costs.ct_pt_mul_us +
        ops.get('rescales', 0) * costs.rescale_us +
        ops.get('rotations', 0) * costs.rotation_us +
        ops.get('additions', 0) * costs.ct_ct_add_us
    )
    return time_us / 1000  # Convert to ms


def honest_comparison(
    hidden_size: int,
    rank: int,
    batch_size: int,
):
    """Run honest comparison showing ALL costs."""

    costs = HEOperationCosts()

    simd = analyze_simd_batching(hidden_size, rank, batch_size)
    cpmm = analyze_moai_cpmm_honest(hidden_size, rank, batch_size)

    simd_time = compute_total_time(simd, costs)
    cpmm_time = compute_total_time(cpmm, costs)

    print(f"\nConfiguration: h={hidden_size}, r={rank}, b={batch_size}")
    print("=" * 70)

    print(f"\n{'Operation':<20} {'SIMD Batching':<20} {'MOAI CPMM':<20}")
    print("-" * 70)

    simd_rot = simd.get('rotations', 0)
    cpmm_rot = cpmm.get('rotations', 'N/A') if 'error' not in cpmm else 'N/A'

    simd_ctpt = simd.get('ct_pt_muls', 0)
    cpmm_ctpt = cpmm.get('ct_pt_muls', 'N/A') if 'error' not in cpmm else 'N/A'

    simd_resc = simd.get('rescales', 0)
    cpmm_resc = cpmm.get('rescales', 'N/A') if 'error' not in cpmm else 'N/A'

    simd_add = simd.get('additions', 0)
    cpmm_add = cpmm.get('additions', 'N/A') if 'error' not in cpmm else 'N/A'

    print(f"{'Rotations':<20} {simd_rot:<20} {cpmm_rot:<20}")
    print(f"{'Ct×Pt Muls':<20} {simd_ctpt:<20} {cpmm_ctpt:<20}")
    print(f"{'Rescales':<20} {simd_resc:<20} {cpmm_resc:<20}")
    print(f"{'Additions':<20} {simd_add:<20} {cpmm_add:<20}")
    print("-" * 70)
    print(f"{'Est. Time (ms)':<20} {simd_time:<20.2f} {cpmm_time if cpmm_time < float('inf') else 'N/A':<20}")

    return simd, cpmm, simd_time, cpmm_time


def main():
    print("=" * 70)
    print("HONEST TRADE-OFF ANALYSIS: SIMD Batching vs MOAI CPMM")
    print("=" * 70)

    print("""
THE MISLEADING CLAIM I MADE:
============================
"MOAI CPMM achieves 0 rotations by aligning W[i,j] with x[j] slots"

WHAT I FORGOT:
==============
Element-wise Ct×Pt gives: [x[0]*W[i,0], x[1]*W[i,1], ..., x[h-1]*W[i,h-1]]

But we need the DOT PRODUCT: y[i] = Σ_j x[j]*W[i,j]

This SUMMATION still requires rotations! Specifically log2(h) rotations.

THE REAL TRADE-OFFS:
====================
""")

    # Test configurations
    configs = [
        (256, 16, 4),
        (512, 16, 4),
        (1024, 16, 4),
        (2048, 16, 4),
    ]

    print("\n" + "=" * 90)
    print("FULL OPERATION COUNT COMPARISON")
    print("=" * 90)

    print(f"\n{'Config':<18} {'SIMD Rot':<10} {'SIMD Ct×Pt':<12} {'CPMM Rot':<10} {'CPMM Ct×Pt':<12} {'Winner':<15}")
    print("-" * 90)

    for h, r, b in configs:
        simd, cpmm, simd_time, cpmm_time = honest_comparison(h, r, b)

        simd_rot = simd.get('rotations', 0)
        cpmm_rot = cpmm.get('rotations', 'N/A') if 'error' not in cpmm else 'N/A'

        simd_ctpt = simd.get('ct_pt_muls', 0)
        cpmm_ctpt = cpmm.get('ct_pt_muls', 'N/A') if 'error' not in cpmm else 'N/A'

        # Determine winner
        if 'error' in cpmm:
            winner = "SIMD (CPMM N/A)"
        elif simd_time < cpmm_time:
            winner = f"SIMD ({simd_time/cpmm_time:.1f}x faster)"
        elif cpmm_time < simd_time:
            winner = f"CPMM ({cpmm_time/simd_time:.1f}x faster)"
        else:
            winner = "TIE"

        config = f"h={h},r={r},b={b}"
        print(f"{config:<18} {simd_rot:<10} {simd_ctpt:<12} {cpmm_rot:<10} {cpmm_ctpt:<12} {winner:<15}")

    print("-" * 90)

    print("""

KEY INSIGHTS:
=============

1. ROTATIONS:
   - SIMD Batching: log2(num_blocks) ≈ 1-4 rotations
   - MOAI CPMM: log2(hidden_size) ≈ 8-12 rotations for dot product sum
   - SIMD WINS on rotations!

2. Ct×Pt MULTIPLICATIONS:
   - SIMD Batching: num_blocks ≈ 2-16 Ct×Pt
   - MOAI CPMM: hidden_size ≈ 256-4096 Ct×Pt (one per output row!)
   - SIMD WINS massively on Ct×Pt count!

3. TOTAL TIME:
   - Rotations cost 500μs each (expensive)
   - Ct×Pt costs 100μs each (cheaper but adds up)
   - SIMD is faster overall due to fewer operations

THE REAL MOAI INSIGHT:
======================
The MOAI paper's actual contribution is:
- COLUMN PACKING allows efficient Ct×Pt without diagonal rotation
- But it's used WITHIN the standard algorithm, not to replace it
- The paper achieves speedup through PARALLELISM, not rotation elimination

MY ERROR:
=========
I conflated "no rotation for value gathering" with "no rotation for computation"
The summation step ALWAYS needs rotations in CKKS.

CONCLUSION:
===========
SIMD Speculative Batching is actually MORE efficient for LoRA because:
1. Pre-computed AB reduces to single matmul
2. Block-based approach minimizes both rotations AND Ct×Pt count
3. MOAI CPMM's advantage is for different use cases (large matrices, specific patterns)
""")


if __name__ == "__main__":
    main()
