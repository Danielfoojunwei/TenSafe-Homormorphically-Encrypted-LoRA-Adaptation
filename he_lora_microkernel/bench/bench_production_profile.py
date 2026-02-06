"""
Benchmark: PRODUCTION Profile (N=32768) vs TURBO Profile (N=16384)

Demonstrates the rotation reduction achieved by using larger polynomial
degree for Llama-scale models (h >= 2048).

Key insight:
- N=16384 gives 8192 slots → block_size=512 → more blocks → more rotations
- N=32768 gives 16384 slots → block_size=1024 → fewer blocks → fewer rotations
"""

import math
import sys
from dataclasses import dataclass
from typing import List

# Add parent path for imports
sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

from he_lora_microkernel.compiler.ckks_params import (
    CKKSProfile,
    get_profile,
    select_optimal_profile,
)
from he_lora_microkernel.compiler.packer import (
    compute_optimal_block_size,
)


@dataclass
class ProfileComparison:
    """Comparison of two CKKS profiles."""
    profile_name: str
    poly_degree: int
    slot_count: int
    block_size: int
    num_blocks: int
    rotations: int
    mul_plain_ops: int
    rescale_ops: int
    estimated_time_us: float


def analyze_profile(
    profile: CKKSProfile,
    hidden_size: int,
    lora_rank: int,
    batch_size: int,
) -> ProfileComparison:
    """Analyze a profile for given workload parameters."""
    params = get_profile(profile)

    # Compute optimal block size
    block_size = compute_optimal_block_size(
        hidden_size=hidden_size,
        batch_size=batch_size,
        slot_count=params.slot_count,
    )

    # Compute number of blocks
    num_blocks = math.ceil(hidden_size / block_size)

    # Compute rotations (tree reduction for cross-block accumulation)
    rotations = 0 if num_blocks == 1 else int(math.ceil(math.log2(num_blocks)))

    # Operation counts (simplified model)
    # Pre-computed AB approach: one matmul with num_blocks blocks
    mul_plain_ops = num_blocks
    rescale_ops = num_blocks

    # Time estimate (microseconds)
    rotation_cost = 500  # μs
    mul_plain_cost = 100  # μs
    rescale_cost = 50  # μs
    encrypt_decrypt_cost = 400  # μs total

    estimated_time = (
        rotations * rotation_cost +
        mul_plain_ops * mul_plain_cost +
        rescale_ops * rescale_cost +
        encrypt_decrypt_cost
    )

    return ProfileComparison(
        profile_name=profile.value.upper(),
        poly_degree=params.poly_modulus_degree,
        slot_count=params.slot_count,
        block_size=block_size,
        num_blocks=num_blocks,
        rotations=rotations,
        mul_plain_ops=mul_plain_ops,
        rescale_ops=rescale_ops,
        estimated_time_us=estimated_time,
    )


def print_comparison_table(comparisons: List[ProfileComparison], config_name: str):
    """Print comparison table for a configuration."""
    print(f"\n{'='*80}")
    print(f"Configuration: {config_name}")
    print(f"{'='*80}")

    headers = [
        "Profile", "N", "Slots", "Block", "Blocks", "Rotations",
        "Ct×Pt", "Rescale", "Est. Time (μs)"
    ]

    # Print header
    print(f"\n{headers[0]:<12} {headers[1]:<7} {headers[2]:<7} {headers[3]:<7} "
          f"{headers[4]:<7} {headers[5]:<10} {headers[6]:<7} {headers[7]:<8} {headers[8]:<15}")
    print("-" * 90)

    # Print rows
    for c in comparisons:
        print(f"{c.profile_name:<12} {c.poly_degree:<7} {c.slot_count:<7} "
              f"{c.block_size:<7} {c.num_blocks:<7} {c.rotations:<10} "
              f"{c.mul_plain_ops:<7} {c.rescale_ops:<8} {c.estimated_time_us:<15.0f}")

    # Calculate improvement
    if len(comparisons) >= 2:
        baseline = comparisons[0]  # TURBO
        optimized = comparisons[1]  # PRODUCTION

        rotation_reduction = (
            (baseline.rotations - optimized.rotations) / max(baseline.rotations, 1) * 100
        )
        time_reduction = (
            (baseline.estimated_time_us - optimized.estimated_time_us) /
            baseline.estimated_time_us * 100
        )

        print("\n  Improvement with PRODUCTION profile:")
        print(f"    Rotation reduction: {rotation_reduction:.1f}%")
        print(f"    Time reduction: {time_reduction:.1f}%")
        print(f"    Blocks: {baseline.num_blocks} → {optimized.num_blocks}")


def run_benchmarks():
    """Run benchmarks for various configurations."""
    print("=" * 80)
    print("PRODUCTION PROFILE OPTIMIZATION BENCHMARK")
    print("=" * 80)
    print("\nComparing N=16384 (TURBO) vs N=32768 (PRODUCTION) for Llama-scale models")
    print("Goal: Demonstrate rotation reduction from larger SIMD slot count")

    # Test configurations (Llama-scale)
    configs = [
        # (hidden_size, lora_rank, batch_size, name)
        (2048, 16, 8, "h=2048, r=16, b=8 (Llama 2B scale)"),
        (4096, 16, 8, "h=4096, r=16, b=8 (Llama 8B scale)"),
        (4096, 32, 8, "h=4096, r=32, b=8 (Higher rank)"),
        (5120, 16, 8, "h=5120, r=16, b=8 (Llama 13B scale)"),
        (8192, 16, 8, "h=8192, r=16, b=8 (Llama 70B scale)"),
    ]

    profiles_to_compare = [CKKSProfile.TURBO, CKKSProfile.PRODUCTION]

    for hidden_size, lora_rank, batch_size, name in configs:
        comparisons = []
        for profile in profiles_to_compare:
            try:
                comp = analyze_profile(profile, hidden_size, lora_rank, batch_size)
                comparisons.append(comp)
            except Exception as e:
                print(f"  {profile.value}: Error - {e}")

        if comparisons:
            print_comparison_table(comparisons, name)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: PRODUCTION PROFILE BENEFITS")
    print("=" * 80)
    print("""
Key Findings:
1. PRODUCTION profile (N=32768) provides 16384 slots vs 8192 for TURBO
2. Larger slot count enables block_size=1024 vs 512
3. Fewer blocks = fewer rotations for cross-block accumulation

Trade-offs:
- PRODUCTION uses 2x memory per ciphertext
- Slightly higher per-operation latency due to larger polynomials
- Benefits outweigh costs for Llama-scale models (h >= 2048)

Recommendation:
- Use PRODUCTION profile for production Llama 8B/70B deployments
- Use TURBO for smaller models or memory-constrained environments
""")

    # Automatic profile selection test
    print("\n" + "=" * 80)
    print("AUTOMATIC PROFILE SELECTION")
    print("=" * 80)

    test_cases = [
        (512, 16, 4, "Small model"),
        (1024, 16, 8, "Medium model"),
        (2048, 16, 8, "Llama 2B scale"),
        (4096, 16, 8, "Llama 8B scale"),
        (8192, 16, 8, "Llama 70B scale"),
    ]

    print(f"\n{'Config':<30} {'Selected Profile':<15} {'Reason'}")
    print("-" * 70)

    for h, r, b, name in test_cases:
        params = select_optimal_profile(h, r, b)
        reason = ""
        if params.profile == CKKSProfile.PRODUCTION:
            reason = "h >= 2048, optimizes rotations"
        elif params.profile == CKKSProfile.TURBO:
            reason = "Large config, needs depth"
        elif params.profile == CKKSProfile.SAFE:
            reason = "Precision requirement"
        else:
            reason = "Default for small workloads"

        print(f"h={h}, r={r}, b={b} ({name:<15}) {params.profile.value:<15} {reason}")


if __name__ == "__main__":
    run_benchmarks()
