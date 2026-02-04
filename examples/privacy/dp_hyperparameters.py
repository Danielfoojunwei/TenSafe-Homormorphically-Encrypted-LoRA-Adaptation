#!/usr/bin/env python3
"""
Differential Privacy Hyperparameter Tuning with TenSafe

This example demonstrates how to tune hyperparameters for DP training
to achieve the best privacy-utility tradeoff. Proper tuning is essential
for getting useful models under privacy constraints.

What this example demonstrates:
- Key hyperparameters affecting DP training
- Strategies for noise multiplier selection
- Batch size optimization for DP
- Privacy amplification through sampling

Key concepts:
- Noise multiplier vs epsilon tradeoff
- Batch size impact on privacy
- Gradient clipping strategies
- Learning rate scaling for DP

Prerequisites:
- TenSafe server running

Expected Output:
    Hyperparameter Analysis
    =======================
    Noise=0.5: epsilon=16.0, utility=0.95
    Noise=1.0: epsilon=8.0,  utility=0.88
    Noise=2.0: epsilon=4.0,  utility=0.72

    Recommended: noise=1.0 for epsilon=8.0 target
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import math

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class DPHyperparameters:
    """Hyperparameters for DP training."""
    noise_multiplier: float
    max_grad_norm: float
    batch_size: int
    learning_rate: float
    num_epochs: int


@dataclass
class DPAnalysisResult:
    """Result of DP hyperparameter analysis."""
    hyperparams: DPHyperparameters
    estimated_epsilon: float
    estimated_utility: float
    privacy_amplification: float


def compute_epsilon(
    noise_multiplier: float,
    sampling_rate: float,
    num_steps: int,
    delta: float = 1e-5
) -> float:
    """Compute epsilon using simplified RDP accounting."""
    base_epsilon = sampling_rate * math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier
    total_epsilon = base_epsilon * math.sqrt(num_steps)
    return min(total_epsilon, 100.0)


def estimate_utility(noise_multiplier: float, learning_rate: float) -> float:
    """Estimate utility for given hyperparameters."""
    base_utility = 0.95
    noise_penalty = 0.1 * math.log1p(noise_multiplier)
    lr_bonus = 0.05 if 1e-4 <= learning_rate <= 5e-4 else 0.0
    return max(0.5, base_utility - noise_penalty + lr_bonus)


def analyze_hyperparameters(
    target_epsilon: float,
    dataset_size: int,
    delta: float = 1e-5
) -> List[DPAnalysisResult]:
    """Analyze different hyperparameter configurations."""
    results = []

    noise_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    batch_sizes = [32, 64, 128, 256]

    for noise in noise_multipliers:
        for batch_size in batch_sizes:
            sampling_rate = batch_size / dataset_size
            steps_per_epoch = dataset_size // batch_size
            num_epochs = 5
            num_steps = steps_per_epoch * num_epochs

            epsilon = compute_epsilon(noise, sampling_rate, num_steps, delta)
            if epsilon > target_epsilon * 2:
                continue

            lr = 1e-4 * (noise / 1.0)
            utility = estimate_utility(noise, lr)

            hyperparams = DPHyperparameters(
                noise_multiplier=noise,
                max_grad_norm=1.0,
                batch_size=batch_size,
                learning_rate=lr,
                num_epochs=num_epochs,
            )

            results.append(DPAnalysisResult(
                hyperparams=hyperparams,
                estimated_epsilon=epsilon,
                estimated_utility=utility,
                privacy_amplification=1.0 / (1.0 + sampling_rate),
            ))

    results.sort(key=lambda x: (-x.estimated_utility, x.estimated_epsilon))
    return results


def main():
    """Demonstrate DP hyperparameter tuning."""

    # =========================================================================
    # Step 1: Understanding DP hyperparameters
    # =========================================================================
    print("=" * 60)
    print("DP HYPERPARAMETER TUNING")
    print("=" * 60)
    print("""
    Key hyperparameters for DP training:

    1. NOISE MULTIPLIER (sigma)
       - Controls amount of noise added to gradients
       - Higher noise = more privacy, less utility
       - Typical range: 0.5 - 3.0

    2. MAX GRADIENT NORM (C)
       - Clips per-sample gradients to bound sensitivity
       - Too low: gradient information lost
       - Too high: more noise needed
       - Typical: 0.1 - 10.0 (often 1.0)

    3. BATCH SIZE
       - Larger batches = more privacy (privacy amplification)
       - DP benefits from larger batches more than non-DP

    4. LEARNING RATE
       - Often needs to be higher for DP training
       - Compensates for noisy gradients
    """)

    # =========================================================================
    # Step 2: Configure analysis
    # =========================================================================
    print("\nConfiguring hyperparameter analysis...")

    target_epsilon = 8.0
    dataset_size = 50000
    delta = 1e-5

    print(f"  Target epsilon: {target_epsilon}")
    print(f"  Dataset size: {dataset_size}")
    print(f"  Delta: {delta}")

    # =========================================================================
    # Step 3: Analyze hyperparameter combinations
    # =========================================================================
    print("\nAnalyzing hyperparameter combinations...")
    print("-" * 60)

    results = analyze_hyperparameters(target_epsilon, dataset_size, delta)

    print("\nTop configurations (sorted by utility):")
    print("-" * 60)
    print(f"{'Noise':>6} {'Batch':>6} {'LR':>10} {'Epsilon':>8} {'Utility':>8}")
    print("-" * 60)

    for result in results[:10]:
        hp = result.hyperparams
        print(f"{hp.noise_multiplier:>6.2f} {hp.batch_size:>6} "
              f"{hp.learning_rate:>10.2e} {result.estimated_epsilon:>8.2f} "
              f"{result.estimated_utility:>8.2f}")

    # =========================================================================
    # Step 4: Recommendations
    # =========================================================================
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    meeting_target = [r for r in results if r.estimated_epsilon <= target_epsilon]
    if meeting_target:
        best = meeting_target[0]
        print(f"""
    Recommended configuration for epsilon={target_epsilon}:

    Noise multiplier: {best.hyperparams.noise_multiplier}
    Max gradient norm: {best.hyperparams.max_grad_norm}
    Batch size: {best.hyperparams.batch_size}
    Learning rate: {best.hyperparams.learning_rate:.2e}
    Epochs: {best.hyperparams.num_epochs}

    Expected results:
    - Epsilon: {best.estimated_epsilon:.2f}
    - Estimated utility: {best.estimated_utility:.2f}
    """)

    print("""
    General tuning tips:

    1. Start with noise_multiplier=1.0 and adjust
    2. Use the largest batch size that fits in memory
    3. Increase learning rate 2-10x compared to non-DP
    4. Monitor validation loss to detect underfitting
    5. Use learning rate warmup for stability
    """)


if __name__ == "__main__":
    main()
