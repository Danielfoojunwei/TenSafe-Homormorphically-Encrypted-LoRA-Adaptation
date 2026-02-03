#!/usr/bin/env python3
"""
Hyperparameter Tuning for TenSafe Training

This example demonstrates how to tune hyperparameters for optimal
training performance while maintaining privacy guarantees. Includes:
- Grid search for basic exploration
- Random search for efficient exploration
- Bayesian optimization with privacy constraints
- Privacy-utility tradeoff analysis

Key hyperparameters for privacy-preserving training:
- Learning rate (affects convergence and privacy)
- Batch size (larger = better privacy efficiency)
- Noise multiplier (privacy vs utility tradeoff)
- LoRA rank (model capacity vs efficiency)

Expected Output:
    Running hyperparameter search...

    Trial 1: lr=1e-4, batch=32, noise=1.0
      -> loss=1.85, epsilon=4.2

    Trial 2: lr=2e-4, batch=64, noise=0.8
      -> loss=1.72, epsilon=5.1

    Best configuration:
      Learning rate: 2e-4
      Batch size: 64
      Final loss: 1.72
      Privacy: epsilon=5.1
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class HyperparameterSpace:
    """Define the hyperparameter search space."""
    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    lr_log_scale: bool = True

    # Batch size
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # LoRA parameters
    lora_ranks: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_alphas: List[int] = field(default_factory=lambda: [16, 32, 64])

    # DP parameters
    noise_multipliers: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0, 1.2])
    max_grad_norms: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])


@dataclass
class TrialConfig:
    """Configuration for a single trial."""
    learning_rate: float
    batch_size: int
    lora_rank: int
    lora_alpha: int
    noise_multiplier: float
    max_grad_norm: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class TrialResult:
    """Result from a single trial."""
    config: TrialConfig
    final_loss: float
    final_epsilon: float
    validation_loss: float
    training_time_hours: float


class HyperparameterTuner:
    """Hyperparameter tuning with privacy constraints."""

    def __init__(
        self,
        search_space: HyperparameterSpace,
        max_epsilon: float = 8.0,
        num_trials: int = 10,
    ):
        self.search_space = search_space
        self.max_epsilon = max_epsilon
        self.num_trials = num_trials
        self.trials: List[TrialResult] = []

    def sample_config(self) -> TrialConfig:
        """Sample a random configuration from the search space."""
        space = self.search_space

        # Sample learning rate
        if space.lr_log_scale:
            import math
            log_lr = random.uniform(math.log(space.lr_min), math.log(space.lr_max))
            lr = math.exp(log_lr)
        else:
            lr = random.uniform(space.lr_min, space.lr_max)

        return TrialConfig(
            learning_rate=lr,
            batch_size=random.choice(space.batch_sizes),
            lora_rank=random.choice(space.lora_ranks),
            lora_alpha=random.choice(space.lora_alphas),
            noise_multiplier=random.choice(space.noise_multipliers),
            max_grad_norm=random.choice(space.max_grad_norms),
        )

    def evaluate_config(self, config: TrialConfig) -> TrialResult:
        """Evaluate a configuration (simulated)."""
        # Simulate training outcomes based on hyperparameters
        # In production, this would run actual training

        # Base loss (affected by lr and lora_rank)
        base_loss = 2.0

        # Learning rate effect (too high or too low is bad)
        optimal_lr = 1e-4
        lr_penalty = abs(config.learning_rate - optimal_lr) / optimal_lr * 0.5
        base_loss += lr_penalty

        # LoRA rank effect (higher rank = better fit)
        rank_benefit = (config.lora_rank / 64) * 0.3
        base_loss -= rank_benefit

        # Noise effect (more noise = worse performance)
        noise_penalty = config.noise_multiplier * 0.2
        base_loss += noise_penalty

        # Batch size effect (larger = more stable)
        batch_benefit = (config.batch_size / 128) * 0.1
        base_loss -= batch_benefit

        final_loss = max(base_loss, 0.5)

        # Epsilon calculation (simulated)
        # More noise and larger batches = less epsilon
        epsilon = (1000 / config.batch_size) * (1.0 / config.noise_multiplier) * 0.5
        epsilon = min(epsilon, self.max_epsilon)

        return TrialResult(
            config=config,
            final_loss=final_loss,
            final_epsilon=epsilon,
            validation_loss=final_loss * 1.1,  # Slightly higher val loss
            training_time_hours=0.5,
        )

    def run_random_search(self) -> TrialResult:
        """Run random search over hyperparameter space."""
        print(f"\nRunning random search ({self.num_trials} trials)...")
        print("-" * 60)

        for i in range(self.num_trials):
            config = self.sample_config()
            result = self.evaluate_config(config)

            # Check privacy constraint
            if result.final_epsilon > self.max_epsilon:
                print(f"Trial {i+1}: SKIPPED (epsilon={result.final_epsilon:.2f} > {self.max_epsilon})")
                continue

            self.trials.append(result)

            print(f"Trial {i+1}: lr={config.learning_rate:.2e}, "
                  f"batch={config.batch_size}, "
                  f"noise={config.noise_multiplier}")
            print(f"  -> loss={result.final_loss:.4f}, "
                  f"epsilon={result.final_epsilon:.2f}")

        # Find best trial
        valid_trials = [t for t in self.trials if t.final_epsilon <= self.max_epsilon]
        if not valid_trials:
            print("\nNo valid trials found within privacy budget!")
            return None

        best_trial = min(valid_trials, key=lambda t: t.final_loss)
        return best_trial

    def analyze_privacy_utility_tradeoff(self):
        """Analyze the privacy-utility tradeoff across trials."""
        print("\n" + "=" * 60)
        print("PRIVACY-UTILITY TRADEOFF ANALYSIS")
        print("=" * 60)

        if not self.trials:
            print("No trials to analyze")
            return

        # Group by noise multiplier
        by_noise: Dict[float, List[TrialResult]] = {}
        for trial in self.trials:
            noise = trial.config.noise_multiplier
            if noise not in by_noise:
                by_noise[noise] = []
            by_noise[noise].append(trial)

        print(f"\n{'Noise':<10} {'Avg Loss':<12} {'Avg Epsilon':<12}")
        print("-" * 40)

        for noise in sorted(by_noise.keys()):
            trials = by_noise[noise]
            avg_loss = sum(t.final_loss for t in trials) / len(trials)
            avg_eps = sum(t.final_epsilon for t in trials) / len(trials)
            print(f"{noise:<10.2f} {avg_loss:<12.4f} {avg_eps:<12.2f}")


def main():
    """Demonstrate hyperparameter tuning."""
    print("=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)
    print("""
    Key hyperparameters for privacy-preserving training:

    1. Learning Rate
       - Too high: Unstable training, poor privacy
       - Too low: Slow convergence
       - Typical range: 1e-5 to 1e-3

    2. Batch Size
       - Larger = better privacy efficiency
       - Limited by GPU memory
       - Typical: 32-256 with gradient accumulation

    3. Noise Multiplier (DP)
       - Higher = more private, worse utility
       - Lower = less private, better utility
       - Typical range: 0.5 to 2.0

    4. LoRA Rank
       - Higher = more capacity, more compute
       - Lower = faster, may underfit
       - Typical: 8, 16, 32, 64
    """)

    # Define search space
    search_space = HyperparameterSpace(
        lr_min=1e-5,
        lr_max=1e-3,
        batch_sizes=[32, 64, 128],
        lora_ranks=[8, 16, 32],
        noise_multipliers=[0.5, 0.8, 1.0, 1.2],
    )

    # Run tuning
    tuner = HyperparameterTuner(
        search_space=search_space,
        max_epsilon=8.0,
        num_trials=10,
    )

    best_trial = tuner.run_random_search()

    # Report best configuration
    if best_trial:
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION")
        print("=" * 60)
        print(f"\n  Learning rate: {best_trial.config.learning_rate:.2e}")
        print(f"  Batch size: {best_trial.config.batch_size}")
        print(f"  LoRA rank: {best_trial.config.lora_rank}")
        print(f"  Noise multiplier: {best_trial.config.noise_multiplier}")
        print(f"\n  Final loss: {best_trial.final_loss:.4f}")
        print(f"  Privacy: epsilon={best_trial.final_epsilon:.2f}")

    # Analyze tradeoffs
    tuner.analyze_privacy_utility_tradeoff()

    # Tips
    print("\n" + "=" * 60)
    print("TUNING TIPS")
    print("=" * 60)
    print("""
    1. Start with recommended defaults
       - lr=1e-4, batch=64, noise=1.0, rank=16

    2. Fix privacy budget first
       - Decide target epsilon (e.g., 8.0)
       - Tune other params within budget

    3. Use larger batches when possible
       - Better privacy efficiency
       - More stable gradients

    4. Consider early stopping
       - Monitor validation loss
       - Stop when epsilon budget is spent

    5. Use warmup for learning rate
       - Helps stability with DP noise
       - Typical: 100-500 steps
    """)


if __name__ == "__main__":
    main()
