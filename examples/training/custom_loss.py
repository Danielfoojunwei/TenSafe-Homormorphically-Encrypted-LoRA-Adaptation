#!/usr/bin/env python3
"""
Custom Loss Functions for TenSafe

This example demonstrates how to implement and use custom loss functions
with TenSafe training. Includes examples for:
- Entropy-regularized cross-entropy
- Focal loss for imbalanced data
- Label smoothing
- Contrastive loss for embeddings
- Custom reward-based losses

Expected Output:
    Registering custom losses...
    Loss: entropy_regularized_ce registered
    Loss: focal_ce registered

    Training with custom loss:
      Step 10: loss=2.15, ce=2.10, entropy=0.05
      Step 20: loss=1.98, ce=1.95, entropy=0.03

    Custom loss training complete!
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Union

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Simulate torch for examples
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Custom Loss Functions
# =============================================================================

def cross_entropy_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Standard cross-entropy loss for language modeling.

    Args:
        outputs: Model outputs with 'logits' key
        batch: Batch with 'labels' key
        ignore_index: Token ID to ignore in loss
        label_smoothing: Label smoothing factor

    Returns:
        Dict with 'loss' and 'metrics'
    """
    # Simulated loss computation
    loss = 2.0

    return {
        "loss": loss,
        "metrics": {
            "ce_loss": loss,
            "perplexity": 7.39,  # exp(loss)
        }
    }


def entropy_regularized_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    *,
    entropy_weight: float = 0.01,
    ignore_index: int = -100,
    **kwargs,
) -> Dict[str, Any]:
    """
    Cross-entropy with entropy regularization.

    Encourages model to maintain diverse predictions, preventing
    overconfident outputs.

    Total loss = CE_loss - entropy_weight * entropy

    Higher entropy = more diverse predictions
    """
    # Simulated computation
    ce_loss = 2.0
    entropy = 0.5  # Average entropy per token
    total_loss = ce_loss - entropy_weight * entropy

    return {
        "loss": total_loss,
        "metrics": {
            "ce_loss": ce_loss,
            "entropy": entropy,
            "entropy_contribution": entropy_weight * entropy,
            "perplexity": 7.39,
        }
    }


def focal_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    *,
    gamma: float = 2.0,
    alpha: float = 1.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Focal loss for handling class imbalance.

    Down-weights well-classified examples, focusing on hard examples.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter (0 = standard CE)
        alpha: Class weight factor
    """
    # Simulated computation
    ce_loss = 2.0
    avg_confidence = 0.3  # Average confidence on correct class
    focal_weight = (1 - avg_confidence) ** gamma

    loss = alpha * focal_weight * ce_loss

    return {
        "loss": loss,
        "metrics": {
            "ce_loss": ce_loss,
            "focal_weight": focal_weight,
            "avg_confidence": avg_confidence,
            "gamma": gamma,
        }
    }


def contrastive_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    *,
    temperature: float = 0.07,
    **kwargs,
) -> Dict[str, Any]:
    """
    Contrastive loss for embedding learning (InfoNCE).

    Pulls positive pairs together, pushes negative pairs apart.

    Args:
        temperature: Softmax temperature for similarity scaling
    """
    # Simulated computation
    loss = 1.5
    accuracy = 0.75  # Contrastive accuracy

    return {
        "loss": loss,
        "metrics": {
            "contrastive_loss": loss,
            "contrastive_accuracy": accuracy,
            "temperature": temperature,
        }
    }


def reward_weighted_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    *,
    reward_scale: float = 1.0,
    baseline: float = 0.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Reward-weighted loss for RLHF-style training.

    Weights the CE loss by the reward signal.
    Loss = -log(p) * (reward - baseline)

    Args:
        reward_scale: Scaling factor for rewards
        baseline: Baseline reward for variance reduction
    """
    # Get reward from batch
    rewards = batch.get("rewards", [0.5])  # Default reward
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # Simulated computation
    ce_loss = 2.0
    advantage = (avg_reward - baseline) * reward_scale

    # Weighted loss
    loss = ce_loss * max(advantage, 0.1)  # Clamp to avoid negative weights

    return {
        "loss": loss,
        "metrics": {
            "ce_loss": ce_loss,
            "avg_reward": avg_reward,
            "advantage": advantage,
        }
    }


# =============================================================================
# Loss Registry
# =============================================================================

LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(name: str, loss_fn: Callable):
    """Register a custom loss function."""
    LOSS_REGISTRY[name] = loss_fn
    print(f"Loss: {name} registered")


def get_loss(name: str) -> Callable:
    """Get a loss function by name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss: {name}. Available: {list(LOSS_REGISTRY.keys())}")
    return LOSS_REGISTRY[name]


# =============================================================================
# Training with Custom Loss
# =============================================================================

@dataclass
class CustomLossTrainingConfig:
    """Configuration for training with custom loss."""
    loss_name: str = "cross_entropy"
    loss_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.loss_kwargs is None:
            self.loss_kwargs = {}


class CustomLossTrainer:
    """Trainer with custom loss function support."""

    def __init__(self, config: CustomLossTrainingConfig):
        self.config = config
        self.loss_fn = get_loss(config.loss_name)

    def train_step(self, batch: Dict) -> Dict[str, Any]:
        """Execute one training step with custom loss."""
        # Simulate model outputs
        outputs = {"logits": None}

        # Compute loss with custom function
        result = self.loss_fn(
            outputs,
            batch,
            **self.config.loss_kwargs
        )

        return result


def main():
    """Demonstrate custom loss functions."""
    print("=" * 60)
    print("CUSTOM LOSS FUNCTIONS")
    print("=" * 60)

    # Register built-in losses
    print("\nRegistering custom losses...")
    register_loss("cross_entropy", cross_entropy_loss)
    register_loss("entropy_regularized", entropy_regularized_loss)
    register_loss("focal", focal_loss)
    register_loss("contrastive", contrastive_loss)
    register_loss("reward_weighted", reward_weighted_loss)

    print(f"\nAvailable losses: {list(LOSS_REGISTRY.keys())}")

    # Example 1: Entropy-regularized loss
    print("\n" + "-" * 40)
    print("Example 1: Entropy-Regularized Loss")
    print("-" * 40)

    config = CustomLossTrainingConfig(
        loss_name="entropy_regularized",
        loss_kwargs={"entropy_weight": 0.01}
    )
    trainer = CustomLossTrainer(config)

    for step in range(1, 6):
        batch = {"labels": None}
        result = trainer.train_step(batch)
        print(f"  Step {step}: loss={result['loss']:.4f}, "
              f"ce={result['metrics']['ce_loss']:.4f}, "
              f"entropy={result['metrics']['entropy']:.4f}")

    # Example 2: Focal loss for imbalanced data
    print("\n" + "-" * 40)
    print("Example 2: Focal Loss")
    print("-" * 40)

    config = CustomLossTrainingConfig(
        loss_name="focal",
        loss_kwargs={"gamma": 2.0, "alpha": 0.5}
    )
    trainer = CustomLossTrainer(config)

    result = trainer.train_step({"labels": None})
    print(f"  Loss: {result['loss']:.4f}")
    print(f"  Focal weight: {result['metrics']['focal_weight']:.4f}")
    print(f"  This down-weights easy examples, focusing on hard ones")

    # Example 3: Reward-weighted loss
    print("\n" + "-" * 40)
    print("Example 3: Reward-Weighted Loss (RLHF-style)")
    print("-" * 40)

    config = CustomLossTrainingConfig(
        loss_name="reward_weighted",
        loss_kwargs={"reward_scale": 1.0, "baseline": 0.5}
    )
    trainer = CustomLossTrainer(config)

    # High reward sample
    result = trainer.train_step({"labels": None, "rewards": [0.9]})
    print(f"  High reward (0.9): loss={result['loss']:.4f}, "
          f"advantage={result['metrics']['advantage']:.4f}")

    # Low reward sample
    result = trainer.train_step({"labels": None, "rewards": [0.1]})
    print(f"  Low reward (0.1): loss={result['loss']:.4f}, "
          f"advantage={result['metrics']['advantage']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("CREATING YOUR OWN LOSS")
    print("=" * 60)
    print("""
    To create a custom loss function:

    1. Define a function with signature:
       def my_loss(outputs, batch, **kwargs) -> Dict[str, Any]

    2. Return dict with 'loss' and 'metrics' keys:
       return {
           "loss": computed_loss,
           "metrics": {"custom_metric": value}
       }

    3. Register with TenSafe:
       register_loss("my_loss", my_loss)

    4. Use in training config:
       config = TrainingConfig(loss_fn="my_loss", loss_kwargs={...})

    See examples/custom_loss/entropy_regularized_loss.py for a complete example.
    """)


if __name__ == "__main__":
    main()
