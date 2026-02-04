#!/usr/bin/env python3
"""
Weights & Biases Integration with TenSafe

This example demonstrates how to integrate TenSafe training with Weights &
Biases (W&B) for experiment tracking, visualization, and collaboration.
W&B provides powerful tools for ML experiment management.

What this example demonstrates:
- Setting up W&B project for TenSafe experiments
- Logging training metrics and DP privacy budget
- Tracking hyperparameters and configurations
- Saving and versioning model artifacts

Key concepts:
- Experiment tracking: Log metrics, configs, and artifacts
- Privacy metrics: Track epsilon consumption over time
- Artifact versioning: Version control for trained adapters
- Team collaboration: Share experiments with your team

Prerequisites:
- TenSafe server running
- W&B account (wandb.ai)
- pip install wandb

Expected Output:
    W&B Integration Demo

    Initialized W&B run: lora-dp-training-001
    Project: tensafe-experiments

    Training progress:
    Step 100: loss=2.45, epsilon=0.5
    Step 200: loss=1.89, epsilon=1.2
    ...

    View experiment at: https://wandb.ai/your-team/tensafe-experiments/runs/abc123
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import time
import random

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class WandBConfig:
    """Configuration for W&B integration."""
    project: str
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    group: Optional[str] = None


@dataclass
class TrainingMetrics:
    """Training metrics to log."""
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    epsilon: float
    delta: float
    tokens_processed: int


class WandBTracker:
    """Track TenSafe training experiments with Weights & Biases."""

    def __init__(self, config: WandBConfig, hyperparameters: Dict[str, Any]):
        self.config = config
        self.hyperparameters = hyperparameters
        self._initialized = False
        self._run_id = f"run-{random.randint(10000, 99999)}"
        self._run_url = f"https://wandb.ai/{config.entity or 'team'}/{config.project}/runs/{self._run_id}"

    def init(self) -> str:
        """Initialize W&B run."""
        print(f"  Initializing W&B run...")
        print(f"    Project: {self.config.project}")
        print(f"    Run name: {self.config.run_name}")
        print(f"    Tags: {self.config.tags}")
        self._initialized = True
        return self._run_id

    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration/hyperparameters."""
        if not self._initialized:
            raise RuntimeError("W&B not initialized. Call init() first.")
        print(f"  Logged {len(config)} config parameters")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to W&B."""
        if not self._initialized:
            raise RuntimeError("W&B not initialized. Call init() first.")
        # In production, this would call wandb.log(metrics, step=step)

    def log_artifact(self, name: str, artifact_type: str, metadata: Dict[str, Any]) -> str:
        """Log an artifact to W&B."""
        artifact_id = f"artifact-{random.randint(1000, 9999)}"
        print(f"  Logged artifact: {name} (type: {artifact_type})")
        return artifact_id

    def summary(self, metrics: Dict[str, Any]) -> None:
        """Update run summary metrics."""
        print(f"  Updated summary with {len(metrics)} metrics")

    def finish(self) -> None:
        """Finish the W&B run."""
        print(f"  Run finished: {self._run_url}")
        self._initialized = False

    @property
    def run_url(self) -> str:
        return self._run_url


class TenSafeWandBIntegration:
    """Integration layer between TenSafe and W&B."""

    def __init__(self, wandb_config: WandBConfig, training_config: Dict[str, Any]):
        self.wandb_config = wandb_config
        self.training_config = training_config
        self.tracker = WandBTracker(wandb_config, training_config)
        self._step = 0
        self._total_epsilon = 0.0

    def setup(self) -> None:
        """Set up the integration."""
        self.tracker.init()
        self.tracker.log_config(self.training_config)

    def log_training_step(self, loss: float, gradient_norm: float,
                          epsilon_step: float, learning_rate: float) -> None:
        """Log a training step."""
        self._step += 1
        self._total_epsilon += epsilon_step

        metrics = {
            "train/loss": loss,
            "train/gradient_norm": gradient_norm,
            "train/learning_rate": learning_rate,
            "privacy/epsilon_step": epsilon_step,
            "privacy/epsilon_total": self._total_epsilon,
            "privacy/budget_remaining": self.training_config.get("target_epsilon", 8.0) - self._total_epsilon,
        }
        self.tracker.log(metrics, step=self._step)

    def log_evaluation(self, eval_loss: float, eval_metrics: Dict[str, float]) -> None:
        """Log evaluation metrics."""
        metrics = {"eval/loss": eval_loss}
        metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        self.tracker.log(metrics, step=self._step)

    def log_checkpoint(self, checkpoint_path: str, metadata: Dict[str, Any]) -> str:
        """Log a checkpoint as an artifact."""
        artifact_name = f"checkpoint-step-{self._step}"
        return self.tracker.log_artifact(
            name=artifact_name,
            artifact_type="model",
            metadata={
                "step": self._step,
                "epsilon": self._total_epsilon,
                "path": checkpoint_path,
                **metadata,
            }
        )

    def finish(self, final_metrics: Dict[str, Any]) -> str:
        """Finish tracking and return run URL."""
        self.tracker.summary(final_metrics)
        self.tracker.finish()
        return self.tracker.run_url


def main():
    """Demonstrate W&B integration with TenSafe."""

    # =========================================================================
    # Step 1: Understanding W&B Integration
    # =========================================================================
    print("=" * 60)
    print("WEIGHTS & BIASES INTEGRATION")
    print("=" * 60)
    print("""
    Why integrate TenSafe with W&B?

    1. EXPERIMENT TRACKING
       - Log metrics, hyperparameters, and configs
       - Compare different training runs
       - Reproduce experiments easily

    2. PRIVACY BUDGET VISUALIZATION
       - Track epsilon consumption over time
       - Set alerts for budget thresholds
       - Visualize privacy-utility tradeoffs

    3. ARTIFACT VERSIONING
       - Version control for TGSP adapters
       - Track lineage from data to model
       - Easy rollback to previous versions

    4. TEAM COLLABORATION
       - Share experiments with team members
       - Collaborate on hyperparameter tuning
       - Document findings with notes
    """)

    # =========================================================================
    # Step 2: Configure W&B Integration
    # =========================================================================
    print("\nConfiguring W&B integration...")

    wandb_config = WandBConfig(
        project="tensafe-experiments",
        entity="ml-team",
        run_name="lora-dp-training-001",
        tags=["lora", "differential-privacy", "llama-3"],
        notes="Testing DP-SGD with noise multiplier 1.0",
        group="hyperparameter-sweep",
    )

    training_config = {
        "model": "meta-llama/Llama-3-8B",
        "lora_rank": 16,
        "lora_alpha": 32.0,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "batch_size": 8,
        "learning_rate": 1e-4,
        "noise_multiplier": 1.0,
        "max_grad_norm": 1.0,
        "target_epsilon": 8.0,
        "target_delta": 1e-5,
        "num_epochs": 3,
    }

    print(f"  Project: {wandb_config.project}")
    print(f"  Run name: {wandb_config.run_name}")
    print(f"  Tags: {wandb_config.tags}")

    # =========================================================================
    # Step 3: Initialize Integration
    # =========================================================================
    print("\n" + "=" * 60)
    print("INITIALIZING W&B RUN")
    print("=" * 60)

    integration = TenSafeWandBIntegration(wandb_config, training_config)
    integration.setup()

    print("\nConfiguration logged:")
    for key, value in list(training_config.items())[:5]:
        print(f"  {key}: {value}")
    print(f"  ... and {len(training_config) - 5} more")

    # =========================================================================
    # Step 4: Simulate Training with Logging
    # =========================================================================
    print("\n" + "=" * 60)
    print("TRAINING WITH W&B LOGGING")
    print("=" * 60)

    num_steps = 500
    checkpoint_interval = 100
    eval_interval = 50

    print(f"\nStarting training for {num_steps} steps...")
    print("-" * 50)

    for step in range(1, num_steps + 1):
        # Simulate training step
        loss = 3.0 * (0.95 ** (step / 50)) + random.gauss(0, 0.1)
        gradient_norm = 0.5 + random.gauss(0, 0.1)
        epsilon_step = 0.01 + random.gauss(0, 0.001)
        learning_rate = 1e-4 * (1 - step / num_steps)  # Linear decay

        # Log training metrics
        integration.log_training_step(loss, gradient_norm, epsilon_step, learning_rate)

        # Evaluation
        if step % eval_interval == 0:
            eval_loss = loss * 1.1
            eval_metrics = {
                "perplexity": 2 ** eval_loss,
                "accuracy": min(0.95, 0.5 + step / 1000),
            }
            integration.log_evaluation(eval_loss, eval_metrics)
            print(f"Step {step}: loss={loss:.4f}, epsilon={integration._total_epsilon:.3f}, "
                  f"eval_loss={eval_loss:.4f}")

        # Checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_path = f"/checkpoints/step-{step}.tgsp"
            artifact_id = integration.log_checkpoint(
                checkpoint_path,
                metadata={"loss": loss, "eval_loss": eval_loss}
            )

    # =========================================================================
    # Step 5: Finalize and View Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINALIZING W&B RUN")
    print("=" * 60)

    final_metrics = {
        "final_loss": loss,
        "final_epsilon": integration._total_epsilon,
        "total_steps": num_steps,
        "checkpoints_saved": num_steps // checkpoint_interval,
    }

    run_url = integration.finish(final_metrics)

    print(f"""
    Training complete!

    Final metrics:
    - Loss: {final_metrics['final_loss']:.4f}
    - Total epsilon: {final_metrics['final_epsilon']:.3f}
    - Steps: {final_metrics['total_steps']}
    - Checkpoints: {final_metrics['checkpoints_saved']}

    View experiment at: {run_url}
    """)

    # =========================================================================
    # Best Practices
    # =========================================================================
    print("=" * 60)
    print("W&B INTEGRATION BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for effective experiment tracking:

    1. Log comprehensively
       - Training loss, gradient norms, learning rate
       - Privacy budget (epsilon/delta per step and total)
       - Evaluation metrics at regular intervals

    2. Use meaningful run names
       - Include key hyperparameters: "lora-r16-eps8-nm1.0"
       - Use groups for related experiments
       - Add descriptive tags

    3. Version your artifacts
       - Log checkpoints as W&B artifacts
       - Include metadata (step, epsilon, metrics)
       - Use artifact aliases for production models

    4. Set up alerts
       - Alert when privacy budget exceeds threshold
       - Monitor for training anomalies
       - Track resource utilization

    5. Document experiments
       - Add notes explaining experiment purpose
       - Record findings and observations
       - Link to related experiments
    """)


if __name__ == "__main__":
    main()
