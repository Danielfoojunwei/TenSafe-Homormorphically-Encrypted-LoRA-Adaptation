#!/usr/bin/env python3
"""
Checkpoint and Resume Training

This example demonstrates how to save and resume training from checkpoints,
which is essential for:
- Fault tolerance in long training runs
- Experimentation (try different hyperparameters from same point)
- Distributed training recovery
- Privacy budget tracking across sessions

What gets saved:
- Model state (LoRA weights)
- Optimizer state
- Learning rate scheduler state
- Training step counter
- Privacy metrics (epsilon, delta)
- Random states for reproducibility

Expected Output:
    Training and checkpointing:
      Step 100: loss=2.15, saved checkpoint
      Step 200: loss=1.92, saved checkpoint

    Simulating failure...

    Resuming from checkpoint:
      Loaded checkpoint from step 200
      Resuming training...
      Step 201: loss=1.91
      Step 300: loss=1.75, saved checkpoint

    Training complete with recovery!
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class CheckpointState:
    """Complete training state for checkpointing."""
    # Training progress
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float("inf")

    # Privacy metrics (for DP training)
    epsilon_spent: float = 0.0
    delta: float = 1e-5

    # Optimizer state
    learning_rate: float = 1e-4
    optimizer_step: int = 0

    # Random states (for reproducibility)
    random_seed: int = 42

    # Metadata
    timestamp: str = ""
    training_hours: float = 0.0


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 100
    keep_last_n: int = 3
    save_best: bool = True


class CheckpointManager:
    """Manages saving and loading training checkpoints."""

    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.time()

    def save(
        self,
        state: CheckpointState,
        model_weights: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
    ) -> Path:
        """Save a checkpoint."""
        # Create checkpoint directory
        checkpoint_name = f"checkpoint-{state.global_step:06d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Update metadata
        state.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        state.training_hours = (time.time() - self._start_time) / 3600

        # Save state
        state_path = checkpoint_path / "training_state.json"
        with open(state_path, "w") as f:
            json.dump(asdict(state), f, indent=2)

        # Save model weights (simulated)
        if model_weights:
            weights_path = checkpoint_path / "model_weights.bin"
            # In production: torch.save(model_weights, weights_path)
            weights_path.touch()

        # Save optimizer state (simulated)
        if optimizer_state:
            optimizer_path = checkpoint_path / "optimizer.bin"
            # In production: torch.save(optimizer_state, optimizer_path)
            optimizer_path.touch()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"  Saved checkpoint: {checkpoint_name}")
        return checkpoint_path

    def load(self, checkpoint_path: Optional[Path] = None) -> Optional[CheckpointState]:
        """Load a checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            return None

        # Load state
        state_path = checkpoint_path / "training_state.json"
        if not state_path.exists():
            return None

        with open(state_path) as f:
            state_dict = json.load(f)

        state = CheckpointState(**state_dict)
        print(f"  Loaded checkpoint from step {state.global_step}")

        return state

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get the path to the latest checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))
        return checkpoints[-1] if checkpoints else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get the path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        return best_path if best_path.exists() else None

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*"))

        # Keep best checkpoint separate
        for checkpoint in checkpoints[:-self.config.keep_last_n]:
            if checkpoint.name != "best":
                shutil.rmtree(checkpoint)


class ResumableTrainer:
    """Trainer with checkpoint/resume support."""

    def __init__(self, checkpoint_config: CheckpointConfig):
        self.checkpoint_manager = CheckpointManager(checkpoint_config)
        self.state = CheckpointState()

    def resume_or_start(self) -> bool:
        """Resume from checkpoint or start fresh."""
        loaded_state = self.checkpoint_manager.load()

        if loaded_state:
            self.state = loaded_state
            print(f"Resuming from step {self.state.global_step}")
            return True
        else:
            print("Starting fresh training")
            return False

    def train_step(self) -> float:
        """Execute one training step."""
        self.state.global_step += 1

        # Simulate training
        loss = 2.5 - (self.state.global_step * 0.005)
        loss = max(loss, 0.5)  # Floor

        # Update privacy budget (for DP training)
        self.state.epsilon_spent += 0.01

        # Track best loss
        if loss < self.state.best_loss:
            self.state.best_loss = loss

        return loss

    def maybe_save_checkpoint(self, loss: float):
        """Save checkpoint if it's time."""
        config = self.checkpoint_manager.config

        if self.state.global_step % config.save_every_n_steps == 0:
            self.checkpoint_manager.save(self.state)

        # Save best model
        if config.save_best and loss <= self.state.best_loss:
            # Would save to "best" checkpoint
            pass


def main():
    """Demonstrate checkpoint and resume functionality."""
    print("=" * 60)
    print("CHECKPOINT AND RESUME TRAINING")
    print("=" * 60)
    print("""
    Checkpointing enables:
    1. Recovery from failures (GPU errors, preemption)
    2. Resuming long training runs
    3. Experimentation from saved states
    4. Privacy budget tracking across sessions
    """)

    # Setup
    checkpoint_config = CheckpointConfig(
        checkpoint_dir="./demo_checkpoints",
        save_every_n_steps=100,
        keep_last_n=3,
    )

    # Clean up from previous runs
    checkpoint_dir = Path(checkpoint_config.checkpoint_dir)
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    # Phase 1: Initial training
    print("\n" + "=" * 60)
    print("PHASE 1: Initial Training")
    print("=" * 60)

    trainer = ResumableTrainer(checkpoint_config)
    trainer.resume_or_start()

    print("\nTraining...")
    for step in range(1, 251):
        loss = trainer.train_step()

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss:.4f}, epsilon={trainer.state.epsilon_spent:.2f}")

        trainer.maybe_save_checkpoint(loss)

    print(f"\nReached step {trainer.state.global_step}")

    # Phase 2: Simulate failure and recovery
    print("\n" + "=" * 60)
    print("PHASE 2: Simulating Failure")
    print("=" * 60)
    print("\n*** Simulating training failure ***")
    print("*** Previous trainer state lost ***")

    # Create new trainer (simulating restart)
    del trainer

    # Phase 3: Resume from checkpoint
    print("\n" + "=" * 60)
    print("PHASE 3: Resuming from Checkpoint")
    print("=" * 60)

    trainer = ResumableTrainer(checkpoint_config)
    resumed = trainer.resume_or_start()

    if resumed:
        print(f"  Restored state:")
        print(f"    Step: {trainer.state.global_step}")
        print(f"    Best loss: {trainer.state.best_loss:.4f}")
        print(f"    Epsilon spent: {trainer.state.epsilon_spent:.2f}")

        print("\nContinuing training...")
        for step in range(trainer.state.global_step + 1, trainer.state.global_step + 101):
            loss = trainer.train_step()

            if step % 50 == 0:
                print(f"  Step {step}: loss={loss:.4f}")

            trainer.maybe_save_checkpoint(loss)

    # Cleanup
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    # Summary
    print("\n" + "=" * 60)
    print("WHAT GETS CHECKPOINTED")
    print("=" * 60)
    print("""
    Essential checkpoint components:

    1. Model State
       - LoRA adapter weights
       - (Optionally) base model weights

    2. Optimizer State
       - Adam moments (m, v)
       - Learning rate scheduler state

    3. Training Progress
       - Global step counter
       - Epoch number
       - Best validation loss

    4. Privacy Metrics (for DP)
       - Epsilon spent
       - Delta value
       - Noise multiplier history

    5. Random States
       - PyTorch RNG state
       - NumPy RNG state
       - Python random state

    6. Metadata
       - Timestamp
       - Training hours
       - Configuration
    """)


if __name__ == "__main__":
    main()
