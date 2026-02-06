"""
Base classes for RLVR algorithms.

Defines the interface that all RL algorithms must implement.
"""

from __future__ import annotations

# Import trajectory types
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tensafe.rlvr.rollout import TrajectoryBatch


@dataclass
class AlgorithmConfig:
    """Base configuration for RL algorithms."""

    # Learning rate
    learning_rate: float = 1e-5

    # Discount factor (not typically used for single-turn RL)
    gamma: float = 1.0

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Baseline settings
    use_baseline: bool = True
    baseline_decay: float = 0.99

    # Entropy bonus (encourages exploration)
    entropy_coef: float = 0.0

    # KL penalty to stay close to reference policy
    kl_coef: float = 0.0
    kl_target: float = 0.0

    # Advantage normalization
    normalize_advantages: bool = True

    # Gradient accumulation
    gradient_accumulation_steps: int = 1


@dataclass
class UpdateResult:
    """Result from a policy update step."""

    # Loss values
    policy_loss: float
    entropy_loss: float = 0.0
    kl_loss: float = 0.0
    total_loss: float = 0.0

    # Gradient info
    grad_norm: float = 0.0
    clipped: bool = False

    # Training statistics
    mean_advantage: float = 0.0
    mean_logprob: float = 0.0
    entropy: float = 0.0
    kl_div: float = 0.0

    # Step info
    step: int = 0
    trajectories_used: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "policy_loss": self.policy_loss,
            "entropy_loss": self.entropy_loss,
            "kl_loss": self.kl_loss,
            "total_loss": self.total_loss,
            "grad_norm": self.grad_norm,
            "mean_advantage": self.mean_advantage,
            "mean_logprob": self.mean_logprob,
            "entropy": self.entropy,
            "kl_div": self.kl_div,
        }


class TrainingClient(Protocol):
    """Protocol for training clients used by RL algorithms."""

    def forward_backward(self, batch: Dict[str, Any]) -> Any:
        """Compute forward-backward pass."""
        ...

    def optim_step(self, apply_dp_noise: bool = True) -> Any:
        """Apply optimizer step."""
        ...

    @property
    def step(self) -> int:
        """Current training step."""
        ...


class RLAlgorithm(ABC):
    """
    Abstract base class for RL algorithms.

    All RL algorithms must implement the `update` method which takes
    a batch of trajectories and performs a policy update.
    """

    def __init__(self, config: AlgorithmConfig):
        """
        Initialize the algorithm.

        Args:
            config: Algorithm configuration
        """
        self.config = config
        self._step = 0
        self._baseline = 0.0

    @property
    def step(self) -> int:
        """Current update step."""
        return self._step

    @abstractmethod
    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """
        Perform a policy update step.

        Args:
            batch: Batch of trajectories with computed rewards/advantages
            client: Optional training client for gradient computation

        Returns:
            UpdateResult with update statistics
        """
        pass

    def prepare_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        """
        Prepare a trajectory batch for training.

        This includes computing advantages and optionally normalizing.

        Args:
            batch: Raw trajectory batch with rewards

        Returns:
            Prepared batch with computed advantages
        """
        # Update baseline
        if self.config.use_baseline:
            batch_mean = batch.mean_reward
            self._baseline = (
                self.config.baseline_decay * self._baseline
                + (1 - self.config.baseline_decay) * batch_mean
            )

        # Compute advantages
        batch.compute_advantages(
            baseline=self._baseline if self.config.use_baseline else None,
            normalize=self.config.normalize_advantages,
        )

        return batch

    def get_state(self) -> Dict[str, Any]:
        """
        Get algorithm state for checkpointing.

        Returns:
            Dictionary containing algorithm state
        """
        return {
            "step": self._step,
            "baseline": self._baseline,
            "config": {
                "learning_rate": self.config.learning_rate,
                "gamma": self.config.gamma,
                "max_grad_norm": self.config.max_grad_norm,
                "use_baseline": self.config.use_baseline,
                "baseline_decay": self.config.baseline_decay,
                "entropy_coef": self.config.entropy_coef,
                "kl_coef": self.config.kl_coef,
                "normalize_advantages": self.config.normalize_advantages,
            },
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load algorithm state from checkpoint.

        Args:
            state: State dictionary from get_state()
        """
        self._step = state.get("step", 0)
        self._baseline = state.get("baseline", 0.0)


class MockRLAlgorithm(RLAlgorithm):
    """
    Mock RL algorithm for testing without real training.

    Simulates policy updates with deterministic behavior.
    """

    def __init__(self, config: Optional[AlgorithmConfig] = None):
        super().__init__(config or AlgorithmConfig())
        self._update_history: List[Dict[str, float]] = []

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """Simulate a policy update."""
        # Prepare batch
        batch = self.prepare_batch(batch)

        # Compute mock statistics
        mean_advantage = sum(t.advantage for t in batch) / len(batch)
        mean_logprob = sum(t.mean_logprob for t in batch) / len(batch)

        # Simulate policy loss: -mean(advantage * logprob)
        policy_loss = -mean_advantage * mean_logprob

        # Mock entropy
        entropy = 0.5 - self._step * 0.001  # Decreasing entropy

        # Mock KL divergence
        kl_div = 0.01 * self._step

        # Total loss
        total_loss = (
            policy_loss
            - self.config.entropy_coef * entropy
            + self.config.kl_coef * kl_div
        )

        self._step += 1

        result = UpdateResult(
            policy_loss=policy_loss,
            entropy_loss=-self.config.entropy_coef * entropy,
            kl_loss=self.config.kl_coef * kl_div,
            total_loss=total_loss,
            grad_norm=1.5 + 0.1 * (hash(str(self._step)) % 10) / 10,
            mean_advantage=mean_advantage,
            mean_logprob=mean_logprob,
            entropy=entropy,
            kl_div=kl_div,
            step=self._step,
            trajectories_used=len(batch),
        )

        self._update_history.append(result.to_dict())

        return result
