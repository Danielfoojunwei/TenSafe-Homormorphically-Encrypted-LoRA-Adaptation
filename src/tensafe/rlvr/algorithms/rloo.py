"""
RLOO (Reinforcement Learning with Leave-One-Out) Algorithm

Implements leave-one-out baseline estimation for policy gradient.
For each trajectory in a group, the baseline is the mean reward of
all OTHER trajectories in the same group. This provides a lower-variance
baseline than the full group mean (which includes the trajectory itself).

Key properties:
- Unbiased advantage estimation
- Lower variance than batch-mean baseline
- No learned value function required (critic-free)
- Naturally handles per-prompt normalization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import AlgorithmConfig, RLAlgorithm, TrainingClient, UpdateResult
from ..rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class RLOOConfig(AlgorithmConfig):
    """Configuration for RLOO algorithm."""

    # PPO-style clipping range
    clip_range: float = 0.2

    # Number of update epochs per batch
    update_epochs: int = 1

    # Mini-batch size
    minibatch_size: int = 8

    # Reward scaling
    reward_scale: float = 1.0

    # Epsilon for numerical stability
    eps: float = 1e-8

    # Fallback baseline for single-sample groups
    fallback_to_batch_mean: bool = True


class RLOO(RLAlgorithm):
    """
    Reinforcement Learning with Leave-One-Out baseline.

    For each trajectory_i in a group of responses to the same prompt:
        baseline_i = mean(rewards of all other trajectories in group)
        advantage_i = reward_i - baseline_i

    When a group has only one trajectory, falls back to the batch-level
    mean reward as the baseline.
    """

    def __init__(self, config: Optional[RLOOConfig] = None):
        super().__init__(config or RLOOConfig())
        self.config: RLOOConfig = self.config
        self._old_logprobs: Dict[int, List[float]] = {}

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """Perform RLOO policy update."""
        # Compute LOO advantages
        self._compute_rloo_advantages(batch)

        # Store old log probs
        if not self._old_logprobs:
            self._store_old_logprobs(batch)

        # Compute losses
        policy_loss = self._compute_policy_loss(batch)
        entropy_loss = self._compute_entropy_loss(batch)
        kl_loss = self._compute_kl_loss(batch)

        total_loss = (
            policy_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.kl_coef * kl_loss
        )

        # Forward-backward
        grad_norm = 0.0
        if client is not None:
            training_batch = self._trajectories_to_batch(batch)
            client.forward_backward(training_batch)
            client.optim_step(apply_dp_noise=True)
            grad_norm = self.config.max_grad_norm

        self._step += 1
        self._old_logprobs.clear()

        mean_adv = sum(t.advantage for t in batch) / len(batch)
        mean_lp = sum(t.mean_logprob for t in batch) / len(batch)

        return UpdateResult(
            policy_loss=policy_loss,
            entropy_loss=-self.config.entropy_coef * entropy_loss,
            kl_loss=self.config.kl_coef * kl_loss,
            total_loss=total_loss,
            grad_norm=grad_norm,
            mean_advantage=mean_adv,
            mean_logprob=mean_lp,
            entropy=entropy_loss,
            kl_div=kl_loss,
            step=self._step,
            trajectories_used=len(batch),
        )

    def _compute_rloo_advantages(self, batch: TrajectoryBatch) -> None:
        """Compute leave-one-out advantages."""
        # Group by prompt
        groups: Dict[str, List[int]] = {}
        for i, traj in enumerate(batch.trajectories):
            if traj.prompt not in groups:
                groups[traj.prompt] = []
            groups[traj.prompt].append(i)

        batch_mean = batch.mean_reward

        for prompt, indices in groups.items():
            rewards = [
                batch.trajectories[i].reward * self.config.reward_scale
                for i in indices
            ]
            n = len(rewards)
            group_sum = sum(rewards)

            if n <= 1 and self.config.fallback_to_batch_mean:
                # Single sample: use batch mean as baseline
                for idx, i in enumerate(indices):
                    batch.trajectories[i].advantage = rewards[idx] - batch_mean
            else:
                # Leave-one-out baseline
                for idx, i in enumerate(indices):
                    if n > 1:
                        loo_baseline = (group_sum - rewards[idx]) / (n - 1)
                    else:
                        loo_baseline = 0.0
                    batch.trajectories[i].advantage = rewards[idx] - loo_baseline

        # Normalize across batch
        if self.config.normalize_advantages:
            advantages = [t.advantage for t in batch.trajectories]
            if len(advantages) > 1:
                mean_a = sum(advantages) / len(advantages)
                var_a = sum((a - mean_a) ** 2 for a in advantages) / len(advantages)
                std_a = math.sqrt(var_a) + self.config.eps
                for traj in batch.trajectories:
                    traj.advantage = (traj.advantage - mean_a) / std_a

    def _store_old_logprobs(self, batch: TrajectoryBatch) -> None:
        for i, traj in enumerate(batch.trajectories):
            self._old_logprobs[i] = traj.logprobs.copy()

    def _compute_policy_loss(self, batch: TrajectoryBatch) -> float:
        total_loss = 0.0
        total_tokens = 0

        for i, traj in enumerate(batch.trajectories):
            old_lps = self._old_logprobs.get(i, traj.logprobs)
            for t in range(len(traj.logprobs)):
                log_ratio = traj.logprobs[t] - old_lps[t]
                ratio = math.exp(min(log_ratio, 20.0))

                surr1 = ratio * traj.advantage
                clipped_ratio = max(
                    min(ratio, 1.0 + self.config.clip_range),
                    1.0 - self.config.clip_range,
                )
                surr2 = clipped_ratio * traj.advantage

                total_loss += -min(surr1, surr2)
                total_tokens += 1

        return total_loss / total_tokens if total_tokens > 0 else 0.0

    def _compute_entropy_loss(self, batch: TrajectoryBatch) -> float:
        total_entropy = 0.0
        total_tokens = 0
        for traj in batch.trajectories:
            for lp in traj.logprobs:
                total_entropy += -lp
                total_tokens += 1
        return total_entropy / total_tokens if total_tokens > 0 else 0.0

    def _compute_kl_loss(self, batch: TrajectoryBatch) -> float:
        if self.config.kl_coef == 0.0:
            return 0.0

        total_kl = 0.0
        total_tokens = 0
        for i, traj in enumerate(batch.trajectories):
            old_lps = self._old_logprobs.get(i, traj.logprobs)
            for t in range(len(traj.logprobs)):
                log_ratio = traj.logprobs[t] - old_lps[t]
                ratio = math.exp(min(log_ratio, 20.0))
                kl = ratio * log_ratio - (ratio - 1.0)
                total_kl += max(kl, 0.0)
                total_tokens += 1
        return total_kl / total_tokens if total_tokens > 0 else 0.0

    def _trajectories_to_batch(self, batch: TrajectoryBatch) -> Dict[str, Any]:
        return {
            "prompts": batch.prompts,
            "responses": batch.responses,
            "advantages": batch.advantages,
            "logprobs": [t.logprobs for t in batch.trajectories],
            "old_logprobs": [
                self._old_logprobs.get(i, t.logprobs)
                for i, t in enumerate(batch.trajectories)
            ],
            "loss_type": "rloo",
            "clip_range": self.config.clip_range,
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["rloo_config"] = {
            "clip_range": self.config.clip_range,
            "fallback_to_batch_mean": self.config.fallback_to_batch_mean,
        }
        return state
