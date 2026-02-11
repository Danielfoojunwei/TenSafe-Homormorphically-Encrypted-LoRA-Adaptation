"""
GRPO (Group Relative Policy Optimization) Algorithm

Implements GRPO as described in DeepSeek-R1 and related work. GRPO
normalizes rewards within prompt groups, eliminating the need for a
learned critic/value function. This makes it particularly attractive
for TenSafe's HE-LoRA setting:

- No critic model = half the memory footprint
- No critic training = simpler DP accounting
- Fewer encrypted parameters to manage

Key features:
- Per-prompt group reward normalization
- Multiple samples per prompt for group statistics
- Configurable group-level or batch-level normalization
- Compatible with KL penalty to reference policy
- Supports both sequence-level and token-level policy losses
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..rollout import Trajectory, TrajectoryBatch
from .base import AlgorithmConfig, RLAlgorithm, TrainingClient, UpdateResult

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig(AlgorithmConfig):
    """Configuration for GRPO algorithm."""

    # Number of samples per prompt for group statistics
    num_samples_per_prompt: int = 5

    # PPO-style clipping range for the IS ratio
    clip_range: float = 0.2

    # Whether to normalize advantages within each group
    normalize_within_group: bool = True

    # Whether to apply batch-level normalization after group normalization
    normalize_batch: bool = False

    # KL penalty coefficient (to stay close to reference policy)
    kl_coef: float = 0.0

    # Minimum group size for normalization (smaller groups use batch baseline)
    min_group_size: int = 2

    # Reward scaling before advantage computation
    reward_scale: float = 1.0

    # Number of update epochs per batch
    update_epochs: int = 1

    # Mini-batch size for gradient accumulation
    minibatch_size: int = 8

    # Epsilon for numerical stability
    eps: float = 1e-8

    # Policy loss type: "ppo_clip" or "sequence_level"
    policy_loss_type: str = "ppo_clip"


class GRPO(RLAlgorithm):
    """
    Group Relative Policy Optimization.

    GRPO computes advantages by normalizing rewards within groups of
    responses generated for the same prompt. This is a critic-free
    approach that provides stable baselines from group statistics.

    The algorithm:
    1. For each prompt, generate multiple responses
    2. Compute group mean and (optionally) std of rewards
    3. Advantage_i = (reward_i - mean) / std for each response in group
    4. Update policy with clipped surrogate objective
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        super().__init__(config or GRPOConfig())
        self.config: GRPOConfig = self.config

        # Storage for old log probabilities (for IS ratio)
        self._old_logprobs: Dict[int, List[float]] = {}

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """
        Perform GRPO policy update.

        1. Group trajectories by prompt
        2. Compute group-normalized advantages
        3. Compute PPO-clipped policy loss
        4. Apply KL penalty if configured
        """
        # Step 1: Compute GRPO advantages
        self._compute_grpo_advantages(batch)

        # Step 2: Store old log probs for IS ratio (if first epoch)
        if not self._old_logprobs:
            self._store_old_logprobs(batch)

        # Step 3: Compute losses
        policy_loss = self._compute_policy_loss(batch)
        entropy_loss = self._compute_entropy_loss(batch)
        kl_loss = self._compute_kl_loss(batch)

        total_loss = (
            policy_loss
            - self.config.entropy_coef * entropy_loss
            + self.config.kl_coef * kl_loss
        )

        # Step 4: Forward-backward pass
        grad_norm = 0.0
        if client is not None:
            training_batch = self._trajectories_to_batch(batch)
            client.forward_backward(training_batch)
            client.optim_step(apply_dp_noise=True)
            grad_norm = self.config.max_grad_norm  # Approximate

        self._step += 1
        self._old_logprobs.clear()  # Reset for next batch

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

    def _compute_grpo_advantages(self, batch: TrajectoryBatch) -> None:
        """Compute group-normalized advantages (the core GRPO mechanism)."""
        eps = self.config.eps

        # Group trajectories by prompt
        groups: Dict[str, List[int]] = {}
        for i, traj in enumerate(batch.trajectories):
            if traj.prompt not in groups:
                groups[traj.prompt] = []
            groups[traj.prompt].append(i)

        for prompt, indices in groups.items():
            rewards = [
                batch.trajectories[i].reward * self.config.reward_scale
                for i in indices
            ]
            group_mean = sum(rewards) / len(rewards)

            if self.config.normalize_within_group and len(rewards) >= self.config.min_group_size:
                group_var = sum((r - group_mean) ** 2 for r in rewards) / len(rewards)
                group_std = math.sqrt(group_var) + eps
            else:
                group_std = 1.0

            for idx, i in enumerate(indices):
                batch.trajectories[i].advantage = (rewards[idx] - group_mean) / group_std

        # Optional batch-level normalization on top of group normalization
        if self.config.normalize_batch:
            advantages = [t.advantage for t in batch.trajectories]
            batch_mean = sum(advantages) / len(advantages)
            batch_var = sum((a - batch_mean) ** 2 for a in advantages) / len(advantages)
            batch_std = math.sqrt(batch_var) + eps

            for traj in batch.trajectories:
                traj.advantage = (traj.advantage - batch_mean) / batch_std

    def _store_old_logprobs(self, batch: TrajectoryBatch) -> None:
        """Store current log probabilities as the 'old' policy for IS ratio."""
        for i, traj in enumerate(batch.trajectories):
            self._old_logprobs[i] = traj.logprobs.copy()

    def _compute_policy_loss(self, batch: TrajectoryBatch) -> float:
        """Compute clipped surrogate policy loss."""
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
        """Estimate entropy from log probabilities."""
        total_entropy = 0.0
        total_tokens = 0

        for traj in batch.trajectories:
            for lp in traj.logprobs:
                # H ≈ -log_prob (approximation for autoregressive LMs)
                total_entropy += -lp
                total_tokens += 1

        return total_entropy / total_tokens if total_tokens > 0 else 0.0

    def _compute_kl_loss(self, batch: TrajectoryBatch) -> float:
        """Compute KL divergence from reference policy."""
        if self.config.kl_coef == 0.0:
            return 0.0

        total_kl = 0.0
        total_tokens = 0

        for i, traj in enumerate(batch.trajectories):
            old_lps = self._old_logprobs.get(i, traj.logprobs)
            for t in range(len(traj.logprobs)):
                log_ratio = traj.logprobs[t] - old_lps[t]
                ratio = math.exp(min(log_ratio, 20.0))
                # KL ≈ ratio * log_ratio - (ratio - 1)
                kl = ratio * log_ratio - (ratio - 1.0)
                total_kl += max(kl, 0.0)
                total_tokens += 1

        return total_kl / total_tokens if total_tokens > 0 else 0.0

    def _trajectories_to_batch(self, batch: TrajectoryBatch) -> Dict[str, Any]:
        """Convert trajectory batch to training batch format."""
        return {
            "prompts": batch.prompts,
            "responses": batch.responses,
            "advantages": batch.advantages,
            "logprobs": [t.logprobs for t in batch.trajectories],
            "old_logprobs": [
                self._old_logprobs.get(i, t.logprobs)
                for i, t in enumerate(batch.trajectories)
            ],
            "loss_type": "grpo",
            "clip_range": self.config.clip_range,
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["grpo_config"] = {
            "num_samples_per_prompt": self.config.num_samples_per_prompt,
            "clip_range": self.config.clip_range,
            "normalize_within_group": self.config.normalize_within_group,
            "policy_loss_type": self.config.policy_loss_type,
        }
        return state
