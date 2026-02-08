"""
REINFORCE++ Algorithm

An enhanced version of REINFORCE that incorporates temporal discounting
and per-token advantage whitening. Instead of a single scalar advantage
for the entire trajectory, REINFORCE++ distributes rewards across tokens
with exponential discounting and normalizes per-token advantages.

Key improvements over vanilla REINFORCE:
- Temporal discounting distributes credit across tokens
- Per-token whitening reduces variance of gradient estimates
- Backward accumulation captures future reward contribution
- Compatible with KL penalty and entropy bonus
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
class REINFORCEPPConfig(AlgorithmConfig):
    """Configuration for REINFORCE++ algorithm."""

    # Temporal discount factor for per-token reward distribution
    gamma: float = 0.99

    # PPO-style clip range (optional, 0 = no clipping)
    clip_range: float = 0.0

    # Whether to apply per-token temporal whitening
    temporal_whitening: bool = True

    # Reward scaling
    reward_scale: float = 1.0

    # Mini-batch size
    minibatch_size: int = 8

    # Epsilon for numerical stability
    eps: float = 1e-8


class REINFORCEPP(RLAlgorithm):
    """
    REINFORCE++ with temporal discounting and whitening.

    For each trajectory:
    1. Distribute the reward to the last response token
    2. Accumulate discounted returns backward: G_t = r_t + gamma * G_{t+1}
    3. Apply temporal whitening: normalize per-token returns
    4. Compute policy gradient with per-token advantages

    This gives more informative gradients than vanilla REINFORCE, which
    applies the same scalar advantage to all tokens.
    """

    def __init__(self, config: Optional[REINFORCEPPConfig] = None):
        super().__init__(config or REINFORCEPPConfig())
        self.config: REINFORCEPPConfig = self.config
        self._old_logprobs: Dict[int, List[float]] = {}

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """Perform REINFORCE++ policy update."""
        # Compute per-token advantages with temporal discounting
        per_token_advantages = self._compute_temporal_advantages(batch)

        # Store old log probs for IS ratio (if clip_range > 0)
        if self.config.clip_range > 0 and not self._old_logprobs:
            for i, traj in enumerate(batch.trajectories):
                self._old_logprobs[i] = traj.logprobs.copy()

        # Compute losses using per-token advantages
        policy_loss = self._compute_policy_loss(batch, per_token_advantages)
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
            training_batch = self._trajectories_to_batch(batch, per_token_advantages)
            client.forward_backward(training_batch)
            client.optim_step(apply_dp_noise=True)
            grad_norm = self.config.max_grad_norm

        self._step += 1
        self._old_logprobs.clear()

        # Set trajectory-level advantage as mean of per-token advantages
        for i, traj in enumerate(batch.trajectories):
            token_advs = per_token_advantages[i]
            traj.advantage = (
                sum(token_advs) / len(token_advs) if token_advs else 0.0
            )

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

    def _compute_temporal_advantages(
        self, batch: TrajectoryBatch
    ) -> List[List[float]]:
        """
        Compute per-token advantages using temporal discounting and whitening.

        For each trajectory:
        1. Place the reward at the last token
        2. Backward accumulate: G_t = r_t + gamma * G_{t+1}
        3. Optionally apply temporal whitening (normalize per-token)
        """
        eps = self.config.eps
        gamma = self.config.gamma
        all_advantages = []

        for traj in batch.trajectories:
            n_tokens = traj.num_response_tokens
            if n_tokens == 0:
                all_advantages.append([])
                continue

            # Step 1: Distribute reward to last token
            per_token_rewards = [0.0] * n_tokens
            per_token_rewards[-1] = traj.reward * self.config.reward_scale

            # Step 2: Backward accumulation of discounted returns
            returns = [0.0] * n_tokens
            running_return = 0.0
            for t in range(n_tokens - 1, -1, -1):
                running_return = per_token_rewards[t] + gamma * running_return
                returns[t] = running_return

            # Step 3: Temporal whitening
            if self.config.temporal_whitening and n_tokens > 1:
                mean_ret = sum(returns) / n_tokens
                var_ret = sum((r - mean_ret) ** 2 for r in returns) / n_tokens
                std_ret = math.sqrt(var_ret) + eps
                whitened = [(r - mean_ret) / std_ret for r in returns]
            else:
                whitened = returns

            all_advantages.append(whitened)

        return all_advantages

    def _compute_policy_loss(
        self,
        batch: TrajectoryBatch,
        per_token_advantages: List[List[float]],
    ) -> float:
        """Compute policy loss with per-token advantages."""
        total_loss = 0.0
        total_tokens = 0

        for i, traj in enumerate(batch.trajectories):
            token_advs = per_token_advantages[i]
            old_lps = self._old_logprobs.get(i, traj.logprobs)

            for t in range(len(traj.logprobs)):
                if t >= len(token_advs):
                    break

                advantage = token_advs[t]

                if self.config.clip_range > 0:
                    # PPO-style clipping
                    log_ratio = traj.logprobs[t] - old_lps[t]
                    ratio = math.exp(min(log_ratio, 20.0))

                    surr1 = ratio * advantage
                    clipped_ratio = max(
                        min(ratio, 1.0 + self.config.clip_range),
                        1.0 - self.config.clip_range,
                    )
                    surr2 = clipped_ratio * advantage
                    total_loss += -min(surr1, surr2)
                else:
                    # Vanilla policy gradient
                    total_loss += -traj.logprobs[t] * advantage

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

    def _trajectories_to_batch(
        self,
        batch: TrajectoryBatch,
        per_token_advantages: List[List[float]],
    ) -> Dict[str, Any]:
        return {
            "prompts": batch.prompts,
            "responses": batch.responses,
            "advantages": batch.advantages,
            "token_advantages": per_token_advantages,
            "logprobs": [t.logprobs for t in batch.trajectories],
            "old_logprobs": [
                self._old_logprobs.get(i, t.logprobs)
                for i, t in enumerate(batch.trajectories)
            ],
            "loss_type": "reinforce_pp",
            "gamma": self.config.gamma,
        }

    def get_state(self) -> Dict[str, Any]:
        state = super().get_state()
        state["reinforce_pp_config"] = {
            "gamma": self.config.gamma,
            "clip_range": self.config.clip_range,
            "temporal_whitening": self.config.temporal_whitening,
        }
        return state
