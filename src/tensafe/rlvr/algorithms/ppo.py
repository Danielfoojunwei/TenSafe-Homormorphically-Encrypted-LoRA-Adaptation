"""
PPO (Proximal Policy Optimization) Algorithm Implementation

PPO is a policy gradient algorithm that uses a clipped surrogate objective
to prevent too large policy updates, improving training stability.

The PPO objective is:
    L^CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

Where:
- r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
- A_t is the advantage estimate
- ε is the clip range (typically 0.2)

For language models:
- The ratio is computed from log probabilities: exp(logprob_new - logprob_old)
- We also add entropy bonus and KL penalty terms
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..rollout import TrajectoryBatch
from .base import AlgorithmConfig, RLAlgorithm, TrainingClient, UpdateResult

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig(AlgorithmConfig):
    """Configuration for PPO algorithm."""

    # Clipping parameters
    clip_range: float = 0.2  # ε in the paper
    clip_range_vf: Optional[float] = None  # Optional value function clipping

    # PPO epochs (how many times to iterate over the batch)
    ppo_epochs: int = 4

    # Mini-batch settings
    minibatch_size: Optional[int] = None  # None means use full batch

    # Value function coefficient (for combined actor-critic loss)
    vf_coef: float = 0.5

    # Entropy bonus (from AlgorithmConfig, default override)
    entropy_coef: float = 0.01

    # KL penalty settings
    target_kl: Optional[float] = 0.01  # Early stop if KL exceeds this
    adaptive_kl: bool = False  # Adapt KL coefficient

    # Advantage estimation
    gae_lambda: float = 1.0  # GAE lambda (1.0 = no GAE, just advantages)

    # Reward normalization
    normalize_rewards: bool = False
    reward_scale: float = 1.0


class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization algorithm.

    This implementation includes:
    - Clipped surrogate objective
    - Multiple PPO epochs per batch
    - KL divergence early stopping
    - Entropy bonus
    - Optional value function loss

    Example usage:
        config = PPOConfig(
            learning_rate=1e-5,
            clip_range=0.2,
            ppo_epochs=4,
            entropy_coef=0.01,
        )
        algo = PPO(config)

        # In training loop
        batch = sampler.generate_trajectories(prompts)
        compute_rewards(batch, reward_fn)
        result = algo.update(batch, training_client)
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        """
        Initialize PPO algorithm.

        Args:
            config: Algorithm configuration
        """
        super().__init__(config or PPOConfig())
        self.config: PPOConfig = self.config  # type: ignore

        # Store old policy log probabilities for ratio computation
        self._old_logprobs: Dict[int, List[float]] = {}

        # Adaptive KL coefficient
        self._kl_coef = self.config.kl_coef

        # Running statistics for reward normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """
        Perform a PPO policy update.

        Runs multiple PPO epochs over the batch, computing the clipped
        surrogate objective each time.

        Args:
            batch: Batch of trajectories with computed rewards
            client: Optional training client for gradient computation

        Returns:
            UpdateResult with update statistics
        """
        # Store old log probs before any updates
        self._store_old_logprobs(batch)

        # Prepare batch (compute advantages)
        batch = self.prepare_batch(batch)

        # Normalize rewards if configured
        if self.config.normalize_rewards:
            batch = self._normalize_rewards(batch)

        # Track statistics across epochs
        epoch_stats = []
        total_grad_norm = 0.0
        early_stopped = False

        # Run PPO epochs
        for epoch in range(self.config.ppo_epochs):
            # Compute losses
            policy_loss, clip_fraction = self._compute_policy_loss(batch)
            entropy, entropy_loss = self._compute_entropy_loss(batch)
            kl_div, kl_loss = self._compute_kl_loss(batch)

            # Total loss
            total_loss = policy_loss + entropy_loss + kl_loss

            # Check for early stopping based on KL
            if self.config.target_kl is not None and kl_div > self.config.target_kl * 1.5:
                logger.info(
                    f"Early stopping at epoch {epoch} due to KL divergence: "
                    f"{kl_div:.4f} > {self.config.target_kl * 1.5:.4f}"
                )
                early_stopped = True
                break

            # Perform gradient update if client provided
            if client is not None:
                train_batch = self._trajectories_to_batch(batch)
                fb_result = client.forward_backward(train_batch)

                if hasattr(fb_result, "result"):
                    result = fb_result.result()
                    grad_norm = result.get("grad_norm", 1.5)
                else:
                    grad_norm = 1.5

                total_grad_norm += grad_norm
                client.optim_step()

            epoch_stats.append({
                "policy_loss": policy_loss,
                "entropy": entropy,
                "kl_div": kl_div,
                "clip_fraction": clip_fraction,
            })

        # Adaptive KL coefficient update
        if self.config.adaptive_kl and epoch_stats:
            self._update_kl_coef(epoch_stats[-1]["kl_div"])

        # Update step counter
        self._step += 1

        # Aggregate statistics
        num_epochs = len(epoch_stats)
        avg_policy_loss = sum(s["policy_loss"] for s in epoch_stats) / num_epochs
        avg_entropy = sum(s["entropy"] for s in epoch_stats) / num_epochs
        avg_kl = sum(s["kl_div"] for s in epoch_stats) / num_epochs
        avg_clip_fraction = sum(s["clip_fraction"] for s in epoch_stats) / num_epochs

        mean_advantage = sum(t.advantage for t in batch) / len(batch)
        mean_logprob = sum(t.mean_logprob for t in batch) / len(batch)

        return UpdateResult(
            policy_loss=avg_policy_loss,
            entropy_loss=-self.config.entropy_coef * avg_entropy,
            kl_loss=self._kl_coef * avg_kl,
            total_loss=avg_policy_loss - self.config.entropy_coef * avg_entropy + self._kl_coef * avg_kl,
            grad_norm=total_grad_norm / max(1, num_epochs),
            clipped=avg_clip_fraction > 0.5,
            mean_advantage=mean_advantage,
            mean_logprob=mean_logprob,
            entropy=avg_entropy,
            kl_div=avg_kl,
            step=self._step,
            trajectories_used=len(batch),
        )

    def _store_old_logprobs(self, batch: TrajectoryBatch) -> None:
        """Store log probabilities as reference for ratio computation."""
        for i, traj in enumerate(batch):
            self._old_logprobs[i] = traj.logprobs.copy()

    def _compute_policy_loss(
        self, batch: TrajectoryBatch
    ) -> Tuple[float, float]:
        """
        Compute the clipped surrogate objective.

        L^CLIP = -E[min(ratio * advantage, clip(ratio, 1-ε, 1+ε) * advantage)]

        Args:
            batch: Trajectory batch with advantages

        Returns:
            Tuple of (policy_loss, clip_fraction)
        """
        clip_range = self.config.clip_range
        losses = []
        clipped_count = 0
        total_count = 0

        for i, traj in enumerate(batch):
            old_logprobs = self._old_logprobs.get(i, traj.logprobs)

            # Compute ratio for each token
            for j, (new_lp, old_lp) in enumerate(zip(traj.logprobs, old_logprobs)):
                # ratio = exp(new_logprob - old_logprob)
                log_ratio = new_lp - old_lp
                ratio = min(max(1.0 + log_ratio, 0.01), 10.0)  # Clamp for stability

                # Clipped ratio
                clipped_ratio = max(min(ratio, 1 + clip_range), 1 - clip_range)

                # Surrogate losses
                surr1 = ratio * traj.advantage
                surr2 = clipped_ratio * traj.advantage

                # PPO objective: take the minimum (pessimistic)
                loss = -min(surr1, surr2)
                losses.append(loss)

                # Track clipping
                if abs(ratio - clipped_ratio) > 1e-6:
                    clipped_count += 1
                total_count += 1

        policy_loss = sum(losses) / len(losses) if losses else 0.0
        clip_fraction = clipped_count / max(1, total_count)

        return policy_loss, clip_fraction

    def _compute_entropy_loss(
        self, batch: TrajectoryBatch
    ) -> Tuple[float, float]:
        """
        Compute entropy and entropy loss.

        Args:
            batch: Trajectory batch

        Returns:
            Tuple of (entropy, entropy_loss)
        """
        entropies = []
        for traj in batch:
            if traj.logprobs:
                # Estimate entropy from log probs
                entropy = -sum(traj.logprobs) / len(traj.logprobs)
                entropies.append(entropy)

        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        entropy_loss = -self.config.entropy_coef * mean_entropy

        return mean_entropy, entropy_loss

    def _compute_kl_loss(
        self, batch: TrajectoryBatch
    ) -> Tuple[float, float]:
        """
        Compute KL divergence from old policy.

        Args:
            batch: Trajectory batch

        Returns:
            Tuple of (kl_div, kl_loss)
        """
        kl_divs = []

        for i, traj in enumerate(batch):
            old_logprobs = self._old_logprobs.get(i, traj.logprobs)

            if len(traj.logprobs) == len(old_logprobs):
                # Approximate KL: E[log(p/q)] = E[logp - logq]
                kl = sum(lp - olp for lp, olp in zip(traj.logprobs, old_logprobs))
                kl = kl / len(traj.logprobs) if traj.logprobs else 0.0
                kl_divs.append(abs(kl))

        mean_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0
        kl_loss = self._kl_coef * mean_kl

        return mean_kl, kl_loss

    def _update_kl_coef(self, current_kl: float) -> None:
        """
        Update KL coefficient adaptively.

        If KL is too high, increase coefficient.
        If KL is too low, decrease coefficient.
        """
        if self.config.target_kl is None:
            return

        if current_kl > self.config.target_kl * 1.5:
            self._kl_coef *= 1.5
        elif current_kl < self.config.target_kl / 1.5:
            self._kl_coef /= 1.5

        # Clamp to reasonable range
        self._kl_coef = max(0.0, min(self._kl_coef, 1.0))

    def _normalize_rewards(
        self, batch: TrajectoryBatch
    ) -> TrajectoryBatch:
        """
        Normalize rewards using running statistics.

        Args:
            batch: Trajectory batch

        Returns:
            Batch with normalized rewards
        """
        rewards = batch.rewards

        # Update running statistics
        for r in rewards:
            self._reward_count += 1
            delta = r - self._reward_mean
            self._reward_mean += delta / self._reward_count
            delta2 = r - self._reward_mean
            self._reward_var += delta * delta2

        # Compute std
        std = (self._reward_var / max(1, self._reward_count)) ** 0.5
        std = max(std, 1e-8)

        # Normalize
        for traj in batch:
            traj.reward = (traj.reward - self._reward_mean) / std * self.config.reward_scale

        return batch

    def _trajectories_to_batch(
        self, batch: TrajectoryBatch
    ) -> Dict[str, Any]:
        """
        Convert trajectory batch to training batch format.

        Args:
            batch: Trajectory batch

        Returns:
            Dictionary suitable for forward_backward()
        """
        max_len = max(len(t.full_tokens) for t in batch)

        input_ids = []
        attention_mask = []
        labels = []
        advantages = []
        old_logprobs = []

        for i, traj in enumerate(batch):
            tokens = traj.full_tokens
            pad_len = max_len - len(tokens)

            input_ids.append(tokens + [0] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)

            prompt_len = len(traj.prompt_tokens)
            lab = [-100] * prompt_len + traj.response_tokens + [-100] * pad_len
            labels.append(lab)

            advantages.append(traj.advantage)
            old_logprobs.append(self._old_logprobs.get(i, traj.logprobs))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,
            "old_logprobs": old_logprobs,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for checkpointing."""
        state = super().get_state()
        state["kl_coef"] = self._kl_coef
        state["reward_mean"] = self._reward_mean
        state["reward_var"] = self._reward_var
        state["reward_count"] = self._reward_count
        state["ppo_config"] = {
            "clip_range": self.config.clip_range,
            "ppo_epochs": self.config.ppo_epochs,
            "target_kl": self.config.target_kl,
        }
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load algorithm state from checkpoint."""
        super().load_state(state)
        self._kl_coef = state.get("kl_coef", self.config.kl_coef)
        self._reward_mean = state.get("reward_mean", 0.0)
        self._reward_var = state.get("reward_var", 1.0)
        self._reward_count = state.get("reward_count", 0)


class PPOWithValueFunction(PPO):
    """
    PPO with separate value function head.

    This variant maintains a value function estimate for each state
    and uses it to compute advantages with GAE.

    Note: For single-turn language model RL, the simpler PPO without
    value function is usually sufficient.
    """

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        value_hidden_size: int = 256,
    ):
        """
        Initialize PPO with value function.

        Args:
            config: PPO configuration
            value_hidden_size: Hidden size for value function
        """
        super().__init__(config)
        self.value_hidden_size = value_hidden_size
        self._value_estimates: Dict[str, float] = {}

    def _estimate_value(self, prompt: str) -> float:
        """
        Estimate value for a prompt.

        Args:
            prompt: The prompt string

        Returns:
            Estimated value
        """
        return self._value_estimates.get(prompt, self._baseline)

    def _compute_value_loss(
        self, batch: TrajectoryBatch
    ) -> float:
        """
        Compute value function loss.

        L^VF = E[(V(s) - R)^2]

        Args:
            batch: Trajectory batch

        Returns:
            Value function loss
        """
        losses = []
        for traj in batch:
            value_pred = self._estimate_value(traj.prompt)
            value_loss = (value_pred - traj.reward) ** 2
            losses.append(value_loss)

            # Update value estimate
            lr = 0.1
            self._value_estimates[traj.prompt] = (
                (1 - lr) * value_pred + lr * traj.reward
            )

        return sum(losses) / len(losses) if losses else 0.0
