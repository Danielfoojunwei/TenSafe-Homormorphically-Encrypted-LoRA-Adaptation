"""
REINFORCE Algorithm Implementation

REINFORCE is a basic policy gradient algorithm that estimates the gradient
of the expected reward with respect to policy parameters.

The policy gradient is:
    ∇J(θ) = E[∑_t ∇log π(a_t|s_t; θ) * (R_t - b)]

Where:
- π is the policy (the language model)
- R_t is the reward (or return)
- b is a baseline (typically the average reward)

For language models:
- Actions are tokens
- States are the sequence of previous tokens
- The policy gradient becomes: ∇log P(response|prompt; θ) * advantage
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import AlgorithmConfig, RLAlgorithm, TrainingClient, UpdateResult
from ..rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class REINFORCEConfig(AlgorithmConfig):
    """Configuration for REINFORCE algorithm."""

    # Variance reduction
    use_baseline: bool = True
    baseline_decay: float = 0.99

    # Entropy bonus to encourage exploration
    entropy_coef: float = 0.01

    # KL penalty to reference policy (optional)
    kl_coef: float = 0.0

    # Reward scaling
    reward_scale: float = 1.0

    # Mini-batch settings
    minibatch_size: Optional[int] = None  # None means use full batch


class REINFORCE(RLAlgorithm):
    """
    REINFORCE policy gradient algorithm.

    This implementation includes:
    - Baseline subtraction for variance reduction
    - Advantage normalization
    - Optional entropy bonus
    - Optional KL penalty to reference policy

    Example usage:
        config = REINFORCEConfig(
            learning_rate=1e-5,
            entropy_coef=0.01,
            use_baseline=True,
        )
        algo = REINFORCE(config)

        # In training loop
        batch = sampler.generate_trajectories(prompts)
        compute_rewards(batch, reward_fn)
        result = algo.update(batch, training_client)
    """

    def __init__(self, config: Optional[REINFORCEConfig] = None):
        """
        Initialize REINFORCE algorithm.

        Args:
            config: Algorithm configuration
        """
        super().__init__(config or REINFORCEConfig())
        self.config: REINFORCEConfig = self.config  # type: ignore

        # Reference policy logprobs (for KL penalty)
        self._ref_logprobs: Dict[int, List[float]] = {}

    def update(
        self,
        batch: TrajectoryBatch,
        client: Optional[TrainingClient] = None,
    ) -> UpdateResult:
        """
        Perform a REINFORCE policy update.

        The update computes:
            loss = -mean(advantage * sum(logprobs))
            loss += entropy_coef * entropy_loss
            loss += kl_coef * kl_loss

        Args:
            batch: Batch of trajectories with computed rewards
            client: Optional training client for gradient computation

        Returns:
            UpdateResult with update statistics
        """
        # Prepare batch (compute advantages)
        batch = self.prepare_batch(batch)

        # Compute policy gradient loss
        policy_loss = self._compute_policy_loss(batch)

        # Compute entropy loss
        entropy, entropy_loss = self._compute_entropy_loss(batch)

        # Compute KL loss (if using KL penalty)
        kl_div, kl_loss = self._compute_kl_loss(batch)

        # Total loss
        total_loss = policy_loss + entropy_loss + kl_loss

        # Compute gradient statistics
        grad_norm = 0.0
        clipped = False

        # If we have a training client, perform actual backward pass
        if client is not None:
            # Create a training batch from trajectories
            train_batch = self._trajectories_to_batch(batch)

            # Forward-backward
            fb_result = client.forward_backward(train_batch)

            # Get result
            if hasattr(fb_result, "result"):
                result = fb_result.result()
                grad_norm = result.get("grad_norm", result.grad_norm if hasattr(result, "grad_norm") else 0.0)
            else:
                grad_norm = 1.5  # Default for mock

            # Optim step
            client.optim_step()

            # Check if gradient was clipped
            clipped = grad_norm > self.config.max_grad_norm

        # Update step counter
        self._step += 1

        # Compute statistics
        mean_advantage = sum(t.advantage for t in batch) / len(batch)
        mean_logprob = sum(t.mean_logprob for t in batch) / len(batch)

        return UpdateResult(
            policy_loss=policy_loss,
            entropy_loss=entropy_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
            grad_norm=grad_norm,
            clipped=clipped,
            mean_advantage=mean_advantage,
            mean_logprob=mean_logprob,
            entropy=entropy,
            kl_div=kl_div,
            step=self._step,
            trajectories_used=len(batch),
        )

    def _compute_policy_loss(self, batch: TrajectoryBatch) -> float:
        """
        Compute the policy gradient loss.

        loss = -mean(advantage * sum(logprobs))

        Args:
            batch: Trajectory batch with advantages

        Returns:
            Scalar policy loss
        """
        # REINFORCE loss: -E[advantage * log_prob]
        losses = []
        for traj in batch:
            # Sum of log probs for the response
            total_logprob = traj.total_logprob
            # Scaled by advantage
            loss = -traj.advantage * total_logprob * self.config.reward_scale
            losses.append(loss)

        return sum(losses) / len(losses) if losses else 0.0

    def _compute_entropy_loss(self, batch: TrajectoryBatch) -> tuple[float, float]:
        """
        Compute entropy and entropy loss.

        Higher entropy = more diverse/uncertain predictions.
        We typically want to maximize entropy (minimize negative entropy).

        Args:
            batch: Trajectory batch

        Returns:
            Tuple of (entropy, entropy_loss)
        """
        # Estimate entropy from log probs
        # entropy ≈ -mean(logprob)
        entropies = []
        for traj in batch:
            if traj.logprobs:
                # Approximate entropy from mean logprob
                # Higher (less negative) logprobs = lower entropy
                entropy = -sum(traj.logprobs) / len(traj.logprobs)
                entropies.append(entropy)

        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        # Entropy loss: we want to maximize entropy, so loss is negative entropy
        entropy_loss = -self.config.entropy_coef * mean_entropy

        return mean_entropy, entropy_loss

    def _compute_kl_loss(self, batch: TrajectoryBatch) -> tuple[float, float]:
        """
        Compute KL divergence from reference policy.

        KL(π_θ || π_ref) ≈ mean(logprob_θ - logprob_ref)

        Args:
            batch: Trajectory batch

        Returns:
            Tuple of (kl_div, kl_loss)
        """
        if self.config.kl_coef == 0.0:
            return 0.0, 0.0

        # For now, approximate KL as deviation from initial policy
        # In practice, you'd store reference logprobs at start of training
        kl_divs = []
        for traj in batch:
            idx = hash(traj.prompt) % 1000000
            ref_logprobs = self._ref_logprobs.get(idx, traj.logprobs)

            # KL divergence: mean difference in logprobs
            if len(traj.logprobs) == len(ref_logprobs):
                kl = sum(lp - rlp for lp, rlp in zip(traj.logprobs, ref_logprobs))
                kl = kl / len(traj.logprobs) if traj.logprobs else 0.0
                kl_divs.append(abs(kl))

        mean_kl = sum(kl_divs) / len(kl_divs) if kl_divs else 0.0
        kl_loss = self.config.kl_coef * mean_kl

        return mean_kl, kl_loss

    def _trajectories_to_batch(self, batch: TrajectoryBatch) -> Dict[str, Any]:
        """
        Convert trajectory batch to training batch format.

        Args:
            batch: Trajectory batch

        Returns:
            Dictionary suitable for forward_backward()
        """
        # Create input_ids, attention_mask, and labels
        max_len = max(len(t.full_tokens) for t in batch)

        input_ids = []
        attention_mask = []
        labels = []
        advantages = []

        for traj in batch:
            tokens = traj.full_tokens
            pad_len = max_len - len(tokens)

            # Pad sequences
            input_ids.append(tokens + [0] * pad_len)
            attention_mask.append([1] * len(tokens) + [0] * pad_len)

            # Labels: -100 for prompt tokens (don't compute loss), actual for response
            prompt_len = len(traj.prompt_tokens)
            lab = [-100] * prompt_len + traj.response_tokens + [-100] * pad_len
            labels.append(lab)

            # Store advantage for loss weighting
            advantages.append(traj.advantage)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "advantages": advantages,  # Custom field for RL
        }

    def store_reference_logprobs(self, batch: TrajectoryBatch) -> None:
        """
        Store log probabilities as reference (for KL penalty).

        Call this at the start of training to establish reference policy.

        Args:
            batch: Initial trajectory batch
        """
        for traj in batch:
            idx = hash(traj.prompt) % 1000000
            self._ref_logprobs[idx] = traj.logprobs.copy()

    def get_state(self) -> Dict[str, Any]:
        """Get algorithm state for checkpointing."""
        state = super().get_state()
        state["ref_logprobs_count"] = len(self._ref_logprobs)
        # Note: ref_logprobs could be large, consider not saving
        return state


class REINFORCEWithBaseline(REINFORCE):
    """
    REINFORCE with learned value baseline.

    This variant uses a value function to estimate the baseline,
    which can provide better variance reduction than the simple
    moving average baseline.

    Note: This is a more advanced variant typically not needed
    for single-turn RL on language models.
    """

    def __init__(
        self,
        config: Optional[REINFORCEConfig] = None,
        value_lr: float = 1e-4,
    ):
        """
        Initialize REINFORCE with value baseline.

        Args:
            config: REINFORCE configuration
            value_lr: Learning rate for value function
        """
        super().__init__(config)
        self.value_lr = value_lr
        self._value_estimates: Dict[str, float] = {}

    def _estimate_value(self, prompt: str) -> float:
        """
        Estimate value for a prompt.

        Args:
            prompt: The prompt string

        Returns:
            Estimated value (expected return)
        """
        # Simple lookup-based value function
        return self._value_estimates.get(prompt, self._baseline)

    def _update_value(self, prompt: str, return_: float) -> None:
        """
        Update value estimate for a prompt.

        Uses exponential moving average.

        Args:
            prompt: The prompt string
            return_: Observed return
        """
        current = self._value_estimates.get(prompt, self._baseline)
        updated = (1 - self.value_lr) * current + self.value_lr * return_
        self._value_estimates[prompt] = updated

    def prepare_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        """Prepare batch with per-prompt baselines."""
        # Compute advantages using per-prompt baselines
        for traj in batch:
            baseline = self._estimate_value(traj.prompt)
            traj.advantage = traj.reward - baseline

            # Update value estimate
            self._update_value(traj.prompt, traj.reward)

        # Optionally normalize
        if self.config.normalize_advantages and len(batch) > 1:
            advantages = [t.advantage for t in batch]
            mean_adv = sum(advantages) / len(advantages)
            std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5

            if std_adv > 1e-8:
                for traj in batch:
                    traj.advantage = (traj.advantage - mean_adv) / (std_adv + 1e-8)

        return batch
