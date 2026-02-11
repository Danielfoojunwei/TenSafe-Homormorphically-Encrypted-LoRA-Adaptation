"""
RLVR Advantage Estimator Registry

Provides a pluggable registry for advantage estimation functions,
following the same pattern as the reward/loss registries.

Supported estimators:
- baseline: Simple baseline subtraction (mean reward)
- grpo: Group Relative Policy Optimization (per-prompt group normalization)
- rloo: Reinforcement Learning with Leave-One-Out baseline
- reinforce_pp: REINFORCE++ with temporal discount and whitening
- gae: Generalized Advantage Estimation (requires value function)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

from .rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class AdvantageResult:
    """Result from advantage computation."""

    advantages: List[float]
    baseline_value: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AdvantageEstimatorFn(Protocol):
    """Protocol for advantage estimator functions."""

    def __call__(
        self,
        batch: TrajectoryBatch,
        *,
        group_key: Optional[str] = None,
        gamma: float = 1.0,
        lam: float = 0.95,
        **kwargs: Any,
    ) -> AdvantageResult: ...


# Global advantage estimator registry
_ADVANTAGE_REGISTRY: Dict[str, AdvantageEstimatorFn] = {}


def register_advantage(
    name: str,
    fn: Optional[AdvantageEstimatorFn] = None,
    *,
    overwrite: bool = False,
) -> Any:
    """Register an advantage estimator. Can be used as a decorator."""

    def decorator(f: AdvantageEstimatorFn) -> AdvantageEstimatorFn:
        if name in _ADVANTAGE_REGISTRY and not overwrite:
            raise ValueError(f"Advantage estimator '{name}' already registered")
        _ADVANTAGE_REGISTRY[name] = f
        logger.debug(f"Registered advantage estimator: {name}")
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def resolve_advantage(name: str) -> AdvantageEstimatorFn:
    """Resolve an advantage estimator by name."""
    if name not in _ADVANTAGE_REGISTRY:
        available = list(_ADVANTAGE_REGISTRY.keys())
        raise ValueError(
            f"Unknown advantage estimator '{name}'. Available: {available}"
        )
    return _ADVANTAGE_REGISTRY[name]


def list_advantage_estimators() -> List[str]:
    """List all registered advantage estimators."""
    return list(_ADVANTAGE_REGISTRY.keys())


def apply_advantage(
    name: str,
    batch: TrajectoryBatch,
    normalize: bool = True,
    **kwargs: Any,
) -> TrajectoryBatch:
    """
    Apply a named advantage estimator to a batch and update trajectory advantages.

    Args:
        name: Registered estimator name
        batch: Trajectory batch with computed rewards
        normalize: Whether to normalize advantages after computation
        **kwargs: Additional arguments passed to the estimator

    Returns:
        The batch with updated advantages on each trajectory
    """
    estimator = resolve_advantage(name)
    result = estimator(batch, **kwargs)

    for traj, adv in zip(batch.trajectories, result.advantages):
        traj.advantage = adv

    if normalize:
        _normalize_advantages(batch)

    return batch


def _normalize_advantages(batch: TrajectoryBatch) -> None:
    """Normalize advantages to zero mean and unit variance."""
    advantages = [t.advantage for t in batch.trajectories]
    if len(advantages) < 2:
        return

    mean = sum(advantages) / len(advantages)
    var = sum((a - mean) ** 2 for a in advantages) / len(advantages)
    std = math.sqrt(var) + 1e-8

    for traj in batch.trajectories:
        traj.advantage = (traj.advantage - mean) / std


# ==============================================================================
# Built-in Advantage Estimators
# ==============================================================================


@register_advantage("baseline")
def baseline_advantage(
    batch: TrajectoryBatch,
    *,
    group_key: Optional[str] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
    baseline_value: Optional[float] = None,
    **kwargs: Any,
) -> AdvantageResult:
    """
    Simple baseline subtraction advantage estimation.

    Advantage = reward - baseline, where baseline defaults to batch mean reward.
    """
    if baseline_value is None:
        baseline_value = batch.mean_reward

    advantages = [t.reward - baseline_value for t in batch.trajectories]
    return AdvantageResult(advantages=advantages, baseline_value=baseline_value)


@register_advantage("grpo")
def grpo_advantage(
    batch: TrajectoryBatch,
    *,
    group_key: Optional[str] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
    eps: float = 1e-8,
    normalize_within_group: bool = True,
    **kwargs: Any,
) -> AdvantageResult:
    """
    Group Relative Policy Optimization advantage estimation.

    For each prompt group, normalizes rewards by group mean and (optionally)
    group standard deviation. This eliminates the need for a learned critic.

    When group_key is None, groups are determined by matching prompts.
    When all prompts are unique, falls back to batch-level normalization.

    Reference: DeepSeek-R1 / GRPO (Shao et al., 2024)
    """
    key = group_key or "prompt"

    # Group trajectories by prompt
    groups: Dict[str, List[int]] = {}
    for i, traj in enumerate(batch.trajectories):
        group_id = getattr(traj, key) if hasattr(traj, key) else traj.prompt
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)

    advantages = [0.0] * len(batch.trajectories)

    for group_id, indices in groups.items():
        group_rewards = [batch.trajectories[i].reward for i in indices]
        group_mean = sum(group_rewards) / len(group_rewards)

        if normalize_within_group and len(group_rewards) > 1:
            group_var = sum((r - group_mean) ** 2 for r in group_rewards) / len(
                group_rewards
            )
            group_std = math.sqrt(group_var) + eps
        else:
            group_std = 1.0

        for i in indices:
            advantages[i] = (batch.trajectories[i].reward - group_mean) / group_std

    return AdvantageResult(
        advantages=advantages,
        baseline_value=batch.mean_reward,
        metadata={"num_groups": len(groups)},
    )


@register_advantage("rloo")
def rloo_advantage(
    batch: TrajectoryBatch,
    *,
    group_key: Optional[str] = None,
    gamma: float = 1.0,
    lam: float = 0.95,
    eps: float = 1e-8,
    **kwargs: Any,
) -> AdvantageResult:
    """
    Reinforcement Learning with Leave-One-Out baseline.

    For each trajectory in a group, the baseline is the mean of all OTHER
    trajectories' rewards in the same group. This gives a lower-variance
    baseline compared to using the group mean (which includes the trajectory
    itself).

    For groups with a single trajectory, falls back to batch-level baseline.
    """
    key = group_key or "prompt"

    # Group trajectories by prompt
    groups: Dict[str, List[int]] = {}
    for i, traj in enumerate(batch.trajectories):
        group_id = getattr(traj, key) if hasattr(traj, key) else traj.prompt
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(i)

    batch_mean = batch.mean_reward
    advantages = [0.0] * len(batch.trajectories)

    for group_id, indices in groups.items():
        group_rewards = [batch.trajectories[i].reward for i in indices]
        group_sum = sum(group_rewards)
        n = len(group_rewards)

        if n <= 1:
            # Single sample in group, use batch-level baseline
            for i in indices:
                advantages[i] = batch.trajectories[i].reward - batch_mean
        else:
            # Leave-one-out: baseline_i = (sum - reward_i) / (n - 1)
            for idx, i in enumerate(indices):
                loo_baseline = (group_sum - group_rewards[idx]) / (n - 1)
                advantages[i] = batch.trajectories[i].reward - loo_baseline

    return AdvantageResult(
        advantages=advantages,
        baseline_value=batch_mean,
        metadata={"num_groups": len(groups)},
    )


@register_advantage("reinforce_pp")
def reinforce_pp_advantage(
    batch: TrajectoryBatch,
    *,
    group_key: Optional[str] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    eps: float = 1e-8,
    **kwargs: Any,
) -> AdvantageResult:
    """
    REINFORCE++ advantage estimation with temporal discount and whitening.

    Accumulates discounted rewards backward through each trajectory's
    response tokens, then applies per-token temporal whitening.

    For single-turn RL (where reward is a single scalar at the end),
    this distributes the reward across tokens with temporal discounting.
    """
    advantages = []

    for traj in batch.trajectories:
        n_tokens = traj.num_response_tokens
        if n_tokens == 0:
            advantages.append(0.0)
            continue

        # Distribute reward across tokens with temporal discounting
        # For single-turn: reward only at the last token
        per_token_rewards = [0.0] * n_tokens
        per_token_rewards[-1] = traj.reward

        # Accumulate discounted returns backward
        returns = [0.0] * n_tokens
        running_return = 0.0
        for t in range(n_tokens - 1, -1, -1):
            running_return = per_token_rewards[t] + gamma * running_return
            returns[t] = running_return

        # Temporal whitening: normalize returns
        if n_tokens > 1:
            mean_ret = sum(returns) / n_tokens
            var_ret = sum((r - mean_ret) ** 2 for r in returns) / n_tokens
            std_ret = math.sqrt(var_ret) + eps
            whitened = [(r - mean_ret) / std_ret for r in returns]
        else:
            whitened = returns

        # Use mean of whitened returns as the trajectory-level advantage
        advantages.append(sum(whitened) / n_tokens)

    return AdvantageResult(
        advantages=advantages,
        baseline_value=0.0,
        metadata={"gamma": gamma},
    )


@register_advantage("gae")
def gae_advantage(
    batch: TrajectoryBatch,
    *,
    group_key: Optional[str] = None,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_estimates: Optional[List[float]] = None,
    eps: float = 1e-8,
    **kwargs: Any,
) -> AdvantageResult:
    """
    Generalized Advantage Estimation (Schulman et al., 2016).

    Requires value function estimates for each trajectory. When not provided,
    uses batch mean reward as a constant value estimate.

    GAE interpolates between high-bias/low-variance (lam=0) and
    low-bias/high-variance (lam=1) advantage estimates.
    """
    n = len(batch.trajectories)

    if value_estimates is None:
        # Fallback: use batch mean as constant value estimate
        v = batch.mean_reward
        value_estimates = [v] * n

    advantages = []
    for i, traj in enumerate(batch.trajectories):
        n_tokens = traj.num_response_tokens
        if n_tokens == 0:
            advantages.append(0.0)
            continue

        v_current = value_estimates[i]
        v_next = 0.0  # Terminal state value

        # For single-turn: one-step TD error
        # delta = reward + gamma * V(s') - V(s)
        delta = traj.reward + gamma * v_next - v_current

        # For single-step, GAE reduces to delta
        # For multi-step, would accumulate: A = sum_{t} (gamma*lam)^t * delta_t
        advantage = delta

        advantages.append(advantage)

    return AdvantageResult(
        advantages=advantages,
        baseline_value=sum(value_estimates) / n if n > 0 else 0.0,
        metadata={"gamma": gamma, "lambda": lam},
    )
