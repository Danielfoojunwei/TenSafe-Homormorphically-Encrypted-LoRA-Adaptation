"""
RLVR Policy Loss Registry

Provides a pluggable registry for policy loss functions, matching SkyRL's
policy loss diversity while staying within TenSafe's architecture.

Supported losses:
- ppo_clip: Standard PPO clipped surrogate objective
- gspo: Sequence-level importance sampling (reduces token-level variance)
- sapo: Temperature-based gating adapting to advantage sign
- cispo: Gradient-space clipping of importance weights
- clip_cov: Token subset selection based on advantage/log-prob covariance
- kl_cov: KL regularization based on covariance selection
- cross_entropy: Simple NLL for supervised fine-tuning fallback
- importance_sampling: Off-policy correction without clipping
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PolicyLossInput:
    """Input data for policy loss computation."""

    # Per-token log probabilities under current policy
    logprobs: List[List[float]]
    # Per-token log probabilities under old/reference policy
    old_logprobs: List[List[float]]
    # Per-trajectory advantages
    advantages: List[float]
    # Loss masks (which tokens to include)
    loss_masks: Optional[List[List[float]]] = None
    # Per-token advantages (for token-level losses)
    token_advantages: Optional[List[List[float]]] = None


@dataclass
class PolicyLossResult:
    """Result from policy loss computation."""

    loss: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyLossFn(Protocol):
    """Protocol for policy loss functions."""

    def __call__(self, inp: PolicyLossInput, **kwargs: Any) -> PolicyLossResult: ...


# Global policy loss registry
_POLICY_LOSS_REGISTRY: Dict[str, PolicyLossFn] = {}


def register_policy_loss(
    name: str,
    fn: Optional[PolicyLossFn] = None,
    *,
    overwrite: bool = False,
) -> Any:
    """Register a policy loss function. Can be used as a decorator."""

    def decorator(f: PolicyLossFn) -> PolicyLossFn:
        if name in _POLICY_LOSS_REGISTRY and not overwrite:
            raise ValueError(f"Policy loss '{name}' already registered")
        _POLICY_LOSS_REGISTRY[name] = f
        logger.debug(f"Registered policy loss: {name}")
        return f

    if fn is not None:
        return decorator(fn)
    return decorator


def resolve_policy_loss(name: str) -> PolicyLossFn:
    """Resolve a policy loss by name."""
    if name not in _POLICY_LOSS_REGISTRY:
        available = list(_POLICY_LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown policy loss '{name}'. Available: {available}")
    return _POLICY_LOSS_REGISTRY[name]


def list_policy_losses() -> List[str]:
    """List all registered policy losses."""
    return list(_POLICY_LOSS_REGISTRY.keys())


# ==============================================================================
# Utility functions
# ==============================================================================


def _compute_log_ratios(
    logprobs: List[float], old_logprobs: List[float]
) -> List[float]:
    """Compute per-token log importance ratios."""
    return [lp - olp for lp, olp in zip(logprobs, old_logprobs)]


def _safe_exp(x: float, max_val: float = 20.0) -> float:
    """Numerically stable exp."""
    return math.exp(min(x, max_val))


def _masked_mean(
    values: List[float], mask: Optional[List[float]] = None
) -> float:
    """Compute masked mean of values."""
    if mask is None:
        return sum(values) / len(values) if values else 0.0
    total = sum(v * m for v, m in zip(values, mask))
    count = sum(mask)
    return total / count if count > 0 else 0.0


# ==============================================================================
# Built-in Policy Losses
# ==============================================================================


@register_policy_loss("ppo_clip")
def ppo_clip_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    dual_clip: Optional[float] = None,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Standard PPO clipped surrogate objective.

    Clips the importance sampling ratio within [1-eps, 1+eps] to prevent
    large policy updates. Optional dual-clip for negative advantages.
    """
    total_loss = 0.0
    total_tokens = 0
    clip_fraction = 0.0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue

            log_ratio = logprobs[t] - old_logprobs[t]
            ratio = _safe_exp(log_ratio)

            surr1 = ratio * advantage
            surr2 = max(min(ratio, 1.0 + clip_range), 1.0 - clip_range) * advantage

            if dual_clip is not None and advantage < 0:
                token_loss = -max(min(surr1, surr2), dual_clip * advantage)
            else:
                token_loss = -min(surr1, surr2)

            total_loss += token_loss
            total_tokens += 1

            if abs(ratio - 1.0) > clip_range:
                clip_fraction += 1.0

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    clip_frac = clip_fraction / total_tokens if total_tokens > 0 else 0.0

    return PolicyLossResult(
        loss=avg_loss,
        metadata={"clip_fraction": clip_frac, "tokens_used": total_tokens},
    )


@register_policy_loss("gspo")
def gspo_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Sequence-level importance sampling (GSPO).

    Instead of per-token IS ratios, computes the IS ratio at the sequence
    level (sum of log-ratios), reducing token-level variance.
    """
    total_loss = 0.0
    n_seqs = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        # Sequence-level log ratio
        seq_log_ratio = 0.0
        n_tokens = 0
        for t in range(len(logprobs)):
            if mask[t] >= 0.5:
                seq_log_ratio += logprobs[t] - old_logprobs[t]
                n_tokens += 1

        if n_tokens == 0:
            continue

        # Average per-token log ratio for numerical stability
        avg_log_ratio = seq_log_ratio / n_tokens
        ratio = _safe_exp(avg_log_ratio)

        surr1 = ratio * advantage
        surr2 = max(min(ratio, 1.0 + clip_range), 1.0 - clip_range) * advantage
        seq_loss = -min(surr1, surr2)

        total_loss += seq_loss
        n_seqs += 1

    avg_loss = total_loss / n_seqs if n_seqs > 0 else 0.0
    return PolicyLossResult(loss=avg_loss, metadata={"sequences_used": n_seqs})


@register_policy_loss("sapo")
def sapo_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    beta: float = 0.1,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Self-Adaptive Policy Optimization (SAPO).

    Uses temperature-based gating that adapts to the sign of the advantage.
    For positive advantages, encourages exploration; for negative advantages,
    applies stronger regularization.
    """
    total_loss = 0.0
    total_tokens = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue

            log_ratio = logprobs[t] - old_logprobs[t]
            ratio = _safe_exp(log_ratio)

            # SAPO: temperature-gated clipping
            if advantage >= 0:
                # Positive advantage: standard clip with exploration bonus
                gate = min(ratio, 1.0 + clip_range)
                token_loss = -gate * advantage
            else:
                # Negative advantage: tighter clip + KL penalty
                gate = max(ratio, 1.0 - clip_range)
                kl_penalty = beta * (ratio * log_ratio - (ratio - 1.0))
                token_loss = -gate * advantage + kl_penalty

            total_loss += token_loss
            total_tokens += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return PolicyLossResult(loss=avg_loss, metadata={"tokens_used": total_tokens})


@register_policy_loss("cispo")
def cispo_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Clipping in Policy Space Optimization (CISPO).

    Clips importance weights in gradient space rather than loss space.
    The gradient contribution of each token is bounded, preventing any
    single token from dominating the update.
    """
    total_loss = 0.0
    total_tokens = 0
    gradient_clips = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue

            log_ratio = logprobs[t] - old_logprobs[t]
            ratio = _safe_exp(log_ratio)

            # CISPO: clip the gradient contribution
            # gradient = advantage * ratio * d(log_pi)/d(theta)
            # Instead of clipping ratio in loss, we clip the effective
            # gradient weight: min(ratio, 1+eps) for positive advantage,
            # max(ratio, 1-eps) for negative advantage
            if advantage >= 0:
                effective_ratio = min(ratio, 1.0 + clip_range)
                if ratio > 1.0 + clip_range:
                    # In gradient space: set gradient to zero when clipped
                    token_loss = -(1.0 + clip_range) * advantage
                    gradient_clips += 1
                else:
                    token_loss = -ratio * advantage
            else:
                effective_ratio = max(ratio, 1.0 - clip_range)
                if ratio < 1.0 - clip_range:
                    token_loss = -(1.0 - clip_range) * advantage
                    gradient_clips += 1
                else:
                    token_loss = -ratio * advantage

            total_loss += token_loss
            total_tokens += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    grad_clip_frac = gradient_clips / total_tokens if total_tokens > 0 else 0.0

    return PolicyLossResult(
        loss=avg_loss,
        metadata={"gradient_clip_fraction": grad_clip_frac, "tokens_used": total_tokens},
    )


@register_policy_loss("clip_cov")
def clip_cov_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    cov_threshold: float = 0.0,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Covariance-based token selection with PPO clipping (Clip-Cov).

    Selects a subset of tokens based on the covariance between advantages
    and log-probability ratios. Only tokens with positive covariance
    contribute to the loss, focusing the update on informative tokens.
    """
    total_loss = 0.0
    total_tokens = 0
    selected_tokens = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        # Compute per-token log ratios
        log_ratios = []
        valid_indices = []
        for t in range(len(logprobs)):
            if mask[t] >= 0.5:
                log_ratios.append(logprobs[t] - old_logprobs[t])
                valid_indices.append(t)

        if not log_ratios:
            continue

        # Compute covariance signal: advantage * log_ratio
        # Tokens where this is positive are "aligned" with the update
        for idx, t in enumerate(valid_indices):
            cov_signal = advantage * log_ratios[idx]

            if cov_signal >= cov_threshold:
                # Include this token in the loss
                ratio = _safe_exp(log_ratios[idx])
                surr1 = ratio * advantage
                surr2 = (
                    max(min(ratio, 1.0 + clip_range), 1.0 - clip_range) * advantage
                )
                token_loss = -min(surr1, surr2)
                total_loss += token_loss
                selected_tokens += 1

            total_tokens += 1

    avg_loss = total_loss / selected_tokens if selected_tokens > 0 else 0.0
    select_frac = selected_tokens / total_tokens if total_tokens > 0 else 0.0

    return PolicyLossResult(
        loss=avg_loss,
        metadata={
            "selection_fraction": select_frac,
            "selected_tokens": selected_tokens,
            "total_tokens": total_tokens,
        },
    )


@register_policy_loss("kl_cov")
def kl_cov_loss(
    inp: PolicyLossInput,
    *,
    clip_range: float = 0.2,
    kl_coef: float = 0.1,
    cov_threshold: float = 0.0,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    KL regularization based on covariance selection (KL-Cov).

    Similar to Clip-Cov but uses KL divergence regularization for tokens
    that fall outside the covariance threshold, instead of dropping them.
    """
    policy_loss = 0.0
    kl_loss = 0.0
    total_tokens = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue

            log_ratio = logprobs[t] - old_logprobs[t]
            ratio = _safe_exp(log_ratio)
            cov_signal = advantage * log_ratio

            if cov_signal >= cov_threshold:
                # Standard PPO clip for selected tokens
                surr1 = ratio * advantage
                surr2 = (
                    max(min(ratio, 1.0 + clip_range), 1.0 - clip_range) * advantage
                )
                policy_loss += -min(surr1, surr2)
            else:
                # KL regularization for non-selected tokens
                # KL ≈ ratio * log_ratio - (ratio - 1)
                kl_approx = ratio * log_ratio - (ratio - 1.0)
                kl_loss += kl_coef * kl_approx

            total_tokens += 1

    avg_policy = policy_loss / total_tokens if total_tokens > 0 else 0.0
    avg_kl = kl_loss / total_tokens if total_tokens > 0 else 0.0

    return PolicyLossResult(
        loss=avg_policy + avg_kl,
        metadata={
            "policy_loss": avg_policy,
            "kl_loss": avg_kl,
            "tokens_used": total_tokens,
        },
    )


@register_policy_loss("cross_entropy")
def cross_entropy_loss(
    inp: PolicyLossInput,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Simple negative log-likelihood loss for supervised fine-tuning mode.

    Ignores advantages and old_logprobs, just maximizes probability of
    the response tokens. Useful as a warm-start or SFT fallback within
    the RL pipeline.
    """
    total_loss = 0.0
    total_tokens = 0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue
            total_loss += -logprobs[t]
            total_tokens += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    return PolicyLossResult(loss=avg_loss, metadata={"tokens_used": total_tokens})


@register_policy_loss("importance_sampling")
def importance_sampling_loss(
    inp: PolicyLossInput,
    *,
    clip_min: float = 0.0,
    clip_max: float = 10.0,
    **kwargs: Any,
) -> PolicyLossResult:
    """
    Off-policy correction using probability ratios without PPO clipping.

    Uses raw importance sampling ratios (optionally bounded) to correct
    for policy drift. Less conservative than PPO clipping, useful for
    off-policy or async training settings.
    """
    total_loss = 0.0
    total_tokens = 0
    mean_ratio = 0.0

    for seq_idx in range(len(inp.logprobs)):
        logprobs = inp.logprobs[seq_idx]
        old_logprobs = inp.old_logprobs[seq_idx]
        advantage = inp.advantages[seq_idx]
        mask = inp.loss_masks[seq_idx] if inp.loss_masks else [1.0] * len(logprobs)

        for t in range(len(logprobs)):
            if mask[t] < 0.5:
                continue

            log_ratio = logprobs[t] - old_logprobs[t]
            ratio = _safe_exp(log_ratio)

            # Bound the ratio for numerical stability
            ratio = max(clip_min, min(clip_max, ratio))

            total_loss += -ratio * advantage
            mean_ratio += ratio
            total_tokens += 1

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    avg_ratio = mean_ratio / total_tokens if total_tokens > 0 else 1.0

    return PolicyLossResult(
        loss=avg_loss,
        metadata={"mean_ratio": avg_ratio, "tokens_used": total_tokens},
    )
