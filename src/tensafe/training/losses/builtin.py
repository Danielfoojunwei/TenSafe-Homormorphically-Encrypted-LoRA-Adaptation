"""
Built-in loss functions for TenSafe training.

This module provides commonly used loss functions that are automatically
registered in the loss registry.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

from .base import LossReturn

# Try to import torch, but provide mock implementations for testing
try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None  # type: ignore
    F = None  # type: ignore

# Import registry for registration
from .registry import register_loss


def _ensure_torch() -> None:
    """Raise an error if torch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for built-in loss functions. "
            "Install with: pip install torch"
        )


@register_loss("token_ce")
def token_cross_entropy_loss(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Token-level cross-entropy loss for language modeling.

    This is the standard loss for causal language model fine-tuning (SFT).

    Args:
        outputs: Model outputs containing 'logits' key
        batch: Batch containing 'labels' key
        ignore_index: Label value to ignore in loss computation (default: -100)
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: Reduction method: 'none', 'mean', 'sum' (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with loss and perplexity metric

    Example:
        >>> loss_fn = resolve_loss("token_ce", ignore_index=-100)
        >>> result = loss_fn({"logits": logits}, {"labels": labels})
    """
    _ensure_torch()

    # Extract logits and labels
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("outputs must contain 'logits' key")
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        raise ValueError("Cannot extract logits from outputs")

    labels = batch.get("labels")
    if labels is None:
        raise ValueError("batch must contain 'labels' key")

    # Reshape for cross entropy
    # logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
    # labels: [batch_size, seq_len] -> [batch_size * seq_len]
    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    # Compute perplexity (only meaningful for mean reduction)
    if reduction == "mean":
        perplexity = torch.exp(loss).item()
    else:
        perplexity = float("nan")

    # Count non-ignored tokens
    num_tokens = (shift_labels != ignore_index).sum().item()

    return {
        "loss": loss,
        "metrics": {
            "perplexity": perplexity,
            "num_tokens": num_tokens,
        },
    }


@register_loss("margin_ranking")
def margin_ranking_loss(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    margin: float = 1.0,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Margin ranking loss for preference learning.

    Computes loss based on the margin between preferred and rejected outputs.
    Useful for RLHF reward model training.

    Args:
        outputs: Model outputs containing:
            - 'chosen_scores' or 'logits_chosen': Scores for preferred samples
            - 'rejected_scores' or 'logits_rejected': Scores for rejected samples
        batch: Batch (not used directly, but may contain metadata)
        margin: Margin for the ranking loss (default: 1.0)
        reduction: Reduction method: 'none', 'mean', 'sum' (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with loss and accuracy metric

    Example:
        >>> result = margin_ranking_loss(
        ...     {"chosen_scores": chosen, "rejected_scores": rejected},
        ...     batch
        ... )
    """
    _ensure_torch()

    # Extract scores
    if isinstance(outputs, dict):
        chosen = outputs.get("chosen_scores") or outputs.get("logits_chosen")
        rejected = outputs.get("rejected_scores") or outputs.get("logits_rejected")
    else:
        raise ValueError("outputs must be a dict for margin ranking loss")

    if chosen is None or rejected is None:
        raise ValueError(
            "outputs must contain 'chosen_scores'/'logits_chosen' and "
            "'rejected_scores'/'logits_rejected' keys"
        )

    # Target: chosen should be ranked higher (target = 1)
    target = torch.ones_like(chosen)

    loss = F.margin_ranking_loss(
        chosen,
        rejected,
        target,
        margin=margin,
        reduction=reduction,
    )

    # Compute accuracy: how often is chosen > rejected?
    accuracy = (chosen > rejected).float().mean().item()

    return {
        "loss": loss,
        "metrics": {
            "accuracy": accuracy,
            "margin": margin,
            "mean_chosen": chosen.mean().item(),
            "mean_rejected": rejected.mean().item(),
        },
    }


@register_loss("contrastive")
def contrastive_loss(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    temperature: float = 0.07,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Contrastive loss (InfoNCE / NT-Xent style).

    Useful for representation learning and embedding models.

    Args:
        outputs: Model outputs containing:
            - 'embeddings_a': First set of embeddings [batch_size, embed_dim]
            - 'embeddings_b': Second set of embeddings [batch_size, embed_dim]
            Or just 'embeddings' for self-supervised setting
        batch: Batch (may contain additional info)
        temperature: Temperature for softmax scaling (default: 0.07)
        reduction: Reduction method (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with loss and similarity metrics

    Example:
        >>> result = contrastive_loss(
        ...     {"embeddings_a": emb1, "embeddings_b": emb2},
        ...     batch,
        ...     temperature=0.1
        ... )
    """
    _ensure_torch()

    if isinstance(outputs, dict):
        emb_a = outputs.get("embeddings_a") or outputs.get("embeddings")
        emb_b = outputs.get("embeddings_b")

        if emb_b is None:
            # Self-supervised: compare embeddings to themselves
            emb_b = emb_a
    else:
        raise ValueError("outputs must be a dict for contrastive loss")

    if emb_a is None:
        raise ValueError("outputs must contain 'embeddings_a' or 'embeddings' key")

    # Normalize embeddings
    emb_a = F.normalize(emb_a, p=2, dim=-1)
    emb_b = F.normalize(emb_b, p=2, dim=-1)

    # Compute similarity matrix
    batch_size = emb_a.size(0)
    similarity = torch.matmul(emb_a, emb_b.T) / temperature

    # Labels: diagonal entries are positives (i matches with i)
    labels = torch.arange(batch_size, device=emb_a.device)

    # Symmetric loss: both directions
    loss_a_to_b = F.cross_entropy(similarity, labels, reduction=reduction)
    loss_b_to_a = F.cross_entropy(similarity.T, labels, reduction=reduction)
    loss = (loss_a_to_b + loss_b_to_a) / 2

    # Metrics
    diag_similarity = torch.diag(similarity * temperature).mean().item()

    return {
        "loss": loss,
        "metrics": {
            "temperature": temperature,
            "mean_positive_similarity": diag_similarity,
            "batch_size": batch_size,
        },
    }


@register_loss("mse")
def mse_loss(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Mean squared error loss.

    Useful for regression tasks or continuous value prediction.

    Args:
        outputs: Model outputs containing 'predictions' or 'logits'
        batch: Batch containing 'targets' or 'labels'
        reduction: Reduction method (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with loss and RMSE metric
    """
    _ensure_torch()

    # Extract predictions
    if isinstance(outputs, dict):
        preds = outputs.get("predictions") or outputs.get("logits") or outputs.get("output")
    else:
        preds = outputs

    if preds is None:
        raise ValueError("Cannot extract predictions from outputs")

    # Extract targets
    targets = batch.get("targets") or batch.get("labels")
    if targets is None:
        raise ValueError("batch must contain 'targets' or 'labels' key")

    loss = F.mse_loss(preds, targets, reduction=reduction)

    # Compute RMSE
    rmse = torch.sqrt(loss).item() if reduction == "mean" else float("nan")

    return {
        "loss": loss,
        "metrics": {
            "rmse": rmse,
            "mean_pred": preds.mean().item(),
            "mean_target": targets.mean().item(),
        },
    }


# Mock implementations for testing without torch
class MockLossFunctions:
    """Mock loss functions for testing without PyTorch."""

    @staticmethod
    def token_cross_entropy(
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        """Mock token CE loss."""
        # Simulate loss based on batch size
        batch_size = len(batch.get("input_ids", [[]])) or 1
        loss = 2.5 - (kwargs.get("step", 0) * 0.01)
        return {
            "loss": max(0.1, loss),
            "metrics": {
                "perplexity": math.exp(loss),
                "num_tokens": batch_size * 128,
            },
        }

    @staticmethod
    def margin_ranking(
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        **kwargs: Any,
    ) -> LossReturn:
        """Mock margin ranking loss."""
        return {
            "loss": 0.5,
            "metrics": {
                "accuracy": 0.75,
                "margin": kwargs.get("margin", 1.0),
            },
        }


def get_mock_loss(name: str) -> Any:
    """Get a mock loss function for testing."""
    mocks = {
        "token_ce": MockLossFunctions.token_cross_entropy,
        "margin_ranking": MockLossFunctions.margin_ranking,
    }
    return mocks.get(name)
