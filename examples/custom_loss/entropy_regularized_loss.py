"""
Example: Entropy-Regularized Cross-Entropy Loss

This example demonstrates how to create a custom loss function that adds
entropy regularization to the standard cross-entropy loss.

Entropy regularization encourages the model to maintain diverse predictions,
which can help prevent overconfident outputs and improve generalization.

Usage:
    # As a dotted path
    loss_fn = resolve_loss("examples.custom_loss.entropy_regularized_loss:entropy_regularized_ce")

    # Or import directly
    from examples.custom_loss.entropy_regularized_loss import entropy_regularized_ce
    loss_fn = resolve_loss(entropy_regularized_ce)
"""

from __future__ import annotations

from typing import Any, Dict, Union

# Try to import torch
try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import the base types
import sys
from pathlib import Path

# Add tensafe to path if not installed
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.training.losses.base import LossReturn


def entropy_regularized_ce(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    ignore_index: int = -100,
    entropy_weight: float = 0.01,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Cross-entropy loss with entropy regularization.

    The total loss is:
        L = CE_loss - entropy_weight * entropy

    Higher entropy means more uniform probability distributions, which can
    help prevent overconfident predictions.

    Args:
        outputs: Model outputs containing 'logits' key
            Shape: [batch_size, seq_len, vocab_size]
        batch: Batch containing 'labels' key
            Shape: [batch_size, seq_len]
        ignore_index: Label value to ignore (default: -100)
        entropy_weight: Weight for entropy regularization (default: 0.01)
            - Higher values encourage more diverse predictions
            - Set to 0 for standard CE loss
        label_smoothing: Label smoothing factor (default: 0.0)
        reduction: Reduction method (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with:
            - loss: Total loss (CE - entropy_weight * entropy)
            - metrics: CE loss, entropy, perplexity
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this loss function")

    # Extract logits
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("outputs must contain 'logits' key")
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        raise ValueError("Cannot extract logits from outputs")

    # Extract labels
    labels = batch.get("labels")
    if labels is None:
        raise ValueError("batch must contain 'labels' key")

    # Shift for causal LM (predict next token)
    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute cross-entropy loss
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )

    # Compute entropy of the predicted distribution
    # entropy = -sum(p * log(p))
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    # Mask out ignored positions
    mask = (shift_labels != ignore_index).float()
    if reduction == "mean":
        masked_entropy = (entropy * mask.unsqueeze(-1).expand_as(entropy)).sum()
        masked_entropy = masked_entropy / mask.sum().clamp(min=1)
    else:
        masked_entropy = entropy

    # Total loss: CE - entropy_weight * entropy
    # Subtracting entropy encourages higher entropy (more diverse predictions)
    total_loss = ce_loss - entropy_weight * masked_entropy.mean()

    # Compute metrics
    perplexity = torch.exp(ce_loss).item()
    avg_entropy = masked_entropy.mean().item() if reduction == "mean" else float("nan")
    num_tokens = mask.sum().item()

    return {
        "loss": total_loss,
        "metrics": {
            "ce_loss": ce_loss.item(),
            "entropy": avg_entropy,
            "entropy_contribution": (entropy_weight * avg_entropy),
            "perplexity": perplexity,
            "num_tokens": num_tokens,
        },
    }


def focal_cross_entropy(
    outputs: Union[Dict[str, Any], Any],
    batch: Dict[str, Any],
    *,
    ignore_index: int = -100,
    gamma: float = 2.0,
    alpha: float = 1.0,
    reduction: str = "mean",
    **kwargs: Any,
) -> LossReturn:
    """
    Focal loss for handling class imbalance.

    Focal loss down-weights well-classified examples and focuses on hard examples:
        FL = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        outputs: Model outputs containing 'logits' key
        batch: Batch containing 'labels' key
        ignore_index: Label value to ignore (default: -100)
        gamma: Focusing parameter (default: 2.0)
            - gamma=0 is equivalent to standard CE
            - Higher gamma focuses more on hard examples
        alpha: Weighting factor (default: 1.0)
        reduction: Reduction method (default: 'mean')
        **kwargs: Additional arguments (ignored)

    Returns:
        LossReturn with loss and metrics
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for this loss function")

    # Extract logits and labels
    if isinstance(outputs, dict):
        logits = outputs.get("logits")
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        raise ValueError("Cannot extract logits from outputs")

    labels = batch.get("labels")
    if labels is None:
        raise ValueError("batch must contain 'labels' key")

    # Shift for causal LM
    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)

    # Create mask for valid tokens
    mask = (flat_labels != ignore_index)

    # Compute probabilities
    probs = F.softmax(flat_logits, dim=-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)

    # Get probability of correct class
    # p_t = probs[true_label]
    valid_labels = flat_labels.clone()
    valid_labels[~mask] = 0  # Temporarily set to valid index

    p_t = probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
    log_p_t = log_probs.gather(1, valid_labels.unsqueeze(1)).squeeze(1)

    # Focal loss: -alpha * (1 - p_t)^gamma * log(p_t)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = -focal_weight * log_p_t

    # Apply mask and reduce
    masked_loss = focal_loss * mask.float()

    if reduction == "mean":
        loss = masked_loss.sum() / mask.sum().clamp(min=1)
    elif reduction == "sum":
        loss = masked_loss.sum()
    else:
        loss = masked_loss

    # Metrics
    ce_loss = -log_p_t[mask].mean().item() if mask.any() else 0.0
    avg_p_t = p_t[mask].mean().item() if mask.any() else 0.0

    return {
        "loss": loss,
        "metrics": {
            "ce_loss": ce_loss,
            "perplexity": float("nan") if ce_loss == 0 else 2.71828 ** ce_loss,
            "avg_confidence": avg_p_t,
            "gamma": gamma,
            "alpha": alpha,
        },
    }


# Example of using the register_loss decorator
try:
    from tensafe.training.losses import register_loss

    # Register our custom losses
    register_loss("entropy_ce", entropy_regularized_ce)
    register_loss("focal_ce", focal_cross_entropy)

except ImportError:
    # tensafe not installed as package; users will use dotted path import
    pass


if __name__ == "__main__":
    # Demo: show how to use the custom losses
    print("=" * 60)
    print("Custom Loss Function Example")
    print("=" * 60)
    print()
    print("This module provides two custom loss functions:")
    print()
    print("1. entropy_regularized_ce")
    print("   - Standard CE loss with entropy regularization")
    print("   - Encourages diverse predictions")
    print("   - Parameters: entropy_weight, label_smoothing")
    print()
    print("2. focal_cross_entropy")
    print("   - Focal loss for handling class imbalance")
    print("   - Focuses on hard examples")
    print("   - Parameters: gamma, alpha")
    print()
    print("Usage:")
    print("  from tensafe.training.losses import resolve_loss")
    print()
    print("  # Method 1: Dotted path import")
    print("  loss_fn = resolve_loss(")
    print('      "examples.custom_loss.entropy_regularized_loss:entropy_regularized_ce",')
    print("      entropy_weight=0.01")
    print("  )")
    print()
    print("  # Method 2: After registration")
    print('  loss_fn = resolve_loss("entropy_ce", entropy_weight=0.01)')
    print()
    print("  # Use in training loop")
    print('  result = loss_fn({"logits": logits}, {"labels": labels})')
    print('  print(f"Loss: {result[\'loss\']}, Metrics: {result[\'metrics\']}")')
