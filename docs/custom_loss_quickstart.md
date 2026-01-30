# Custom Loss Function Quickstart

This guide shows how to use TenSafe's pluggable loss function system.

## Overview

TenSafe supports custom loss functions through a registry-based system. You can use:

- **Built-in losses**: Pre-registered losses like `token_ce`, `margin_ranking`
- **Custom functions**: Your own Python callables
- **Module imports**: Loss functions from external modules via dotted paths

## Quick Start

### Using Built-in Losses

```python
from tensafe.training.losses import resolve_loss

# Resolve a built-in loss by name
loss_fn = resolve_loss("token_ce")

# Use with default parameters
outputs = {"logits": model_logits}
batch = {"labels": input_labels}
result = loss_fn(outputs, batch)

print(f"Loss: {result['loss']}")
```

### Using Custom Loss Functions

```python
from tensafe.training.losses import resolve_loss, register_loss

# Option 1: Pass a callable directly
def my_custom_loss(outputs, batch, **kwargs):
    logits = outputs["logits"]
    labels = batch["labels"]
    # Your loss computation here
    loss = compute_loss(logits, labels)
    return {"loss": loss}

loss_fn = resolve_loss(my_custom_loss)

# Option 2: Register for reuse
register_loss("my_loss", my_custom_loss)
loss_fn = resolve_loss("my_loss")
```

### Using Losses from External Modules

```python
from tensafe.training.losses import resolve_loss

# Import from a module using dotted path
loss_fn = resolve_loss("my_package.losses:custom_loss")

# With default kwargs
loss_fn = resolve_loss(
    "my_package.losses:weighted_loss",
    weight=0.5,
    ignore_index=-100,
)
```

## Configuration via YAML

Losses can be configured in `configs/train_sft.yaml`:

```yaml
loss:
  type: token_ce  # or dotted path: "my_module:custom_loss"
  kwargs:
    ignore_index: -100
    label_smoothing: 0.1
    reduction: mean
```

## Built-in Losses

| Name | Description | Common Args |
|------|-------------|-------------|
| `token_ce` | Token-level cross-entropy | `ignore_index`, `label_smoothing` |
| `margin_ranking` | Margin ranking loss | `margin` |
| `contrastive` | Contrastive loss | `margin`, `temperature` |
| `mse` | Mean squared error | `reduction` |

## Creating a Custom Loss

### Loss Function Protocol

Loss functions must follow this protocol:

```python
from typing import Any, Dict, TypedDict

class LossReturn(TypedDict, total=False):
    loss: float  # Required: the loss value
    metrics: Dict[str, float]  # Optional: additional metrics
    auxiliary: Dict[str, Any]  # Optional: auxiliary outputs

def my_loss(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
    **kwargs,
) -> LossReturn:
    """
    Compute loss from model outputs and batch.

    Args:
        outputs: Dictionary containing model outputs (e.g., logits)
        batch: Dictionary containing batch data (e.g., labels)
        **kwargs: Additional parameters

    Returns:
        LossReturn with at least 'loss' key
    """
    ...
```

### Example: Entropy-Regularized Loss

```python
from tensafe.training.losses import register_loss

@register_loss("entropy_regularized")
def entropy_regularized_loss(outputs, batch, entropy_weight=0.1, **kwargs):
    """Cross-entropy with entropy regularization."""
    logits = outputs["logits"]
    labels = batch["labels"]

    # Standard CE loss
    ce_loss = compute_cross_entropy(logits, labels)

    # Entropy term (encourages diverse predictions)
    probs = softmax(logits)
    entropy = -sum(probs * log(probs))

    # Combined loss
    total_loss = ce_loss - entropy_weight * entropy

    return {
        "loss": total_loss,
        "metrics": {
            "ce_loss": ce_loss,
            "entropy": entropy,
        },
    }
```

### Example: Class-Based Loss

```python
from tensafe.training.losses import register_loss

class FocalLoss:
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, outputs, batch, **kwargs):
        logits = outputs["logits"]
        labels = batch["labels"]

        # Focal loss computation
        ce_loss = cross_entropy(logits, labels)
        pt = exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return {"loss": focal_loss.mean()}

# Register with default parameters
register_loss("focal_loss", FocalLoss())
```

## Testing Your Custom Loss

```python
import pytest
from tensafe.training.losses import resolve_loss

def test_custom_loss():
    loss_fn = resolve_loss("my_custom_loss")

    outputs = {"logits": [[0.1, 0.9], [0.8, 0.2]]}
    batch = {"labels": [1, 0]}

    result = loss_fn(outputs, batch)

    assert "loss" in result
    assert result["loss"] >= 0
```

## Integration with Training

```python
from tensafe.training.losses import resolve_loss

# In your training script
loss_fn = resolve_loss(config.loss.type, **config.loss.kwargs)

for batch in dataloader:
    outputs = model(batch["input_ids"])
    loss_result = loss_fn(outputs, batch)

    loss = loss_result["loss"]
    loss.backward()

    # Log additional metrics if available
    if "metrics" in loss_result:
        for name, value in loss_result["metrics"].items():
            log_metric(name, value)
```

## Best Practices

1. **Return a dictionary**: Always return `{"loss": ...}` at minimum
2. **Handle missing keys**: Check for required keys in outputs/batch
3. **Support kwargs**: Accept `**kwargs` for flexibility
4. **Add metrics**: Return useful metrics for debugging
5. **Test thoroughly**: Verify behavior with edge cases

## See Also

- [Baseline SFT Documentation](dev/baseline.md)
- [RLVR Quickstart](rlvr_quickstart.md)
- [Training Configuration](dev/training_config.md)
