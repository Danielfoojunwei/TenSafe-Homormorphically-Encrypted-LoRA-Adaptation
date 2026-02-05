"""
LoRA+ Optimizer Implementation

Implements the LoRA+ optimization strategy from:
"LoRA+: Efficient Low Rank Adaptation of Large Models"
(Hayou et al., ICML 2024) https://arxiv.org/abs/2402.12354

Key insight: Using the same learning rate for A and B matrices is suboptimal.
The B matrix should have a higher learning rate (λ × lr_A where λ >> 1).

Benefits:
- Up to 2x training speedup
- 1-2% performance improvements
- Same computational cost as standard LoRA
"""

from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
import re


@dataclass
class LoRAPlusConfig:
    """Configuration for LoRA+ optimization."""

    # Base learning rate (applied to A matrices and non-LoRA params)
    base_lr: float = 2e-4

    # Learning rate ratio for B matrices (λ in the paper)
    # Recommended: 8-16 for most cases
    lora_plus_ratio: float = 16.0

    # Weight decay
    weight_decay: float = 0.01

    # Adam betas
    betas: Tuple[float, float] = (0.9, 0.999)

    # Epsilon
    eps: float = 1e-8

    @property
    def lr_a(self) -> float:
        """Learning rate for A matrices."""
        return self.base_lr

    @property
    def lr_b(self) -> float:
        """Learning rate for B matrices (λ × lr_A)."""
        return self.base_lr * self.lora_plus_ratio


def get_lora_param_groups(
    model,
    base_lr: float = 2e-4,
    lora_plus_ratio: float = 16.0,
    weight_decay: float = 0.01,
    exclude_bias: bool = True,
    exclude_layernorm: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create parameter groups for LoRA+ optimization.

    Separates parameters into groups:
    1. LoRA A matrices (down projection): base_lr
    2. LoRA B matrices (up projection): base_lr * lora_plus_ratio
    3. Other trainable parameters: base_lr

    Args:
        model: The model (should be a PEFT model with LoRA)
        base_lr: Base learning rate for A matrices
        lora_plus_ratio: Ratio for B matrix learning rate (λ)
        weight_decay: Weight decay (applied to non-bias, non-LN params)
        exclude_bias: Exclude bias from weight decay
        exclude_layernorm: Exclude LayerNorm from weight decay

    Returns:
        List of parameter group dicts for optimizer
    """
    # Patterns to identify LoRA parameters
    lora_a_pattern = re.compile(r".*lora_A.*")
    lora_b_pattern = re.compile(r".*lora_B.*")

    # Patterns for no weight decay
    no_decay_patterns = []
    if exclude_bias:
        no_decay_patterns.append(re.compile(r".*bias.*"))
    if exclude_layernorm:
        no_decay_patterns.extend([
            re.compile(r".*layer_?norm.*", re.IGNORECASE),
            re.compile(r".*ln_.*"),
        ])

    # Categorize parameters
    lora_a_params = []
    lora_a_params_no_decay = []
    lora_b_params = []
    lora_b_params_no_decay = []
    other_params = []
    other_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if should have no weight decay
        no_decay = any(p.match(name) for p in no_decay_patterns)

        # Categorize by type
        if lora_a_pattern.match(name):
            if no_decay:
                lora_a_params_no_decay.append(param)
            else:
                lora_a_params.append(param)
        elif lora_b_pattern.match(name):
            if no_decay:
                lora_b_params_no_decay.append(param)
            else:
                lora_b_params.append(param)
        else:
            if no_decay:
                other_params_no_decay.append(param)
            else:
                other_params.append(param)

    # Create parameter groups
    param_groups = []

    lr_b = base_lr * lora_plus_ratio

    # LoRA A parameters (with weight decay)
    if lora_a_params:
        param_groups.append({
            "params": lora_a_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "lora_A",
        })

    # LoRA A parameters (no weight decay)
    if lora_a_params_no_decay:
        param_groups.append({
            "params": lora_a_params_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "lora_A_no_decay",
        })

    # LoRA B parameters (with weight decay) - HIGHER LR
    if lora_b_params:
        param_groups.append({
            "params": lora_b_params,
            "lr": lr_b,
            "weight_decay": weight_decay,
            "name": "lora_B",
        })

    # LoRA B parameters (no weight decay) - HIGHER LR
    if lora_b_params_no_decay:
        param_groups.append({
            "params": lora_b_params_no_decay,
            "lr": lr_b,
            "weight_decay": 0.0,
            "name": "lora_B_no_decay",
        })

    # Other trainable parameters (with weight decay)
    if other_params:
        param_groups.append({
            "params": other_params,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "other",
        })

    # Other trainable parameters (no weight decay)
    if other_params_no_decay:
        param_groups.append({
            "params": other_params_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "other_no_decay",
        })

    return param_groups


def create_lora_plus_optimizer(
    model,
    config: Optional[LoRAPlusConfig] = None,
    base_lr: float = 2e-4,
    lora_plus_ratio: float = 16.0,
    weight_decay: float = 0.01,
    optimizer_cls: Optional[type] = None,
):
    """
    Create an optimizer configured for LoRA+ training.

    Args:
        model: The model to optimize
        config: LoRAPlusConfig (if provided, overrides other args)
        base_lr: Base learning rate
        lora_plus_ratio: Ratio for B matrix learning rate
        weight_decay: Weight decay
        optimizer_cls: Optimizer class (defaults to AdamW)

    Returns:
        Configured optimizer

    Example:
        >>> from tensafe.lora_best_practices import create_lora_plus_optimizer
        >>> optimizer = create_lora_plus_optimizer(
        ...     model,
        ...     base_lr=2e-4,
        ...     lora_plus_ratio=16.0,
        ... )
    """
    if config is not None:
        base_lr = config.base_lr
        lora_plus_ratio = config.lora_plus_ratio
        weight_decay = config.weight_decay

    # Get parameter groups
    param_groups = get_lora_param_groups(
        model,
        base_lr=base_lr,
        lora_plus_ratio=lora_plus_ratio,
        weight_decay=weight_decay,
    )

    # Default to AdamW
    if optimizer_cls is None:
        try:
            from torch.optim import AdamW
            optimizer_cls = AdamW
        except ImportError:
            raise ImportError("PyTorch is required for optimizer creation")

    # Create optimizer
    optimizer = optimizer_cls(param_groups)

    return optimizer


class LoRAPlusOptimizer:
    """
    Wrapper class for LoRA+ optimization.

    Provides a convenient interface for LoRA+ training with automatic
    parameter group creation and learning rate scheduling support.
    """

    def __init__(
        self,
        model,
        base_lr: float = 2e-4,
        lora_plus_ratio: float = 16.0,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        """
        Initialize LoRA+ optimizer.

        Args:
            model: The model to optimize
            base_lr: Base learning rate for A matrices
            lora_plus_ratio: Multiplier for B matrix learning rate
            weight_decay: Weight decay factor
            betas: Adam beta parameters
            eps: Adam epsilon
        """
        self.model = model
        self.base_lr = base_lr
        self.lora_plus_ratio = lora_plus_ratio
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps

        self._param_groups = get_lora_param_groups(
            model,
            base_lr=base_lr,
            lora_plus_ratio=lora_plus_ratio,
            weight_decay=weight_decay,
        )

        self._optimizer = None

    def get_optimizer(self, optimizer_cls=None):
        """Get the underlying optimizer."""
        if self._optimizer is None:
            if optimizer_cls is None:
                try:
                    from torch.optim import AdamW
                    optimizer_cls = AdamW
                except ImportError:
                    raise ImportError("PyTorch required")

            self._optimizer = optimizer_cls(
                self._param_groups,
                betas=self.betas,
                eps=self.eps,
            )

        return self._optimizer

    def get_param_groups(self) -> List[Dict]:
        """Get parameter groups for external optimizer creation."""
        return self._param_groups

    def get_lr_info(self) -> Dict[str, float]:
        """Get learning rate information for logging."""
        return {
            "base_lr": self.base_lr,
            "lora_a_lr": self.base_lr,
            "lora_b_lr": self.base_lr * self.lora_plus_ratio,
            "lora_plus_ratio": self.lora_plus_ratio,
        }

    def state_dict(self) -> Dict:
        """Get optimizer state dict."""
        if self._optimizer is not None:
            return self._optimizer.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict):
        """Load optimizer state dict."""
        if self._optimizer is not None:
            self._optimizer.load_state_dict(state_dict)


def analyze_lora_parameters(model) -> Dict[str, Any]:
    """
    Analyze LoRA parameters in a model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with parameter statistics
    """
    lora_a_count = 0
    lora_b_count = 0
    lora_a_params = 0
    lora_b_params = 0
    other_count = 0
    other_params = 0

    lora_a_pattern = re.compile(r".*lora_A.*")
    lora_b_pattern = re.compile(r".*lora_B.*")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        numel = param.numel()

        if lora_a_pattern.match(name):
            lora_a_count += 1
            lora_a_params += numel
        elif lora_b_pattern.match(name):
            lora_b_count += 1
            lora_b_params += numel
        else:
            other_count += 1
            other_params += numel

    total_params = lora_a_params + lora_b_params + other_params

    return {
        "lora_a_matrices": lora_a_count,
        "lora_a_parameters": lora_a_params,
        "lora_b_matrices": lora_b_count,
        "lora_b_parameters": lora_b_params,
        "other_trainable": other_count,
        "other_parameters": other_params,
        "total_trainable_parameters": total_params,
        "lora_fraction": (lora_a_params + lora_b_params) / max(total_params, 1),
    }


# Documentation
LORA_PLUS_EXPLANATION = """
LoRA+ Optimization Strategy
===========================

Standard LoRA uses the same learning rate for both A and B matrices:
    A: (r x in_features), initialized from N(0, σ²)
    B: (out_features x r), initialized to zeros

LoRA+ insight: The optimal learning rates are different!
    - B matrix should have a HIGHER learning rate
    - Specifically: lr_B = λ × lr_A where λ >> 1

Why this works:
1. A and B have different roles in the forward pass
2. B controls the "direction" of adaptation
3. A controls the "magnitude" of input projection
4. With large embedding dimensions, equal LRs lead to suboptimal learning

Recommended λ values:
    - λ = 16: Default, works well for most cases
    - λ = 8: More conservative, for sensitive training
    - λ = 32: Aggressive, may speed up training further

Example:
    # Standard LoRA (suboptimal)
    optimizer = AdamW(model.parameters(), lr=2e-4)

    # LoRA+ (optimal)
    param_groups = get_lora_param_groups(model, base_lr=2e-4, lora_plus_ratio=16)
    optimizer = AdamW(param_groups)

Results from paper:
    - Up to 2x speedup in training
    - 1-2% improvement in final performance
    - No additional computational cost
"""
