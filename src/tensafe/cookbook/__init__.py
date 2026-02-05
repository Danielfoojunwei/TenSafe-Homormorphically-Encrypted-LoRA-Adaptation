"""
TenSafe Cookbook - Practical examples and utilities for fine-tuning language models.

This cookbook provides higher-level abstractions and example implementations built
on the TenSafe training API, inspired by the Tinker Cookbook patterns.

Key Components:
- Renderers: Convert between tokens and structured chat messages
- Completers: Token and message-level sampling abstractions
- Hyperparameter utilities: LoRA configuration and learning rate scaling
- Model info: Model metadata and recommended renderers
- Recipes: Complete training examples for various use cases

Use Cases:
- Chat supervised learning
- Math reasoning with RL
- Preference learning (RLHF pipeline)
- Tool use training
- Prompt distillation
- Multi-agent training
"""

__version__ = "0.1.0"

from .completers import (
    MessageCompleter,
    TinkerMessageCompleter,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from .hyperparam_utils import (
    get_lora_lr,
    get_lora_lr_multiplier,
    get_lora_lr_over_full_finetune_lr,
    get_lora_param_count,
)
from .model_info import (
    ModelAttributes,
    get_model_attributes,
    get_recommended_renderer_name,
    get_recommended_renderer_names,
)
from .tokenizer_utils import Tokenizer, get_tokenizer

__all__ = [
    # Completers
    "TokenCompleter",
    "MessageCompleter",
    "TinkerTokenCompleter",
    "TinkerMessageCompleter",
    "TokensWithLogprobs",
    # Hyperparameters
    "get_lora_lr",
    "get_lora_lr_multiplier",
    "get_lora_lr_over_full_finetune_lr",
    "get_lora_param_count",
    # Model info
    "ModelAttributes",
    "get_model_attributes",
    "get_recommended_renderer_name",
    "get_recommended_renderer_names",
    # Tokenizer
    "Tokenizer",
    "get_tokenizer",
]
