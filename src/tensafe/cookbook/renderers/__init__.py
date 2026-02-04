"""
Renderers for converting between tokens and structured chat messages.

Renderers handle the token-level formatting and prompt building for different
language models, with particular focus on:
- Correspondence between generation prompts and supervised examples
- Response parsing from generated tokens
- Stop sequence handling

Each renderer is model-specific to handle the unique special tokens and
formatting conventions used by different model families.

Available Renderers:
- Llama3Renderer: For Meta's Llama 3.x models
- DeepSeekV3Renderer: For DeepSeek V3 models
- Qwen3Renderer: For Alibaba's Qwen 3 models
- KimiK2Renderer: For Moonshot's Kimi K2 model
- RoleColonRenderer: Generic role:content format

Usage:
    from tensafe.cookbook.renderers import Llama3Renderer, get_renderer
    from tensafe.cookbook import get_tokenizer

    tokenizer = get_tokenizer("meta-llama/Llama-3.1-8B")
    renderer = Llama3Renderer(tokenizer)

    # Build generation prompt
    messages = [
        {"role": "user", "content": "Hello!"},
    ]
    tokens = renderer.build_generation_prompt(messages)

    # Build supervised example
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    tokens, weights = renderer.build_supervised_example(messages)
"""

from .base import (
    Message,
    MessageChunk,
    Renderer,
    RenderContext,
    RenderedMessage,
    StopCondition,
    SupervisedWeightMode,
)
from .deepseek_v3 import DeepSeekV3Renderer
from .kimi_k2 import KimiK2Renderer
from .llama3 import Llama3Renderer
from .qwen3 import Qwen3Renderer
from .role_colon import RoleColonRenderer

# Registry of available renderers
RENDERER_REGISTRY: dict[str, type[Renderer]] = {
    "llama3": Llama3Renderer,
    "llama3-instruct": Llama3Renderer,
    "deepseek-v3": DeepSeekV3Renderer,
    "deepseek": DeepSeekV3Renderer,
    "qwen3": Qwen3Renderer,
    "qwen": Qwen3Renderer,
    "kimi-k2": KimiK2Renderer,
    "kimi": KimiK2Renderer,
    "role-colon": RoleColonRenderer,
    "generic": RoleColonRenderer,
}


def get_renderer(renderer_name: str, tokenizer) -> Renderer:
    """
    Get a renderer instance by name.

    Args:
        renderer_name: Name of the renderer (e.g., "llama3", "deepseek-v3")
        tokenizer: Tokenizer instance for the model

    Returns:
        Renderer instance

    Raises:
        ValueError: If renderer_name is not recognized
    """
    name_lower = renderer_name.lower().replace("_", "-")

    if name_lower not in RENDERER_REGISTRY:
        available = ", ".join(sorted(RENDERER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown renderer: {renderer_name}. Available: {available}"
        )

    renderer_cls = RENDERER_REGISTRY[name_lower]
    return renderer_cls(tokenizer)


__all__ = [
    # Base types
    "Renderer",
    "Message",
    "MessageChunk",
    "RenderedMessage",
    "RenderContext",
    "StopCondition",
    "SupervisedWeightMode",
    # Model-specific renderers
    "Llama3Renderer",
    "DeepSeekV3Renderer",
    "Qwen3Renderer",
    "KimiK2Renderer",
    "RoleColonRenderer",
    # Factory
    "get_renderer",
    "RENDERER_REGISTRY",
]
