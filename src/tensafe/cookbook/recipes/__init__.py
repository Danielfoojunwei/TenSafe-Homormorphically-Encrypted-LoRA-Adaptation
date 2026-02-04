"""
TenSafe Cookbook Recipes - Complete training examples for various use cases.

This module provides production-ready training recipes that demonstrate
how to use the TenSafe API for different fine-tuning scenarios.

Available Recipes:

1. Chat Supervised Learning (chat_sl)
   - Fine-tune on conversational datasets
   - Supports multi-turn dialogues
   - Uses standard cross-entropy loss

2. Math Reasoning (math_rl)
   - Improve LLM performance on mathematical problems
   - Uses GRPO-style reward centering
   - Supports GSM8K and custom datasets

3. Preference Learning (preference)
   - Three-stage RLHF pipeline
   - Stage 1: Supervised fine-tuning
   - Stage 2: Reward model training
   - Stage 3: RL optimization (PPO/DPO)

4. Tool Use (tool_use)
   - Train models to use tools/APIs
   - Supports function calling patterns
   - Retrieval tool integration

5. Prompt Distillation (distillation)
   - Internalize complex instructions into model weights
   - Reduce inference-time prompt length
   - Maintain instruction-following quality

6. Multi-Agent Training (multi_agent)
   - Competitive/cooperative multi-agent setups
   - Self-play training
   - Arena-style evaluation

Each recipe provides:
- Configuration classes
- Training loop implementations
- Dataset builders
- Evaluation utilities
"""

from .chat_sl import ChatSLConfig, ChatSLTrainer, run_chat_sl
from .distillation import DistillationConfig, DistillationTrainer, run_distillation
from .math_rl import MathRLConfig, MathRLTrainer, run_math_rl
from .multi_agent import MultiAgentConfig, MultiAgentTrainer, run_multi_agent
from .preference import (
    DPOConfig,
    PreferenceConfig,
    PreferenceTrainer,
    RewardModelConfig,
    run_preference_learning,
)
from .tool_use import ToolUseConfig, ToolUseTrainer, run_tool_use

__all__ = [
    # Chat SL
    "ChatSLConfig",
    "ChatSLTrainer",
    "run_chat_sl",
    # Math RL
    "MathRLConfig",
    "MathRLTrainer",
    "run_math_rl",
    # Preference Learning
    "PreferenceConfig",
    "RewardModelConfig",
    "DPOConfig",
    "PreferenceTrainer",
    "run_preference_learning",
    # Tool Use
    "ToolUseConfig",
    "ToolUseTrainer",
    "run_tool_use",
    # Distillation
    "DistillationConfig",
    "DistillationTrainer",
    "run_distillation",
    # Multi-Agent
    "MultiAgentConfig",
    "MultiAgentTrainer",
    "run_multi_agent",
]
