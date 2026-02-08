"""
RLVR Algorithms

This module provides RL algorithms for fine-tuning language models:
- REINFORCE: Basic policy gradient algorithm
- REINFORCE++: Enhanced REINFORCE with temporal discounting and whitening
- PPO: Proximal Policy Optimization with clipped objective
- GRPO: Group Relative Policy Optimization (critic-free, per-group normalization)
- RLOO: Reinforcement Learning with Leave-One-Out baseline
"""

from .base import AlgorithmConfig, RLAlgorithm, MockRLAlgorithm, UpdateResult
from .reinforce import REINFORCE, REINFORCEConfig, REINFORCEWithBaseline
from .reinforce_pp import REINFORCEPP, REINFORCEPPConfig
from .ppo import PPO, PPOConfig, PPOWithValueFunction
from .grpo import GRPO, GRPOConfig
from .rloo import RLOO, RLOOConfig

__all__ = [
    # Base
    "RLAlgorithm",
    "AlgorithmConfig",
    "MockRLAlgorithm",
    "UpdateResult",
    # REINFORCE
    "REINFORCE",
    "REINFORCEConfig",
    "REINFORCEWithBaseline",
    # REINFORCE++
    "REINFORCEPP",
    "REINFORCEPPConfig",
    # PPO
    "PPO",
    "PPOConfig",
    "PPOWithValueFunction",
    # GRPO
    "GRPO",
    "GRPOConfig",
    # RLOO
    "RLOO",
    "RLOOConfig",
]
