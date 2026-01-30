"""
TenSafe RLVR (Reinforcement Learning with Verifiable Rewards) Module

This module provides RL fine-tuning capabilities for language models using
LoRA adapters. It supports:

- Rollout sampling from the current policy
- Pluggable reward functions
- REINFORCE and PPO algorithms
- Trajectory storage and replay

Example usage:
    from tensafe.rlvr import RLVRTrainer, resolve_reward
    from tensafe.rlvr.algorithms import REINFORCE

    # Create reward function
    reward_fn = resolve_reward("my_rewards:keyword_reward")

    # Create trainer
    trainer = RLVRTrainer(
        training_client=tc,
        reward_fn=reward_fn,
        algorithm=REINFORCE(lr=1e-5),
    )

    # Training loop
    for batch in prompt_loader:
        metrics = trainer.step(batch)
        print(f"Reward: {metrics['mean_reward']}")
"""

from .rollout import MockRolloutSampler, RolloutSampler, Trajectory, TrajectoryBatch
from .reward import RewardFn, resolve_reward, register_reward, get_registered_rewards
from .buffers import TrajectoryBuffer
from .trainer import RLVRTrainer
from .config import RLVRConfig
from .algorithms import PPO, PPOConfig, REINFORCE, REINFORCEConfig, RLAlgorithm

__all__ = [
    # Rollout
    "RolloutSampler",
    "MockRolloutSampler",
    "Trajectory",
    "TrajectoryBatch",
    # Reward
    "RewardFn",
    "resolve_reward",
    "register_reward",
    "get_registered_rewards",
    # Buffers
    "TrajectoryBuffer",
    # Trainer
    "RLVRTrainer",
    # Config
    "RLVRConfig",
    # Algorithms
    "RLAlgorithm",
    "REINFORCE",
    "REINFORCEConfig",
    "PPO",
    "PPOConfig",
]
