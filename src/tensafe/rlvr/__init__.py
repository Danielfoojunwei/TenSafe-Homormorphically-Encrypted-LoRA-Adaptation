"""
TenSafe RLVR (Reinforcement Learning with Verifiable Rewards) Module

This module provides RL fine-tuning capabilities for language models using
LoRA adapters. It supports:

- Rollout sampling from the current policy
- Pluggable reward functions
- Five RL algorithms: REINFORCE, REINFORCE++, PPO, GRPO, RLOO
- Pluggable advantage estimators (baseline, GRPO, RLOO, REINFORCE++, GAE)
- Pluggable policy losses (PPO clip, GSPO, SAPO, CISPO, Clip-Cov, KL-Cov, etc.)
- Off-policy correction for async training (TIS ratios, outlier masking)
- Async rollout buffer with staleness control for HE-LoRA latency hiding
- Lightweight environment protocol with RewardFn auto-wrapper
- Micro-batch gradient accumulation with DP-SGD coordination
- Trajectory storage and replay

Example usage:
    from tensafe.rlvr import RLVRTrainer, resolve_reward
    from tensafe.rlvr.algorithms import GRPO, GRPOConfig

    # Create reward function
    reward_fn = resolve_reward("my_rewards:keyword_reward")

    # Create trainer with GRPO (critic-free, ideal for HE-LoRA)
    trainer = RLVRTrainer(
        training_client=tc,
        reward_fn=reward_fn,
        algorithm=GRPO(GRPOConfig(num_samples_per_prompt=5)),
    )

    # Training loop
    for batch in prompt_loader:
        metrics = trainer.step(batch)
        print(f"Reward: {metrics['mean_reward']}")
"""

from .rollout import MockRolloutSampler, RolloutSampler, Trajectory, TrajectoryBatch
from .reward import RewardFn, resolve_reward, register_reward, get_registered_rewards
from .buffers import TrajectoryBuffer, PrioritizedTrajectoryBuffer
from .trainer import RLVRTrainer
from .config import RLVRConfig
from .algorithms import (
    PPO, PPOConfig, PPOWithValueFunction,
    REINFORCE, REINFORCEConfig, REINFORCEWithBaseline,
    REINFORCEPP, REINFORCEPPConfig,
    GRPO, GRPOConfig,
    RLOO, RLOOConfig,
    RLAlgorithm, AlgorithmConfig, MockRLAlgorithm, UpdateResult,
)
from .advantages import (
    register_advantage, resolve_advantage, list_advantage_estimators,
    apply_advantage, AdvantageResult,
)
from .policy_losses import (
    register_policy_loss, resolve_policy_loss, list_policy_losses,
    PolicyLossInput, PolicyLossResult,
)
from .off_policy import (
    OffPolicyConfig, CorrectionResult,
    apply_off_policy_correction,
    compute_token_tis_ratios, compute_sequence_tis_ratios,
    mask_outlier_tokens, mask_outlier_sequences,
    compute_staleness_weights,
)
from .env import (
    Environment, Observation, StepResult,
    SingleTurnEnv, MultiTurnEnv, BatchEnvRunner,
    RewardShapingWrapper, TurnLimitWrapper,
    register_env, make_env, list_envs, wrap_reward_fn,
)
from .async_rollout import (
    AsyncRolloutConfig, AsyncRolloutBuffer,
    AsyncGenerationWorker, AsyncRolloutOrchestrator,
    StalenessManager,
)
from .micro_batch import (
    MicroBatchConfig, GradientAccumulator,
    MicroBatchContext, DPAwareMicroBatcher,
)

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
    "PrioritizedTrajectoryBuffer",
    # Trainer
    "RLVRTrainer",
    # Config
    "RLVRConfig",
    # Algorithms
    "RLAlgorithm",
    "AlgorithmConfig",
    "MockRLAlgorithm",
    "UpdateResult",
    "REINFORCE",
    "REINFORCEConfig",
    "REINFORCEWithBaseline",
    "REINFORCEPP",
    "REINFORCEPPConfig",
    "PPO",
    "PPOConfig",
    "PPOWithValueFunction",
    "GRPO",
    "GRPOConfig",
    "RLOO",
    "RLOOConfig",
    # Advantage Estimators
    "register_advantage",
    "resolve_advantage",
    "list_advantage_estimators",
    "apply_advantage",
    "AdvantageResult",
    # Policy Losses
    "register_policy_loss",
    "resolve_policy_loss",
    "list_policy_losses",
    "PolicyLossInput",
    "PolicyLossResult",
    # Off-Policy Correction
    "OffPolicyConfig",
    "CorrectionResult",
    "apply_off_policy_correction",
    "compute_token_tis_ratios",
    "compute_sequence_tis_ratios",
    "mask_outlier_tokens",
    "mask_outlier_sequences",
    "compute_staleness_weights",
    # Environment
    "Environment",
    "Observation",
    "StepResult",
    "SingleTurnEnv",
    "MultiTurnEnv",
    "BatchEnvRunner",
    "RewardShapingWrapper",
    "TurnLimitWrapper",
    "register_env",
    "make_env",
    "list_envs",
    "wrap_reward_fn",
    # Async Rollout
    "AsyncRolloutConfig",
    "AsyncRolloutBuffer",
    "AsyncGenerationWorker",
    "AsyncRolloutOrchestrator",
    "StalenessManager",
    # Micro-Batch
    "MicroBatchConfig",
    "GradientAccumulator",
    "MicroBatchContext",
    "DPAwareMicroBatcher",
]
