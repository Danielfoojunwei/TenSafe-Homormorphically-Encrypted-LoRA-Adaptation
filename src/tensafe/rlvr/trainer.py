"""
RLVR Trainer

High-level trainer class that orchestrates RLVR training.
Supports all five RL algorithms (REINFORCE, REINFORCE++, PPO, GRPO, RLOO),
pluggable advantage estimators, pluggable policy losses, off-policy
correction, async rollout generation, micro-batch gradient accumulation,
and environment-based reward computation.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol

from .config import RLVRConfig
from .rollout import MockRolloutSampler, RolloutSampler, Trajectory, TrajectoryBatch
from .reward import RewardFn, resolve_reward
from .buffers import RolloutCollector, TrajectoryBuffer
from .algorithms.base import AlgorithmConfig, RLAlgorithm, UpdateResult
from .algorithms.reinforce import REINFORCE, REINFORCEConfig
from .algorithms.reinforce_pp import REINFORCEPP, REINFORCEPPConfig
from .algorithms.ppo import PPO, PPOConfig
from .algorithms.grpo import GRPO, GRPOConfig
from .algorithms.rloo import RLOO, RLOOConfig

logger = logging.getLogger(__name__)


class TrainingClient(Protocol):
    """Protocol for training clients."""

    def forward_backward(self, batch: Dict[str, Any]) -> Any:
        ...

    def optim_step(self, apply_dp_noise: bool = True) -> Any:
        ...

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Any:
        ...

    @property
    def step(self) -> int:
        ...


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    step: int
    mean_reward: float
    std_reward: float
    max_reward: float
    min_reward: float
    policy_loss: float
    entropy: float
    kl_div: float
    grad_norm: float
    trajectories_collected: int
    tokens_generated: int
    time_rollout_ms: float
    time_reward_ms: float
    time_update_ms: float
    total_time_ms: float

    # Extended metrics (optional)
    advantage_estimator: Optional[str] = None
    policy_loss_type: Optional[str] = None
    off_policy_metadata: Optional[Dict[str, float]] = None
    micro_batch_metadata: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        d = {
            "step": self.step,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "max_reward": self.max_reward,
            "min_reward": self.min_reward,
            "policy_loss": self.policy_loss,
            "entropy": self.entropy,
            "kl_div": self.kl_div,
            "grad_norm": self.grad_norm,
            "trajectories": self.trajectories_collected,
            "tokens": self.tokens_generated,
            "time_rollout_ms": self.time_rollout_ms,
            "time_reward_ms": self.time_reward_ms,
            "time_update_ms": self.time_update_ms,
            "total_time_ms": self.total_time_ms,
        }
        if self.advantage_estimator:
            d["advantage_estimator"] = self.advantage_estimator
        if self.policy_loss_type:
            d["policy_loss_type"] = self.policy_loss_type
        return d


class RLVRTrainer:
    """
    High-level RLVR trainer.

    Orchestrates the full RLVR training loop:
    1. Generate rollouts from current policy
    2. Compute rewards for generated responses
    3. Compute advantages (via pluggable estimator)
    4. Update policy using chosen RL algorithm
    5. Optionally apply off-policy correction for stale data
    6. Optionally use micro-batch gradient accumulation

    Supports all five algorithms:
    - reinforce: Vanilla REINFORCE with moving-average baseline
    - reinforce_pp: REINFORCE++ with temporal discounting and whitening
    - ppo: PPO with clipped surrogate objective
    - grpo: Group Relative Policy Optimization (critic-free)
    - rloo: Leave-One-Out baseline estimation
    """

    def __init__(
        self,
        training_client: Optional[TrainingClient] = None,
        config: Optional[RLVRConfig] = None,
        reward_fn: Optional[RewardFn] = None,
        algorithm: Optional[RLAlgorithm] = None,
        sampler: Optional[RolloutSampler] = None,
    ):
        self.config = config or RLVRConfig()
        self.client = training_client

        random.seed(self.config.seed)

        # Initialize reward function
        if reward_fn is not None:
            self.reward_fn = reward_fn
        else:
            self.reward_fn = resolve_reward(
                self.config.reward_fn,
                **self.config.reward_kwargs,
            )

        # Initialize algorithm
        if algorithm is not None:
            self.algorithm = algorithm
        else:
            self.algorithm = self._create_algorithm()

        # Initialize sampler
        if sampler is not None:
            self.sampler = sampler
        elif training_client is not None:
            self.sampler = RolloutSampler(
                client=training_client,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )
        else:
            self.sampler = MockRolloutSampler(
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                seed=self.config.seed,
            )

        # Initialize buffer
        self.buffer = TrajectoryBuffer(
            max_size=self.config.buffer_size,
            seed=self.config.seed,
        )

        # Initialize collector
        self.collector = RolloutCollector(buffer=self.buffer, seed=self.config.seed)

        # Initialize async rollout orchestrator (if enabled)
        self._async_orchestrator = None
        if self.config.async_rollout_enabled:
            from .async_rollout import AsyncRolloutConfig, AsyncRolloutOrchestrator

            async_config = AsyncRolloutConfig(
                max_staleness_steps=self.config.async_max_staleness_steps,
                max_buffer_size=self.config.async_max_buffer_size,
                max_generation_slots=self.config.async_max_generation_slots,
                min_batch_size=self.config.async_min_batch_size,
                batch_timeout=self.config.async_batch_timeout,
                track_consumed_uids=self.config.async_track_consumed_uids,
            )
            self._async_orchestrator = AsyncRolloutOrchestrator(
                config=async_config,
                num_workers=self.config.async_num_workers,
            )

        # Initialize micro-batch accumulator (if enabled)
        self._micro_batcher = None
        if self.config.micro_batch_size > 0:
            from .micro_batch import MicroBatchConfig, GradientAccumulator, DPAwareMicroBatcher

            mb_config = MicroBatchConfig(
                micro_batch_size=self.config.micro_batch_size,
                effective_batch_size=self.config.effective_batch_size,
                scale_gradients=self.config.micro_batch_scale_gradients,
                dp_enabled=self.config.dp_micro_batch_enabled,
                dp_max_grad_norm=self.config.dp_max_grad_norm,
                dp_noise_multiplier=self.config.dp_noise_multiplier,
                max_grad_norm=self.config.max_grad_norm,
            )
            if self.config.dp_micro_batch_enabled:
                self._micro_batcher = DPAwareMicroBatcher(mb_config)
            else:
                self._micro_batcher = GradientAccumulator(mb_config)

        # Initialize environment (if configured)
        self._environment = None
        if self.config.environment is not None:
            from .env import make_env

            self._environment = make_env(
                self.config.environment,
                **self.config.environment_kwargs,
            )

        # Training state
        self._step = 0
        self._total_trajectories = 0
        self._total_tokens = 0
        self._metrics_history: List[TrainingMetrics] = []

    def _create_algorithm(self) -> RLAlgorithm:
        """Create the RL algorithm based on config."""
        algo = self.config.algorithm

        if algo == "reinforce":
            algo_config = REINFORCEConfig(
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                max_grad_norm=self.config.max_grad_norm,
                use_baseline=self.config.use_baseline,
                baseline_decay=self.config.baseline_decay,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                normalize_advantages=self.config.normalize_advantages,
            )
            return REINFORCE(algo_config)

        elif algo == "reinforce_pp":
            algo_config = REINFORCEPPConfig(
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                max_grad_norm=self.config.max_grad_norm,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                normalize_advantages=self.config.normalize_advantages,
                temporal_whitening=self.config.reinforce_pp_temporal_whitening,
            )
            return REINFORCEPP(algo_config)

        elif algo == "ppo":
            algo_config = PPOConfig(
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                max_grad_norm=self.config.max_grad_norm,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                normalize_advantages=self.config.normalize_advantages,
                clip_range=self.config.ppo_clip_eps,
                ppo_epochs=self.config.ppo_epochs,
                minibatch_size=self.config.ppo_minibatch_size,
                vf_coef=self.config.vf_coef,
            )
            return PPO(algo_config)

        elif algo == "grpo":
            algo_config = GRPOConfig(
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                max_grad_norm=self.config.max_grad_norm,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                normalize_advantages=self.config.normalize_advantages,
                num_samples_per_prompt=self.config.num_samples_per_prompt,
                clip_range=self.config.ppo_clip_eps,
                normalize_within_group=self.config.grpo_normalize_within_group,
                normalize_batch=self.config.grpo_normalize_batch,
                min_group_size=self.config.grpo_min_group_size,
            )
            return GRPO(algo_config)

        elif algo == "rloo":
            algo_config = RLOOConfig(
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                max_grad_norm=self.config.max_grad_norm,
                entropy_coef=self.config.entropy_coef,
                kl_coef=self.config.kl_coef,
                normalize_advantages=self.config.normalize_advantages,
                clip_range=self.config.ppo_clip_eps,
                fallback_to_batch_mean=self.config.rloo_fallback_to_batch_mean,
            )
            return RLOO(algo_config)

        else:
            raise ValueError(
                f"Unknown algorithm: {algo}. "
                f"Supported: reinforce, reinforce_pp, ppo, grpo, rloo"
            )

    def step(self, prompts: List[str]) -> TrainingMetrics:
        """
        Perform one training step.

        Args:
            prompts: List of prompts for rollout generation

        Returns:
            TrainingMetrics for this step
        """
        step_start = time.time()

        # 1. Generate rollouts
        rollout_start = time.time()
        batch = self.sampler.generate_trajectories(prompts)
        time_rollout = (time.time() - rollout_start) * 1000

        # 2. Compute rewards
        reward_start = time.time()
        self._compute_rewards(batch)
        time_reward = (time.time() - reward_start) * 1000

        # 3. Apply advantage estimator override (if configured)
        if self.config.advantage_estimator is not None:
            from .advantages import apply_advantage

            apply_advantage(
                self.config.advantage_estimator,
                batch,
                normalize=self.config.normalize_advantages,
                gamma=self.config.gamma,
            )

        # 4. Collect trajectories
        self.collector.collect(batch)

        # 5. Apply off-policy correction (if enabled)
        off_policy_meta = None
        if self.config.off_policy_enabled:
            off_policy_meta = self._apply_off_policy_correction(batch)

        # 6. Update policy
        update_start = time.time()
        update_result = self.algorithm.update(batch, self.client)
        time_update = (time.time() - update_start) * 1000

        # 7. Advance async orchestrator (if enabled)
        if self._async_orchestrator is not None:
            self._async_orchestrator.advance_step()

        # Update counters
        self._step += 1
        self._total_trajectories += len(batch)
        self._total_tokens += sum(t.num_response_tokens for t in batch)

        total_time = (time.time() - step_start) * 1000

        metrics = TrainingMetrics(
            step=self._step,
            mean_reward=batch.mean_reward,
            std_reward=batch.std_reward,
            max_reward=max(t.reward for t in batch),
            min_reward=min(t.reward for t in batch),
            policy_loss=update_result.policy_loss,
            entropy=update_result.entropy,
            kl_div=update_result.kl_div,
            grad_norm=update_result.grad_norm,
            trajectories_collected=len(batch),
            tokens_generated=sum(t.num_response_tokens for t in batch),
            time_rollout_ms=time_rollout,
            time_reward_ms=time_reward,
            time_update_ms=time_update,
            total_time_ms=total_time,
            advantage_estimator=self.config.advantage_estimator,
            policy_loss_type=self.config.policy_loss,
            off_policy_metadata=off_policy_meta,
        )

        self._metrics_history.append(metrics)
        return metrics

    def _compute_rewards(self, batch: TrajectoryBatch) -> None:
        """Compute rewards for all trajectories in the batch."""
        for traj in batch.trajectories:
            reward = self.reward_fn(
                prompt=traj.prompt,
                response=traj.response,
                meta=traj.metadata,
            )

            reward = reward * self.config.reward_scale

            if self.config.reward_clip is not None:
                reward = max(-self.config.reward_clip, min(self.config.reward_clip, reward))

            traj.reward = reward

    def _apply_off_policy_correction(
        self, batch: TrajectoryBatch
    ) -> Optional[Dict[str, float]]:
        """Apply off-policy correction to the batch."""
        from .off_policy import OffPolicyConfig, apply_off_policy_correction

        config = OffPolicyConfig(
            tis_clip_min=self.config.off_policy_tis_clip_min,
            tis_clip_max=self.config.off_policy_tis_clip_max,
            seq_ratio_max=self.config.off_policy_seq_ratio_max,
            seq_ratio_min=self.config.off_policy_seq_ratio_min,
            outlier_z_threshold=self.config.off_policy_outlier_z_threshold,
            drop_outlier_sequences=self.config.off_policy_drop_outlier_sequences,
            staleness_decay=self.config.off_policy_staleness_decay,
        )

        logprobs = [t.logprobs for t in batch.trajectories]
        # Use current logprobs as both current and old (in sync mode)
        # In async mode, old logprobs would come from generation time
        old_logprobs = [
            t.metadata.get("old_logprobs", t.logprobs)
            for t in batch.trajectories
        ]

        generation_steps = [
            t.metadata.get("generation_step") for t in batch.trajectories
        ]
        has_gen_steps = all(gs is not None for gs in generation_steps)

        result = apply_off_policy_correction(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            config=config,
            generation_steps=generation_steps if has_gen_steps else None,
            current_step=self._step,
        )

        # Apply sequence mask: drop trajectories with extreme staleness
        if any(not m for m in result.sequence_mask):
            kept = [
                t
                for t, m in zip(batch.trajectories, result.sequence_mask)
                if m
            ]
            if kept:
                batch.trajectories[:] = kept
                # Reset cached stats
                batch._mean_reward = None
                batch._std_reward = None

        return result.metadata

    def evaluate(
        self,
        prompts: List[str],
        num_samples: int = 1,
    ) -> Dict[str, float]:
        """Evaluate current policy on given prompts."""
        all_rewards = []
        all_lengths = []

        for _ in range(num_samples):
            batch = self.sampler.generate_trajectories(prompts)
            self._compute_rewards(batch)

            for traj in batch.trajectories:
                all_rewards.append(traj.reward)
                all_lengths.append(traj.num_response_tokens)

        return {
            "eval_mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "eval_std_reward": self._std(all_rewards),
            "eval_mean_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0.0,
            "eval_samples": len(all_rewards),
        }

    def _std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = {
            "step": self._step,
            "total_trajectories": self._total_trajectories,
            "total_tokens": self._total_tokens,
            "buffer_size": len(self.buffer),
            "algorithm": self.config.algorithm,
            "algorithm_step": self.algorithm.step,
            "config": self.config.to_dict(),
        }

        if self._async_orchestrator is not None:
            stats["async_rollout"] = self._async_orchestrator.get_stats()

        if self._micro_batcher is not None:
            stats["micro_batch"] = self._micro_batcher.get_stats()

        return stats

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get history of training metrics."""
        return [m.to_dict() for m in self._metrics_history]

    def save_checkpoint(self, path: str) -> Dict[str, Any]:
        """Save training checkpoint."""
        checkpoint = {
            "step": self._step,
            "total_trajectories": self._total_trajectories,
            "total_tokens": self._total_tokens,
            "algorithm_state": self.algorithm.get_state(),
            "config": self.config.to_dict(),
            "metrics_history": self.get_metrics_history(),
        }

        # Save async state if applicable
        if self._async_orchestrator is not None:
            checkpoint["async_state"] = self._async_orchestrator.save_state()

        import json
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")
        return checkpoint

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        import json
        with open(path) as f:
            checkpoint = json.load(f)

        self._step = checkpoint["step"]
        self._total_trajectories = checkpoint["total_trajectories"]
        self._total_tokens = checkpoint["total_tokens"]
        self.algorithm.load_state(checkpoint["algorithm_state"])

        # Restore async state if applicable
        if self._async_orchestrator is not None and "async_state" in checkpoint:
            self._async_orchestrator.load_state(checkpoint["async_state"])

        logger.info(f"Loaded checkpoint from {path} (step {self._step})")

    def train(
        self,
        prompt_iterator: Iterator[List[str]],
        total_steps: Optional[int] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
    ) -> List[TrainingMetrics]:
        """
        Run full training loop.

        Args:
            prompt_iterator: Iterator yielding batches of prompts
            total_steps: Override config.total_steps
            callback: Optional callback called after each step

        Returns:
            List of training metrics for all steps
        """
        total_steps = total_steps or self.config.total_steps
        all_metrics = []

        for step_num, prompts in enumerate(prompt_iterator):
            if step_num >= total_steps:
                break

            metrics = self.step(prompts)
            all_metrics.append(metrics)

            if step_num % self.config.log_interval == 0:
                logger.info(
                    f"Step {metrics.step}: "
                    f"reward={metrics.mean_reward:.3f} "
                    f"loss={metrics.policy_loss:.4f} "
                    f"entropy={metrics.entropy:.3f} "
                    f"[{self.config.algorithm}]"
                )

            if callback is not None:
                callback(metrics)

            if self.config.save_interval > 0 and step_num % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{step_num}.json")

        return all_metrics
