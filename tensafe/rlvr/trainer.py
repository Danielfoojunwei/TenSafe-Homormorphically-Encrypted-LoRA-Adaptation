"""
RLVR Trainer

High-level trainer class that orchestrates RLVR training.
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

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
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


class RLVRTrainer:
    """
    High-level RLVR trainer.

    Orchestrates the full RLVR training loop:
    1. Generate rollouts from current policy
    2. Compute rewards for generated responses
    3. Update policy using chosen RL algorithm

    Example usage:
        trainer = RLVRTrainer(
            training_client=tc,
            config=RLVRConfig(algorithm="reinforce"),
        )

        for prompts in prompt_loader:
            metrics = trainer.step(prompts)
            print(f"Step {metrics.step}: reward={metrics.mean_reward:.3f}")

        trainer.save_checkpoint("checkpoint.pt")
    """

    def __init__(
        self,
        training_client: Optional[TrainingClient] = None,
        config: Optional[RLVRConfig] = None,
        reward_fn: Optional[RewardFn] = None,
        algorithm: Optional[RLAlgorithm] = None,
        sampler: Optional[RolloutSampler] = None,
    ):
        """
        Initialize the RLVR trainer.

        Args:
            training_client: Client for training operations
            config: RLVR configuration
            reward_fn: Reward function (overrides config.reward_fn)
            algorithm: RL algorithm (overrides config.algorithm)
            sampler: Rollout sampler (created from client if not provided)
        """
        self.config = config or RLVRConfig()
        self.client = training_client

        # Set random seed
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
            # Use mock sampler for testing
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

        # Training state
        self._step = 0
        self._total_trajectories = 0
        self._total_tokens = 0
        self._metrics_history: List[TrainingMetrics] = []

    def _create_algorithm(self) -> RLAlgorithm:
        """Create the RL algorithm based on config."""
        if self.config.algorithm == "reinforce":
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
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

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

        # 3. Collect trajectories
        self.collector.collect(batch)

        # 4. Update policy
        update_start = time.time()
        update_result = self.algorithm.update(batch, self.client)
        time_update = (time.time() - update_start) * 1000

        # Update counters
        self._step += 1
        self._total_trajectories += len(batch)
        self._total_tokens += sum(t.num_response_tokens for t in batch)

        total_time = (time.time() - step_start) * 1000

        # Create metrics
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
        )

        self._metrics_history.append(metrics)

        return metrics

    def _compute_rewards(self, batch: TrajectoryBatch) -> None:
        """
        Compute rewards for all trajectories in the batch.

        Args:
            batch: The trajectory batch
        """
        for traj in batch.trajectories:
            # Compute reward
            reward = self.reward_fn(
                prompt=traj.prompt,
                response=traj.response,
                meta=traj.metadata,
            )

            # Scale reward
            reward = reward * self.config.reward_scale

            # Clip reward if configured
            if self.config.reward_clip is not None:
                reward = max(-self.config.reward_clip, min(self.config.reward_clip, reward))

            traj.reward = reward

    def evaluate(
        self,
        prompts: List[str],
        num_samples: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate current policy on given prompts.

        Args:
            prompts: Prompts to evaluate on
            num_samples: Number of samples per prompt

        Returns:
            Evaluation metrics
        """
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
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "step": self._step,
            "total_trajectories": self._total_trajectories,
            "total_tokens": self._total_tokens,
            "buffer_size": len(self.buffer),
            "algorithm_step": self.algorithm.step,
            "config": self.config.to_dict(),
        }

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get history of training metrics."""
        return [m.to_dict() for m in self._metrics_history]

    def save_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint

        Returns:
            Checkpoint dictionary
        """
        checkpoint = {
            "step": self._step,
            "total_trajectories": self._total_trajectories,
            "total_tokens": self._total_tokens,
            "algorithm_state": self.algorithm.get_state(),
            "config": self.config.to_dict(),
            "metrics_history": self.get_metrics_history(),
        }

        # Save to file
        import json

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")

        return checkpoint

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        import json

        with open(path) as f:
            checkpoint = json.load(f)

        self._step = checkpoint["step"]
        self._total_trajectories = checkpoint["total_trajectories"]
        self._total_tokens = checkpoint["total_tokens"]
        self.algorithm.load_state(checkpoint["algorithm_state"])

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

            # Training step
            metrics = self.step(prompts)
            all_metrics.append(metrics)

            # Logging
            if step_num % self.config.log_interval == 0:
                logger.info(
                    f"Step {metrics.step}: "
                    f"reward={metrics.mean_reward:.3f} "
                    f"loss={metrics.policy_loss:.4f} "
                    f"entropy={metrics.entropy:.3f}"
                )

            # Callback
            if callback is not None:
                callback(metrics)

            # Save checkpoint
            if self.config.save_interval > 0 and step_num % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{step_num}.json")

        return all_metrics
