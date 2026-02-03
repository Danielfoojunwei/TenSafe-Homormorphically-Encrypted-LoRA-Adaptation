"""
RLVR Trajectory Buffers

Provides storage for trajectories collected during RLVR training.
Supports mini-batch sampling and experience replay.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, Optional

from .rollout import Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


@dataclass
class BufferStats:
    """Statistics about the trajectory buffer."""

    total_trajectories: int = 0
    total_tokens: int = 0
    mean_reward: float = 0.0
    max_reward: float = float("-inf")
    min_reward: float = float("inf")
    mean_response_length: int = 0


class TrajectoryBuffer:
    """
    Buffer for storing and sampling trajectories.

    Supports:
    - Fixed-size circular buffer
    - Random sampling for mini-batches
    - Statistics tracking
    - Filtering by reward threshold
    """

    def __init__(
        self,
        max_size: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Initialize the trajectory buffer.

        Args:
            max_size: Maximum number of trajectories to store
            seed: Random seed for sampling
        """
        self.max_size = max_size
        self.seed = seed

        self._buffer: Deque[Trajectory] = deque(maxlen=max_size)
        self._rng = random.Random(seed)

        # Running statistics
        self._total_added = 0
        self._reward_sum = 0.0
        self._token_sum = 0

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Trajectory]:
        return iter(self._buffer)

    def add(self, trajectory: Trajectory) -> None:
        """
        Add a single trajectory to the buffer.

        Args:
            trajectory: The trajectory to add
        """
        self._buffer.append(trajectory)
        self._total_added += 1
        self._reward_sum += trajectory.reward
        self._token_sum += trajectory.num_response_tokens

    def add_batch(self, batch: TrajectoryBatch) -> None:
        """
        Add a batch of trajectories to the buffer.

        Args:
            batch: The trajectory batch to add
        """
        for traj in batch.trajectories:
            self.add(traj)

    def sample(self, batch_size: int) -> TrajectoryBatch:
        """
        Sample a random mini-batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            TrajectoryBatch containing sampled trajectories
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self._buffer))
        sampled = self._rng.sample(list(self._buffer), batch_size)

        return TrajectoryBatch(trajectories=sampled)

    def sample_recent(self, batch_size: int) -> TrajectoryBatch:
        """
        Sample from the most recent trajectories.

        Args:
            batch_size: Number of trajectories to sample

        Returns:
            TrajectoryBatch from recent trajectories
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self._buffer))
        recent = list(self._buffer)[-batch_size:]

        return TrajectoryBatch(trajectories=recent)

    def get_all(self) -> TrajectoryBatch:
        """
        Get all trajectories in the buffer.

        Returns:
            TrajectoryBatch containing all trajectories
        """
        return TrajectoryBatch(trajectories=list(self._buffer))

    def filter_by_reward(
        self,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
    ) -> TrajectoryBatch:
        """
        Filter trajectories by reward threshold.

        Args:
            min_reward: Minimum reward (inclusive)
            max_reward: Maximum reward (inclusive)

        Returns:
            TrajectoryBatch with filtered trajectories
        """
        filtered = []
        for traj in self._buffer:
            if min_reward is not None and traj.reward < min_reward:
                continue
            if max_reward is not None and traj.reward > max_reward:
                continue
            filtered.append(traj)

        return TrajectoryBatch(trajectories=filtered)

    def get_top_k(self, k: int) -> TrajectoryBatch:
        """
        Get top k trajectories by reward.

        Args:
            k: Number of top trajectories to return

        Returns:
            TrajectoryBatch with top k trajectories
        """
        sorted_trajs = sorted(self._buffer, key=lambda t: t.reward, reverse=True)
        return TrajectoryBatch(trajectories=sorted_trajs[:k])

    def get_stats(self) -> BufferStats:
        """
        Compute statistics about the buffer.

        Returns:
            BufferStats with current statistics
        """
        if len(self._buffer) == 0:
            return BufferStats()

        rewards = [t.reward for t in self._buffer]
        lengths = [t.num_response_tokens for t in self._buffer]

        return BufferStats(
            total_trajectories=len(self._buffer),
            total_tokens=sum(lengths),
            mean_reward=sum(rewards) / len(rewards),
            max_reward=max(rewards),
            min_reward=min(rewards),
            mean_response_length=sum(lengths) // len(lengths) if lengths else 0,
        )

    def clear(self) -> None:
        """Clear all trajectories from the buffer."""
        self._buffer.clear()
        self._total_added = 0
        self._reward_sum = 0.0
        self._token_sum = 0

    def iter_minibatches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[TrajectoryBatch]:
        """
        Iterate over the buffer in mini-batches.

        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle before iterating

        Yields:
            TrajectoryBatch for each mini-batch
        """
        trajectories = list(self._buffer)

        if shuffle:
            self._rng.shuffle(trajectories)

        for i in range(0, len(trajectories), batch_size):
            batch = trajectories[i : i + batch_size]
            yield TrajectoryBatch(trajectories=batch)


class PrioritizedTrajectoryBuffer(TrajectoryBuffer):
    """
    Trajectory buffer with prioritized sampling.

    Trajectories with higher rewards are sampled more frequently.
    """

    def __init__(
        self,
        max_size: int = 10000,
        seed: Optional[int] = None,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        """
        Initialize prioritized buffer.

        Args:
            max_size: Maximum number of trajectories
            seed: Random seed
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling correction (0 = no correction, 1 = full)
        """
        super().__init__(max_size, seed)
        self.alpha = alpha
        self.beta = beta
        self._priorities: Deque[float] = deque(maxlen=max_size)

    def add(self, trajectory: Trajectory) -> None:
        """Add trajectory with priority based on reward."""
        super().add(trajectory)
        # Priority based on reward (shifted to be positive)
        priority = abs(trajectory.reward) + 1.0
        self._priorities.append(priority ** self.alpha)

    def sample(self, batch_size: int) -> TrajectoryBatch:
        """Sample with prioritization."""
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self._buffer))

        # Compute sampling probabilities
        priorities = list(self._priorities)
        total_priority = sum(priorities)
        probs = [p / total_priority for p in priorities]

        # Sample indices based on priorities
        indices = []
        trajectories = list(self._buffer)

        for _ in range(batch_size):
            r = self._rng.random()
            cumsum = 0.0
            for i, p in enumerate(probs):
                cumsum += p
                if r <= cumsum:
                    indices.append(i)
                    break

        sampled = [trajectories[i] for i in indices]
        return TrajectoryBatch(trajectories=sampled)

    def clear(self) -> None:
        """Clear buffer and priorities."""
        super().clear()
        self._priorities.clear()


class RolloutCollector:
    """
    Collects rollouts and manages trajectory buffering.

    Provides a higher-level interface for collecting trajectories
    during RLVR training.
    """

    def __init__(
        self,
        buffer: Optional[TrajectoryBuffer] = None,
        buffer_size: int = 10000,
        seed: Optional[int] = None,
    ):
        """
        Initialize rollout collector.

        Args:
            buffer: Optional pre-existing buffer
            buffer_size: Size of new buffer if not provided
            seed: Random seed
        """
        self.buffer = buffer or TrajectoryBuffer(max_size=buffer_size, seed=seed)
        self._episode_count = 0
        self._step_count = 0

    def collect(
        self,
        batch: TrajectoryBatch,
        compute_rewards: bool = True,
    ) -> Dict[str, float]:
        """
        Collect a batch of trajectories.

        Args:
            batch: The trajectory batch to collect
            compute_rewards: Whether rewards have been computed

        Returns:
            Collection statistics
        """
        self.buffer.add_batch(batch)
        self._episode_count += len(batch)
        self._step_count += sum(t.num_response_tokens for t in batch)

        return {
            "episodes_collected": len(batch),
            "total_episodes": self._episode_count,
            "total_steps": self._step_count,
            "buffer_size": len(self.buffer),
            "mean_reward": batch.mean_reward,
        }

    def get_training_batch(self, batch_size: int) -> TrajectoryBatch:
        """
        Get a batch for training.

        Args:
            batch_size: Desired batch size

        Returns:
            TrajectoryBatch for training
        """
        return self.buffer.sample(batch_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        buffer_stats = self.buffer.get_stats()
        return {
            "episode_count": self._episode_count,
            "step_count": self._step_count,
            "buffer_size": len(self.buffer),
            "mean_reward": buffer_stats.mean_reward,
            "max_reward": buffer_stats.max_reward,
            "min_reward": buffer_stats.min_reward,
        }
