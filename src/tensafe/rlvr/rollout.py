"""
RLVR Rollout Sampler

Handles generating responses from the current policy and collecting
trajectories for reinforcement learning updates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union

logger = logging.getLogger(__name__)

# Try to import numpy, provide fallback
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None  # type: ignore


@dataclass
class Trajectory:
    """
    A single trajectory (prompt + generated response).

    Stores all information needed for policy gradient computation.
    """

    # Input
    prompt: str
    prompt_tokens: List[int]

    # Generated output
    response: str
    response_tokens: List[int]

    # Log probabilities of generated tokens
    logprobs: List[float]

    # Attention mask for the full sequence
    attention_mask: List[int]

    # Reward information (filled after reward computation)
    reward: float = 0.0
    advantage: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_tokens(self) -> List[int]:
        """Return the full token sequence (prompt + response)."""
        return self.prompt_tokens + self.response_tokens

    @property
    def num_response_tokens(self) -> int:
        """Number of generated response tokens."""
        return len(self.response_tokens)

    @property
    def total_logprob(self) -> float:
        """Sum of log probabilities for the response."""
        return sum(self.logprobs)

    @property
    def mean_logprob(self) -> float:
        """Mean log probability per token."""
        if not self.logprobs:
            return 0.0
        return sum(self.logprobs) / len(self.logprobs)


@dataclass
class TrajectoryBatch:
    """
    A batch of trajectories for vectorized operations.

    Provides methods for computing batch-level statistics and
    preparing data for policy updates.
    """

    trajectories: List[Trajectory]

    # Cached batch statistics
    _mean_reward: Optional[float] = field(default=None, repr=False)
    _std_reward: Optional[float] = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]

    @property
    def prompts(self) -> List[str]:
        """All prompts in the batch."""
        return [t.prompt for t in self.trajectories]

    @property
    def responses(self) -> List[str]:
        """All responses in the batch."""
        return [t.response for t in self.trajectories]

    @property
    def rewards(self) -> List[float]:
        """All rewards in the batch."""
        return [t.reward for t in self.trajectories]

    @property
    def advantages(self) -> List[float]:
        """All advantages in the batch."""
        return [t.advantage for t in self.trajectories]

    @property
    def mean_reward(self) -> float:
        """Mean reward across the batch."""
        if self._mean_reward is None:
            rewards = self.rewards
            self._mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        return self._mean_reward

    @property
    def std_reward(self) -> float:
        """Standard deviation of rewards."""
        if self._std_reward is None:
            rewards = self.rewards
            if len(rewards) < 2:
                self._std_reward = 0.0
            else:
                mean = self.mean_reward
                variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
                self._std_reward = variance ** 0.5
        return self._std_reward

    def compute_advantages(
        self,
        baseline: Optional[float] = None,
        normalize: bool = True,
    ) -> None:
        """
        Compute advantages for all trajectories.

        Args:
            baseline: Baseline value to subtract (default: batch mean)
            normalize: Whether to normalize advantages
        """
        if baseline is None:
            baseline = self.mean_reward

        for traj in self.trajectories:
            traj.advantage = traj.reward - baseline

        if normalize and self.std_reward > 1e-8:
            for traj in self.trajectories:
                traj.advantage = traj.advantage / (self.std_reward + 1e-8)

    def get_logprobs_tensor(self) -> List[List[float]]:
        """Get log probabilities as a list of lists (for conversion to tensor)."""
        return [t.logprobs for t in self.trajectories]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "prompts": self.prompts,
            "responses": self.responses,
            "rewards": self.rewards,
            "advantages": self.advantages,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "num_trajectories": len(self),
        }


class SamplingClient(Protocol):
    """Protocol for clients that can generate samples."""

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Any:
        """Generate samples from prompts."""
        ...


class RolloutSampler:
    """
    Generates rollouts (trajectories) from the current policy.

    The sampler takes a batch of prompts and generates responses,
    collecting log probabilities for policy gradient computation.
    """

    def __init__(
        self,
        client: SamplingClient,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        stop_sequences: Optional[List[str]] = None,
    ):
        """
        Initialize the rollout sampler.

        Args:
            client: Training client for generating samples
            max_new_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop_sequences: Sequences that stop generation
        """
        self.client = client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.stop_sequences = stop_sequences or []

    def generate_trajectories(
        self,
        prompts: List[str],
        prompt_tokens: Optional[List[List[int]]] = None,
    ) -> TrajectoryBatch:
        """
        Generate trajectories for a batch of prompts.

        Args:
            prompts: List of prompt strings
            prompt_tokens: Optional pre-tokenized prompts

        Returns:
            TrajectoryBatch containing all generated trajectories
        """
        logger.debug(f"Generating trajectories for {len(prompts)} prompts")

        # Generate samples
        sample_result = self.client.sample(
            prompts=prompts,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )

        # Convert samples to trajectories
        trajectories = []
        samples = sample_result.get("samples", []) if isinstance(sample_result, dict) else sample_result.samples

        for i, sample in enumerate(samples):
            # Extract sample data
            if isinstance(sample, dict):
                prompt = sample.get("prompt", prompts[i] if i < len(prompts) else "")
                response = sample.get("completion", "")
                tokens_generated = sample.get("tokens_generated", len(response.split()))
            else:
                prompt = sample.prompt
                response = sample.completion
                tokens_generated = sample.tokens_generated

            # Create mock token IDs and logprobs for now
            # In a real implementation, these would come from the model
            prompt_toks = prompt_tokens[i] if prompt_tokens else list(range(len(prompt.split())))
            response_toks = list(range(tokens_generated))

            # Mock log probabilities (would come from model in real impl)
            # Use slightly negative values as typical for log probs
            logprobs = [-0.5 - (j * 0.01) for j in range(tokens_generated)]

            traj = Trajectory(
                prompt=prompt,
                prompt_tokens=prompt_toks,
                response=response,
                response_tokens=response_toks,
                logprobs=logprobs,
                attention_mask=[1] * (len(prompt_toks) + len(response_toks)),
                metadata={"sample_index": i},
            )
            trajectories.append(traj)

        return TrajectoryBatch(trajectories=trajectories)

    def sample_single(self, prompt: str) -> Trajectory:
        """
        Generate a single trajectory.

        Args:
            prompt: The prompt string

        Returns:
            A single Trajectory
        """
        batch = self.generate_trajectories([prompt])
        return batch[0]


class MockRolloutSampler(RolloutSampler):
    """
    Mock rollout sampler for testing without a real model.

    Generates deterministic mock responses based on prompts.
    """

    def __init__(
        self,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        seed: int = 42,
    ):
        """
        Initialize mock sampler.

        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (affects mock randomness)
            seed: Random seed for reproducibility
        """
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.seed = seed
        self._rng_state = seed

    def _next_random(self) -> float:
        """Simple PRNG for deterministic mock generation."""
        self._rng_state = (self._rng_state * 1103515245 + 12345) & 0x7FFFFFFF
        return self._rng_state / 0x7FFFFFFF

    def generate_trajectories(
        self,
        prompts: List[str],
        prompt_tokens: Optional[List[List[int]]] = None,
    ) -> TrajectoryBatch:
        """Generate mock trajectories."""
        trajectories = []

        for i, prompt in enumerate(prompts):
            # Generate deterministic mock response
            words = prompt.split()
            response_len = min(10 + len(words), self.max_new_tokens // 4)

            # Generate response based on prompt
            response_words = []
            for j in range(response_len):
                # Use deterministic "random" word selection
                word_idx = int(self._next_random() * len(words)) if words else 0
                if words:
                    response_words.append(words[word_idx % len(words)])
                else:
                    response_words.append(f"word{j}")

            response = " ".join(response_words)

            # Mock tokens and logprobs
            prompt_toks = prompt_tokens[i] if prompt_tokens else list(range(len(words)))
            response_toks = list(range(response_len))
            logprobs = [-0.3 - (self._next_random() * 0.4) for _ in range(response_len)]

            traj = Trajectory(
                prompt=prompt,
                prompt_tokens=prompt_toks,
                response=response,
                response_tokens=response_toks,
                logprobs=logprobs,
                attention_mask=[1] * (len(prompt_toks) + len(response_toks)),
                metadata={"sample_index": i, "mock": True},
            )
            trajectories.append(traj)

        return TrajectoryBatch(trajectories=trajectories)
