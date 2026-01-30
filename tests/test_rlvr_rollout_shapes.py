"""
Tests for RLVR rollout shapes and data structures.

Verifies that trajectories and trajectory batches have correct structure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr.rollout import (
    MockRolloutSampler,
    RolloutSampler,
    Trajectory,
    TrajectoryBatch,
)


class TestTrajectory:
    """Tests for Trajectory dataclass."""

    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        traj = Trajectory(
            prompt="Hello",
            prompt_tokens=[1, 2, 3],
            response="World",
            response_tokens=[4, 5],
            logprobs=[-0.5, -0.3],
            attention_mask=[1, 1, 1, 1, 1],
        )

        assert traj.prompt == "Hello"
        assert len(traj.prompt_tokens) == 3
        assert len(traj.response_tokens) == 2
        assert len(traj.logprobs) == 2

    def test_full_tokens(self):
        """Test full_tokens property."""
        traj = Trajectory(
            prompt="A",
            prompt_tokens=[1, 2],
            response="B",
            response_tokens=[3, 4, 5],
            logprobs=[-0.5, -0.5, -0.5],
            attention_mask=[1, 1, 1, 1, 1],
        )

        assert traj.full_tokens == [1, 2, 3, 4, 5]

    def test_num_response_tokens(self):
        """Test num_response_tokens property."""
        traj = Trajectory(
            prompt="P",
            prompt_tokens=[1],
            response="R",
            response_tokens=[2, 3, 4],
            logprobs=[-0.1, -0.2, -0.3],
            attention_mask=[1, 1, 1, 1],
        )

        assert traj.num_response_tokens == 3

    def test_total_logprob(self):
        """Test total_logprob property."""
        traj = Trajectory(
            prompt="P",
            prompt_tokens=[1],
            response="R",
            response_tokens=[2, 3],
            logprobs=[-0.5, -0.3],
            attention_mask=[1, 1, 1],
        )

        assert abs(traj.total_logprob - (-0.8)) < 1e-6

    def test_mean_logprob(self):
        """Test mean_logprob property."""
        traj = Trajectory(
            prompt="P",
            prompt_tokens=[1],
            response="R",
            response_tokens=[2, 3, 4, 5],
            logprobs=[-0.4, -0.2, -0.6, -0.8],
            attention_mask=[1, 1, 1, 1, 1],
        )

        expected_mean = sum([-0.4, -0.2, -0.6, -0.8]) / 4
        assert abs(traj.mean_logprob - expected_mean) < 1e-6

    def test_reward_and_advantage(self):
        """Test reward and advantage fields."""
        traj = Trajectory(
            prompt="P",
            prompt_tokens=[1],
            response="R",
            response_tokens=[2],
            logprobs=[-0.5],
            attention_mask=[1, 1],
            reward=1.5,
            advantage=0.3,
        )

        assert traj.reward == 1.5
        assert traj.advantage == 0.3


class TestTrajectoryBatch:
    """Tests for TrajectoryBatch."""

    @pytest.fixture
    def sample_batch(self) -> TrajectoryBatch:
        """Create a sample trajectory batch."""
        trajectories = []
        for i in range(5):
            traj = Trajectory(
                prompt=f"Prompt {i}",
                prompt_tokens=list(range(5)),
                response=f"Response {i}",
                response_tokens=list(range(10)),
                logprobs=[-0.5] * 10,
                attention_mask=[1] * 15,
                reward=float(i),
            )
            trajectories.append(traj)
        return TrajectoryBatch(trajectories=trajectories)

    def test_batch_length(self, sample_batch):
        """Test batch length."""
        assert len(sample_batch) == 5

    def test_batch_iteration(self, sample_batch):
        """Test iterating over batch."""
        count = 0
        for traj in sample_batch:
            assert isinstance(traj, Trajectory)
            count += 1
        assert count == 5

    def test_batch_indexing(self, sample_batch):
        """Test indexing into batch."""
        assert sample_batch[0].prompt == "Prompt 0"
        assert sample_batch[4].prompt == "Prompt 4"

    def test_prompts_property(self, sample_batch):
        """Test prompts property."""
        prompts = sample_batch.prompts
        assert len(prompts) == 5
        assert prompts[0] == "Prompt 0"

    def test_rewards_property(self, sample_batch):
        """Test rewards property."""
        rewards = sample_batch.rewards
        assert len(rewards) == 5
        assert rewards == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_mean_reward(self, sample_batch):
        """Test mean_reward property."""
        expected = sum([0.0, 1.0, 2.0, 3.0, 4.0]) / 5
        assert abs(sample_batch.mean_reward - expected) < 1e-6

    def test_std_reward(self, sample_batch):
        """Test std_reward property."""
        rewards = [0.0, 1.0, 2.0, 3.0, 4.0]
        mean = sum(rewards) / len(rewards)
        variance = sum((r - mean) ** 2 for r in rewards) / len(rewards)
        expected_std = variance ** 0.5

        assert abs(sample_batch.std_reward - expected_std) < 1e-6

    def test_compute_advantages(self, sample_batch):
        """Test compute_advantages method."""
        sample_batch.compute_advantages(baseline=2.0, normalize=False)

        # Advantages should be reward - baseline
        assert abs(sample_batch[0].advantage - (-2.0)) < 1e-6
        assert abs(sample_batch[2].advantage - 0.0) < 1e-6
        assert abs(sample_batch[4].advantage - 2.0) < 1e-6

    def test_compute_advantages_normalized(self, sample_batch):
        """Test normalized advantage computation."""
        sample_batch.compute_advantages(normalize=True)

        # Mean should be approximately 0
        advantages = sample_batch.advantages
        mean_adv = sum(advantages) / len(advantages)
        assert abs(mean_adv) < 1e-6

    def test_to_dict(self, sample_batch):
        """Test to_dict method."""
        d = sample_batch.to_dict()

        assert "prompts" in d
        assert "responses" in d
        assert "rewards" in d
        assert "mean_reward" in d
        assert d["num_trajectories"] == 5


class TestMockRolloutSampler:
    """Tests for MockRolloutSampler."""

    def test_sampler_creates_trajectories(self):
        """Test that sampler creates trajectories."""
        sampler = MockRolloutSampler(
            max_new_tokens=32,
            temperature=0.7,
            seed=42,
        )

        prompts = ["Hello", "World", "Test"]
        batch = sampler.generate_trajectories(prompts)

        assert len(batch) == 3
        assert all(isinstance(t, Trajectory) for t in batch)

    def test_sampler_deterministic(self):
        """Test that same seed produces same results."""
        sampler1 = MockRolloutSampler(seed=42)
        sampler2 = MockRolloutSampler(seed=42)

        prompts = ["Test prompt"]

        batch1 = sampler1.generate_trajectories(prompts)
        batch2 = sampler2.generate_trajectories(prompts)

        assert batch1[0].response == batch2[0].response
        assert batch1[0].logprobs == batch2[0].logprobs

    def test_trajectory_shapes_match(self):
        """Test that trajectory shapes are consistent."""
        sampler = MockRolloutSampler(max_new_tokens=64, seed=42)

        batch = sampler.generate_trajectories(["Prompt 1", "Prompt 2"])

        for traj in batch:
            # Number of logprobs should match response tokens
            assert len(traj.logprobs) == len(traj.response_tokens)

            # Attention mask should cover full sequence
            assert len(traj.attention_mask) == len(traj.full_tokens)

    def test_logprobs_are_negative(self):
        """Test that log probabilities are negative (valid log probs)."""
        sampler = MockRolloutSampler(seed=42)

        batch = sampler.generate_trajectories(["Test"])

        for traj in batch:
            assert all(lp < 0 for lp in traj.logprobs)


class TestRolloutSamplerShapes:
    """Additional shape tests for rollout generation."""

    def test_batch_prompt_token_alignment(self):
        """Test that prompts and tokens align."""
        sampler = MockRolloutSampler(seed=42)

        prompts = ["Short", "A longer prompt here", "X"]
        batch = sampler.generate_trajectories(prompts)

        for i, traj in enumerate(batch):
            assert traj.prompt == prompts[i]
            # Prompt tokens should be non-empty
            assert len(traj.prompt_tokens) > 0

    def test_metadata_preserved(self):
        """Test that metadata is attached to trajectories."""
        sampler = MockRolloutSampler(seed=42)

        batch = sampler.generate_trajectories(["Test 1", "Test 2"])

        for i, traj in enumerate(batch):
            assert "sample_index" in traj.metadata
            assert traj.metadata["sample_index"] == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
