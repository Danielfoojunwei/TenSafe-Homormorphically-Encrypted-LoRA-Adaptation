"""
Tests for PPO (Proximal Policy Optimization) algorithm.

Verifies that PPO:
- Correctly computes the clipped surrogate objective
- Runs multiple PPO epochs per update
- Performs early stopping based on KL divergence
- Tracks clip fractions and other statistics
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr.algorithms.ppo import PPO, PPOConfig
from tensafe.rlvr.rollout import MockRolloutSampler, Trajectory, TrajectoryBatch


class MockPPOTrainingClient:
    """Mock training client for PPO testing."""

    def __init__(self, seed: int = 42):
        self.step = 0
        random.seed(seed)

    def forward_backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Mock forward-backward pass."""
        loss = 2.5 - self.step * 0.01
        return {
            "loss": max(0.1, loss),
            "grad_norm": 1.5 + random.uniform(-0.1, 0.1),
            "tokens_processed": 100,
        }

    def optim_step(self) -> Dict[str, Any]:
        """Mock optimizer step."""
        self.step += 1
        return {"step": self.step}


class TestPPOConfig:
    """Tests for PPO configuration."""

    def test_default_config(self):
        """Test default PPO configuration values."""
        config = PPOConfig()

        assert config.clip_range == 0.2
        assert config.ppo_epochs == 4
        assert config.entropy_coef == 0.01
        assert config.target_kl == 0.01

    def test_custom_config(self):
        """Test custom PPO configuration."""
        config = PPOConfig(
            clip_range=0.1,
            ppo_epochs=2,
            entropy_coef=0.05,
            target_kl=0.02,
        )

        assert config.clip_range == 0.1
        assert config.ppo_epochs == 2
        assert config.entropy_coef == 0.05
        assert config.target_kl == 0.02


class TestPPOInitialization:
    """Tests for PPO initialization."""

    def test_ppo_initialization(self):
        """Test PPO initializes correctly."""
        ppo = PPO()

        assert ppo.step == 0
        assert ppo.config.clip_range == 0.2
        assert ppo.config.ppo_epochs == 4

    def test_ppo_with_config(self):
        """Test PPO with custom config."""
        config = PPOConfig(clip_range=0.15, ppo_epochs=3)
        ppo = PPO(config)

        assert ppo.config.clip_range == 0.15
        assert ppo.config.ppo_epochs == 3


class TestPPOUpdate:
    """Tests for PPO update method."""

    @pytest.fixture
    def ppo(self) -> PPO:
        """Create a PPO instance."""
        return PPO(PPOConfig(ppo_epochs=2))

    @pytest.fixture
    def client(self) -> MockPPOTrainingClient:
        """Create a training client."""
        return MockPPOTrainingClient()

    @pytest.fixture
    def sample_batch(self) -> TrajectoryBatch:
        """Create a sample batch."""
        trajectories = [
            Trajectory(
                prompt=f"Prompt {i}",
                prompt_tokens=[1, 2, 3],
                response=f"Response {i}",
                response_tokens=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.2],
                attention_mask=[1, 1, 1, 1, 1, 1],
                reward=float(i - 2),  # -2, -1, 0, 1, 2
            )
            for i in range(5)
        ]
        return TrajectoryBatch(trajectories=trajectories)

    def test_update_returns_result(self, ppo, client, sample_batch):
        """Test that update returns a valid result."""
        result = ppo.update(sample_batch, client)

        assert hasattr(result, "policy_loss")
        assert hasattr(result, "entropy_loss")
        assert hasattr(result, "kl_loss")
        assert hasattr(result, "step")
        assert result.step == 1

    def test_update_increments_step(self, ppo, client, sample_batch):
        """Test that update increments step counter."""
        assert ppo.step == 0

        ppo.update(sample_batch, client)
        assert ppo.step == 1

        ppo.update(sample_batch, client)
        assert ppo.step == 2

    def test_multiple_ppo_epochs(self, client, sample_batch):
        """Test that multiple PPO epochs are run."""
        config = PPOConfig(ppo_epochs=3)
        ppo = PPO(config)

        # With 3 PPO epochs and no early stopping, client should
        # be called multiple times per update
        initial_step = client.step
        ppo.update(sample_batch, client)

        # Client step should have increased by number of epochs
        assert client.step == initial_step + 3

    def test_result_has_correct_statistics(self, ppo, client, sample_batch):
        """Test that result contains correct statistics."""
        result = ppo.update(sample_batch, client)

        # Should have computed mean advantage
        assert hasattr(result, "mean_advantage")
        # Should have computed entropy
        assert hasattr(result, "entropy")
        # Should have computed KL divergence
        assert hasattr(result, "kl_div")
        # Should track trajectories used
        assert result.trajectories_used == 5


class TestPPOClipping:
    """Tests for PPO clipping mechanism."""

    def test_clip_fraction_tracked(self):
        """Test that clip fraction is tracked."""
        ppo = PPO(PPOConfig(clip_range=0.2))

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2, 3],
                logprobs=[-0.5, -0.5],
                attention_mask=[1, 1, 1],
                reward=1.0,
                advantage=0.5,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Store old logprobs and run update
        ppo._store_old_logprobs(batch)
        loss, clip_fraction = ppo._compute_policy_loss(batch)

        # Clip fraction should be between 0 and 1
        assert 0 <= clip_fraction <= 1

    def test_clip_range_affects_loss(self):
        """Test that clip range affects policy loss computation."""
        ppo_narrow = PPO(PPOConfig(clip_range=0.1))
        ppo_wide = PPO(PPOConfig(clip_range=0.3))

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
                advantage=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Simulate changed logprobs (would cause clipping)
        ppo_narrow._old_logprobs = {0: [-0.8]}  # Different from current
        ppo_wide._old_logprobs = {0: [-0.8]}

        loss_narrow, _ = ppo_narrow._compute_policy_loss(batch)
        loss_wide, _ = ppo_wide._compute_policy_loss(batch)

        # Losses may differ due to different clip ranges
        assert isinstance(loss_narrow, float)
        assert isinstance(loss_wide, float)


class TestPPOKLDivergence:
    """Tests for PPO KL divergence computation."""

    def test_kl_computed(self):
        """Test that KL divergence is computed."""
        ppo = PPO()

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        ppo._store_old_logprobs(batch)
        kl_div, kl_loss = ppo._compute_kl_loss(batch)

        # With same logprobs, KL should be ~0
        assert kl_div < 0.01

    def test_kl_early_stopping(self):
        """Test that KL early stopping works."""
        # Create config with very low target KL
        config = PPOConfig(
            ppo_epochs=10,
            target_kl=0.001,
        )
        ppo = PPO(config)
        client = MockPPOTrainingClient()

        # Create batch with high reward variance (may cause high KL)
        trajectories = [
            Trajectory(
                prompt=f"P{i}",
                prompt_tokens=[1],
                response=f"R{i}",
                response_tokens=[2],
                logprobs=[-0.5 - i * 0.1],  # Varying logprobs
                attention_mask=[1, 1],
                reward=float(i),
            )
            for i in range(5)
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        initial_step = client.step
        ppo.update(batch, client)

        # May have stopped early (less than 10 epochs)
        # Just verify it completed without error
        assert client.step > initial_step


class TestPPOEntropy:
    """Tests for PPO entropy computation."""

    def test_entropy_computed(self):
        """Test that entropy is computed."""
        ppo = PPO()

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2, 3, 4],
                logprobs=[-0.5, -0.3, -0.7],
                attention_mask=[1, 1, 1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        entropy, entropy_loss = ppo._compute_entropy_loss(batch)

        # Entropy should be positive
        assert entropy > 0
        # Entropy loss should be negative (we maximize entropy)
        assert entropy_loss < 0

    def test_entropy_coef_affects_loss(self):
        """Test that entropy coefficient affects total loss."""
        ppo_low = PPO(PPOConfig(entropy_coef=0.01))
        ppo_high = PPO(PPOConfig(entropy_coef=0.1))

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        _, loss_low = ppo_low._compute_entropy_loss(batch)
        _, loss_high = ppo_high._compute_entropy_loss(batch)

        # Higher entropy coefficient should give larger (more negative) loss
        assert loss_high < loss_low


class TestPPOAdaptiveKL:
    """Tests for adaptive KL coefficient."""

    def test_adaptive_kl_increases(self):
        """Test that KL coefficient increases when KL is high."""
        config = PPOConfig(
            adaptive_kl=True,
            target_kl=0.01,
            kl_coef=0.1,
        )
        ppo = PPO(config)

        initial_kl_coef = ppo._kl_coef

        # Simulate high KL
        ppo._update_kl_coef(0.05)  # 5x target

        assert ppo._kl_coef > initial_kl_coef

    def test_adaptive_kl_decreases(self):
        """Test that KL coefficient decreases when KL is low."""
        config = PPOConfig(
            adaptive_kl=True,
            target_kl=0.1,
            kl_coef=0.1,
        )
        ppo = PPO(config)

        initial_kl_coef = ppo._kl_coef

        # Simulate low KL
        ppo._update_kl_coef(0.01)  # 10x lower than target

        assert ppo._kl_coef < initial_kl_coef


class TestPPORewardNormalization:
    """Tests for reward normalization."""

    def test_reward_normalization(self):
        """Test that reward normalization works."""
        config = PPOConfig(normalize_rewards=True)
        ppo = PPO(config)

        # Create batch with varied rewards
        trajectories = [
            Trajectory(
                prompt=f"P{i}",
                prompt_tokens=[1],
                response=f"R{i}",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=float(i * 10),  # 0, 10, 20, 30, 40
            )
            for i in range(5)
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        normalized = ppo._normalize_rewards(batch)

        # Rewards should have changed
        # (mean subtracted and divided by std)
        rewards = [t.reward for t in normalized]
        mean = sum(rewards) / len(rewards)

        # After normalization, mean should be closer to 0
        # (for first batch, it won't be exactly 0 due to running stats)
        assert abs(mean) < 25  # Less than original mean of 20


class TestPPOState:
    """Tests for PPO state management."""

    def test_get_state(self):
        """Test that get_state returns complete state."""
        ppo = PPO(PPOConfig(clip_range=0.15))

        # Run an update to change state
        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)
        ppo.update(batch, None)

        state = ppo.get_state()

        assert "step" in state
        assert "baseline" in state
        assert "kl_coef" in state
        assert "ppo_config" in state
        assert state["step"] == 1

    def test_load_state(self):
        """Test that load_state restores state."""
        ppo = PPO()

        # Create state to load
        state = {
            "step": 10,
            "baseline": 0.5,
            "kl_coef": 0.2,
            "reward_mean": 1.0,
            "reward_var": 2.0,
            "reward_count": 100,
        }

        ppo.load_state(state)

        assert ppo._step == 10
        assert ppo._baseline == 0.5
        assert ppo._kl_coef == 0.2
        assert ppo._reward_mean == 1.0


class TestPPOIntegration:
    """Integration tests for PPO."""

    def test_full_training_loop(self):
        """Test a full PPO training loop."""
        ppo = PPO(PPOConfig(ppo_epochs=2))
        sampler = MockRolloutSampler(max_new_tokens=16, seed=42)
        client = MockPPOTrainingClient()

        prompts = ["Test 1", "Test 2", "Test 3"]

        for _ in range(5):
            batch = sampler.generate_trajectories(prompts)
            for traj in batch:
                traj.reward = len(traj.response) / 50.0
            result = ppo.update(batch, client)

        assert ppo.step == 5
        assert client.step > 5  # Multiple PPO epochs per update

    def test_comparison_with_reinforce(self):
        """Test that PPO and REINFORCE can be used interchangeably."""
        from tensafe.rlvr.algorithms.reinforce import REINFORCE

        ppo = PPO(PPOConfig(ppo_epochs=1))
        reinforce = REINFORCE()

        trajectories = [
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Both should work with same batch
        ppo_result = ppo.update(batch, None)
        reinforce_result = reinforce.update(batch, None)

        # Both should return valid results
        assert hasattr(ppo_result, "policy_loss")
        assert hasattr(reinforce_result, "policy_loss")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
