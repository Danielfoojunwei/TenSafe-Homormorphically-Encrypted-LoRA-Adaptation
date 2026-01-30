"""
Tests for RLVR checkpoint compatibility.

Verifies that:
- Algorithm state can be saved and restored
- Training can resume from checkpoints
- State is compatible across algorithm versions
- RLVR checkpoints are compatible with SFT checkpoints
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr.algorithms.ppo import PPO, PPOConfig
from tensafe.rlvr.algorithms.reinforce import REINFORCE, REINFORCEConfig
from tensafe.rlvr.algorithms.base import AlgorithmConfig, MockRLAlgorithm
from tensafe.rlvr.rollout import MockRolloutSampler, Trajectory, TrajectoryBatch


class MockCheckpointStore:
    """Mock checkpoint storage for testing."""

    def __init__(self):
        self.checkpoints: Dict[str, bytes] = {}
        self._id_counter = 0

    def save(self, state: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Save state and return checkpoint ID."""
        self._id_counter += 1
        checkpoint_id = f"ckpt-{self._id_counter:04d}"

        checkpoint = {
            "state": state,
            "metadata": metadata or {},
            "version": "1.0.0",
        }

        self.checkpoints[checkpoint_id] = json.dumps(checkpoint).encode()
        return checkpoint_id

    def load(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load checkpoint by ID."""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        data = json.loads(self.checkpoints[checkpoint_id])
        return data["state"]

    def get_metadata(self, checkpoint_id: str) -> Dict[str, Any]:
        """Get checkpoint metadata."""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        data = json.loads(self.checkpoints[checkpoint_id])
        return data["metadata"]


class TestREINFORCECheckpoint:
    """Tests for REINFORCE checkpoint save/load."""

    @pytest.fixture
    def reinforce(self) -> REINFORCE:
        """Create REINFORCE instance."""
        return REINFORCE(REINFORCEConfig(
            use_baseline=True,
            normalize_advantages=True,
            entropy_coef=0.02,
        ))

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        """Create checkpoint store."""
        return MockCheckpointStore()

    def test_save_initial_state(self, reinforce, store):
        """Test saving initial state."""
        state = reinforce.get_state()
        checkpoint_id = store.save(state)

        assert checkpoint_id is not None
        assert checkpoint_id in store.checkpoints

    def test_save_after_training(self, reinforce, store):
        """Test saving state after training."""
        # Train a few steps
        for i in range(5):
            trajectories = [
                Trajectory(
                    prompt=f"P{i}",
                    prompt_tokens=[1],
                    response=f"R{i}",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=float(i),
                ),
            ]
            batch = TrajectoryBatch(trajectories=trajectories)
            reinforce.update(batch, None)

        # Save state
        state = reinforce.get_state()
        checkpoint_id = store.save(state, {"step": reinforce.step})

        # Verify state
        loaded = store.load(checkpoint_id)
        assert loaded["step"] == 5

    def test_load_restores_state(self, store):
        """Test that loading restores algorithm state."""
        # Create and train algorithm
        reinforce1 = REINFORCE(REINFORCEConfig(entropy_coef=0.03))

        for i in range(3):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=float(i),
                ),
            ])
            reinforce1.update(batch, None)

        # Save checkpoint
        checkpoint_id = store.save(reinforce1.get_state())

        # Create new algorithm and load state
        reinforce2 = REINFORCE(REINFORCEConfig(entropy_coef=0.03))
        reinforce2.load_state(store.load(checkpoint_id))

        # Verify state matches
        assert reinforce2.step == reinforce1.step
        assert reinforce2._baseline == reinforce1._baseline

    def test_resume_training(self, store):
        """Test resuming training from checkpoint."""
        reinforce1 = REINFORCE()

        # Train phase 1
        for i in range(3):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=1.0,
                ),
            ])
            reinforce1.update(batch, None)

        # Save checkpoint
        checkpoint_id = store.save(reinforce1.get_state())

        # Create new algorithm and resume
        reinforce2 = REINFORCE()
        reinforce2.load_state(store.load(checkpoint_id))

        # Train phase 2
        for i in range(2):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=1.0,
                ),
            ])
            reinforce2.update(batch, None)

        # Total steps should be 3 + 2 = 5
        assert reinforce2.step == 5


class TestPPOCheckpoint:
    """Tests for PPO checkpoint save/load."""

    @pytest.fixture
    def ppo(self) -> PPO:
        """Create PPO instance."""
        return PPO(PPOConfig(
            clip_range=0.2,
            ppo_epochs=2,
            adaptive_kl=True,
        ))

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        """Create checkpoint store."""
        return MockCheckpointStore()

    def test_save_ppo_state(self, ppo, store):
        """Test saving PPO state."""
        # Train a bit to change state
        batch = TrajectoryBatch(trajectories=[
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ])
        ppo.update(batch, None)

        # Save
        state = ppo.get_state()
        checkpoint_id = store.save(state)

        # Verify PPO-specific state
        loaded = store.load(checkpoint_id)
        assert "kl_coef" in loaded
        assert "ppo_config" in loaded

    def test_load_ppo_state(self, store):
        """Test loading PPO state."""
        ppo1 = PPO(PPOConfig(adaptive_kl=True))

        # Simulate adaptive KL changes
        ppo1._kl_coef = 0.15
        ppo1._step = 10
        ppo1._baseline = 0.5

        checkpoint_id = store.save(ppo1.get_state())

        ppo2 = PPO()
        ppo2.load_state(store.load(checkpoint_id))

        assert ppo2._kl_coef == 0.15
        assert ppo2._step == 10
        assert ppo2._baseline == 0.5


class TestCrossAlgorithmCompatibility:
    """Tests for compatibility between different algorithms."""

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        return MockCheckpointStore()

    def test_mock_algorithm_checkpoint(self, store):
        """Test MockRLAlgorithm can save/load."""
        mock = MockRLAlgorithm()

        # Train
        batch = TrajectoryBatch(trajectories=[
            Trajectory(
                prompt="P",
                prompt_tokens=[1],
                response="R",
                response_tokens=[2],
                logprobs=[-0.5],
                attention_mask=[1, 1],
                reward=1.0,
            ),
        ])
        mock.update(batch, None)

        # Save and load
        checkpoint_id = store.save(mock.get_state())
        loaded = store.load(checkpoint_id)

        # Verify basic state
        assert loaded["step"] == 1

    def test_state_structure_compatible(self, store):
        """Test that all algorithms have compatible state structure."""
        algorithms = [
            REINFORCE(),
            PPO(),
            MockRLAlgorithm(),
        ]

        required_keys = {"step", "baseline", "config"}

        for algo in algorithms:
            state = algo.get_state()
            for key in required_keys:
                assert key in state, f"{algo.__class__.__name__} missing {key}"


class TestCheckpointVersioning:
    """Tests for checkpoint versioning and migration."""

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        return MockCheckpointStore()

    def test_save_with_version(self, store):
        """Test that checkpoints include version."""
        algo = REINFORCE()
        checkpoint_id = store.save(algo.get_state())

        # Check raw checkpoint data
        data = json.loads(store.checkpoints[checkpoint_id])
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_metadata_preserved(self, store):
        """Test that metadata is preserved in checkpoint."""
        algo = REINFORCE()
        metadata = {
            "experiment_name": "test_exp",
            "created_at": "2024-01-15T12:00:00",
            "notes": "Test checkpoint",
        }

        checkpoint_id = store.save(algo.get_state(), metadata)

        loaded_meta = store.get_metadata(checkpoint_id)
        assert loaded_meta == metadata


class TestMultipleCheckpoints:
    """Tests for managing multiple checkpoints."""

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        return MockCheckpointStore()

    def test_save_multiple_checkpoints(self, store):
        """Test saving multiple checkpoints."""
        algo = REINFORCE()
        checkpoint_ids = []

        for i in range(5):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=float(i),
                ),
            ])
            algo.update(batch, None)
            checkpoint_ids.append(store.save(algo.get_state()))

        assert len(checkpoint_ids) == 5
        assert len(set(checkpoint_ids)) == 5  # All unique

    def test_load_specific_checkpoint(self, store):
        """Test loading a specific checkpoint from history."""
        algo = REINFORCE()
        checkpoints = {}

        for step in [5, 10, 15]:
            algo._step = step
            checkpoint_id = store.save(algo.get_state())
            checkpoints[step] = checkpoint_id

        # Load middle checkpoint
        algo2 = REINFORCE()
        algo2.load_state(store.load(checkpoints[10]))
        assert algo2.step == 10


class TestCheckpointIntegrity:
    """Tests for checkpoint data integrity."""

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        return MockCheckpointStore()

    def test_state_serializable(self):
        """Test that all state values are JSON-serializable."""
        algorithms = [
            REINFORCE(),
            PPO(),
            MockRLAlgorithm(),
        ]

        for algo in algorithms:
            state = algo.get_state()
            # Should not raise
            json_str = json.dumps(state)
            loaded = json.loads(json_str)
            assert loaded is not None

    def test_float_precision_preserved(self, store):
        """Test that float precision is preserved."""
        algo = REINFORCE()
        algo._baseline = 0.123456789012345

        checkpoint_id = store.save(algo.get_state())
        loaded = store.load(checkpoint_id)

        # JSON should preserve reasonable precision
        assert abs(loaded["baseline"] - 0.123456789012345) < 1e-10

    def test_config_preserved(self, store):
        """Test that config values are preserved."""
        config = REINFORCEConfig(
            learning_rate=5e-5,
            entropy_coef=0.03,
            normalize_advantages=False,
        )
        algo = REINFORCE(config)

        checkpoint_id = store.save(algo.get_state())
        loaded = store.load(checkpoint_id)

        assert loaded["config"]["learning_rate"] == 5e-5
        assert loaded["config"]["entropy_coef"] == 0.03
        assert loaded["config"]["normalize_advantages"] is False


class TestTrainingContinuity:
    """Tests for training continuity across checkpoints."""

    @pytest.fixture
    def sampler(self) -> MockRolloutSampler:
        return MockRolloutSampler(max_new_tokens=16, seed=42)

    def test_baseline_continuity(self):
        """Test that baseline is maintained across checkpoint load."""
        algo1 = REINFORCE(REINFORCEConfig(baseline_decay=0.9))

        # Train to build up baseline
        for i in range(10):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=float(i),
                ),
            ])
            algo1.update(batch, None)

        baseline_before = algo1._baseline

        # Save and load
        state = algo1.get_state()
        algo2 = REINFORCE(REINFORCEConfig(baseline_decay=0.9))
        algo2.load_state(state)

        assert algo2._baseline == baseline_before

    def test_reward_stats_continuity(self):
        """Test that PPO reward stats are maintained."""
        ppo1 = PPO(PPOConfig(normalize_rewards=True))

        # Train to build up stats
        for i in range(20):
            batch = TrajectoryBatch(trajectories=[
                Trajectory(
                    prompt="P",
                    prompt_tokens=[1],
                    response="R",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=float(i),
                ),
            ])
            ppo1.prepare_batch(batch)
            ppo1._normalize_rewards(batch)

        # Save and load
        state = ppo1.get_state()
        ppo2 = PPO(PPOConfig(normalize_rewards=True))
        ppo2.load_state(state)

        assert ppo2._reward_mean == ppo1._reward_mean
        assert ppo2._reward_count == ppo1._reward_count


class TestErrorHandling:
    """Tests for checkpoint error handling."""

    @pytest.fixture
    def store(self) -> MockCheckpointStore:
        return MockCheckpointStore()

    def test_load_nonexistent_raises(self, store):
        """Test that loading nonexistent checkpoint raises."""
        with pytest.raises(ValueError, match="Checkpoint not found"):
            store.load("nonexistent-checkpoint")

    def test_load_partial_state(self):
        """Test loading state with missing fields."""
        algo = REINFORCE()

        # Partial state (missing some fields)
        partial_state = {
            "step": 5,
        }

        # Should not raise, use defaults for missing
        algo.load_state(partial_state)
        assert algo._step == 5
        assert algo._baseline == 0.0  # Default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
