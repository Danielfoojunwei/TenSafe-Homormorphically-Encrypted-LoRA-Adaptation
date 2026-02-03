"""
Tests for verifying only LoRA parameters are updated during RLVR.

Ensures that base model parameters remain frozen while LoRA adapters
are updated during training.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add tensafe to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensafe.rlvr.algorithms.reinforce import REINFORCE, REINFORCEConfig
from tensafe.rlvr.rollout import MockRolloutSampler, Trajectory, TrajectoryBatch


@dataclass
class MockParameter:
    """Mock parameter with name and values."""

    name: str
    values: List[float]
    requires_grad: bool = True

    def checksum(self) -> str:
        """Compute checksum of parameter values."""
        data = json.dumps(self.values, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def clone(self) -> MockParameter:
        """Create a deep copy of the parameter."""
        return MockParameter(
            name=self.name,
            values=self.values.copy(),
            requires_grad=self.requires_grad,
        )


@dataclass
class MockLoRALayer:
    """Mock LoRA layer with A and B matrices."""

    name: str
    lora_A: MockParameter = field(default_factory=lambda: MockParameter("lora_A", [0.01] * 16))
    lora_B: MockParameter = field(default_factory=lambda: MockParameter("lora_B", [0.0] * 16))

    def __post_init__(self):
        self.lora_A.name = f"{self.name}.lora_A"
        self.lora_B.name = f"{self.name}.lora_B"

    def get_params(self) -> List[MockParameter]:
        return [self.lora_A, self.lora_B]

    def get_checksums(self) -> Dict[str, str]:
        return {
            "lora_A": self.lora_A.checksum(),
            "lora_B": self.lora_B.checksum(),
        }


class MockModel:
    """
    Mock model with base parameters and LoRA layers.

    Simulates a transformer model with:
    - Frozen base parameters (embeddings, attention weights, etc.)
    - Trainable LoRA adapters attached to attention layers
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # Base model parameters (frozen)
        self.base_params = {
            "embeddings.weight": MockParameter(
                "embeddings.weight",
                [random.uniform(-0.1, 0.1) for _ in range(256)],
                requires_grad=False,
            ),
            "ln_1.weight": MockParameter(
                "ln_1.weight",
                [1.0] * 64,
                requires_grad=False,
            ),
            "ln_1.bias": MockParameter(
                "ln_1.bias",
                [0.0] * 64,
                requires_grad=False,
            ),
            "attention.q_proj.weight": MockParameter(
                "attention.q_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(256)],
                requires_grad=False,
            ),
            "attention.k_proj.weight": MockParameter(
                "attention.k_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(256)],
                requires_grad=False,
            ),
            "attention.v_proj.weight": MockParameter(
                "attention.v_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(256)],
                requires_grad=False,
            ),
            "attention.o_proj.weight": MockParameter(
                "attention.o_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(256)],
                requires_grad=False,
            ),
            "mlp.up_proj.weight": MockParameter(
                "mlp.up_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(512)],
                requires_grad=False,
            ),
            "mlp.down_proj.weight": MockParameter(
                "mlp.down_proj.weight",
                [random.uniform(-0.02, 0.02) for _ in range(512)],
                requires_grad=False,
            ),
            "ln_f.weight": MockParameter(
                "ln_f.weight",
                [1.0] * 64,
                requires_grad=False,
            ),
            "lm_head.weight": MockParameter(
                "lm_head.weight",
                [random.uniform(-0.02, 0.02) for _ in range(1024)],
                requires_grad=False,
            ),
        }

        # LoRA layers (trainable)
        self.lora_layers = {
            "attention.q_proj": MockLoRALayer("attention.q_proj"),
            "attention.k_proj": MockLoRALayer("attention.k_proj"),
            "attention.v_proj": MockLoRALayer("attention.v_proj"),
            "attention.o_proj": MockLoRALayer("attention.o_proj"),
        }

    def get_base_checksums(self) -> Dict[str, str]:
        """Get checksums for all base parameters."""
        return {name: param.checksum() for name, param in self.base_params.items()}

    def get_lora_checksums(self) -> Dict[str, Dict[str, str]]:
        """Get checksums for all LoRA parameters."""
        return {name: layer.get_checksums() for name, layer in self.lora_layers.items()}

    def get_trainable_params(self) -> List[MockParameter]:
        """Get all trainable parameters (LoRA only)."""
        params = []
        for layer in self.lora_layers.values():
            params.extend(layer.get_params())
        return params

    def get_all_params(self) -> List[MockParameter]:
        """Get all parameters."""
        return list(self.base_params.values()) + self.get_trainable_params()

    def update_lora_params(self, gradients: Dict[str, List[float]], lr: float = 1e-4):
        """
        Update LoRA parameters with gradients.

        Args:
            gradients: Gradient values for each LoRA parameter
            lr: Learning rate
        """
        for layer in self.lora_layers.values():
            for param in layer.get_params():
                if param.name in gradients:
                    grad = gradients[param.name]
                    # Simple SGD update
                    param.values = [
                        v - lr * g for v, g in zip(param.values, grad)
                    ]


class MockLoRATrainingClient:
    """
    Mock training client that tracks parameter updates.

    Ensures only LoRA parameters are updated during training.
    """

    def __init__(self, seed: int = 42):
        self.model = MockModel(seed)
        self.step = 0
        self._update_history: List[Dict[str, Any]] = []
        self._base_checksums_at_init = self.model.get_base_checksums()
        self._lora_checksums_at_init = self.model.get_lora_checksums()

    def forward_backward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute forward-backward pass.

        Returns mock loss and computes gradients only for LoRA params.
        """
        # Compute mock loss
        loss = 2.5 - self.step * 0.01
        loss = max(0.1, loss)

        # Compute mock gradients (only for LoRA params)
        gradients = {}
        for layer in self.model.lora_layers.values():
            for param in layer.get_params():
                # Random gradient scaled by advantage
                avg_advantage = sum(batch.get("advantages", [0.0])) / max(
                    1, len(batch.get("advantages", [0.0]))
                )
                grad = [random.uniform(-0.01, 0.01) * (1 + abs(avg_advantage)) for _ in param.values]
                gradients[param.name] = grad

        # Store gradients for optim_step
        self._pending_gradients = gradients

        return {
            "loss": loss,
            "grad_norm": sum(
                sum(abs(g) for g in grads)
                for grads in gradients.values()
            ) ** 0.5,
            "tokens_processed": len(batch.get("input_ids", [[]])) * len(batch.get("input_ids", [[]])[0]),
        }

    def optim_step(self) -> Dict[str, Any]:
        """Apply optimizer step to LoRA parameters only."""
        if hasattr(self, "_pending_gradients"):
            self.model.update_lora_params(self._pending_gradients, lr=1e-4)
            del self._pending_gradients

        self.step += 1

        # Record update
        self._update_history.append({
            "step": self.step,
            "base_checksums": self.model.get_base_checksums(),
            "lora_checksums": self.model.get_lora_checksums(),
        })

        return {"step": self.step}

    def verify_base_frozen(self) -> bool:
        """Verify that base parameters haven't changed."""
        current_checksums = self.model.get_base_checksums()
        return current_checksums == self._base_checksums_at_init

    def verify_lora_updated(self) -> bool:
        """Verify that LoRA parameters have changed."""
        current_checksums = self.model.get_lora_checksums()
        return current_checksums != self._lora_checksums_at_init

    def get_update_report(self) -> Dict[str, Any]:
        """Get a report of parameter updates."""
        return {
            "total_steps": self.step,
            "base_frozen": self.verify_base_frozen(),
            "lora_updated": self.verify_lora_updated() if self.step > 0 else False,
            "initial_base_checksums": self._base_checksums_at_init,
            "current_base_checksums": self.model.get_base_checksums(),
            "initial_lora_checksums": self._lora_checksums_at_init,
            "current_lora_checksums": self.model.get_lora_checksums(),
        }


class TestLoRAOnlyUpdates:
    """Tests for verifying only LoRA parameters are updated."""

    @pytest.fixture
    def client(self) -> MockLoRATrainingClient:
        """Create a training client."""
        return MockLoRATrainingClient(seed=42)

    @pytest.fixture
    def reinforce(self) -> REINFORCE:
        """Create REINFORCE algorithm."""
        return REINFORCE(REINFORCEConfig())

    @pytest.fixture
    def sampler(self) -> MockRolloutSampler:
        """Create rollout sampler."""
        return MockRolloutSampler(max_new_tokens=16, seed=42)

    def test_base_params_frozen_after_one_step(self, client, reinforce):
        """Test that base parameters remain frozen after one training step."""
        # Record initial checksums
        initial_base = client.model.get_base_checksums()

        # Create a simple batch
        trajectories = [
            Trajectory(
                prompt="Test prompt",
                prompt_tokens=[1, 2, 3],
                response="Test response",
                response_tokens=[4, 5, 6],
                logprobs=[-0.5, -0.3, -0.2],
                attention_mask=[1, 1, 1, 1, 1, 1],
                reward=1.0,
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Run one update
        reinforce.update(batch, client)

        # Verify base params unchanged
        current_base = client.model.get_base_checksums()
        assert current_base == initial_base, "Base parameters should remain frozen"

    def test_lora_params_updated_after_one_step(self, client, reinforce):
        """Test that LoRA parameters are updated after one training step."""
        # Record initial checksums
        initial_lora = client.model.get_lora_checksums()

        # Create a batch with non-zero reward to trigger gradient
        trajectories = [
            Trajectory(
                prompt="Test",
                prompt_tokens=[1],
                response="Response",
                response_tokens=[2, 3],
                logprobs=[-0.5, -0.3],
                attention_mask=[1, 1, 1],
                reward=1.0,
                advantage=1.0,  # Non-zero advantage
            ),
        ]
        batch = TrajectoryBatch(trajectories=trajectories)

        # Run one update
        reinforce.update(batch, client)

        # Verify LoRA params changed
        current_lora = client.model.get_lora_checksums()
        assert current_lora != initial_lora, "LoRA parameters should be updated"

    def test_base_params_frozen_after_many_steps(self, client, reinforce, sampler):
        """Test that base parameters remain frozen after many training steps."""
        # Record initial checksums
        initial_base = client.model.get_base_checksums()

        # Run multiple training iterations
        for _ in range(10):
            batch = sampler.generate_trajectories(["Test prompt"])
            for traj in batch:
                traj.reward = random.uniform(-1, 1)
            reinforce.update(batch, client)

        # Verify base params unchanged
        current_base = client.model.get_base_checksums()
        assert current_base == initial_base, (
            f"Base parameters should remain frozen after {client.step} steps"
        )

    def test_lora_params_progressively_updated(self, client, reinforce, sampler):
        """Test that LoRA parameters are progressively updated over steps."""
        lora_checksums_history = [client.model.get_lora_checksums()]

        # Run training iterations and track LoRA changes
        for _ in range(5):
            batch = sampler.generate_trajectories(["Test"])
            for traj in batch:
                traj.reward = 1.0
            reinforce.update(batch, client)
            lora_checksums_history.append(client.model.get_lora_checksums())

        # Verify LoRA params change between steps
        changes = 0
        for i in range(1, len(lora_checksums_history)):
            if lora_checksums_history[i] != lora_checksums_history[i - 1]:
                changes += 1

        assert changes >= 3, f"LoRA should change in most steps, but only changed {changes}/5 times"

    def test_client_report_shows_frozen_base(self, client, reinforce, sampler):
        """Test that client's update report shows frozen base params."""
        # Run some training
        for _ in range(3):
            batch = sampler.generate_trajectories(["Prompt"])
            for traj in batch:
                traj.reward = 0.5
            reinforce.update(batch, client)

        # Get report
        report = client.get_update_report()

        assert report["base_frozen"] is True
        assert report["total_steps"] == 3

    def test_specific_base_param_unchanged(self, client, reinforce):
        """Test specific base parameters like embeddings remain unchanged."""
        # Get initial embedding checksum
        initial_embed = client.model.base_params["embeddings.weight"].checksum()
        initial_lm_head = client.model.base_params["lm_head.weight"].checksum()

        # Run update
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
        reinforce.update(batch, client)

        # Verify specific params unchanged
        assert client.model.base_params["embeddings.weight"].checksum() == initial_embed
        assert client.model.base_params["lm_head.weight"].checksum() == initial_lm_head

    def test_all_base_layer_types_frozen(self, client, reinforce):
        """Test all types of base layers remain frozen."""
        # Record initial state for different layer types
        layer_types = {
            "embeddings": "embeddings.weight",
            "layer_norm": "ln_1.weight",
            "attention": "attention.q_proj.weight",
            "mlp": "mlp.up_proj.weight",
            "lm_head": "lm_head.weight",
        }
        initial_checksums = {
            name: client.model.base_params[param].checksum()
            for name, param in layer_types.items()
        }

        # Run multiple updates
        for _ in range(5):
            trajectories = [
                Trajectory(
                    prompt=f"P{i}",
                    prompt_tokens=[1],
                    response=f"R{i}",
                    response_tokens=[2],
                    logprobs=[-0.5],
                    attention_mask=[1, 1],
                    reward=random.uniform(-1, 1),
                )
                for i in range(3)
            ]
            batch = TrajectoryBatch(trajectories=trajectories)
            reinforce.update(batch, client)

        # Verify all layer types frozen
        for name, param in layer_types.items():
            current = client.model.base_params[param].checksum()
            assert current == initial_checksums[name], (
                f"{name} layer ({param}) should remain frozen"
            )


class TestLoRAParameterGradients:
    """Tests for LoRA parameter gradient computation."""

    def test_gradients_only_for_trainable_params(self):
        """Test that gradients are only computed for trainable params."""
        client = MockLoRATrainingClient(seed=42)

        # Get list of trainable params
        trainable_params = client.model.get_trainable_params()
        trainable_names = {p.name for p in trainable_params}

        # Verify all trainable params are LoRA params
        for name in trainable_names:
            assert "lora" in name.lower(), f"{name} should be a LoRA parameter"

    def test_base_params_not_trainable(self):
        """Test that base parameters are marked as not trainable."""
        client = MockLoRATrainingClient(seed=42)

        for name, param in client.model.base_params.items():
            assert param.requires_grad is False, (
                f"Base param {name} should have requires_grad=False"
            )

    def test_lora_params_trainable(self):
        """Test that LoRA parameters are marked as trainable."""
        client = MockLoRATrainingClient(seed=42)

        for layer in client.model.lora_layers.values():
            for param in layer.get_params():
                assert param.requires_grad is True, (
                    f"LoRA param {param.name} should have requires_grad=True"
                )


class TestChecksumIntegrity:
    """Tests for checksum computation integrity."""

    def test_checksum_deterministic(self):
        """Test that checksums are deterministic."""
        param = MockParameter("test", [1.0, 2.0, 3.0])
        checksum1 = param.checksum()
        checksum2 = param.checksum()
        assert checksum1 == checksum2

    def test_checksum_changes_with_values(self):
        """Test that checksums change when values change."""
        param = MockParameter("test", [1.0, 2.0, 3.0])
        checksum1 = param.checksum()

        param.values[0] = 1.0001  # Small change
        checksum2 = param.checksum()

        assert checksum1 != checksum2

    def test_model_checksum_consistency(self):
        """Test that model checksums are consistent across instances."""
        model1 = MockModel(seed=42)
        model2 = MockModel(seed=42)

        assert model1.get_base_checksums() == model2.get_base_checksums()
        assert model1.get_lora_checksums() == model2.get_lora_checksums()


class TestIntegrationWithRLVR:
    """Integration tests with full RLVR training loop."""

    def test_full_training_loop_preserves_base(self):
        """Test that a full training loop preserves base parameters."""
        client = MockLoRATrainingClient(seed=42)
        reinforce = REINFORCE(REINFORCEConfig(
            use_baseline=True,
            normalize_advantages=True,
        ))
        sampler = MockRolloutSampler(max_new_tokens=16, seed=42)

        # Record initial base checksums
        initial_base = client.model.get_base_checksums()

        # Simulate full training loop
        prompts = ["Prompt A", "Prompt B", "Prompt C"]
        num_epochs = 3
        steps_per_epoch = 5

        for epoch in range(num_epochs):
            for step in range(steps_per_epoch):
                batch = sampler.generate_trajectories(prompts)
                for traj in batch:
                    # Simple reward based on response length
                    traj.reward = len(traj.response) / 100.0
                reinforce.update(batch, client)

        # Final verification
        final_base = client.model.get_base_checksums()
        total_steps = num_epochs * steps_per_epoch

        assert final_base == initial_base, (
            f"Base parameters should remain frozen after {total_steps} steps"
        )
        assert client.step == total_steps

    def test_checkpointing_preserves_invariant(self):
        """Test that the frozen-base invariant holds across checkpointing."""
        client = MockLoRATrainingClient(seed=42)
        reinforce = REINFORCE(REINFORCEConfig())

        # Train for a few steps
        for i in range(3):
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
            reinforce.update(batch, client)

        # Get state (simulating checkpoint save)
        algo_state = reinforce.get_state()
        report = client.get_update_report()

        # Verify invariant
        assert report["base_frozen"] is True
        assert algo_state["step"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
