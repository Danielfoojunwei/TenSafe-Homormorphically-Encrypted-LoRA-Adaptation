"""
Comprehensive tests for TenSafe Unified Pipeline.

Tests the integrated functionality of:
- Unified configuration system
- Production gates and feature flags
- Function registry (loss/reward)
- HE backend abstraction
- Training pipeline orchestration
- Inference integration
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

# Enable toy HE mode for testing
os.environ["TENSAFE_TOY_HE"] = "1"
os.environ["TENSAFE_ENV"] = "test"


# ==============================================================================
# Configuration Tests
# ==============================================================================


class TestUnifiedConfiguration:
    """Tests for the unified configuration system."""

    def test_default_config_creation(self):
        """Test creating a default configuration."""
        from tensafe.core.config import TenSafeConfig, TrainingMode

        config = TenSafeConfig()

        assert config.version == "1.0"
        assert config.training.mode == TrainingMode.SFT
        assert config.lora.enabled is True
        assert config.dp.enabled is True

    def test_config_with_custom_values(self):
        """Test configuration with custom values."""
        from tensafe.core.config import (
            LoRAConfig,
            ModelConfig,
            TenSafeConfig,
            TrainingConfig,
            TrainingMode,
        )

        config = TenSafeConfig(
            model=ModelConfig(name="custom-model", max_seq_length=1024),
            training=TrainingConfig(mode=TrainingMode.RLVR, total_steps=500),
            lora=LoRAConfig(rank=8, alpha=16.0),
        )

        assert config.model.name == "custom-model"
        assert config.model.max_seq_length == 1024
        assert config.training.mode == TrainingMode.RLVR
        assert config.training.total_steps == 500
        assert config.lora.rank == 8
        assert config.lora.scaling == 2.0  # alpha/rank

    def test_rlvr_config_auto_creation(self):
        """Test that RLVR config is auto-created for RLVR mode."""
        from tensafe.core.config import TenSafeConfig, TrainingConfig, TrainingMode

        config = TenSafeConfig(
            training=TrainingConfig(mode=TrainingMode.RLVR),
        )

        assert config.rlvr is not None
        assert config.rlvr.algorithm == "reinforce"

    def test_config_validation(self):
        """Test configuration validation."""
        from tensafe.core.config import DPConfig, LoRAConfig, TenSafeConfig

        config = TenSafeConfig(
            lora=LoRAConfig(rank=256),  # Very high rank
            dp=DPConfig(target_epsilon=0.5),  # Very tight epsilon
        )

        issues = config.validate()

        assert any("rank" in issue.lower() for issue in issues)
        assert any("epsilon" in issue.lower() for issue in issues)

    def test_config_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        from tensafe.core.config import TenSafeConfig, TrainingConfig, TrainingMode

        original = TenSafeConfig(
            training=TrainingConfig(mode=TrainingMode.SFT, total_steps=1000),
        )

        data = original.to_dict()
        restored = TenSafeConfig.from_dict(data)

        assert restored.training.mode == original.training.mode
        assert restored.training.total_steps == original.training.total_steps

    def test_config_save_and_load(self):
        """Test saving and loading configuration."""
        from tensafe.core.config import (
            TenSafeConfig,
            TrainingConfig,
            TrainingMode,
            load_config,
            save_config,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            original = TenSafeConfig(
                training=TrainingConfig(mode=TrainingMode.SFT),
            )

            save_config(original, config_path)
            loaded = load_config(config_path)

            assert loaded.training.mode == original.training.mode

    def test_create_default_config(self):
        """Test the create_default_config helper."""
        from tensafe.core.config import HEMode, TrainingMode, create_default_config

        config = create_default_config(
            mode=TrainingMode.RLVR,
            model_name="test-model",
            with_dp=True,
            with_he=True,
        )

        assert config.training.mode == TrainingMode.RLVR
        assert config.model.name == "test-model"
        assert config.dp.enabled is True
        assert config.he.mode == HEMode.PRODUCTION


# ==============================================================================
# Production Gates Tests
# ==============================================================================


class TestProductionGates:
    """Tests for production gating and feature flags."""

    def test_gate_allowed_by_env(self):
        """Test gate allowed via environment variable."""
        from tensafe.core.gates import FeatureGate, GateStatus

        gate = FeatureGate(
            name="test_gate",
            default_allowed=False,
            env_var="TEST_GATE_ENABLED",
        )

        # Without env var
        assert gate.check() == GateStatus.DENIED

        # With env var
        with patch.dict(os.environ, {"TEST_GATE_ENABLED": "1"}):
            assert gate.check() == GateStatus.ALLOWED

    def test_gate_denied_in_production(self):
        """Test gate denied in production environment."""
        from tensafe.core.gates import FeatureGate, GateStatus

        gate = FeatureGate(
            name="dev_only_gate",
            default_allowed=True,
            production_allowed=False,
        )

        # In test environment
        with patch.dict(os.environ, {"TENSAFE_ENV": "test"}):
            assert gate.check() == GateStatus.ALLOWED

        # In production environment
        with patch.dict(os.environ, {"TENSAFE_ENV": "production"}):
            assert gate.check() == GateStatus.DENIED

    def test_production_gates_toy_he(self):
        """Test the TOY_HE production gate."""
        from tensafe.core.gates import GateStatus, ProductionGates

        # Should be allowed in test env with TENSAFE_TOY_HE=1
        status = ProductionGates.TOY_HE.check()
        assert status in (GateStatus.ALLOWED, GateStatus.AUDIT)

    def test_require_gate_decorator(self):
        """Test the require_gate decorator."""
        from tensafe.core.gates import FeatureGate, GateDeniedError, require_gate

        gate = FeatureGate(
            name="required_gate",
            default_allowed=False,
        )

        @require_gate(gate)
        def protected_function():
            return "success"

        with pytest.raises(GateDeniedError):
            protected_function()

    def test_production_check(self):
        """Test comprehensive production check."""
        from tensafe.core.config import HEConfig, HEMode, TenSafeConfig
        from tensafe.core.gates import production_check

        # Valid config
        config = TenSafeConfig()
        result = production_check(config)
        assert result.valid

        # Config with toy HE in production
        config_with_toy = TenSafeConfig(
            he=HEConfig(mode=HEMode.TOY),
        )

        with patch.dict(os.environ, {"TENSAFE_ENV": "production"}):
            result = production_check(config_with_toy)
            assert not result.valid


# ==============================================================================
# Function Registry Tests
# ==============================================================================


class TestFunctionRegistry:
    """Tests for the unified function registry."""

    def test_loss_registry_registration(self):
        """Test registering loss functions."""
        from tensafe.core.registry import get_loss_registry, register_function

        registry = get_loss_registry()

        @register_function("test_loss", registry="loss")
        def test_loss(outputs, batch, **kwargs):
            return {"loss": 0.5}

        assert "test_loss" in registry

    def test_loss_resolution_by_name(self):
        """Test resolving loss function by name."""
        from tensafe.core.registry import resolve_function

        loss_fn = resolve_function("token_ce", registry="loss")
        assert callable(loss_fn)

    def test_reward_registry(self):
        """Test the reward registry."""
        from tensafe.core.registry import get_reward_registry

        registry = get_reward_registry()

        # Built-in rewards should be registered
        assert "keyword_contains" in registry
        assert "length_penalty" in registry

    def test_reward_resolution(self):
        """Test resolving reward functions."""
        from tensafe.core.registry import resolve_function

        reward_fn = resolve_function("keyword_contains", registry="reward")
        result = reward_fn("prompt", "response with keyword", keywords=["keyword"])
        assert result > 0

    def test_function_with_default_kwargs(self):
        """Test resolving function with default kwargs."""
        from tensafe.core.registry import resolve_function

        reward_fn = resolve_function(
            "keyword_contains",
            registry="reward",
            positive_reward=2.0,
            negative_reward=-1.0,
        )

        # Should use bound kwargs
        result = reward_fn("prompt", "response with keyword", keywords=["keyword"])
        assert result == 2.0


# ==============================================================================
# HE Backend Tests
# ==============================================================================


class TestHEBackendInterface:
    """Tests for the unified HE backend interface."""

    def test_toy_backend_creation(self):
        """Test creating toy HE backend."""
        from tensafe.core.he_interface import HEParams, ToyHEBackend

        backend = ToyHEBackend(HEParams())
        backend.setup()

        assert backend.is_setup
        assert not backend.is_production_ready

    def test_toy_backend_encrypt_decrypt(self):
        """Test encrypt/decrypt with toy backend."""
        from tensafe.core.he_interface import HEParams, ToyHEBackend

        backend = ToyHEBackend(HEParams())
        backend.setup()

        plaintext = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(plaintext)
        decrypted = backend.decrypt(ct)

        np.testing.assert_array_almost_equal(plaintext, decrypted)

    def test_toy_backend_lora_delta(self):
        """Test LoRA delta computation with toy backend."""
        from tensafe.core.he_interface import HEParams, ToyHEBackend

        backend = ToyHEBackend(HEParams())
        backend.setup()

        # Create test data
        x = np.random.randn(64).astype(np.float64)
        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01

        ct_x = backend.encrypt(x)
        ct_result = backend.lora_delta(ct_x, lora_a, lora_b, scaling=0.5)
        result = backend.decrypt(ct_result)

        # Compare with plaintext computation
        expected = 0.5 * (x @ lora_a.T @ lora_b.T)
        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_get_backend_auto(self):
        """Test auto-selecting backend."""
        from tensafe.core.he_interface import HEBackendType, get_backend

        backend = get_backend(HEBackendType.AUTO)
        assert backend.is_setup

    def test_list_available_backends(self):
        """Test listing available backends."""
        from tensafe.core.he_interface import list_available_backends

        available = list_available_backends()
        assert isinstance(available, list)
        # Simulation should be available in test env (replaces legacy "toy")
        assert "simulation" in available


# ==============================================================================
# Pipeline Tests
# ==============================================================================


class TestUnifiedPipeline:
    """Tests for the unified training pipeline."""

    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        from tensafe.core.config import TenSafeConfig
        from tensafe.core.pipeline import PipelineState, TenSafePipeline

        config = TenSafeConfig()
        pipeline = TenSafePipeline(config, validate_production=False)

        assert pipeline.state == PipelineState.INITIALIZED

    def test_pipeline_setup(self):
        """Test pipeline setup."""
        from tensafe.core.config import TenSafeConfig
        from tensafe.core.pipeline import TenSafePipeline

        config = TenSafeConfig()
        pipeline = TenSafePipeline(config, validate_production=False)
        pipeline.setup()

        assert pipeline._training_mode is not None

    def test_pipeline_sft_training(self):
        """Test SFT training with pipeline."""
        from tensafe.core.config import TenSafeConfig, TrainingConfig
        from tensafe.core.pipeline import PipelineState, TenSafePipeline

        config = TenSafeConfig(
            training=TrainingConfig(total_steps=10, log_interval=5),
        )

        pipeline = TenSafePipeline(config, validate_production=False)
        pipeline.setup()

        result = pipeline.train()

        assert result.success
        assert result.total_steps >= 9  # May be 9 due to 0-indexing
        assert pipeline.state == PipelineState.COMPLETED

    def test_pipeline_rlvr_training(self):
        """Test RLVR training with pipeline."""
        from tensafe.core.config import TenSafeConfig, TrainingConfig, TrainingMode
        from tensafe.core.pipeline import TenSafePipeline

        config = TenSafeConfig(
            training=TrainingConfig(mode=TrainingMode.RLVR, total_steps=5),
        )

        pipeline = TenSafePipeline(config, validate_production=False)
        pipeline.setup()

        result = pipeline.train()

        assert result.success

    def test_pipeline_callbacks(self):
        """Test pipeline event callbacks."""
        from tensafe.core.config import TenSafeConfig, TrainingConfig
        from tensafe.core.pipeline import PipelineEvent, TenSafePipeline

        events_received = []

        def callback(event: PipelineEvent, payload: Dict[str, Any]):
            events_received.append(event)

        config = TenSafeConfig(
            training=TrainingConfig(total_steps=5),
        )

        pipeline = TenSafePipeline(config, validate_production=False)
        pipeline.register_callback(callback)
        pipeline.setup()
        pipeline.train()

        assert PipelineEvent.STATE_CHANGE in events_received
        assert PipelineEvent.STEP_START in events_received
        assert PipelineEvent.STEP_END in events_received

    def test_pipeline_from_config_file(self):
        """Test creating pipeline from config file."""
        from tensafe.core.config import TenSafeConfig, save_config
        from tensafe.core.pipeline import TenSafePipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            config = TenSafeConfig()
            save_config(config, config_path)

            pipeline = TenSafePipeline.from_config(config_path)
            assert pipeline is not None


# ==============================================================================
# Inference Tests
# ==============================================================================


class TestUnifiedInference:
    """Tests for the unified inference module."""

    def test_inference_creation(self):
        """Test creating inference engine."""
        from tensafe.core.inference import InferenceMode, TenSafeInference

        inference = TenSafeInference(mode=InferenceMode.PLAINTEXT)
        assert inference._mode == InferenceMode.PLAINTEXT

    def test_inference_forward_no_lora(self):
        """Test forward pass without LoRA."""
        from tensafe.core.inference import InferenceMode, TenSafeInference

        inference = TenSafeInference(mode=InferenceMode.NONE)

        x = np.random.randn(64).astype(np.float64)
        result = inference.forward(x)

        np.testing.assert_array_equal(result.output, x)

    def test_inference_forward_plaintext_lora(self):
        """Test forward pass with plaintext LoRA."""
        from tensafe.core.inference import InferenceMode, TenSafeInference

        inference = TenSafeInference(mode=InferenceMode.PLAINTEXT)

        # Register LoRA weights
        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        inference.register_lora_weights("q_proj", lora_a, lora_b)

        x = np.random.randn(64).astype(np.float64)
        result = inference.forward(x, module_name="q_proj")

        assert result.output is not None
        assert result.lora_time_ms > 0

    def test_inference_forward_he_lora(self):
        """Test forward pass with HE LoRA."""
        from tensafe.core.inference import InferenceMode, TenSafeInference

        inference = TenSafeInference(mode=InferenceMode.HE_ONLY)

        # Register LoRA weights
        lora_a = np.random.randn(8, 64).astype(np.float64) * 0.01
        lora_b = np.random.randn(64, 8).astype(np.float64) * 0.01
        inference.register_lora_weights("q_proj", lora_a, lora_b)

        x = np.random.randn(64).astype(np.float64)
        result = inference.forward(x, module_name="q_proj")

        assert result.output is not None
        assert result.mode == "he_only"
        assert result.he_metrics is not None

    def test_inference_generate(self):
        """Test text generation."""
        from tensafe.core.inference import TenSafeInference

        inference = TenSafeInference()
        result = inference.generate("Hello, world!")

        assert result.text is not None
        assert result.tokens is not None
        assert len(result.tokens) > 0

    def test_inference_batch_generate(self):
        """Test batch text generation."""
        from tensafe.core.inference import TenSafeInference

        inference = TenSafeInference()
        prompts = ["Hello!", "How are you?", "What is AI?"]

        batch_result = inference.generate_batch(prompts)

        assert len(batch_result.results) == 3
        assert batch_result.total_time_ms > 0


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests for the unified pipeline."""

    def test_full_sft_workflow(self):
        """Test complete SFT workflow."""
        from tensafe.core.config import (
            TenSafeConfig,
            TrainingConfig,
            load_config,
            save_config,
        )
        from tensafe.core.pipeline import TenSafePipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save config
            config = TenSafeConfig(
                training=TrainingConfig(
                    total_steps=5,
                    output_dir=tmpdir,
                    save_interval=3,
                ),
            )

            config_path = Path(tmpdir) / "config.yaml"
            save_config(config, config_path)

            # Load and train
            loaded_config = load_config(config_path)
            pipeline = TenSafePipeline(loaded_config, validate_production=False)
            pipeline.setup()

            result = pipeline.train()

            assert result.success

    def test_full_rlvr_workflow(self):
        """Test complete RLVR workflow."""
        from tensafe.core.config import (
            RLVRConfig,
            TenSafeConfig,
            TrainingConfig,
            TrainingMode,
        )
        from tensafe.core.pipeline import TenSafePipeline

        config = TenSafeConfig(
            training=TrainingConfig(
                mode=TrainingMode.RLVR,
                total_steps=3,
            ),
            rlvr=RLVRConfig(
                algorithm="reinforce",
                reward_fn="keyword_contains",
            ),
        )

        pipeline = TenSafePipeline(config, validate_production=False)
        pipeline.setup()

        result = pipeline.train()

        assert result.success

    def test_he_integration(self):
        """Test HE integration in pipeline."""
        from tensafe.core.config import HEConfig, HEMode, TenSafeConfig
        from tensafe.core.he_interface import HEBackendType, get_backend

        config = TenSafeConfig(
            he=HEConfig(mode=HEMode.TOY),
        )

        # Get backend
        backend = get_backend(HEBackendType.TOY)

        # Test operations
        x = np.array([1.0, 2.0, 3.0])
        ct = backend.encrypt(x)
        decrypted = backend.decrypt(ct)

        np.testing.assert_array_almost_equal(x, decrypted)


# ==============================================================================
# CLI Tests
# ==============================================================================


class TestCLI:
    """Tests for the CLI module."""

    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess

        result = subprocess.run(
            ["python", "-m", "tensafe.cli", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/user/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation",
        )

        assert result.returncode == 0
        assert "TenSafe" in result.stdout

    def test_cli_config_create(self):
        """Test CLI config create command."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_config.yaml"

            result = subprocess.run(
                [
                    "python", "-m", "tensafe.cli",
                    "config", "create",
                    "--mode", "sft",
                    "--output", str(output_path),
                ],
                capture_output=True,
                text=True,
                cwd="/home/user/TenSafe-Homormorphically-Encrypted-LoRA-Adaptation",
            )

            assert result.returncode == 0
            assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
