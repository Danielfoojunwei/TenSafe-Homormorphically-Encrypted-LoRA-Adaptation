"""Integration Tests for Competitive Feature Implementations.

Tests the new integrations implemented in the competitive analysis:
- vLLM backend integration
- Ray Train distributed training
- OpenTelemetry observability
- W&B and MLflow callbacks
- HuggingFace Hub integration
- Kernel optimizations (Liger)
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


# ============================================================================
# vLLM Backend Tests
# ============================================================================

class TestVLLMBackendConfig:
    """Tests for vLLM backend configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig, HESchemeType, CKKSProfile

        config = TenSafeVLLMConfig(model_path="test-model")

        assert config.model_path == "test-model"
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.9
        assert config.enable_he_lora == False  # No TSSP package
        assert config.he_scheme == HESchemeType.CKKS
        assert config.ckks_profile == CKKSProfile.FAST

    def test_config_with_tssp(self):
        """Test configuration with TSSP package."""
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        config = TenSafeVLLMConfig(
            model_path="test-model",
            tssp_package_path="/path/to/package.tssp",
            enable_he_lora=True,
        )

        assert config.enable_he_lora == True
        assert config.tssp_package_path == "/path/to/package.tssp"

    def test_config_validation(self):
        """Test configuration validation."""
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        with pytest.raises(ValueError):
            TenSafeVLLMConfig(
                model_path="test-model",
                gpu_memory_utilization=1.5,  # Invalid: > 1
            )

    def test_to_vllm_args(self):
        """Test conversion to vLLM arguments."""
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        config = TenSafeVLLMConfig(
            model_path="test-model",
            tensor_parallel_size=2,
            max_model_len=4096,
        )

        args = config.to_vllm_args()

        assert args["model"] == "test-model"
        assert args["tensor_parallel_size"] == 2
        assert args["max_model_len"] == 4096


class TestHELoRAHooks:
    """Tests for HE-LoRA forward hooks."""

    def test_hook_creation(self):
        """Test creating HE-LoRA hook."""
        from tensorguard.backends.vllm.hooks import HELoRAHook, HELoRAConfig
        import torch

        config = HELoRAConfig(
            hidden_size=4096,
            rank=16,
            alpha=32.0,
        )

        lora_a = torch.randn(16, 4096)
        lora_b = torch.zeros(4096, 16)

        hook = HELoRAHook(
            layer_name="test_layer",
            lora_a=lora_a,
            lora_b=lora_b,
            config=config,
        )

        assert hook.layer_name == "test_layer"
        assert hook.scaling == 32.0 / 16  # alpha / rank

    def test_hook_application(self):
        """Test applying HE-LoRA hook."""
        from tensorguard.backends.vllm.hooks import HELoRAHook, HELoRAConfig
        import torch

        config = HELoRAConfig(hidden_size=64, rank=4, alpha=8.0)

        lora_a = torch.randn(4, 64) * 0.01
        lora_b = torch.randn(64, 4) * 0.01

        hook = HELoRAHook(
            layer_name="test",
            lora_a=lora_a,
            lora_b=lora_b,
            config=config,
        )

        # Create mock module and inputs
        module = Mock()
        input_tensor = torch.randn(2, 8, 64)  # batch, seq, hidden
        output_tensor = torch.randn(2, 8, 64)

        # Apply hook
        result = hook(module, (input_tensor,), output_tensor)

        # Should return modified tensor
        assert result.shape == output_tensor.shape
        # Result should be different from original output
        assert not torch.allclose(result, output_tensor)

    def test_hook_manager(self):
        """Test HE-LoRA hook manager."""
        from tensorguard.backends.vllm.hooks import HELoRAHookManager, HELoRAConfig
        import torch
        import torch.nn as nn

        config = HELoRAConfig(hidden_size=64, rank=4)
        manager = HELoRAHookManager(config, target_modules=["linear"])

        # Create simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 64)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        # Create LoRA weights
        lora_weights = {
            "linear": (torch.randn(4, 64), torch.randn(64, 4))
        }

        # Register hooks
        num_hooks = manager.register_hooks(model, lora_weights)

        assert num_hooks == 1

        # Cleanup
        manager.remove_hooks()


class TestVLLMEngine:
    """Tests for TenSafe vLLM engine."""

    def test_engine_initialization(self):
        """Test engine initialization."""
        from tensorguard.backends.vllm.engine import TenSafeVLLMEngine
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        config = TenSafeVLLMConfig(model_path="test-model")

        engine = TenSafeVLLMEngine(config)

        assert engine.config == config
        assert engine._initialized == True

    def test_simulated_generation(self):
        """Test simulated generation (no vLLM)."""
        from tensorguard.backends.vllm.engine import TenSafeVLLMEngine
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        config = TenSafeVLLMConfig(model_path="test-model")
        engine = TenSafeVLLMEngine(config)

        results = engine.generate(["Hello, world!"])

        assert len(results) == 1
        assert results[0].prompt == "Hello, world!"
        assert len(results[0].outputs) > 0

    def test_engine_metrics(self):
        """Test engine metrics collection."""
        from tensorguard.backends.vllm.engine import TenSafeVLLMEngine
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig

        config = TenSafeVLLMConfig(model_path="test-model")
        engine = TenSafeVLLMEngine(config)

        # Generate some requests
        engine.generate(["Test 1", "Test 2"])

        metrics = engine.get_metrics()

        assert "uptime_seconds" in metrics
        assert metrics["total_requests"] == 2


# ============================================================================
# Ray Train Tests
# ============================================================================

class TestRayTrainConfig:
    """Tests for Ray Train configuration."""

    def test_config_defaults(self):
        """Test default configuration."""
        from tensorguard.distributed.ray_trainer import TenSafeRayConfig

        config = TenSafeRayConfig()

        assert config.num_workers == 4
        assert config.use_gpu == True
        assert config.batch_size_per_worker == 8
        assert config.secure_aggregation == True

    def test_config_with_dp(self):
        """Test configuration with differential privacy."""
        from tensorguard.distributed.ray_trainer import TenSafeRayConfig
        from tensorguard.distributed.dp_distributed import DPAccountingResult

        # Create DP config
        @dataclass
        class MockDPConfig:
            enabled: bool = True
            noise_multiplier: float = 1.0
            max_grad_norm: float = 1.0
            target_epsilon: float = 8.0
            target_delta: float = 1e-5

        config = TenSafeRayConfig(
            num_workers=2,
            dp_config=MockDPConfig(),
        )

        assert config.dp_config.enabled == True
        assert config.dp_config.target_epsilon == 8.0


class TestDistributedDP:
    """Tests for distributed DP-SGD."""

    def test_dp_optimizer(self):
        """Test DP optimizer wrapper."""
        from tensorguard.distributed.dp_distributed import DistributedDPOptimizer
        import torch

        # Create simple model and optimizer
        model = torch.nn.Linear(10, 5)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dp_optimizer = DistributedDPOptimizer(
            base_optimizer=base_optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            num_workers=2,
        )

        # Run a step
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.sum()
        loss.backward()
        dp_optimizer.step()

        # Check privacy accounting
        epsilon, delta = dp_optimizer.get_privacy_spent()

        assert epsilon > 0
        assert delta == 1e-5

    def test_secure_aggregator(self):
        """Test secure gradient aggregation."""
        from tensorguard.distributed.dp_distributed import SecureGradientAggregator
        import torch

        aggregator = SecureGradientAggregator(num_workers=3)

        # Setup pairwise masks
        seeds_0 = aggregator.setup_pairwise_masks(0)
        seeds_1 = aggregator.setup_pairwise_masks(1)
        seeds_2 = aggregator.setup_pairwise_masks(2)

        # Create gradients
        grad_0 = torch.ones(10)
        grad_1 = torch.ones(10) * 2
        grad_2 = torch.ones(10) * 3

        # Mask gradients
        masked_0 = aggregator.mask_gradient(grad_0, 0, seeds_0)
        masked_1 = aggregator.mask_gradient(grad_1, 1, seeds_1)
        masked_2 = aggregator.mask_gradient(grad_2, 2, seeds_2)

        # Aggregate
        result = aggregator.aggregate_gradients(
            [masked_0, masked_1, masked_2],
            [0, 1, 2]
        )

        # Should be close to sum of original gradients
        expected = grad_0 + grad_1 + grad_2
        assert torch.allclose(result, expected, atol=1e-5)


# ============================================================================
# Observability Tests
# ============================================================================

class TestObservabilitySetup:
    """Tests for observability setup."""

    def test_metrics_creation(self):
        """Test creating TenSafe metrics."""
        from tensorguard.observability.setup import TenSafeMetrics, ObservabilityConfig

        config = ObservabilityConfig(
            service_name="test-service",
            metrics_enabled=True,
        )

        metrics = TenSafeMetrics(config)

        # Record some metrics
        metrics.record_inference_latency(0.1)
        metrics.record_inference_tokens(100)
        metrics.record_privacy_budget(5.0, 1e-5, "training-1")

        summary = metrics.get_summary()

        assert "counters" in summary
        assert "histograms" in summary
        assert summary["counters"]["inference_tokens"] == 100

    def test_he_lora_metrics(self):
        """Test HE-LoRA specific metrics."""
        from tensorguard.observability.setup import TenSafeMetrics

        metrics = TenSafeMetrics()

        metrics.record_he_lora_latency(0.001, layer="q_proj")
        metrics.record_he_lora_operation(layer="q_proj")

        summary = metrics.get_summary()

        assert len(summary["histograms"]["he_lora_latency"]) == 1


class TestTracingMiddleware:
    """Tests for tracing middleware."""

    def test_middleware_creation(self):
        """Test creating tracing middleware."""
        from tensorguard.observability.middleware import TenSafeTracingMiddleware

        app = Mock()
        middleware = TenSafeTracingMiddleware(
            app,
            service_name="test",
            redact_sensitive=True,
        )

        assert middleware.service_name == "test"
        assert middleware.redact_sensitive == True

    def test_sensitive_detection(self):
        """Test sensitive field detection."""
        from tensorguard.observability.middleware import TenSafeTracingMiddleware

        middleware = TenSafeTracingMiddleware(Mock())

        assert middleware._is_sensitive("Authorization") == True
        assert middleware._is_sensitive("X-API-Key") == True
        assert middleware._is_sensitive("Content-Type") == False


# ============================================================================
# MLOps Integration Tests
# ============================================================================

class TestWandbCallback:
    """Tests for W&B callback."""

    def test_callback_creation(self):
        """Test creating W&B callback."""
        from tensorguard.integrations.wandb_callback import TenSafeWandbCallback, TenSafeWandbConfig

        config = TenSafeWandbConfig(project="test-project")
        callback = TenSafeWandbCallback(config)

        assert callback.config.project == "test-project"
        assert callback.config.log_model_weights == False

    def test_redaction(self):
        """Test config redaction."""
        from tensorguard.integrations.wandb_callback import TenSafeWandbCallback

        callback = TenSafeWandbCallback()

        config = {
            "model": "llama",
            "api_key": "secret123",
            "password": "pass123",
        }

        redacted = callback._redact_config(config)

        assert redacted["model"] == "llama"
        assert redacted["api_key"] == "[REDACTED]"
        assert redacted["password"] == "[REDACTED]"


class TestMLflowCallback:
    """Tests for MLflow callback."""

    def test_callback_creation(self):
        """Test creating MLflow callback."""
        from tensorguard.integrations.mlflow_callback import TenSafeMLflowCallback, TenSafeMLflowConfig

        config = TenSafeMLflowConfig(experiment_name="test-exp")
        callback = TenSafeMLflowCallback(config)

        assert callback.config.experiment_name == "test-exp"


class TestHFHubIntegration:
    """Tests for HuggingFace Hub integration."""

    def test_integration_creation(self):
        """Test creating HF Hub integration."""
        from tensorguard.integrations.hf_hub import TenSafeHFHubIntegration, TenSafeHFHubConfig

        config = TenSafeHFHubConfig(private=True)
        hub = TenSafeHFHubIntegration(config)

        assert hub.config.private == True
        assert hub.config.include_encrypted_weights == False

    def test_model_card_generation(self):
        """Test model card generation."""
        from tensorguard.integrations.hf_hub import TenSafeHFHubIntegration

        hub = TenSafeHFHubIntegration()

        manifest_data = {
            "name": "test-model",
            "base_model": "llama-3-8b",
            "package_id": "pkg-123",
        }

        privacy_info = {
            "epsilon": 8.0,
            "delta": 1e-5,
        }

        model_card = hub._generate_model_card(
            manifest_data=manifest_data,
            privacy_info=privacy_info,
            repo_id="user/test-model",
        )

        assert "test-model" in model_card
        assert "Privacy Information" in model_card
        assert "8.0" in model_card


# ============================================================================
# Kernel Optimization Tests
# ============================================================================

class TestLigerIntegration:
    """Tests for Liger kernel integration."""

    def test_config_creation(self):
        """Test creating Liger config."""
        from tensorguard.optimizations.liger_integration import LigerOptimizationConfig

        config = LigerOptimizationConfig(
            enable_rope=True,
            enable_rms_norm=True,
        )

        assert config.enable_rope == True
        assert config.model_type == "auto"

    def test_model_type_detection(self):
        """Test model type detection."""
        from tensorguard.optimizations.liger_integration import _detect_model_type
        import torch.nn as nn

        class LlamaModel(nn.Module):
            pass

        class MistralModel(nn.Module):
            pass

        assert _detect_model_type(LlamaModel()) == "llama"
        assert _detect_model_type(MistralModel()) == "mistral"


class TestTrainingOptimizations:
    """Tests for training optimizations."""

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing application."""
        from tensorguard.optimizations.training_optimizations import apply_gradient_checkpointing
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.checkpointing_enabled = False

            def gradient_checkpointing_enable(self):
                self.checkpointing_enabled = True

        model = MockModel()
        apply_gradient_checkpointing(model)

        assert model.checkpointing_enabled == True

    def test_optimized_trainer(self):
        """Test optimized trainer creation."""
        from tensorguard.optimizations.training_optimizations import (
            TenSafeOptimizedTrainer,
            TrainingOptimizationConfig,
        )
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset

        # Create simple model and dataset
        model = nn.Linear(10, 5)
        dataset = TensorDataset(
            torch.randn(100, 10),
            torch.randn(100, 5)
        )

        config = TrainingOptimizationConfig(
            mixed_precision=False,
            gradient_checkpointing=False,
        )

        trainer = TenSafeOptimizedTrainer(
            model=model,
            train_dataset=dataset,
            config=config,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None


# ============================================================================
# End-to-End Integration Tests
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_inference_pipeline(self):
        """Test full inference pipeline with privacy features."""
        from tensorguard.backends.vllm.engine import TenSafeVLLMEngine
        from tensorguard.backends.vllm.config import TenSafeVLLMConfig
        from tensorguard.observability.setup import TenSafeMetrics

        # Setup
        config = TenSafeVLLMConfig(model_path="test-model")
        engine = TenSafeVLLMEngine(config)
        metrics = TenSafeMetrics()

        # Generate
        results = engine.generate(["Test prompt 1", "Test prompt 2"])

        # Record metrics
        for r in results:
            metrics.record_inference_request("success")
            metrics.record_inference_tokens(50)

        # Verify
        assert len(results) == 2
        summary = metrics.get_summary()
        assert summary["counters"]["inference_tokens"] == 100

    def test_privacy_aware_training_simulation(self):
        """Test privacy-aware training simulation."""
        from tensorguard.distributed.dp_distributed import DistributedDPOptimizer
        from tensorguard.integrations.wandb_callback import TenSafeWandbCallback
        import torch

        # Setup model
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.AdamW(model.parameters())

        dp_optimizer = DistributedDPOptimizer(
            base_optimizer=optimizer,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        callback = TenSafeWandbCallback()

        # Simulate training steps
        for step in range(5):
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            dp_optimizer.step()
            dp_optimizer.zero_grad()

        # Check privacy budget
        epsilon, delta = dp_optimizer.get_privacy_spent()
        assert epsilon > 0

        # Check accounting
        result = dp_optimizer.get_accounting_result()
        assert result.steps == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
