"""
Tests for LoRA best practices configuration.
"""

import math

import pytest

from tensafe.lora_best_practices.config import (
    TARGET_MODULE_PRESETS,
    LoRABestPracticesConfig,
    LoRAScalingMethod,
    LoRAVariant,
    TargetModulePreset,
    create_optimal_config,
)


class TestLoRABestPracticesConfig:
    """Tests for LoRABestPracticesConfig."""

    def test_default_values(self):
        """Test that defaults follow research best practices."""
        config = LoRABestPracticesConfig()

        # Research-backed defaults
        assert config.rank == 32
        assert config.alpha == 64.0  # 2 * rank
        assert config.dropout == 0.0  # Research shows 0 is fine
        assert config.target_preset == TargetModulePreset.ALL_LINEAR
        assert config.use_rslora is True  # For rank > 32
        assert config.learning_rate == 2e-4  # 10x full FT
        assert config.lora_plus_ratio == 16.0

    def test_alpha_defaults_to_2x_rank(self):
        """Test alpha defaults to 2 * rank."""
        for rank in [8, 16, 32, 64, 128]:
            config = LoRABestPracticesConfig(rank=rank)
            assert config.alpha == 2.0 * rank

    def test_explicit_alpha_preserved(self):
        """Test explicitly set alpha is not overwritten."""
        config = LoRABestPracticesConfig(rank=32, alpha=128.0)
        assert config.alpha == 128.0

    def test_rslora_auto_enabled_for_high_rank(self):
        """Test rsLoRA is automatically enabled for rank > 32."""
        config_low = LoRABestPracticesConfig(rank=16, use_rslora=False)
        # For low rank, user preference is respected
        assert config_low.use_rslora is False

        config_high = LoRABestPracticesConfig(rank=64, use_rslora=False)
        # For high rank, rsLoRA is auto-enabled
        assert config_high.use_rslora is True

    def test_scaling_factor_standard(self):
        """Test standard scaling: α/r."""
        config = LoRABestPracticesConfig(
            rank=32,
            alpha=64.0,
            use_rslora=False,
            scaling_method=LoRAScalingMethod.STANDARD,
        )
        expected = 64.0 / 32  # 2.0
        assert abs(config.scaling_factor - expected) < 1e-6

    def test_scaling_factor_rslora(self):
        """Test rsLoRA scaling: α/√r."""
        config = LoRABestPracticesConfig(
            rank=32,
            alpha=64.0,
            use_rslora=True,
        )
        expected = 64.0 / math.sqrt(32)
        assert abs(config.scaling_factor - expected) < 1e-6

    def test_target_modules_from_preset(self):
        """Test target modules are correctly set from preset."""
        config = LoRABestPracticesConfig(
            target_preset=TargetModulePreset.ALL_LINEAR,
            architecture="llama",
        )
        expected = TARGET_MODULE_PRESETS["llama"][TargetModulePreset.ALL_LINEAR]
        assert config.target_modules == expected

    def test_all_linear_includes_mlp(self):
        """Test ALL_LINEAR preset includes MLP layers."""
        config = LoRABestPracticesConfig(
            target_preset=TargetModulePreset.ALL_LINEAR,
            architecture="llama",
        )
        # MLP layers
        assert "gate_proj" in config.target_modules
        assert "up_proj" in config.target_modules
        assert "down_proj" in config.target_modules
        # Attention layers
        assert "q_proj" in config.target_modules
        assert "k_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert "o_proj" in config.target_modules

    def test_to_peft_config(self):
        """Test conversion to PEFT config format."""
        config = LoRABestPracticesConfig(rank=32)
        peft_config = config.to_peft_config()

        assert peft_config["r"] == 32
        assert peft_config["lora_alpha"] == 64.0
        assert peft_config["lora_dropout"] == 0.0
        assert peft_config["use_rslora"] is True
        assert "target_modules" in peft_config

    def test_serialization_round_trip(self):
        """Test serialization and deserialization."""
        original = LoRABestPracticesConfig(
            rank=64,
            alpha=128.0,
            dropout=0.1,
            use_dora=True,
        )
        serialized = original.to_dict()
        restored = LoRABestPracticesConfig.from_dict(serialized)

        assert restored.rank == original.rank
        assert restored.alpha == original.alpha
        assert restored.dropout == original.dropout
        assert restored.use_dora == original.use_dora

    def test_dora_variant_enables_use_dora(self):
        """Test DoRA variant automatically sets use_dora."""
        config = LoRABestPracticesConfig(variant=LoRAVariant.DORA)
        assert config.use_dora is True

    def test_learning_rates(self):
        """Test LoRA+ learning rate calculation."""
        config = LoRABestPracticesConfig(
            learning_rate=2e-4,
            lora_plus_ratio=16.0,
        )
        lrs = config.get_learning_rates()

        assert lrs["lora_A"] == 2e-4
        assert lrs["lora_B"] == 2e-4 * 16.0
        assert lrs["default"] == 2e-4

    def test_validation_rejects_invalid_rank(self):
        """Test validation rejects invalid rank."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoRABestPracticesConfig(rank=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            LoRABestPracticesConfig(rank=-1)

    def test_validation_rejects_invalid_alpha(self):
        """Test validation rejects invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoRABestPracticesConfig(rank=32, alpha=0)

    def test_estimate_trainable_params(self):
        """Test trainable parameter estimation."""
        config = LoRABestPracticesConfig(rank=32)
        # For LLaMA 7B: hidden=4096, layers=32
        params = config.estimate_trainable_params(hidden_size=4096, num_layers=32)

        # Each adapter: 2 * rank * hidden = 2 * 32 * 4096 = 262144
        # 7 modules per layer, 32 layers = 224 adapters
        expected = 2 * 32 * 4096 * 7 * 32
        assert params == expected


class TestCreateOptimalConfig:
    """Tests for create_optimal_config utility."""

    def test_small_dataset(self):
        """Test config for small dataset."""
        config = create_optimal_config(
            dataset_size=5000,
            task_complexity="simple",
        )
        assert config.rank <= 16  # Low rank for small dataset

    def test_medium_dataset(self):
        """Test config for medium dataset."""
        config = create_optimal_config(
            dataset_size=50000,
            task_complexity="medium",
        )
        assert 16 <= config.rank <= 64

    def test_large_dataset(self):
        """Test config for large dataset."""
        config = create_optimal_config(
            dataset_size=200000,
            task_complexity="complex",
        )
        assert config.rank >= 64

    def test_memory_constraint_enables_qlora(self):
        """Test memory constraint enables QLoRA."""
        config = create_optimal_config(
            dataset_size=50000,
            memory_constraint_gb=12.0,
        )
        assert config.use_4bit is True

    def test_respects_architecture(self):
        """Test architecture is preserved."""
        config = create_optimal_config(
            dataset_size=50000,
            architecture="qwen",
        )
        assert config.architecture == "qwen"


class TestTargetModulePresets:
    """Tests for target module presets."""

    def test_llama_presets_exist(self):
        """Test LLaMA presets are defined."""
        assert "llama" in TARGET_MODULE_PRESETS
        llama = TARGET_MODULE_PRESETS["llama"]

        assert TargetModulePreset.ATTENTION_ONLY in llama
        assert TargetModulePreset.MLP_ONLY in llama
        assert TargetModulePreset.ALL_LINEAR in llama
        assert TargetModulePreset.QKV_ONLY in llama

    def test_attention_only_no_mlp(self):
        """Test attention-only preset has no MLP modules."""
        llama = TARGET_MODULE_PRESETS["llama"]
        attn_only = llama[TargetModulePreset.ATTENTION_ONLY]

        assert "gate_proj" not in attn_only
        assert "up_proj" not in attn_only
        assert "down_proj" not in attn_only
        assert "q_proj" in attn_only

    def test_mlp_only_no_attention(self):
        """Test MLP-only preset has no attention modules."""
        llama = TARGET_MODULE_PRESETS["llama"]
        mlp_only = llama[TargetModulePreset.MLP_ONLY]

        assert "q_proj" not in mlp_only
        assert "k_proj" not in mlp_only
        assert "v_proj" not in mlp_only
        assert "gate_proj" in mlp_only
