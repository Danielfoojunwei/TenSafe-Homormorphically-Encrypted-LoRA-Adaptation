"""
Tests for LoRA configuration presets.
"""

import pytest

from tensafe.lora_best_practices.presets import (
    PresetType,
    PRESET_CONFIGS,
    get_preset_config,
    list_presets,
    recommend_preset,
)
from tensafe.lora_best_practices.config import TargetModulePreset


class TestPresetConfigs:
    """Tests for preset configurations."""

    def test_all_presets_exist(self):
        """Test all preset types have configurations."""
        for preset_type in PresetType:
            assert preset_type in PRESET_CONFIGS, f"Missing preset: {preset_type}"

    def test_instruction_tuning_preset(self):
        """Test instruction tuning preset."""
        config = PRESET_CONFIGS[PresetType.INSTRUCTION_TUNING]

        assert config.rank == 32
        assert config.alpha == 64.0
        assert config.target_preset == TargetModulePreset.ALL_LINEAR
        assert config.use_rslora is True

    def test_memory_efficient_uses_qlora(self):
        """Test memory efficient preset uses QLoRA."""
        config = PRESET_CONFIGS[PresetType.MEMORY_EFFICIENT]

        assert config.use_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_high_quality_uses_dora(self):
        """Test high quality preset uses DoRA."""
        config = PRESET_CONFIGS[PresetType.HIGH_QUALITY]

        assert config.use_dora is True
        assert config.rank >= 64  # Higher rank for quality

    def test_privacy_preserving_freezes_a(self):
        """Test privacy preserving preset uses FFA-LoRA."""
        config = PRESET_CONFIGS[PresetType.PRIVACY_PRESERVING]

        assert config.freeze_a_matrix is True
        assert config.use_he_compatible_init is True

    def test_llama_presets_have_llama_architecture(self):
        """Test LLaMA presets have correct architecture."""
        for preset_type in [PresetType.LLAMA_7B, PresetType.LLAMA_13B, PresetType.LLAMA_70B]:
            config = PRESET_CONFIGS[preset_type]
            assert config.architecture == "llama"

    def test_llama_70b_uses_qlora(self):
        """Test LLaMA 70B uses QLoRA due to size."""
        config = PRESET_CONFIGS[PresetType.LLAMA_70B]
        assert config.use_4bit is True


class TestGetPresetConfig:
    """Tests for get_preset_config function."""

    def test_basic_usage(self):
        """Test basic preset retrieval."""
        config = get_preset_config(PresetType.STANDARD)

        assert config is not None
        assert config.rank > 0

    def test_with_overrides(self):
        """Test preset with overrides."""
        config = get_preset_config(
            PresetType.STANDARD,
            rank=128,
            learning_rate=1e-4,
        )

        assert config.rank == 128
        assert config.learning_rate == 1e-4

    def test_override_preserves_other_values(self):
        """Test overrides preserve other preset values."""
        base_config = PRESET_CONFIGS[PresetType.INSTRUCTION_TUNING]
        modified_config = get_preset_config(
            PresetType.INSTRUCTION_TUNING,
            rank=64,
        )

        # Modified value
        assert modified_config.rank == 64

        # Preserved values
        assert modified_config.dropout == base_config.dropout
        assert modified_config.use_rslora == base_config.use_rslora

    def test_invalid_preset_raises(self):
        """Test invalid preset raises error."""
        with pytest.raises((ValueError, KeyError)):
            get_preset_config("nonexistent_preset")


class TestListPresets:
    """Tests for list_presets function."""

    def test_returns_dict(self):
        """Test returns dictionary."""
        presets = list_presets()
        assert isinstance(presets, dict)

    def test_contains_all_presets(self):
        """Test contains all preset types."""
        presets = list_presets()
        for preset_type in PresetType:
            assert preset_type.value in presets


class TestRecommendPreset:
    """Tests for recommend_preset function."""

    def test_privacy_takes_precedence(self):
        """Test privacy requirement returns privacy preset."""
        preset = recommend_preset(
            task_type="instruction",
            vram_gb=48.0,
            privacy_required=True,
        )
        assert preset == PresetType.PRIVACY_PRESERVING

    def test_low_vram_returns_memory_efficient(self):
        """Test low VRAM returns memory efficient preset."""
        preset = recommend_preset(
            task_type="instruction",
            vram_gb=12.0,
        )
        assert preset == PresetType.MEMORY_EFFICIENT

    def test_task_specific_recommendation(self):
        """Test task-specific recommendations."""
        assert recommend_preset(task_type="code") == PresetType.CODE
        assert recommend_preset(task_type="math") == PresetType.MATH
        assert recommend_preset(task_type="chat") == PresetType.CHAT

    def test_quality_priority(self):
        """Test quality priority returns high quality."""
        preset = recommend_preset(
            task_type="general",
            quality_priority=True,
        )
        assert preset == PresetType.HIGH_QUALITY

    def test_speed_priority(self):
        """Test speed priority returns fast training."""
        preset = recommend_preset(
            task_type="general",
            speed_priority=True,
        )
        assert preset == PresetType.FAST_TRAINING

    def test_default_is_standard(self):
        """Test default recommendation is standard."""
        preset = recommend_preset(task_type="general")
        assert preset == PresetType.STANDARD
