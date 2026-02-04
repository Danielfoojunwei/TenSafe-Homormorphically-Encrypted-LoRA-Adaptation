"""
Tests for LoRA scaling methods.
"""

import pytest
import math

from tensafe.lora_best_practices.scaling import (
    StandardLoRAScaling,
    RSLoRAScaling,
    UnitScaling,
    compute_lora_scaling,
    analyze_scaling_stability,
    get_optimal_alpha,
    recommend_scaling_method,
)


class TestStandardLoRAScaling:
    """Tests for standard LoRA scaling."""

    def test_compute_scaling(self):
        """Test standard scaling: α/r."""
        scaling = StandardLoRAScaling()

        # α=64, r=32 -> 2.0
        result = scaling.compute_scaling(alpha=64.0, rank=32)
        assert abs(result - 2.0) < 1e-6

        # α=32, r=8 -> 4.0
        result = scaling.compute_scaling(alpha=32.0, rank=8)
        assert abs(result - 4.0) < 1e-6

    def test_name(self):
        """Test scaling method name."""
        scaling = StandardLoRAScaling()
        assert scaling.get_name() == "standard"

    def test_invalid_rank(self):
        """Test rejection of invalid rank."""
        scaling = StandardLoRAScaling()

        with pytest.raises(ValueError):
            scaling.compute_scaling(alpha=64.0, rank=0)

        with pytest.raises(ValueError):
            scaling.compute_scaling(alpha=64.0, rank=-1)


class TestRSLoRAScaling:
    """Tests for rsLoRA scaling."""

    def test_compute_scaling(self):
        """Test rsLoRA scaling: α/√r."""
        scaling = RSLoRAScaling()

        # α=64, r=32 -> 64/√32 ≈ 11.31
        result = scaling.compute_scaling(alpha=64.0, rank=32)
        expected = 64.0 / math.sqrt(32)
        assert abs(result - expected) < 1e-6

    def test_name(self):
        """Test scaling method name."""
        scaling = RSLoRAScaling()
        assert scaling.get_name() == "rslora"

    def test_higher_scaling_than_standard(self):
        """Test rsLoRA gives higher scaling for high ranks."""
        standard = StandardLoRAScaling()
        rslora = RSLoRAScaling()

        for rank in [64, 128, 256]:
            s_standard = standard.compute_scaling(alpha=128.0, rank=rank)
            s_rslora = rslora.compute_scaling(alpha=128.0, rank=rank)

            # rsLoRA should give higher scaling factor for high ranks
            assert s_rslora > s_standard


class TestUnitScaling:
    """Tests for unit scaling."""

    def test_always_returns_one(self):
        """Test unit scaling always returns 1.0."""
        scaling = UnitScaling()

        assert scaling.compute_scaling(alpha=64.0, rank=32) == 1.0
        assert scaling.compute_scaling(alpha=128.0, rank=8) == 1.0
        assert scaling.compute_scaling(alpha=1.0, rank=256) == 1.0


class TestComputeLoRAScaling:
    """Tests for compute_lora_scaling utility."""

    def test_default_uses_rslora(self):
        """Test default is rsLoRA."""
        result = compute_lora_scaling(alpha=64.0, rank=32)
        expected = 64.0 / math.sqrt(32)
        assert abs(result - expected) < 1e-6

    def test_explicit_standard(self):
        """Test explicit standard scaling."""
        result = compute_lora_scaling(alpha=64.0, rank=32, use_rslora=False)
        expected = 64.0 / 32
        assert abs(result - expected) < 1e-6


class TestGetOptimalAlpha:
    """Tests for get_optimal_alpha utility."""

    def test_rslora_alpha(self):
        """Test alpha calculation for rsLoRA."""
        # Want scaling=1.0, rank=64, rsLoRA
        # scaling = alpha / √rank -> alpha = scaling * √rank
        alpha = get_optimal_alpha(rank=64, target_scaling=1.0, use_rslora=True)
        expected = 1.0 * math.sqrt(64)  # 8.0
        assert abs(alpha - expected) < 1e-6

    def test_standard_alpha(self):
        """Test alpha calculation for standard scaling."""
        # Want scaling=2.0, rank=32, standard
        # scaling = alpha / rank -> alpha = scaling * rank
        alpha = get_optimal_alpha(rank=32, target_scaling=2.0, use_rslora=False)
        expected = 2.0 * 32  # 64.0
        assert abs(alpha - expected) < 1e-6


class TestRecommendScalingMethod:
    """Tests for recommend_scaling_method utility."""

    def test_low_rank_either(self):
        """Test low rank recommendation."""
        result = recommend_scaling_method(rank=16)
        assert result == "standard"  # Either works

    def test_medium_rank_rslora(self):
        """Test medium rank recommendation."""
        result = recommend_scaling_method(rank=64)
        assert result == "rslora"

    def test_high_rank_rslora(self):
        """Test high rank recommendation."""
        result = recommend_scaling_method(rank=256)
        assert result == "rslora"


class TestAnalyzeScalingStability:
    """Tests for scaling analysis."""

    def test_returns_analysis(self):
        """Test analysis is returned."""
        analysis = analyze_scaling_stability(alpha=64.0)

        assert analysis is not None
        assert "standard" in analysis.scaling_factors
        assert "rslora" in analysis.scaling_factors

    def test_custom_ranks(self):
        """Test with custom rank list."""
        analysis = analyze_scaling_stability(
            alpha=64.0,
            ranks=[8, 32, 128],
        )

        assert 8 in analysis.scaling_factors["standard"]
        assert 32 in analysis.scaling_factors["standard"]
        assert 128 in analysis.scaling_factors["standard"]
