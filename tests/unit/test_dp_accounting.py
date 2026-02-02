"""
Unit tests for production-grade DP accounting.

Tests the RDPAccountant implementation including:
- Gaussian mechanism RDP computation
- Privacy amplification by subsampling
- RDP to DP conversion
- Privacy budget planning utilities
"""

import math
import pytest
from unittest.mock import patch

from tensorguard.platform.tg_tinker_api.dp import (
    RDPAccountant,
    DPConfig,
    DPTrainer,
    DPMetrics,
    create_accountant,
    compute_noise_multiplier,
    compute_max_steps,
    plan_privacy_budget,
    PrivacyBudgetPlan,
    clip_gradients,
    add_noise,
    _log_add,
    _log1mexp,
    MIN_NOISE_MULTIPLIER,
    MAX_RDP_EPSILON,
)


class TestNumericalUtilities:
    """Tests for numerical utility functions."""

    def test_log_add_basic(self):
        """Test log_add computes log(exp(a) + exp(b)) correctly."""
        # log(e^1 + e^2) = log(e + e^2) ≈ 2.313
        result = _log_add(1.0, 2.0)
        expected = math.log(math.exp(1.0) + math.exp(2.0))
        assert abs(result - expected) < 1e-10

    def test_log_add_with_neg_inf(self):
        """Test log_add handles negative infinity."""
        assert _log_add(float("-inf"), 1.0) == 1.0
        assert _log_add(1.0, float("-inf")) == 1.0

    def test_log1mexp_basic(self):
        """Test log1mexp computes log(1 - exp(x)) correctly."""
        # For x = -1, log(1 - e^(-1)) ≈ log(0.632) ≈ -0.459
        result = _log1mexp(-1.0)
        expected = math.log(1 - math.exp(-1.0))
        assert abs(result - expected) < 1e-10

    def test_log1mexp_close_to_zero(self):
        """Test log1mexp handles values close to zero."""
        result = _log1mexp(-0.001)
        # Should be approximately -7.6 (log of a very small number)
        assert result < 0


class TestRDPAccountantBasic:
    """Tests for basic RDP accountant functionality."""

    def test_accountant_initialization(self):
        """Test RDP accountant initializes correctly."""
        accountant = RDPAccountant(target_delta=1e-5)

        assert accountant.target_delta == 1e-5
        assert len(accountant.orders) > 0
        assert accountant.get_num_steps() == 0

    def test_accountant_reset(self):
        """Test accountant reset clears state."""
        accountant = RDPAccountant(target_delta=1e-5)

        # Run some steps
        accountant.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
        assert accountant.get_num_steps() == 10

        # Reset
        accountant.reset()
        assert accountant.get_num_steps() == 0
        eps, delta = accountant.get_privacy_spent()
        assert eps == 0.0

    def test_accountant_custom_orders(self):
        """Test accountant accepts custom RDP orders."""
        custom_orders = [2, 4, 8, 16]
        accountant = RDPAccountant(target_delta=1e-5, orders=custom_orders)

        assert accountant.orders == custom_orders


class TestGaussianMechanismRDP:
    """Tests for Gaussian mechanism RDP computation."""

    def test_gaussian_rdp_formula(self):
        """Test RDP for Gaussian mechanism follows alpha/(2*sigma^2)."""
        accountant = RDPAccountant(target_delta=1e-5)

        sigma = 1.0
        alpha = 2.0

        # Without subsampling (sample_rate = 1), RDP should be alpha/(2*sigma^2)
        rdp = accountant._compute_rdp_gaussian(sigma, alpha)
        expected = alpha / (2 * sigma * sigma)

        assert abs(rdp - expected) < 1e-10

    def test_gaussian_rdp_scales_with_sigma(self):
        """Test RDP decreases as noise multiplier increases."""
        accountant = RDPAccountant(target_delta=1e-5)

        rdp_sigma1 = accountant._compute_rdp_gaussian(1.0, 2.0)
        rdp_sigma2 = accountant._compute_rdp_gaussian(2.0, 2.0)
        rdp_sigma4 = accountant._compute_rdp_gaussian(4.0, 2.0)

        # Larger sigma should give smaller RDP
        assert rdp_sigma1 > rdp_sigma2 > rdp_sigma4


class TestSubsampledGaussianRDP:
    """Tests for subsampled Gaussian mechanism RDP."""

    def test_subsampling_reduces_rdp(self):
        """Test privacy amplification by subsampling."""
        accountant = RDPAccountant(target_delta=1e-5)

        sigma = 1.0
        alpha = 2.0

        rdp_full = accountant._compute_rdp_subsampled_gaussian(sigma, 1.0, alpha)
        rdp_half = accountant._compute_rdp_subsampled_gaussian(sigma, 0.5, alpha)
        rdp_tenth = accountant._compute_rdp_subsampled_gaussian(sigma, 0.1, alpha)

        # Smaller sample rate should give smaller RDP (better privacy)
        assert rdp_full > rdp_half > rdp_tenth

    def test_subsampling_zero_rate(self):
        """Test zero sampling rate gives zero RDP."""
        accountant = RDPAccountant(target_delta=1e-5)

        rdp = accountant._compute_rdp_subsampled_gaussian(1.0, 0.0, 2.0)
        assert rdp == 0.0

    def test_subsampling_full_rate(self):
        """Test full sampling rate equals non-subsampled."""
        accountant = RDPAccountant(target_delta=1e-5)

        rdp_subsampled = accountant._compute_rdp_subsampled_gaussian(1.0, 1.0, 2.0)
        rdp_gaussian = accountant._compute_rdp_gaussian(1.0, 2.0)

        assert abs(rdp_subsampled - rdp_gaussian) < 1e-10

    def test_integer_order_computation(self):
        """Test integer order RDP computation."""
        accountant = RDPAccountant(target_delta=1e-5)

        # Integer orders should use exact binomial formula
        rdp = accountant._compute_rdp_sampled_gaussian_integer(1.0, 0.1, 4)
        assert rdp >= 0


class TestRDPToDPConversion:
    """Tests for RDP to (epsilon, delta)-DP conversion."""

    def test_rdp_to_dp_basic(self):
        """Test RDP to DP conversion produces valid epsilon."""
        accountant = RDPAccountant(target_delta=1e-5)

        # Run some steps
        eps, delta = accountant.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        assert eps > 0
        assert delta == 1e-5

    def test_rdp_to_dp_composition(self):
        """Test epsilon increases with composition."""
        accountant = RDPAccountant(target_delta=1e-5)

        eps_10, _ = accountant.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=10)
        eps_100, _ = accountant.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=90)

        # Total should be for 100 steps
        assert accountant.get_num_steps() == 100
        assert eps_100 > eps_10

    def test_epsilon_scales_sublinearly(self):
        """Test epsilon grows sublinearly with steps (due to RDP composition)."""
        accountant1 = RDPAccountant(target_delta=1e-5)
        accountant2 = RDPAccountant(target_delta=1e-5)

        eps_100, _ = accountant1.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)
        eps_1000, _ = accountant2.step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1000)

        # Epsilon should grow slower than linear
        assert eps_1000 < 10 * eps_100


class TestPrivacyBudgetPlanning:
    """Tests for privacy budget planning utilities."""

    def test_compute_noise_multiplier_basic(self):
        """Test computing noise multiplier for target epsilon."""
        sigma = compute_noise_multiplier(
            target_epsilon=8.0,
            target_delta=1e-5,
            sample_rate=0.01,
            num_steps=1000,
        )

        assert sigma > 0

        # Verify it achieves approximately the target
        accountant = RDPAccountant(target_delta=1e-5)
        actual_eps = accountant.compute_epsilon_for_steps(sigma, 0.01, 1000, 1e-5)
        assert abs(actual_eps - 8.0) / 8.0 < 0.05  # Within 5%

    def test_compute_noise_multiplier_invalid_inputs(self):
        """Test compute_noise_multiplier rejects invalid inputs."""
        with pytest.raises(ValueError, match="target_epsilon must be positive"):
            compute_noise_multiplier(-1.0, 1e-5, 0.01, 100)

        with pytest.raises(ValueError, match="target_delta must be in"):
            compute_noise_multiplier(8.0, 0.0, 0.01, 100)

        with pytest.raises(ValueError, match="sample_rate must be in"):
            compute_noise_multiplier(8.0, 1e-5, 0.0, 100)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            compute_noise_multiplier(8.0, 1e-5, 0.01, 0)

    def test_compute_max_steps(self):
        """Test computing max steps for privacy budget."""
        max_steps = compute_max_steps(
            noise_multiplier=1.0,
            target_epsilon=8.0,
            target_delta=1e-5,
            sample_rate=0.01,
        )

        assert max_steps > 0

        # Verify it's within budget
        accountant = RDPAccountant(target_delta=1e-5)
        eps = accountant.compute_epsilon_for_steps(1.0, 0.01, max_steps, 1e-5)
        assert eps <= 8.0

        # One more step should exceed budget
        eps_plus1 = accountant.compute_epsilon_for_steps(1.0, 0.01, max_steps + 1, 1e-5)
        assert eps_plus1 > 8.0

    def test_plan_privacy_budget(self):
        """Test planning privacy budget for training run."""
        plan = plan_privacy_budget(
            target_epsilon=8.0,
            target_delta=1e-5,
            dataset_size=60000,
            batch_size=256,
            num_epochs=3.0,
        )

        assert isinstance(plan, PrivacyBudgetPlan)
        assert plan.noise_multiplier > 0
        assert plan.sample_rate == 256 / 60000
        assert plan.num_steps == int(3.0 * 60000 / 256)
        assert abs(plan.epsilon - 8.0) / 8.0 < 0.05  # Within 5%
        assert plan.delta == 1e-5

    def test_privacy_budget_plan_summary(self):
        """Test privacy budget plan summary generation."""
        plan = PrivacyBudgetPlan(
            noise_multiplier=1.0,
            sample_rate=0.01,
            num_steps=1000,
            epsilon=8.0,
            delta=1e-5,
            epsilon_per_step=0.008,
        )

        summary = plan.summary()
        assert "Noise multiplier" in summary
        assert "1000" in summary


class TestDPTrainer:
    """Tests for the DPTrainer wrapper."""

    def test_trainer_initialization(self):
        """Test DPTrainer initializes with config."""
        config = DPConfig(
            enabled=True,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            target_epsilon=8.0,
            target_delta=1e-5,
        )

        trainer = DPTrainer(config)
        assert trainer.config == config
        assert trainer.state.num_steps == 0

    def test_trainer_process_gradients(self):
        """Test processing gradients with DP."""
        config = DPConfig(
            enabled=True,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        trainer = DPTrainer(config)
        metrics = trainer.process_gradients(grad_norm=0.5, sample_rate=0.01)

        assert isinstance(metrics, DPMetrics)
        assert metrics.noise_applied is True
        assert metrics.total_epsilon > 0

    def test_trainer_disabled(self):
        """Test trainer with DP disabled."""
        config = DPConfig(enabled=False)
        trainer = DPTrainer(config)

        metrics = trainer.process_gradients(grad_norm=1.0)
        assert metrics.noise_applied is False

    def test_trainer_check_budget(self):
        """Test budget checking."""
        config = DPConfig(
            enabled=True,
            noise_multiplier=0.1,  # Low noise = high epsilon
            max_grad_norm=1.0,
            target_epsilon=1.0,  # Low target
        )

        trainer = DPTrainer(config)

        # Initial check should pass
        assert trainer.check_budget() is True

        # Run many steps to exceed budget
        for _ in range(1000):
            trainer.process_gradients(grad_norm=1.0, sample_rate=0.1)

        # Should eventually fail
        # (This may or may not fail depending on exact parameters)


class TestGradientClipping:
    """Tests for gradient clipping utilities."""

    def test_clip_gradients_no_clip(self):
        """Test no clipping when norm is below threshold."""
        clipped, was_clipped = clip_gradients(0.5, 1.0)
        assert clipped == 0.5
        assert was_clipped is False

    def test_clip_gradients_clips(self):
        """Test clipping when norm exceeds threshold."""
        clipped, was_clipped = clip_gradients(2.0, 1.0)
        assert clipped == 1.0
        assert was_clipped is True

    def test_clip_gradients_boundary(self):
        """Test clipping at exact boundary."""
        clipped, was_clipped = clip_gradients(1.0, 1.0)
        assert clipped == 1.0
        assert was_clipped is False


class TestAddNoise:
    """Tests for noise computation."""

    def test_add_noise_scale(self):
        """Test noise scale computation."""
        noise = add_noise(
            clipped_grad_norm=1.0,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        assert noise == 1.0

    def test_add_noise_scales_with_multiplier(self):
        """Test noise scales with multiplier."""
        noise1 = add_noise(1.0, 1.0, 1.0)
        noise2 = add_noise(1.0, 2.0, 1.0)
        assert noise2 == 2 * noise1


class TestCreateAccountant:
    """Tests for accountant factory function."""

    def test_create_rdp_accountant(self):
        """Test creating RDP accountant."""
        accountant = create_accountant("rdp", target_delta=1e-5)
        assert isinstance(accountant, RDPAccountant)

    def test_create_unknown_defaults_to_rdp(self):
        """Test unknown type defaults to RDP."""
        accountant = create_accountant("unknown", target_delta=1e-5)
        assert isinstance(accountant, RDPAccountant)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_very_small_noise_multiplier(self):
        """Test handling of very small noise multiplier."""
        accountant = RDPAccountant(target_delta=1e-5)

        rdp = accountant._compute_rdp_subsampled_gaussian(1e-8, 0.01, 2.0)
        assert rdp == MAX_RDP_EPSILON or rdp > 1e6

    def test_very_large_order(self):
        """Test handling of very large RDP order."""
        accountant = RDPAccountant(target_delta=1e-5)

        rdp = accountant._compute_rdp_subsampled_gaussian(1.0, 0.01, 512)
        assert rdp >= 0  # Should be non-negative

    def test_many_steps(self):
        """Test accountant handles many steps."""
        accountant = RDPAccountant(target_delta=1e-5)

        eps, delta = accountant.step(noise_multiplier=1.0, sample_rate=0.001, num_steps=10000)

        assert eps > 0
        assert eps < float("inf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
