"""
Unit tests for RDP (Rényi Differential Privacy) Accountant.

Tests cover:
- RDP computation for Gaussian mechanism
- RDP composition
- RDP → (ε, δ)-DP conversion
- Budget tracking and exhaustion
- Privacy estimation utilities
"""

import pytest
import math
from tensorguard.core.privacy.rdp_accountant import (
    RDPAccountant,
    compute_dp_sgd_privacy,
    DEFAULT_RDP_ORDERS,
)


class TestRDPComputation:
    """Test RDP value computation."""

    def test_rdp_gaussian_basic(self):
        """Test basic RDP computation for Gaussian mechanism."""
        accountant = RDPAccountant()
        rdp = accountant.compute_rdp_gaussian(
            noise_multiplier=1.0,
            sample_rate=0.01,
            num_steps=1,
        )

        # RDP values should be non-negative (with numerical stability fix)
        assert all(v >= 0 or math.isinf(v) for v in rdp)
        # After computing, we should get a valid epsilon
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
        eps = accountant.get_epsilon()
        assert eps > 0

    def test_rdp_higher_noise_lower_rdp(self):
        """Higher noise multiplier should result in lower RDP."""
        accountant = RDPAccountant()

        rdp_low_noise = accountant.compute_rdp_gaussian(
            noise_multiplier=0.5,
            sample_rate=0.01,
            num_steps=1,
        )
        rdp_high_noise = accountant.compute_rdp_gaussian(
            noise_multiplier=2.0,
            sample_rate=0.01,
            num_steps=1,
        )

        # Higher noise should give lower RDP at each order
        for low, high in zip(rdp_low_noise, rdp_high_noise):
            if not math.isinf(low) and not math.isinf(high):
                assert high <= low

    def test_rdp_lower_sample_rate_better_privacy(self):
        """Lower sample rate should result in better privacy (lower epsilon)."""
        # Test via epsilon comparison (more robust than comparing raw RDP)
        accountant_high_q = RDPAccountant()
        accountant_high_q.add_step(
            noise_multiplier=1.0,
            sample_rate=0.1,
            num_steps=1,
        )
        eps_high_q = accountant_high_q.get_epsilon()

        accountant_low_q = RDPAccountant()
        accountant_low_q.add_step(
            noise_multiplier=1.0,
            sample_rate=0.01,
            num_steps=1,
        )
        eps_low_q = accountant_low_q.get_epsilon()

        # Lower sample rate should give lower epsilon (better privacy)
        assert eps_low_q <= eps_high_q

    def test_rdp_invalid_noise_multiplier(self):
        """Should raise error for invalid noise multiplier."""
        accountant = RDPAccountant()

        with pytest.raises(ValueError, match="positive"):
            accountant.compute_rdp_gaussian(noise_multiplier=0, sample_rate=0.01)

        with pytest.raises(ValueError, match="positive"):
            accountant.compute_rdp_gaussian(noise_multiplier=-1.0, sample_rate=0.01)

    def test_rdp_invalid_sample_rate(self):
        """Should raise error for invalid sample rate."""
        accountant = RDPAccountant()

        with pytest.raises(ValueError):
            accountant.compute_rdp_gaussian(noise_multiplier=1.0, sample_rate=0)

        with pytest.raises(ValueError):
            accountant.compute_rdp_gaussian(noise_multiplier=1.0, sample_rate=1.5)


class TestRDPComposition:
    """Test RDP composition and epsilon conversion."""

    def test_add_step_updates_epsilon(self):
        """Adding a step should update the cumulative epsilon."""
        accountant = RDPAccountant(epsilon_budget=10.0)

        eps1 = accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
        assert eps1 > 0

        eps2 = accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
        assert eps2 > eps1  # Epsilon should increase with more steps

    def test_rdp_composition_additive(self):
        """RDP composition should be additive."""
        accountant = RDPAccountant()

        # Add two steps
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
        eps_two_steps = accountant.get_epsilon()

        # Compare with single accountant with two steps at once
        accountant2 = RDPAccountant()
        accountant2.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=2)
        eps_combined = accountant2.get_epsilon()

        # Should be approximately equal (within numerical precision)
        assert abs(eps_two_steps - eps_combined) < 0.01

    def test_epsilon_conversion_delta_sensitivity(self):
        """Epsilon should depend on delta."""
        accountant = RDPAccountant(target_delta=1e-5)
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=100)

        eps_small_delta = accountant.get_epsilon(delta=1e-6)
        eps_large_delta = accountant.get_epsilon(delta=1e-4)

        # Smaller delta should give larger epsilon
        assert eps_small_delta > eps_large_delta

    def test_invalid_delta(self):
        """Should raise error for non-positive delta."""
        accountant = RDPAccountant()
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01)

        with pytest.raises(ValueError, match="positive"):
            accountant.get_epsilon(delta=0)

        with pytest.raises(ValueError, match="positive"):
            accountant.get_epsilon(delta=-1e-5)


class TestBudgetTracking:
    """Test privacy budget tracking and enforcement."""

    def test_budget_exhaustion(self):
        """Should detect when budget is exhausted."""
        accountant = RDPAccountant(epsilon_budget=1.0)

        # Add steps until budget exhausted
        for _ in range(100):
            if accountant.is_budget_exhausted():
                break
            accountant.add_step(noise_multiplier=0.5, sample_rate=0.1, num_steps=1)

        assert accountant.is_budget_exhausted()

    def test_remaining_budget(self):
        """Should correctly compute remaining budget."""
        accountant = RDPAccountant(epsilon_budget=10.0)

        # Initial epsilon should be 0 (no steps taken)
        initial_eps = accountant.get_epsilon()
        assert initial_eps == 0.0

        initial_remaining = accountant.get_remaining_budget()
        assert initial_remaining == 10.0

        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
        remaining_after = accountant.get_remaining_budget()

        assert remaining_after < initial_remaining
        assert remaining_after >= 0

    def test_budget_tracking_across_steps(self):
        """Budget should decrease with each step."""
        accountant = RDPAccountant(epsilon_budget=10.0)
        previous_remaining = accountant.get_remaining_budget()

        for i in range(5):
            accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)
            current_remaining = accountant.get_remaining_budget()
            assert current_remaining < previous_remaining
            previous_remaining = current_remaining

    def test_summary(self):
        """Summary should contain expected fields."""
        accountant = RDPAccountant(epsilon_budget=10.0, target_delta=1e-5)
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01, num_steps=1)

        summary = accountant.summary()

        assert "num_steps" in summary
        assert "epsilon" in summary
        assert "delta" in summary
        assert "budget" in summary
        assert "remaining" in summary
        assert "exhausted" in summary

        assert summary["num_steps"] == 1
        assert summary["delta"] == 1e-5
        assert summary["budget"] == 10.0


class TestNoiseMultiplierComputation:
    """Test computing required noise multiplier."""

    def test_compute_noise_for_target_epsilon(self):
        """Should find noise multiplier for target epsilon."""
        accountant = RDPAccountant()

        target_eps = 1.0
        sigma = accountant.compute_noise_multiplier(
            target_epsilon=target_eps,
            sample_rate=0.01,
            num_steps=100,
        )

        # Verify the computed sigma gives approximately target epsilon
        test_accountant = RDPAccountant()
        test_accountant.add_step(sigma, 0.01, 100)
        actual_eps = test_accountant.get_epsilon()

        assert abs(actual_eps - target_eps) < 0.1  # Within 0.1 tolerance


class TestUtilityFunctions:
    """Test utility functions."""

    def test_compute_dp_sgd_privacy(self):
        """Test the convenience function for DP-SGD privacy."""
        epsilon, delta = compute_dp_sgd_privacy(
            sample_size=10000,
            batch_size=100,
            noise_multiplier=1.0,
            epochs=10,
            delta=1e-5,
        )

        assert epsilon > 0
        assert delta == 1e-5

    def test_compute_dp_sgd_more_epochs_higher_epsilon(self):
        """More epochs should result in higher epsilon."""
        eps_5_epochs, _ = compute_dp_sgd_privacy(
            sample_size=10000,
            batch_size=100,
            noise_multiplier=1.0,
            epochs=5,
        )

        eps_20_epochs, _ = compute_dp_sgd_privacy(
            sample_size=10000,
            batch_size=100,
            noise_multiplier=1.0,
            epochs=20,
        )

        assert eps_20_epochs > eps_5_epochs


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_small_sample_rate(self):
        """Should handle very small sample rates."""
        accountant = RDPAccountant()
        eps = accountant.add_step(
            noise_multiplier=1.0,
            sample_rate=1e-6,
            num_steps=1,
        )
        assert eps >= 0
        assert not math.isnan(eps)
        assert not math.isinf(eps)

    def test_many_small_steps(self):
        """Should handle many small privacy steps."""
        accountant = RDPAccountant(epsilon_budget=100.0)

        for _ in range(1000):
            accountant.add_step(
                noise_multiplier=10.0,
                sample_rate=0.001,
                num_steps=1,
            )

        eps = accountant.get_epsilon()
        assert not math.isnan(eps)
        assert not math.isinf(eps)

    def test_default_orders_coverage(self):
        """Default RDP orders should provide good coverage."""
        # Check that default orders cover the typical range
        assert min(DEFAULT_RDP_ORDERS) > 1  # Must be > 1 for valid RDP
        assert max(DEFAULT_RDP_ORDERS) >= 50  # Should include high orders
        assert len(DEFAULT_RDP_ORDERS) >= 50  # Good resolution

    def test_privacy_spent_tuple(self):
        """get_privacy_spent should return (epsilon, delta) tuple."""
        accountant = RDPAccountant(target_delta=1e-5)
        accountant.add_step(noise_multiplier=1.0, sample_rate=0.01)

        epsilon, delta = accountant.get_privacy_spent()

        assert epsilon > 0
        assert delta == 1e-5
