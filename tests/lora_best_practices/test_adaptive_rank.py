"""
Tests for adaptive rank selection.
"""

import pytest

from tensafe.lora_best_practices.adaptive_rank import (
    AdaptiveRankSelector,
    RankSelectionStrategy,
    estimate_optimal_rank,
)


class TestAdaptiveRankSelector:
    """Tests for AdaptiveRankSelector."""

    def test_init(self):
        """Test initialization."""
        selector = AdaptiveRankSelector(
            hidden_size=4096,
            num_layers=32,
            max_memory_gb=24.0,
        )
        assert selector.hidden_size == 4096
        assert selector.num_layers == 32
        assert selector.max_memory_gb == 24.0

    def test_small_dataset_low_rank(self):
        """Test small datasets get low ranks."""
        selector = AdaptiveRankSelector()
        rank = selector.estimate_rank_from_dataset(num_examples=5000)
        assert rank <= 16

    def test_medium_dataset_medium_rank(self):
        """Test medium datasets get medium ranks."""
        selector = AdaptiveRankSelector()
        rank = selector.estimate_rank_from_dataset(num_examples=50000)
        assert 16 <= rank <= 64

    def test_large_dataset_high_rank(self):
        """Test large datasets get high ranks."""
        selector = AdaptiveRankSelector()
        rank = selector.estimate_rank_from_dataset(num_examples=200000)
        assert rank >= 64

    def test_rank_is_power_of_two(self):
        """Test estimated rank is power of 2."""
        selector = AdaptiveRankSelector()
        for n_examples in [1000, 5000, 25000, 100000]:
            rank = selector.estimate_rank_from_dataset(num_examples=n_examples)
            # Check if power of 2
            assert rank & (rank - 1) == 0 or rank == 4

    def test_rank_clamped_to_valid_range(self):
        """Test rank is clamped to valid range."""
        selector = AdaptiveRankSelector()

        # Very small dataset
        rank = selector.estimate_rank_from_dataset(num_examples=10)
        assert rank >= selector.MIN_RANK

        # Very large dataset
        rank = selector.estimate_rank_from_dataset(num_examples=10_000_000)
        assert rank <= selector.MAX_RANK

    def test_memory_estimation(self):
        """Test memory estimation for rank."""
        selector = AdaptiveRankSelector(hidden_size=4096, num_layers=32)

        # Memory should increase with rank
        mem_16 = selector.estimate_memory_for_rank(rank=16)
        mem_32 = selector.estimate_memory_for_rank(rank=32)
        mem_64 = selector.estimate_memory_for_rank(rank=64)

        assert mem_32 > mem_16
        assert mem_64 > mem_32
        assert mem_32 == pytest.approx(mem_16 * 2, rel=0.01)

    def test_max_rank_for_memory(self):
        """Test maximum rank calculation for memory budget."""
        selector = AdaptiveRankSelector(hidden_size=4096, num_layers=32)

        # Should get higher rank with more memory
        rank_small = selector.max_rank_for_memory(memory_budget_gb=0.1)
        rank_large = selector.max_rank_for_memory(memory_budget_gb=1.0)

        assert rank_large > rank_small

    def test_recommendation(self):
        """Test recommendation generation."""
        selector = AdaptiveRankSelector()
        recommendation = selector.recommend(
            num_examples=50000,
            task_complexity="medium",
        )

        assert recommendation is not None
        assert recommendation.recommended_rank > 0
        assert recommendation.min_rank <= recommendation.recommended_rank
        assert recommendation.max_rank >= recommendation.recommended_rank
        assert 0 <= recommendation.confidence <= 1
        assert recommendation.reasoning

    def test_complexity_affects_rank(self):
        """Test task complexity affects recommended rank."""
        selector = AdaptiveRankSelector()

        rec_simple = selector.recommend(num_examples=50000, task_complexity="simple")
        rec_medium = selector.recommend(num_examples=50000, task_complexity="medium")
        rec_complex = selector.recommend(num_examples=50000, task_complexity="complex")

        assert rec_simple.recommended_rank <= rec_medium.recommended_rank
        assert rec_medium.recommended_rank <= rec_complex.recommended_rank

    def test_memory_constraint_limits_rank(self):
        """Test memory constraint limits recommended rank."""
        selector_unlimited = AdaptiveRankSelector(max_memory_gb=None)
        selector_limited = AdaptiveRankSelector(max_memory_gb=0.05)  # Very limited

        rec_unlimited = selector_unlimited.recommend(num_examples=100000)
        rec_limited = selector_limited.recommend(num_examples=100000)

        assert rec_limited.recommended_rank < rec_unlimited.recommended_rank

    def test_dataset_analysis(self):
        """Test dataset analysis."""
        selector = AdaptiveRankSelector()
        analysis = selector.analyze_dataset(num_examples=50000)

        assert analysis.num_examples == 50000
        assert analysis.recommended_rank > 0
        assert 0 <= analysis.capacity_ratio <= 1


class TestEstimateOptimalRank:
    """Tests for estimate_optimal_rank utility."""

    def test_basic_usage(self):
        """Test basic usage."""
        rank = estimate_optimal_rank(
            num_examples=50000,
            hidden_size=4096,
        )
        assert 16 <= rank <= 128

    def test_complexity_parameter(self):
        """Test complexity parameter."""
        rank_simple = estimate_optimal_rank(
            num_examples=50000,
            task_complexity="simple",
        )
        rank_complex = estimate_optimal_rank(
            num_examples=50000,
            task_complexity="complex",
        )

        assert rank_simple <= rank_complex
