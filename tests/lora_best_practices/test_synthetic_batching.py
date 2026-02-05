"""
Tests for Synthetic Batching System.
"""

import pytest

from tensafe.lora_best_practices.synthetic_batching import (
    SpeculationStrategy,
    SpeculativeToken,
    SpeculationBatch,
    VerificationResult,
    SyntheticBatchMetrics,
    HEBatchConfig,
    LookaheadHEBatcher,
    HybridSyntheticBatcher,
    SyntheticBatchExecutor,
    analyze_synthetic_batching_performance,
)


class TestSpeculativeToken:
    """Tests for SpeculativeToken dataclass."""

    def test_basic_creation(self):
        """Test basic token creation."""
        token = SpeculativeToken(
            token_id=1234,
            position=10,
            confidence=0.8,
            source=SpeculationStrategy.DRAFT_MODEL,
        )
        assert token.token_id == 1234
        assert token.position == 10
        assert token.confidence == 0.8
        assert token.source == SpeculationStrategy.DRAFT_MODEL


class TestSpeculationBatch:
    """Tests for SpeculationBatch dataclass."""

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        tokens = [
            SpeculativeToken(
                token_id=i,
                position=i,
                confidence=0.8,
                source=SpeculationStrategy.LOOKAHEAD,
            )
            for i in range(8)
        ]
        batch = SpeculationBatch(
            tokens=tokens,
            batch_size=8,
            target_acceptance_rate=0.7,
        )

        assert batch.effective_batch_size == 5  # int(8 * 0.7)


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_tokens_generated_all_accepted(self):
        """Test tokens generated when all accepted."""
        result = VerificationResult(
            accepted_tokens=[1, 2, 3, 4],
            accepted_count=4,
            rejection_position=-1,
            bonus_token=None,
            total_speculated=4,
            acceptance_rate=1.0,
        )
        assert result.tokens_generated == 4

    def test_tokens_generated_with_rejection(self):
        """Test tokens generated with rejection and bonus."""
        result = VerificationResult(
            accepted_tokens=[1, 2],
            accepted_count=2,
            rejection_position=2,
            bonus_token=5,
            total_speculated=4,
            acceptance_rate=0.5,
        )
        assert result.tokens_generated == 3  # 2 accepted + 1 bonus


class TestSyntheticBatchMetrics:
    """Tests for metrics calculation."""

    def test_acceptance_rate(self):
        """Test acceptance rate calculation."""
        metrics = SyntheticBatchMetrics(
            total_speculated_tokens=100,
            total_accepted_tokens=70,
        )
        assert metrics.acceptance_rate == 0.7

    def test_tokens_per_second(self):
        """Test throughput calculation."""
        metrics = SyntheticBatchMetrics(
            total_tokens_generated=1000,
            total_time_ms=500.0,  # 0.5 seconds
        )
        assert metrics.tokens_per_second == 2000.0


class TestLookaheadHEBatcher:
    """Tests for LookaheadHEBatcher."""

    def test_initialization(self):
        """Test batcher initialization."""
        batcher = LookaheadHEBatcher(
            max_speculation_depth=8,
            target_batch_size=128,
        )
        assert batcher.max_speculation_depth == 8
        assert batcher.target_batch_size == 128

    def test_generate_speculation_batch(self):
        """Test speculation generation."""
        batcher = LookaheadHEBatcher(max_speculation_depth=4)

        # Seed the n-gram cache
        context = [1, 2, 3, 4, 5, 6, 7, 8]
        batcher.update_ngram_cache(context)

        batch = batcher.generate_speculation_batch(
            context_tokens=context,
            hidden_states=None,
            num_speculate=4,
        )

        assert isinstance(batch, SpeculationBatch)
        assert batch.batch_size <= 4

    def test_adaptive_depth(self):
        """Test adaptive speculation depth."""
        batcher = LookaheadHEBatcher(
            max_speculation_depth=8,
            adaptive_depth=True,
        )

        # High acceptance should increase depth
        for _ in range(5):
            batcher.adapt_speculation_depth(0.9)

        # Should have increased (or stayed at max)
        assert batcher.current_speculation_depth >= 8

        # Low acceptance should decrease
        for _ in range(10):
            batcher.adapt_speculation_depth(0.3)

        assert batcher.current_speculation_depth < 8


class TestHybridSyntheticBatcher:
    """Tests for HybridSyntheticBatcher."""

    def test_initialization_without_draft(self):
        """Test initialization without draft model."""
        batcher = HybridSyntheticBatcher(
            draft_model=None,
            use_lookahead=True,
        )
        assert len(batcher.strategies) == 1  # Only lookahead

    def test_strategy_selection(self):
        """Test strategy selection."""
        batcher = HybridSyntheticBatcher(use_lookahead=True)

        # Should be able to generate
        batch = batcher.generate_speculation_batch(
            context_tokens=[1, 2, 3, 4, 5],
            hidden_states=None,
            num_speculate=4,
        )
        assert isinstance(batch, SpeculationBatch)


class TestHEBatchConfig:
    """Tests for HE batch configuration."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HEBatchConfig()
        assert config.target_batch_size == 128
        assert config.max_speculation_depth == 8
        assert config.slot_count == 8192
        assert config.scale_bits == 45


class TestAnalyzePerformance:
    """Tests for performance analysis function."""

    def test_basic_analysis(self):
        """Test basic performance analysis."""
        result = analyze_synthetic_batching_performance(
            prompt_length=1000,
            response_length=500,
            speculation_depth=8,
            acceptance_rate=0.7,
        )

        assert "prefill" in result
        assert "decode" in result
        assert "total" in result
        assert "comparison" in result

    def test_prefill_efficiency(self):
        """Test that prefill achieves near-full batch efficiency."""
        result = analyze_synthetic_batching_performance(
            prompt_length=1000,
            response_length=0,  # Only prefill
        )

        # Prefill should use few batches
        assert result["prefill"]["batches"] == 8  # ceil(1000/128)

    def test_higher_acceptance_faster(self):
        """Test higher acceptance rate gives faster throughput."""
        result_low = analyze_synthetic_batching_performance(acceptance_rate=0.5)
        result_high = analyze_synthetic_batching_performance(acceptance_rate=0.9)

        assert result_high["total"]["throughput_toks"] > result_low["total"]["throughput_toks"]

    def test_speedup_vs_naive(self):
        """Test speedup is significant vs naive."""
        result = analyze_synthetic_batching_performance(
            speculation_depth=8,
            acceptance_rate=0.7,
        )

        # Should have significant speedup
        assert result["comparison"]["speedup_vs_naive"] > 5


class TestSyntheticBatchExecutor:
    """Tests for SyntheticBatchExecutor."""

    def test_initialization(self):
        """Test executor initialization."""
        batcher = LookaheadHEBatcher()
        config = HEBatchConfig()

        executor = SyntheticBatchExecutor(
            batcher=batcher,
            he_config=config,
        )

        assert executor.batcher is batcher
        assert executor.he_config is config

    def test_prefill(self):
        """Test prefill phase."""
        batcher = LookaheadHEBatcher()
        config = HEBatchConfig(target_batch_size=128)
        executor = SyntheticBatchExecutor(batcher=batcher, he_config=config)

        prompt_tokens = list(range(100))
        hidden_states, processed = executor.prefill(prompt_tokens, he_context=None)

        assert processed == prompt_tokens
        assert executor.metrics.total_tokens_generated == 100

    def test_decode_step(self):
        """Test single decode step."""
        batcher = LookaheadHEBatcher(max_speculation_depth=4)
        config = HEBatchConfig()
        executor = SyntheticBatchExecutor(batcher=batcher, he_config=config)

        new_tokens, hidden, result = executor.decode_step(
            context_tokens=list(range(10)),
            hidden_states=None,
            he_context=None,
        )

        assert isinstance(new_tokens, list)
        assert isinstance(result, VerificationResult)
