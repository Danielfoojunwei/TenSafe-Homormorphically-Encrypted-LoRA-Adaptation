"""Tests for cookbook evaluation metrics."""

import math

import pytest


class TestAccuracyMetric:
    """Tests for accuracy metric."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        from tensafe.cookbook.eval.metrics import accuracy

        predictions = ["a", "b", "c"]
        references = ["a", "b", "c"]

        result = accuracy(predictions, references)
        assert result == 1.0

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        from tensafe.cookbook.eval.metrics import accuracy

        predictions = ["x", "y", "z"]
        references = ["a", "b", "c"]

        result = accuracy(predictions, references)
        assert result == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        from tensafe.cookbook.eval.metrics import accuracy

        predictions = ["a", "x", "c"]
        references = ["a", "b", "c"]

        result = accuracy(predictions, references)
        assert result == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        """Test case-insensitive comparison."""
        from tensafe.cookbook.eval.metrics import accuracy

        predictions = ["A", "B"]
        references = ["a", "b"]

        result = accuracy(predictions, references, case_sensitive=False)
        assert result == 1.0

    def test_case_sensitive(self):
        """Test case-sensitive comparison."""
        from tensafe.cookbook.eval.metrics import accuracy

        predictions = ["A", "B"]
        references = ["a", "b"]

        result = accuracy(predictions, references, case_sensitive=True)
        assert result == 0.0


class TestExactMatchMetric:
    """Tests for exact match metric."""

    def test_exact_match(self):
        """Test exact match scoring."""
        from tensafe.cookbook.eval.metrics import exact_match

        predictions = ["The answer is 42", "Hello world"]
        references = ["the answer is 42", "hello world"]

        result = exact_match(predictions, references)
        assert result == 1.0

    def test_no_match(self):
        """Test no match case."""
        from tensafe.cookbook.eval.metrics import exact_match

        predictions = ["wrong answer"]
        references = ["correct answer"]

        result = exact_match(predictions, references)
        assert result == 0.0

    def test_whitespace_normalization(self):
        """Test whitespace handling."""
        from tensafe.cookbook.eval.metrics import exact_match

        predictions = ["  hello   world  "]
        references = ["hello world"]

        result = exact_match(predictions, references)
        assert result == 1.0


class TestF1Metric:
    """Tests for F1 metric."""

    def test_perfect_f1(self):
        """Test perfect F1 score."""
        from tensafe.cookbook.eval.metrics import f1_score

        predictions = ["the cat sat"]
        references = ["the cat sat"]

        result = f1_score(predictions, references)
        assert result == 1.0

    def test_partial_overlap(self):
        """Test partial token overlap."""
        from tensafe.cookbook.eval.metrics import f1_score

        predictions = ["the cat sat on the mat"]
        references = ["the cat sat"]

        result = f1_score(predictions, references)
        assert 0 < result < 1

    def test_no_overlap(self):
        """Test no token overlap."""
        from tensafe.cookbook.eval.metrics import f1_score

        predictions = ["foo bar"]
        references = ["baz qux"]

        result = f1_score(predictions, references)
        assert result == 0.0


class TestBLEUMetric:
    """Tests for BLEU metric."""

    def test_perfect_bleu(self):
        """Test perfect BLEU score."""
        from tensafe.cookbook.eval.metrics import bleu

        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]

        result = bleu(predictions, references)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_partial_bleu(self):
        """Test partial BLEU score."""
        from tensafe.cookbook.eval.metrics import bleu

        predictions = ["the cat sat on the mat"]
        references = ["the dog lay on the floor"]

        result = bleu(predictions, references)
        assert 0 < result < 1

    def test_no_match_bleu(self):
        """Test BLEU with no matching n-grams."""
        from tensafe.cookbook.eval.metrics import bleu

        predictions = ["a b c d e"]
        references = ["x y z w v"]

        # With smoothing, should still be > 0
        result = bleu(predictions, references, smoothing=True)
        assert result >= 0


class TestPerplexityMetric:
    """Tests for perplexity metric."""

    def test_perplexity_computation(self):
        """Test basic perplexity computation."""
        from tensafe.cookbook.eval.metrics import perplexity

        log_probs = [-1.0, -1.0, -1.0]  # All same probability

        result = perplexity(log_probs)
        expected = math.exp(1.0)  # exp(-(-1.0))
        assert result == pytest.approx(expected)

    def test_perplexity_lower_is_better(self):
        """Test that higher log probs give lower perplexity."""
        from tensafe.cookbook.eval.metrics import perplexity

        high_prob = [-0.1, -0.1, -0.1]
        low_prob = [-2.0, -2.0, -2.0]

        ppl_high = perplexity(high_prob)
        ppl_low = perplexity(low_prob)

        assert ppl_high < ppl_low
