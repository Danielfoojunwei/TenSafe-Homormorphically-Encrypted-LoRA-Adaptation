"""
Common evaluation metrics.

Provides implementations of standard metrics for different tasks:
- Accuracy for classification
- Exact match for QA
- F1 score for token-level evaluation
- BLEU for translation/generation
- Perplexity for language modeling
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional, Sequence

from .base import Metric, MetricResult


class AccuracyMetric(Metric):
    """Simple accuracy metric."""

    name = "accuracy"

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        normalize: bool = True,
        case_sensitive: bool = False,
        **kwargs,
    ) -> MetricResult:
        """
        Compute accuracy.

        Args:
            predictions: Model predictions
            references: Ground truth references
            normalize: Whether to strip whitespace
            case_sensitive: Whether comparison is case-sensitive

        Returns:
            MetricResult with accuracy value
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        correct = 0
        for pred, ref in zip(predictions, references):
            if normalize:
                pred = pred.strip()
                ref = ref.strip()
            if not case_sensitive:
                pred = pred.lower()
                ref = ref.lower()
            if pred == ref:
                correct += 1

        acc = correct / len(predictions) if predictions else 0.0

        return MetricResult(
            name=self.name,
            value=acc,
            metadata={"correct": correct, "total": len(predictions)},
        )


class ExactMatchMetric(Metric):
    """Exact match metric for QA tasks."""

    name = "exact_match"

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        normalize_fn: Optional[callable] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Compute exact match score.

        Args:
            predictions: Model predictions
            references: Ground truth references
            normalize_fn: Optional normalization function

        Returns:
            MetricResult with EM score
        """

        def default_normalize(text: str) -> str:
            """Default normalization: lowercase, strip, remove punctuation."""
            import re

            text = text.lower().strip()
            text = re.sub(r"[^\w\s]", "", text)
            text = " ".join(text.split())
            return text

        normalize = normalize_fn or default_normalize

        matches = 0
        for pred, ref in zip(predictions, references):
            if normalize(pred) == normalize(ref):
                matches += 1

        em = matches / len(predictions) if predictions else 0.0

        return MetricResult(
            name=self.name,
            value=em,
            metadata={"matches": matches, "total": len(predictions)},
        )


class F1Metric(Metric):
    """Token-level F1 score."""

    name = "f1"

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> MetricResult:
        """
        Compute token-level F1 score.

        Args:
            predictions: Model predictions
            references: Ground truth references

        Returns:
            MetricResult with F1 score
        """
        f1_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            pred_counter = Counter(pred_tokens)
            ref_counter = Counter(ref_tokens)

            # Count common tokens
            common = sum((pred_counter & ref_counter).values())

            if common == 0:
                f1_scores.append(0.0)
                continue

            precision = common / len(pred_tokens) if pred_tokens else 0.0
            recall = common / len(ref_tokens) if ref_tokens else 0.0

            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return MetricResult(
            name=self.name,
            value=avg_f1,
            metadata={"scores": f1_scores},
        )


class BLEUMetric(Metric):
    """BLEU score for translation/generation tasks."""

    name = "bleu"

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        max_n: int = 4,
        smoothing: bool = True,
        **kwargs,
    ) -> MetricResult:
        """
        Compute BLEU score.

        Args:
            predictions: Model predictions
            references: Ground truth references
            max_n: Maximum n-gram order
            smoothing: Whether to apply smoothing

        Returns:
            MetricResult with BLEU score
        """

        def get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-gram counts."""
            return Counter(
                tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
            )

        bleu_scores = []

        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()

            if not pred_tokens:
                bleu_scores.append(0.0)
                continue

            # Compute n-gram precisions
            precisions = []
            for n in range(1, max_n + 1):
                pred_ngrams = get_ngrams(pred_tokens, n)
                ref_ngrams = get_ngrams(ref_tokens, n)

                if not pred_ngrams:
                    if smoothing:
                        precisions.append(1 / (len(pred_tokens) + 1))
                    else:
                        precisions.append(0.0)
                    continue

                matches = sum((pred_ngrams & ref_ngrams).values())
                total = sum(pred_ngrams.values())

                if smoothing and matches == 0:
                    matches = 1
                    total += 1

                precisions.append(matches / total if total > 0 else 0.0)

            # Compute geometric mean
            if all(p > 0 for p in precisions):
                log_avg = sum(math.log(p) for p in precisions) / len(precisions)
                precision_score = math.exp(log_avg)
            else:
                precision_score = 0.0

            # Brevity penalty
            if len(pred_tokens) < len(ref_tokens):
                bp = math.exp(1 - len(ref_tokens) / len(pred_tokens))
            else:
                bp = 1.0

            bleu = bp * precision_score
            bleu_scores.append(bleu)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        return MetricResult(
            name=self.name,
            value=avg_bleu,
            metadata={"scores": bleu_scores},
        )


class PerplexityMetric(Metric):
    """Perplexity for language modeling."""

    name = "perplexity"

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        log_probs: Optional[List[float]] = None,
        **kwargs,
    ) -> MetricResult:
        """
        Compute perplexity from log probabilities.

        Args:
            predictions: Not used directly
            references: Not used directly
            log_probs: Log probabilities for each token

        Returns:
            MetricResult with perplexity
        """
        if log_probs is None:
            # Can't compute without log probs
            return MetricResult(
                name=self.name,
                value=float("inf"),
                metadata={"error": "log_probs required"},
            )

        if not log_probs:
            return MetricResult(name=self.name, value=float("inf"))

        # Perplexity = exp(-1/N * sum(log_probs))
        avg_log_prob = sum(log_probs) / len(log_probs)
        perplexity = math.exp(-avg_log_prob)

        return MetricResult(
            name=self.name,
            value=perplexity,
            metadata={"avg_log_prob": avg_log_prob, "num_tokens": len(log_probs)},
        )


# Convenience functions
def accuracy(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """Compute accuracy."""
    metric = AccuracyMetric()
    return metric.compute(predictions, references, **kwargs).value


def exact_match(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """Compute exact match score."""
    metric = ExactMatchMetric()
    return metric.compute(predictions, references, **kwargs).value


def f1_score(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """Compute F1 score."""
    metric = F1Metric()
    return metric.compute(predictions, references, **kwargs).value


def bleu(
    predictions: List[str],
    references: List[str],
    **kwargs,
) -> float:
    """Compute BLEU score."""
    metric = BLEUMetric()
    return metric.compute(predictions, references, **kwargs).value


def perplexity(
    log_probs: List[float],
    **kwargs,
) -> float:
    """Compute perplexity from log probabilities."""
    metric = PerplexityMetric()
    return metric.compute([], [], log_probs=log_probs, **kwargs).value
