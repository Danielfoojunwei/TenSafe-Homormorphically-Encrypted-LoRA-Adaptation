"""
TenSafe Cookbook Evaluation Abstractions.

Provides evaluation utilities and benchmark integrations for assessing
model quality during and after training.

Key Components:

1. Evaluator - Base class for implementing custom evaluators
2. Benchmark Integration - Support for InspectAI and other benchmarks
3. Metrics - Common metrics for different tasks

Supported Benchmarks:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math)
- HumanEval (Code generation)
- TruthfulQA
- Custom benchmarks via InspectAI

Usage:
    from tensafe.cookbook.eval import Evaluator, run_evaluation

    evaluator = Evaluator(
        benchmarks=["mmlu", "gsm8k"],
        model_name="meta-llama/Llama-3.1-8B",
    )
    results = await evaluator.evaluate(client)
"""

from .base import (
    BenchmarkConfig,
    BenchmarkResult,
    Evaluator,
    EvaluatorConfig,
    Metric,
    MetricResult,
)
from .benchmarks import (
    GSM8KEvaluator,
    HumanEvalEvaluator,
    MMLUEvaluator,
)
from .metrics import accuracy, bleu, exact_match, f1_score, perplexity

__all__ = [
    # Base classes
    "Evaluator",
    "EvaluatorConfig",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Metric",
    "MetricResult",
    # Benchmark evaluators
    "MMLUEvaluator",
    "GSM8KEvaluator",
    "HumanEvalEvaluator",
    # Metrics
    "accuracy",
    "exact_match",
    "f1_score",
    "bleu",
    "perplexity",
]
