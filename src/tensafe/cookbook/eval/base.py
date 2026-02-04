"""
Base evaluation classes and abstractions.

Provides the foundation for implementing custom evaluators and
integrating with benchmark suites.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class SamplingClient(Protocol):
    """Protocol for sampling clients."""

    def sample(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        **kwargs,
    ) -> Any:
        ...


@dataclass
class MetricResult:
    """Result from computing a metric."""

    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"{self.name}: {self.value:.4f}"


class Metric(ABC):
    """Base class for evaluation metrics."""

    name: str = "metric"

    @abstractmethod
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs,
    ) -> MetricResult:
        """
        Compute the metric.

        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional arguments

        Returns:
            MetricResult with computed value
        """
        pass


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark."""

    name: str
    dataset: str = ""
    split: str = "test"
    num_samples: Optional[int] = None  # None = use all
    max_tokens: int = 256
    temperature: float = 0.0  # Greedy by default for evaluation
    few_shot: int = 0  # Number of few-shot examples
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])


@dataclass
class BenchmarkResult:
    """Result from running a benchmark."""

    name: str
    metrics: Dict[str, float]
    num_samples: int
    num_correct: int = 0
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """Accuracy if available."""
        return self.metrics.get("accuracy", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "num_correct": self.num_correct,
            "accuracy": self.accuracy,
        }


@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator."""

    model_name: str = "meta-llama/Llama-3.1-8B"
    benchmarks: List[str] = field(default_factory=lambda: ["mmlu"])
    benchmark_configs: Dict[str, BenchmarkConfig] = field(default_factory=dict)

    # Generation settings
    max_tokens: int = 256
    temperature: float = 0.0
    batch_size: int = 32

    # Logging
    log_examples: bool = True
    num_examples_to_log: int = 10


class Evaluator:
    """
    Main evaluator class for running benchmarks.

    Orchestrates benchmark execution and result aggregation.
    """

    # Registry of available benchmarks
    BENCHMARK_REGISTRY: Dict[str, type] = {}

    def __init__(
        self,
        config: EvaluatorConfig,
        client: SamplingClient,
    ):
        """
        Initialize evaluator.

        Args:
            config: Evaluator configuration
            client: Sampling client for model inference
        """
        self.config = config
        self.client = client

        # Load renderers
        from ..tokenizer_utils import get_tokenizer
        from ..renderers import get_renderer
        from ..model_info import get_recommended_renderer_name

        self.tokenizer = get_tokenizer(config.model_name)
        renderer_name = get_recommended_renderer_name(config.model_name)
        self.renderer = get_renderer(renderer_name, self.tokenizer)

    @classmethod
    def register_benchmark(cls, name: str, evaluator_class: type) -> None:
        """Register a benchmark evaluator."""
        cls.BENCHMARK_REGISTRY[name.lower()] = evaluator_class

    def _get_benchmark_evaluator(self, name: str) -> "BenchmarkEvaluator":
        """Get evaluator for a benchmark."""
        name_lower = name.lower()

        if name_lower in self.BENCHMARK_REGISTRY:
            evaluator_class = self.BENCHMARK_REGISTRY[name_lower]
            config = self.config.benchmark_configs.get(
                name, BenchmarkConfig(name=name)
            )
            return evaluator_class(config, self.client, self.renderer)

        raise ValueError(f"Unknown benchmark: {name}")

    async def evaluate(
        self,
        benchmarks: Optional[List[str]] = None,
    ) -> Dict[str, BenchmarkResult]:
        """
        Run evaluation on specified benchmarks.

        Args:
            benchmarks: List of benchmark names (None = use config)

        Returns:
            Dictionary mapping benchmark names to results
        """
        benchmarks = benchmarks or self.config.benchmarks
        results = {}

        for benchmark_name in benchmarks:
            logger.info(f"Running benchmark: {benchmark_name}")

            try:
                evaluator = self._get_benchmark_evaluator(benchmark_name)
                result = await evaluator.run()
                results[benchmark_name] = result

                logger.info(
                    f"  {benchmark_name}: {result.accuracy:.2%} accuracy "
                    f"({result.num_correct}/{result.num_samples})"
                )
            except Exception as e:
                logger.error(f"  Error running {benchmark_name}: {e}")
                results[benchmark_name] = BenchmarkResult(
                    name=benchmark_name,
                    metrics={"error": 1.0},
                    num_samples=0,
                    metadata={"error": str(e)},
                )

        return results

    def summarize(
        self, results: Dict[str, BenchmarkResult]
    ) -> Dict[str, Any]:
        """
        Summarize evaluation results.

        Args:
            results: Results from evaluate()

        Returns:
            Summary dictionary
        """
        summary = {
            "model": self.config.model_name,
            "benchmarks": {},
            "overall_accuracy": 0.0,
        }

        total_correct = 0
        total_samples = 0

        for name, result in results.items():
            summary["benchmarks"][name] = {
                "accuracy": result.accuracy,
                "num_samples": result.num_samples,
                "metrics": result.metrics,
            }
            total_correct += result.num_correct
            total_samples += result.num_samples

        if total_samples > 0:
            summary["overall_accuracy"] = total_correct / total_samples

        return summary


class BenchmarkEvaluator(ABC):
    """
    Base class for benchmark-specific evaluators.

    Each benchmark implements its own evaluation logic.
    """

    name: str = "benchmark"

    def __init__(
        self,
        config: BenchmarkConfig,
        client: SamplingClient,
        renderer: Any,
    ):
        """
        Initialize benchmark evaluator.

        Args:
            config: Benchmark configuration
            client: Sampling client
            renderer: Renderer for prompt formatting
        """
        self.config = config
        self.client = client
        self.renderer = renderer

    @abstractmethod
    async def run(self) -> BenchmarkResult:
        """
        Run the benchmark evaluation.

        Returns:
            BenchmarkResult with evaluation metrics
        """
        pass

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """
        Load the benchmark dataset.

        Returns:
            List of examples
        """
        pass

    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        Format an example into a prompt.

        Args:
            example: Dataset example

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def check_answer(
        self,
        prediction: str,
        example: Dict[str, Any],
    ) -> bool:
        """
        Check if prediction is correct.

        Args:
            prediction: Model prediction
            example: Original example with ground truth

        Returns:
            True if correct
        """
        pass

    async def generate_responses(
        self,
        prompts: List[str],
        batch_size: int = 32,
    ) -> List[str]:
        """
        Generate responses for prompts in batches.

        Args:
            prompts: List of prompts
            batch_size: Batch size for generation

        Returns:
            List of generated responses
        """
        responses = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]

            result = self.client.sample(
                prompts=batch,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            if hasattr(result, "samples"):
                batch_responses = [s.completion for s in result.samples]
            else:
                batch_responses = [
                    s.get("completion", "") for s in result.get("samples", [])
                ]

            responses.extend(batch_responses)

        return responses
