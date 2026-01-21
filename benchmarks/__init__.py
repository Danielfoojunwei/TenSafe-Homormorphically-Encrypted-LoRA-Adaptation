"""
TensorGuardFlow Benchmark Suite

Comprehensive performance benchmarking framework for TensorGuardFlow,
measuring API latency, telemetry throughput, and resource consumption.

Usage:
    # Run all benchmarks
    python -m benchmarks.runner

    # Run with specific load level
    python -m benchmarks.runner --load heavy

    # Analyze results
    python -c "from benchmarks.analyzer import analyze_results; analyze_results('artifacts/benchmarks/benchmark_results_latest.json')"

    # Run regression tests
    python -m benchmarks.regression_test artifacts/benchmarks/benchmark_results_latest.json
"""

from .config import (
    BenchmarkConfig,
    LoadConfig,
    ThresholdConfig,
    LoadLevel,
    get_config,
    get_hardware_info,
)

from .runner import BenchmarkRunner

__version__ = "1.0.0"
__all__ = [
    "BenchmarkConfig",
    "LoadConfig",
    "ThresholdConfig",
    "LoadLevel",
    "BenchmarkRunner",
    "get_config",
    "get_hardware_info",
]
