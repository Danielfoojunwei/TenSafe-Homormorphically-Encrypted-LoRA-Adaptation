"""
End-to-End Benchmarks for HE-LoRA Microkernel

These benchmarks measure complete HE-LoRA inference:
  - Per-sequence tok/s
  - Aggregate tok/s
  - Rotations per token
  - Keyswitches per token
  - HE time percentage

Benchmarks are run across:
  - Multiple batch sizes
  - Multiple hidden sizes
  - Multiple ranks
"""

import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.compiler import (
    LoRAConfig,
    LoRATargets,
    CKKSProfile,
    get_profile,
    compile_schedule,
    select_optimal_profile,
)
from he_lora_microkernel.runtime import (
    HELoRAExecutor,
    TelemetryCollector,
    PerformanceReporter,
)
from he_lora_microkernel.backend.gpu_ckks_backend import BackendType


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    hidden_size: int
    rank: int
    batch_size: int
    num_tokens: int = 100
    profile: CKKSProfile = CKKSProfile.FAST
    targets: LoRATargets = LoRATargets.QKV


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    config: BenchmarkConfig
    valid: bool = True
    error_message: Optional[str] = None

    # Throughput metrics
    total_tokens: int = 0
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0
    aggregate_tokens_per_second: float = 0.0
    ms_per_token: float = 0.0

    # Operation metrics
    avg_rotations_per_token: float = 0.0
    avg_keyswitches_per_token: float = 0.0
    avg_rescales_per_token: float = 0.0

    # Time breakdown
    avg_he_time_ms: float = 0.0
    avg_encrypt_time_ms: float = 0.0
    avg_compute_time_ms: float = 0.0
    avg_decrypt_time_ms: float = 0.0
    he_time_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config': {
                'hidden_size': self.config.hidden_size,
                'rank': self.config.rank,
                'batch_size': self.config.batch_size,
                'num_tokens': self.config.num_tokens,
                'profile': self.config.profile.value,
                'targets': self.config.targets.value,
            },
            'valid': self.valid,
            'error_message': self.error_message,
            'throughput': {
                'total_tokens': self.total_tokens,
                'total_time_ms': self.total_time_ms,
                'tokens_per_second': self.tokens_per_second,
                'aggregate_tokens_per_second': self.aggregate_tokens_per_second,
                'ms_per_token': self.ms_per_token,
            },
            'operations': {
                'avg_rotations_per_token': self.avg_rotations_per_token,
                'avg_keyswitches_per_token': self.avg_keyswitches_per_token,
                'avg_rescales_per_token': self.avg_rescales_per_token,
            },
            'timing': {
                'avg_he_time_ms': self.avg_he_time_ms,
                'avg_encrypt_time_ms': self.avg_encrypt_time_ms,
                'avg_compute_time_ms': self.avg_compute_time_ms,
                'avg_decrypt_time_ms': self.avg_decrypt_time_ms,
                'he_time_percentage': self.he_time_percentage,
            },
        }


class EndToEndBenchmarker:
    """Runs end-to-end HE-LoRA benchmarks."""

    def __init__(
        self,
        backend_type: BackendType = BackendType.SIMULATION,
        warmup_tokens: int = 5,
    ):
        """
        Initialize benchmarker.

        Args:
            backend_type: Backend to use
            warmup_tokens: Warmup tokens before timing
        """
        self.backend_type = backend_type
        self.warmup_tokens = warmup_tokens
        self.rng = np.random.default_rng(42)

    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        result = BenchmarkResult(config=config)

        try:
            # Auto-select optimal profile for large configurations
            try:
                ckks_params = select_optimal_profile(
                    hidden_size=config.hidden_size,
                    lora_rank=config.rank,
                    batch_size=config.batch_size,
                )
                profile = ckks_params.profile
            except ValueError:
                # Fall back to specified profile
                profile = config.profile
                ckks_params = get_profile(profile)

            # Create LoRA config
            lora_config = LoRAConfig(
                hidden_size=config.hidden_size,
                rank=config.rank,
                alpha=2.0 * config.rank,
                targets=config.targets,
                batch_size=config.batch_size,
                max_context_length=config.num_tokens + self.warmup_tokens,
                ckks_profile=profile,
            )

            # Compile
            schedule = compile_schedule(lora_config, ckks_params)

            if not schedule.is_valid:
                result.valid = False
                result.error_message = str(schedule.validation_errors)
                return result

            # Create executor
            executor = HELoRAExecutor(schedule, self.backend_type)

            # Generate weights
            A = self.rng.standard_normal(
                (config.hidden_size, config.rank)
            ).astype(np.float64) * 0.01
            B = self.rng.standard_normal(
                (config.rank, config.hidden_size)
            ).astype(np.float64) * 0.01

            executor.load_weights(A, B, lora_config.alpha)

            # Warmup
            for i in range(self.warmup_tokens):
                x = self.rng.standard_normal(
                    (config.batch_size, config.hidden_size)
                ).astype(np.float64)
                executor.execute_token(x, position=i)

            # Reset statistics after warmup
            executor.reset_statistics()

            # Benchmark
            start_time = time.perf_counter()

            for i in range(config.num_tokens):
                x = self.rng.standard_normal(
                    (config.batch_size, config.hidden_size)
                ).astype(np.float64)
                executor.execute_token(x, position=i + self.warmup_tokens)

            end_time = time.perf_counter()

            # Collect results
            total_time_ms = (end_time - start_time) * 1000
            stats = executor.get_statistics()
            counters = stats['backend_counters']

            result.total_tokens = config.num_tokens
            result.total_time_ms = total_time_ms
            result.tokens_per_second = config.num_tokens * 1000 / total_time_ms
            result.aggregate_tokens_per_second = (
                config.num_tokens * config.batch_size * 1000 / total_time_ms
            )
            result.ms_per_token = total_time_ms / config.num_tokens

            result.avg_rotations_per_token = counters['rotations'] / config.num_tokens
            result.avg_keyswitches_per_token = counters['keyswitches'] / config.num_tokens
            result.avg_rescales_per_token = counters['rescales'] / config.num_tokens

            result.avg_he_time_ms = counters['total_time_ms'] / config.num_tokens
            result.avg_encrypt_time_ms = counters['encrypt_time_ms'] / config.num_tokens
            result.avg_compute_time_ms = counters['compute_time_ms'] / config.num_tokens
            result.avg_decrypt_time_ms = counters['decrypt_time_ms'] / config.num_tokens

            if result.ms_per_token > 0:
                result.he_time_percentage = (
                    result.avg_he_time_ms / result.ms_per_token * 100
                )

        except Exception as e:
            result.valid = False
            result.error_message = str(e)

        return result

    def run_sweep(
        self,
        batch_sizes: List[int] = [1, 4, 8],
        hidden_sizes: List[int] = [512, 1024],
        ranks: List[int] = [8, 16],
        num_tokens: int = 50,
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks across parameter sweep.

        Args:
            batch_sizes: Batch sizes to test
            hidden_sizes: Hidden sizes to test
            ranks: Ranks to test
            num_tokens: Tokens per benchmark

        Returns:
            List of benchmark results
        """
        results = []

        total_configs = len(batch_sizes) * len(hidden_sizes) * len(ranks)
        current = 0

        print(f"Running {total_configs} benchmark configurations...")
        print()

        for hidden_size in hidden_sizes:
            for rank in ranks:
                for batch_size in batch_sizes:
                    current += 1
                    config = BenchmarkConfig(
                        hidden_size=hidden_size,
                        rank=rank,
                        batch_size=batch_size,
                        num_tokens=num_tokens,
                    )

                    print(
                        f"  [{current}/{total_configs}] "
                        f"h={hidden_size}, r={rank}, b={batch_size}...",
                        end=" ", flush=True
                    )

                    result = self.run_benchmark(config)
                    results.append(result)

                    if result.valid:
                        print(f"{result.tokens_per_second:.1f} tok/s")
                    else:
                        print(f"FAILED: {result.error_message}")

        return results

    def generate_report(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """Generate benchmark report."""
        valid_results = [r for r in results if r.valid]
        failed_results = [r for r in results if not r.valid]

        # Find best configurations
        best_throughput = max(valid_results, key=lambda r: r.tokens_per_second) if valid_results else None
        best_aggregate = max(valid_results, key=lambda r: r.aggregate_tokens_per_second) if valid_results else None
        min_rotations = min(valid_results, key=lambda r: r.avg_rotations_per_token) if valid_results else None

        return {
            'summary': {
                'total_configs': len(results),
                'valid_configs': len(valid_results),
                'failed_configs': len(failed_results),
            },
            'best_configurations': {
                'highest_throughput': best_throughput.to_dict() if best_throughput else None,
                'highest_aggregate': best_aggregate.to_dict() if best_aggregate else None,
                'lowest_rotations': min_rotations.to_dict() if min_rotations else None,
            },
            'all_results': [r.to_dict() for r in results],
        }


def run_e2e_benchmarks(
    backend_type: str = "SIMULATION",
    batch_sizes: List[int] = [1, 4, 8],
    hidden_sizes: List[int] = [512, 1024],
    ranks: List[int] = [8, 16],
    num_tokens: int = 50,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run end-to-end benchmarks.

    Args:
        backend_type: Backend type string
        batch_sizes: Batch sizes to test
        hidden_sizes: Hidden sizes to test
        ranks: Ranks to test
        num_tokens: Tokens per benchmark
        output_file: Optional file to save results

    Returns:
        Benchmark report
    """
    backend = BackendType[backend_type]
    benchmarker = EndToEndBenchmarker(backend_type=backend)

    results = benchmarker.run_sweep(
        batch_sizes=batch_sizes,
        hidden_sizes=hidden_sizes,
        ranks=ranks,
        num_tokens=num_tokens,
    )

    report = benchmarker.generate_report(results)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HE-LoRA End-to-End Benchmarks")
    parser.add_argument("--backend", default="SIMULATION", help="Backend type")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--num-tokens", type=int, default=50)
    parser.add_argument("--output", default=None, help="Output file")

    args = parser.parse_args()

    report = run_e2e_benchmarks(
        backend_type=args.backend,
        batch_sizes=args.batch_sizes,
        hidden_sizes=args.hidden_sizes,
        ranks=args.ranks,
        num_tokens=args.num_tokens,
        output_file=args.output,
    )

    print("\n" + "=" * 60)
    print("END-TO-END BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total configurations: {report['summary']['total_configs']}")
    print(f"Valid configurations: {report['summary']['valid_configs']}")
    print(f"Failed configurations: {report['summary']['failed_configs']}")
    print()

    if report['best_configurations']['highest_throughput']:
        best = report['best_configurations']['highest_throughput']
        print(f"Best throughput: {best['throughput']['tokens_per_second']:.1f} tok/s")
        print(f"  Config: h={best['config']['hidden_size']}, "
              f"r={best['config']['rank']}, b={best['config']['batch_size']}")
