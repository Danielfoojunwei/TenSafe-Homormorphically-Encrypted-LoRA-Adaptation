"""
Microbenchmarks for HE-LoRA Microkernel

These benchmarks measure individual operation costs:
  - Encryption time
  - Decryption time
  - Ct×Pt multiplication time
  - Rotation time
  - Rescale time

Results are used to validate the cost model and identify bottlenecks.
"""

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from he_lora_microkernel.backend.gpu_ckks_backend import (
    BackendType,
    create_backend,
)
from he_lora_microkernel.compiler import (
    CKKSProfile,
    get_profile,
)


@dataclass
class MicrobenchResult:
    """Result of a microbenchmark."""
    operation: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Microbenchmarker:
    """Runs microbenchmarks on GPU CKKS backend."""

    def __init__(
        self,
        backend_type: BackendType = BackendType.SIMULATION,
        profile: CKKSProfile = CKKSProfile.FAST,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 100,
    ):
        """
        Initialize microbenchmarker.

        Args:
            backend_type: Backend to benchmark
            profile: CKKS profile
            warmup_iterations: Warmup iterations before timing
            benchmark_iterations: Timed iterations
        """
        self.ckks_params = get_profile(profile)
        self.backend = create_backend(backend_type, self.ckks_params)
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

        # Pre-generate test data
        self.rng = np.random.default_rng(42)
        self.test_vector = self.rng.standard_normal(self.ckks_params.slot_count)

    def _time_operation(
        self,
        operation_fn,
        iterations: int,
    ) -> List[float]:
        """Time an operation for multiple iterations."""
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation_fn()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
        return times

    def _run_benchmark(
        self,
        name: str,
        operation_fn,
    ) -> MicrobenchResult:
        """Run a single benchmark."""
        # Warmup
        for _ in range(self.warmup_iterations):
            operation_fn()

        # Benchmark
        times = self._time_operation(operation_fn, self.benchmark_iterations)

        return MicrobenchResult(
            operation=name,
            iterations=self.benchmark_iterations,
            total_time_ms=sum(times),
            avg_time_ms=np.mean(times),
            min_time_ms=np.min(times),
            max_time_ms=np.max(times),
            std_time_ms=np.std(times),
        )

    def bench_encrypt(self) -> MicrobenchResult:
        """Benchmark encryption."""
        return self._run_benchmark(
            "encrypt",
            lambda: self.backend.encrypt(self.test_vector)
        )

    def bench_decrypt(self) -> MicrobenchResult:
        """Benchmark decryption."""
        ct = self.backend.encrypt(self.test_vector)
        return self._run_benchmark(
            "decrypt",
            lambda: self.backend.decrypt(ct)
        )

    def bench_mul_plain(self) -> MicrobenchResult:
        """Benchmark Ct×Pt multiplication."""
        ct = self.backend.encrypt(self.test_vector)
        pt = self.backend.encode_plaintext(self.test_vector)
        return self._run_benchmark(
            "mul_plain",
            lambda: self.backend.mul_plain(ct, pt)
        )

    def bench_add(self) -> MicrobenchResult:
        """Benchmark Ct+Ct addition."""
        ct1 = self.backend.encrypt(self.test_vector)
        ct2 = self.backend.encrypt(self.test_vector)
        return self._run_benchmark(
            "add",
            lambda: self.backend.add(ct1, ct2)
        )

    def bench_rotate(self) -> MicrobenchResult:
        """Benchmark rotation."""
        ct = self.backend.encrypt(self.test_vector)
        return self._run_benchmark(
            "rotate",
            lambda: self.backend.rotate(ct, 1)
        )

    def bench_rescale(self) -> MicrobenchResult:
        """Benchmark rescale."""
        ct = self.backend.encrypt(self.test_vector)
        pt = self.backend.encode_plaintext(self.test_vector)
        ct_mul = self.backend.mul_plain(ct, pt)
        return self._run_benchmark(
            "rescale",
            lambda: self.backend.rescale(ct_mul)
        )

    def run_all(self) -> Dict[str, MicrobenchResult]:
        """Run all microbenchmarks."""
        results = {}

        print("Running microbenchmarks...")
        print(f"  Backend: {type(self.backend).__name__}")
        print(f"  Profile: {self.ckks_params.profile.value}")
        print(f"  Iterations: {self.benchmark_iterations}")
        print()

        benchmarks = [
            ("encrypt", self.bench_encrypt),
            ("decrypt", self.bench_decrypt),
            ("mul_plain", self.bench_mul_plain),
            ("add", self.bench_add),
            ("rotate", self.bench_rotate),
            ("rescale", self.bench_rescale),
        ]

        for name, bench_fn in benchmarks:
            print(f"  Benchmarking {name}...", end=" ", flush=True)
            result = bench_fn()
            results[name] = result
            print(f"{result.avg_time_ms:.3f} ms (±{result.std_time_ms:.3f})")

        return results

    def generate_report(self, results: Dict[str, MicrobenchResult]) -> Dict[str, Any]:
        """Generate benchmark report."""
        return {
            'backend': type(self.backend).__name__,
            'profile': self.ckks_params.profile.value,
            'poly_modulus_degree': self.ckks_params.poly_modulus_degree,
            'slot_count': self.ckks_params.slot_count,
            'iterations': self.benchmark_iterations,
            'results': {k: v.to_dict() for k, v in results.items()},
        }


def run_microbenchmarks(
    backend_type: str = "SIMULATION",
    profile: str = "FAST",
    iterations: int = 100,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run microbenchmarks and optionally save results.

    Args:
        backend_type: Backend type string
        profile: CKKS profile string
        iterations: Number of iterations
        output_file: Optional file to save results

    Returns:
        Benchmark report
    """
    backend = BackendType[backend_type]
    ckks_profile = CKKSProfile[profile]

    benchmarker = Microbenchmarker(
        backend_type=backend,
        profile=ckks_profile,
        benchmark_iterations=iterations,
    )

    results = benchmarker.run_all()
    report = benchmarker.generate_report(results)

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HE-LoRA Microbenchmarks")
    parser.add_argument("--backend", default="SIMULATION", help="Backend type")
    parser.add_argument("--profile", default="FAST", help="CKKS profile")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations")
    parser.add_argument("--output", default=None, help="Output file")

    args = parser.parse_args()

    report = run_microbenchmarks(
        backend_type=args.backend,
        profile=args.profile,
        iterations=args.iterations,
        output_file=args.output,
    )

    print("\n" + "=" * 60)
    print("MICROBENCHMARK SUMMARY")
    print("=" * 60)
    for op, result in report['results'].items():
        print(f"  {op:12s}: {result['avg_time_ms']:8.3f} ms")
