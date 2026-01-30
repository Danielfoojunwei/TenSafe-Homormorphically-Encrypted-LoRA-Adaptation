"""
Cryptographic Benchmarking for N2HE.

Provides reproducible benchmarking of HE operations to:
    - Measure performance of encryption/decryption/computation
    - Estimate noise budget consumption
    - Compare simulation vs native N2HE performance
    - Generate compliance evidence for encrypted compute

This supports architectural option 3: reproducible cryptographic benchmarking.
"""

import logging
import secrets
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import (
    HESchemeParams,
    N2HEContext,
    create_context,
)
from .keys import HEKeyManager
from .serialization import CiphertextFormat, CiphertextSerializer

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark operation."""

    operation: str
    iterations: int
    total_time_ms: float
    mean_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    ops_per_second: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional metrics
    noise_budget_consumed: Optional[float] = None
    memory_bytes: Optional[int] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "operation": self.operation,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "mean_time_ms": self.mean_time_ms,
            "std_dev_ms": self.std_dev_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "ops_per_second": self.ops_per_second,
            "timestamp": self.timestamp.isoformat(),
            "noise_budget_consumed": self.noise_budget_consumed,
            "memory_bytes": self.memory_bytes,
            "extra_metrics": self.extra_metrics,
        }


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    suite_id: str
    name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    params: Optional[HESchemeParams] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Environment info
    platform: str = ""
    simulation_mode: bool = True

    def add_result(self, result: BenchmarkResult) -> None:
        """Add a benchmark result."""
        self.results.append(result)

    def get_summary(self) -> Dict[str, Any]:
        """Get suite summary."""
        return {
            "suite_id": self.suite_id,
            "name": self.name,
            "num_benchmarks": len(self.results),
            "total_time_ms": sum(r.total_time_ms for r in self.results),
            "params_hash": self.params.get_hash() if self.params else None,
            "simulation_mode": self.simulation_mode,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = self.get_summary()
        data["results"] = [r.to_dict() for r in self.results]
        return data


class N2HEBenchmark:
    """
    Benchmarking suite for N2HE operations.

    Provides reproducible benchmarks for:
        - Key generation
        - Encryption/Decryption
        - Homomorphic operations (add, multiply, matmul)
        - LoRA delta computation
        - Serialization/deserialization
    """

    def __init__(
        self,
        params: Optional[HESchemeParams] = None,
        warmup_iterations: int = 3,
        default_iterations: int = 10,
    ):
        """
        Initialize benchmark suite.

        Args:
            params: HE scheme parameters
            warmup_iterations: Warmup iterations before timing
            default_iterations: Default iterations for benchmarks
        """
        self.params = params or HESchemeParams.default_lora_params()
        self.warmup_iterations = warmup_iterations
        self.default_iterations = default_iterations

        # Context for benchmarking
        self._context: Optional[N2HEContext] = None
        self._key_manager: Optional[HEKeyManager] = None

    def _get_context(self) -> N2HEContext:
        """Get or create benchmark context."""
        if self._context is None:
            self._context = create_context(profile="lora", use_simulation=True)
            self._context.generate_keys()
        return self._context

    def _get_key_manager(self) -> HEKeyManager:
        """Get or create key manager."""
        if self._key_manager is None:
            self._key_manager = HEKeyManager()
        return self._key_manager

    def _benchmark_operation(
        self,
        name: str,
        operation,
        iterations: Optional[int] = None,
        **kwargs,
    ) -> BenchmarkResult:
        """
        Run a benchmark for an operation.

        Args:
            name: Operation name
            operation: Callable to benchmark
            iterations: Number of iterations
            **kwargs: Additional arguments to pass to operation

        Returns:
            BenchmarkResult
        """
        iterations = iterations or self.default_iterations

        # Warmup
        for _ in range(self.warmup_iterations):
            operation(**kwargs)

        # Timing
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation(**kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        total_time = sum(times)
        mean_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

        return BenchmarkResult(
            operation=name,
            iterations=iterations,
            total_time_ms=total_time,
            mean_time_ms=mean_time,
            std_dev_ms=std_dev,
            min_time_ms=min(times),
            max_time_ms=max(times),
            ops_per_second=1000.0 / mean_time if mean_time > 0 else 0,
        )

    def benchmark_keygen(self, iterations: Optional[int] = None) -> BenchmarkResult:
        """Benchmark key generation."""

        def keygen_op():
            ctx = create_context(profile="lora", use_simulation=True)
            ctx.generate_keys()

        return self._benchmark_operation("keygen", keygen_op, iterations)

    def benchmark_encryption(
        self,
        plaintext_size: int = 1024,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark encryption."""
        ctx = self._get_context()
        plaintext = np.random.randint(0, 1000, size=plaintext_size, dtype=np.int64)

        def encrypt_op():
            ctx.encrypt(plaintext)

        result = self._benchmark_operation("encryption", encrypt_op, iterations)
        result.extra_metrics["plaintext_size"] = plaintext_size
        return result

    def benchmark_decryption(
        self,
        plaintext_size: int = 1024,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark decryption."""
        ctx = self._get_context()
        plaintext = np.random.randint(0, 1000, size=plaintext_size, dtype=np.int64)
        ciphertext = ctx.encrypt(plaintext)

        def decrypt_op():
            ctx.decrypt(ciphertext)

        result = self._benchmark_operation("decryption", decrypt_op, iterations)
        result.extra_metrics["plaintext_size"] = plaintext_size
        return result

    def benchmark_homomorphic_add(
        self,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark homomorphic addition."""
        ctx = self._get_context()
        pt1 = np.array([100], dtype=np.int64)
        pt2 = np.array([200], dtype=np.int64)
        ct1 = ctx.encrypt(pt1)
        ct2 = ctx.encrypt(pt2)

        def add_op():
            ctx.scheme.add(ct1, ct2)

        result = self._benchmark_operation("homomorphic_add", add_op, iterations)

        # Measure noise consumption
        ct_result = ctx.scheme.add(ct1, ct2)
        if hasattr(ct_result, "noise_budget") and hasattr(ct1, "noise_budget"):
            result.noise_budget_consumed = (ct1.noise_budget or 0) - (
                ct_result.noise_budget or 0
            )

        return result

    def benchmark_homomorphic_multiply(
        self,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark plaintext-ciphertext multiplication."""
        ctx = self._get_context()
        pt = np.array([100], dtype=np.int64)
        ct = ctx.encrypt(pt)
        scalar = np.array([5], dtype=np.int64)

        def mul_op():
            ctx.scheme.multiply(ct, scalar)

        result = self._benchmark_operation("homomorphic_multiply", mul_op, iterations)

        ct_result = ctx.scheme.multiply(ct, scalar)
        if hasattr(ct_result, "noise_budget") and hasattr(ct, "noise_budget"):
            result.noise_budget_consumed = (ct.noise_budget or 0) - (
                ct_result.noise_budget or 0
            )

        return result

    def benchmark_encrypted_matmul(
        self,
        matrix_size: Tuple[int, int] = (16, 16),
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark encrypted matrix multiplication."""
        ctx = self._get_context()
        pt = np.array([100], dtype=np.int64)
        ct = ctx.encrypt(pt)
        weight = np.random.randn(*matrix_size).astype(np.float32)

        def matmul_op():
            ctx.scheme.matmul(ct, weight, ctx._ek)

        result = self._benchmark_operation("encrypted_matmul", matmul_op, iterations)
        result.extra_metrics["matrix_size"] = list(matrix_size)

        ct_result = ctx.scheme.matmul(ct, weight, ctx._ek)
        if hasattr(ct_result, "noise_budget") and hasattr(ct, "noise_budget"):
            result.noise_budget_consumed = (ct.noise_budget or 0) - (
                ct_result.noise_budget or 0
            )

        return result

    def benchmark_lora_delta(
        self,
        rank: int = 16,
        hidden_dim: int = 256,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark encrypted LoRA delta computation."""
        ctx = self._get_context()
        pt = np.random.randint(0, 1000, size=hidden_dim, dtype=np.int64)
        ct = ctx.encrypt(pt)

        lora_a = np.random.randn(rank, hidden_dim).astype(np.float32) * 0.02
        lora_b = np.random.randn(hidden_dim, rank).astype(np.float32) * 0.02

        def lora_op():
            ctx.encrypted_lora_delta(ct, lora_a, lora_b, scaling=2.0)

        result = self._benchmark_operation("lora_delta", lora_op, iterations)
        result.extra_metrics["rank"] = rank
        result.extra_metrics["hidden_dim"] = hidden_dim

        return result

    def benchmark_serialization(
        self,
        format: CiphertextFormat = CiphertextFormat.BINARY,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark ciphertext serialization."""
        ctx = self._get_context()
        pt = np.array([100], dtype=np.int64)
        ct = ctx.encrypt(pt)
        serializer = CiphertextSerializer()

        def serialize_op():
            serializer.serialize(ct, format)

        result = self._benchmark_operation(
            f"serialize_{format.value}", serialize_op, iterations
        )

        serialized = serializer.serialize(ct, format)
        result.memory_bytes = len(serialized.data)

        return result

    def benchmark_deserialization(
        self,
        format: CiphertextFormat = CiphertextFormat.BINARY,
        iterations: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark ciphertext deserialization."""
        ctx = self._get_context()
        pt = np.array([100], dtype=np.int64)
        ct = ctx.encrypt(pt)
        serializer = CiphertextSerializer()
        serialized = serializer.serialize(ct, format)

        def deserialize_op():
            serializer.deserialize(serialized, ctx.params)

        return self._benchmark_operation(
            f"deserialize_{format.value}", deserialize_op, iterations
        )

    def run_full_suite(
        self,
        name: str = "N2HE Full Benchmark",
        iterations: Optional[int] = None,
    ) -> BenchmarkSuite:
        """
        Run the full benchmark suite.

        Args:
            name: Suite name
            iterations: Iterations per benchmark

        Returns:
            BenchmarkSuite with all results
        """
        import platform

        suite = BenchmarkSuite(
            suite_id=f"suite-{secrets.token_hex(6)}",
            name=name,
            params=self.params,
            platform=f"{platform.system()} {platform.machine()}",
            simulation_mode=True,  # Currently always simulation
        )

        logger.info(f"Starting benchmark suite: {name}")

        # Key generation
        suite.add_result(self.benchmark_keygen(iterations))

        # Encryption/Decryption
        for size in [256, 1024]:
            suite.add_result(self.benchmark_encryption(size, iterations))
            suite.add_result(self.benchmark_decryption(size, iterations))

        # Homomorphic operations
        suite.add_result(self.benchmark_homomorphic_add(iterations))
        suite.add_result(self.benchmark_homomorphic_multiply(iterations))

        # Matrix operations
        for size in [(8, 8), (16, 16), (32, 32)]:
            suite.add_result(self.benchmark_encrypted_matmul(size, iterations))

        # LoRA operations
        for rank in [8, 16, 32]:
            suite.add_result(self.benchmark_lora_delta(rank, 256, iterations))

        # Serialization
        for fmt in [CiphertextFormat.BINARY, CiphertextFormat.JSON]:
            suite.add_result(self.benchmark_serialization(fmt, iterations))
            suite.add_result(self.benchmark_deserialization(fmt, iterations))

        suite.completed_at = datetime.utcnow()

        logger.info(
            f"Completed benchmark suite: {len(suite.results)} benchmarks, "
            f"total time: {suite.get_summary()['total_time_ms']:.2f}ms"
        )

        return suite


def run_quick_benchmark() -> BenchmarkSuite:
    """Run a quick benchmark for smoke testing."""
    benchmark = N2HEBenchmark(warmup_iterations=1, default_iterations=3)
    return benchmark.run_full_suite(name="N2HE Quick Benchmark")


def run_full_benchmark() -> BenchmarkSuite:
    """Run a comprehensive benchmark."""
    benchmark = N2HEBenchmark(warmup_iterations=5, default_iterations=20)
    return benchmark.run_full_suite(name="N2HE Full Benchmark")


def generate_benchmark_report(suite: BenchmarkSuite) -> str:
    """
    Generate a human-readable benchmark report.

    Args:
        suite: Benchmark suite results

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 70,
        f"N2HE Benchmark Report: {suite.name}",
        "=" * 70,
        f"Suite ID: {suite.suite_id}",
        f"Platform: {suite.platform}",
        f"Simulation Mode: {suite.simulation_mode}",
        f"Parameters Hash: {suite.params.get_hash()[:16]}..." if suite.params else "",
        f"Started: {suite.started_at.isoformat()}",
        f"Completed: {suite.completed_at.isoformat() if suite.completed_at else 'N/A'}",
        "",
        "-" * 70,
        f"{'Operation':<30} {'Mean (ms)':<12} {'Std Dev':<10} {'Ops/sec':<12}",
        "-" * 70,
    ]

    for result in suite.results:
        lines.append(
            f"{result.operation:<30} {result.mean_time_ms:<12.3f} "
            f"{result.std_dev_ms:<10.3f} {result.ops_per_second:<12.1f}"
        )

    lines.extend([
        "-" * 70,
        f"Total Time: {sum(r.total_time_ms for r in suite.results):.2f}ms",
        f"Benchmarks Run: {len(suite.results)}",
        "=" * 70,
    ])

    return "\n".join(lines)


@dataclass
class ComplianceEvidence:
    """Compliance evidence from N2HE benchmarking."""

    evidence_id: str
    benchmark_suite: BenchmarkSuite
    security_claims: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "generated_at": self.generated_at.isoformat(),
            "benchmark_summary": self.benchmark_suite.get_summary(),
            "security_claims": self.security_claims,
        }


def generate_compliance_evidence(
    suite: BenchmarkSuite,
) -> ComplianceEvidence:
    """
    Generate compliance evidence from benchmark results.

    Args:
        suite: Completed benchmark suite

    Returns:
        ComplianceEvidence for audit
    """
    # Compute security claims based on parameters
    security_claims = {}
    if suite.params:
        security_claims = {
            "scheme_type": suite.params.scheme_type.value,
            "security_level_bits": suite.params.security_level,
            "lattice_dimension": suite.params.n,
            "ciphertext_modulus_bits": int(np.log2(suite.params.q)),
            "noise_standard_deviation": suite.params.std_dev,
        }

    return ComplianceEvidence(
        evidence_id=f"n2he-evidence-{secrets.token_hex(8)}",
        benchmark_suite=suite,
        security_claims=security_claims,
    )
