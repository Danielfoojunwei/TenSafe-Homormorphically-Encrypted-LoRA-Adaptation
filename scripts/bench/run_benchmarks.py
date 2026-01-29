#!/usr/bin/env python3
"""
Performance Benchmark Runner for Privacy Tinker

Runs benchmarks across privacy modes and collects:
- p50/p95 latency
- Throughput (tok/s where applicable)
- Memory usage (RSS delta)
- Breakdown timings

Modes:
- smoke: Quick CI benchmark (~5 min)
- full: Complete benchmark suite (~30 min)
"""

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import psutil for memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed, memory tracking disabled")


@dataclass
class BenchmarkResult:
    """Result from a single benchmark scenario."""
    scenario: str
    privacy_mode: str
    iterations: int
    warmup_iterations: int

    # Latency metrics (milliseconds)
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    latency_std_ms: float

    # Throughput (operations per second)
    throughput_ops: float

    # Memory (MB)
    memory_rss_start_mb: float
    memory_rss_end_mb: float
    memory_rss_delta_mb: float

    # Breakdown timings (milliseconds)
    breakdown: Dict[str, float] = field(default_factory=dict)

    # Raw data for analysis
    raw_latencies_ms: List[float] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str
    git_sha: str
    git_branch: str
    mode: str  # smoke or full
    machine_info: Dict[str, Any]
    scenarios: List[BenchmarkResult]
    overhead_analysis: Dict[str, Any]


def get_git_info() -> tuple:
    """Get current git SHA and branch."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        sha = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        branch = "unknown"

    return sha, branch


def get_machine_info() -> Dict[str, Any]:
    """Collect machine information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
    }

    if HAS_PSUTIL:
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_available_gb"] = round(mem.available / (1024**3), 2)

    return info


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


# ============================================================================
# Benchmark Scenarios
# ============================================================================

def benchmark_schema_validation(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark Pydantic schema validation (TG-Tinker schemas)."""
    from tg_tinker.schemas import (
        TrainingConfig, LoRAConfig, DPConfig,
        ForwardBackwardRequest, BatchData,
    )

    # Sample data
    config_data = {
        "model_ref": "meta-llama/Llama-3-8B",
        "lora_config": {"rank": 16, "alpha": 32, "dropout": 0.1},
        "dp_config": {"enabled": True, "noise_multiplier": 1.0, "max_grad_norm": 1.0},
    }

    batch_data = {
        "input_ids": [[1, 2, 3, 4, 5] * 100],
        "attention_mask": [[1, 1, 1, 1, 1] * 100],
        "labels": [[1, 2, 3, 4, 5] * 100],
    }

    # Warmup
    for _ in range(warmup):
        TrainingConfig(**config_data)
        BatchData(**batch_data)

    gc.collect()
    mem_start = get_memory_mb()
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        TrainingConfig(**config_data)
        BatchData(**batch_data)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)  # Convert to ms

    mem_end = get_memory_mb()
    latencies_ms = latencies

    return BenchmarkResult(
        scenario="schema_validation",
        privacy_mode="off",
        iterations=iterations,
        warmup_iterations=warmup,
        latency_p50_ms=percentile(latencies_ms, 50),
        latency_p95_ms=percentile(latencies_ms, 95),
        latency_mean_ms=statistics.mean(latencies_ms),
        latency_min_ms=min(latencies_ms),
        latency_max_ms=max(latencies_ms),
        latency_std_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
        throughput_ops=iterations / (sum(latencies_ms) / 1000),
        memory_rss_start_mb=mem_start,
        memory_rss_end_mb=mem_end,
        memory_rss_delta_mb=mem_end - mem_start,
        raw_latencies_ms=latencies_ms[:100],  # Keep first 100 for analysis
    )


def benchmark_encryption(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark AES-256-GCM encryption (artifact storage)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    import secrets

    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(12)
    data = b"x" * 1024 * 1024  # 1MB payload
    aad = b"artifact_id|tenant_id|training_client_id"

    aesgcm = AESGCM(key)

    # Warmup
    for _ in range(warmup):
        ciphertext = aesgcm.encrypt(nonce, data, aad)
        aesgcm.decrypt(nonce, ciphertext, aad)

    gc.collect()
    mem_start = get_memory_mb()
    latencies = []

    for _ in range(iterations):
        nonce = secrets.token_bytes(12)
        start = time.perf_counter_ns()
        ciphertext = aesgcm.encrypt(nonce, data, aad)
        aesgcm.decrypt(nonce, ciphertext, aad)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)

    mem_end = get_memory_mb()

    return BenchmarkResult(
        scenario="encryption_1mb",
        privacy_mode="off",
        iterations=iterations,
        warmup_iterations=warmup,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_mean_ms=statistics.mean(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops=iterations / (sum(latencies) / 1000),
        memory_rss_start_mb=mem_start,
        memory_rss_end_mb=mem_end,
        memory_rss_delta_mb=mem_end - mem_start,
        breakdown={"encrypt_decrypt_cycle": statistics.mean(latencies)},
        raw_latencies_ms=latencies[:100],
    )


def benchmark_hash_chain(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark hash-chained audit logging."""
    import hashlib
    import json as json_module
    from datetime import datetime

    def compute_hash(prev_hash: str, entry: dict) -> str:
        data = json_module.dumps(entry, sort_keys=True)
        combined = f"{prev_hash}|{data}"
        return hashlib.sha256(combined.encode()).hexdigest()

    prev_hash = "0" * 64
    entry = {
        "entry_id": "audit-12345",
        "tenant_id": "tenant-abc",
        "training_client_id": "tc-xyz",
        "operation": "forward_backward",
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Warmup
    for _ in range(warmup):
        compute_hash(prev_hash, entry)

    gc.collect()
    mem_start = get_memory_mb()
    latencies = []

    for i in range(iterations):
        entry["entry_id"] = f"audit-{i}"
        start = time.perf_counter_ns()
        prev_hash = compute_hash(prev_hash, entry)
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)

    mem_end = get_memory_mb()

    return BenchmarkResult(
        scenario="hash_chain",
        privacy_mode="off",
        iterations=iterations,
        warmup_iterations=warmup,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_mean_ms=statistics.mean(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops=iterations / (sum(latencies) / 1000),
        memory_rss_start_mb=mem_start,
        memory_rss_end_mb=mem_end,
        memory_rss_delta_mb=mem_end - mem_start,
        raw_latencies_ms=latencies[:100],
    )


def benchmark_dp_accountant(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark differential privacy accountant."""
    sys.path.insert(0, str(Path.cwd() / "src"))

    try:
        from tensorguard.platform.tg_tinker_api.dp import RDPAccountant
    except ImportError:
        # Return empty result if DP module not available
        return BenchmarkResult(
            scenario="dp_accountant",
            privacy_mode="off",
            iterations=0,
            warmup_iterations=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_mean_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            latency_std_ms=0,
            throughput_ops=0,
            memory_rss_start_mb=0,
            memory_rss_end_mb=0,
            memory_rss_delta_mb=0,
        )

    accountant = RDPAccountant(target_delta=1e-5)
    noise_multiplier = 1.0
    sample_rate = 0.01

    # Warmup
    for _ in range(warmup):
        accountant.step(noise_multiplier, sample_rate)

    gc.collect()
    mem_start = get_memory_mb()
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter_ns()
        accountant.step(noise_multiplier, sample_rate)
        accountant.get_privacy_spent()
        end = time.perf_counter_ns()
        latencies.append((end - start) / 1_000_000)

    mem_end = get_memory_mb()

    return BenchmarkResult(
        scenario="dp_accountant",
        privacy_mode="off",
        iterations=iterations,
        warmup_iterations=warmup,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_mean_ms=statistics.mean(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops=iterations / (sum(latencies) / 1000) if sum(latencies) > 0 else 0,
        memory_rss_start_mb=mem_start,
        memory_rss_end_mb=mem_end,
        memory_rss_delta_mb=mem_end - mem_start,
        raw_latencies_ms=latencies[:100],
    )


def benchmark_artifact_store(iterations: int, warmup: int) -> BenchmarkResult:
    """Benchmark encrypted artifact storage."""
    sys.path.insert(0, str(Path.cwd() / "src"))

    try:
        from tensorguard.platform.tg_tinker_api.storage import (
            LocalStorageBackend, EncryptedArtifactStore, KeyManager
        )
    except ImportError:
        return BenchmarkResult(
            scenario="artifact_store",
            privacy_mode="off",
            iterations=0,
            warmup_iterations=0,
            latency_p50_ms=0,
            latency_p95_ms=0,
            latency_mean_ms=0,
            latency_min_ms=0,
            latency_max_ms=0,
            latency_std_ms=0,
            throughput_ops=0,
            memory_rss_start_mb=0,
            memory_rss_end_mb=0,
            memory_rss_delta_mb=0,
        )

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalStorageBackend(tmpdir)
        key_manager = KeyManager()
        store = EncryptedArtifactStore(backend, key_manager)

        # 100KB payload
        data = b"x" * 100 * 1024

        # Warmup
        for i in range(warmup):
            artifact = store.save_artifact(
                data, "tenant-1", f"tc-warmup-{i}", "checkpoint"
            )
            store.load_artifact(artifact)

        gc.collect()
        mem_start = get_memory_mb()
        latencies = []

        for i in range(iterations):
            start = time.perf_counter_ns()
            artifact = store.save_artifact(
                data, "tenant-1", f"tc-bench-{i}", "checkpoint"
            )
            store.load_artifact(artifact)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1_000_000)

        mem_end = get_memory_mb()

    return BenchmarkResult(
        scenario="artifact_store_100kb",
        privacy_mode="off",
        iterations=iterations,
        warmup_iterations=warmup,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        latency_mean_ms=statistics.mean(latencies),
        latency_min_ms=min(latencies),
        latency_max_ms=max(latencies),
        latency_std_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops=iterations / (sum(latencies) / 1000) if sum(latencies) > 0 else 0,
        memory_rss_start_mb=mem_start,
        memory_rss_end_mb=mem_end,
        memory_rss_delta_mb=mem_end - mem_start,
        breakdown={"save_load_cycle": statistics.mean(latencies)},
        raw_latencies_ms=latencies[:100],
    )


# ============================================================================
# Main Runner
# ============================================================================

SCENARIOS_SMOKE = [
    ("schema_validation", benchmark_schema_validation, 100, 10),
    ("encryption_1mb", benchmark_encryption, 50, 5),
    ("hash_chain", benchmark_hash_chain, 500, 50),
    ("dp_accountant", benchmark_dp_accountant, 100, 10),
]

SCENARIOS_FULL = [
    ("schema_validation", benchmark_schema_validation, 1000, 100),
    ("encryption_1mb", benchmark_encryption, 200, 20),
    ("hash_chain", benchmark_hash_chain, 5000, 500),
    ("dp_accountant", benchmark_dp_accountant, 1000, 100),
    ("artifact_store_100kb", benchmark_artifact_store, 100, 10),
]


def run_benchmarks(
    mode: str = "smoke",
    output_dir: Optional[Path] = None,
) -> BenchmarkReport:
    """Run benchmark suite."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha, git_branch = get_git_info()

    if output_dir is None:
        output_dir = Path("reports/bench") / git_sha

    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = SCENARIOS_SMOKE if mode == "smoke" else SCENARIOS_FULL

    print(f"\nBenchmark Runner")
    print(f"================")
    print(f"Mode: {mode}")
    print(f"Timestamp: {timestamp}")
    print(f"Git SHA: {git_sha}")
    print(f"Output: {output_dir}")
    print(f"Scenarios: {len(scenarios)}")

    machine_info = get_machine_info()
    results = []

    for name, func, iterations, warmup in scenarios:
        print(f"\n  Running: {name} ({iterations} iterations, {warmup} warmup)...")
        try:
            result = func(iterations, warmup)
            results.append(result)
            print(f"    p50: {result.latency_p50_ms:.3f}ms, p95: {result.latency_p95_ms:.3f}ms, throughput: {result.throughput_ops:.1f} ops/s")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Calculate overhead analysis
    overhead_analysis = {}
    baseline_results = {r.scenario: r for r in results}

    # Schema validation baseline
    if "schema_validation" in baseline_results:
        overhead_analysis["schema_validation_baseline_p50_ms"] = baseline_results["schema_validation"].latency_p50_ms

    # Encryption overhead
    if "encryption_1mb" in baseline_results:
        overhead_analysis["encryption_overhead_p50_ms"] = baseline_results["encryption_1mb"].latency_p50_ms

    report = BenchmarkReport(
        timestamp=timestamp,
        git_sha=git_sha,
        git_branch=git_branch,
        mode=mode,
        machine_info=machine_info,
        scenarios=[asdict(r) for r in results],
        overhead_analysis=overhead_analysis,
    )

    # Write JSON report
    json_file = output_dir / f"bench_{mode}.json"
    with open(json_file, "w") as f:
        json.dump(asdict(report), f, indent=2)

    # Write markdown report
    md_file = output_dir / f"bench_{mode}.md"
    with open(md_file, "w") as f:
        f.write(f"# Benchmark Report ({mode})\n\n")
        f.write(f"**Timestamp**: {timestamp}\n")
        f.write(f"**Git SHA**: {git_sha}\n")
        f.write(f"**Git Branch**: {git_branch}\n\n")

        f.write("## Machine Info\n\n")
        for k, v in machine_info.items():
            f.write(f"- **{k}**: {v}\n")

        f.write("\n## Results\n\n")
        f.write("| Scenario | p50 (ms) | p95 (ms) | Mean (ms) | Throughput (ops/s) | Memory Delta (MB) |\n")
        f.write("|----------|----------|----------|-----------|-------------------|------------------|\n")

        for r in results:
            f.write(f"| {r.scenario} | {r.latency_p50_ms:.3f} | {r.latency_p95_ms:.3f} | {r.latency_mean_ms:.3f} | {r.throughput_ops:.1f} | {r.memory_rss_delta_mb:.2f} |\n")

    print(f"\n{'='*60}")
    print(f"Benchmarks Complete")
    print(f"{'='*60}")
    print(f"JSON: {json_file}")
    print(f"Markdown: {md_file}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Run Privacy Tinker benchmarks")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Benchmark mode (smoke: ~5min, full: ~30min)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )

    args = parser.parse_args()

    run_benchmarks(
        mode=args.mode,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
