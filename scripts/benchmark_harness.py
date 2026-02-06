#!/usr/bin/env python
"""
TenSafe Benchmark Harness

Runs benchmarks for SFT and RLVR training to establish performance baselines
and track regressions.

Usage:
    python scripts/benchmark_harness.py
    python scripts/benchmark_harness.py --suite sft
    python scripts/benchmark_harness.py --suite rlvr
    python scripts/benchmark_harness.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    suite: str
    duration_seconds: float
    iterations: int
    throughput: float  # iterations per second
    metrics: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "suite": self.suite,
            "duration_seconds": round(self.duration_seconds, 4),
            "iterations": self.iterations,
            "throughput": round(self.throughput, 2),
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""

    results: List[BenchmarkResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    platform: str = field(default_factory=lambda: sys.platform)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    duration_seconds: float = 0.0

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "platform": self.platform,
            "python_version": self.python_version,
            "duration_seconds": round(self.duration_seconds, 2),
            "num_benchmarks": len(self.results),
            "num_passed": sum(1 for r in self.results if r.success),
            "results": [r.to_dict() for r in self.results],
        }

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {self.timestamp}")
        print(f"Duration: {self.duration_seconds:.2f}s")
        print(f"Benchmarks: {len(self.results)} total, {sum(1 for r in self.results if r.success)} passed")
        print("-" * 60)

        for result in self.results:
            status = "✓" if result.success else "✗"
            print(f"  [{status}] {result.name}")
            print(f"      Duration: {result.duration_seconds:.4f}s")
            print(f"      Throughput: {result.throughput:.2f} iter/s")
            if result.metrics:
                for k, v in result.metrics.items():
                    print(f"      {k}: {v:.4f}")
            if result.error:
                print(f"      Error: {result.error}")

        print("=" * 60)


def run_benchmark(
    name: str,
    suite: str,
    func: Callable[[], Dict[str, float]],
    iterations: int = 100,
    warmup: int = 5,
) -> BenchmarkResult:
    """
    Run a benchmark function multiple times.

    Args:
        name: Benchmark name
        suite: Suite name (sft, rlvr, etc.)
        func: Function to benchmark (returns metrics dict)
        iterations: Number of iterations
        warmup: Number of warmup iterations

    Returns:
        BenchmarkResult
    """
    try:
        # Warmup
        for _ in range(warmup):
            func()

        # Timed run
        start = time.perf_counter()
        all_metrics: List[Dict[str, float]] = []

        for _ in range(iterations):
            metrics = func()
            all_metrics.append(metrics)

        duration = time.perf_counter() - start

        # Aggregate metrics
        aggregated: Dict[str, float] = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics]
                aggregated[f"{key}_mean"] = statistics.mean(values)
                if len(values) > 1:
                    aggregated[f"{key}_std"] = statistics.stdev(values)

        return BenchmarkResult(
            name=name,
            suite=suite,
            duration_seconds=duration,
            iterations=iterations,
            throughput=iterations / duration,
            metrics=aggregated,
        )

    except Exception as e:
        return BenchmarkResult(
            name=name,
            suite=suite,
            duration_seconds=0,
            iterations=0,
            throughput=0,
            success=False,
            error=str(e),
        )


# ==============================================================================
# SFT Benchmarks
# ==============================================================================


def benchmark_sft_forward_backward() -> Dict[str, float]:
    """Benchmark SFT forward-backward pass."""
    from scripts.baseline_sft_smoke import MinimalTrainingClient, create_synthetic_batch

    client = MinimalTrainingClient()
    batch = create_synthetic_batch(batch_size=4, seq_len=128)

    start = time.perf_counter()
    result = client.forward_backward(batch)
    duration = time.perf_counter() - start

    return {
        "loss": result["loss"],
        "grad_norm": result["grad_norm"],
        "duration": duration,
    }


def benchmark_sft_optim_step() -> Dict[str, float]:
    """Benchmark SFT optimizer step."""
    from scripts.baseline_sft_smoke import MinimalTrainingClient, create_synthetic_batch

    client = MinimalTrainingClient()
    batch = create_synthetic_batch(batch_size=4, seq_len=128)
    client.forward_backward(batch)

    start = time.perf_counter()
    client.optim_step()
    duration = time.perf_counter() - start

    return {"step": float(client.step), "duration": duration}


def benchmark_sft_full_step() -> Dict[str, float]:
    """Benchmark full SFT training step."""
    from scripts.baseline_sft_smoke import MinimalTrainingClient, create_synthetic_batch

    client = MinimalTrainingClient()
    batch = create_synthetic_batch(batch_size=4, seq_len=128)

    start = time.perf_counter()
    fb_result = client.forward_backward(batch)
    client.optim_step()
    duration = time.perf_counter() - start

    return {
        "loss": fb_result["loss"],
        "duration": duration,
    }


def benchmark_sft_checkpoint() -> Dict[str, float]:
    """Benchmark SFT checkpoint save/load."""
    from scripts.baseline_sft_smoke import MinimalTrainingClient, create_synthetic_batch

    client = MinimalTrainingClient()
    batch = create_synthetic_batch()

    # Train a bit
    for _ in range(3):
        client.forward_backward(batch)
        client.optim_step()

    # Benchmark save
    start = time.perf_counter()
    save_result = client.save_state()
    save_duration = time.perf_counter() - start

    # Benchmark load
    start = time.perf_counter()
    client.load_state(save_result["artifact_id"], save_result["_state_bytes"])
    load_duration = time.perf_counter() - start

    return {
        "save_duration": save_duration,
        "load_duration": load_duration,
        "checkpoint_size_bytes": float(save_result["size_bytes"]),
    }


# ==============================================================================
# Loss Function Benchmarks
# ==============================================================================


def benchmark_loss_resolution() -> Dict[str, float]:
    """Benchmark loss function resolution."""
    from tensafe.training.losses import resolve_loss

    start = time.perf_counter()
    loss_fn = resolve_loss("token_ce")
    duration = time.perf_counter() - start

    return {"duration": duration}


def benchmark_loss_computation() -> Dict[str, float]:
    """Benchmark loss computation (using mock when PyTorch not available)."""
    from tensafe.training.losses.builtin import MockLossFunctions

    # Use mock loss function which doesn't require PyTorch
    loss_fn = MockLossFunctions.token_cross_entropy

    # Mock outputs and batch
    outputs = {"logits": [[0.1] * 1000] * 8}
    batch = {"input_ids": [[1] * 128] * 8, "labels": [[1] * 128] * 8}

    start = time.perf_counter()
    result = loss_fn(outputs, batch)
    duration = time.perf_counter() - start

    return {
        "loss": result["loss"],
        "duration": duration,
    }


# ==============================================================================
# RLVR Benchmarks
# ==============================================================================


def benchmark_trajectory_creation() -> Dict[str, float]:
    """Benchmark trajectory creation."""
    from tensafe.rlvr.rollout import Trajectory

    start = time.perf_counter()
    traj = Trajectory(
        prompt="Test prompt",
        prompt_tokens=list(range(10)),
        response="Test response",
        response_tokens=list(range(20)),
        logprobs=[-0.5] * 20,
        attention_mask=[1] * 30,
        reward=1.0,
    )
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "num_tokens": float(len(traj.full_tokens)),
    }


def benchmark_batch_creation() -> Dict[str, float]:
    """Benchmark trajectory batch creation."""
    from tensafe.rlvr.rollout import Trajectory, TrajectoryBatch

    trajectories = [
        Trajectory(
            prompt=f"Prompt {i}",
            prompt_tokens=list(range(10)),
            response=f"Response {i}",
            response_tokens=list(range(20)),
            logprobs=[-0.5] * 20,
            attention_mask=[1] * 30,
            reward=float(i),
        )
        for i in range(32)
    ]

    start = time.perf_counter()
    batch = TrajectoryBatch(trajectories=trajectories)
    _ = batch.mean_reward
    _ = batch.std_reward
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "batch_size": float(len(batch)),
    }


def benchmark_advantage_computation() -> Dict[str, float]:
    """Benchmark advantage computation."""
    from tensafe.rlvr.rollout import Trajectory, TrajectoryBatch

    trajectories = [
        Trajectory(
            prompt=f"Prompt {i}",
            prompt_tokens=[1],
            response=f"Response {i}",
            response_tokens=[2],
            logprobs=[-0.5],
            attention_mask=[1, 1],
            reward=float(i) - 16,
        )
        for i in range(32)
    ]
    batch = TrajectoryBatch(trajectories=trajectories)

    start = time.perf_counter()
    batch.compute_advantages(normalize=True)
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "mean_advantage": sum(t.advantage for t in batch) / len(batch),
    }


def benchmark_reinforce_update() -> Dict[str, float]:
    """Benchmark REINFORCE update."""
    from tensafe.rlvr.algorithms.reinforce import REINFORCE, REINFORCEConfig
    from tensafe.rlvr.rollout import Trajectory, TrajectoryBatch

    reinforce = REINFORCE(REINFORCEConfig(normalize_advantages=True))

    trajectories = [
        Trajectory(
            prompt=f"Prompt {i}",
            prompt_tokens=[1],
            response=f"Response {i}",
            response_tokens=[2, 3],
            logprobs=[-0.5, -0.3],
            attention_mask=[1, 1, 1],
            reward=float(i) - 4,
        )
        for i in range(8)
    ]
    batch = TrajectoryBatch(trajectories=trajectories)

    start = time.perf_counter()
    result = reinforce.update(batch, None)
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "policy_loss": result.policy_loss,
    }


def benchmark_ppo_update() -> Dict[str, float]:
    """Benchmark PPO update."""
    from tensafe.rlvr.algorithms.ppo import PPO, PPOConfig
    from tensafe.rlvr.rollout import Trajectory, TrajectoryBatch

    ppo = PPO(PPOConfig(ppo_epochs=2))

    trajectories = [
        Trajectory(
            prompt=f"Prompt {i}",
            prompt_tokens=[1],
            response=f"Response {i}",
            response_tokens=[2, 3],
            logprobs=[-0.5, -0.3],
            attention_mask=[1, 1, 1],
            reward=float(i) - 4,
        )
        for i in range(8)
    ]
    batch = TrajectoryBatch(trajectories=trajectories)

    start = time.perf_counter()
    result = ppo.update(batch, None)
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "policy_loss": result.policy_loss,
    }


def benchmark_reward_computation() -> Dict[str, float]:
    """Benchmark reward function computation."""
    from tensafe.rlvr.reward import resolve_reward

    reward_fn = resolve_reward("keyword_contains", keywords=["target", "goal"])

    start = time.perf_counter()
    for i in range(100):
        _ = reward_fn(f"Prompt {i}", f"Response with target word {i}")
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "calls_per_second": 100 / duration,
    }


def benchmark_rollout_generation() -> Dict[str, float]:
    """Benchmark rollout generation."""
    from tensafe.rlvr.rollout import MockRolloutSampler

    sampler = MockRolloutSampler(max_new_tokens=32, seed=42)
    prompts = [f"Prompt {i}" for i in range(8)]

    start = time.perf_counter()
    batch = sampler.generate_trajectories(prompts)
    duration = time.perf_counter() - start

    return {
        "duration": duration,
        "trajectories_per_second": len(batch) / duration,
    }


# ==============================================================================
# Benchmark Suites
# ==============================================================================


def get_sft_benchmarks() -> List[tuple]:
    """Get SFT benchmark functions."""
    return [
        ("sft_forward_backward", benchmark_sft_forward_backward, 100),
        ("sft_optim_step", benchmark_sft_optim_step, 100),
        ("sft_full_step", benchmark_sft_full_step, 100),
        ("sft_checkpoint", benchmark_sft_checkpoint, 20),
        ("loss_resolution", benchmark_loss_resolution, 100),
        ("loss_computation", benchmark_loss_computation, 100),
    ]


def get_rlvr_benchmarks() -> List[tuple]:
    """Get RLVR benchmark functions."""
    return [
        ("trajectory_creation", benchmark_trajectory_creation, 1000),
        ("batch_creation", benchmark_batch_creation, 100),
        ("advantage_computation", benchmark_advantage_computation, 100),
        ("reinforce_update", benchmark_reinforce_update, 50),
        ("ppo_update", benchmark_ppo_update, 50),
        ("reward_computation", benchmark_reward_computation, 50),
        ("rollout_generation", benchmark_rollout_generation, 50),
    ]


def run_suite(
    suite_name: str,
    benchmarks: List[tuple],
    report: BenchmarkReport,
    verbose: bool = True,
) -> None:
    """Run a benchmark suite."""
    if verbose:
        print(f"\nRunning {suite_name} benchmarks...")
        print("-" * 40)

    for name, func, iterations in benchmarks:
        if verbose:
            print(f"  {name}... ", end="", flush=True)

        result = run_benchmark(
            name=name,
            suite=suite_name,
            func=func,
            iterations=iterations,
            warmup=5,
        )
        report.add_result(result)

        if verbose:
            if result.success:
                print(f"done ({result.throughput:.1f} iter/s)")
            else:
                print(f"FAILED: {result.error}")


def main():
    parser = argparse.ArgumentParser(description="TenSafe Benchmark Harness")
    parser.add_argument(
        "--suite",
        choices=["all", "sft", "rlvr"],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (less output)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TenSafe Benchmark Harness")
    print("=" * 60)

    report = BenchmarkReport()
    start_time = time.perf_counter()
    verbose = not args.quiet

    if args.suite in ("all", "sft"):
        run_suite("sft", get_sft_benchmarks(), report, verbose)

    if args.suite in ("all", "rlvr"):
        run_suite("rlvr", get_rlvr_benchmarks(), report, verbose)

    report.duration_seconds = time.perf_counter() - start_time
    report.print_summary()

    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Return exit code based on success
    all_passed = all(r.success for r in report.results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
