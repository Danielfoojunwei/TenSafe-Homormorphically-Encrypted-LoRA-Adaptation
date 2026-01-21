#!/usr/bin/env python3
"""
TensorGuardFlow Benchmark Runner

Main entry point for running comprehensive performance benchmarks.
Orchestrates HTTP, telemetry, and resource benchmarks.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    BenchmarkConfig,
    LoadLevel,
    LOAD_PRESETS,
    get_config,
    get_hardware_info,
)
from .http_bench import run_http_benchmark
from .telemetry_bench import run_telemetry_benchmark
from .resource_monitor import ResourceMonitor, run_resource_benchmark


class BenchmarkRunner:
    """
    Orchestrates all benchmark suites and produces consolidated results.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: Dict[str, Any] = {
            "metadata": {},
            "benchmarks": {},
            "summary": {},
        }
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_metadata(self) -> Dict[str, Any]:
        """Collect benchmark metadata."""
        hardware = get_hardware_info()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "tensorguard_version": "3.0.0",  # TODO: get from package
            "benchmark_version": "1.0.0",
            "config": {
                "base_url": self.config.base_url,
                "load_level": {
                    "concurrent_users": self.config.load.concurrent_users,
                    "requests_per_second": self.config.load.requests_per_second,
                    "duration_seconds": self.config.load.duration_seconds,
                    "batch_size": self.config.load.batch_size,
                },
                "thresholds": {
                    "p95_latency_ms": self.config.thresholds.p95_latency_ms,
                    "min_rps": self.config.thresholds.min_rps,
                    "max_error_rate": self.config.thresholds.max_error_rate,
                },
            },
            "environment": {
                "cpu": hardware["cpu_info"],
                "memory_gb": hardware["memory_gb"],
                "os": hardware["os_info"],
                "python": hardware["python_version"],
            },
        }

    def run_http_benchmarks(self) -> Dict[str, Any]:
        """Run HTTP API benchmarks with resource monitoring."""
        print("\n" + "=" * 70)
        print("HTTP API BENCHMARKS")
        print("=" * 70)

        monitor = ResourceMonitor(sample_interval=0.5)
        monitor.start()

        try:
            http_results = run_http_benchmark(self.config)
            resource_metrics = monitor.stop()

            http_results["resources"] = resource_metrics.to_dict()
            return http_results

        except Exception as e:
            monitor.stop()
            print(f"HTTP benchmark error: {e}")
            return {"error": str(e)}

    def run_telemetry_benchmarks(self) -> Dict[str, Any]:
        """Run telemetry ingestion benchmarks with resource monitoring."""
        print("\n" + "=" * 70)
        print("TELEMETRY INGESTION BENCHMARKS")
        print("=" * 70)

        monitor = ResourceMonitor(sample_interval=0.5)
        monitor.start()

        try:
            telemetry_results = run_telemetry_benchmark(self.config)
            resource_metrics = monitor.stop()

            telemetry_results["resources"] = resource_metrics.to_dict()
            return telemetry_results

        except Exception as e:
            monitor.stop()
            print(f"Telemetry benchmark error: {e}")
            return {"error": str(e)}

    def run_resource_baseline(self) -> Dict[str, Any]:
        """Run resource baseline measurement."""
        print("\n" + "=" * 70)
        print("RESOURCE BASELINE")
        print("=" * 70)

        return run_resource_benchmark(duration_seconds=10.0)

    def _calculate_summary(self) -> Dict[str, Any]:
        """Calculate overall benchmark summary."""
        summary = {
            "passed": True,
            "checks": [],
        }

        # Check HTTP benchmarks
        if "http_api" in self.results["benchmarks"]:
            http = self.results["benchmarks"]["http_api"]
            if "summary" in http:
                avg_p95 = http["summary"].get("avg_p95_latency_ms", 0)
                avg_rps = http["summary"].get("avg_rps", 0)
                avg_error = http["summary"].get("avg_error_rate", 0)

                # Check latency threshold
                latency_check = avg_p95 <= self.config.thresholds.p95_latency_ms
                summary["checks"].append({
                    "name": "http_p95_latency",
                    "passed": latency_check,
                    "value": avg_p95,
                    "threshold": self.config.thresholds.p95_latency_ms,
                })
                if not latency_check:
                    summary["passed"] = False

                # Check throughput threshold
                rps_check = avg_rps >= self.config.thresholds.min_rps
                summary["checks"].append({
                    "name": "http_throughput",
                    "passed": rps_check,
                    "value": avg_rps,
                    "threshold": self.config.thresholds.min_rps,
                })
                if not rps_check:
                    summary["passed"] = False

                # Check error rate
                error_check = avg_error <= self.config.thresholds.max_error_rate
                summary["checks"].append({
                    "name": "http_error_rate",
                    "passed": error_check,
                    "value": avg_error,
                    "threshold": self.config.thresholds.max_error_rate,
                })
                if not error_check:
                    summary["passed"] = False

        # Check telemetry benchmarks
        if "telemetry_ingest" in self.results["benchmarks"]:
            telemetry = self.results["benchmarks"]["telemetry_ingest"]
            if "metrics" in telemetry:
                metrics = telemetry["metrics"]
                events_per_sec = metrics.get("throughput", {}).get("events_per_second", 0)
                acceptance_rate = metrics.get("acceptance_rate", 0)
                p95_latency = metrics.get("latency", {}).get("p95_ms", 0)

                # Check events throughput
                events_check = events_per_sec >= self.config.thresholds.min_events_per_second
                summary["checks"].append({
                    "name": "telemetry_events_per_second",
                    "passed": events_check,
                    "value": events_per_sec,
                    "threshold": self.config.thresholds.min_events_per_second,
                })
                if not events_check:
                    summary["passed"] = False

                # Check acceptance rate (should be high)
                accept_check = acceptance_rate >= 0.95
                summary["checks"].append({
                    "name": "telemetry_acceptance_rate",
                    "passed": accept_check,
                    "value": acceptance_rate,
                    "threshold": 0.95,
                })
                if not accept_check:
                    summary["passed"] = False

        return summary

    def run_all(self) -> Dict[str, Any]:
        """Run all configured benchmarks."""
        print("\n" + "=" * 70)
        print("TENSORGUARDFLOW BENCHMARK SUITE")
        print("=" * 70)
        print(f"Server: {self.config.base_url}")
        print(f"Output: {self.output_dir}")
        print()

        start_time = time.time()

        # Collect metadata
        self.results["metadata"] = self._collect_metadata()

        # Run baseline resource measurement
        if self.config.run_resource_bench:
            self.results["benchmarks"]["resource_baseline"] = self.run_resource_baseline()

        # Run HTTP benchmarks
        if self.config.run_api_bench:
            self.results["benchmarks"]["http_api"] = self.run_http_benchmarks()

        # Run telemetry benchmarks
        if self.config.run_ingest_bench:
            self.results["benchmarks"]["telemetry_ingest"] = self.run_telemetry_benchmarks()

        # Calculate summary
        self.results["summary"] = self._calculate_summary()
        self.results["metadata"]["total_duration_seconds"] = time.time() - start_time

        # Save results
        self._save_results()

        # Print summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")

        # Also save as latest
        latest_path = self.output_dir / "benchmark_results_latest.json"
        with open(latest_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        summary = self.results.get("summary", {})
        passed = summary.get("passed", False)

        status = "PASSED" if passed else "FAILED"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        print(f"\nOverall Status: {color}{status}{reset}")
        print()

        for check in summary.get("checks", []):
            check_status = "PASS" if check["passed"] else "FAIL"
            check_color = "\033[92m" if check["passed"] else "\033[91m"
            print(f"  [{check_color}{check_status}{reset}] {check['name']}: "
                  f"{check['value']:.2f} (threshold: {check['threshold']:.2f})")

        duration = self.results.get("metadata", {}).get("total_duration_seconds", 0)
        print(f"\nTotal duration: {duration:.1f}s")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TensorGuardFlow Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (moderate load)
  python -m benchmarks.runner

  # Run with light load for quick test
  python -m benchmarks.runner --load light

  # Run against specific server
  python -m benchmarks.runner --url http://production:8000

  # Run only HTTP benchmarks
  python -m benchmarks.runner --http-only

  # Run only telemetry benchmarks
  python -m benchmarks.runner --telemetry-only
        """
    )

    parser.add_argument(
        "--url",
        default=os.getenv("TG_BENCH_URL", "http://localhost:8000"),
        help="Base URL for TensorGuardFlow server"
    )

    parser.add_argument(
        "--load",
        choices=["light", "moderate", "heavy", "stress"],
        default="moderate",
        help="Load level preset (default: moderate)"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Benchmark duration in seconds"
    )

    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help="Number of concurrent users (overrides preset)"
    )

    parser.add_argument(
        "--output",
        default="artifacts/benchmarks",
        help="Output directory for results"
    )

    parser.add_argument(
        "--http-only",
        action="store_true",
        help="Run only HTTP API benchmarks"
    )

    parser.add_argument(
        "--telemetry-only",
        action="store_true",
        help="Run only telemetry ingestion benchmarks"
    )

    parser.add_argument(
        "--no-resources",
        action="store_true",
        help="Disable resource monitoring"
    )

    args = parser.parse_args()

    # Build configuration
    load_level = LoadLevel(args.load)
    config = get_config(load_level)

    config.base_url = args.url
    config.output_dir = args.output
    config.load.duration_seconds = args.duration

    if args.concurrent:
        config.load.concurrent_users = args.concurrent

    if args.http_only:
        config.run_ingest_bench = False
        config.run_resource_bench = False

    if args.telemetry_only:
        config.run_api_bench = False
        config.run_resource_bench = False

    if args.no_resources:
        config.run_resource_bench = False

    # Run benchmarks
    runner = BenchmarkRunner(config)
    results = runner.run_all()

    # Exit with appropriate code
    sys.exit(0 if results.get("summary", {}).get("passed", False) else 1)


if __name__ == "__main__":
    main()
