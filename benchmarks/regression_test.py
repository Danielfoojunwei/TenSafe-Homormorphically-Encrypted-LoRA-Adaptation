"""
Performance Regression Test

Integration test that fails if performance regresses beyond thresholds.
Suitable for CI/CD pipelines to catch performance degradation early.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .config import BenchmarkConfig, ThresholdConfig, LoadLevel, get_config


@dataclass
class RegressionResult:
    """Result of a regression check."""
    name: str
    passed: bool
    current_value: float
    threshold: float
    baseline_value: Optional[float] = None
    deviation_percent: Optional[float] = None
    message: str = ""


class RegressionTest:
    """
    Performance regression testing framework.

    Compares current benchmark results against:
    1. Fixed thresholds (always enforced)
    2. Historical baseline (if available)
    """

    def __init__(
        self,
        results_path: str,
        baseline_path: Optional[str] = None,
        thresholds: Optional[ThresholdConfig] = None,
    ):
        self.results_path = Path(results_path)
        self.baseline_path = Path(baseline_path) if baseline_path else None
        self.thresholds = thresholds or ThresholdConfig()

        self.results: Dict[str, Any] = {}
        self.baseline: Optional[Dict[str, Any]] = None
        self.checks: List[RegressionResult] = []

        self._load_data()

    def _load_data(self):
        """Load results and baseline data."""
        if self.results_path.exists():
            with open(self.results_path) as f:
                self.results = json.load(f)
        else:
            raise FileNotFoundError(f"Results not found: {self.results_path}")

        if self.baseline_path and self.baseline_path.exists():
            with open(self.baseline_path) as f:
                self.baseline = json.load(f)

    def check_http_latency(self) -> List[RegressionResult]:
        """Check HTTP API latency thresholds."""
        checks = []

        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if not http_results:
            return checks

        # Check overall average p95 latency
        summary = http_results.get("summary", {})
        avg_p95 = summary.get("avg_p95_latency_ms", 0)

        checks.append(RegressionResult(
            name="http_avg_p95_latency",
            passed=avg_p95 <= self.thresholds.p95_latency_ms,
            current_value=avg_p95,
            threshold=self.thresholds.p95_latency_ms,
            message=f"Average HTTP p95 latency: {avg_p95:.1f}ms (threshold: {self.thresholds.p95_latency_ms}ms)"
        ))

        # Check individual endpoints
        endpoints = http_results.get("endpoints", {})
        for path, metrics in endpoints.items():
            latency = metrics.get("latency", {})
            p95 = latency.get("p95_ms", 0)
            p99 = latency.get("p99_ms", 0)

            # p95 threshold check
            checks.append(RegressionResult(
                name=f"http_p95_{path.replace('/', '_')}",
                passed=p95 <= self.thresholds.p95_latency_ms,
                current_value=p95,
                threshold=self.thresholds.p95_latency_ms,
                message=f"{path} p95 latency: {p95:.1f}ms"
            ))

            # p99 threshold check
            checks.append(RegressionResult(
                name=f"http_p99_{path.replace('/', '_')}",
                passed=p99 <= self.thresholds.p99_latency_ms,
                current_value=p99,
                threshold=self.thresholds.p99_latency_ms,
                message=f"{path} p99 latency: {p99:.1f}ms"
            ))

        return checks

    def check_http_throughput(self) -> List[RegressionResult]:
        """Check HTTP API throughput thresholds."""
        checks = []

        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if not http_results:
            return checks

        summary = http_results.get("summary", {})
        avg_rps = summary.get("avg_rps", 0)

        checks.append(RegressionResult(
            name="http_avg_throughput",
            passed=avg_rps >= self.thresholds.min_rps,
            current_value=avg_rps,
            threshold=self.thresholds.min_rps,
            message=f"Average HTTP throughput: {avg_rps:.0f} RPS (minimum: {self.thresholds.min_rps})"
        ))

        return checks

    def check_http_errors(self) -> List[RegressionResult]:
        """Check HTTP API error rates."""
        checks = []

        http_results = self.results.get("benchmarks", {}).get("http_api", {})
        if not http_results:
            return checks

        summary = http_results.get("summary", {})
        avg_error = summary.get("avg_error_rate", 0)

        checks.append(RegressionResult(
            name="http_error_rate",
            passed=avg_error <= self.thresholds.max_error_rate,
            current_value=avg_error,
            threshold=self.thresholds.max_error_rate,
            message=f"HTTP error rate: {avg_error:.2%} (maximum: {self.thresholds.max_error_rate:.2%})"
        ))

        return checks

    def check_telemetry_throughput(self) -> List[RegressionResult]:
        """Check telemetry ingestion throughput."""
        checks = []

        telemetry_results = self.results.get("benchmarks", {}).get("telemetry_ingest", {})
        if not telemetry_results:
            return checks

        metrics = telemetry_results.get("metrics", {})
        events_per_sec = metrics.get("throughput", {}).get("events_per_second", 0)

        checks.append(RegressionResult(
            name="telemetry_events_per_second",
            passed=events_per_sec >= self.thresholds.min_events_per_second,
            current_value=events_per_sec,
            threshold=self.thresholds.min_events_per_second,
            message=f"Telemetry throughput: {events_per_sec:.0f} events/s (minimum: {self.thresholds.min_events_per_second})"
        ))

        return checks

    def check_telemetry_latency(self) -> List[RegressionResult]:
        """Check telemetry ingestion latency."""
        checks = []

        telemetry_results = self.results.get("benchmarks", {}).get("telemetry_ingest", {})
        if not telemetry_results:
            return checks

        metrics = telemetry_results.get("metrics", {})
        latency = metrics.get("latency", {})
        p95 = latency.get("p95_ms", 0)

        checks.append(RegressionResult(
            name="telemetry_p95_latency",
            passed=p95 <= self.thresholds.p95_latency_ms,
            current_value=p95,
            threshold=self.thresholds.p95_latency_ms,
            message=f"Telemetry p95 latency: {p95:.1f}ms (threshold: {self.thresholds.p95_latency_ms}ms)"
        ))

        return checks

    def check_resource_usage(self) -> List[RegressionResult]:
        """Check resource utilization."""
        checks = []

        # Get resource metrics from any benchmark that has them
        resource_metrics = None
        for bench_name in ["http_api", "telemetry_ingest"]:
            bench = self.results.get("benchmarks", {}).get(bench_name, {})
            if "resources" in bench:
                resource_metrics = bench["resources"]
                break

        if not resource_metrics:
            return checks

        cpu = resource_metrics.get("cpu", {})
        memory = resource_metrics.get("memory", {})

        cpu_max = cpu.get("max_percent", 0)
        checks.append(RegressionResult(
            name="cpu_utilization",
            passed=cpu_max <= self.thresholds.max_cpu_percent,
            current_value=cpu_max,
            threshold=self.thresholds.max_cpu_percent,
            message=f"Peak CPU: {cpu_max:.0f}% (maximum: {self.thresholds.max_cpu_percent}%)"
        ))

        memory_max = memory.get("max_used_mb", 0)
        checks.append(RegressionResult(
            name="memory_utilization",
            passed=memory_max <= self.thresholds.max_memory_mb,
            current_value=memory_max,
            threshold=self.thresholds.max_memory_mb,
            message=f"Peak Memory: {memory_max:.0f}MB (maximum: {self.thresholds.max_memory_mb}MB)"
        ))

        return checks

    def check_baseline_regression(self) -> List[RegressionResult]:
        """Check for regression against baseline (if available)."""
        checks = []

        if not self.baseline:
            return checks

        # Compare HTTP throughput
        current_http = self.results.get("benchmarks", {}).get("http_api", {}).get("summary", {})
        baseline_http = self.baseline.get("benchmarks", {}).get("http_api", {}).get("summary", {})

        if current_http and baseline_http:
            current_rps = current_http.get("avg_rps", 0)
            baseline_rps = baseline_http.get("avg_rps", 1)

            if baseline_rps > 0:
                deviation = ((current_rps - baseline_rps) / baseline_rps) * 100
                # Allow up to 20% regression
                passed = deviation >= -20

                checks.append(RegressionResult(
                    name="http_throughput_regression",
                    passed=passed,
                    current_value=current_rps,
                    threshold=baseline_rps * 0.8,
                    baseline_value=baseline_rps,
                    deviation_percent=deviation,
                    message=f"HTTP throughput: {current_rps:.0f} RPS ({deviation:+.1f}% vs baseline)"
                ))

        # Compare telemetry throughput
        current_tel = self.results.get("benchmarks", {}).get("telemetry_ingest", {}).get("metrics", {})
        baseline_tel = self.baseline.get("benchmarks", {}).get("telemetry_ingest", {}).get("metrics", {})

        if current_tel and baseline_tel:
            current_events = current_tel.get("throughput", {}).get("events_per_second", 0)
            baseline_events = baseline_tel.get("throughput", {}).get("events_per_second", 1)

            if baseline_events > 0:
                deviation = ((current_events - baseline_events) / baseline_events) * 100
                passed = deviation >= -20

                checks.append(RegressionResult(
                    name="telemetry_throughput_regression",
                    passed=passed,
                    current_value=current_events,
                    threshold=baseline_events * 0.8,
                    baseline_value=baseline_events,
                    deviation_percent=deviation,
                    message=f"Telemetry throughput: {current_events:.0f} events/s ({deviation:+.1f}% vs baseline)"
                ))

        return checks

    def run_all_checks(self) -> bool:
        """Run all regression checks and return overall pass/fail."""
        self.checks = []

        # Run all check categories
        self.checks.extend(self.check_http_latency())
        self.checks.extend(self.check_http_throughput())
        self.checks.extend(self.check_http_errors())
        self.checks.extend(self.check_telemetry_throughput())
        self.checks.extend(self.check_telemetry_latency())
        self.checks.extend(self.check_resource_usage())
        self.checks.extend(self.check_baseline_regression())

        # Calculate overall result
        all_passed = all(check.passed for check in self.checks)

        return all_passed

    def print_report(self):
        """Print regression test report."""
        print("\n" + "=" * 70)
        print("PERFORMANCE REGRESSION TEST RESULTS")
        print("=" * 70)

        passed_checks = [c for c in self.checks if c.passed]
        failed_checks = [c for c in self.checks if not c.passed]

        print(f"\nTotal checks: {len(self.checks)}")
        print(f"Passed: {len(passed_checks)}")
        print(f"Failed: {len(failed_checks)}")

        if failed_checks:
            print("\n--- FAILED CHECKS ---")
            for check in failed_checks:
                print(f"\n[FAIL] {check.name}")
                print(f"       {check.message}")
                print(f"       Current: {check.current_value:.2f}, Threshold: {check.threshold:.2f}")
                if check.baseline_value is not None:
                    print(f"       Baseline: {check.baseline_value:.2f}, Deviation: {check.deviation_percent:+.1f}%")

        if passed_checks:
            print("\n--- PASSED CHECKS ---")
            for check in passed_checks:
                print(f"[PASS] {check.name}: {check.message}")

        overall = "PASSED" if not failed_checks else "FAILED"
        color = "\033[92m" if not failed_checks else "\033[91m"
        reset = "\033[0m"

        print(f"\n{color}Overall Result: {overall}{reset}")
        print("=" * 70)

    def to_junit_xml(self) -> str:
        """Generate JUnit XML format for CI integration."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<testsuite name="PerformanceRegression" tests="{len(self.checks)}" '
                    f'failures="{len([c for c in self.checks if not c.passed])}">')

        for check in self.checks:
            lines.append(f'  <testcase name="{check.name}" classname="performance">')
            if not check.passed:
                lines.append(f'    <failure message="{check.message}">')
                lines.append(f'      Current: {check.current_value}, Threshold: {check.threshold}')
                lines.append('    </failure>')
            lines.append('  </testcase>')

        lines.append('</testsuite>')
        return '\n'.join(lines)


def run_regression_test(
    results_path: str,
    baseline_path: Optional[str] = None,
    output_junit: Optional[str] = None,
) -> bool:
    """
    Run performance regression tests.

    Args:
        results_path: Path to benchmark results JSON
        baseline_path: Optional path to baseline results for comparison
        output_junit: Optional path to write JUnit XML report

    Returns:
        True if all checks pass, False otherwise
    """
    test = RegressionTest(results_path, baseline_path)
    passed = test.run_all_checks()
    test.print_report()

    if output_junit:
        junit_xml = test.to_junit_xml()
        with open(output_junit, 'w') as f:
            f.write(junit_xml)
        print(f"\nJUnit report written to: {output_junit}")

    return passed


# Entry point for pytest
class TestPerformanceRegression:
    """Pytest-compatible test class for CI integration."""

    @staticmethod
    def test_performance_thresholds():
        """Test that performance meets thresholds."""
        results_path = os.getenv(
            "TG_BENCH_RESULTS",
            "artifacts/benchmarks/benchmark_results_latest.json"
        )

        if not Path(results_path).exists():
            import pytest
            pytest.skip(f"Benchmark results not found: {results_path}")

        baseline_path = os.getenv("TG_BENCH_BASELINE")

        test = RegressionTest(results_path, baseline_path)
        passed = test.run_all_checks()

        # Collect failures for assertion message
        failures = [c for c in test.checks if not c.passed]
        failure_messages = [f"{c.name}: {c.message}" for c in failures]

        assert passed, f"Performance regression detected:\n" + "\n".join(failure_messages)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance Regression Test")
    parser.add_argument("results", help="Path to benchmark results JSON")
    parser.add_argument("--baseline", help="Path to baseline results for comparison")
    parser.add_argument("--junit", help="Output path for JUnit XML report")

    args = parser.parse_args()

    passed = run_regression_test(args.results, args.baseline, args.junit)
    sys.exit(0 if passed else 1)
