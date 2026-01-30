#!/usr/bin/env python3
"""
N2HE Benchmark Runner

Runs comprehensive benchmarks for the N2HE integration and generates reports.

Usage:
    python scripts/n2he/run_benchmark.py [--mode smoke|full] [--output-dir DIR]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensorguard.n2he.benchmark import (
    N2HEBenchmark,
    run_quick_benchmark,
    run_full_benchmark,
    generate_benchmark_report,
    generate_compliance_evidence,
)
from tensorguard.n2he._native import is_native_available, get_native_version
from tensorguard.n2he.core import HESchemeParams


def main():
    parser = argparse.ArgumentParser(description="N2HE Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["smoke", "quick", "full"],
        default="quick",
        help="Benchmark mode (default: quick)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/n2he",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print header
    print("=" * 70)
    print("N2HE Benchmark Suite")
    print("=" * 70)
    print()

    # Check native library status
    native_available = is_native_available()
    native_version = get_native_version()

    print(f"Native N2HE Library: {'Available' if native_available else 'Not Available (using simulation)'}")
    if native_version:
        print(f"Native Version: {native_version}")
    print()

    # Run benchmarks
    print(f"Running {args.mode} benchmark...")
    print()

    if args.mode == "smoke":
        benchmark = N2HEBenchmark(warmup_iterations=1, default_iterations=3)
        suite = benchmark.run_full_suite(name="N2HE Smoke Benchmark", iterations=3)
    elif args.mode == "quick":
        suite = run_quick_benchmark()
    else:
        suite = run_full_benchmark()

    # Generate report
    report = generate_benchmark_report(suite)
    print(report)
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save text report
    report_file = output_dir / f"benchmark_{args.mode}_{timestamp}.txt"
    report_file.write_text(report)
    print(f"Report saved to: {report_file}")

    # Save JSON results
    json_file = output_dir / f"benchmark_{args.mode}_{timestamp}.json"
    json_file.write_text(json.dumps(suite.to_dict(), indent=2, default=str))
    print(f"JSON results saved to: {json_file}")

    # Generate compliance evidence
    evidence = generate_compliance_evidence(suite)
    evidence_file = output_dir / f"evidence_{args.mode}_{timestamp}.json"
    evidence_file.write_text(json.dumps(evidence.to_dict(), indent=2, default=str))
    print(f"Compliance evidence saved to: {evidence_file}")

    print()
    print("=" * 70)
    print("Benchmark Complete!")
    print("=" * 70)

    # Output JSON if requested
    if args.json:
        print()
        print(json.dumps(suite.to_dict(), indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
