#!/usr/bin/env python3
"""
Test Matrix Runner for Privacy Tinker

Runs pytest across multiple privacy mode configurations and collects results.

Privacy Modes:
- off: Baseline (no privacy features)
- tdx_base_only: TDX enclave only (mock)
- tdx_plus_moai_lora: TDX + MOAI encrypted LoRA (mock)
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestResult:
    """Result from a single test run."""
    mode: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_seconds: float
    warnings: int
    exit_code: int
    log_file: str


@dataclass
class MatrixResult:
    """Result from the full test matrix."""
    timestamp: str
    git_sha: str
    git_branch: str
    modes_tested: List[str]
    results: Dict[str, TestResult]
    all_passed: bool
    total_tests: int
    total_failures: int


# Privacy mode configurations
PRIVACY_MODES = {
    "off": {
        "TINKER_PRIVACY_MODE": "off",
        "description": "Baseline (no privacy features)",
    },
    "tdx_base_only": {
        "TINKER_PRIVACY_MODE": "tdx_base_only",
        "TINKER_TDX_MODE": "mock",
        "description": "TDX enclave only (mock)",
    },
    "tdx_plus_moai_lora": {
        "TINKER_PRIVACY_MODE": "tdx_plus_moai_lora",
        "TINKER_TDX_MODE": "mock",
        "TINKER_MOAI_MODE": "mock",
        "TINKER_STRICT_PRIVACY": "1",
        "description": "TDX + MOAI encrypted LoRA (mock, strict)",
    },
}


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


def run_tests_for_mode(
    mode_name: str,
    env_vars: Dict[str, str],
    output_dir: Path,
    test_paths: List[str] = None,
) -> TestResult:
    """Run pytest with specific environment configuration."""

    if test_paths is None:
        test_paths = ["tests/unit/", "tests/integration/"]

    log_file = output_dir / f"test_{mode_name}.log"
    junit_file = output_dir / f"test_{mode_name}.xml"

    # Build environment
    env = os.environ.copy()
    env.update(env_vars)
    env["PYTHONPATH"] = str(Path.cwd() / "src")

    # Build pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        *test_paths,
        "-v", "--tb=short",
        f"--junitxml={junit_file}",
        "--ignore=tests/integration/test_bench_evidence.py",
        "--ignore=tests/regression/",
    ]

    print(f"\n{'='*60}")
    print(f"Running tests in mode: {mode_name}")
    print(f"Environment: {env_vars}")
    print(f"{'='*60}\n")

    start_time = datetime.now()

    with open(log_file, "w") as log:
        result = subprocess.run(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(Path.cwd()),
        )
        log.write(result.stdout)

    duration = (datetime.now() - start_time).total_seconds()

    # Parse results from output
    passed = failed = skipped = errors = warnings = 0

    for line in result.stdout.split("\n"):
        line_lower = line.lower()
        if "passed" in line_lower and ("failed" in line_lower or "error" in line_lower or "=" in line):
            # Parse summary line like "10 passed, 2 failed, 1 error in 5.23s"
            parts = line.split()
            for i, part in enumerate(parts):
                if "passed" in part and i > 0:
                    try:
                        passed = int(parts[i-1])
                    except ValueError:
                        pass
                if "failed" in part and i > 0:
                    try:
                        failed = int(parts[i-1])
                    except ValueError:
                        pass
                if "skipped" in part and i > 0:
                    try:
                        skipped = int(parts[i-1])
                    except ValueError:
                        pass
                if "error" in part.lower() and i > 0:
                    try:
                        errors = int(parts[i-1])
                    except ValueError:
                        pass
                if "warning" in part.lower() and i > 0:
                    try:
                        warnings = int(parts[i-1])
                    except ValueError:
                        pass

    return TestResult(
        mode=mode_name,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=errors,
        duration_seconds=duration,
        warnings=warnings,
        exit_code=result.returncode,
        log_file=str(log_file),
    )


def run_test_matrix(
    output_dir: Optional[Path] = None,
    modes: Optional[List[str]] = None,
) -> MatrixResult:
    """Run the full test matrix across all privacy modes."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = Path("reports/qa") / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)

    if modes is None:
        modes = list(PRIVACY_MODES.keys())

    git_sha, git_branch = get_git_info()

    print(f"\nTest Matrix Runner")
    print(f"==================")
    print(f"Timestamp: {timestamp}")
    print(f"Git SHA: {git_sha}")
    print(f"Git Branch: {git_branch}")
    print(f"Modes: {modes}")
    print(f"Output: {output_dir}")

    results = {}

    for mode_name in modes:
        if mode_name not in PRIVACY_MODES:
            print(f"Warning: Unknown mode {mode_name}, skipping")
            continue

        mode_config = PRIVACY_MODES[mode_name]
        env_vars = {k: v for k, v in mode_config.items() if k != "description"}

        result = run_tests_for_mode(mode_name, env_vars, output_dir)
        results[mode_name] = result

        status = "PASS" if result.exit_code == 0 else "FAIL"
        print(f"\n[{status}] Mode {mode_name}: {result.passed} passed, {result.failed} failed, {result.errors} errors")

    # Calculate totals
    total_tests = sum(r.passed + r.failed + r.errors for r in results.values())
    total_failures = sum(r.failed + r.errors for r in results.values())
    all_passed = all(r.exit_code == 0 for r in results.values())

    matrix_result = MatrixResult(
        timestamp=timestamp,
        git_sha=git_sha,
        git_branch=git_branch,
        modes_tested=modes,
        results={k: asdict(v) for k, v in results.items()},
        all_passed=all_passed,
        total_tests=total_tests,
        total_failures=total_failures,
    )

    # Write JSON result
    result_file = output_dir / "matrix_result.json"
    with open(result_file, "w") as f:
        json.dump(asdict(matrix_result), f, indent=2)

    # Write markdown summary
    summary_file = output_dir / "matrix_summary.md"
    with open(summary_file, "w") as f:
        f.write(f"# Test Matrix Results\n\n")
        f.write(f"**Timestamp**: {timestamp}\n")
        f.write(f"**Git SHA**: {git_sha}\n")
        f.write(f"**Git Branch**: {git_branch}\n")
        f.write(f"**Overall Status**: {'PASS' if all_passed else 'FAIL'}\n\n")

        f.write("## Results by Mode\n\n")
        f.write("| Mode | Status | Passed | Failed | Errors | Duration |\n")
        f.write("|------|--------|--------|--------|--------|----------|\n")

        for mode_name, result in results.items():
            status = "PASS" if result.exit_code == 0 else "FAIL"
            f.write(f"| {mode_name} | {status} | {result.passed} | {result.failed} | {result.errors} | {result.duration_seconds:.1f}s |\n")

        f.write(f"\n**Total Tests**: {total_tests}\n")
        f.write(f"**Total Failures**: {total_failures}\n")

    print(f"\n{'='*60}")
    print(f"Test Matrix Complete")
    print(f"{'='*60}")
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print(f"Total Tests: {total_tests}")
    print(f"Total Failures: {total_failures}")
    print(f"Results: {result_file}")
    print(f"Summary: {summary_file}")

    return matrix_result


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run test matrix across privacy modes")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(PRIVACY_MODES.keys()),
        help="Specific modes to test (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for results",
    )

    args = parser.parse_args()

    result = run_test_matrix(
        output_dir=args.output_dir,
        modes=args.modes,
    )

    # Exit with failure if any tests failed
    sys.exit(0 if result.all_passed else 1)


if __name__ == "__main__":
    main()
