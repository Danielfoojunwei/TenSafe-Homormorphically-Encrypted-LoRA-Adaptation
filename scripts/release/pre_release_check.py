#!/usr/bin/env python3
"""
Pre-Release Verification Script.

Runs all quality gates that must pass before a release can be cut.
Exit code 0 = all checks pass, ready for release.
Exit code 1 = one or more checks failed.

Usage:
    python scripts/release/pre_release_check.py --version 3.0.1
    python scripts/release/pre_release_check.py --version 3.0.1 --fix
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class CheckResult:
    """Result of a single check."""

    name: str
    passed: bool
    message: str
    fixable: bool = False


def run_command(cmd: List[str], capture: bool = True) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=300,
        )
        output = result.stdout + result.stderr if capture else ""
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except FileNotFoundError:
        return 1, f"Command not found: {cmd[0]}"


def check_ruff_format() -> CheckResult:
    """Check code formatting with ruff."""
    code, output = run_command(["python", "-m", "ruff", "format", "--check", "src/", "tests/"])
    if code == 0:
        return CheckResult("ruff format", True, "Code is properly formatted")
    return CheckResult("ruff format", False, "Code needs formatting. Run: ruff format src/ tests/", fixable=True)


def check_ruff_lint() -> CheckResult:
    """Check linting with ruff."""
    code, output = run_command(["python", "-m", "ruff", "check", "src/", "tests/"])
    if code == 0:
        return CheckResult("ruff lint", True, "No linting errors")
    # Count errors
    error_count = output.count("\n") - 1 if output else 0
    return CheckResult("ruff lint", False, f"{error_count} linting issues. Run: ruff check --fix src/ tests/", fixable=True)


def check_pyright() -> CheckResult:
    """Check type annotations with pyright."""
    code, output = run_command(["python", "-m", "pyright", "--project", "pyrightconfig.json"])
    if code == 0:
        return CheckResult("pyright", True, "No type errors")
    # Extract error count
    if "error" in output.lower():
        return CheckResult("pyright", False, "Type errors found. Run: pyright --project pyrightconfig.json")
    return CheckResult("pyright", True, "No type errors (warnings only)")


def check_tests() -> CheckResult:
    """Run test suite."""
    code, output = run_command(["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-q"])
    if code == 0:
        return CheckResult("tests", True, "All tests pass")
    # Extract failure count
    if "failed" in output.lower():
        return CheckResult("tests", False, "Some tests failed. Run: pytest tests/ -v")
    return CheckResult("tests", False, f"Test run failed with code {code}")


def check_security_tests() -> CheckResult:
    """Run security-specific tests."""
    code, output = run_command(["python", "-m", "pytest", "tests/security/", "-v", "--tb=short", "-q"])
    if code == 0:
        return CheckResult("security tests", True, "All security tests pass")
    return CheckResult("security tests", False, "Security tests failed. Run: pytest tests/security/ -v")


def check_version_consistency(expected_version: str) -> CheckResult:
    """Check that version is consistent across files."""
    import re

    files_to_check = [
        ("pyproject.toml", r'version = "([^"]+)"'),
        ("src/tensorguard/platform/main.py", r'version="([^"]+)"'),
    ]

    versions_found = {}
    for filepath, pattern in files_to_check:
        try:
            with open(filepath) as f:
                content = f.read()
            match = re.search(pattern, content)
            if match:
                versions_found[filepath] = match.group(1)
            else:
                versions_found[filepath] = "NOT FOUND"
        except FileNotFoundError:
            versions_found[filepath] = "FILE NOT FOUND"

    all_match = all(v == expected_version for v in versions_found.values())
    if all_match:
        return CheckResult("version consistency", True, f"All files have version {expected_version}")

    details = ", ".join(f"{k}: {v}" for k, v in versions_found.items())
    return CheckResult("version consistency", False, f"Version mismatch. Expected {expected_version}, found: {details}")


def check_no_debug_code() -> CheckResult:
    """Check for debug statements that shouldn't be in release."""
    patterns = [
        "import pdb",
        "breakpoint()",
        "print(__",
        "TODO: REMOVE",
        "FIXME: BEFORE RELEASE",
    ]

    code, output = run_command(
        ["grep", "-rn", "--include=*.py"]
        + [item for p in patterns for item in ["-e", p]]
        + ["src/"]
    )

    if code != 0:  # grep returns 1 if no matches
        return CheckResult("no debug code", True, "No debug statements found")

    lines = output.strip().split("\n") if output.strip() else []
    if lines:
        return CheckResult("no debug code", False, f"Found {len(lines)} debug statement(s). Review and remove.")
    return CheckResult("no debug code", True, "No debug statements found")


def check_secrets_scan() -> CheckResult:
    """Check for exposed secrets."""
    patterns = [
        r"api_key\s*=\s*['\"][^'\"]+['\"]",
        r"password\s*=\s*['\"][^'\"]+['\"]",
        r"secret\s*=\s*['\"][^'\"]+['\"]",
        r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    ]

    # Simple grep-based check
    for pattern in patterns:
        code, output = run_command(["grep", "-rEn", "--include=*.py", pattern, "src/"])
        if code == 0 and output.strip():
            # Filter out test files and known safe patterns
            lines = [l for l in output.strip().split("\n") if "test" not in l.lower() and "example" not in l.lower()]
            if lines:
                return CheckResult("secrets scan", False, f"Potential secrets found. Review: {lines[0][:80]}...")

    return CheckResult("secrets scan", True, "No exposed secrets detected")


def run_all_checks(version: str) -> List[CheckResult]:
    """Run all pre-release checks."""
    checks = [
        ("Code formatting", check_ruff_format),
        ("Linting", check_ruff_lint),
        ("Type checking", check_pyright),
        ("Unit tests", check_tests),
        ("Security tests", check_security_tests),
        ("Version consistency", lambda: check_version_consistency(version)),
        ("Debug code", check_no_debug_code),
        ("Secrets scan", check_secrets_scan),
    ]

    results = []
    for name, check_fn in checks:
        print(f"Running: {name}...", end=" ", flush=True)
        try:
            result = check_fn()
            status = "PASS" if result.passed else "FAIL"
            print(status)
            results.append(result)
        except Exception as e:
            print("ERROR")
            results.append(CheckResult(name, False, f"Check failed with error: {e}"))

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-release verification")
    parser.add_argument("--version", required=True, help="Expected version number (e.g., 3.0.1)")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix auto-fixable issues")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Pre-Release Verification for v{args.version}")
    print(f"{'='*60}\n")

    if args.fix:
        print("Auto-fixing enabled. Running formatters first...\n")
        subprocess.run(["python", "-m", "ruff", "format", "src/", "tests/"])
        subprocess.run(["python", "-m", "ruff", "check", "--fix", "src/", "tests/"])
        print()

    results = run_all_checks(args.version)

    print(f"\n{'='*60}")
    print("  Results Summary")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        icon = "" if result.passed else ""
        print(f"{icon} [{status}] {result.name}")
        if not result.passed:
            print(f"      {result.message}")
            failed += 1
        else:
            passed += 1

    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed")
    print(f"{'='*60}\n")

    if failed > 0:
        print("Release blocked. Please fix the above issues.\n")
        sys.exit(1)
    else:
        print(f"All checks passed! Ready to release v{args.version}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
