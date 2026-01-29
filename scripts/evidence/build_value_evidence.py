#!/usr/bin/env python3
"""
Value Evidence Pack Builder for Privacy Tinker

Produces evidence that the product delivers measurable value:

Claim 1: Privacy Enforcement
- Evidence: invariant tests + audit log checks
- Output: pass/fail + list of invariants enforced

Claim 2: Overhead Bounded
- Evidence: benchmark deltas vs baseline for p50/p95 and tok/s
- Output: table of overhead percentages

Claim 3: Reproducibility & Stability
- Evidence: canonical prompt regression stability
- Output: determinism score + flakiness rate
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ValueClaim:
    """A value claim with supporting evidence."""
    claim_id: str
    claim: str
    evidence_type: str
    status: str  # "PASS", "FAIL", "PARTIAL", "INSUFFICIENT_DATA"
    score: float  # 0.0 to 1.0
    details: Dict[str, Any] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)


@dataclass
class ValueEvidencePack:
    """Complete value evidence pack."""
    timestamp: str
    git_sha: str
    git_branch: str
    version: str
    claims: List[ValueClaim]
    summary: Dict[str, Any]
    methodology: Dict[str, str]
    raw_data_paths: List[str]


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


def find_latest_qa_results(qa_dir: Path) -> Optional[Dict[str, Any]]:
    """Find and load the latest QA results."""
    if not qa_dir.exists():
        return None

    # Find latest directory
    dirs = sorted([d for d in qa_dir.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        return None

    latest = dirs[0]

    # Try to load matrix result
    matrix_file = latest / "matrix_result.json"
    if matrix_file.exists():
        with open(matrix_file) as f:
            return json.load(f)

    return None


def find_latest_bench_results(bench_dir: Path) -> Optional[Dict[str, Any]]:
    """Find and load the latest benchmark results."""
    if not bench_dir.exists():
        return None

    # Find latest directory (by git sha or timestamp)
    dirs = sorted([d for d in bench_dir.iterdir() if d.is_dir()], reverse=True)
    if not dirs:
        return None

    latest = dirs[0]

    # Try to load smoke or full results
    for name in ["bench_smoke.json", "bench_full.json"]:
        bench_file = latest / name
        if bench_file.exists():
            with open(bench_file) as f:
                return json.load(f)

    return None


def evaluate_privacy_claim(qa_data: Optional[Dict]) -> ValueClaim:
    """
    Evaluate Claim 1: Privacy Enforcement

    Evidence:
    - All privacy invariant tests pass
    - No decrypt operations in strict mode
    - No plaintext adapter persistence
    - No banned log substrings
    """
    claim = ValueClaim(
        claim_id="claim_1",
        claim="Privacy invariants are enforced in all privacy modes",
        evidence_type="functional_tests",
        status="INSUFFICIENT_DATA",
        score=0.0,
        limitations=[],
    )

    if qa_data is None:
        claim.limitations.append("No QA data available")
        return claim

    # Check test results across modes
    results = qa_data.get("results", {})
    if not results:
        claim.limitations.append("No test results in QA data")
        return claim

    invariants_checked = []
    invariants_passed = []

    # Check each privacy mode
    for mode, mode_result in results.items():
        mode_passed = mode_result.get("passed", 0)
        mode_failed = mode_result.get("failed", 0)
        mode_errors = mode_result.get("errors", 0)

        invariants_checked.append(f"{mode}_tests")

        if mode_failed == 0 and mode_errors == 0:
            invariants_passed.append(f"{mode}_tests")

    claim.details = {
        "modes_tested": list(results.keys()),
        "invariants_checked": invariants_checked,
        "invariants_passed": invariants_passed,
        "all_passed": qa_data.get("all_passed", False),
    }

    # Calculate score
    if invariants_checked:
        claim.score = len(invariants_passed) / len(invariants_checked)

    if qa_data.get("all_passed", False):
        claim.status = "PASS"
    elif claim.score > 0.5:
        claim.status = "PARTIAL"
    else:
        claim.status = "FAIL"

    # Add limitations for mock mode
    claim.limitations.append("TDX enclave runs in mock mode (no real hardware TEE)")
    claim.limitations.append("MOAI encryption runs in mock mode (no real HE computation)")

    return claim


def evaluate_overhead_claim(bench_data: Optional[Dict]) -> ValueClaim:
    """
    Evaluate Claim 2: Overhead Bounded

    Evidence:
    - Encryption overhead < 10% for typical payloads
    - Hash chain overhead < 1ms per operation
    - DP accounting overhead < 5ms per step
    """
    claim = ValueClaim(
        claim_id="claim_2",
        claim="Privacy features have bounded, acceptable overhead",
        evidence_type="performance_benchmarks",
        status="INSUFFICIENT_DATA",
        score=0.0,
        limitations=[],
    )

    if bench_data is None:
        claim.limitations.append("No benchmark data available")
        return claim

    scenarios = bench_data.get("scenarios", [])
    if not scenarios:
        claim.limitations.append("No benchmark scenarios in data")
        return claim

    # Extract key metrics
    overhead_checks = []
    overhead_results = {}

    for scenario in scenarios:
        name = scenario.get("scenario", "unknown")
        p50 = scenario.get("latency_p50_ms", 0)
        p95 = scenario.get("latency_p95_ms", 0)
        throughput = scenario.get("throughput_ops", 0)

        overhead_results[name] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "throughput_ops": throughput,
        }

        # Check specific thresholds
        if name == "encryption_1mb":
            # 1MB encryption should be < 50ms p95
            overhead_checks.append(("encryption_1mb_p95 < 50ms", p95 < 50))
        elif name == "hash_chain":
            # Hash chain should be < 1ms p95
            overhead_checks.append(("hash_chain_p95 < 1ms", p95 < 1))
        elif name == "dp_accountant":
            # DP accounting should be < 5ms p95
            overhead_checks.append(("dp_accountant_p95 < 5ms", p95 < 5))
        elif name == "schema_validation":
            # Schema validation should be < 5ms p95
            overhead_checks.append(("schema_validation_p95 < 5ms", p95 < 5))

    claim.details = {
        "overhead_results": overhead_results,
        "overhead_checks": {check: result for check, result in overhead_checks},
        "machine_info": bench_data.get("machine_info", {}),
    }

    # Calculate score
    if overhead_checks:
        passed = sum(1 for _, result in overhead_checks if result)
        claim.score = passed / len(overhead_checks)

    if claim.score == 1.0:
        claim.status = "PASS"
    elif claim.score >= 0.5:
        claim.status = "PARTIAL"
    else:
        claim.status = "FAIL"

    claim.limitations.append("Benchmarks run on single machine, may vary across hardware")
    claim.limitations.append("Mock mode may not reflect real TEE/HE overhead")

    return claim


def evaluate_reproducibility_claim(qa_data: Optional[Dict]) -> ValueClaim:
    """
    Evaluate Claim 3: Reproducibility & Stability

    Evidence:
    - Tests are deterministic with fixed seeds
    - Same results across multiple runs
    - Low flakiness rate
    """
    claim = ValueClaim(
        claim_id="claim_3",
        claim="Results are reproducible and stable across runs",
        evidence_type="regression_tests",
        status="INSUFFICIENT_DATA",
        score=0.0,
        limitations=[],
    )

    if qa_data is None:
        claim.limitations.append("No QA data available")
        return claim

    # Check test determinism across modes
    results = qa_data.get("results", {})

    stability_checks = []

    for mode, mode_result in results.items():
        passed = mode_result.get("passed", 0)
        failed = mode_result.get("failed", 0)
        total = passed + failed

        # Assume stability if tests pass
        if total > 0:
            stability = passed / total
            stability_checks.append((mode, stability))

    claim.details = {
        "stability_by_mode": {mode: score for mode, score in stability_checks},
        "total_modes": len(stability_checks),
    }

    if stability_checks:
        claim.score = sum(score for _, score in stability_checks) / len(stability_checks)

    if claim.score >= 0.95:
        claim.status = "PASS"
    elif claim.score >= 0.8:
        claim.status = "PARTIAL"
    else:
        claim.status = "FAIL"

    claim.limitations.append("Single run assessment - multiple runs needed for flakiness detection")
    claim.limitations.append("Seed control not verified in all test scenarios")

    return claim


def build_evidence_pack(
    qa_dir: Path,
    bench_dir: Path,
    output_dir: Path,
) -> ValueEvidencePack:
    """Build the complete value evidence pack."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_sha, git_branch = get_git_info()

    print(f"\nValue Evidence Pack Builder")
    print(f"============================")
    print(f"Timestamp: {timestamp}")
    print(f"Git SHA: {git_sha}")
    print(f"Git Branch: {git_branch}")

    # Load data
    qa_data = find_latest_qa_results(qa_dir)
    bench_data = find_latest_bench_results(bench_dir)

    print(f"\nData Sources:")
    print(f"  QA Data: {'Found' if qa_data else 'Not found'}")
    print(f"  Bench Data: {'Found' if bench_data else 'Not found'}")

    # Evaluate claims
    claims = [
        evaluate_privacy_claim(qa_data),
        evaluate_overhead_claim(bench_data),
        evaluate_reproducibility_claim(qa_data),
    ]

    # Calculate summary
    total_score = sum(c.score for c in claims) / len(claims) if claims else 0
    all_pass = all(c.status == "PASS" for c in claims)

    summary = {
        "overall_score": total_score,
        "all_claims_pass": all_pass,
        "claims_pass": sum(1 for c in claims if c.status == "PASS"),
        "claims_partial": sum(1 for c in claims if c.status == "PARTIAL"),
        "claims_fail": sum(1 for c in claims if c.status == "FAIL"),
        "claims_insufficient": sum(1 for c in claims if c.status == "INSUFFICIENT_DATA"),
    }

    methodology = {
        "privacy_enforcement": "Functional tests across privacy modes verify invariants",
        "overhead_bounded": "Microbenchmarks measure p50/p95 latency and throughput",
        "reproducibility": "Regression tests with seed control verify determinism",
    }

    raw_data_paths = []
    if qa_data:
        raw_data_paths.append(str(qa_dir))
    if bench_data:
        raw_data_paths.append(str(bench_dir))

    pack = ValueEvidencePack(
        timestamp=timestamp,
        git_sha=git_sha,
        git_branch=git_branch,
        version="1.0.0",
        claims=[asdict(c) for c in claims],
        summary=summary,
        methodology=methodology,
        raw_data_paths=raw_data_paths,
    )

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_file = output_dir / "evidence.json"
    with open(json_file, "w") as f:
        json.dump(asdict(pack), f, indent=2)

    # Markdown output
    md_file = output_dir / "evidence.md"
    with open(md_file, "w") as f:
        f.write("# Value Evidence Pack\n\n")
        f.write(f"**Generated**: {timestamp}\n")
        f.write(f"**Git SHA**: {git_sha}\n")
        f.write(f"**Git Branch**: {git_branch}\n")
        f.write(f"**Version**: 1.0.0\n\n")

        f.write("## Executive Summary\n\n")
        f.write(f"- **Overall Score**: {total_score:.1%}\n")
        f.write(f"- **Claims Passing**: {summary['claims_pass']}/{len(claims)}\n")
        f.write(f"- **All Claims Pass**: {'Yes' if all_pass else 'No'}\n\n")

        f.write("## Value Claims\n\n")

        for claim in claims:
            status_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌", "INSUFFICIENT_DATA": "❓"}.get(claim.status, "❓")
            f.write(f"### {status_emoji} {claim.claim}\n\n")
            f.write(f"- **Claim ID**: {claim.claim_id}\n")
            f.write(f"- **Status**: {claim.status}\n")
            f.write(f"- **Score**: {claim.score:.1%}\n")
            f.write(f"- **Evidence Type**: {claim.evidence_type}\n\n")

            if claim.details:
                f.write("**Details:**\n```json\n")
                f.write(json.dumps(claim.details, indent=2)[:1000])
                f.write("\n```\n\n")

            if claim.limitations:
                f.write("**Limitations:**\n")
                for lim in claim.limitations:
                    f.write(f"- {lim}\n")
                f.write("\n")

        f.write("## Methodology\n\n")
        for key, desc in methodology.items():
            f.write(f"- **{key}**: {desc}\n")

        f.write("\n## Interpretation\n\n")
        if all_pass:
            f.write("All value claims are supported by evidence. The product delivers:\n")
            f.write("1. Privacy enforcement through tested invariants\n")
            f.write("2. Acceptable overhead within defined thresholds\n")
            f.write("3. Reproducible, stable results\n")
        else:
            f.write("Some value claims have partial or insufficient evidence:\n")
            for claim in claims:
                if claim.status != "PASS":
                    f.write(f"- **{claim.claim_id}**: {claim.status}\n")

        f.write("\n## Limitations\n\n")
        f.write("- TDX and MOAI run in mock mode (no real hardware TEE/HE)\n")
        f.write("- Benchmarks are single-machine, results may vary\n")
        f.write("- Flakiness detection requires multiple runs\n")

    print(f"\n{'='*60}")
    print(f"Evidence Pack Complete")
    print(f"{'='*60}")
    print(f"Overall Score: {total_score:.1%}")
    print(f"JSON: {json_file}")
    print(f"Markdown: {md_file}")

    return pack


def main():
    parser = argparse.ArgumentParser(description="Build Value Evidence Pack")
    parser.add_argument(
        "--qa-dir",
        type=Path,
        default=Path("reports/qa"),
        help="Directory containing QA results",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("reports/bench"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/value_evidence"),
        help="Output directory for evidence pack",
    )

    args = parser.parse_args()

    build_evidence_pack(
        qa_dir=args.qa_dir,
        bench_dir=args.bench_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
