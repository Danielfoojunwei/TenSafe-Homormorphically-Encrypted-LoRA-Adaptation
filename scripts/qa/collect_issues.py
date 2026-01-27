#!/usr/bin/env python3
"""
TensorGuardFlow Issue Collector

Parses test results and static analysis to generate issues.json.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


def parse_junit_xml(xml_path: str) -> List[Dict[str, Any]]:
    """Parse JUnit XML for failures."""
    issues = []

    if not os.path.exists(xml_path):
        return issues

    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for testcase in root.iter("testcase"):
            failure = testcase.find("failure")
            error = testcase.find("error")

            if failure is not None or error is not None:
                element = failure if failure is not None else error
                issues.append({
                    "title": f"Test failure: {testcase.get('classname')}.{testcase.get('name')}",
                    "severity": "P1",
                    "component": infer_component(testcase.get("classname", "")),
                    "reproduction_steps": [
                        f"Run: pytest {testcase.get('classname', '').replace('.', '/')}::{testcase.get('name')}"
                    ],
                    "expected": "Test should pass",
                    "actual": element.get("message", "Test failed"),
                    "logs_path": xml_path,
                    "proposed_fix": "Investigate test failure and fix underlying issue",
                    "regression_test": testcase.get("name")
                })
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")

    return issues


def parse_static_checks(md_path: str) -> List[Dict[str, Any]]:
    """Parse static checks markdown for issues."""
    issues = []

    if not os.path.exists(md_path):
        return issues

    try:
        with open(md_path, "r") as f:
            content = f.read()

        # Look for ruff errors
        ruff_match = re.search(r"## Linting.*?```(.*?)```", content, re.DOTALL)
        if ruff_match:
            ruff_output = ruff_match.group(1)
            error_lines = [l for l in ruff_output.split("\n") if ": E" in l or ": F" in l]

            if len(error_lines) > 0:
                issues.append({
                    "title": f"Lint issues: {len(error_lines)} violations",
                    "severity": "P2",
                    "component": "code_quality",
                    "reproduction_steps": ["Run: ruff check src/"],
                    "expected": "No lint violations",
                    "actual": f"{len(error_lines)} violations found",
                    "logs_path": md_path,
                    "proposed_fix": "Run: ruff check src/ --fix",
                    "regression_test": None
                })

        # Look for security issues
        if "ISSUES FOUND" in content or "High severity" in content.lower():
            issues.append({
                "title": "Security scan found potential issues",
                "severity": "P0",
                "component": "security",
                "reproduction_steps": ["Run: bandit -r src/"],
                "expected": "No security issues",
                "actual": "Security issues detected",
                "logs_path": md_path,
                "proposed_fix": "Review and address security findings",
                "regression_test": None
            })

    except Exception as e:
        print(f"Error parsing {md_path}: {e}")

    return issues


def parse_perf_baseline(json_path: str) -> List[Dict[str, Any]]:
    """Parse performance baseline for budget violations."""
    issues = []

    if not os.path.exists(json_path):
        return issues

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        for scenario in data.get("scenarios", []):
            for op in scenario.get("operations", []):
                if "passed" in op and not op["passed"]:
                    issues.append({
                        "title": f"Performance budget exceeded: {op.get('operation')}",
                        "severity": "P1",
                        "component": "performance",
                        "reproduction_steps": [
                            f"Run: python benchmarks/perf_baseline/run_perf.py"
                        ],
                        "expected": f"Latency < {op.get('budget_ms', 'N/A')}ms",
                        "actual": f"Latency = {op.get('latency_ms', op.get('avg_latency_ms', 'N/A'))}ms",
                        "logs_path": json_path,
                        "proposed_fix": "Investigate slow code path and optimize",
                        "regression_test": f"test_metrics_endpoint_latency_budget"
                    })

    except Exception as e:
        print(f"Error parsing {json_path}: {e}")

    return issues


def infer_component(class_name: str) -> str:
    """Infer component from test class name."""
    class_lower = class_name.lower()

    if "n2he" in class_lower or "privacy" in class_lower:
        return "n2he"
    elif "dashboard" in class_lower or "metrics" in class_lower:
        return "dashboard"
    elif "rollback" in class_lower or "promote" in class_lower:
        return "orchestrator"
    elif "evidence" in class_lower or "tgsp" in class_lower:
        return "evidence"
    elif "export" in class_lower or "integration" in class_lower:
        return "integrations"
    elif "concurrent" in class_lower or "race" in class_lower:
        return "concurrency"
    else:
        return "core"


def collect_issues(reports_dir: str) -> Dict[str, Any]:
    """Collect all issues from QA reports."""

    issues = []

    # Parse JUnit XMLs
    for xml_file in Path(reports_dir).glob("junit_*.xml"):
        issues.extend(parse_junit_xml(str(xml_file)))

    # Parse static checks
    static_path = Path(reports_dir) / "static_checks.md"
    issues.extend(parse_static_checks(str(static_path)))

    # Parse performance baseline
    perf_path = Path(reports_dir) / "perf_baseline.json"
    issues.extend(parse_perf_baseline(str(perf_path)))

    # Sort by severity
    severity_order = {"P0": 0, "P1": 1, "P2": 2}
    issues.sort(key=lambda x: severity_order.get(x.get("severity", "P2"), 3))

    # Add IDs
    for i, issue in enumerate(issues):
        issue["id"] = f"TGQA-{i+1:04d}"

    result = {
        "generated_at": datetime.now().isoformat(),
        "reports_dir": reports_dir,
        "total_issues": len(issues),
        "by_severity": {
            "P0": len([i for i in issues if i.get("severity") == "P0"]),
            "P1": len([i for i in issues if i.get("severity") == "P1"]),
            "P2": len([i for i in issues if i.get("severity") == "P2"]),
        },
        "issues": issues
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect QA issues")
    parser.add_argument("reports_dir", help="Path to QA reports directory")
    parser.add_argument("--output", "-o", help="Output file path", default=None)
    args = parser.parse_args()

    result = collect_issues(args.reports_dir)

    output_path = args.output or os.path.join(args.reports_dir, "issues.json")

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Collected {result['total_issues']} issues")
    print(f"  P0 (Blocker): {result['by_severity']['P0']}")
    print(f"  P1 (Major): {result['by_severity']['P1']}")
    print(f"  P2 (Minor): {result['by_severity']['P2']}")
    print(f"Output: {output_path}")

    return result


if __name__ == "__main__":
    main()
