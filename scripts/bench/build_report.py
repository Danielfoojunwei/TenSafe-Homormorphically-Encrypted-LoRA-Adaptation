#!/usr/bin/env python3
"""
Unified Benchmark Report Builder

Generates a comprehensive benchmark report that includes:
- Training metrics
- Evaluation results
- Performance benchmarks
- Privacy & Security compliance evidence

Usage:
    python scripts/bench/build_report.py [--git-sha SHA] [--output-dir PATH]

Outputs:
    reports/bench/<git_sha>/unified_report.md
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class UnifiedReportBuilder:
    """Builds a unified benchmark + compliance report."""

    def __init__(
        self,
        git_sha: str,
        bench_dir: Path,
        compliance_dir: Path,
        output_dir: Path,
    ):
        self.git_sha = git_sha
        self.bench_dir = bench_dir
        self.compliance_dir = compliance_dir
        self.output_dir = output_dir

        # Load data
        self.training_data = self._load_training_data()
        self.eval_data = self._load_eval_data()
        self.perf_data = self._load_perf_data()
        self.compliance_data = self._load_compliance_data()

    def _load_json_safe(self, path: Path) -> Optional[Dict]:
        """Safely load a JSON file."""
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _load_training_data(self) -> Optional[Dict]:
        """Load training benchmark data."""
        # Try multiple possible locations
        candidates = [
            self.bench_dir / self.git_sha / "training.json",
            self.bench_dir / "training_results.json",
            self.bench_dir / self.git_sha / "training_metrics.json",
        ]
        for path in candidates:
            data = self._load_json_safe(path)
            if data:
                return data
        return None

    def _load_eval_data(self) -> Optional[Dict]:
        """Load evaluation benchmark data."""
        candidates = [
            self.bench_dir / self.git_sha / "eval.json",
            self.bench_dir / "eval_results.json",
            self.bench_dir / self.git_sha / "eval_metrics.json",
        ]
        for path in candidates:
            data = self._load_json_safe(path)
            if data:
                return data
        return None

    def _load_perf_data(self) -> Optional[Dict]:
        """Load performance benchmark data."""
        candidates = [
            self.bench_dir / "benchmark_results_latest.json",
            Path("artifacts/benchmarks/benchmark_results_latest.json"),
            self.bench_dir / self.git_sha / "perf.json",
        ]
        for path in candidates:
            data = self._load_json_safe(path)
            if data:
                return data
        return None

    def _load_compliance_data(self) -> Optional[Dict]:
        """Load compliance evidence data."""
        # Try git SHA directory first
        metrics_path = self.compliance_dir / self.git_sha / "metrics.json"
        if not metrics_path.exists():
            # Try finding any metrics file
            metrics_files = list(self.compliance_dir.glob("*/metrics.json"))
            if metrics_files:
                metrics_path = metrics_files[-1]  # Most recent

        return self._load_json_safe(metrics_path)

    def _load_evidence_data(self) -> Optional[Dict]:
        """Load compliance evidence report."""
        evidence_path = self.compliance_dir / self.git_sha / "evidence.json"
        if not evidence_path.exists():
            evidence_files = list(self.compliance_dir.glob("*/evidence.json"))
            if evidence_files:
                evidence_path = evidence_files[-1]

        return self._load_json_safe(evidence_path)

    def generate_report(self) -> str:
        """Generate the unified Markdown report."""
        lines = []

        # Header
        lines.append("# Llama3 LoRA Benchmark Report")
        lines.append("")
        lines.append(f"> **Git SHA**: `{self.git_sha}`")
        lines.append(f"> **Generated**: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Executive Summary](#executive-summary)")
        lines.append("2. [Training Metrics](#training-metrics)")
        lines.append("3. [Evaluation Results](#evaluation-results)")
        lines.append("4. [Performance Benchmarks](#performance-benchmarks)")
        lines.append("5. [Privacy & Security Evidence](#privacy--security-evidence)")
        lines.append("6. [Artifacts](#artifacts)")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.extend(self._generate_executive_summary())
        lines.append("")

        # Training Metrics
        lines.append("## Training Metrics")
        lines.append("")
        lines.extend(self._generate_training_section())
        lines.append("")

        # Evaluation Results
        lines.append("## Evaluation Results")
        lines.append("")
        lines.extend(self._generate_eval_section())
        lines.append("")

        # Performance Benchmarks
        lines.append("## Performance Benchmarks")
        lines.append("")
        lines.extend(self._generate_perf_section())
        lines.append("")

        # Privacy & Security Evidence
        lines.append("## Privacy & Security Evidence")
        lines.append("")
        lines.extend(self._generate_compliance_section())
        lines.append("")

        # Artifacts
        lines.append("## Artifacts")
        lines.append("")
        lines.extend(self._generate_artifacts_section())
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("_Generated by TensorGuard Unified Report Builder_")
        lines.append("")

        return "\n".join(lines)

    def _generate_executive_summary(self) -> List[str]:
        """Generate executive summary section."""
        lines = []

        # Overall status
        training_ok = self.training_data is not None
        eval_ok = self.eval_data is not None
        perf_ok = self.perf_data is not None
        compliance_ok = self.compliance_data is not None

        lines.append("| Component | Status |")
        lines.append("|-----------|--------|")
        lines.append(f"| Training | {'Available' if training_ok else 'Not Run'} |")
        lines.append(f"| Evaluation | {'Available' if eval_ok else 'Not Run'} |")
        lines.append(f"| Performance | {'Available' if perf_ok else 'Not Run'} |")
        lines.append(f"| Compliance | {'Available' if compliance_ok else 'Not Run'} |")
        lines.append("")

        # Key metrics summary
        if self.compliance_data:
            pii_found = self.compliance_data.get("pii_scan", {}).get("total_pii_found", 0)
            secrets_found = self.compliance_data.get("secrets_hygiene", {}).get("secrets_found", 0)
            encryption = self.compliance_data.get("encryption", {}).get("at_rest_encryption_enabled", False)
            audit = self.compliance_data.get("audit_logging", {}).get("audit_log_enabled", False)

            lines.append("### Key Compliance Indicators")
            lines.append("")
            lines.append(f"- **PII Exposure**: {pii_found} instances detected")
            lines.append(f"- **Secrets Hygiene**: {secrets_found} potential secrets")
            lines.append(f"- **Encryption at Rest**: {'Enabled' if encryption else 'Not Configured'}")
            lines.append(f"- **Audit Logging**: {'Enabled' if audit else 'Not Configured'}")
            lines.append("")

        return lines

    def _generate_training_section(self) -> List[str]:
        """Generate training metrics section."""
        lines = []

        if self.training_data:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            # Extract common training metrics
            for key in ["loss", "learning_rate", "epoch", "steps", "duration_seconds"]:
                if key in self.training_data:
                    lines.append(f"| {key.replace('_', ' ').title()} | {self.training_data[key]} |")

            lines.append("")
        else:
            lines.append("_Training metrics not available for this run._")
            lines.append("")
            lines.append("To generate training metrics, run:")
            lines.append("```bash")
            lines.append("make bench-llama3")
            lines.append("```")

        return lines

    def _generate_eval_section(self) -> List[str]:
        """Generate evaluation results section."""
        lines = []

        if self.eval_data:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            # Extract common eval metrics
            for key in ["accuracy", "perplexity", "f1_score", "bleu_score", "rouge_score"]:
                if key in self.eval_data:
                    value = self.eval_data[key]
                    if isinstance(value, float):
                        lines.append(f"| {key.replace('_', ' ').title()} | {value:.4f} |")
                    else:
                        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

            lines.append("")
        else:
            lines.append("_Evaluation results not available for this run._")
            lines.append("")

        return lines

    def _generate_perf_section(self) -> List[str]:
        """Generate performance benchmarks section."""
        lines = []

        if self.perf_data:
            # Check for HTTP benchmark results
            if "http_benchmark" in self.perf_data:
                http = self.perf_data["http_benchmark"]
                lines.append("### HTTP Performance")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")

                for key in ["requests_per_second", "avg_latency_ms", "p99_latency_ms", "error_rate"]:
                    if key in http:
                        lines.append(f"| {key.replace('_', ' ').title()} | {http[key]} |")

                lines.append("")

            # Check for resource metrics
            if "resource_metrics" in self.perf_data:
                res = self.perf_data["resource_metrics"]
                lines.append("### Resource Usage")
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|--------|-------|")

                for key in ["cpu_percent", "memory_percent", "disk_percent"]:
                    if key in res:
                        lines.append(f"| {key.replace('_', ' ').title()} | {res[key]:.1f}% |")

                lines.append("")
        else:
            lines.append("_Performance benchmarks not available for this run._")
            lines.append("")

        return lines

    def _generate_compliance_section(self) -> List[str]:
        """Generate compliance evidence section."""
        lines = []

        if self.compliance_data:
            # Link to full evidence report
            evidence_path = f"reports/compliance/{self.git_sha}/evidence.md"
            lines.append(f"Full evidence report: [{evidence_path}]({evidence_path})")
            lines.append("")

            # Encryption Posture
            lines.append("### Encryption Posture")
            lines.append("")
            encryption = self.compliance_data.get("encryption", {})
            lines.append("| Control | Status |")
            lines.append("|---------|--------|")
            lines.append(f"| At-Rest Encryption | {'Enabled' if encryption.get('at_rest_encryption_enabled') else 'Not Configured'} |")
            lines.append(f"| In-Transit Encryption | {'Enabled' if encryption.get('in_transit_encryption_enabled') else 'Not Configured'} |")
            lines.append(f"| Post-Quantum Crypto | {'Enabled' if encryption.get('pqc_enabled') else 'Not Enabled'} |")
            lines.append(f"| Key Rotation | {'Configured' if encryption.get('key_rotation_configured') else 'Not Configured'} |")
            lines.append("")

            # PII Scan Results
            lines.append("### PII Scan Results")
            lines.append("")
            pii = self.compliance_data.get("pii_scan", {})
            total_pii = pii.get("total_pii_found", 0)
            lines.append(f"**Total PII Detected**: {total_pii}")
            lines.append("")

            if total_pii > 0:
                lines.append("_Warning: PII was detected. Review and remediate before production deployment._")
            else:
                lines.append("_No PII detected in scanned files._")
            lines.append("")

            # Audit Logging
            lines.append("### Audit Logging")
            lines.append("")
            audit = self.compliance_data.get("audit_logging", {})
            lines.append("| Control | Status |")
            lines.append("|---------|--------|")
            lines.append(f"| Audit Log Enabled | {'Yes' if audit.get('audit_log_enabled') else 'No'} |")
            lines.append(f"| Integrity Verified | {'Yes' if audit.get('log_integrity_verified') else 'No'} |")
            lines.append(f"| Event Coverage | {audit.get('event_coverage_pct', 0):.1f}% |")
            lines.append("")

            # Data Retention
            lines.append("### Data Retention")
            lines.append("")
            data_inv = self.compliance_data.get("data_inventory", {})
            lines.append(f"**Retention Policy Present**: {'Yes' if data_inv.get('retention_policy_present') else 'No'}")
            lines.append("")

        else:
            lines.append("_Compliance evidence not available for this run._")
            lines.append("")
            lines.append("To generate compliance evidence, run:")
            lines.append("```bash")
            lines.append("make compliance")
            lines.append("```")
            lines.append("")

        return lines

    def _generate_artifacts_section(self) -> List[str]:
        """Generate artifacts listing section."""
        lines = []

        lines.append("| Artifact | Path | Description |")
        lines.append("|----------|------|-------------|")

        # Benchmark artifacts
        lines.append(f"| Benchmark Report | `reports/bench/{self.git_sha}/` | Training and eval results |")

        # Compliance artifacts
        lines.append(f"| Metrics Bundle | `reports/compliance/{self.git_sha}/metrics.json` | Raw compliance metrics |")
        lines.append(f"| Evidence Report | `reports/compliance/{self.git_sha}/evidence.md` | Human-readable evidence |")
        lines.append(f"| Evidence Data | `reports/compliance/{self.git_sha}/evidence.json` | Structured evidence |")

        # Documentation
        lines.append("| Control Matrix | `docs/compliance/CONTROL_MATRIX.md` | Control framework mapping |")
        lines.append("| Data Flow | `docs/compliance/DATA_FLOW.md` | Data flow documentation |")
        lines.append("| Threat Model | `docs/compliance/THREAT_MODEL.md` | Security threat analysis |")

        return lines

    def save_report(self) -> str:
        """Save the report to file."""
        output_path = self.output_dir / self.git_sha
        output_path.mkdir(parents=True, exist_ok=True)

        report_file = output_path / "unified_report.md"
        report_content = self.generate_report()

        with open(report_file, 'w') as f:
            f.write(report_content)

        return str(report_file)


def get_git_sha() -> str:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Build unified benchmark + compliance report"
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Git SHA to use (default: current HEAD)",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("reports/bench"),
        help="Benchmark reports directory",
    )
    parser.add_argument(
        "--compliance-dir",
        type=Path,
        default=Path("reports/compliance"),
        help="Compliance reports directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/bench"),
        help="Output directory for unified report",
    )

    args = parser.parse_args()

    git_sha = args.git_sha or get_git_sha()

    builder = UnifiedReportBuilder(
        git_sha=git_sha,
        bench_dir=args.bench_dir.resolve(),
        compliance_dir=args.compliance_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )

    report_path = builder.save_report()
    print(f"Unified report generated: {report_path}")


if __name__ == "__main__":
    main()
