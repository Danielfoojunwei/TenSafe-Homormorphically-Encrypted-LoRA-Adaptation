#!/usr/bin/env python3
"""
Compliance Evidence Report Builder

Generates human-readable (Markdown) and machine-readable (JSON) compliance
evidence reports mapped to ISO/IEC 27701, ISO/IEC 27001, and SOC 2 controls.

Usage:
    python scripts/compliance/build_compliance_evidence.py [--metrics-file PATH] [--output-dir PATH]

Outputs:
    reports/compliance/<git_sha>/evidence.md
    reports/compliance/<git_sha>/evidence.json
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# =============================================================================
# Control Mapping Definitions
# =============================================================================

CONTROL_FAMILIES = {
    "iso27701": {
        "PIM-1": {
            "name": "Consent & Purpose Limitation",
            "description": "Ensure personal data is processed only for specified, legitimate purposes",
            "metrics": ["data_inventory.purpose_tags_present", "data_inventory.classification_applied"],
        },
        "PIM-2": {
            "name": "Data Minimization",
            "description": "Collect and process only data necessary for the stated purpose",
            "metrics": ["data_minimization.columns_dropped_pct", "data_minimization.examples_filtered_pct"],
        },
        "PIM-3": {
            "name": "Data Retention",
            "description": "Retain data only as long as necessary; ensure secure disposal",
            "metrics": ["data_inventory.retention_policy_present"],
        },
        "PIM-4": {
            "name": "Privacy by Design",
            "description": "Embed privacy into system design and default settings",
            "metrics": ["pii_scan.pii_redaction_enabled", "encryption.at_rest_encryption_enabled"],
        },
    },
    "iso27001": {
        "ISMS-AC": {
            "name": "Access Control",
            "description": "Ensure authorized access and prevent unauthorized access (A.9)",
            "metrics": [
                "access_control.authn_enabled",
                "access_control.authz_model",
                "access_control.default_deny",
            ],
        },
        "ISMS-CRYPTO": {
            "name": "Cryptography",
            "description": "Protect data confidentiality and integrity through encryption (A.10)",
            "metrics": [
                "encryption.at_rest_encryption_enabled",
                "encryption.in_transit_encryption_enabled",
                "encryption.key_rotation_configured",
                "encryption.pqc_enabled",
            ],
        },
        "ISMS-LOG": {
            "name": "Logging & Monitoring",
            "description": "Record and monitor security events (A.12.4)",
            "metrics": [
                "audit_logging.audit_log_enabled",
                "audit_logging.log_integrity_verified",
                "audit_logging.event_coverage_pct",
            ],
        },
        "ISMS-CHANGE": {
            "name": "Change Management",
            "description": "Control changes to systems and ensure traceability (A.12.1, A.14.2)",
            "metrics": [
                "change_management.git_sha",
                "change_management.dirty_tree",
                "change_management.dependency_lockfile_present",
            ],
        },
        "ISMS-SUPPLIER": {
            "name": "Supplier Management",
            "description": "Manage security in supplier relationships (A.15)",
            "metrics": ["secrets_hygiene.scan_passed"],
        },
        "ISMS-BACKUP": {
            "name": "Backup & Recovery",
            "description": "Ensure data availability and recovery capability (A.12.3, A.17)",
            "metrics": ["availability.benchmark_success_rate"],
        },
    },
    "soc2": {
        "TSC-SEC": {
            "name": "Security",
            "description": "Protect against unauthorized access (CC1-CC9)",
            "metrics": [
                "access_control.authn_enabled",
                "encryption.at_rest_encryption_enabled",
                "secrets_hygiene.secrets_found",
            ],
        },
        "TSC-AVAIL": {
            "name": "Availability",
            "description": "Ensure system availability per SLAs (A1)",
            "metrics": [
                "availability.benchmark_success_rate",
                "availability.timeout_retry_configured",
                "availability.health_check_enabled",
            ],
        },
        "TSC-CONF": {
            "name": "Confidentiality",
            "description": "Protect confidential information (C1)",
            "metrics": [
                "encryption.at_rest_encryption_enabled",
                "encryption.in_transit_encryption_enabled",
                "data_inventory.classification_applied",
            ],
        },
        "TSC-PI": {
            "name": "Processing Integrity",
            "description": "Ensure complete, valid, accurate, and timely processing (PI1)",
            "metrics": [
                "processing_integrity.dataset_hash",
                "processing_integrity.validation_passed",
                "processing_integrity.regression_test_passed",
            ],
        },
        "TSC-PRIV": {
            "name": "Privacy",
            "description": "Collect, use, retain, disclose personal information per policy (P1-P8)",
            "metrics": [
                "pii_scan.total_pii_found",
                "pii_scan.pii_redaction_enabled",
                "data_inventory.retention_policy_present",
            ],
        },
    },
}


@dataclass
class ControlEvidence:
    """Evidence for a single control."""
    control_id: str
    control_name: str
    standard: str
    description: str
    metrics: List[str]
    metric_values: Dict[str, Any]
    status: str  # "Evidence Present", "Partial", "Gap"
    artifact_refs: List[str]
    notes: str = ""


@dataclass
class EvidenceReport:
    """Complete evidence report."""
    metadata: Dict[str, Any]
    executive_summary: Dict[str, Any]
    system_scope: Dict[str, Any]
    control_mapping: List[ControlEvidence]
    metrics_appendix: Dict[str, Any]
    limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "executive_summary": self.executive_summary,
            "system_scope": self.system_scope,
            "control_mapping": [asdict(c) for c in self.control_mapping],
            "metrics_appendix": self.metrics_appendix,
            "limitations": self.limitations,
        }


class EvidenceBuilder:
    """Builds compliance evidence reports from collected metrics."""

    def __init__(
        self,
        metrics_file: Path,
        output_dir: Path,
        bench_dir: Optional[Path] = None,
    ):
        self.metrics_file = metrics_file
        self.output_dir = output_dir
        self.bench_dir = bench_dir

        # Load metrics
        with open(metrics_file) as f:
            self.metrics = json.load(f)

        self.git_sha = self.metrics.get("git_sha", "unknown")

    def _get_metric_value(self, metric_path: str) -> Any:
        """Get a metric value by dot-notation path."""
        parts = metric_path.split(".")
        value = self.metrics
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

    def _evaluate_control_status(self, control_id: str, metric_paths: List[str]) -> tuple:
        """Evaluate the status of a control based on its metrics."""
        values = {}
        present_count = 0
        pass_count = 0

        for path in metric_paths:
            value = self._get_metric_value(path)
            values[path] = value

            if value is not None:
                present_count += 1

                # Evaluate pass/fail based on metric type
                if isinstance(value, bool):
                    if value:
                        pass_count += 1
                elif isinstance(value, (int, float)):
                    # Specific handling for different metrics
                    if "secrets_found" in path or "pii_found" in path:
                        if value == 0:
                            pass_count += 1
                    elif "coverage" in path or "success_rate" in path:
                        if value >= 80:
                            pass_count += 1
                    else:
                        pass_count += 1  # Non-zero numeric values are considered present
                elif isinstance(value, str) and value:
                    pass_count += 1

        if present_count == 0:
            status = "Gap"
        elif pass_count == present_count:
            status = "Evidence Present"
        else:
            status = "Partial"

        return status, values

    def build_control_evidence(self) -> List[ControlEvidence]:
        """Build evidence for all controls."""
        evidence_list = []

        for standard, controls in CONTROL_FAMILIES.items():
            for control_id, control_info in controls.items():
                status, values = self._evaluate_control_status(
                    control_id, control_info["metrics"]
                )

                # Determine artifact references
                artifact_refs = []
                if self.metrics_file.exists():
                    artifact_refs.append(str(self.metrics_file))

                evidence = ControlEvidence(
                    control_id=control_id,
                    control_name=control_info["name"],
                    standard=standard.upper(),
                    description=control_info["description"],
                    metrics=control_info["metrics"],
                    metric_values=values,
                    status=status,
                    artifact_refs=artifact_refs,
                )
                evidence_list.append(evidence)

        return evidence_list

    def build_executive_summary(self, control_evidence: List[ControlEvidence]) -> Dict[str, Any]:
        """Build the executive summary."""
        # Count status by standard
        status_by_standard = {}
        for evidence in control_evidence:
            if evidence.standard not in status_by_standard:
                status_by_standard[evidence.standard] = {"present": 0, "partial": 0, "gap": 0}

            if evidence.status == "Evidence Present":
                status_by_standard[evidence.standard]["present"] += 1
            elif evidence.status == "Partial":
                status_by_standard[evidence.standard]["partial"] += 1
            else:
                status_by_standard[evidence.standard]["gap"] += 1

        # Key findings
        pii_found = self._get_metric_value("pii_scan.total_pii_found") or 0
        secrets_found = self._get_metric_value("secrets_hygiene.secrets_found") or 0
        encryption_enabled = self._get_metric_value("encryption.at_rest_encryption_enabled") or False
        audit_enabled = self._get_metric_value("audit_logging.audit_log_enabled") or False

        return {
            "report_date": datetime.now(timezone.utc).isoformat(),
            "git_sha": self.git_sha,
            "environment": self.metrics.get("environment", "unknown"),
            "disclaimer": (
                "This report contains empirical evidence collected from system telemetry. "
                "It does not constitute a certification or compliance attestation. "
                "All metrics are objective, machine-readable, and reproducible."
            ),
            "coverage_summary": status_by_standard,
            "key_findings": {
                "pii_exposure_count": pii_found,
                "secrets_exposed_count": secrets_found,
                "encryption_at_rest_enabled": encryption_enabled,
                "audit_logging_enabled": audit_enabled,
            },
            "total_controls_assessed": len(control_evidence),
            "evidence_present": sum(1 for e in control_evidence if e.status == "Evidence Present"),
            "partial_evidence": sum(1 for e in control_evidence if e.status == "Partial"),
            "gaps_identified": sum(1 for e in control_evidence if e.status == "Gap"),
        }

    def build_system_scope(self) -> Dict[str, Any]:
        """Build the system scope section."""
        return {
            "system_name": "TensorGuard / TensorTrace",
            "version": self.metrics.get("git_sha", "unknown"),
            "description": "Post-quantum secure MLOps platform for privacy-preserving federated learning",
            "components": [
                "Training Pipeline (LoRA fine-tuning)",
                "Inference Serving",
                "Artifact Store",
                "Audit Logging System",
                "Telemetry Collector",
            ],
            "data_flows": [
                "Training data ingestion -> Preprocessing -> Model training -> Adapter storage",
                "Inference request -> Model serving -> Response generation -> Logging",
            ],
            "data_flow_doc": "docs/compliance/DATA_FLOW.md",
            "threat_model_doc": "docs/compliance/THREAT_MODEL.md",
        }

    def build_limitations(self) -> List[str]:
        """Build the limitations section."""
        limitations = [
            "This evidence pack is generated from automated system telemetry and does not replace manual security assessments.",
            "PII detection uses heuristic pattern matching and may have false positives/negatives.",
            "Secrets scanning covers common patterns but may not detect all sensitive data.",
            "Encryption posture is assessed from configuration files; actual implementation verification requires additional testing.",
            "Audit log integrity verification assumes the log system is correctly configured.",
            "This is evidence collection, not a compliance certification or attestation.",
        ]

        # Add dynamic limitations based on gaps
        if not self._get_metric_value("access_control.authn_enabled"):
            limitations.append("Authentication enforcement could not be verified from configuration.")

        if not self._get_metric_value("data_inventory.retention_policy_present"):
            limitations.append("Data retention policy configuration was not found.")

        return limitations

    def build_report(self) -> EvidenceReport:
        """Build the complete evidence report."""
        control_evidence = self.build_control_evidence()

        return EvidenceReport(
            metadata={
                "report_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generator": "TensorGuard Compliance Evidence Builder",
                "git_sha": self.git_sha,
                "metrics_source": str(self.metrics_file),
            },
            executive_summary=self.build_executive_summary(control_evidence),
            system_scope=self.build_system_scope(),
            control_mapping=control_evidence,
            metrics_appendix=self.metrics,
            limitations=self.build_limitations(),
        )

    def generate_markdown(self, report: EvidenceReport) -> str:
        """Generate Markdown report."""
        lines = []

        # Header
        lines.append("# Privacy & Security Compliance Evidence Report")
        lines.append("")
        lines.append(f"> **Generated**: {report.metadata['generated_at']}")
        lines.append(f"> **Git SHA**: `{report.metadata['git_sha']}`")
        lines.append(f"> **Environment**: {report.executive_summary.get('environment', 'N/A')}")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Disclaimer
        lines.append("## Disclaimer")
        lines.append("")
        lines.append(f"_{report.executive_summary['disclaimer']}_")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Total Controls Assessed**: {report.executive_summary['total_controls_assessed']}")
        lines.append("")
        lines.append("| Status | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| Evidence Present | {report.executive_summary['evidence_present']} |")
        lines.append(f"| Partial Evidence | {report.executive_summary['partial_evidence']} |")
        lines.append(f"| Gaps Identified | {report.executive_summary['gaps_identified']} |")
        lines.append("")

        lines.append("### Key Findings")
        lines.append("")
        findings = report.executive_summary["key_findings"]
        lines.append(f"- **PII Exposure Count**: {findings['pii_exposure_count']}")
        lines.append(f"- **Secrets Exposed**: {findings['secrets_exposed_count']}")
        lines.append(f"- **Encryption at Rest**: {'Enabled' if findings['encryption_at_rest_enabled'] else 'Not Verified'}")
        lines.append(f"- **Audit Logging**: {'Enabled' if findings['audit_logging_enabled'] else 'Not Verified'}")
        lines.append("")

        # System Scope
        lines.append("## System Scope")
        lines.append("")
        lines.append(f"**System**: {report.system_scope['system_name']}")
        lines.append(f"**Version**: `{report.system_scope['version']}`")
        lines.append("")
        lines.append(f"_{report.system_scope['description']}_")
        lines.append("")
        lines.append("### Components")
        lines.append("")
        for component in report.system_scope["components"]:
            lines.append(f"- {component}")
        lines.append("")
        lines.append(f"See [{report.system_scope['data_flow_doc']}]({report.system_scope['data_flow_doc']}) for detailed data flow documentation.")
        lines.append("")

        # Control Mapping Matrix
        lines.append("## Control Mapping Matrix")
        lines.append("")

        # Group by standard
        by_standard = {}
        for evidence in report.control_mapping:
            if evidence.standard not in by_standard:
                by_standard[evidence.standard] = []
            by_standard[evidence.standard].append(evidence)

        for standard, evidences in by_standard.items():
            lines.append(f"### {standard}")
            lines.append("")
            lines.append("| Control ID | Control Name | Status | Key Metrics |")
            lines.append("|------------|--------------|--------|-------------|")

            for e in evidences:
                # Format key metric values
                metric_summary = []
                for m, v in list(e.metric_values.items())[:2]:
                    metric_name = m.split(".")[-1]
                    if isinstance(v, bool):
                        metric_summary.append(f"{metric_name}: {'Yes' if v else 'No'}")
                    elif isinstance(v, float):
                        metric_summary.append(f"{metric_name}: {v:.1f}")
                    elif v is not None:
                        metric_summary.append(f"{metric_name}: {v}")

                status_emoji = "+" if e.status == "Evidence Present" else ("~" if e.status == "Partial" else "-")
                lines.append(f"| {e.control_id} | {e.control_name} | {status_emoji} {e.status} | {', '.join(metric_summary) or 'N/A'} |")

            lines.append("")

        # Metrics Appendix
        lines.append("## Metrics Appendix")
        lines.append("")
        lines.append("### Data Inventory")
        lines.append("```json")
        lines.append(json.dumps(self.metrics.get("data_inventory", {}), indent=2))
        lines.append("```")
        lines.append("")

        lines.append("### PII Scan Results")
        lines.append("```json")
        pii_summary = {
            "total_pii_found": self.metrics.get("pii_scan", {}).get("total_pii_found", 0),
            "redaction_enabled": self.metrics.get("pii_scan", {}).get("pii_redaction_enabled", False),
        }
        lines.append(json.dumps(pii_summary, indent=2))
        lines.append("```")
        lines.append("")

        lines.append("### Encryption Posture")
        lines.append("```json")
        lines.append(json.dumps(self.metrics.get("encryption", {}), indent=2))
        lines.append("```")
        lines.append("")

        lines.append("### Audit Logging")
        lines.append("```json")
        lines.append(json.dumps(self.metrics.get("audit_logging", {}), indent=2))
        lines.append("```")
        lines.append("")

        # Limitations
        lines.append("## Limitations & Gaps")
        lines.append("")
        for i, limitation in enumerate(report.limitations, 1):
            lines.append(f"{i}. {limitation}")
        lines.append("")

        # Evidence Artifacts
        lines.append("## Evidence Artifacts")
        lines.append("")
        lines.append("| Artifact | Path | Description |")
        lines.append("|----------|------|-------------|")
        lines.append(f"| Metrics Bundle | `reports/compliance/{self.git_sha}/metrics.json` | Raw metrics data |")
        lines.append(f"| Evidence Report | `reports/compliance/{self.git_sha}/evidence.md` | This report |")
        lines.append(f"| Evidence Data | `reports/compliance/{self.git_sha}/evidence.json` | Structured evidence |")
        lines.append(f"| Control Matrix | `docs/compliance/CONTROL_MATRIX.md` | Control definitions |")
        lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append(f"_Report generated by TensorGuard Compliance Evidence Builder v1.0_")
        lines.append("")

        return "\n".join(lines)

    def save_report(self, report: EvidenceReport) -> tuple:
        """Save both Markdown and JSON reports."""
        output_path = self.output_dir / self.git_sha
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_file = output_path / "evidence.json"
        with open(json_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Save Markdown
        md_file = output_path / "evidence.md"
        md_content = self.generate_markdown(report)
        with open(md_file, 'w') as f:
            f.write(md_content)

        return str(md_file), str(json_file)


def main():
    parser = argparse.ArgumentParser(
        description="Build compliance evidence reports from collected metrics"
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        help="Path to metrics.json file (default: auto-detect from latest git SHA)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/compliance"),
        help="Output directory for evidence reports",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=None,
        help="Path to benchmark reports directory",
    )

    args = parser.parse_args()

    # Auto-detect metrics file if not provided
    if args.metrics_file is None:
        # Get current git SHA
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            git_sha = result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
        except Exception:
            git_sha = "unknown"

        args.metrics_file = args.output_dir / git_sha / "metrics.json"

    if not args.metrics_file.exists():
        print(f"Error: Metrics file not found: {args.metrics_file}")
        print("Run collect_privacy_security_metrics.py first.")
        sys.exit(1)

    builder = EvidenceBuilder(
        metrics_file=args.metrics_file,
        output_dir=args.output_dir,
        bench_dir=args.bench_dir,
    )

    print(f"Building evidence report from: {args.metrics_file}")
    report = builder.build_report()
    md_file, json_file = builder.save_report(report)

    print(f"\nEvidence report generated!")
    print(f"  Markdown: {md_file}")
    print(f"  JSON: {json_file}")
    print(f"\nSummary:")
    print(f"  Controls assessed: {report.executive_summary['total_controls_assessed']}")
    print(f"  Evidence present: {report.executive_summary['evidence_present']}")
    print(f"  Partial evidence: {report.executive_summary['partial_evidence']}")
    print(f"  Gaps identified: {report.executive_summary['gaps_identified']}")


if __name__ == "__main__":
    main()
