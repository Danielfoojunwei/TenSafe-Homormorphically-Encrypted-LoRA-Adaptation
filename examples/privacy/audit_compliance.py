"""
Audit and Compliance Example

Demonstrates TenSafe's audit logging and compliance features for
regulatory requirements (SOC 2, HIPAA, GDPR).

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python audit_compliance.py
"""

import os
from datetime import datetime, timedelta
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))

    # Query audit logs
    print("=== Audit Log Query ===")
    logs = client.audit.get_logs(
        limit=10,
        operations=["create_training_client", "forward_backward", "save_state"],
    )

    for log in logs:
        print(f"[{log.timestamp}] {log.operation}")
        print(f"  Tenant: {log.tenant_id}")
        print(f"  Request hash: {log.request_hash[:32]}...")
        print(f"  Success: {log.success}")
        if log.dp_metrics:
            print(f"  DP: ε={log.dp_metrics.get('total_epsilon', 'N/A')}")
        print()

    # Verify audit chain integrity
    print("=== Chain Integrity Verification ===")
    verification = client.audit.verify_chain(
        start_date=datetime.utcnow() - timedelta(days=7),
        end_date=datetime.utcnow(),
    )
    print(f"Chain valid: {verification.is_valid}")
    print(f"Entries verified: {verification.entries_count}")
    if not verification.is_valid:
        print(f"First invalid entry: {verification.first_invalid_id}")

    # Generate compliance report
    print("\n=== Compliance Report ===")
    report = client.audit.generate_compliance_report(
        framework="SOC2",
        period_start=datetime.utcnow() - timedelta(days=30),
        period_end=datetime.utcnow(),
    )

    print(f"Framework: {report.framework}")
    print(f"Period: {report.period_start} to {report.period_end}")
    print(f"Controls evaluated: {len(report.controls)}")
    print(f"Passed: {report.controls_passed}")
    print(f"Failed: {report.controls_failed}")

    # Export audit logs for external SIEM
    print("\n=== Export for SIEM ===")
    export = client.audit.export(
        format="json",  # or "csv", "splunk"
        start_date=datetime.utcnow() - timedelta(days=1),
        include_request_bodies=False,  # Privacy: exclude raw data
    )
    print(f"Exported {export.record_count} records to {export.download_url}")


if __name__ == "__main__":
    main()
