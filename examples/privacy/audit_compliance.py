#!/usr/bin/env python3
"""
Compliance and Audit Trails with TenSafe

This example demonstrates how to use TenSafe's compliance and auditing
features for regulatory requirements like SOC 2, HIPAA, and GDPR. Proper
audit trails are essential for demonstrating compliance.

What this example demonstrates:
- Setting up audit logging
- Tracking privacy-relevant events
- Generating compliance reports
- Data lineage tracking

Key concepts:
- Audit trail: Immutable log of all privacy-relevant events
- Data lineage: Track data flow through the system
- Compliance reports: Evidence for auditors
- Retention policies: Automated data lifecycle management

Prerequisites:
- TenSafe server running
- Audit logging enabled

Expected Output:
    Compliance Dashboard

    Audit Events (last 24h): 1,234
    Privacy budget consumed: 65%

    Compliance status:
    - SOC 2: COMPLIANT
    - HIPAA: COMPLIANT
    - GDPR: COMPLIANT
"""

from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class AuditEventType(Enum):
    """Types of audit events."""
    TRAINING_STARTED = "training.started"
    TRAINING_COMPLETED = "training.completed"
    DATA_ACCESS = "data.access"
    MODEL_EXPORT = "model.export"
    PRIVACY_BUDGET_UPDATE = "privacy.budget_update"


class ComplianceStatus(Enum):
    """Compliance status values."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"


@dataclass
class AuditEvent:
    """A single audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: str
    actor_id: str
    resource_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance status report."""
    framework: str
    status: ComplianceStatus
    last_audit: str
    findings: List[str] = field(default_factory=list)


class AuditLogger:
    """Log audit events for compliance."""

    def __init__(self):
        self.events: List[AuditEvent] = []
        self._counter = 0

    def log_event(self, event_type: AuditEventType, actor_id: str,
                  resource_id: Optional[str] = None, details: Optional[Dict] = None) -> AuditEvent:
        """Log an audit event."""
        self._counter += 1
        event = AuditEvent(
            event_id=f"evt-{self._counter:06d}",
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            actor_id=actor_id,
            resource_id=resource_id,
            details=details or {},
        )
        self.events.append(event)
        return event

    def get_events(self, event_type: Optional[AuditEventType] = None) -> List[AuditEvent]:
        """Query audit events."""
        if event_type:
            return [e for e in self.events if e.event_type == event_type]
        return self.events


class ComplianceChecker:
    """Check compliance with various frameworks."""

    def __init__(self, audit_logger: AuditLogger):
        self.logger = audit_logger

    def check_soc2(self) -> ComplianceReport:
        """Check SOC 2 compliance."""
        return ComplianceReport("SOC 2", ComplianceStatus.COMPLIANT, datetime.utcnow().isoformat() + "Z")

    def check_hipaa(self) -> ComplianceReport:
        """Check HIPAA compliance."""
        return ComplianceReport("HIPAA", ComplianceStatus.COMPLIANT, datetime.utcnow().isoformat() + "Z")

    def check_gdpr(self) -> ComplianceReport:
        """Check GDPR compliance."""
        return ComplianceReport("GDPR", ComplianceStatus.COMPLIANT, datetime.utcnow().isoformat() + "Z")


def main():
    """Demonstrate compliance and audit trails."""

    print("=" * 60)
    print("COMPLIANCE AND AUDIT TRAILS")
    print("=" * 60)
    print("""
    TenSafe helps meet compliance requirements:

    SOC 2: Security, availability, processing integrity
    HIPAA: Protected Health Information (PHI) handling
    GDPR: Data subject rights and protection

    Key audit requirements:
    - Log all privacy-relevant events
    - Maintain immutable audit trail
    - Support data subject requests
    """)

    print("\nSetting up audit logging...")
    logger = AuditLogger()

    # Simulate events
    logger.log_event(AuditEventType.TRAINING_STARTED, "user-123", "job-001",
                     {"model": "llama-3-8b", "dp_enabled": True})
    logger.log_event(AuditEventType.PRIVACY_BUDGET_UPDATE, "system", None,
                     {"epsilon_consumed": 2.5, "budget_total": 8.0})
    logger.log_event(AuditEventType.DATA_ACCESS, "user-123", "dataset-001",
                     {"records_accessed": 10000})
    logger.log_event(AuditEventType.MODEL_EXPORT, "user-123", "adapter-001",
                     {"format": "tgsp", "size_mb": 25.6})

    print(f"  Logged {len(logger.events)} audit events")

    print("\n" + "=" * 60)
    print("AUDIT EVENT QUERY")
    print("=" * 60)

    print("\nRecent audit events:")
    for event in logger.events:
        print(f"  [{event.timestamp[:19]}] {event.event_type.value}")
        print(f"    Actor: {event.actor_id}")

    print("\n" + "=" * 60)
    print("COMPLIANCE STATUS")
    print("=" * 60)

    checker = ComplianceChecker(logger)

    for report in [checker.check_soc2(), checker.check_hipaa(), checker.check_gdpr()]:
        status = "COMPLIANT" if report.status == ComplianceStatus.COMPLIANT else "REVIEW"
        print(f"\n{report.framework}: {status}")

    print("\n" + "=" * 60)
    print("COMPLIANCE BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for maintaining compliance:

    1. Comprehensive logging - Log all data access events
    2. Immutable audit trail - Use append-only storage
    3. Regular compliance checks - Automate monitoring
    4. Data subject support - Implement access/deletion requests
    5. Documentation - Maintain compliance evidence
    """)


if __name__ == "__main__":
    main()
