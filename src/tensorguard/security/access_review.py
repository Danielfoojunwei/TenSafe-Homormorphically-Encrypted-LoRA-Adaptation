"""
Automated Access Review Module.

Provides periodic access review capabilities:
- Scheduled access reviews
- Orphaned account detection
- Privilege escalation detection
- Access certification workflows
- Compliance reporting

Compliance Requirements:
- SOC 2 CC6.2-CC6.3: Access management
- ISO 27001 A.5.18: Access rights review
- HIPAA §164.312(a): Access control

Usage:
    from tensorguard.security.access_review import (
        AccessReviewManager,
        schedule_access_review,
        detect_orphaned_accounts,
    )
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

ACCESS_REVIEW_PERIOD_DAYS = int(os.getenv("TG_ACCESS_REVIEW_PERIOD_DAYS", "90"))
PRIVILEGED_REVIEW_PERIOD_DAYS = int(os.getenv("TG_PRIVILEGED_REVIEW_PERIOD_DAYS", "30"))
INACTIVE_THRESHOLD_DAYS = int(os.getenv("TG_INACTIVE_THRESHOLD_DAYS", "90"))
AUTO_DISABLE_INACTIVE = os.getenv("TG_AUTO_DISABLE_INACTIVE", "false").lower() == "true"


class ReviewStatus(str, Enum):
    """Access review status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    CERTIFIED = "certified"
    REVOKED = "revoked"
    MODIFIED = "modified"
    EXPIRED = "expired"


class ReviewDecision(str, Enum):
    """Access review decision."""

    APPROVE = "approve"  # Maintain current access
    REVOKE = "revoke"  # Remove access
    MODIFY = "modify"  # Change access level
    ESCALATE = "escalate"  # Escalate for further review


class RiskLevel(str, Enum):
    """Risk level for access."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AccessEntry:
    """An access entry to be reviewed."""

    entry_id: str
    user_id: str
    user_email: str
    user_name: str
    role: str
    resource: str
    permissions: List[str]
    granted_at: datetime
    granted_by: Optional[str] = None
    last_used: Optional[datetime] = None
    risk_level: RiskLevel = RiskLevel.LOW
    is_privileged: bool = False
    review_notes: Optional[str] = None

    def days_since_last_use(self) -> Optional[int]:
        """Get days since last use."""
        if not self.last_used:
            return None
        return (datetime.now(timezone.utc) - self.last_used).days

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "role": self.role,
            "resource": self.resource,
            "permissions": self.permissions,
            "granted_at": self.granted_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "risk_level": self.risk_level.value,
            "is_privileged": self.is_privileged,
            "days_since_last_use": self.days_since_last_use(),
        }


@dataclass
class AccessReviewItem:
    """Individual item in an access review."""

    item_id: str
    access_entry: AccessEntry
    status: ReviewStatus = ReviewStatus.PENDING
    decision: Optional[ReviewDecision] = None
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    justification: Optional[str] = None
    new_permissions: Optional[List[str]] = None  # For MODIFY decisions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "access_entry": self.access_entry.to_dict(),
            "status": self.status.value,
            "decision": self.decision.value if self.decision else None,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }


@dataclass
class AccessReview:
    """An access review campaign."""

    review_id: str
    name: str
    description: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=14)
    )
    items: List[AccessReviewItem] = field(default_factory=list)
    status: ReviewStatus = ReviewStatus.PENDING
    created_by: Optional[str] = None
    completed_at: Optional[datetime] = None
    scope: Optional[str] = None  # "all", "privileged", "resource:X"

    @property
    def total_items(self) -> int:
        """Get total number of items."""
        return len(self.items)

    @property
    def completed_items(self) -> int:
        """Get number of completed items."""
        return sum(
            1 for item in self.items
            if item.status not in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]
        )

    @property
    def progress_percent(self) -> float:
        """Get review progress percentage."""
        if not self.items:
            return 100.0
        return (self.completed_items / self.total_items) * 100

    def is_overdue(self) -> bool:
        """Check if review is overdue."""
        return datetime.now(timezone.utc) > self.due_date and self.status != ReviewStatus.CERTIFIED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat(),
            "status": self.status.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "progress_percent": self.progress_percent,
            "is_overdue": self.is_overdue(),
        }


@dataclass
class AccessAnomaly:
    """Detected access anomaly."""

    anomaly_id: str
    anomaly_type: str
    user_id: str
    description: str
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    risk_level: RiskLevel = RiskLevel.MEDIUM
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "user_id": self.user_id,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
            "risk_level": self.risk_level.value,
            "resolved": self.resolved,
        }


class AccessReviewManager:
    """
    Automated Access Review Manager.

    Handles periodic access reviews and anomaly detection.
    """

    # Privileged roles requiring more frequent review
    PRIVILEGED_ROLES = {"org_admin", "site_admin", "security_admin", "root"}

    def __init__(
        self,
        audit_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None,
        user_provider: Optional[Callable] = None,
        access_provider: Optional[Callable] = None,
        access_revoker: Optional[Callable] = None,
    ):
        """
        Initialize Access Review Manager.

        Args:
            audit_callback: Callback for audit logging
            notification_callback: Callback for notifications
            user_provider: Function to get user list
            access_provider: Function to get access entries for a user
            access_revoker: Function to revoke access
        """
        self._audit_callback = audit_callback
        self._notification_callback = notification_callback
        self._user_provider = user_provider
        self._access_provider = access_provider
        self._access_revoker = access_revoker
        self._reviews: Dict[str, AccessReview] = {}
        self._anomalies: List[AccessAnomaly] = []

    def _audit_log(
        self,
        event: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> None:
        """Log access review event."""
        audit_record = {
            "event": f"access_review.{event}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "compliance_relevant": True,
        }

        if self._audit_callback:
            self._audit_callback("security.access_review." + event, user_id, audit_record)

        logger.info(f"ACCESS_REVIEW: {event} | {details}")

    def create_review(
        self,
        name: str,
        scope: str = "all",
        created_by: Optional[str] = None,
        due_days: int = 14,
        description: Optional[str] = None,
    ) -> AccessReview:
        """
        Create a new access review campaign.

        Args:
            name: Review campaign name
            scope: Review scope ("all", "privileged", "resource:X")
            created_by: Creator user ID
            due_days: Days until due
            description: Optional description

        Returns:
            Created AccessReview
        """
        review = AccessReview(
            review_id=str(uuid4()),
            name=name,
            description=description,
            due_date=datetime.now(timezone.utc) + timedelta(days=due_days),
            created_by=created_by,
            scope=scope,
        )

        # Populate review items
        access_entries = self._get_access_entries(scope)
        for entry in access_entries:
            item = AccessReviewItem(
                item_id=str(uuid4()),
                access_entry=entry,
            )
            review.items.append(item)

        self._reviews[review.review_id] = review

        self._audit_log(
            "created",
            {
                "review_id": review.review_id,
                "name": name,
                "scope": scope,
                "total_items": review.total_items,
            },
            created_by,
        )

        return review

    def _get_access_entries(self, scope: str) -> List[AccessEntry]:
        """Get access entries based on scope."""
        entries = []

        # Use provider if available
        if self._access_provider:
            entries = self._access_provider(scope)
        else:
            # Placeholder - return sample entries
            entries = [
                AccessEntry(
                    entry_id=str(uuid4()),
                    user_id="user_1",
                    user_email="admin@example.com",
                    user_name="Admin User",
                    role="org_admin",
                    resource="platform",
                    permissions=["*"],
                    granted_at=datetime.now(timezone.utc) - timedelta(days=180),
                    is_privileged=True,
                    risk_level=RiskLevel.HIGH,
                ),
                AccessEntry(
                    entry_id=str(uuid4()),
                    user_id="user_2",
                    user_email="operator@example.com",
                    user_name="Operator User",
                    role="operator",
                    resource="training",
                    permissions=["read", "write"],
                    granted_at=datetime.now(timezone.utc) - timedelta(days=90),
                    last_used=datetime.now(timezone.utc) - timedelta(days=30),
                    is_privileged=False,
                    risk_level=RiskLevel.MEDIUM,
                ),
            ]

        # Filter by scope
        if scope == "privileged":
            entries = [e for e in entries if e.is_privileged]
        elif scope.startswith("resource:"):
            resource = scope.split(":")[1]
            entries = [e for e in entries if e.resource == resource]

        return entries

    def submit_decision(
        self,
        review_id: str,
        item_id: str,
        decision: ReviewDecision,
        reviewer_id: str,
        justification: str,
        new_permissions: Optional[List[str]] = None,
    ) -> bool:
        """
        Submit a review decision for an item.

        Args:
            review_id: Review campaign ID
            item_id: Review item ID
            decision: Review decision
            reviewer_id: Reviewer user ID
            justification: Justification for decision
            new_permissions: New permissions for MODIFY decisions

        Returns:
            True if decision recorded
        """
        review = self._reviews.get(review_id)
        if not review:
            return False

        item = next((i for i in review.items if i.item_id == item_id), None)
        if not item:
            return False

        item.decision = decision
        item.reviewer_id = reviewer_id
        item.reviewed_at = datetime.now(timezone.utc)
        item.justification = justification
        item.new_permissions = new_permissions

        # Update status based on decision
        if decision == ReviewDecision.APPROVE:
            item.status = ReviewStatus.CERTIFIED
        elif decision == ReviewDecision.REVOKE:
            item.status = ReviewStatus.REVOKED
            self._execute_revocation(item.access_entry)
        elif decision == ReviewDecision.MODIFY:
            item.status = ReviewStatus.MODIFIED
            self._execute_modification(item.access_entry, new_permissions or [])
        elif decision == ReviewDecision.ESCALATE:
            item.status = ReviewStatus.IN_PROGRESS

        self._audit_log(
            "decision_submitted",
            {
                "review_id": review_id,
                "item_id": item_id,
                "user_id": item.access_entry.user_id,
                "decision": decision.value,
                "justification": justification[:100],
            },
            reviewer_id,
        )

        # Check if review is complete
        if review.completed_items == review.total_items:
            review.status = ReviewStatus.CERTIFIED
            review.completed_at = datetime.now(timezone.utc)
            self._audit_log(
                "completed",
                {"review_id": review_id, "total_items": review.total_items},
            )

        return True

    def _execute_revocation(self, entry: AccessEntry) -> None:
        """Execute access revocation."""
        if self._access_revoker:
            self._access_revoker(entry.user_id, entry.resource, entry.permissions)

        self._audit_log(
            "access_revoked",
            {
                "user_id": entry.user_id,
                "resource": entry.resource,
                "permissions": entry.permissions,
            },
            entry.user_id,
        )

    def _execute_modification(
        self,
        entry: AccessEntry,
        new_permissions: List[str],
    ) -> None:
        """Execute access modification."""
        self._audit_log(
            "access_modified",
            {
                "user_id": entry.user_id,
                "resource": entry.resource,
                "old_permissions": entry.permissions,
                "new_permissions": new_permissions,
            },
            entry.user_id,
        )

    def detect_anomalies(self) -> List[AccessAnomaly]:
        """
        Detect access anomalies.

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Get all access entries
        entries = self._get_access_entries("all")

        for entry in entries:
            # Check for inactive accounts with access
            if entry.last_used:
                days_inactive = entry.days_since_last_use()
                if days_inactive and days_inactive > INACTIVE_THRESHOLD_DAYS:
                    anomaly = AccessAnomaly(
                        anomaly_id=str(uuid4()),
                        anomaly_type="inactive_account",
                        user_id=entry.user_id,
                        description=f"Account inactive for {days_inactive} days but has {entry.role} access",
                        risk_level=RiskLevel.MEDIUM if not entry.is_privileged else RiskLevel.HIGH,
                    )
                    anomalies.append(anomaly)
                    self._anomalies.append(anomaly)

            # Check for privileged access without recent activity
            if entry.is_privileged:
                days_since_grant = (datetime.now(timezone.utc) - entry.granted_at).days
                if days_since_grant > PRIVILEGED_REVIEW_PERIOD_DAYS and entry.last_used is None:
                    anomaly = AccessAnomaly(
                        anomaly_id=str(uuid4()),
                        anomaly_type="unused_privileged_access",
                        user_id=entry.user_id,
                        description=f"Privileged access ({entry.role}) never used since grant {days_since_grant} days ago",
                        risk_level=RiskLevel.HIGH,
                    )
                    anomalies.append(anomaly)
                    self._anomalies.append(anomaly)

        if anomalies:
            self._audit_log(
                "anomalies_detected",
                {
                    "count": len(anomalies),
                    "types": list(set(a.anomaly_type for a in anomalies)),
                },
            )

        return anomalies

    def get_overdue_reviews(self) -> List[AccessReview]:
        """Get all overdue reviews."""
        return [r for r in self._reviews.values() if r.is_overdue()]

    def get_pending_reviews(self) -> List[AccessReview]:
        """Get all pending reviews."""
        return [
            r for r in self._reviews.values()
            if r.status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS]
        ]

    def get_review(self, review_id: str) -> Optional[AccessReview]:
        """Get a specific review."""
        return self._reviews.get(review_id)

    def get_unresolved_anomalies(self) -> List[AccessAnomaly]:
        """Get unresolved anomalies."""
        return [a for a in self._anomalies if not a.resolved]

    def resolve_anomaly(
        self,
        anomaly_id: str,
        resolved_by: str,
        resolution_notes: str,
    ) -> bool:
        """
        Mark an anomaly as resolved.

        Args:
            anomaly_id: Anomaly ID
            resolved_by: Resolver user ID
            resolution_notes: Resolution notes

        Returns:
            True if resolved
        """
        anomaly = next((a for a in self._anomalies if a.anomaly_id == anomaly_id), None)
        if not anomaly:
            return False

        anomaly.resolved = True
        anomaly.resolved_at = datetime.now(timezone.utc)
        anomaly.resolved_by = resolved_by
        anomaly.resolution_notes = resolution_notes

        self._audit_log(
            "anomaly_resolved",
            {
                "anomaly_id": anomaly_id,
                "anomaly_type": anomaly.anomaly_type,
                "resolved_by": resolved_by,
            },
        )

        return True

    def generate_compliance_report(self) -> Dict[str, Any]:
        """
        Generate access review compliance report.

        Returns:
            Compliance report data
        """
        total_reviews = len(self._reviews)
        completed_reviews = sum(
            1 for r in self._reviews.values()
            if r.status == ReviewStatus.CERTIFIED
        )
        overdue_reviews = len(self.get_overdue_reviews())

        total_anomalies = len(self._anomalies)
        resolved_anomalies = sum(1 for a in self._anomalies if a.resolved)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "review_period_days": ACCESS_REVIEW_PERIOD_DAYS,
            "privileged_review_period_days": PRIVILEGED_REVIEW_PERIOD_DAYS,
            "reviews": {
                "total": total_reviews,
                "completed": completed_reviews,
                "overdue": overdue_reviews,
                "completion_rate": (completed_reviews / total_reviews * 100) if total_reviews else 100,
            },
            "anomalies": {
                "total": total_anomalies,
                "resolved": resolved_anomalies,
                "unresolved": total_anomalies - resolved_anomalies,
            },
            "compliance_status": "compliant" if overdue_reviews == 0 else "non_compliant",
        }

        self._audit_log("compliance_report_generated", report)

        return report


# Singleton instance
_access_review_manager: Optional[AccessReviewManager] = None


def get_access_review_manager() -> AccessReviewManager:
    """Get or create the default access review manager."""
    global _access_review_manager
    if _access_review_manager is None:
        _access_review_manager = AccessReviewManager()
    return _access_review_manager
