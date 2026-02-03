"""
Data Subject Access Request (DSAR) Module.

Provides automated handling of data subject rights requests:
- Right to Access (Article 15 GDPR)
- Right to Rectification (Article 16)
- Right to Erasure (Article 17)
- Right to Restrict Processing (Article 18)
- Right to Data Portability (Article 20)
- Right to Object (Article 21)

Compliance Requirements:
- GDPR Articles 12-22
- ISO 27701 A.7.3.6: Access, correction, and/or erasure
- SOC 2 TSC Privacy (P1-P8)

Usage:
    from tensorguard.security.dsar import (
        DSARManager,
        create_access_request,
        create_deletion_request,
    )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# GDPR requires response within 30 days (extendable to 90 for complex requests)
DSAR_RESPONSE_DAYS = int(os.getenv("TG_DSAR_RESPONSE_DAYS", "30"))
DSAR_EXTENSION_DAYS = int(os.getenv("TG_DSAR_EXTENSION_DAYS", "60"))
AUTO_VERIFY_IDENTITY = os.getenv("TG_DSAR_AUTO_VERIFY", "false").lower() == "true"


class DSARType(str, Enum):
    """Types of Data Subject Access Requests."""

    ACCESS = "access"  # Right to access
    RECTIFICATION = "rectification"  # Right to rectification
    ERASURE = "erasure"  # Right to erasure (right to be forgotten)
    RESTRICTION = "restriction"  # Right to restrict processing
    PORTABILITY = "portability"  # Right to data portability
    OBJECTION = "objection"  # Right to object


class DSARStatus(str, Enum):
    """DSAR processing status."""

    SUBMITTED = "submitted"
    IDENTITY_VERIFICATION = "identity_verification"
    IN_PROGRESS = "in_progress"
    EXTENDED = "extended"
    COMPLETED = "completed"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"


class DenialReason(str, Enum):
    """Reasons for denying a DSAR."""

    IDENTITY_NOT_VERIFIED = "identity_not_verified"
    EXCESSIVE_REQUESTS = "excessive_requests"
    LEGAL_OBLIGATION = "legal_obligation"  # Legal requirement to retain
    PUBLIC_INTEREST = "public_interest"
    LEGAL_CLAIMS = "legal_claims"  # Needed for legal claims
    NOT_APPLICABLE = "not_applicable"  # Request type not applicable


@dataclass
class DataCategory:
    """Category of personal data."""

    category_id: str
    name: str
    description: str
    data_sources: List[str]
    retention_days: int
    legal_basis: str
    can_delete: bool = True
    can_export: bool = True


@dataclass
class DSARRequest:
    """Data Subject Access Request record."""

    request_id: str
    subject_id: str  # Data subject identifier (user ID or email)
    subject_email: str
    request_type: DSARType
    status: DSARStatus = DSARStatus.SUBMITTED
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=DSAR_RESPONSE_DAYS))
    identity_verified: bool = False
    identity_verified_at: Optional[datetime] = None
    identity_verification_method: Optional[str] = None
    specific_data_requested: Optional[List[str]] = None
    reason: Optional[str] = None
    completed_at: Optional[datetime] = None
    denial_reason: Optional[DenialReason] = None
    denial_explanation: Optional[str] = None
    exported_data_path: Optional[str] = None
    deleted_data_categories: List[str] = field(default_factory=list)
    processing_notes: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    extended: bool = False
    extension_reason: Optional[str] = None

    def is_overdue(self) -> bool:
        """Check if request is overdue."""
        return datetime.now(timezone.utc) > self.due_date and self.status not in [
            DSARStatus.COMPLETED,
            DSARStatus.DENIED,
            DSARStatus.WITHDRAWN,
        ]

    def days_remaining(self) -> int:
        """Get days remaining until due date."""
        delta = self.due_date - datetime.now(timezone.utc)
        return max(0, delta.days)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "subject_id": self.subject_id,
            "subject_email": self.subject_email,
            "request_type": self.request_type.value,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "due_date": self.due_date.isoformat(),
            "identity_verified": self.identity_verified,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "days_remaining": self.days_remaining(),
            "is_overdue": self.is_overdue(),
        }


class DSARManager:
    """
    Data Subject Access Request Manager.

    Handles the full lifecycle of DSARs including:
    - Request submission and tracking
    - Identity verification
    - Data collection and export
    - Secure deletion
    - Audit logging
    """

    # Default data categories for the platform
    DEFAULT_DATA_CATEGORIES = [
        DataCategory(
            category_id="user_profile",
            name="User Profile Data",
            description="Basic user information (name, email, role)",
            data_sources=["users_table"],
            retention_days=365,
            legal_basis="contract",
            can_delete=True,
            can_export=True,
        ),
        DataCategory(
            category_id="authentication",
            name="Authentication Data",
            description="Login history, tokens, MFA data",
            data_sources=["auth_logs", "mfa_table"],
            retention_days=90,
            legal_basis="contract",
            can_delete=True,
            can_export=True,
        ),
        DataCategory(
            category_id="training_data",
            name="Training Data Contributions",
            description="Data provided for model training",
            data_sources=["training_datasets"],
            retention_days=365,
            legal_basis="consent",
            can_delete=True,
            can_export=True,
        ),
        DataCategory(
            category_id="inference_logs",
            name="Inference Request Logs",
            description="API requests and responses",
            data_sources=["inference_logs"],
            retention_days=30,
            legal_basis="legitimate_interest",
            can_delete=True,
            can_export=True,
        ),
        DataCategory(
            category_id="audit_logs",
            name="Security Audit Logs",
            description="Security and compliance audit trail",
            data_sources=["audit_logs"],
            retention_days=365,
            legal_basis="legal_obligation",
            can_delete=False,  # Required for compliance
            can_export=True,
        ),
    ]

    def __init__(
        self,
        data_categories: Optional[List[DataCategory]] = None,
        audit_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None,
        data_collector: Optional[Callable] = None,
        data_deleter: Optional[Callable] = None,
    ):
        """
        Initialize DSAR Manager.

        Args:
            data_categories: Custom data categories
            audit_callback: Callback for audit logging
            notification_callback: Callback for notifications
            data_collector: Function to collect data for a subject
            data_deleter: Function to delete data for a subject
        """
        self._categories = {
            c.category_id: c
            for c in (data_categories or self.DEFAULT_DATA_CATEGORIES)
        }
        self._audit_callback = audit_callback
        self._notification_callback = notification_callback
        self._data_collector = data_collector
        self._data_deleter = data_deleter
        self._requests: Dict[str, DSARRequest] = {}

    def _audit_log(
        self,
        event: str,
        request_id: str,
        subject_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log DSAR event for audit trail."""
        audit_record = {
            "event": f"dsar.{event}",
            "request_id": request_id,
            "subject_id": subject_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "compliance_relevant": True,
        }

        if self._audit_callback:
            self._audit_callback("privacy.dsar." + event, subject_id, audit_record)

        logger.info(f"DSAR: {event} | request={request_id} | subject={subject_id}")

    def _notify(self, event: str, request: DSARRequest) -> None:
        """Send notification for DSAR event."""
        if self._notification_callback:
            self._notification_callback(
                event=f"dsar.{event}",
                email=request.subject_email,
                data=request.to_dict(),
            )

    def submit_request(
        self,
        subject_id: str,
        subject_email: str,
        request_type: DSARType,
        specific_data: Optional[List[str]] = None,
        reason: Optional[str] = None,
    ) -> DSARRequest:
        """
        Submit a new DSAR.

        Args:
            subject_id: Data subject identifier
            subject_email: Data subject email
            request_type: Type of request
            specific_data: Optional list of specific data categories
            reason: Optional reason for request

        Returns:
            Created DSARRequest
        """
        request = DSARRequest(
            request_id=str(uuid4()),
            subject_id=subject_id,
            subject_email=subject_email,
            request_type=request_type,
            specific_data_requested=specific_data,
            reason=reason,
        )

        self._requests[request.request_id] = request

        self._audit_log(
            "submitted",
            request.request_id,
            subject_id,
            {
                "type": request_type.value,
                "due_date": request.due_date.isoformat(),
            },
        )

        self._notify("submitted", request)

        return request

    def verify_identity(
        self,
        request_id: str,
        verification_method: str,
        verified_by: Optional[str] = None,
    ) -> bool:
        """
        Verify data subject identity.

        Args:
            request_id: Request to verify
            verification_method: How identity was verified
            verified_by: Who performed verification

        Returns:
            True if verification recorded
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        request.identity_verified = True
        request.identity_verified_at = datetime.now(timezone.utc)
        request.identity_verification_method = verification_method
        request.status = DSARStatus.IN_PROGRESS

        self._audit_log(
            "identity_verified",
            request_id,
            request.subject_id,
            {
                "method": verification_method,
                "verified_by": verified_by,
            },
        )

        return True

    def extend_deadline(
        self,
        request_id: str,
        reason: str,
        extended_by: str,
    ) -> bool:
        """
        Extend DSAR deadline (allowed by GDPR for complex requests).

        Args:
            request_id: Request to extend
            reason: Reason for extension
            extended_by: Who approved extension

        Returns:
            True if extension granted
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.extended:
            return False  # Can only extend once

        original_due = request.due_date
        request.due_date = request.due_date + timedelta(days=DSAR_EXTENSION_DAYS)
        request.extended = True
        request.extension_reason = reason
        request.status = DSARStatus.EXTENDED
        request.processing_notes.append(
            f"Extended by {extended_by}: {reason}"
        )

        self._audit_log(
            "extended",
            request_id,
            request.subject_id,
            {
                "original_due": original_due.isoformat(),
                "new_due": request.due_date.isoformat(),
                "reason": reason,
                "extended_by": extended_by,
            },
        )

        self._notify("extended", request)

        return True

    def collect_subject_data(
        self,
        request_id: str,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Collect all data for a data subject.

        Args:
            request_id: DSAR request ID
            categories: Specific categories to collect (default: all)

        Returns:
            Dictionary of collected data by category
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if not request.identity_verified:
            raise ValueError("Identity must be verified before data collection")

        categories = categories or request.specific_data_requested or list(self._categories.keys())
        collected_data = {}

        for cat_id in categories:
            category = self._categories.get(cat_id)
            if not category or not category.can_export:
                continue

            # Use custom collector or placeholder
            if self._data_collector:
                data = self._data_collector(
                    subject_id=request.subject_id,
                    category=category,
                )
            else:
                # Placeholder - in production, connect to actual data sources
                data = {
                    "category": category.name,
                    "description": category.description,
                    "legal_basis": category.legal_basis,
                    "retention_days": category.retention_days,
                    "data": f"[Data from {category.data_sources}]",
                }

            collected_data[cat_id] = data

        self._audit_log(
            "data_collected",
            request_id,
            request.subject_id,
            {"categories": list(collected_data.keys())},
        )

        return collected_data

    def export_subject_data(
        self,
        request_id: str,
        format: str = "json",
    ) -> str:
        """
        Export data subject's data in portable format.

        Args:
            request_id: DSAR request ID
            format: Export format (json, csv)

        Returns:
            Path to exported data file
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.request_type not in [DSARType.ACCESS, DSARType.PORTABILITY]:
            raise ValueError("Export only valid for access/portability requests")

        # Collect data
        data = self.collect_subject_data(request_id)

        # Export to file (in production, use secure storage)
        export_dir = os.getenv("TG_DSAR_EXPORT_DIR", "/tmp/dsar_exports")
        os.makedirs(export_dir, exist_ok=True)

        filename = f"dsar_{request_id}_{request.subject_id}.{format}"
        filepath = os.path.join(export_dir, filename)

        if format == "json":
            export_content = {
                "request_id": request_id,
                "subject_id": request.subject_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            with open(filepath, "w") as f:
                json.dump(export_content, f, indent=2, default=str)

        request.exported_data_path = filepath

        self._audit_log(
            "data_exported",
            request_id,
            request.subject_id,
            {"format": format, "categories": list(data.keys())},
        )

        return filepath

    def process_erasure_request(
        self,
        request_id: str,
        categories: Optional[List[str]] = None,
        processed_by: str = "system",
    ) -> Dict[str, bool]:
        """
        Process erasure (right to be forgotten) request.

        Args:
            request_id: DSAR request ID
            categories: Specific categories to delete (default: all deletable)
            processed_by: Who processed the deletion

        Returns:
            Dictionary of deletion results by category
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.request_type != DSARType.ERASURE:
            raise ValueError("This is not an erasure request")

        if not request.identity_verified:
            raise ValueError("Identity must be verified before erasure")

        categories = categories or list(self._categories.keys())
        results = {}

        for cat_id in categories:
            category = self._categories.get(cat_id)
            if not category:
                continue

            if not category.can_delete:
                results[cat_id] = False
                request.processing_notes.append(
                    f"Cannot delete {category.name}: {category.legal_basis}"
                )
                continue

            # Use custom deleter or placeholder
            if self._data_deleter:
                success = self._data_deleter(
                    subject_id=request.subject_id,
                    category=category,
                )
            else:
                # Placeholder - in production, connect to actual deletion
                success = True

            results[cat_id] = success
            if success:
                request.deleted_data_categories.append(cat_id)

        self._audit_log(
            "erasure_processed",
            request_id,
            request.subject_id,
            {
                "deleted": request.deleted_data_categories,
                "retained": [k for k, v in results.items() if not v],
                "processed_by": processed_by,
            },
        )

        return results

    def complete_request(
        self,
        request_id: str,
        completed_by: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Mark DSAR as completed.

        Args:
            request_id: Request to complete
            completed_by: Who completed it
            notes: Optional completion notes

        Returns:
            True if completed successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        request.status = DSARStatus.COMPLETED
        request.completed_at = datetime.now(timezone.utc)
        if notes:
            request.processing_notes.append(f"Completed: {notes}")

        self._audit_log(
            "completed",
            request_id,
            request.subject_id,
            {
                "completed_by": completed_by,
                "days_taken": (request.completed_at - request.submitted_at).days,
                "was_overdue": request.is_overdue(),
            },
        )

        self._notify("completed", request)

        return True

    def deny_request(
        self,
        request_id: str,
        reason: DenialReason,
        explanation: str,
        denied_by: str,
    ) -> bool:
        """
        Deny a DSAR with documented reason.

        Args:
            request_id: Request to deny
            reason: Predefined denial reason
            explanation: Detailed explanation
            denied_by: Who denied the request

        Returns:
            True if denied successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        request.status = DSARStatus.DENIED
        request.denial_reason = reason
        request.denial_explanation = explanation
        request.completed_at = datetime.now(timezone.utc)

        self._audit_log(
            "denied",
            request_id,
            request.subject_id,
            {
                "reason": reason.value,
                "explanation": explanation,
                "denied_by": denied_by,
            },
        )

        self._notify("denied", request)

        return True

    def get_overdue_requests(self) -> List[DSARRequest]:
        """Get all overdue requests."""
        return [r for r in self._requests.values() if r.is_overdue()]

    def get_pending_requests(self) -> List[DSARRequest]:
        """Get all pending requests."""
        return [
            r for r in self._requests.values()
            if r.status in [
                DSARStatus.SUBMITTED,
                DSARStatus.IDENTITY_VERIFICATION,
                DSARStatus.IN_PROGRESS,
                DSARStatus.EXTENDED,
            ]
        ]

    def get_request(self, request_id: str) -> Optional[DSARRequest]:
        """Get a specific request."""
        return self._requests.get(request_id)

    def get_subject_requests(self, subject_id: str) -> List[DSARRequest]:
        """Get all requests for a data subject."""
        return [r for r in self._requests.values() if r.subject_id == subject_id]


# Singleton instance
_dsar_manager: Optional[DSARManager] = None


def get_dsar_manager() -> DSARManager:
    """Get or create the default DSAR manager."""
    global _dsar_manager
    if _dsar_manager is None:
        _dsar_manager = DSARManager()
    return _dsar_manager
