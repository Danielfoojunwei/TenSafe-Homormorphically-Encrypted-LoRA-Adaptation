"""
Emergency Access (Break-Glass) Module.

Provides emergency access procedures for HIPAA compliance:
- Break-glass access for emergency situations
- Time-limited emergency tokens
- Enhanced audit logging
- Mandatory post-incident review
- Automatic access revocation

Compliance Requirements:
- HIPAA §164.312(a)(2)(ii): Emergency access procedure (REQUIRED)
- SOC 2 CC6.1: Logical access controls
- ISO 27001 A.5.24: Information security incident management

Usage:
    from tensorguard.security.emergency_access import (
        EmergencyAccessManager,
        request_emergency_access,
        verify_emergency_token,
    )
"""

import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Emergency access defaults
EMERGENCY_TOKEN_DURATION_MINUTES = int(os.getenv("TG_EMERGENCY_TOKEN_DURATION", "60"))
MAX_EMERGENCY_DURATION_HOURS = int(os.getenv("TG_MAX_EMERGENCY_DURATION", "4"))
REQUIRE_JUSTIFICATION = os.getenv("TG_REQUIRE_EMERGENCY_JUSTIFICATION", "true").lower() == "true"
REQUIRE_INCIDENT_ID = os.getenv("TG_REQUIRE_EMERGENCY_INCIDENT_ID", "true").lower() == "true"
AUTO_NOTIFY_ADMINS = os.getenv("TG_EMERGENCY_NOTIFY_ADMINS", "true").lower() == "true"

# Break-glass approvers (user IDs or roles)
BREAK_GLASS_APPROVERS = os.getenv("TG_BREAK_GLASS_APPROVERS", "org_admin").split(",")


class EmergencyAccessReason(str, Enum):
    """Predefined emergency access reasons (HIPAA-compliant)."""

    PATIENT_CARE = "patient_care"  # Direct patient care emergency
    SYSTEM_FAILURE = "system_failure"  # System failure requiring immediate access
    SECURITY_INCIDENT = "security_incident"  # Active security incident response
    DISASTER_RECOVERY = "disaster_recovery"  # Disaster recovery operations
    REGULATORY_REQUEST = "regulatory_request"  # Regulatory/legal requirement
    OTHER = "other"  # Requires detailed justification


class EmergencyAccessStatus(str, Enum):
    """Emergency access request status."""

    REQUESTED = "requested"
    APPROVED = "approved"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    REVIEW_PENDING = "review_pending"
    REVIEWED = "reviewed"


@dataclass
class EmergencyAccessRequest:
    """Emergency access request record."""

    request_id: str
    user_id: str
    reason: EmergencyAccessReason
    justification: str
    status: EmergencyAccessStatus = EmergencyAccessStatus.REQUESTED
    incident_id: Optional[str] = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revocation_reason: Optional[str] = None
    accessed_resources: List[str] = field(default_factory=list)
    review_completed: bool = False
    review_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    def is_active(self) -> bool:
        """Check if emergency access is currently active."""
        if self.status != EmergencyAccessStatus.ACTIVE:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "reason": self.reason.value,
            "justification": self.justification,
            "status": self.status.value,
            "incident_id": self.incident_id,
            "requested_at": self.requested_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approved_by": self.approved_by,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "accessed_resources": self.accessed_resources,
            "review_completed": self.review_completed,
        }


@dataclass
class EmergencyToken:
    """Emergency access token."""

    token_id: str
    request_id: str
    user_id: str
    token_hash: str
    created_at: datetime
    expires_at: datetime
    scope: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return datetime.now(timezone.utc) > self.expires_at


class EmergencyAccessManager:
    """
    Emergency Access (Break-Glass) Manager.

    Handles emergency access requests, approval, and audit for HIPAA compliance.

    Key Features:
    - Pre-authenticated emergency access for critical situations
    - Time-limited access tokens
    - Enhanced audit logging (all access logged)
    - Mandatory post-incident review
    - Automatic expiration and revocation
    """

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        audit_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None,
    ):
        """
        Initialize Emergency Access Manager.

        Args:
            redis_client: Optional Redis client for distributed storage
            audit_callback: Callback for audit logging
            notification_callback: Callback for admin notifications
        """
        self._redis = redis_client
        self._audit_callback = audit_callback
        self._notification_callback = notification_callback
        self._requests: Dict[str, EmergencyAccessRequest] = {}
        self._tokens: Dict[str, EmergencyToken] = {}
        self._active_by_user: Dict[str, str] = {}  # user_id -> request_id

    def _audit_log(
        self,
        event: str,
        user_id: str,
        request_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log emergency access event for audit trail."""
        audit_record = {
            "event": event,
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "compliance_relevant": True,  # Always compliance-relevant
        }

        if self._audit_callback:
            self._audit_callback("emergency_access." + event, user_id, audit_record)

        # Always log to system logger with CRITICAL level for visibility
        logger.critical(f"EMERGENCY_ACCESS: {event} | user={user_id} | request={request_id} | {details}")

    def _notify_admins(self, event: str, request: EmergencyAccessRequest) -> None:
        """Notify administrators of emergency access events."""
        if AUTO_NOTIFY_ADMINS and self._notification_callback:
            self._notification_callback(
                event=f"emergency_access.{event}",
                data={
                    "request_id": request.request_id,
                    "user_id": request.user_id,
                    "reason": request.reason.value,
                    "justification": request.justification,
                },
            )

    def request_emergency_access(
        self,
        user_id: str,
        reason: EmergencyAccessReason,
        justification: str,
        incident_id: Optional[str] = None,
        duration_minutes: int = EMERGENCY_TOKEN_DURATION_MINUTES,
    ) -> EmergencyAccessRequest:
        """
        Request emergency (break-glass) access.

        Args:
            user_id: User requesting access
            reason: Predefined reason for emergency access
            justification: Detailed justification (free text)
            incident_id: Optional incident ticket ID
            duration_minutes: Requested access duration

        Returns:
            EmergencyAccessRequest record

        Raises:
            ValueError: If required fields are missing or validation fails
        """
        # Validate inputs
        if REQUIRE_JUSTIFICATION and len(justification.strip()) < 20:
            raise ValueError("Justification must be at least 20 characters")

        if REQUIRE_INCIDENT_ID and not incident_id:
            raise ValueError("Incident ID is required for emergency access")

        if duration_minutes > MAX_EMERGENCY_DURATION_HOURS * 60:
            raise ValueError(f"Duration cannot exceed {MAX_EMERGENCY_DURATION_HOURS} hours")

        # Check for existing active request
        if user_id in self._active_by_user:
            existing_id = self._active_by_user[user_id]
            existing = self._requests.get(existing_id)
            if existing and existing.is_active():
                raise ValueError(f"Active emergency access already exists: {existing_id}")

        # Create request
        request = EmergencyAccessRequest(
            request_id=str(uuid4()),
            user_id=user_id,
            reason=reason,
            justification=justification,
            incident_id=incident_id,
        )

        self._requests[request.request_id] = request

        self._audit_log(
            "requested",
            user_id,
            request.request_id,
            {
                "reason": reason.value,
                "justification": justification[:100],
                "incident_id": incident_id,
                "duration_requested": duration_minutes,
            },
        )

        self._notify_admins("requested", request)

        return request

    def approve_emergency_access(
        self,
        request_id: str,
        approver_id: str,
        duration_minutes: int = EMERGENCY_TOKEN_DURATION_MINUTES,
        scope: Optional[List[str]] = None,
    ) -> EmergencyToken:
        """
        Approve emergency access request and issue token.

        Args:
            request_id: Request to approve
            approver_id: User approving the request
            duration_minutes: Approved access duration
            scope: Optional list of allowed resources/actions

        Returns:
            EmergencyToken for access

        Raises:
            ValueError: If request not found or cannot be approved
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.status not in [EmergencyAccessStatus.REQUESTED]:
            raise ValueError(f"Request cannot be approved in status: {request.status}")

        # Enforce duration limits
        duration_minutes = min(duration_minutes, MAX_EMERGENCY_DURATION_HOURS * 60)

        # Update request
        now = datetime.now(timezone.utc)
        request.status = EmergencyAccessStatus.ACTIVE
        request.approved_at = now
        request.approved_by = approver_id
        request.expires_at = now + timedelta(minutes=duration_minutes)

        # Generate token
        token_value = secrets.token_urlsafe(32)
        token = EmergencyToken(
            token_id=str(uuid4()),
            request_id=request_id,
            user_id=request.user_id,
            token_hash=hashlib.sha256(token_value.encode()).hexdigest(),
            created_at=now,
            expires_at=request.expires_at,
            scope=scope or ["*"],  # Default: all access
        )

        self._tokens[token.token_id] = token
        self._active_by_user[request.user_id] = request_id

        self._audit_log(
            "approved",
            request.user_id,
            request_id,
            {
                "approver": approver_id,
                "duration_minutes": duration_minutes,
                "scope": scope,
                "expires_at": request.expires_at.isoformat(),
            },
        )

        self._notify_admins("approved", request)

        # Return token value (only returned once!)
        return EmergencyToken(
            token_id=token.token_id,
            request_id=request_id,
            user_id=request.user_id,
            token_hash=token_value,  # Return actual value, not hash
            created_at=now,
            expires_at=request.expires_at,
            scope=token.scope,
        )

    def self_approve_emergency_access(
        self,
        request_id: str,
        user_id: str,
        duration_minutes: int = EMERGENCY_TOKEN_DURATION_MINUTES,
    ) -> EmergencyToken:
        """
        Self-approve emergency access (break-glass).

        Used when no approver is available during true emergency.
        Creates heightened audit trail and requires mandatory review.

        Args:
            request_id: Request to self-approve
            user_id: User approving their own request
            duration_minutes: Access duration (limited for self-approval)

        Returns:
            EmergencyToken for access
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        if request.user_id != user_id:
            raise ValueError("Can only self-approve your own request")

        # Self-approval has reduced duration
        max_self_approve_minutes = min(60, EMERGENCY_TOKEN_DURATION_MINUTES)
        duration_minutes = min(duration_minutes, max_self_approve_minutes)

        # Mark for mandatory review
        request.status = EmergencyAccessStatus.REVIEW_PENDING

        self._audit_log(
            "self_approved",
            user_id,
            request_id,
            {
                "warning": "SELF-APPROVED EMERGENCY ACCESS - MANDATORY REVIEW REQUIRED",
                "duration_minutes": duration_minutes,
                "reason": request.reason.value,
            },
        )

        # Issue token with limited scope
        return self.approve_emergency_access(
            request_id,
            approver_id=f"SELF:{user_id}",
            duration_minutes=duration_minutes,
            scope=["read:*"],  # Limited to read-only for self-approval
        )

    def verify_emergency_token(
        self,
        token_value: str,
        resource: Optional[str] = None,
    ) -> Optional[EmergencyAccessRequest]:
        """
        Verify an emergency access token.

        Args:
            token_value: Token to verify
            resource: Optional resource being accessed

        Returns:
            EmergencyAccessRequest if valid, None otherwise
        """
        token_hash = hashlib.sha256(token_value.encode()).hexdigest()

        # Find matching token
        for token in self._tokens.values():
            if hmac.compare_digest(token.token_hash, token_hash):
                if token.is_expired():
                    return None

                request = self._requests.get(token.request_id)
                if not request or not request.is_active():
                    return None

                # Check scope
                if resource:
                    scope_allowed = "*" in token.scope or resource in token.scope
                    if not scope_allowed:
                        for scope_pattern in token.scope:
                            if scope_pattern.endswith("*") and resource.startswith(scope_pattern[:-1]):
                                scope_allowed = True
                                break
                    if not scope_allowed:
                        self._audit_log(
                            "access_denied",
                            request.user_id,
                            request.request_id,
                            {"resource": resource, "scope": token.scope},
                        )
                        return None

                # Log access
                if resource:
                    request.accessed_resources.append(resource)
                    self._audit_log(
                        "resource_accessed",
                        request.user_id,
                        request.request_id,
                        {"resource": resource},
                    )

                return request

        return None

    def revoke_emergency_access(
        self,
        request_id: str,
        revoker_id: str,
        reason: str,
    ) -> bool:
        """
        Revoke active emergency access.

        Args:
            request_id: Request to revoke
            revoker_id: User revoking the access
            reason: Reason for revocation

        Returns:
            True if revoked successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status != EmergencyAccessStatus.ACTIVE:
            return False

        request.status = EmergencyAccessStatus.REVOKED
        request.revoked_at = datetime.now(timezone.utc)
        request.revoked_by = revoker_id
        request.revocation_reason = reason

        # Remove from active tracking
        if request.user_id in self._active_by_user:
            del self._active_by_user[request.user_id]

        # Invalidate tokens
        for token in list(self._tokens.values()):
            if token.request_id == request_id:
                del self._tokens[token.token_id]

        self._audit_log(
            "revoked",
            request.user_id,
            request_id,
            {"revoker": revoker_id, "reason": reason},
        )

        self._notify_admins("revoked", request)

        return True

    def complete_post_incident_review(
        self,
        request_id: str,
        reviewer_id: str,
        notes: str,
        access_was_appropriate: bool,
    ) -> bool:
        """
        Complete mandatory post-incident review.

        HIPAA requires review of emergency access to ensure
        it was appropriate and justified.

        Args:
            request_id: Request to review
            reviewer_id: User performing review
            notes: Review notes
            access_was_appropriate: Whether access was justified

        Returns:
            True if review completed successfully
        """
        request = self._requests.get(request_id)
        if not request:
            return False

        if request.status not in [
            EmergencyAccessStatus.EXPIRED,
            EmergencyAccessStatus.REVOKED,
            EmergencyAccessStatus.REVIEW_PENDING,
        ]:
            return False

        request.review_completed = True
        request.review_notes = notes
        request.reviewed_by = reviewer_id
        request.reviewed_at = datetime.now(timezone.utc)
        request.status = EmergencyAccessStatus.REVIEWED

        self._audit_log(
            "review_completed",
            request.user_id,
            request_id,
            {
                "reviewer": reviewer_id,
                "appropriate": access_was_appropriate,
                "notes_preview": notes[:100],
                "resources_accessed": len(request.accessed_resources),
            },
        )

        if not access_was_appropriate:
            self._audit_log(
                "inappropriate_access_flagged",
                request.user_id,
                request_id,
                {
                    "reviewer": reviewer_id,
                    "notes": notes,
                    "warning": "SECURITY REVIEW REQUIRED",
                },
            )

        return True

    def get_pending_reviews(self) -> List[EmergencyAccessRequest]:
        """Get all requests pending post-incident review."""
        pending = []
        for request in self._requests.values():
            if request.status in [
                EmergencyAccessStatus.EXPIRED,
                EmergencyAccessStatus.REVOKED,
                EmergencyAccessStatus.REVIEW_PENDING,
            ]:
                if not request.review_completed:
                    pending.append(request)
        return pending

    def cleanup_expired(self) -> int:
        """
        Clean up expired emergency access.

        Returns:
            Number of requests expired
        """
        count = 0
        now = datetime.now(timezone.utc)

        for request in self._requests.values():
            if request.status == EmergencyAccessStatus.ACTIVE:
                if request.expires_at and now > request.expires_at:
                    request.status = EmergencyAccessStatus.EXPIRED
                    if request.user_id in self._active_by_user:
                        del self._active_by_user[request.user_id]
                    self._audit_log(
                        "expired",
                        request.user_id,
                        request.request_id,
                        {"accessed_resources_count": len(request.accessed_resources)},
                    )
                    count += 1

        # Clean up expired tokens
        for token_id in list(self._tokens.keys()):
            if self._tokens[token_id].is_expired():
                del self._tokens[token_id]

        return count


# Singleton instance
_emergency_manager: Optional[EmergencyAccessManager] = None


def get_emergency_access_manager() -> EmergencyAccessManager:
    """Get or create the default emergency access manager."""
    global _emergency_manager
    if _emergency_manager is None:
        _emergency_manager = EmergencyAccessManager()
    return _emergency_manager
