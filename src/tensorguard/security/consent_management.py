"""
Consent Management Module.

Provides dynamic consent tracking and management:
- Granular consent collection
- Purpose-based consent
- Consent withdrawal
- Consent audit trail
- Consent preferences API

Compliance Requirements:
- GDPR Article 7: Conditions for consent
- ISO 27701 A.7.2.3-A.7.2.4: Consent management
- SOC 2 TSC Privacy P2: Choice and consent

Usage:
    from tensorguard.security.consent_management import (
        ConsentManager,
        record_consent,
        check_consent,
        withdraw_consent,
    )
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONSENT_VERSION = os.getenv("TG_CONSENT_VERSION", "1.0")
REQUIRE_EXPLICIT_CONSENT = os.getenv("TG_REQUIRE_EXPLICIT_CONSENT", "true").lower() == "true"


class ConsentPurpose(str, Enum):
    """Predefined consent purposes (GDPR-aligned)."""

    ESSENTIAL = "essential"  # Essential for service operation
    ANALYTICS = "analytics"  # Usage analytics
    PERSONALIZATION = "personalization"  # Personalized experience
    MARKETING = "marketing"  # Marketing communications
    THIRD_PARTY = "third_party"  # Third-party sharing
    RESEARCH = "research"  # Research and development
    MODEL_TRAINING = "model_training"  # ML model training
    DATA_EXPORT = "data_export"  # Data export to other services


class ConsentStatus(str, Enum):
    """Consent status."""

    GRANTED = "granted"
    DENIED = "denied"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    PENDING = "pending"


class ConsentMethod(str, Enum):
    """How consent was collected."""

    EXPLICIT_OPT_IN = "explicit_opt_in"  # User explicitly opted in
    IMPLICIT = "implicit"  # Implied consent (legitimate interest)
    CONTRACT = "contract"  # Consent via contract
    LEGAL_OBLIGATION = "legal_obligation"  # Required by law
    API = "api"  # Consent via API
    UI = "ui"  # Consent via user interface


@dataclass
class ConsentRecord:
    """Individual consent record."""

    consent_id: str
    subject_id: str
    purpose: ConsentPurpose
    status: ConsentStatus
    method: ConsentMethod
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    withdrawn_at: Optional[datetime] = None
    withdrawal_reason: Optional[str] = None
    version: str = CONSENT_VERSION
    scope: Optional[str] = None  # Specific scope within purpose
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "consent_id": self.consent_id,
            "subject_id": self.subject_id,
            "purpose": self.purpose.value,
            "status": self.status.value,
            "method": self.method.value,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "withdrawn_at": self.withdrawn_at.isoformat() if self.withdrawn_at else None,
            "version": self.version,
            "is_valid": self.is_valid(),
        }


@dataclass
class ConsentPreferences:
    """User's consent preferences."""

    subject_id: str
    consents: Dict[ConsentPurpose, ConsentRecord]
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_consent(self, purpose: ConsentPurpose) -> bool:
        """Check if consent exists for a purpose."""
        consent = self.consents.get(purpose)
        return consent is not None and consent.is_valid()

    def get_granted_purposes(self) -> List[ConsentPurpose]:
        """Get list of purposes with valid consent."""
        return [p for p, c in self.consents.items() if c.is_valid()]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "subject_id": self.subject_id,
            "last_updated": self.last_updated.isoformat(),
            "consents": {p.value: c.to_dict() for p, c in self.consents.items()},
            "granted_purposes": [p.value for p in self.get_granted_purposes()],
        }


class ConsentManager:
    """
    Consent Management System.

    Handles collection, storage, and verification of user consents.
    """

    # Purposes that cannot be withdrawn (essential for service)
    ESSENTIAL_PURPOSES = {ConsentPurpose.ESSENTIAL}

    def __init__(
        self,
        audit_callback: Optional[Callable] = None,
        notification_callback: Optional[Callable] = None,
    ):
        """
        Initialize Consent Manager.

        Args:
            audit_callback: Callback for audit logging
            notification_callback: Callback for notifications
        """
        self._audit_callback = audit_callback
        self._notification_callback = notification_callback
        self._preferences: Dict[str, ConsentPreferences] = {}
        self._consent_history: List[ConsentRecord] = []

    def _audit_log(
        self,
        event: str,
        subject_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Log consent event for audit trail."""
        audit_record = {
            "event": f"consent.{event}",
            "subject_id": subject_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
            "compliance_relevant": True,
        }

        if self._audit_callback:
            self._audit_callback("privacy.consent." + event, subject_id, audit_record)

        logger.info(f"CONSENT: {event} | subject={subject_id} | {details}")

    def _get_preferences(self, subject_id: str) -> ConsentPreferences:
        """Get or create consent preferences for a subject."""
        if subject_id not in self._preferences:
            self._preferences[subject_id] = ConsentPreferences(
                subject_id=subject_id,
                consents={},
            )
        return self._preferences[subject_id]

    def record_consent(
        self,
        subject_id: str,
        purpose: ConsentPurpose,
        method: ConsentMethod,
        granted: bool = True,
        scope: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsentRecord:
        """
        Record a consent decision.

        Args:
            subject_id: Data subject identifier
            purpose: Purpose of consent
            method: How consent was collected
            granted: Whether consent was granted
            scope: Specific scope within purpose
            expires_at: Optional expiration date
            metadata: Additional metadata

        Returns:
            Created ConsentRecord
        """
        prefs = self._get_preferences(subject_id)

        consent = ConsentRecord(
            consent_id=str(uuid4()),
            subject_id=subject_id,
            purpose=purpose,
            status=ConsentStatus.GRANTED if granted else ConsentStatus.DENIED,
            method=method,
            granted_at=datetime.now(timezone.utc) if granted else None,
            expires_at=expires_at,
            scope=scope,
            metadata=metadata or {},
        )

        prefs.consents[purpose] = consent
        prefs.last_updated = datetime.now(timezone.utc)
        self._consent_history.append(consent)

        self._audit_log(
            "recorded",
            subject_id,
            {
                "purpose": purpose.value,
                "granted": granted,
                "method": method.value,
                "consent_id": consent.consent_id,
            },
        )

        return consent

    def withdraw_consent(
        self,
        subject_id: str,
        purpose: ConsentPurpose,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Withdraw consent for a purpose.

        Args:
            subject_id: Data subject identifier
            purpose: Purpose to withdraw consent for
            reason: Optional reason for withdrawal

        Returns:
            True if consent was withdrawn

        Raises:
            ValueError: If purpose is essential and cannot be withdrawn
        """
        if purpose in self.ESSENTIAL_PURPOSES:
            raise ValueError(f"Cannot withdraw consent for essential purpose: {purpose}")

        prefs = self._get_preferences(subject_id)
        consent = prefs.consents.get(purpose)

        if not consent or consent.status != ConsentStatus.GRANTED:
            return False

        consent.status = ConsentStatus.WITHDRAWN
        consent.withdrawn_at = datetime.now(timezone.utc)
        consent.withdrawal_reason = reason
        prefs.last_updated = datetime.now(timezone.utc)

        self._audit_log(
            "withdrawn",
            subject_id,
            {
                "purpose": purpose.value,
                "reason": reason,
                "consent_id": consent.consent_id,
            },
        )

        if self._notification_callback:
            self._notification_callback(
                event="consent.withdrawn",
                subject_id=subject_id,
                data={"purpose": purpose.value},
            )

        return True

    def check_consent(
        self,
        subject_id: str,
        purpose: ConsentPurpose,
        scope: Optional[str] = None,
    ) -> bool:
        """
        Check if valid consent exists for a purpose.

        Args:
            subject_id: Data subject identifier
            purpose: Purpose to check
            scope: Optional specific scope

        Returns:
            True if valid consent exists
        """
        prefs = self._get_preferences(subject_id)
        consent = prefs.consents.get(purpose)

        if not consent or not consent.is_valid():
            return False

        # Check scope if specified
        if scope and consent.scope and consent.scope != scope:
            return False

        return True

    def require_consent(
        self,
        subject_id: str,
        purpose: ConsentPurpose,
        scope: Optional[str] = None,
    ) -> bool:
        """
        Require consent, raising error if not present.

        Args:
            subject_id: Data subject identifier
            purpose: Required purpose
            scope: Optional specific scope

        Returns:
            True if consent is valid

        Raises:
            PermissionError: If consent is not granted
        """
        if not self.check_consent(subject_id, purpose, scope):
            self._audit_log(
                "consent_required",
                subject_id,
                {"purpose": purpose.value, "scope": scope, "granted": False},
            )
            raise PermissionError(f"Consent required for: {purpose.value}")

        return True

    def get_preferences(self, subject_id: str) -> ConsentPreferences:
        """
        Get consent preferences for a subject.

        Args:
            subject_id: Data subject identifier

        Returns:
            ConsentPreferences for the subject
        """
        return self._get_preferences(subject_id)

    def get_consent_history(
        self,
        subject_id: str,
        purpose: Optional[ConsentPurpose] = None,
    ) -> List[ConsentRecord]:
        """
        Get consent history for a subject.

        Args:
            subject_id: Data subject identifier
            purpose: Optional filter by purpose

        Returns:
            List of consent records
        """
        history = [c for c in self._consent_history if c.subject_id == subject_id]
        if purpose:
            history = [c for c in history if c.purpose == purpose]
        return sorted(history, key=lambda c: c.granted_at or datetime.min, reverse=True)

    def update_consent(
        self,
        subject_id: str,
        purpose: ConsentPurpose,
        granted: bool,
        method: ConsentMethod = ConsentMethod.API,
    ) -> ConsentRecord:
        """
        Update consent status (grant or deny).

        Args:
            subject_id: Data subject identifier
            purpose: Purpose to update
            granted: Whether to grant consent
            method: How consent was updated

        Returns:
            Updated ConsentRecord
        """
        if granted:
            return self.record_consent(subject_id, purpose, method, granted=True)
        else:
            self.withdraw_consent(subject_id, purpose)
            return self._get_preferences(subject_id).consents.get(purpose)

    def batch_update_consent(
        self,
        subject_id: str,
        consents: Dict[ConsentPurpose, bool],
        method: ConsentMethod = ConsentMethod.UI,
    ) -> Dict[ConsentPurpose, ConsentRecord]:
        """
        Update multiple consents at once.

        Args:
            subject_id: Data subject identifier
            consents: Dictionary of purpose -> granted
            method: How consents were collected

        Returns:
            Dictionary of updated records
        """
        results = {}
        for purpose, granted in consents.items():
            if granted:
                results[purpose] = self.record_consent(
                    subject_id, purpose, method, granted=True
                )
            else:
                if purpose not in self.ESSENTIAL_PURPOSES:
                    self.withdraw_consent(subject_id, purpose)
                    results[purpose] = self._get_preferences(subject_id).consents.get(purpose)

        self._audit_log(
            "batch_updated",
            subject_id,
            {
                "purposes_updated": [p.value for p in results.keys()],
                "method": method.value,
            },
        )

        return results

    def get_subjects_with_consent(
        self,
        purpose: ConsentPurpose,
    ) -> List[str]:
        """
        Get all subjects who have granted consent for a purpose.

        Args:
            purpose: Purpose to check

        Returns:
            List of subject IDs
        """
        return [
            subject_id
            for subject_id, prefs in self._preferences.items()
            if prefs.has_consent(purpose)
        ]

    def expire_old_consents(self) -> int:
        """
        Mark expired consents.

        Returns:
            Number of consents expired
        """
        count = 0
        now = datetime.now(timezone.utc)

        for prefs in self._preferences.values():
            for consent in prefs.consents.values():
                if consent.status == ConsentStatus.GRANTED:
                    if consent.expires_at and now > consent.expires_at:
                        consent.status = ConsentStatus.EXPIRED
                        count += 1
                        self._audit_log(
                            "expired",
                            prefs.subject_id,
                            {"purpose": consent.purpose.value},
                        )

        return count


# Singleton instance
_consent_manager: Optional[ConsentManager] = None


def get_consent_manager() -> ConsentManager:
    """Get or create the default consent manager."""
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager()
    return _consent_manager


# Convenience functions
def record_consent(
    subject_id: str,
    purpose: ConsentPurpose,
    method: ConsentMethod,
    granted: bool = True,
) -> ConsentRecord:
    """Record a consent decision."""
    return get_consent_manager().record_consent(subject_id, purpose, method, granted)


def check_consent(subject_id: str, purpose: ConsentPurpose) -> bool:
    """Check if consent exists."""
    return get_consent_manager().check_consent(subject_id, purpose)


def withdraw_consent(subject_id: str, purpose: ConsentPurpose) -> bool:
    """Withdraw consent."""
    return get_consent_manager().withdraw_consent(subject_id, purpose)
