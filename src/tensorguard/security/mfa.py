"""
Multi-Factor Authentication (MFA) Module.

Provides TOTP-based MFA for administrative accounts:
- TOTP (Time-based One-Time Password) per RFC 6238
- Backup codes for account recovery
- MFA enforcement policies
- Audit logging of MFA events

Compliance Requirements:
- SOC 2 CC6.1: Logical access controls
- ISO 27001 A.8.5: Secure authentication
- HIPAA §164.312(d): Person or entity authentication

Usage:
    from tensorguard.security.mfa import (
        MFAManager,
        generate_totp_secret,
        verify_totp,
        generate_backup_codes,
    )
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

# TOTP Configuration (RFC 6238)
TOTP_DIGITS = 6
TOTP_PERIOD = 30  # seconds
TOTP_ALGORITHM = "sha1"  # RFC 6238 default
TOTP_WINDOW = 1  # Allow 1 period before/after for clock drift

# Backup codes
BACKUP_CODE_COUNT = 10
BACKUP_CODE_LENGTH = 8

# MFA enforcement
MFA_REQUIRED_ROLES = {"org_admin", "site_admin"}
MFA_GRACE_PERIOD_DAYS = int(os.getenv("TG_MFA_GRACE_PERIOD_DAYS", "7"))


class MFAMethod(str, Enum):
    """Supported MFA methods."""

    TOTP = "totp"
    BACKUP_CODE = "backup_code"


class MFAStatus(str, Enum):
    """MFA enrollment status."""

    NOT_ENROLLED = "not_enrolled"
    PENDING_VERIFICATION = "pending_verification"
    ENROLLED = "enrolled"
    DISABLED = "disabled"


# ============================================================================
# TOTP IMPLEMENTATION (RFC 6238)
# ============================================================================


def generate_totp_secret(length: int = 32) -> str:
    """
    Generate a cryptographically secure TOTP secret.

    Args:
        length: Secret length in bytes (default 32 for 256-bit security)

    Returns:
        Base32-encoded secret string
    """
    secret_bytes = secrets.token_bytes(length)
    return base64.b32encode(secret_bytes).decode("utf-8").rstrip("=")


def _compute_hotp(secret: str, counter: int) -> str:
    """
    Compute HOTP value per RFC 4226.

    Args:
        secret: Base32-encoded secret
        counter: Counter value

    Returns:
        HOTP code as zero-padded string
    """
    # Decode secret (handle missing padding)
    secret_padded = secret + "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(secret_padded.upper())

    # Pack counter as big-endian 64-bit integer
    counter_bytes = struct.pack(">Q", counter)

    # Compute HMAC-SHA1
    hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()

    # Dynamic truncation (RFC 4226 Section 5.4)
    offset = hmac_hash[-1] & 0x0F
    binary = struct.unpack(">I", hmac_hash[offset:offset + 4])[0] & 0x7FFFFFFF

    # Generate OTP
    otp = binary % (10 ** TOTP_DIGITS)
    return str(otp).zfill(TOTP_DIGITS)


def generate_totp(secret: str, timestamp: Optional[float] = None) -> str:
    """
    Generate TOTP value for current time.

    Args:
        secret: Base32-encoded secret
        timestamp: Unix timestamp (default: current time)

    Returns:
        TOTP code as string
    """
    if timestamp is None:
        timestamp = time.time()

    counter = int(timestamp // TOTP_PERIOD)
    return _compute_hotp(secret, counter)


def verify_totp(
    secret: str,
    code: str,
    timestamp: Optional[float] = None,
    window: int = TOTP_WINDOW,
) -> bool:
    """
    Verify a TOTP code with window for clock drift.

    Args:
        secret: Base32-encoded secret
        code: Code to verify
        timestamp: Unix timestamp (default: current time)
        window: Number of periods to check before/after

    Returns:
        True if code is valid
    """
    if timestamp is None:
        timestamp = time.time()

    # Sanitize code
    code = code.strip().replace(" ", "").replace("-", "")
    if not code.isdigit() or len(code) != TOTP_DIGITS:
        return False

    current_counter = int(timestamp // TOTP_PERIOD)

    # Check current period and window
    for offset in range(-window, window + 1):
        expected = _compute_hotp(secret, current_counter + offset)
        if hmac.compare_digest(code, expected):
            return True

    return False


def get_totp_uri(
    secret: str,
    account_name: str,
    issuer: str = "TensorGuard",
) -> str:
    """
    Generate otpauth:// URI for QR code generation.

    Args:
        secret: Base32-encoded secret
        account_name: User's account identifier (email)
        issuer: Service name

    Returns:
        otpauth:// URI string
    """
    from urllib.parse import quote

    return (
        f"otpauth://totp/{quote(issuer)}:{quote(account_name)}"
        f"?secret={secret}"
        f"&issuer={quote(issuer)}"
        f"&algorithm=SHA1"
        f"&digits={TOTP_DIGITS}"
        f"&period={TOTP_PERIOD}"
    )


# ============================================================================
# BACKUP CODES
# ============================================================================


def generate_backup_codes(count: int = BACKUP_CODE_COUNT) -> List[str]:
    """
    Generate backup codes for account recovery.

    Args:
        count: Number of codes to generate

    Returns:
        List of backup codes (format: XXXX-XXXX)
    """
    codes = []
    for _ in range(count):
        # Generate random bytes and convert to alphanumeric
        raw = secrets.token_hex(BACKUP_CODE_LENGTH // 2)
        # Format as XXXX-XXXX for readability
        formatted = f"{raw[:4].upper()}-{raw[4:].upper()}"
        codes.append(formatted)
    return codes


def hash_backup_code(code: str) -> str:
    """
    Hash a backup code for secure storage.

    Args:
        code: Backup code to hash

    Returns:
        SHA-256 hash of normalized code
    """
    # Normalize: remove formatting, uppercase
    normalized = code.replace("-", "").replace(" ", "").upper()
    return hashlib.sha256(normalized.encode()).hexdigest()


def verify_backup_code(code: str, hashed_codes: List[str]) -> Optional[int]:
    """
    Verify a backup code against stored hashes.

    Args:
        code: Code to verify
        hashed_codes: List of hashed backup codes

    Returns:
        Index of matching code, or None if not found
    """
    code_hash = hash_backup_code(code)

    for i, stored_hash in enumerate(hashed_codes):
        if hmac.compare_digest(code_hash, stored_hash):
            return i

    return None


# ============================================================================
# MFA MANAGER
# ============================================================================


@dataclass
class MFAEnrollment:
    """MFA enrollment state for a user."""

    user_id: str
    status: MFAStatus = MFAStatus.NOT_ENROLLED
    method: Optional[MFAMethod] = None
    totp_secret: Optional[str] = None
    backup_codes_hash: List[str] = field(default_factory=list)
    used_backup_codes: Set[int] = field(default_factory=set)
    enrolled_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    failed_attempts: int = 0
    locked_until: Optional[datetime] = None


class MFAManager:
    """
    Multi-Factor Authentication Manager.

    Handles MFA enrollment, verification, and policy enforcement.

    Attributes:
        enrollments: In-memory enrollment storage (use Redis/DB in production)
        audit_callback: Optional callback for audit logging
    """

    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        audit_callback: Optional[callable] = None,
    ):
        """
        Initialize MFA Manager.

        Args:
            redis_client: Optional Redis client for distributed storage
            audit_callback: Optional callback for audit logging
        """
        self._redis = redis_client
        self._audit_callback = audit_callback
        self._enrollments: Dict[str, MFAEnrollment] = {}

    def _get_enrollment(self, user_id: str) -> MFAEnrollment:
        """Get or create enrollment for user."""
        if user_id not in self._enrollments:
            self._enrollments[user_id] = MFAEnrollment(user_id=user_id)
        return self._enrollments[user_id]

    def _audit_log(self, event: str, user_id: str, details: Dict[str, Any]) -> None:
        """Log MFA event for audit trail."""
        if self._audit_callback:
            self._audit_callback(event, user_id, details)
        logger.info(f"MFA: {event} | user={user_id} | {details}")

    def is_mfa_required(self, user_role: str) -> bool:
        """
        Check if MFA is required for a user role.

        Args:
            user_role: User's role

        Returns:
            True if MFA is required
        """
        return user_role in MFA_REQUIRED_ROLES

    def get_enrollment_status(self, user_id: str) -> MFAStatus:
        """
        Get MFA enrollment status for a user.

        Args:
            user_id: User identifier

        Returns:
            Current MFA status
        """
        enrollment = self._get_enrollment(user_id)
        return enrollment.status

    def begin_enrollment(self, user_id: str, email: str) -> Dict[str, Any]:
        """
        Begin MFA enrollment process.

        Args:
            user_id: User identifier
            email: User's email for TOTP URI

        Returns:
            Enrollment data including secret and QR code URI
        """
        enrollment = self._get_enrollment(user_id)

        # Generate new TOTP secret
        secret = generate_totp_secret()
        enrollment.totp_secret = secret
        enrollment.status = MFAStatus.PENDING_VERIFICATION
        enrollment.method = MFAMethod.TOTP

        # Generate TOTP URI for QR code
        totp_uri = get_totp_uri(secret, email)

        self._audit_log("mfa.enrollment.started", user_id, {"method": "totp"})

        return {
            "secret": secret,
            "totp_uri": totp_uri,
            "status": enrollment.status.value,
        }

    def complete_enrollment(
        self,
        user_id: str,
        verification_code: str,
    ) -> Dict[str, Any]:
        """
        Complete MFA enrollment by verifying TOTP code.

        Args:
            user_id: User identifier
            verification_code: TOTP code to verify

        Returns:
            Enrollment result including backup codes

        Raises:
            ValueError: If verification fails
        """
        enrollment = self._get_enrollment(user_id)

        if enrollment.status != MFAStatus.PENDING_VERIFICATION:
            raise ValueError("No pending MFA enrollment")

        if not enrollment.totp_secret:
            raise ValueError("No TOTP secret configured")

        # Verify the code
        if not verify_totp(enrollment.totp_secret, verification_code):
            enrollment.failed_attempts += 1
            self._audit_log(
                "mfa.enrollment.verification_failed",
                user_id,
                {"attempts": enrollment.failed_attempts},
            )
            raise ValueError("Invalid verification code")

        # Generate backup codes
        backup_codes = generate_backup_codes()
        enrollment.backup_codes_hash = [hash_backup_code(c) for c in backup_codes]
        enrollment.used_backup_codes = set()

        # Complete enrollment
        enrollment.status = MFAStatus.ENROLLED
        enrollment.enrolled_at = datetime.now(timezone.utc)
        enrollment.failed_attempts = 0

        self._audit_log("mfa.enrollment.completed", user_id, {"method": "totp"})

        return {
            "status": enrollment.status.value,
            "backup_codes": backup_codes,  # Only shown once!
            "message": "MFA enrollment complete. Save your backup codes securely.",
        }

    def verify_mfa(
        self,
        user_id: str,
        code: str,
        method: MFAMethod = MFAMethod.TOTP,
    ) -> bool:
        """
        Verify MFA code during authentication.

        Args:
            user_id: User identifier
            code: MFA code to verify
            method: MFA method (TOTP or backup code)

        Returns:
            True if verification successful

        Raises:
            ValueError: If user is locked out or not enrolled
        """
        enrollment = self._get_enrollment(user_id)

        # Check enrollment status
        if enrollment.status != MFAStatus.ENROLLED:
            raise ValueError("MFA not enrolled")

        # Check lockout
        if enrollment.locked_until:
            if datetime.now(timezone.utc) < enrollment.locked_until:
                remaining = (enrollment.locked_until - datetime.now(timezone.utc)).seconds
                raise ValueError(f"Account locked. Try again in {remaining} seconds")
            else:
                # Clear lockout
                enrollment.locked_until = None
                enrollment.failed_attempts = 0

        # Verify based on method
        verified = False

        if method == MFAMethod.TOTP:
            if enrollment.totp_secret:
                verified = verify_totp(enrollment.totp_secret, code)

        elif method == MFAMethod.BACKUP_CODE:
            code_index = verify_backup_code(code, enrollment.backup_codes_hash)
            if code_index is not None and code_index not in enrollment.used_backup_codes:
                enrollment.used_backup_codes.add(code_index)
                verified = True
                self._audit_log(
                    "mfa.backup_code.used",
                    user_id,
                    {"remaining": len(enrollment.backup_codes_hash) - len(enrollment.used_backup_codes)},
                )

        if verified:
            enrollment.last_used_at = datetime.now(timezone.utc)
            enrollment.failed_attempts = 0
            self._audit_log("mfa.verification.success", user_id, {"method": method.value})
            return True
        else:
            enrollment.failed_attempts += 1
            self._audit_log(
                "mfa.verification.failed",
                user_id,
                {"method": method.value, "attempts": enrollment.failed_attempts},
            )

            # Lock account after max attempts
            if enrollment.failed_attempts >= self.MAX_FAILED_ATTEMPTS:
                enrollment.locked_until = datetime.now(timezone.utc) + \
                    timedelta(minutes=self.LOCKOUT_DURATION_MINUTES)
                self._audit_log(
                    "mfa.account.locked",
                    user_id,
                    {"duration_minutes": self.LOCKOUT_DURATION_MINUTES},
                )

            return False

    def disable_mfa(self, user_id: str, admin_user_id: Optional[str] = None) -> bool:
        """
        Disable MFA for a user (admin action).

        Args:
            user_id: User to disable MFA for
            admin_user_id: Admin performing the action

        Returns:
            True if disabled successfully
        """
        enrollment = self._get_enrollment(user_id)

        if enrollment.status == MFAStatus.NOT_ENROLLED:
            return False

        # Clear all MFA data
        enrollment.status = MFAStatus.DISABLED
        enrollment.totp_secret = None
        enrollment.backup_codes_hash = []
        enrollment.used_backup_codes = set()

        self._audit_log(
            "mfa.disabled",
            user_id,
            {"admin": admin_user_id or "self"},
        )

        return True

    def regenerate_backup_codes(self, user_id: str, totp_code: str) -> List[str]:
        """
        Regenerate backup codes (requires TOTP verification).

        Args:
            user_id: User identifier
            totp_code: Current TOTP code for verification

        Returns:
            New backup codes

        Raises:
            ValueError: If TOTP verification fails
        """
        enrollment = self._get_enrollment(user_id)

        if enrollment.status != MFAStatus.ENROLLED:
            raise ValueError("MFA not enrolled")

        # Verify TOTP first
        if not verify_totp(enrollment.totp_secret, totp_code):
            raise ValueError("Invalid TOTP code")

        # Generate new backup codes
        backup_codes = generate_backup_codes()
        enrollment.backup_codes_hash = [hash_backup_code(c) for c in backup_codes]
        enrollment.used_backup_codes = set()

        self._audit_log("mfa.backup_codes.regenerated", user_id, {})

        return backup_codes

    def get_remaining_backup_codes(self, user_id: str) -> int:
        """
        Get count of remaining unused backup codes.

        Args:
            user_id: User identifier

        Returns:
            Number of unused backup codes
        """
        enrollment = self._get_enrollment(user_id)
        return len(enrollment.backup_codes_hash) - len(enrollment.used_backup_codes)


# Singleton instance
_mfa_manager: Optional[MFAManager] = None


def get_mfa_manager() -> MFAManager:
    """Get or create the default MFA manager."""
    global _mfa_manager
    if _mfa_manager is None:
        _mfa_manager = MFAManager()
    return _mfa_manager
