"""
Data Loss Prevention (DLP) Module.

Provides DLP scanning and prevention capabilities:
- PII detection (SSN, credit cards, emails, phone numbers)
- Custom sensitive data patterns
- Data classification
- Egress monitoring
- Automatic redaction

Compliance Requirements:
- SOC 2 CC6.7: Data transmission protection
- ISO 27001 A.8.12: Data leakage prevention
- HIPAA §164.312(e): Transmission security

Usage:
    from tensorguard.security.dlp import (
        DLPScanner,
        scan_for_pii,
        redact_sensitive_data,
    )
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Set

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DLP_ENABLED = os.getenv("TG_DLP_ENABLED", "true").lower() == "true"
DLP_BLOCK_ON_DETECT = os.getenv("TG_DLP_BLOCK_ON_DETECT", "false").lower() == "true"
DLP_REDACT_ON_DETECT = os.getenv("TG_DLP_REDACT_ON_DETECT", "true").lower() == "true"


class SensitiveDataType(str, Enum):
    """Types of sensitive data detected by DLP."""

    SSN = "ssn"  # Social Security Number
    CREDIT_CARD = "credit_card"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    PASSWORD = "password"
    AWS_KEY = "aws_key"
    PRIVATE_KEY = "private_key"
    JWT_TOKEN = "jwt_token"
    MEDICAL_RECORD = "medical_record"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    CUSTOM = "custom"


class DLPAction(str, Enum):
    """Actions to take when sensitive data is detected."""

    LOG = "log"  # Log only
    REDACT = "redact"  # Redact and continue
    BLOCK = "block"  # Block the operation
    ALERT = "alert"  # Send alert


@dataclass
class DLPPattern:
    """Pattern definition for DLP scanning."""

    pattern_id: str
    name: str
    data_type: SensitiveDataType
    regex: Pattern
    confidence: float  # 0.0-1.0
    action: DLPAction = DLPAction.REDACT
    redaction_template: str = "[REDACTED:{type}]"
    description: Optional[str] = None
    enabled: bool = True


@dataclass
class DLPMatch:
    """A match found during DLP scanning."""

    pattern_id: str
    data_type: SensitiveDataType
    matched_text: str
    start_pos: int
    end_pos: int
    confidence: float
    context: str  # Surrounding text for review
    redacted_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without matched text for security)."""
        return {
            "pattern_id": self.pattern_id,
            "data_type": self.data_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
            "context_hash": hashlib.sha256(self.context.encode()).hexdigest()[:16],
        }


@dataclass
class DLPScanResult:
    """Result of a DLP scan."""

    scan_id: str
    scanned_at: datetime
    content_hash: str
    matches: List[DLPMatch]
    blocked: bool = False
    redacted_content: Optional[str] = None
    action_taken: Optional[DLPAction] = None

    @property
    def has_sensitive_data(self) -> bool:
        """Check if any sensitive data was found."""
        return len(self.matches) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "scan_id": self.scan_id,
            "scanned_at": self.scanned_at.isoformat(),
            "content_hash": self.content_hash,
            "match_count": len(self.matches),
            "data_types_found": list(set(m.data_type.value for m in self.matches)),
            "blocked": self.blocked,
            "action_taken": self.action_taken.value if self.action_taken else None,
        }


class DLPScanner:
    """
    Data Loss Prevention Scanner.

    Scans content for sensitive data and applies configured actions.
    """

    # Built-in patterns for common sensitive data types
    DEFAULT_PATTERNS = [
        DLPPattern(
            pattern_id="ssn_us",
            name="US Social Security Number",
            data_type=SensitiveDataType.SSN,
            regex=re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
            confidence=0.9,
            action=DLPAction.REDACT,
            description="US SSN format XXX-XX-XXXX",
        ),
        DLPPattern(
            pattern_id="ssn_us_no_dash",
            name="US SSN (no dashes)",
            data_type=SensitiveDataType.SSN,
            regex=re.compile(r"\b\d{9}\b"),
            confidence=0.5,  # Lower confidence without dashes
            action=DLPAction.LOG,
            description="9 consecutive digits",
        ),
        DLPPattern(
            pattern_id="credit_card_visa",
            name="Visa Credit Card",
            data_type=SensitiveDataType.CREDIT_CARD,
            regex=re.compile(r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            confidence=0.9,
            action=DLPAction.REDACT,
        ),
        DLPPattern(
            pattern_id="credit_card_mc",
            name="Mastercard Credit Card",
            data_type=SensitiveDataType.CREDIT_CARD,
            regex=re.compile(r"\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
            confidence=0.9,
            action=DLPAction.REDACT,
        ),
        DLPPattern(
            pattern_id="credit_card_amex",
            name="American Express",
            data_type=SensitiveDataType.CREDIT_CARD,
            regex=re.compile(r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b"),
            confidence=0.9,
            action=DLPAction.REDACT,
        ),
        DLPPattern(
            pattern_id="email",
            name="Email Address",
            data_type=SensitiveDataType.EMAIL,
            regex=re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
            confidence=0.95,
            action=DLPAction.LOG,  # Emails often legitimate
        ),
        DLPPattern(
            pattern_id="phone_us",
            name="US Phone Number",
            data_type=SensitiveDataType.PHONE,
            regex=re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            confidence=0.7,
            action=DLPAction.LOG,
        ),
        DLPPattern(
            pattern_id="ip_address",
            name="IP Address",
            data_type=SensitiveDataType.IP_ADDRESS,
            regex=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            confidence=0.8,
            action=DLPAction.LOG,
        ),
        DLPPattern(
            pattern_id="aws_access_key",
            name="AWS Access Key",
            data_type=SensitiveDataType.AWS_KEY,
            regex=re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
            confidence=0.99,
            action=DLPAction.BLOCK,
            description="AWS Access Key ID",
        ),
        DLPPattern(
            pattern_id="aws_secret_key",
            name="AWS Secret Key",
            data_type=SensitiveDataType.AWS_KEY,
            regex=re.compile(r"\b[A-Za-z0-9/+=]{40}\b"),
            confidence=0.6,  # Many false positives
            action=DLPAction.LOG,
        ),
        DLPPattern(
            pattern_id="private_key",
            name="Private Key",
            data_type=SensitiveDataType.PRIVATE_KEY,
            regex=re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
            confidence=0.99,
            action=DLPAction.BLOCK,
        ),
        DLPPattern(
            pattern_id="jwt_token",
            name="JWT Token",
            data_type=SensitiveDataType.JWT_TOKEN,
            regex=re.compile(r"\beyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*\b"),
            confidence=0.95,
            action=DLPAction.REDACT,
        ),
        DLPPattern(
            pattern_id="api_key_generic",
            name="Generic API Key",
            data_type=SensitiveDataType.API_KEY,
            regex=re.compile(r"\b(?:api[_-]?key|apikey)[\"'\s:=]+[a-zA-Z0-9_-]{20,}\b", re.IGNORECASE),
            confidence=0.8,
            action=DLPAction.REDACT,
        ),
        DLPPattern(
            pattern_id="password_field",
            name="Password Field",
            data_type=SensitiveDataType.PASSWORD,
            regex=re.compile(r"\b(?:password|passwd|pwd)[\"'\s:=]+[^\s\"']{8,}\b", re.IGNORECASE),
            confidence=0.85,
            action=DLPAction.REDACT,
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[DLPPattern]] = None,
        audit_callback: Optional[Callable] = None,
        alert_callback: Optional[Callable] = None,
        context_chars: int = 50,
    ):
        """
        Initialize DLP Scanner.

        Args:
            patterns: Custom patterns (default: built-in patterns)
            audit_callback: Callback for audit logging
            alert_callback: Callback for alerts
            context_chars: Characters of context to capture around matches
        """
        self._patterns = {p.pattern_id: p for p in (patterns or self.DEFAULT_PATTERNS)}
        self._audit_callback = audit_callback
        self._alert_callback = alert_callback
        self._context_chars = context_chars
        self._scan_count = 0

    def add_pattern(self, pattern: DLPPattern) -> None:
        """Add a custom DLP pattern."""
        self._patterns[pattern.pattern_id] = pattern

    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove a DLP pattern."""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            return True
        return False

    def scan(
        self,
        content: str,
        context: Optional[str] = None,
        data_types: Optional[Set[SensitiveDataType]] = None,
    ) -> DLPScanResult:
        """
        Scan content for sensitive data.

        Args:
            content: Content to scan
            context: Optional context for audit (e.g., "api_response")
            data_types: Optional filter for specific data types

        Returns:
            DLPScanResult with matches and actions taken
        """
        if not DLP_ENABLED:
            return DLPScanResult(
                scan_id=f"dlp_{self._scan_count}",
                scanned_at=datetime.now(timezone.utc),
                content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
                matches=[],
            )

        self._scan_count += 1
        scan_id = f"dlp_{self._scan_count}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        matches: List[DLPMatch] = []
        should_block = False
        highest_action = DLPAction.LOG

        # Scan with each pattern
        for pattern in self._patterns.values():
            if not pattern.enabled:
                continue

            if data_types and pattern.data_type not in data_types:
                continue

            for match in pattern.regex.finditer(content):
                # Extract context
                start = max(0, match.start() - self._context_chars)
                end = min(len(content), match.end() + self._context_chars)
                match_context = content[start:end]

                dlp_match = DLPMatch(
                    pattern_id=pattern.pattern_id,
                    data_type=pattern.data_type,
                    matched_text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=pattern.confidence,
                    context=match_context,
                )

                matches.append(dlp_match)

                # Determine action
                if pattern.action == DLPAction.BLOCK:
                    should_block = True
                    highest_action = DLPAction.BLOCK
                elif pattern.action == DLPAction.ALERT and highest_action != DLPAction.BLOCK:
                    highest_action = DLPAction.ALERT
                elif pattern.action == DLPAction.REDACT and highest_action not in [DLPAction.BLOCK, DLPAction.ALERT]:
                    highest_action = DLPAction.REDACT

        # Apply actions
        redacted_content = None
        if matches:
            if DLP_REDACT_ON_DETECT and highest_action in [DLPAction.REDACT, DLPAction.BLOCK]:
                redacted_content = self._redact_content(content, matches)

            if DLP_BLOCK_ON_DETECT and should_block:
                pass  # Block will be enforced by caller

            # Audit log
            self._log_scan(scan_id, context, matches, highest_action)

            # Alert if needed
            if highest_action == DLPAction.ALERT and self._alert_callback:
                self._alert_callback(
                    event="dlp.sensitive_data_detected",
                    data={
                        "scan_id": scan_id,
                        "data_types": list(set(m.data_type.value for m in matches)),
                        "match_count": len(matches),
                    },
                )

        return DLPScanResult(
            scan_id=scan_id,
            scanned_at=datetime.now(timezone.utc),
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            matches=matches,
            blocked=should_block and DLP_BLOCK_ON_DETECT,
            redacted_content=redacted_content,
            action_taken=highest_action if matches else None,
        )

    def _redact_content(
        self,
        content: str,
        matches: List[DLPMatch],
    ) -> str:
        """Redact sensitive data from content."""
        # Sort matches by position (descending) to avoid offset issues
        sorted_matches = sorted(matches, key=lambda m: m.start_pos, reverse=True)

        redacted = content
        for match in sorted_matches:
            pattern = self._patterns.get(match.pattern_id)
            if pattern:
                redaction = pattern.redaction_template.format(type=match.data_type.value)
            else:
                redaction = f"[REDACTED:{match.data_type.value}]"

            match.redacted_value = redaction
            redacted = redacted[:match.start_pos] + redaction + redacted[match.end_pos:]

        return redacted

    def _log_scan(
        self,
        scan_id: str,
        context: Optional[str],
        matches: List[DLPMatch],
        action: DLPAction,
    ) -> None:
        """Log DLP scan result."""
        log_data = {
            "scan_id": scan_id,
            "context": context,
            "match_count": len(matches),
            "data_types": list(set(m.data_type.value for m in matches)),
            "action": action.value,
            "high_confidence_matches": sum(1 for m in matches if m.confidence >= 0.9),
        }

        if self._audit_callback:
            self._audit_callback("dlp.scan_completed", None, log_data)

        logger.info(f"DLP: scan_id={scan_id} matches={len(matches)} action={action.value}")

    def validate_luhn(self, number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.

        Args:
            number: Card number to validate

        Returns:
            True if valid Luhn checksum
        """
        digits = [int(d) for d in re.sub(r"\D", "", number)]
        if len(digits) < 13:
            return False

        # Luhn algorithm
        checksum = 0
        for i, digit in enumerate(reversed(digits)):
            if i % 2 == 1:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit

        return checksum % 10 == 0


def scan_for_pii(content: str) -> DLPScanResult:
    """
    Convenience function to scan content for PII.

    Args:
        content: Content to scan

    Returns:
        DLPScanResult
    """
    scanner = DLPScanner()
    return scanner.scan(content)


def redact_sensitive_data(content: str) -> str:
    """
    Convenience function to redact sensitive data.

    Args:
        content: Content to redact

    Returns:
        Redacted content
    """
    scanner = DLPScanner()
    result = scanner.scan(content)
    return result.redacted_content or content


# Singleton instance
_dlp_scanner: Optional[DLPScanner] = None


def get_dlp_scanner() -> DLPScanner:
    """Get or create the default DLP scanner."""
    global _dlp_scanner
    if _dlp_scanner is None:
        _dlp_scanner = DLPScanner()
    return _dlp_scanner
