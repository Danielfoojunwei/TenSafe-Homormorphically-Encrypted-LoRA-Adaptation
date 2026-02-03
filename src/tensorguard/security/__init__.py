"""
TensorGuard Security Hardening Module.

Provides comprehensive security features for production deployments:
- Rate limiting with IP-based throttling
- Token revocation list (TRL) for JWT invalidation
- Security event audit logging
- Content Security Policy (CSP) headers
- Secure memory handling for cryptographic secrets
- Request signing and replay attack prevention
- Constant-time comparison utilities
- Input sanitization
- Multi-Factor Authentication (MFA/TOTP)
- Emergency Access (break-glass procedures)
- Data Subject Access Requests (DSAR)
- Data Loss Prevention (DLP)
- Consent Management
- Access Reviews
- Data Protection Impact Assessment (DPIA)

Usage:
    from tensorguard.security import (
        RateLimiter,
        TokenRevocationList,
        SecurityAuditLog,
        SecureMemory,
        RequestSigner,
        constant_time_compare,
        sanitize_input,
        MFAManager,
        EmergencyAccessManager,
        DSARManager,
        DLPScanner,
        ConsentManager,
        AccessReviewManager,
        DPIAManager,
    )
"""

from .rate_limiter import RateLimiter, RateLimitMiddleware, RateLimitConfig
from .token_revocation import TokenRevocationList, TokenRevocationMiddleware
from .audit import SecurityAuditLog, SecurityEvent, AuditEventType
from .secure_memory import SecureMemory, secure_zero, secure_random
from .request_signing import RequestSigner, ReplayProtection, NonceStore
from .crypto_utils import constant_time_compare, secure_hash, generate_secure_token
from .sanitization import (
    sanitize_input,
    sanitize_path,
    sanitize_html,
    InputValidator,
    ValidationMiddleware,
)
from .csp import ContentSecurityPolicy, CSPMiddleware
from .key_rotation import KeyRotationScheduler, RotationPolicy

# Compliance modules
from .mfa import (
    MFAManager,
    MFAEnrollment,
    MFAMethod,
    MFAStatus,
    generate_totp_secret,
    verify_totp,
    generate_backup_codes,
    get_mfa_manager,
)
from .emergency_access import (
    EmergencyAccessManager,
    EmergencyAccessRequest,
    EmergencyToken,
    EmergencyAccessReason,
    get_emergency_access_manager,
)
from .dsar import (
    DSARManager,
    DSARRequest,
    DSARType,
    DSARStatus,
    get_dsar_manager,
)
from .dlp import (
    DLPScanner,
    DLPPattern,
    DLPMatch,
    DLPScanResult,
    DLPAction,
    get_dlp_scanner,
)
from .consent_management import (
    ConsentManager,
    ConsentRecord,
    ConsentPreferences,
    ConsentPurpose,
    get_consent_manager,
)
from .access_review import (
    AccessReviewManager,
    AccessReview,
    AccessEntry,
    AccessAnomaly,
    ReviewStatus,
    get_access_review_manager,
)
from .dpia import (
    DPIAManager,
    DPIAAssessment,
    DPIAStatus,
    RiskLevel,
    RiskAssessment,
    DataCategory,
    ProcessingType,
    ProcessingBasis,
    get_dpia_manager,
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "RateLimitMiddleware",
    "RateLimitConfig",
    # Token revocation
    "TokenRevocationList",
    "TokenRevocationMiddleware",
    # Audit logging
    "SecurityAuditLog",
    "SecurityEvent",
    "AuditEventType",
    # Secure memory
    "SecureMemory",
    "secure_zero",
    "secure_random",
    # Request signing
    "RequestSigner",
    "ReplayProtection",
    "NonceStore",
    # Crypto utilities
    "constant_time_compare",
    "secure_hash",
    "generate_secure_token",
    # Sanitization
    "sanitize_input",
    "sanitize_path",
    "sanitize_html",
    "InputValidator",
    "ValidationMiddleware",
    # CSP
    "ContentSecurityPolicy",
    "CSPMiddleware",
    # Key rotation
    "KeyRotationScheduler",
    "RotationPolicy",
    # MFA
    "MFAManager",
    "MFAEnrollment",
    "MFAMethod",
    "MFAStatus",
    "generate_totp_secret",
    "verify_totp",
    "generate_backup_codes",
    "get_mfa_manager",
    # Emergency Access
    "EmergencyAccessManager",
    "EmergencyAccessRequest",
    "EmergencyToken",
    "EmergencyAccessReason",
    "get_emergency_access_manager",
    # DSAR
    "DSARManager",
    "DSARRequest",
    "DSARType",
    "DSARStatus",
    "get_dsar_manager",
    # DLP
    "DLPScanner",
    "DLPPattern",
    "DLPMatch",
    "DLPScanResult",
    "DLPAction",
    "get_dlp_scanner",
    # Consent Management
    "ConsentManager",
    "ConsentRecord",
    "ConsentPreferences",
    "ConsentPurpose",
    "get_consent_manager",
    # Access Review
    "AccessReviewManager",
    "AccessReview",
    "AccessEntry",
    "AccessAnomaly",
    "ReviewStatus",
    "get_access_review_manager",
    # DPIA
    "DPIAManager",
    "DPIAAssessment",
    "DPIAStatus",
    "RiskLevel",
    "RiskAssessment",
    "DataCategory",
    "ProcessingType",
    "ProcessingBasis",
    "get_dpia_manager",
]
