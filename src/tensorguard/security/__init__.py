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

Usage:
    from tensorguard.security import (
        RateLimiter,
        TokenRevocationList,
        SecurityAuditLog,
        SecureMemory,
        RequestSigner,
        constant_time_compare,
        sanitize_input,
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
]
