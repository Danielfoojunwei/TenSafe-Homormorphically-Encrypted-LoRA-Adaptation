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

from .audit import AuditEventType, SecurityAuditLog, SecurityEvent
from .crypto_utils import constant_time_compare, generate_secure_token, secure_hash
from .csp import ContentSecurityPolicy, CSPMiddleware
from .key_rotation import KeyRotationScheduler, RotationPolicy
from .rate_limiter import RateLimitConfig, RateLimiter, RateLimitMiddleware
from .request_signing import NonceStore, ReplayProtection, RequestSigner
from .sanitization import (
    InputValidator,
    ValidationMiddleware,
    sanitize_html,
    sanitize_input,
    sanitize_path,
)
from .secure_memory import SecureMemory, secure_random, secure_zero
from .token_revocation import TokenRevocationList, TokenRevocationMiddleware

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
