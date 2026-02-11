"""
TenSafe Confidential Inference Module.

Provides TEE-based confidential inference with:
- HPKE session key exchange (X25519-ChaCha20Poly1305)
- Encrypted prompt/output envelopes
- Attestation-gated key release
- Privacy receipts with cryptographic proof chain
"""

from .middleware import ConfidentialInferenceMiddleware
from .receipt import PrivacyReceipt, PrivacyReceiptGenerator
from .session import ConfidentialSession, ConfidentialSessionManager

__all__ = [
    "ConfidentialSessionManager",
    "ConfidentialSession",
    "ConfidentialInferenceMiddleware",
    "PrivacyReceipt",
    "PrivacyReceiptGenerator",
]
