"""
TG-Tinker: Privacy-First ML Training Platform

A complete privacy-preserving machine learning training system featuring:
- Differential Privacy (DP-SGD) with RDP/Moments/PRV accountants
- AES-256-GCM encrypted artifact storage with per-tenant isolation
- Hash-chained tamper-evident audit logging
- TGSP secure packaging with hybrid PQC signatures (Ed25519 + Dilithium3)
- Async training API with FutureHandle pattern
"""

__version__ = "3.0.0"
__author__ = "Daniel Foo & The TG-Tinker Team"

# Lazy imports to avoid circular dependencies and optional dependencies
__all__ = ["__version__"]

try:
    from .crypto import (
        sign_hybrid,
        verify_hybrid,
        generate_hybrid_keypair,
    )
    __all__.extend(["sign_hybrid", "verify_hybrid", "generate_hybrid_keypair"])
except ImportError:
    # PQC dependencies may not be installed
    sign_hybrid = None
    verify_hybrid = None
    generate_hybrid_keypair = None

try:
    from .tgsp.service import TGSPService
    __all__.append("TGSPService")
except ImportError:
    TGSPService = None

try:
    from .edge import TGSPEdgeClient
    __all__.append("TGSPEdgeClient")
except ImportError:
    TGSPEdgeClient = None
