"""
TenSafe Unified HE Module.

Provides a unified interface to homomorphic encryption backends:
- N2HE (LWE/RLWE, development)
- N2HE-HEXL (CKKS with Intel HEXL, production)
- Toy mode (NOT SECURE, testing only)

Usage:
    from tensafe.he import get_backend, HEParams

    # Auto-select best backend
    backend = get_backend()

    # Explicit backend selection
    backend = get_backend(backend_type="hexl")

    # Operations
    ct = backend.encrypt(plaintext)
    result = backend.lora_delta(ct, lora_a, lora_b)
    plaintext = backend.decrypt(result)
"""

from tensafe.core.he_interface import (
    HEBackendType,
    HEScheme,
    HEParams,
    HEMetrics,
    HEBackendInterface,
    ToyHEBackend,
    N2HEBackendWrapper,
    HEXLBackendWrapper,
    get_backend,
    is_backend_available,
    list_available_backends,
)

__all__ = [
    "HEBackendType",
    "HEScheme",
    "HEParams",
    "HEMetrics",
    "HEBackendInterface",
    "ToyHEBackend",
    "N2HEBackendWrapper",
    "HEXLBackendWrapper",
    "get_backend",
    "is_backend_available",
    "list_available_backends",
]
