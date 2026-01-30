"""
TensorGuard Evidence Module

Provides canonical serialization and evidence storage for audit trails
and cryptographic verification of package manifests.
"""

from .canonical import canonical_bytes, canonical_json
from .store import EvidenceStore, get_store

__all__ = [
    "canonical_bytes",
    "canonical_json",
    "EvidenceStore",
    "get_store",
]
