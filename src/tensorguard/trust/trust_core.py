"""
TrustCore - Trust and Signing Interface

Provides a unified interface for signing evidence bundles, verifying signatures,
and emitting attestation reports. Supports pluggable providers (software, Nitro Enclave).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import time


class TrustProvider(str, Enum):
    """Supported trust providers."""
    SOFTWARE = "software"
    NITRO_ENCLAVE = "nitro_enclave"


@dataclass
class SignatureResult:
    """Result of a signing operation."""
    success: bool
    signature: Optional[str] = None
    key_id: Optional[str] = None
    algorithm: str = "Ed25519"
    timestamp: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "signature": self.signature,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class VerificationResult:
    """Result of a signature verification."""
    valid: bool
    key_id: Optional[str] = None
    signer_identity: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "key_id": self.key_id,
            "signer_identity": self.signer_identity,
            "error": self.error,
        }


@dataclass
class AttestationReport:
    """TEE attestation report."""
    provider: str
    document: Optional[bytes] = None
    pcrs: Optional[Dict[int, str]] = None
    user_data: Optional[bytes] = None
    timestamp: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "document_hash": hashlib.sha256(self.document).hexdigest() if self.document else None,
            "pcrs": self.pcrs,
            "timestamp": self.timestamp,
            "error": self.error,
        }


class TrustCoreProvider(ABC):
    """Abstract base class for trust providers."""
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        pass
    
    @abstractmethod
    def sign(self, data: bytes, key_id: Optional[str] = None) -> SignatureResult:
        """
        Sign data with the provider's signing key.
        
        Args:
            data: Raw bytes to sign
            key_id: Optional specific key to use
            
        Returns:
            SignatureResult with signature or error
        """
        pass
    
    @abstractmethod
    def verify(self, data: bytes, signature: str, key_id: Optional[str] = None) -> VerificationResult:
        """
        Verify a signature.
        
        Args:
            data: Original data that was signed
            signature: Signature to verify (base64 encoded)
            key_id: Key ID that was used to sign
            
        Returns:
            VerificationResult indicating validity
        """
        pass
    
    @abstractmethod
    def get_attestation(self, user_data: Optional[bytes] = None) -> AttestationReport:
        """
        Get attestation report from TEE.
        
        Args:
            user_data: Optional user data to include in attestation
            
        Returns:
            AttestationReport from the trust environment
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available/configured."""
        pass


class TrustCore:
    """
    Main TrustCore interface for signing and attestation.
    
    Manages provider selection and provides high-level signing operations
    for TGSP manifests, evidence chains, and promotion decisions.
    """
    
    _providers: Dict[str, TrustCoreProvider] = {}
    _active_provider: Optional[str] = None
    
    @classmethod
    def register_provider(cls, provider: TrustCoreProvider) -> None:
        """Register a trust provider."""
        cls._providers[provider.provider_id] = provider
    
    @classmethod
    def set_active_provider(cls, provider_id: str) -> None:
        """Set the active provider for signing operations."""
        if provider_id not in cls._providers:
            available = list(cls._providers.keys())
            raise ValueError(f"Provider '{provider_id}' not registered. Available: {available}")
        cls._active_provider = provider_id
    
    @classmethod
    def get_provider(cls, provider_id: Optional[str] = None) -> TrustCoreProvider:
        """Get a specific or the active provider."""
        pid = provider_id or cls._active_provider
        if not pid:
            raise ValueError("No active provider set. Call set_active_provider() first.")
        if pid not in cls._providers:
            raise ValueError(f"Provider '{pid}' not registered.")
        return cls._providers[pid]
    
    @classmethod
    def sign_manifest(cls, manifest_bytes: bytes) -> SignatureResult:
        """Sign a TGSP manifest."""
        provider = cls.get_provider()
        return provider.sign(manifest_bytes)
    
    @classmethod
    def sign_evidence_chain(cls, chain_head_hash: str) -> SignatureResult:
        """Sign an evidence chain head hash."""
        provider = cls.get_provider()
        return provider.sign(chain_head_hash.encode('utf-8'))
    
    @classmethod
    def sign_promotion_decision(cls, decision: Dict[str, Any]) -> SignatureResult:
        """Sign a promotion/release decision."""
        provider = cls.get_provider()
        canonical = json.dumps(decision, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return provider.sign(canonical)
    
    @classmethod
    def verify_signature(cls, data: bytes, signature: str, key_id: Optional[str] = None) -> VerificationResult:
        """Verify a signature."""
        provider = cls.get_provider()
        return provider.verify(data, signature, key_id)
    
    @classmethod
    def get_attestation(cls, user_data: Optional[bytes] = None) -> AttestationReport:
        """Get attestation from the active provider."""
        provider = cls.get_provider()
        return provider.get_attestation(user_data)
    
    @classmethod
    def list_providers(cls) -> Dict[str, bool]:
        """List registered providers and their availability."""
        return {
            pid: provider.is_available() 
            for pid, provider in cls._providers.items()
        }
