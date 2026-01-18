"""
PrivacyCore - Privacy and Encryption Interface

Provides a unified interface for privacy-preserving operations including
feature encryption, encrypted inference, and privacy receipt generation.
Supports N2HE (Homomorphic Encryption) as the primary privacy provider.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import time
import json


class PrivacyMode(str, Enum):
    """Privacy operation modes."""
    OFF = "off"
    N2HE = "n2he"


class N2HEProfile(str, Enum):
    """N2HE feature profiles."""
    ROUTER_ONLY = "router_only"
    ROUTER_PLUS_EVAL = "router_plus_eval"


@dataclass
class EncryptionResult:
    """Result of an encryption operation."""
    success: bool
    ciphertext: Optional[bytes] = None
    ciphertext_hash: Optional[str] = None
    scheme_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "ciphertext_hash": self.ciphertext_hash,
            "scheme_id": self.scheme_id,
            "error": self.error,
        }


@dataclass
class InferenceResult:
    """Result of encrypted inference."""
    success: bool
    encrypted_output: Optional[bytes] = None
    output_hash: Optional[str] = None
    confidence: Optional[float] = None
    adapter_id: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_hash": self.output_hash,
            "confidence": self.confidence,
            "adapter_id": self.adapter_id,
            "error": self.error,
        }


@dataclass
class DecryptionResult:
    """Result of decryption operation."""
    success: bool
    plaintext: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class PrivacyReceipt:
    """
    Privacy receipt documenting a privacy-preserving operation.
    
    Included in TGSP packages and evidence chains to prove
    that operations were performed in privacy-preserving mode.
    """
    provider: str
    provider_version: str
    scheme_profile: str
    scheme_params_hash: str
    operation: str  # "encrypt", "infer", "eval"
    input_hash: str
    output_hash: str
    timestamp: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_version": self.provider_version,
            "scheme_profile": self.scheme_profile,
            "scheme_params_hash": self.scheme_params_hash,
            "operation": self.operation,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    def get_hash(self) -> str:
        """Get hash of this receipt for evidence chain."""
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class PrivacyCoreProvider(ABC):
    """Abstract base class for privacy providers."""
    
    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        pass
    
    @property
    @abstractmethod
    def provider_version(self) -> str:
        """Version of this provider."""
        pass
    
    @abstractmethod
    def encrypt_features(
        self, 
        feature_vector: List[float],
        context_id: Optional[str] = None
    ) -> EncryptionResult:
        """
        Encrypt a feature vector for privacy-preserving routing.
        
        Args:
            feature_vector: Numeric feature vector (e.g., embedding)
            context_id: Optional context identifier for key management
            
        Returns:
            EncryptionResult with ciphertext or error
        """
        pass
    
    @abstractmethod
    def infer_encrypted(
        self,
        ciphertext: bytes,
        model_id: Optional[str] = None
    ) -> InferenceResult:
        """
        Perform inference on encrypted features.
        
        Args:
            ciphertext: Encrypted feature vector
            model_id: Optional model/router identifier
            
        Returns:
            InferenceResult with encrypted output or adapter decision
        """
        pass
    
    @abstractmethod
    def decrypt_prediction(
        self,
        encrypted_prediction: bytes,
        context_id: Optional[str] = None
    ) -> DecryptionResult:
        """
        Decrypt a prediction (optional, for authorized parties only).
        
        Args:
            encrypted_prediction: Encrypted prediction result
            context_id: Context for key retrieval
            
        Returns:
            DecryptionResult with plaintext or error
        """
        pass
    
    @abstractmethod
    def emit_privacy_receipt(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PrivacyReceipt:
        """
        Generate a privacy receipt for an operation.
        
        Args:
            operation: Type of operation performed
            input_hash: Hash of input data
            output_hash: Hash of output data
            metadata: Additional operation metadata
            
        Returns:
            PrivacyReceipt documenting the operation
        """
        pass
    
    @abstractmethod
    def get_scheme_params_hash(self) -> str:
        """Get hash of current encryption scheme parameters."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available/configured."""
        pass


class PrivacyCore:
    """
    Main PrivacyCore interface for privacy-preserving operations.
    
    Manages provider selection and provides high-level operations
    for encrypted routing and evaluation.
    """
    
    _providers: Dict[str, PrivacyCoreProvider] = {}
    _active_provider: Optional[str] = None
    _mode: PrivacyMode = PrivacyMode.OFF
    _profile: N2HEProfile = N2HEProfile.ROUTER_ONLY
    
    @classmethod
    def register_provider(cls, provider: PrivacyCoreProvider) -> None:
        """Register a privacy provider."""
        cls._providers[provider.provider_id] = provider
    
    @classmethod
    def set_mode(cls, mode: PrivacyMode, profile: N2HEProfile = N2HEProfile.ROUTER_ONLY) -> None:
        """Set the privacy mode and profile."""
        cls._mode = mode
        cls._profile = profile
        if mode == PrivacyMode.N2HE and "n2he" in cls._providers:
            cls._active_provider = "n2he"
    
    @classmethod
    def get_mode(cls) -> PrivacyMode:
        """Get current privacy mode."""
        return cls._mode
    
    @classmethod
    def is_enabled(cls) -> bool:
        """Check if privacy mode is enabled."""
        return cls._mode == PrivacyMode.N2HE
    
    @classmethod
    def get_provider(cls, provider_id: Optional[str] = None) -> PrivacyCoreProvider:
        """Get a specific or the active provider."""
        pid = provider_id or cls._active_provider
        if not pid:
            raise ValueError("No active provider set. Enable privacy mode first.")
        if pid not in cls._providers:
            raise ValueError(f"Provider '{pid}' not registered.")
        return cls._providers[pid]
    
    @classmethod
    def encrypt_features(cls, feature_vector: List[float], context_id: Optional[str] = None) -> EncryptionResult:
        """Encrypt features for privacy-preserving routing."""
        if not cls.is_enabled():
            return EncryptionResult(success=False, error="Privacy mode not enabled")
        return cls.get_provider().encrypt_features(feature_vector, context_id)
    
    @classmethod
    def infer_encrypted(cls, ciphertext: bytes, model_id: Optional[str] = None) -> InferenceResult:
        """Perform encrypted inference."""
        if not cls.is_enabled():
            return InferenceResult(success=False, error="Privacy mode not enabled")
        return cls.get_provider().infer_encrypted(ciphertext, model_id)
    
    @classmethod
    def emit_privacy_receipt(
        cls,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[PrivacyReceipt]:
        """Generate privacy receipt if privacy mode is enabled."""
        if not cls.is_enabled():
            return None
        return cls.get_provider().emit_privacy_receipt(operation, input_hash, output_hash, metadata)
    
    @classmethod
    def list_providers(cls) -> Dict[str, bool]:
        """List registered providers and their availability."""
        return {
            pid: provider.is_available()
            for pid, provider in cls._providers.items()
        }
