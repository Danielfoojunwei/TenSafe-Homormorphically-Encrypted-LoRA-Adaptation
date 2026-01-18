"""
Software Signer Provider

Software-based signing implementation using Ed25519.
For development and non-TEE production environments.
"""

import os
import base64
import hashlib
import time
from typing import Optional, Dict, Any

from ..trust_core import (
    TrustCoreProvider,
    SignatureResult,
    VerificationResult,
    AttestationReport,
)

# Optional cryptography import
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SoftwareSigner(TrustCoreProvider):
    """
    Software-based Ed25519 signing provider.
    
    Uses local key files for signing operations.
    Suitable for development and non-TEE production deployments.
    """
    
    def __init__(
        self,
        private_key_path: Optional[str] = None,
        public_key_path: Optional[str] = None,
        key_id: Optional[str] = None,
    ):
        self._private_key_path = private_key_path
        self._public_key_path = public_key_path
        self._key_id = key_id or "software-default"
        self._private_key: Optional[Any] = None
        self._public_key: Optional[Any] = None
        
        if CRYPTO_AVAILABLE:
            self._load_keys()
    
    @property
    def provider_id(self) -> str:
        return "software"
    
    def _load_keys(self) -> None:
        """Load keys from files if paths provided."""
        if self._private_key_path and os.path.exists(self._private_key_path):
            with open(self._private_key_path, 'rb') as f:
                key_data = f.read()
                if len(key_data) == 32:
                    # Raw key bytes
                    self._private_key = Ed25519PrivateKey.from_private_bytes(key_data)
                else:
                    # PEM format
                    self._private_key = serialization.load_pem_private_key(key_data, password=None)
                self._public_key = self._private_key.public_key()
        
        elif self._public_key_path and os.path.exists(self._public_key_path):
            with open(self._public_key_path, 'rb') as f:
                key_data = f.read()
                if len(key_data) == 32:
                    self._public_key = Ed25519PublicKey.from_public_bytes(key_data)
                else:
                    self._public_key = serialization.load_pem_public_key(key_data)
    
    def generate_key_pair(self, save_to: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a new Ed25519 key pair.
        
        Args:
            save_to: Optional directory to save keys
            
        Returns:
            Dict with public_key_hex and key_id
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not installed")
        
        self._private_key = Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        
        pub_bytes = self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )
        pub_hex = pub_bytes.hex()
        self._key_id = f"sw-{pub_hex[:16]}"
        
        if save_to:
            os.makedirs(save_to, exist_ok=True)
            priv_path = os.path.join(save_to, f"{self._key_id}.priv")
            pub_path = os.path.join(save_to, f"{self._key_id}.pub")
            
            with open(priv_path, 'wb') as f:
                f.write(self._private_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            with open(pub_path, 'wb') as f:
                f.write(pub_bytes)
            
            self._private_key_path = priv_path
            self._public_key_path = pub_path
        
        return {"public_key_hex": pub_hex, "key_id": self._key_id}
    
    def sign(self, data: bytes, key_id: Optional[str] = None) -> SignatureResult:
        """Sign data with Ed25519."""
        if not CRYPTO_AVAILABLE:
            return SignatureResult(
                success=False,
                error="cryptography library not installed"
            )
        
        if not self._private_key:
            return SignatureResult(
                success=False,
                error="No private key loaded. Call generate_key_pair() or provide key path."
            )
        
        try:
            signature = self._private_key.sign(data)
            return SignatureResult(
                success=True,
                signature=base64.b64encode(signature).decode('utf-8'),
                key_id=key_id or self._key_id,
                algorithm="Ed25519",
                timestamp=time.time(),
            )
        except Exception as e:
            return SignatureResult(success=False, error=str(e))
    
    def verify(self, data: bytes, signature: str, key_id: Optional[str] = None) -> VerificationResult:
        """Verify an Ed25519 signature."""
        if not CRYPTO_AVAILABLE:
            return VerificationResult(valid=False, error="cryptography library not installed")
        
        if not self._public_key:
            return VerificationResult(valid=False, error="No public key loaded")
        
        try:
            sig_bytes = base64.b64decode(signature)
            self._public_key.verify(sig_bytes, data)
            return VerificationResult(
                valid=True,
                key_id=key_id or self._key_id,
                signer_identity=f"software:{self._key_id}",
            )
        except Exception as e:
            return VerificationResult(valid=False, error=str(e))
    
    def get_attestation(self, user_data: Optional[bytes] = None) -> AttestationReport:
        """
        Software provider does not provide TEE attestation.
        Returns a placeholder report for compatibility.
        """
        return AttestationReport(
            provider="software",
            document=None,
            pcrs=None,
            user_data=user_data,
            timestamp=time.time(),
            error="Software provider does not support TEE attestation",
        )
    
    def is_available(self) -> bool:
        """Check if cryptography is available."""
        return CRYPTO_AVAILABLE


# Auto-register provider if TrustCore is available
def _register_software_provider():
    try:
        from ..trust_core import TrustCore
        provider = SoftwareSigner()
        TrustCore.register_provider(provider)
    except Exception:
        pass  # Will be registered when explicitly instantiated
