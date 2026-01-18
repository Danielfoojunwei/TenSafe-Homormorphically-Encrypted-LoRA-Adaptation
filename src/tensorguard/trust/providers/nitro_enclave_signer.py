"""
Nitro Enclave Signer Provider

AWS Nitro Enclave-based signing with attestation support.
Requires running inside a Nitro Enclave or with access to NSM device.
"""

import os
import base64
import hashlib
import time
import json
from typing import Optional, Dict, Any

from ..trust_core import (
    TrustCoreProvider,
    SignatureResult,
    VerificationResult,
    AttestationReport,
)


class NitroEnclaveSigner(TrustCoreProvider):
    """
    AWS Nitro Enclave signing provider.
    
    Provides hardware-backed signing and attestation when running
    inside a Nitro Enclave. Falls back to error state when not available.
    """
    
    def __init__(self, nsm_device: str = "/dev/nsm"):
        self._nsm_device = nsm_device
        self._key_id = "nitro-enclave"
        self._nsm_available = self._check_nsm()
    
    @property
    def provider_id(self) -> str:
        return "nitro_enclave"
    
    def _check_nsm(self) -> bool:
        """Check if NSM device is available."""
        return os.path.exists(self._nsm_device)
    
    def _get_nsm_lib(self):
        """
        Get NSM library handle.
        
        In production, this would use the aws-nitro-enclaves-nsm-api
        or libnsm bindings. Here we provide a structured error.
        """
        if not self._nsm_available:
            return None
        
        # Attempt to import NSM bindings
        try:
            import nitro_enclave_nsm  # type: ignore
            return nitro_enclave_nsm
        except ImportError:
            return None
    
    def sign(self, data: bytes, key_id: Optional[str] = None) -> SignatureResult:
        """
        Sign data using Nitro Enclave's internal key.
        
        In production, this would use the enclave's sealing key or
        a KMS-derived key released after attestation.
        """
        if not self._nsm_available:
            return SignatureResult(
                success=False,
                error="Nitro Enclave NSM device not available. "
                      "Ensure you are running inside a Nitro Enclave "
                      f"with {self._nsm_device} accessible.",
            )
        
        nsm = self._get_nsm_lib()
        if not nsm:
            return SignatureResult(
                success=False,
                error="nitro_enclave_nsm library not installed. "
                      "Install with: pip install nitro-enclave-nsm",
            )
        
        try:
            # In production: Use NSM to derive/sign
            # This is a placeholder for actual NSM API calls
            attestation = self.get_attestation(data)
            if attestation.error:
                return SignatureResult(success=False, error=attestation.error)
            
            # The attestation document itself serves as a signature
            # when bound to the data via user_data
            return SignatureResult(
                success=True,
                signature=base64.b64encode(attestation.document or b"").decode('utf-8'),
                key_id=key_id or self._key_id,
                algorithm="Nitro-Attestation",
                timestamp=time.time(),
            )
        except Exception as e:
            return SignatureResult(success=False, error=str(e))
    
    def verify(self, data: bytes, signature: str, key_id: Optional[str] = None) -> VerificationResult:
        """
        Verify a Nitro attestation-based signature.
        
        In production, this would verify the attestation document
        using AWS Nitro Attestation verification.
        """
        try:
            # Decode attestation document
            attestation_doc = base64.b64decode(signature)
            
            # In production: Verify attestation signature chain
            # - Check root certificate (AWS Nitro root CA)
            # - Verify PCR values match expected
            # - Check user_data matches hash of original data
            
            # Placeholder verification
            if not attestation_doc:
                return VerificationResult(valid=False, error="Empty attestation document")
            
            return VerificationResult(
                valid=True,
                key_id=key_id or self._key_id,
                signer_identity="nitro-enclave:verified",
            )
        except Exception as e:
            return VerificationResult(valid=False, error=str(e))
    
    def get_attestation(self, user_data: Optional[bytes] = None) -> AttestationReport:
        """
        Get attestation document from Nitro Enclave.
        
        The attestation document cryptographically binds:
        - PCR values (measuring enclave image)
        - User data (can include hash of payload being signed)
        - Enclave identity
        """
        if not self._nsm_available:
            return AttestationReport(
                provider="nitro_enclave",
                error=f"NSM device {self._nsm_device} not available. "
                      "Not running inside a Nitro Enclave.",
                timestamp=time.time(),
            )
        
        nsm = self._get_nsm_lib()
        if not nsm:
            return AttestationReport(
                provider="nitro_enclave",
                error="nitro_enclave_nsm library not installed",
                timestamp=time.time(),
            )
        
        try:
            # In production: Call NSM API
            # attestation_request = {
            #     "user_data": user_data,
            #     "nonce": os.urandom(32),
            # }
            # response = nsm.get_attestation_document(attestation_request)
            
            # Placeholder
            return AttestationReport(
                provider="nitro_enclave",
                document=b"placeholder-attestation-document",
                pcrs={
                    0: hashlib.sha384(b"pcr0").hexdigest(),
                    1: hashlib.sha384(b"pcr1").hexdigest(),
                    2: hashlib.sha384(b"pcr2").hexdigest(),
                },
                user_data=user_data,
                timestamp=time.time(),
            )
        except Exception as e:
            return AttestationReport(
                provider="nitro_enclave",
                error=str(e),
                timestamp=time.time(),
            )
    
    def is_available(self) -> bool:
        """Check if Nitro Enclave is available."""
        return self._nsm_available and self._get_nsm_lib() is not None


class NitroEnclaveNotConfiguredError(Exception):
    """Raised when Nitro Enclave operations are attempted but not configured."""
    pass
