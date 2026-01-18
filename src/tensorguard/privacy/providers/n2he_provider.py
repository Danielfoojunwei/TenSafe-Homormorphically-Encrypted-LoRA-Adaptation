"""
N2HE Provider - Homomorphic Encryption Privacy Provider

Implements PrivacyCoreProvider using N2HE (a simplified HE scheme)
or TenSEAL for production use. Supports both embedded and sidecar modes.
"""

import os
import hashlib
import time
import json
import logging
from typing import Optional, List, Dict, Any

from ..privacy_core import (
    PrivacyCoreProvider,
    EncryptionResult,
    InferenceResult,
    DecryptionResult,
    PrivacyReceipt,
)

logger = logging.getLogger(__name__)

# Version info
N2HE_PROVIDER_VERSION = "1.0.0"


class N2HEProvider(PrivacyCoreProvider):
    """
    N2HE (Homomorphic Encryption) privacy provider.
    
    Supports two deployment modes:
    A) Embedded Python binding (if tenseal/n2he available)
    B) Sidecar service (HTTP/gRPC to external service)
    """
    
    def __init__(
        self,
        mode: str = "embedded",  # "embedded" or "sidecar"
        sidecar_url: Optional[str] = None,
        scheme: str = "bfv",  # "bfv" or "ckks"
        poly_modulus_degree: int = 4096,
        coeff_mod_bit_sizes: Optional[List[int]] = None,
    ):
        self._mode = mode
        self._sidecar_url = sidecar_url or os.getenv("N2HE_SIDECAR_URL", "http://localhost:8765")
        self._scheme = scheme
        self._poly_modulus_degree = poly_modulus_degree
        self._coeff_mod_bit_sizes = coeff_mod_bit_sizes or [40, 20, 40]
        
        self._context = None
        self._public_key = None
        self._secret_key = None
        self._scheme_params_hash = None
        
        if mode == "embedded":
            self._init_embedded()
    
    @property
    def provider_id(self) -> str:
        return "n2he"
    
    @property
    def provider_version(self) -> str:
        return N2HE_PROVIDER_VERSION
    
    def _init_embedded(self) -> None:
        """Initialize embedded TenSEAL context."""
        try:
            import tenseal as ts
            
            if self._scheme == "bfv":
                self._context = ts.context(
                    ts.SCHEME_TYPE.BFV,
                    poly_modulus_degree=self._poly_modulus_degree,
                    plain_modulus=1032193,
                    coeff_mod_bit_sizes=self._coeff_mod_bit_sizes,
                )
            else:  # ckks
                self._context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=self._poly_modulus_degree,
                    coeff_mod_bit_sizes=self._coeff_mod_bit_sizes,
                )
                self._context.global_scale = 2**20
            
            self._context.generate_galois_keys()
            self._context.generate_relin_keys()
            
            # Compute scheme params hash
            params_str = f"{self._scheme}:{self._poly_modulus_degree}:{self._coeff_mod_bit_sizes}"
            self._scheme_params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
            
            logger.info(f"N2HE embedded mode initialized with {self._scheme} scheme")
            
        except ImportError:
            logger.warning("TenSEAL not available - N2HE provider will use sidecar mode")
            self._mode = "sidecar"
            self._scheme_params_hash = "sidecar-mode"
        except Exception as e:
            logger.error(f"Failed to initialize TenSEAL context: {e}")
            self._context = None
    
    def _call_sidecar(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP call to sidecar service."""
        import requests
        
        url = f"{self._sidecar_url}{endpoint}"
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def encrypt_features(
        self,
        feature_vector: List[float],
        context_id: Optional[str] = None
    ) -> EncryptionResult:
        """Encrypt feature vector for privacy-preserving routing."""
        
        if self._mode == "embedded" and self._context:
            try:
                import tenseal as ts
                
                # Encrypt using CKKS for floating point
                encrypted = ts.ckks_vector(self._context, feature_vector)
                ciphertext = encrypted.serialize()
                ciphertext_hash = hashlib.sha256(ciphertext).hexdigest()
                
                return EncryptionResult(
                    success=True,
                    ciphertext=ciphertext,
                    ciphertext_hash=ciphertext_hash,
                    scheme_id=f"n2he-{self._scheme}",
                )
            except Exception as e:
                return EncryptionResult(success=False, error=str(e))
        
        elif self._mode == "sidecar":
            result = self._call_sidecar("/encrypt", {
                "features": feature_vector,
                "context_id": context_id,
            })
            if "error" in result:
                return EncryptionResult(success=False, error=result["error"])
            return EncryptionResult(
                success=True,
                ciphertext=bytes.fromhex(result.get("ciphertext_hex", "")),
                ciphertext_hash=result.get("ciphertext_hash"),
                scheme_id=result.get("scheme_id", "n2he-sidecar"),
            )
        
        return EncryptionResult(success=False, error="No encryption backend available")
    
    def infer_encrypted(
        self,
        ciphertext: bytes,
        model_id: Optional[str] = None
    ) -> InferenceResult:
        """Perform inference on encrypted features."""
        
        if self._mode == "embedded" and self._context:
            try:
                import tenseal as ts
                
                # Deserialize and perform simple routing inference
                # In production, this would use an encrypted routing model
                encrypted_vec = ts.ckks_vector_from(self._context, ciphertext)
                
                # Placeholder: compute sum as routing decision
                # Real implementation would use encrypted linear classifier
                encrypted_result = encrypted_vec.sum()
                result_bytes = encrypted_result.serialize()
                output_hash = hashlib.sha256(result_bytes).hexdigest()
                
                return InferenceResult(
                    success=True,
                    encrypted_output=result_bytes,
                    output_hash=output_hash,
                    adapter_id=None,  # Decision made after decryption
                )
            except Exception as e:
                return InferenceResult(success=False, error=str(e))
        
        elif self._mode == "sidecar":
            result = self._call_sidecar("/infer", {
                "ciphertext_hex": ciphertext.hex(),
                "model_id": model_id,
            })
            if "error" in result:
                return InferenceResult(success=False, error=result["error"])
            return InferenceResult(
                success=True,
                encrypted_output=bytes.fromhex(result.get("output_hex", "")),
                output_hash=result.get("output_hash"),
                confidence=result.get("confidence"),
                adapter_id=result.get("adapter_id"),
            )
        
        return InferenceResult(success=False, error="No inference backend available")
    
    def decrypt_prediction(
        self,
        encrypted_prediction: bytes,
        context_id: Optional[str] = None
    ) -> DecryptionResult:
        """Decrypt prediction result."""
        
        if self._mode == "embedded" and self._context:
            try:
                import tenseal as ts
                
                encrypted = ts.ckks_vector_from(self._context, encrypted_prediction)
                plaintext = encrypted.decrypt()
                
                return DecryptionResult(success=True, plaintext=plaintext)
            except Exception as e:
                return DecryptionResult(success=False, error=str(e))
        
        elif self._mode == "sidecar":
            result = self._call_sidecar("/decrypt", {
                "ciphertext_hex": encrypted_prediction.hex(),
                "context_id": context_id,
            })
            if "error" in result:
                return DecryptionResult(success=False, error=result["error"])
            return DecryptionResult(success=True, plaintext=result.get("plaintext"))
        
        return DecryptionResult(success=False, error="No decryption backend available")
    
    def emit_privacy_receipt(
        self,
        operation: str,
        input_hash: str,
        output_hash: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PrivacyReceipt:
        """Generate privacy receipt for an operation."""
        return PrivacyReceipt(
            provider=self.provider_id,
            provider_version=self.provider_version,
            scheme_profile=self._scheme,
            scheme_params_hash=self._scheme_params_hash or "unknown",
            operation=operation,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    def get_scheme_params_hash(self) -> str:
        """Get hash of encryption scheme parameters."""
        return self._scheme_params_hash or "not-initialized"
    
    def is_available(self) -> bool:
        """Check if N2HE provider is available."""
        if self._mode == "embedded":
            return self._context is not None
        elif self._mode == "sidecar":
            try:
                result = self._call_sidecar("/health", {})
                return result.get("status") == "ok"
            except Exception:
                return False
        return False
    
    def get_health(self) -> Dict[str, Any]:
        """Get provider health status."""
        return {
            "provider": self.provider_id,
            "version": self.provider_version,
            "mode": self._mode,
            "scheme": self._scheme,
            "available": self.is_available(),
            "sidecar_url": self._sidecar_url if self._mode == "sidecar" else None,
        }
