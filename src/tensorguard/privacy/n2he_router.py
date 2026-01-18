"""
N2HE Router - Privacy-Preserving Adapter Routing

Routes adapter selection without exposing plaintext features.
Uses N2HE to encrypt feature vectors and make routing decisions
on encrypted data.
"""

import hashlib
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .privacy_core import PrivacyCore, PrivacyReceipt, EncryptionResult, InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class N2HERoutingDecision:
    """Result of N2HE-based routing decision."""
    adapter_id: Optional[str]
    confidence: float
    encrypted: bool
    decision_hash: str
    privacy_receipt: Optional[PrivacyReceipt]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "confidence": self.confidence,
            "encrypted": self.encrypted,
            "decision_hash": self.decision_hash,
            "privacy_receipt": self.privacy_receipt.to_dict() if self.privacy_receipt else None,
            "error": self.error,
        }


class N2HERouter:
    """
    Privacy-preserving router using N2HE.
    
    Workflow:
    1. Extract minimal feature vector (embedding, metadata, sensitivity flags)
    2. Encrypt feature vector via N2HE
    3. Perform encrypted routing decision
    4. Return adapter decision with privacy receipt
    """
    
    def __init__(self, adapter_candidates: Optional[List[str]] = None):
        """
        Initialize N2HE router.
        
        Args:
            adapter_candidates: List of adapter IDs available for routing
        """
        self.adapter_candidates = adapter_candidates or []
        self._privacy_enabled = PrivacyCore.is_enabled()
    
    def extract_features(
        self,
        embedding: List[float],
        tenant_id: str,
        route_metadata: Optional[Dict[str, Any]] = None,
        sensitivity_flags: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, Any]:
        """
        Extract minimal feature vector for routing.
        
        Note: Raw prompts are NEVER persisted or logged.
        Only embeddings and metadata are used.
        
        Args:
            embedding: Pre-computed embedding vector
            tenant_id: Tenant identifier
            route_metadata: Routing metadata (tags, context)
            sensitivity_flags: Sensitivity indicators
            
        Returns:
            Feature dict ready for encryption
        """
        feature_vector = {
            "embedding": embedding,
            "tenant_id": tenant_id,
            "metadata": route_metadata or {},
            "sensitivity": sensitivity_flags or {},
        }
        
        # Hash the feature vector for receipts (no plaintext persisted)
        feature_hash = hashlib.sha256(
            str(sorted(embedding[:10])).encode()  # Only hash first 10 dims
        ).hexdigest()
        feature_vector["feature_hash"] = feature_hash
        
        return feature_vector
    
    def route_encrypted(
        self,
        features: Dict[str, Any],
        context_id: Optional[str] = None,
    ) -> N2HERoutingDecision:
        """
        Make routing decision on encrypted features.
        
        Args:
            features: Feature dict from extract_features()
            context_id: Optional context for key management
            
        Returns:
            N2HERoutingDecision with adapter selection
        """
        if not PrivacyCore.is_enabled():
            return N2HERoutingDecision(
                adapter_id=None,
                confidence=0.0,
                encrypted=False,
                decision_hash="",
                privacy_receipt=None,
                error="Privacy mode not enabled. Call PrivacyCore.set_mode(PrivacyMode.N2HE) first.",
            )
        
        embedding = features.get("embedding", [])
        if not embedding:
            return N2HERoutingDecision(
                adapter_id=None,
                confidence=0.0,
                encrypted=False,
                decision_hash="",
                privacy_receipt=None,
                error="No embedding provided in features",
            )
        
        # Step 1: Encrypt features
        enc_result = PrivacyCore.encrypt_features(embedding, context_id)
        if not enc_result.success:
            return N2HERoutingDecision(
                adapter_id=None,
                confidence=0.0,
                encrypted=False,
                decision_hash="",
                privacy_receipt=None,
                error=f"Encryption failed: {enc_result.error}",
            )
        
        # Step 2: Perform encrypted inference for routing
        infer_result = PrivacyCore.infer_encrypted(enc_result.ciphertext)
        if not infer_result.success:
            return N2HERoutingDecision(
                adapter_id=None,
                confidence=0.0,
                encrypted=True,
                decision_hash=enc_result.ciphertext_hash or "",
                privacy_receipt=None,
                error=f"Encrypted inference failed: {infer_result.error}",
            )
        
        # Step 3: Map inference result to adapter decision
        adapter_id = infer_result.adapter_id
        if not adapter_id and self.adapter_candidates:
            # Default to first candidate if inference doesn't return specific adapter
            adapter_id = self.adapter_candidates[0]
        
        # Step 4: Generate privacy receipt
        receipt = PrivacyCore.emit_privacy_receipt(
            operation="routing",
            input_hash=features.get("feature_hash", ""),
            output_hash=infer_result.output_hash or "",
            metadata={
                "adapter_id": adapter_id,
                "route_type": "encrypted",
            },
        )
        
        # Compute decision hash for evidence
        decision_data = f"{adapter_id}:{infer_result.output_hash}"
        decision_hash = hashlib.sha256(decision_data.encode()).hexdigest()
        
        return N2HERoutingDecision(
            adapter_id=adapter_id,
            confidence=infer_result.confidence or 0.95,
            encrypted=True,
            decision_hash=decision_hash,
            privacy_receipt=receipt,
        )
    
    def route_plaintext(
        self,
        features: Dict[str, Any],
    ) -> N2HERoutingDecision:
        """
        Fallback plaintext routing (when privacy mode is off).
        
        Args:
            features: Feature dict from extract_features()
            
        Returns:
            N2HERoutingDecision with adapter selection
        """
        # Simple plaintext routing logic
        embedding = features.get("embedding", [])
        
        # Default decision: first candidate or based on simple heuristic
        adapter_id = self.adapter_candidates[0] if self.adapter_candidates else None
        
        decision_hash = hashlib.sha256(
            f"{adapter_id}:{features.get('feature_hash', '')}".encode()
        ).hexdigest()
        
        return N2HERoutingDecision(
            adapter_id=adapter_id,
            confidence=1.0,
            encrypted=False,
            decision_hash=decision_hash,
            privacy_receipt=None,
        )
    
    def route(
        self,
        embedding: List[float],
        tenant_id: str,
        route_metadata: Optional[Dict[str, Any]] = None,
        sensitivity_flags: Optional[Dict[str, bool]] = None,
        force_encrypted: bool = False,
    ) -> N2HERoutingDecision:
        """
        Main routing entry point.
        
        Automatically selects encrypted or plaintext routing based on
        privacy mode configuration.
        
        Args:
            embedding: Pre-computed embedding vector
            tenant_id: Tenant identifier
            route_metadata: Routing metadata
            sensitivity_flags: Sensitivity indicators
            force_encrypted: Force encrypted routing even if privacy mode is off
            
        Returns:
            N2HERoutingDecision
        """
        features = self.extract_features(
            embedding=embedding,
            tenant_id=tenant_id,
            route_metadata=route_metadata,
            sensitivity_flags=sensitivity_flags,
        )
        
        if PrivacyCore.is_enabled() or force_encrypted:
            return self.route_encrypted(features)
        else:
            return self.route_plaintext(features)
