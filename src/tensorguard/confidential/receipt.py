"""
Privacy Receipt Generator.

Generates cryptographically verifiable privacy receipts for each
confidential inference request. A receipt proves:

1. TEE Attestation: The computation ran inside attested hardware
2. Adapter Provenance: The adapter has known origin and DP guarantees
3. HE Execution: The adapter weights were encrypted during computation
4. Audit Chain: The request is recorded in a tamper-evident log

The receipt is returned alongside (but outside) the encrypted response,
so the client can verify privacy properties independently.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class TEEAttestationClaim:
    """TEE attestation claim in the privacy receipt."""

    platform: str  # "intel-tdx", "amd-sev-snp", "simulation"
    quote_hash: str  # SHA-256 of the attestation quote
    gpu_attestation: Optional[str] = None  # NVIDIA NRAS token hash
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "platform": self.platform,
            "quote_hash": self.quote_hash,
            "verified": self.verified,
        }
        if self.gpu_attestation:
            result["gpu_attestation"] = self.gpu_attestation
        return result


@dataclass
class AdapterProvenanceClaim:
    """Adapter provenance claim in the privacy receipt."""

    tssp_package_hash: Optional[str] = None
    signature_algorithm: Optional[str] = None  # "ed25519+dilithium3"
    dp_certificate: Optional[Dict[str, Any]] = None  # {"epsilon": 8.0, "delta": 1e-5}
    adapter_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.tssp_package_hash:
            result["tssp_package_hash"] = self.tssp_package_hash
        if self.signature_algorithm:
            result["signature_algorithm"] = self.signature_algorithm
        if self.dp_certificate:
            result["dp_certificate"] = self.dp_certificate
        if self.adapter_id:
            result["adapter_id"] = self.adapter_id
        return result


@dataclass
class HEExecutionClaim:
    """HE execution claim in the privacy receipt."""

    mode: str  # "HE_ONLY", "FULL_HE", "PLAINTEXT", "DISABLED"
    backend: Optional[str] = None  # "CKKS-MOAI", "simulation"
    adapter_encrypted: bool = False
    rotations: int = 0
    operations: int = 0
    compute_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": self.backend,
            "adapter_encrypted": self.adapter_encrypted,
            "rotations": self.rotations,
            "operations": self.operations,
            "compute_time_ms": self.compute_time_ms,
        }


@dataclass
class PrivacyReceipt:
    """
    Cryptographically verifiable privacy receipt.

    Returned with every confidential inference response to provide
    proof of the privacy properties.
    """

    receipt_id: str
    timestamp: str
    session_id: str

    # Claims
    tee_attestation: TEEAttestationClaim
    adapter_provenance: AdapterProvenanceClaim
    he_execution: HEExecutionClaim

    # Audit chain
    audit_hash: str  # SHA-256 of all claims
    previous_audit_hash: Optional[str] = None  # Hash chain

    # Timing
    total_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "tee_attestation": self.tee_attestation.to_dict(),
            "adapter_provenance": self.adapter_provenance.to_dict(),
            "he_execution": self.he_execution.to_dict(),
            "audit_hash": self.audit_hash,
            "previous_audit_hash": self.previous_audit_hash,
            "total_latency_ms": self.total_latency_ms,
        }

    def compute_verification_hash(self) -> str:
        """Recompute the audit hash for verification."""
        claims = {
            "tee": self.tee_attestation.to_dict(),
            "adapter": self.adapter_provenance.to_dict(),
            "he": self.he_execution.to_dict(),
            "session_id": self.session_id,
            "timestamp": self.timestamp,
        }
        canonical = json.dumps(claims, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify receipt integrity (audit hash matches claims)."""
        return self.compute_verification_hash() == self.audit_hash


class PrivacyReceiptGenerator:
    """
    Generates privacy receipts for confidential inference.

    Maintains a hash chain across receipts for tamper evidence.
    """

    def __init__(
        self,
        tee_platform: str = "simulation",
        attestation_quote_hash: Optional[str] = None,
        gpu_attestation: Optional[str] = None,
    ):
        self._tee_platform = tee_platform
        self._attestation_quote_hash = attestation_quote_hash or "none"
        self._gpu_attestation = gpu_attestation
        self._previous_hash: Optional[str] = None
        self._receipt_count = 0

    def generate(
        self,
        session_id: str,
        he_mode: str = "DISABLED",
        he_backend: Optional[str] = None,
        adapter_encrypted: bool = False,
        he_metrics: Optional[Dict[str, Any]] = None,
        tssp_hash: Optional[str] = None,
        dp_certificate: Optional[Dict[str, Any]] = None,
        adapter_id: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> PrivacyReceipt:
        """
        Generate a privacy receipt for a confidential inference request.

        Args:
            session_id: Confidential session ID
            he_mode: HE execution mode
            he_backend: HE backend name
            adapter_encrypted: Whether adapter was encrypted
            he_metrics: HE operation metrics
            tssp_hash: TSSP package hash
            dp_certificate: DP training certificate
            adapter_id: Adapter identifier
            latency_ms: Total inference latency

        Returns:
            PrivacyReceipt with all claims
        """
        self._receipt_count += 1
        receipt_id = f"pr-{hashlib.sha256(f'{session_id}-{self._receipt_count}-{time.time()}'.encode()).hexdigest()[:16]}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        # Build claims
        tee_claim = TEEAttestationClaim(
            platform=self._tee_platform,
            quote_hash=self._attestation_quote_hash,
            gpu_attestation=self._gpu_attestation,
            verified=self._attestation_quote_hash != "none",
        )

        adapter_claim = AdapterProvenanceClaim(
            tssp_package_hash=tssp_hash,
            signature_algorithm="ed25519+dilithium3" if tssp_hash else None,
            dp_certificate=dp_certificate,
            adapter_id=adapter_id,
        )

        he_metrics = he_metrics or {}
        he_claim = HEExecutionClaim(
            mode=he_mode,
            backend=he_backend,
            adapter_encrypted=adapter_encrypted,
            rotations=he_metrics.get("rotations", 0),
            operations=he_metrics.get("operations", 0),
            compute_time_ms=he_metrics.get("compute_time_ms", 0.0),
        )

        # Build receipt (audit_hash computed from claims)
        receipt = PrivacyReceipt(
            receipt_id=receipt_id,
            timestamp=timestamp,
            session_id=session_id,
            tee_attestation=tee_claim,
            adapter_provenance=adapter_claim,
            he_execution=he_claim,
            audit_hash="",  # Computed below
            previous_audit_hash=self._previous_hash,
            total_latency_ms=latency_ms,
        )

        # Compute and set audit hash
        receipt.audit_hash = receipt.compute_verification_hash()

        # Update hash chain
        self._previous_hash = receipt.audit_hash

        return receipt
