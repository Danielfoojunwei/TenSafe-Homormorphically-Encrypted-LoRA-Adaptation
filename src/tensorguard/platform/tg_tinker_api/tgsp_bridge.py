"""
TG-Tinker to TGSP integration module.

Provides utilities to package TG-Tinker training artifacts as TGSP packages
for secure distribution to edge devices.
"""

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .audit import AuditLogger
    from .storage import EncryptedArtifactStore, TinkerArtifact

logger = logging.getLogger(__name__)


class TinkerTGSPBridge:
    """
    Bridge between TG-Tinker training artifacts and TGSP secure packages.

    Provides methods to:
    - Export training checkpoints as TGSP packages
    - Include DP certificates in TGSP evidence
    - Link audit chain to TGSP provenance
    """

    def __init__(
        self,
        artifact_store: "EncryptedArtifactStore",
        audit_logger: "AuditLogger",
    ):
        """
        Initialize the TGSP bridge.

        Args:
            artifact_store: TG-Tinker encrypted artifact store
            audit_logger: TG-Tinker audit logger for provenance
        """
        self.artifact_store = artifact_store
        self.audit_logger = audit_logger

    def create_tgsp_from_checkpoint(
        self,
        artifact: "TinkerArtifact",
        output_path: str,
        signing_key_path: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        policy_path: Optional[str] = None,
        dp_certificate: Optional[Dict[str, Any]] = None,
        optimization_hints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a TGSP package from a TG-Tinker checkpoint artifact.

        Args:
            artifact: TinkerArtifact from save_state()
            output_path: Path for the output .tgsp file
            signing_key_path: Path to producer signing key
            recipients: List of recipient public key paths (format: "id:path")
            policy_path: Path to OPA/Rego policy file
            dp_certificate: Differential privacy certificate data
            optimization_hints: Hardware optimization metadata

        Returns:
            Package ID of the created TGSP
        """
        try:
            from tensorguard.tgsp.service import TGSPService
        except ImportError:
            raise ImportError("tensorguard.tgsp module required for TGSP packaging")

        # Decrypt the artifact to get plaintext weights
        plaintext = self.artifact_store.load_artifact(artifact)

        # Create temporary directory for packaging
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write weights to temp file
            weights_path = tmp_path / "weights.bin"
            weights_path.write_bytes(plaintext)

            # Create evidence.json from audit chain
            evidence = self._create_evidence(artifact, dp_certificate)
            evidence_path = tmp_path / "evidence.json"
            evidence_path.write_text(json.dumps(evidence, indent=2, default=str))

            # Create DP certificate if provided
            if dp_certificate:
                dp_cert_path = tmp_path / "dp_certificate.json"
                dp_cert_path.write_text(json.dumps(dp_certificate, indent=2, default=str))

            # Create optimization.json if hints provided
            if optimization_hints:
                opt_path = tmp_path / "optimization.json"
                opt_path.write_text(json.dumps(optimization_hints, indent=2))

            # Build payloads list
            payloads = [
                f"adapter:weights:{weights_path}",
                f"evidence:metadata:{evidence_path}",
            ]

            if dp_certificate:
                payloads.append(f"dp_certificate:metadata:{dp_cert_path}")

            if optimization_hints:
                payloads.append(f"optimization:metadata:{opt_path}")

            # Create TGSP package
            service = TGSPService()
            package_id = service.create_package(
                out_path=output_path,
                signing_key_path=signing_key_path,
                payloads=payloads,
                policy_path=policy_path,
                recipients=recipients,
                evidence_report=str(evidence_path),
            )

            logger.info(f"Created TGSP package: {output_path}")
            return package_id

    def _create_evidence(
        self,
        artifact: "TinkerArtifact",
        dp_certificate: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create evidence.json from TG-Tinker artifact and audit chain.

        Args:
            artifact: Source TinkerArtifact
            dp_certificate: Optional DP certificate data

        Returns:
            Evidence dictionary for TGSP package
        """
        # Get audit logs for this training client
        logs = self.audit_logger.get_logs(
            training_client_id=artifact.training_client_id,
            limit=10000,
        )

        # Compute audit chain hash
        if logs:
            first_hash = logs[0].record_hash
            last_hash = logs[-1].record_hash
        else:
            first_hash = "none"
            last_hash = "none"

        # Build evidence
        evidence = {
            "training_run_id": artifact.training_client_id,
            "artifact_id": artifact.id,
            "artifact_type": artifact.artifact_type,
            "content_hash": artifact.content_hash,
            "created_at": artifact.created_at.isoformat(),
            "tenant_id": artifact.tenant_id,
            "encryption_algorithm": artifact.encryption_algorithm,
            "audit_chain": {
                "first_entry_hash": first_hash,
                "last_entry_hash": last_hash,
                "total_entries": len(logs),
            },
        }

        # Add DP metrics if available
        if dp_certificate:
            evidence["privacy"] = {
                "epsilon": dp_certificate.get("total_epsilon"),
                "delta": dp_certificate.get("total_delta"),
                "accountant_type": dp_certificate.get("accountant_type"),
                "noise_multiplier": dp_certificate.get("noise_multiplier"),
                "max_grad_norm": dp_certificate.get("max_grad_norm"),
            }

        # Add metadata from artifact
        if artifact.metadata_json:
            evidence["training_metadata"] = artifact.metadata_json

        return evidence

    def verify_tgsp_provenance(
        self,
        package_path: str,
        public_key_path: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Verify TGSP package and extract provenance information.

        Args:
            package_path: Path to .tgsp file
            public_key_path: Path to producer public key

        Returns:
            Tuple of (is_valid, message, evidence_data)
        """
        try:
            from tensorguard.tgsp.service import TGSPService
        except ImportError:
            return False, "tensorguard.tgsp module not available", None

        service = TGSPService()

        # Verify package signature
        is_valid, msg = service.verify_package(package_path, public_key_path)
        if not is_valid:
            return False, msg, None

        # Extract evidence if available
        evidence = None
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # Try to read evidence without decryption (public metadata)
                from tensorguard.tgsp.format import read_tgsp_header

                header = read_tgsp_header(package_path)
                if header and "evidence" in header:
                    evidence = header.get("evidence")
            except Exception as e:
                logger.warning(f"Could not extract evidence: {e}")

        return True, "Package signature valid", evidence


def create_dp_certificate(
    training_client_id: str,
    total_epsilon: float,
    total_delta: float,
    accountant_type: str = "rdp",
    num_training_steps: int = 0,
    sample_rate: float = 0.0,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    audit_chain_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a DP certificate for inclusion in TGSP packages.

    Args:
        training_client_id: TG-Tinker training client ID
        total_epsilon: Total privacy budget spent (epsilon)
        total_delta: Delta parameter
        accountant_type: Type of privacy accountant used
        num_training_steps: Number of training steps
        sample_rate: Batch sampling rate
        noise_multiplier: Gaussian noise multiplier
        max_grad_norm: Gradient clipping norm
        audit_chain_hash: Hash of the audit chain for verification

    Returns:
        DP certificate dictionary
    """
    import secrets

    return {
        "certificate_id": f"dpc-{secrets.token_hex(8)}",
        "training_client_id": training_client_id,
        "accountant_type": accountant_type,
        "total_epsilon": total_epsilon,
        "total_delta": total_delta,
        "composition_method": f"{accountant_type}_to_dp_conversion",
        "num_training_steps": num_training_steps,
        "sample_rate": sample_rate,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "audit_chain_hash": audit_chain_hash,
        "issued_at": datetime.utcnow().isoformat(),
        "signed_by": "tg-tinker-api",
    }
