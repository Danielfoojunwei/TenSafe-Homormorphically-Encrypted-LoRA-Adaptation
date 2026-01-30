
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
from ..evidence.canonical import canonical_bytes
from ..evidence.store import get_store
import secrets


class HESchemeConfig(BaseModel):
    """Homomorphic encryption scheme configuration."""

    scheme_type: str = Field(default="lwe", description="HE scheme: 'lwe', 'rlwe', 'fhew', 'tfhe', 'ckks'")
    lattice_dimension: int = Field(default=1024, description="Lattice dimension (n)")
    ciphertext_modulus_bits: int = Field(default=32, description="Ciphertext modulus (log2 q)")
    plaintext_modulus_bits: int = Field(default=16, description="Plaintext modulus (log2 t)")
    noise_std_dev: float = Field(default=3.2, description="Gaussian noise standard deviation")
    polynomial_degree: int = Field(default=4096, description="Polynomial ring degree (N)")
    coefficient_modulus_bits: List[int] = Field(
        default_factory=lambda: [60, 40, 40, 60],
        description="Coefficient modulus chain (RNS)"
    )
    security_level: int = Field(default=128, description="Security level in bits (NIST standard)")
    bootstrap_enabled: bool = Field(default=False, description="Whether bootstrapping is enabled")


class HEKeyInfo(BaseModel):
    """Information about HE keys used in the package."""

    public_key_fingerprint: Optional[str] = Field(default=None, description="SHA-256 fingerprint of public key")
    eval_key_fingerprint: Optional[str] = Field(default=None, description="SHA-256 fingerprint of evaluation key")
    key_bundle_id: Optional[str] = Field(default=None, description="Key bundle identifier")
    key_generation_timestamp: Optional[str] = Field(default=None, description="ISO timestamp of key generation")
    capabilities: List[str] = Field(
        default_factory=lambda: ["matmul", "add", "scale"],
        description="Supported HE operations"
    )


class EncryptedArtifactInfo(BaseModel):
    """Information about encrypted compute artifacts in the package."""

    artifact_type: str = Field(default="ciphertext", description="Artifact type: 'ciphertext', 'key', 'proof'")
    artifact_id: str = Field(default="", description="Unique artifact identifier")
    content_hash: str = Field(default="", description="SHA-256 hash of artifact content")
    serialization_format: str = Field(default="binary", description="Serialization format: 'binary', 'json', 'cbor'")
    compression: Optional[str] = Field(default=None, description="Compression algorithm if applied")
    size_bytes: int = Field(default=0, description="Size in bytes")
    noise_budget_estimate: Optional[float] = Field(default=None, description="Estimated remaining noise budget (bits)")


class PrivacyClaims(BaseModel):
    """Privacy claims for TGSP manifest when N2HE is enabled."""

    # Basic mode
    mode: str = Field(default="off", description="Privacy mode: 'off' or 'n2he'")
    provider: Optional[str] = Field(default=None, description="Privacy provider (e.g., 'n2he')")
    provider_version: Optional[str] = Field(default=None, description="N2HE provider version")

    # Profile configuration
    profile: Optional[str] = Field(
        default=None,
        description="N2HE profile: 'router_only', 'router_plus_eval', 'encrypted_lora', 'private_inference'"
    )

    # HE scheme configuration (detailed)
    scheme_config: Optional[HESchemeConfig] = Field(
        default=None,
        description="Detailed HE scheme configuration"
    )
    scheme_params_hash: Optional[str] = Field(
        default=None,
        description="Hash of HE scheme parameters for quick verification"
    )

    # Key information
    key_info: Optional[HEKeyInfo] = Field(
        default=None,
        description="Information about HE keys"
    )

    # Encrypted artifacts in this package
    encrypted_artifacts: List[EncryptedArtifactInfo] = Field(
        default_factory=list,
        description="List of encrypted compute artifacts"
    )

    # Deployment metadata
    sidecar_image_digest: Optional[str] = Field(
        default=None,
        description="Docker digest of N2HE sidecar image"
    )
    encrypted_feature_schema_hash: Optional[str] = Field(
        default=None,
        description="Hash of encrypted feature schema"
    )
    router_model_hash: Optional[str] = Field(
        default=None,
        description="Hash of encrypted router model (if applicable)"
    )

    # Audit and compliance
    privacy_receipt_hash: Optional[str] = Field(
        default=None,
        description="Hash of privacy computation receipt"
    )
    compliance_evidence_ref: Optional[str] = Field(
        default=None,
        description="Reference to compliance evidence pack"
    )

    def has_encrypted_lora(self) -> bool:
        """Check if package contains encrypted LoRA adapters."""
        return self.profile == "encrypted_lora" or any(
            a.artifact_type == "ciphertext" for a in self.encrypted_artifacts
        )

    def get_total_encrypted_size(self) -> int:
        """Get total size of all encrypted artifacts."""
        return sum(a.size_bytes for a in self.encrypted_artifacts)


class PackageManifest(BaseModel):
    tgsp_version: str = "0.2"
    package_id: str = Field(default_factory=lambda: secrets.token_hex(8))
    model_name: str = "unknown"
    model_version: str = "0.0.1"
    author_id: str = "anonymous"
    producer_pubkey_ed25519: Optional[str] = None # Base64 encoded
    created_at: float = Field(default_factory=time.time)
    
    payload_hash: str = "pending" # SHA-256 of encrypted payload (or compressed if v0.1)
    
    content_index: List[Dict[str, str]] = [] # [{path, sha256}]
    
    policy_constraints: Dict[str, Any] = {}
    build_info: Dict[str, str] = {}
    compat_base_model_id: List[str] = [] # For backward compatibility
    
    # Privacy claims for N2HE integration
    privacy: PrivacyClaims = Field(default_factory=PrivacyClaims, description="Privacy claims (N2HE)")
    
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.model_dump())
        
    def to_canonical_cbor(self) -> bytes:
        return self.canonical_bytes()
        
    def get_hash(self) -> str:
        import hashlib
        return hashlib.sha256(self.canonical_bytes()).hexdigest()
    
    def is_privacy_enabled(self) -> bool:
        """Check if privacy mode is enabled."""
        return self.privacy.mode == "n2he"
