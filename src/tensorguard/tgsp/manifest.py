
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
from ..evidence.canonical import canonical_bytes
from ..evidence.store import get_store
import secrets


class PrivacyClaims(BaseModel):
    """Privacy claims for TGSP manifest when N2HE is enabled."""
    mode: str = Field(default="off", description="Privacy mode: 'off' or 'n2he'")
    provider: Optional[str] = Field(default=None, description="Privacy provider (e.g., 'n2he')")
    profile: Optional[str] = Field(default=None, description="N2HE profile: 'router_only' or 'router_plus_eval'")
    sidecar_image_digest: Optional[str] = Field(default=None, description="Docker digest of N2HE sidecar image")
    encrypted_feature_schema_hash: Optional[str] = Field(default=None, description="Hash of encrypted feature schema")
    router_model_hash: Optional[str] = Field(default=None, description="Hash of encrypted router model (if applicable)")
    provider_version: Optional[str] = Field(default=None, description="N2HE provider version")
    scheme_params_hash: Optional[str] = Field(default=None, description="Hash of HE scheme parameters")


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
