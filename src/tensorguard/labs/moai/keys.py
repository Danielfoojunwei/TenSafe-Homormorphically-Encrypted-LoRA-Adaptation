"""
MOAI Key Management System (CKKS)
Distinguishes between Training Keys (N2HE) and Inference Keys (CKKS).
"""

import os
import json
import base64
import secrets
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

from .moai_config import MoaiConfig
from ..core.keys import vault, KeyScope
from ..utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)

@dataclass
class CkksKeyMetadata:
    """Metadata for a tenant's FHE evaluation keys."""
    key_id: str
    tenant_id: str
    created_at: str
    poly_modulus_degree: int
    has_relin_keys: bool
    has_galois_keys: bool
    version: str = "1.0.0"

class MoaiKeyManager:
    """
    Manages generation, storage, and loading of CKKS keys via Unified Vault.
    """
    
    def __init__(self, key_store_path: str = None):
        # We now use the unified vault by default
        self.vault = vault
        self.scope = KeyScope.INFERENCE

    def generate_keypair(self, tenant_id: str, config: MoaiConfig) -> Tuple[str, bytes, bytes, bytes]:
        import tenseal as ts
        key_id = f"moai_{secrets.token_hex(8)}"
        
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.poly_modulus_degree,
            coeff_mod_bit_sizes=config.coeff_modulus_bit_sizes
        )
        ctx.global_scale = config.scale
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        
        # Serialize
        secret_ctx = ctx.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True)
        public_ctx = ctx.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True)
        eval_k = b"" 
        
        # Save to Vault
        self.vault.save_key_artifact(
            scope=self.scope,
            name=key_id,
            data=public_ctx,
            algorithm="CKKS-TenSEAL",
            params={
                "tenant_id": tenant_id,
                "poly_modulus_degree": config.poly_modulus_degree,
                "has_relin_keys": True,
                "has_galois_keys": True
            },
            suffix=".pub"
        )
        
        # Also save secret ctx if required (usually client side)
        self.vault.save_key_artifact(
            scope=self.scope,
            name=key_id,
            data=secret_ctx,
            algorithm="CKKS-TenSEAL",
            suffix=".secret"
        )
        
        return key_id, public_ctx, secret_ctx, eval_k

    def load_metadata(self, key_id: str) -> Optional[CkksKeyMetadata]:
        try:
            _, meta = self.vault.load_key_artifact(self.scope, key_id, suffix=".pub")
            return CkksKeyMetadata(
                key_id=meta.key_id,
                tenant_id=meta.params.get("tenant_id", "unknown"),
                created_at=meta.created_at,
                poly_modulus_degree=meta.params.get("poly_modulus_degree", 8192),
                has_relin_keys=meta.params.get("has_relin_keys", True),
                has_galois_keys=meta.params.get("has_galois_keys", True)
            )
        except:
            return None

    def save_keys(self, key_id: str, pk: bytes, sk: Optional[bytes] = None, eval_k: Optional[bytes] = None):
        """Save binary key artifacts to vault."""
        self.vault.save_key_artifact(self.scope, key_id, pk, "CKKS-TenSEAL", suffix=".pub")
        if sk:
            self.vault.save_key_artifact(self.scope, key_id, sk, "CKKS-TenSEAL", suffix=".secret")
        if eval_k:
            self.vault.save_key_artifact(self.scope, key_id, eval_k, "CKKS-TenSEAL", suffix=".eval")

    def load_public_context(self, key_id: str) -> Tuple[bytes, bytes]:
        """Load public context (and placeholder eval keys) from vault."""
        pk_data, _ = self.vault.load_key_artifact(self.scope, key_id, suffix=".pub")
        # In TenSEAL, eval keys are often bundled in the public context
        try:
            eval_data, _ = self.vault.load_key_artifact(self.scope, key_id, suffix=".eval")
        except FileNotFoundError:
            if is_production():
                raise ProductionGateError(
                    gate_name="MISSING_EVAL_KEYS",
                    message="MOAI eval keys are required in production.",
                    remediation="Generate and store eval keys in the vault with suffix .eval.",
                )
            logger.warning("Eval keys missing for %s; falling back to empty keys in development.", key_id)
            eval_data = b"" # Fallback to empty if not explicitly saved
            
        return pk_data, eval_data
