"""
N2HE Key Management Module.

Provides secure key generation, storage, and distribution for
homomorphic encryption operations in TenSafe.

Key Types:
    - SecretKey (sk): For decryption, held by data owner
    - PublicKey (pk): For encryption, distributed to clients
    - EvaluationKey (ek): For homomorphic operations, distributed to servers

Key Flow:
    1. Data owner generates (sk, pk, ek) via HEKeyManager
    2. pk distributed to clients for encrypting activations
    3. ek distributed to TenSafe server for LoRA computation
    4. sk kept secret by data owner for decrypting results
"""

import base64
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .core import HESchemeParams, N2HEContext, create_context

if TYPE_CHECKING:
    from tensorguard.identity.keys.provider import KeyProvider

logger = logging.getLogger(__name__)


@dataclass
class PublicKey:
    """
    HE public key for encryption.

    Safe to distribute to any party that needs to encrypt data.
    """

    key_id: str
    key_bytes: bytes
    params_hash: str  # Hash of HE scheme parameters
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "key_id": self.key_id,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            "params_hash": self.params_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "key_type": "public",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PublicKey":
        """Deserialize from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_bytes=base64.b64decode(data["key_bytes"]),
            params_hash=data["params_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )

    def get_fingerprint(self) -> str:
        """Get key fingerprint for identification."""
        return f"sha256:{hashlib.sha256(self.key_bytes).hexdigest()[:16]}"


@dataclass
class EvaluationKey:
    """
    HE evaluation key for homomorphic operations.

    Enables computation on encrypted data without access to plaintext.
    Should be distributed to computation servers.

    Includes:
        - Relinearization keys (for multiplication)
        - Rotation keys (for SIMD operations)
        - Key-switching keys (for matrix multiplication)
    """

    key_id: str
    key_bytes: bytes
    params_hash: str
    capabilities: list = field(default_factory=lambda: ["matmul", "add", "scale"])
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "key_id": self.key_id,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            "params_hash": self.params_hash,
            "capabilities": self.capabilities,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "key_type": "evaluation",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationKey":
        """Deserialize from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_bytes=base64.b64decode(data["key_bytes"]),
            params_hash=data["params_hash"],
            capabilities=data.get("capabilities", ["matmul", "add", "scale"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
        )

    def get_fingerprint(self) -> str:
        """Get key fingerprint for identification."""
        return f"sha256:{hashlib.sha256(self.key_bytes).hexdigest()[:16]}"


@dataclass
class SecretKey:
    """
    HE secret key for decryption.

    MUST be kept secret by the data owner. Never distribute.
    """

    key_id: str
    key_bytes: bytes  # Always stored encrypted
    params_hash: str
    wrapped: bool = True  # Whether key_bytes is wrapped (encrypted)
    wrap_key_id: Optional[str] = None  # ID of key used for wrapping
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (key_bytes should be wrapped!)."""
        if not self.wrapped:
            raise ValueError("Cannot serialize unwrapped secret key")
        return {
            "key_id": self.key_id,
            "key_bytes": base64.b64encode(self.key_bytes).decode("ascii"),
            "params_hash": self.params_hash,
            "wrapped": True,
            "wrap_key_id": self.wrap_key_id,
            "created_at": self.created_at.isoformat(),
            "key_type": "secret",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecretKey":
        """Deserialize from dictionary."""
        return cls(
            key_id=data["key_id"],
            key_bytes=base64.b64decode(data["key_bytes"]),
            params_hash=data["params_hash"],
            wrapped=data.get("wrapped", True),
            wrap_key_id=data.get("wrap_key_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
        )


@dataclass
class HEKeyBundle:
    """
    Complete HE key bundle containing all key types.

    Typically generated together and then distributed to different parties:
        - SecretKey → Data owner only
        - PublicKey → Clients for encryption
        - EvaluationKey → Computation servers
    """

    bundle_id: str
    params: HESchemeParams
    secret_key: SecretKey
    public_key: PublicKey
    evaluation_key: EvaluationKey
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_manifest_claims(self) -> Dict[str, Any]:
        """Generate claims for TGSP manifest."""
        return {
            "mode": "n2he",
            "provider": "n2he",
            "provider_version": "0.1.0",
            "scheme_params_hash": self.params.get_hash(),
            "public_key_fingerprint": self.public_key.get_fingerprint(),
            "eval_key_fingerprint": self.evaluation_key.get_fingerprint(),
            "capabilities": self.evaluation_key.capabilities,
        }


class HEKeyManager:
    """
    Manages HE key lifecycle: generation, storage, rotation, and distribution.

    Integrates with TenSafe's existing key management infrastructure:
        - KeyManager (DEK/KEK hierarchy)
        - IdentityKeyManager (HSM/KMS integration)
        - Post-quantum key wrapping

    Key Storage Structure:
        {tenant_id}/
            n2he/
                bundles/
                    {bundle_id}.json     # Encrypted bundle metadata
                secret_keys/
                    {key_id}.enc         # Wrapped secret keys
                public_keys/
                    {key_id}.pub         # Public keys (not encrypted)
                eval_keys/
                    {key_id}.ek          # Evaluation keys (not encrypted)
    """

    NONCE_SIZE = 12  # AES-GCM nonce

    def __init__(
        self,
        master_key: Optional[bytes] = None,
        storage_path: Optional[str] = None,
        key_provider: Optional["KeyProvider"] = None,
    ):
        """
        Initialize HE key manager.

        Args:
            master_key: Master key for wrapping secret keys (32 bytes)
            storage_path: Path for key storage
            key_provider: Optional TensorGuard key provider for integration
        """
        if master_key is None:
            # Generate random master key (production should use vault)
            self._master_key = secrets.token_bytes(32)
            logger.warning(
                "Generated ephemeral master key - use vault in production"
            )
        else:
            if len(master_key) != 32:
                raise ValueError("Master key must be 32 bytes")
            self._master_key = master_key

        self._storage_path = Path(storage_path) if storage_path else None
        self._key_provider = key_provider

        # In-memory cache
        self._bundles: Dict[str, HEKeyBundle] = {}
        self._contexts: Dict[str, N2HEContext] = {}

        # Load existing keys
        if self._storage_path:
            self._load_keys()

    def generate_key_bundle(
        self,
        tenant_id: str,
        params: Optional[HESchemeParams] = None,
        bundle_id: Optional[str] = None,
    ) -> HEKeyBundle:
        """
        Generate a complete HE key bundle for a tenant.

        Args:
            tenant_id: Tenant identifier
            params: HE scheme parameters (default: LoRA-optimized)
            bundle_id: Optional bundle ID (auto-generated if not provided)

        Returns:
            HEKeyBundle with all key types
        """
        params = params or HESchemeParams.default_lora_params()
        bundle_id = bundle_id or f"bundle-{secrets.token_hex(8)}"
        key_id_base = f"n2he-{tenant_id}-{bundle_id}"

        # Create context and generate keys
        ctx = create_context(profile="lora", use_simulation=True)
        ctx.generate_keys()

        params_hash = params.get_hash()
        now = datetime.utcnow()

        # Create key objects
        secret_key = SecretKey(
            key_id=f"{key_id_base}-sk",
            key_bytes=self._wrap_key(ctx._sk),
            params_hash=params_hash,
            wrapped=True,
            wrap_key_id="master",
            created_at=now,
        )

        public_key = PublicKey(
            key_id=f"{key_id_base}-pk",
            key_bytes=ctx._pk,
            params_hash=params_hash,
            created_at=now,
        )

        evaluation_key = EvaluationKey(
            key_id=f"{key_id_base}-ek",
            key_bytes=ctx._ek,
            params_hash=params_hash,
            capabilities=["matmul", "add", "scale", "lora_delta"],
            created_at=now,
        )

        bundle = HEKeyBundle(
            bundle_id=bundle_id,
            params=params,
            secret_key=secret_key,
            public_key=public_key,
            evaluation_key=evaluation_key,
            created_at=now,
        )

        # Cache
        self._bundles[bundle_id] = bundle
        self._contexts[bundle_id] = ctx

        # Persist
        if self._storage_path:
            self._save_bundle(tenant_id, bundle)

        logger.info(
            f"Generated HE key bundle {bundle_id} for tenant {tenant_id}: "
            f"pk={public_key.get_fingerprint()}, ek={evaluation_key.get_fingerprint()}"
        )

        return bundle

    def get_bundle(self, bundle_id: str) -> Optional[HEKeyBundle]:
        """Get a key bundle by ID."""
        return self._bundles.get(bundle_id)

    def get_context(
        self,
        bundle_id: str,
        include_secret_key: bool = False,
    ) -> Optional[N2HEContext]:
        """
        Get N2HE context for a bundle.

        Args:
            bundle_id: Bundle ID
            include_secret_key: Whether to load secret key (for decryption)

        Returns:
            Configured N2HEContext
        """
        if bundle_id in self._contexts:
            ctx = self._contexts[bundle_id]
            if include_secret_key and not ctx.has_secret_key:
                # Load and unwrap secret key
                bundle = self._bundles.get(bundle_id)
                if bundle:
                    sk = self._unwrap_key(bundle.secret_key.key_bytes)
                    ctx.load_keys(sk=sk)
            return ctx

        # Try to load from storage
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return None

        ctx = create_context(profile="lora", use_simulation=True)
        ctx.load_keys(
            pk=bundle.public_key.key_bytes,
            ek=bundle.evaluation_key.key_bytes,
        )

        if include_secret_key:
            sk = self._unwrap_key(bundle.secret_key.key_bytes)
            ctx.load_keys(sk=sk)

        self._contexts[bundle_id] = ctx
        return ctx

    def export_public_key(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        """Export public key for client distribution."""
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return None
        return bundle.public_key.to_dict()

    def export_eval_key(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        """Export evaluation key for server distribution."""
        bundle = self._bundles.get(bundle_id)
        if bundle is None:
            return None
        return bundle.evaluation_key.to_dict()

    def rotate_keys(
        self,
        tenant_id: str,
        old_bundle_id: str,
    ) -> HEKeyBundle:
        """
        Rotate keys by generating a new bundle.

        The old bundle remains valid until explicitly revoked.

        Args:
            tenant_id: Tenant identifier
            old_bundle_id: Previous bundle ID

        Returns:
            New HEKeyBundle
        """
        old_bundle = self._bundles.get(old_bundle_id)
        params = old_bundle.params if old_bundle else None

        new_bundle = self.generate_key_bundle(tenant_id, params)

        logger.info(
            f"Rotated keys for tenant {tenant_id}: "
            f"{old_bundle_id} → {new_bundle.bundle_id}"
        )

        return new_bundle

    def revoke_bundle(self, bundle_id: str) -> bool:
        """
        Revoke a key bundle, preventing further use.

        Args:
            bundle_id: Bundle ID to revoke

        Returns:
            True if revoked, False if not found
        """
        if bundle_id not in self._bundles:
            return False

        # Remove from caches
        del self._bundles[bundle_id]
        self._contexts.pop(bundle_id, None)

        # Remove from storage
        if self._storage_path:
            self._delete_bundle(bundle_id)

        logger.info(f"Revoked HE key bundle {bundle_id}")
        return True

    def _wrap_key(self, key: bytes) -> bytes:
        """Wrap (encrypt) a key with the master key."""
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        aesgcm = AESGCM(self._master_key)
        ciphertext = aesgcm.encrypt(nonce, key, None)
        return nonce + ciphertext

    def _unwrap_key(self, wrapped: bytes) -> bytes:
        """Unwrap (decrypt) a key with the master key."""
        nonce = wrapped[: self.NONCE_SIZE]
        ciphertext = wrapped[self.NONCE_SIZE:]
        aesgcm = AESGCM(self._master_key)
        return aesgcm.decrypt(nonce, ciphertext, None)

    def _save_bundle(self, tenant_id: str, bundle: HEKeyBundle) -> None:
        """Save bundle to storage."""
        if not self._storage_path:
            return

        base = self._storage_path / tenant_id / "n2he"
        (base / "bundles").mkdir(parents=True, exist_ok=True)
        (base / "secret_keys").mkdir(exist_ok=True)
        (base / "public_keys").mkdir(exist_ok=True)
        (base / "eval_keys").mkdir(exist_ok=True)

        # Save bundle metadata
        bundle_data = {
            "bundle_id": bundle.bundle_id,
            "params": bundle.params.to_dict(),
            "secret_key_id": bundle.secret_key.key_id,
            "public_key_id": bundle.public_key.key_id,
            "eval_key_id": bundle.evaluation_key.key_id,
            "created_at": bundle.created_at.isoformat(),
        }
        (base / "bundles" / f"{bundle.bundle_id}.json").write_text(
            json.dumps(bundle_data, indent=2)
        )

        # Save keys
        (base / "secret_keys" / f"{bundle.secret_key.key_id}.enc").write_bytes(
            bundle.secret_key.key_bytes
        )
        (base / "public_keys" / f"{bundle.public_key.key_id}.pub").write_bytes(
            bundle.public_key.key_bytes
        )
        (base / "eval_keys" / f"{bundle.evaluation_key.key_id}.ek").write_bytes(
            bundle.evaluation_key.key_bytes
        )

        # Set permissions on secret key
        secret_key_path = base / "secret_keys" / f"{bundle.secret_key.key_id}.enc"
        os.chmod(secret_key_path, 0o600)

    def _delete_bundle(self, bundle_id: str) -> None:
        """Delete bundle from storage."""
        if not self._storage_path:
            return

        # Find and delete bundle files
        for tenant_dir in self._storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            bundle_file = tenant_dir / "n2he" / "bundles" / f"{bundle_id}.json"
            if bundle_file.exists():
                # Load to get key IDs
                bundle_data = json.loads(bundle_file.read_text())

                # Delete all related files
                bundle_file.unlink()

                sk_file = (
                    tenant_dir
                    / "n2he"
                    / "secret_keys"
                    / f"{bundle_data['secret_key_id']}.enc"
                )
                if sk_file.exists():
                    sk_file.unlink()

                pk_file = (
                    tenant_dir
                    / "n2he"
                    / "public_keys"
                    / f"{bundle_data['public_key_id']}.pub"
                )
                if pk_file.exists():
                    pk_file.unlink()

                ek_file = (
                    tenant_dir
                    / "n2he"
                    / "eval_keys"
                    / f"{bundle_data['eval_key_id']}.ek"
                )
                if ek_file.exists():
                    ek_file.unlink()

                break

    def _load_keys(self) -> None:
        """Load existing keys from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return

        for tenant_dir in self._storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue

            bundles_dir = tenant_dir / "n2he" / "bundles"
            if not bundles_dir.exists():
                continue

            for bundle_file in bundles_dir.glob("*.json"):
                try:
                    bundle_data = json.loads(bundle_file.read_text())
                    bundle_id = bundle_data["bundle_id"]

                    # Load key bytes
                    sk_bytes = (
                        tenant_dir
                        / "n2he"
                        / "secret_keys"
                        / f"{bundle_data['secret_key_id']}.enc"
                    ).read_bytes()

                    pk_bytes = (
                        tenant_dir
                        / "n2he"
                        / "public_keys"
                        / f"{bundle_data['public_key_id']}.pub"
                    ).read_bytes()

                    ek_bytes = (
                        tenant_dir
                        / "n2he"
                        / "eval_keys"
                        / f"{bundle_data['eval_key_id']}.ek"
                    ).read_bytes()

                    params = HESchemeParams.from_dict(bundle_data["params"])
                    params_hash = params.get_hash()
                    created_at = datetime.fromisoformat(bundle_data["created_at"])

                    bundle = HEKeyBundle(
                        bundle_id=bundle_id,
                        params=params,
                        secret_key=SecretKey(
                            key_id=bundle_data["secret_key_id"],
                            key_bytes=sk_bytes,
                            params_hash=params_hash,
                            wrapped=True,
                            created_at=created_at,
                        ),
                        public_key=PublicKey(
                            key_id=bundle_data["public_key_id"],
                            key_bytes=pk_bytes,
                            params_hash=params_hash,
                            created_at=created_at,
                        ),
                        evaluation_key=EvaluationKey(
                            key_id=bundle_data["eval_key_id"],
                            key_bytes=ek_bytes,
                            params_hash=params_hash,
                            created_at=created_at,
                        ),
                        created_at=created_at,
                    )

                    self._bundles[bundle_id] = bundle
                    logger.debug(f"Loaded HE key bundle {bundle_id}")

                except Exception as e:
                    logger.warning(f"Failed to load bundle {bundle_file}: {e}")


def create_key_manager(
    storage_path: Optional[str] = None,
    master_key: Optional[bytes] = None,
) -> HEKeyManager:
    """
    Factory function to create an HE key manager.

    Args:
        storage_path: Path for key storage (default: /tmp/tensafe_n2he_keys)
        master_key: Master key for wrapping (auto-generated if not provided)

    Returns:
        Configured HEKeyManager
    """
    if storage_path is None:
        storage_path = "/tmp/tensafe_n2he_keys"

    return HEKeyManager(
        master_key=master_key,
        storage_path=storage_path,
    )
