"""
HE Key Manager

Manages HE keys for the HAS service:
- Secret key (sk) - NEVER leaves HAS process
- Public key (pk) - Can be shared with MSS
- Evaluation keys (evk) - Galois keys for rotations, relinearization keys

Security properties:
- Keys generated once at startup
- Secret key held only in memory, never serialized
- Audit logging for all key operations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import logging
import time
import hashlib
import secrets

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types of HE keys."""
    SECRET = "secret"
    PUBLIC = "public"
    GALOIS = "galois"
    RELIN = "relin"


@dataclass
class KeyMetadata:
    """Metadata for a key."""
    key_type: KeyType
    created_at: float
    key_hash: str  # SHA256 of key bytes (for audit)
    params_hash: str  # Hash of CKKS parameters used


@dataclass
class KeySet:
    """Complete set of HE keys."""
    secret_key: Any = None
    public_key: Any = None
    galois_keys: Dict[int, Any] = field(default_factory=dict)
    relin_keys: Any = None
    metadata: Dict[str, KeyMetadata] = field(default_factory=dict)


class KeyManager:
    """
    Manages HE keys for the HAS service.

    Security model:
    - Secret key is generated at initialization and held only in memory
    - All key operations are logged for audit
    - Keys are tied to specific CKKS parameters
    """

    def __init__(
        self,
        ckks_params: Optional[Any] = None,
        enable_audit_log: bool = True,
    ):
        """
        Initialize key manager.

        Args:
            ckks_params: CKKS parameters for key generation
            enable_audit_log: Whether to log key operations
        """
        self._ckks_params = ckks_params
        self._enable_audit = enable_audit_log

        # Key storage
        self._keyset: Optional[KeySet] = None
        self._params_hash: Optional[str] = None

        # Backend reference
        self._backend: Optional[Any] = None

        # Audit log
        self._audit_log: List[Dict] = []

        # Required Galois key steps
        self._required_galois_steps: Set[int] = set()

    def initialize(
        self,
        backend: Any,
        galois_steps: Optional[List[int]] = None,
    ) -> bool:
        """
        Initialize keys with the HE backend.

        Args:
            backend: GPU CKKS backend instance
            galois_steps: Required rotation step sizes

        Returns:
            True if initialization successful
        """
        self._backend = backend
        self._params_hash = self._compute_params_hash()

        # Store required Galois steps
        if galois_steps:
            self._required_galois_steps = set(galois_steps)

        # Generate keys
        try:
            self._generate_keys()
            self._audit("KEY_INIT", "Keys generated successfully")
            return True
        except Exception as e:
            logger.error(f"Key generation failed: {e}")
            self._audit("KEY_INIT_FAILED", str(e))
            return False

    def _generate_keys(self) -> None:
        """Generate all HE keys."""
        if self._backend is None:
            raise RuntimeError("Backend not set")

        self._keyset = KeySet()
        created_at = time.time()

        # Generate secret key
        if hasattr(self._backend, 'generate_secret_key'):
            self._keyset.secret_key = self._backend.generate_secret_key()
        else:
            # Mock key for simulation
            self._keyset.secret_key = self._mock_key("secret")

        self._keyset.metadata['secret'] = KeyMetadata(
            key_type=KeyType.SECRET,
            created_at=created_at,
            key_hash=self._hash_key(self._keyset.secret_key),
            params_hash=self._params_hash or "",
        )

        # Generate public key
        if hasattr(self._backend, 'generate_public_key'):
            self._keyset.public_key = self._backend.generate_public_key(
                self._keyset.secret_key
            )
        else:
            self._keyset.public_key = self._mock_key("public")

        self._keyset.metadata['public'] = KeyMetadata(
            key_type=KeyType.PUBLIC,
            created_at=created_at,
            key_hash=self._hash_key(self._keyset.public_key),
            params_hash=self._params_hash or "",
        )

        # Generate relinearization keys
        if hasattr(self._backend, 'generate_relin_keys'):
            self._keyset.relin_keys = self._backend.generate_relin_keys(
                self._keyset.secret_key
            )
        else:
            self._keyset.relin_keys = self._mock_key("relin")

        self._keyset.metadata['relin'] = KeyMetadata(
            key_type=KeyType.RELIN,
            created_at=created_at,
            key_hash=self._hash_key(self._keyset.relin_keys),
            params_hash=self._params_hash or "",
        )

        # Generate Galois keys for required rotation steps
        self._generate_galois_keys(created_at)

        logger.info(f"Generated HE keys with {len(self._keyset.galois_keys)} Galois keys")

    def _generate_galois_keys(self, created_at: float) -> None:
        """Generate Galois keys for rotations."""
        for step in self._required_galois_steps:
            if hasattr(self._backend, 'generate_galois_key'):
                galois_key = self._backend.generate_galois_key(
                    self._keyset.secret_key, step
                )
            else:
                galois_key = self._mock_key(f"galois_{step}")

            self._keyset.galois_keys[step] = galois_key
            self._keyset.metadata[f'galois_{step}'] = KeyMetadata(
                key_type=KeyType.GALOIS,
                created_at=created_at,
                key_hash=self._hash_key(galois_key),
                params_hash=self._params_hash or "",
            )

    def add_galois_key(self, step: int) -> bool:
        """
        Add a Galois key for a new rotation step.

        Args:
            step: Rotation step size

        Returns:
            True if key was added (or already exists)
        """
        if self._keyset is None:
            return False

        if step in self._keyset.galois_keys:
            return True  # Already have this key

        self._required_galois_steps.add(step)

        try:
            created_at = time.time()
            if hasattr(self._backend, 'generate_galois_key'):
                galois_key = self._backend.generate_galois_key(
                    self._keyset.secret_key, step
                )
            else:
                galois_key = self._mock_key(f"galois_{step}")

            self._keyset.galois_keys[step] = galois_key
            self._keyset.metadata[f'galois_{step}'] = KeyMetadata(
                key_type=KeyType.GALOIS,
                created_at=created_at,
                key_hash=self._hash_key(galois_key),
                params_hash=self._params_hash or "",
            )

            self._audit("GALOIS_KEY_ADDED", f"step={step}")
            return True

        except Exception as e:
            logger.error(f"Failed to add Galois key for step {step}: {e}")
            return False

    def get_secret_key(self) -> Optional[Any]:
        """
        Get the secret key.

        WARNING: This should only be called within HAS process.
        """
        if self._keyset is None:
            return None
        self._audit("SECRET_KEY_ACCESS", "Secret key accessed")
        return self._keyset.secret_key

    def get_public_key(self) -> Optional[Any]:
        """Get the public key."""
        if self._keyset is None:
            return None
        return self._keyset.public_key

    def get_galois_key(self, step: int) -> Optional[Any]:
        """Get a Galois key for a specific rotation step."""
        if self._keyset is None:
            return None
        return self._keyset.galois_keys.get(step)

    def get_relin_keys(self) -> Optional[Any]:
        """Get relinearization keys."""
        if self._keyset is None:
            return None
        return self._keyset.relin_keys

    def get_available_galois_steps(self) -> List[int]:
        """Get list of available Galois key steps."""
        if self._keyset is None:
            return []
        return list(sorted(self._keyset.galois_keys.keys()))

    def clear_keys(self) -> None:
        """
        Securely clear all keys from memory.

        Called during shutdown.
        """
        if self._keyset is not None:
            # Attempt to overwrite key data
            if self._keyset.secret_key is not None:
                self._keyset.secret_key = None
            if self._keyset.public_key is not None:
                self._keyset.public_key = None
            for step in list(self._keyset.galois_keys.keys()):
                self._keyset.galois_keys[step] = None
            if self._keyset.relin_keys is not None:
                self._keyset.relin_keys = None

            self._keyset = None

        self._audit("KEYS_CLEARED", "All keys cleared from memory")
        logger.info("HE keys cleared from memory")

    # -------------------------------------------------------------------------
    # AUDIT AND UTILITIES
    # -------------------------------------------------------------------------

    def _audit(self, event: str, details: str) -> None:
        """Log an audit event."""
        if not self._enable_audit:
            return

        entry = {
            'timestamp': time.time(),
            'event': event,
            'details': details,
        }
        self._audit_log.append(entry)

        # Also log to standard logger at INFO level
        logger.info(f"AUDIT: {event} - {details}")

    def get_audit_log(self) -> List[Dict]:
        """Get the audit log."""
        return self._audit_log.copy()

    def _compute_params_hash(self) -> str:
        """Compute hash of CKKS parameters."""
        if self._ckks_params is None:
            return "no_params"

        params_str = str(self._ckks_params)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]

    def _hash_key(self, key: Any) -> str:
        """Compute hash of a key for audit purposes."""
        if key is None:
            return "none"

        # Get bytes representation
        if hasattr(key, 'tobytes'):
            key_bytes = key.tobytes()
        elif hasattr(key, 'serialize'):
            key_bytes = key.serialize()
        elif isinstance(key, bytes):
            key_bytes = key
        else:
            key_bytes = str(key).encode()

        return hashlib.sha256(key_bytes).hexdigest()[:16]

    def _mock_key(self, key_type: str) -> bytes:
        """Generate mock key for simulation mode."""
        return secrets.token_bytes(32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get key manager statistics."""
        return {
            'initialized': self._keyset is not None,
            'params_hash': self._params_hash,
            'galois_key_count': len(self._keyset.galois_keys) if self._keyset else 0,
            'galois_steps': self.get_available_galois_steps(),
            'audit_entries': len(self._audit_log),
        }
