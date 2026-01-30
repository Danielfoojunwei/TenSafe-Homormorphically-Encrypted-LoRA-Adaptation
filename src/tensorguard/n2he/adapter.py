"""
Encrypted LoRA Adapter Runtime.

This module implements the "HE Adapter Runtime" that computes LoRA delta
contributions under homomorphic encryption using N2HE primitives.

Architecture:
    - Base model inference runs in plaintext (client or TEE)
    - TenSafe introduces the HE Adapter Runtime that computes only the
      LoRA delta path under HE
    - Existing TenSafe strengths (audit logs, packaging, PQ signatures)
      become the supply-chain trust layer for shipping LoRA adapters + proofs

What N2HE contributes:
    - HE kernels for the adapter's linear algebra path
    - "Compute LoRA delta without decrypting activations"

Flow:
    1. Client encrypts hidden state activations with public key
    2. Server receives encrypted activations
    3. HE Adapter Runtime computes: delta = scaling * enc(x) @ A^T @ B^T
    4. Server returns encrypted delta
    5. Client decrypts delta with secret key
    6. Client adds delta to base model output

This provides privacy for:
    - User activations (encrypted throughout)
    - User-specific adapter contributions
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .core import (
    Ciphertext,
    HESchemeParams,
    N2HEContext,
)
from .keys import HEKeyBundle, HEKeyManager

logger = logging.getLogger(__name__)


class AdapterMode(Enum):
    """Adapter operation mode."""

    ENCRYPTED = "encrypted"  # Full HE computation
    PLAINTEXT = "plaintext"  # No encryption (baseline)
    HYBRID = "hybrid"  # Encrypt only sensitive layers


@dataclass
class AdapterEncryptionConfig:
    """Configuration for encrypted LoRA adapter computation."""

    mode: AdapterMode = AdapterMode.ENCRYPTED

    # LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0  # No dropout in encrypted mode
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # HE parameters
    he_params: Optional[HESchemeParams] = None
    key_bundle_id: Optional[str] = None

    # Performance tuning
    batch_size: int = 1  # Process one sample at a time for HE
    max_seq_len: int = 512  # Maximum sequence length

    # Noise budget management
    noise_budget_threshold: float = 10.0  # Re-encrypt if budget drops below

    def get_scaling(self) -> float:
        """Get LoRA scaling factor."""
        return self.alpha / self.rank

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mode": self.mode.value,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "he_params": self.he_params.to_dict() if self.he_params else None,
            "key_bundle_id": self.key_bundle_id,
            "batch_size": self.batch_size,
            "max_seq_len": self.max_seq_len,
            "noise_budget_threshold": self.noise_budget_threshold,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterEncryptionConfig":
        """Deserialize from dictionary."""
        he_params = None
        if data.get("he_params"):
            he_params = HESchemeParams.from_dict(data["he_params"])

        return cls(
            mode=AdapterMode(data.get("mode", "encrypted")),
            rank=data.get("rank", 16),
            alpha=data.get("alpha", 32.0),
            dropout=data.get("dropout", 0.0),
            target_modules=data.get(
                "target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
            ),
            he_params=he_params,
            key_bundle_id=data.get("key_bundle_id"),
            batch_size=data.get("batch_size", 1),
            max_seq_len=data.get("max_seq_len", 512),
            noise_budget_threshold=data.get("noise_budget_threshold", 10.0),
        )


@dataclass
class EncryptedLoRAAdapter:
    """
    Represents a LoRA adapter with encrypted-capable weights.

    The adapter weights (A, B matrices) are stored in plaintext since they
    are model parameters. The encryption protects the activations flowing
    through the adapter, not the adapter weights themselves.
    """

    adapter_id: str
    module_name: str  # e.g., "model.layers.0.self_attn.q_proj"

    # LoRA matrices (plaintext)
    lora_a: np.ndarray  # [rank, in_features]
    lora_b: np.ndarray  # [out_features, rank]

    # Configuration
    rank: int
    alpha: float
    scaling: float  # alpha / rank

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    content_hash: Optional[str] = None

    def __post_init__(self):
        if self.content_hash is None:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash for integrity."""
        data = (
            self.adapter_id.encode()
            + self.module_name.encode()
            + self.lora_a.tobytes()
            + self.lora_b.tobytes()
        )
        return f"sha256:{hashlib.sha256(data).hexdigest()}"

    @property
    def in_features(self) -> int:
        """Input dimension."""
        return self.lora_a.shape[1]

    @property
    def out_features(self) -> int:
        """Output dimension."""
        return self.lora_b.shape[0]

    def forward_plaintext(self, x: np.ndarray) -> np.ndarray:
        """
        Compute LoRA delta in plaintext (for testing/baseline).

        Args:
            x: Input activations [batch, seq, in_features]

        Returns:
            LoRA delta [batch, seq, out_features]
        """
        # x @ A^T → [batch, seq, rank]
        intermediate = np.matmul(x, self.lora_a.T)

        # intermediate @ B^T → [batch, seq, out_features]
        delta = np.matmul(intermediate, self.lora_b.T)

        # Apply scaling
        return self.scaling * delta


@dataclass
class EncryptedActivation:
    """
    Encrypted hidden state activation.

    Contains ciphertext for a batch of activations along with metadata
    for tracking and verification.
    """

    ciphertext: Ciphertext
    batch_size: int
    seq_len: int
    hidden_dim: int
    key_bundle_id: str
    encrypted_at: datetime = field(default_factory=datetime.utcnow)

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for audit logging."""
        return {
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "key_bundle_id": self.key_bundle_id,
            "encrypted_at": self.encrypted_at.isoformat(),
            "noise_budget": getattr(self.ciphertext, "noise_budget", None),
        }


@dataclass
class EncryptedDelta:
    """
    Encrypted LoRA delta output.

    Contains the encrypted result of adapter computation.
    """

    ciphertext: Ciphertext
    adapter_id: str
    module_name: str
    key_bundle_id: str
    computed_at: datetime = field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for audit logging."""
        return {
            "adapter_id": self.adapter_id,
            "module_name": self.module_name,
            "key_bundle_id": self.key_bundle_id,
            "computed_at": self.computed_at.isoformat(),
            "computation_time_ms": self.computation_time_ms,
            "noise_budget": getattr(self.ciphertext, "noise_budget", None),
        }


class EncryptedLoRARuntime:
    """
    HE Adapter Runtime for computing LoRA deltas on encrypted activations.

    This is the core component that "uses N2HE" to provide privacy-preserving
    LoRA computation. It:

    1. Receives encrypted activations from clients
    2. Computes LoRA delta under HE: delta = scaling * enc(x) @ A^T @ B^T
    3. Returns encrypted delta to client
    4. Logs all operations for audit

    The client can then decrypt the delta with their secret key and add it
    to their base model output.
    """

    def __init__(
        self,
        config: AdapterEncryptionConfig,
        key_manager: Optional[HEKeyManager] = None,
    ):
        """
        Initialize encrypted LoRA runtime.

        Args:
            config: Adapter encryption configuration
            key_manager: HE key manager (created if not provided)
        """
        self.config = config
        self.key_manager = key_manager or HEKeyManager()

        # N2HE context (loaded when key bundle is set)
        self._context: Optional[N2HEContext] = None

        # Loaded adapters
        self._adapters: Dict[str, EncryptedLoRAAdapter] = {}

        # Metrics
        self._operations_count = 0
        self._total_compute_time_ms = 0.0
        self._errors_count = 0

        # Initialize context if key bundle specified
        if config.key_bundle_id:
            self._load_context(config.key_bundle_id)

    def _load_context(self, bundle_id: str) -> None:
        """Load N2HE context for a key bundle."""
        self._context = self.key_manager.get_context(
            bundle_id, include_secret_key=False
        )
        if self._context is None:
            raise ValueError(f"Key bundle not found: {bundle_id}")
        logger.info(f"Loaded N2HE context for bundle {bundle_id}")

    def register_adapter(
        self,
        adapter_id: str,
        module_name: str,
        lora_a: np.ndarray,
        lora_b: np.ndarray,
        rank: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> EncryptedLoRAAdapter:
        """
        Register a LoRA adapter for encrypted computation.

        Args:
            adapter_id: Unique adapter identifier
            module_name: Target module name (e.g., "model.layers.0.self_attn.q_proj")
            lora_a: LoRA A matrix [rank, in_features]
            lora_b: LoRA B matrix [out_features, rank]
            rank: LoRA rank (inferred from lora_a if not provided)
            alpha: LoRA alpha (default: from config)

        Returns:
            Registered EncryptedLoRAAdapter
        """
        rank = rank or lora_a.shape[0]
        alpha = alpha or self.config.alpha
        scaling = alpha / rank

        adapter = EncryptedLoRAAdapter(
            adapter_id=adapter_id,
            module_name=module_name,
            lora_a=lora_a.astype(np.float32),
            lora_b=lora_b.astype(np.float32),
            rank=rank,
            alpha=alpha,
            scaling=scaling,
        )

        self._adapters[adapter_id] = adapter
        logger.info(
            f"Registered adapter {adapter_id} for {module_name}: "
            f"rank={rank}, alpha={alpha}, shape=({lora_a.shape}, {lora_b.shape})"
        )

        return adapter

    def encrypt_activation(
        self,
        activation: np.ndarray,
        key_bundle_id: Optional[str] = None,
    ) -> EncryptedActivation:
        """
        Encrypt hidden state activations.

        This is typically done by the client before sending to the server.

        Args:
            activation: Hidden state [batch, seq, hidden_dim]
            key_bundle_id: Key bundle to use (default: config bundle)

        Returns:
            EncryptedActivation
        """
        bundle_id = key_bundle_id or self.config.key_bundle_id
        if bundle_id is None:
            raise ValueError("No key bundle specified")

        # Get context with secret key for encryption
        ctx = self.key_manager.get_context(bundle_id, include_secret_key=True)
        if ctx is None:
            raise ValueError(f"Key bundle not found: {bundle_id}")

        # Flatten activation for encryption
        # In real HE, we'd use SIMD packing; here we encrypt first element
        batch_size = activation.shape[0] if activation.ndim >= 1 else 1
        seq_len = activation.shape[1] if activation.ndim >= 2 else 1
        hidden_dim = activation.shape[-1]

        # Encrypt (simplified - real impl would pack more data)
        flat_activation = activation.flatten()[:hidden_dim]
        ciphertext = ctx.encrypt(flat_activation.astype(np.int64))

        return EncryptedActivation(
            ciphertext=ciphertext,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            key_bundle_id=bundle_id,
        )

    def compute_delta(
        self,
        encrypted_activation: EncryptedActivation,
        adapter_id: str,
    ) -> EncryptedDelta:
        """
        Compute encrypted LoRA delta.

        This is the core HE operation: delta = scaling * enc(x) @ A^T @ B^T

        Args:
            encrypted_activation: Encrypted hidden state
            adapter_id: Adapter to apply

        Returns:
            EncryptedDelta with encrypted result
        """
        start_time = time.time()

        adapter = self._adapters.get(adapter_id)
        if adapter is None:
            raise ValueError(f"Adapter not found: {adapter_id}")

        # Get context (should have eval key loaded)
        if self._context is None:
            bundle_id = encrypted_activation.key_bundle_id
            self._load_context(bundle_id)

        # Compute encrypted LoRA delta
        try:
            result_ct = self._context.encrypted_lora_delta(
                encrypted_activation=encrypted_activation.ciphertext,
                lora_a=adapter.lora_a,
                lora_b=adapter.lora_b,
                scaling=adapter.scaling,
            )

            computation_time_ms = (time.time() - start_time) * 1000
            self._operations_count += 1
            self._total_compute_time_ms += computation_time_ms

            return EncryptedDelta(
                ciphertext=result_ct,
                adapter_id=adapter_id,
                module_name=adapter.module_name,
                key_bundle_id=encrypted_activation.key_bundle_id,
                computation_time_ms=computation_time_ms,
            )

        except Exception as e:
            self._errors_count += 1
            logger.error(f"Error computing encrypted delta: {e}")
            raise

    def decrypt_delta(
        self,
        encrypted_delta: EncryptedDelta,
        key_bundle_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Decrypt LoRA delta.

        This is typically done by the client after receiving from server.

        Args:
            encrypted_delta: Encrypted delta from compute_delta
            key_bundle_id: Key bundle to use (must have secret key)

        Returns:
            Decrypted delta as numpy array
        """
        bundle_id = key_bundle_id or encrypted_delta.key_bundle_id

        # Get context with secret key
        ctx = self.key_manager.get_context(bundle_id, include_secret_key=True)
        if ctx is None:
            raise ValueError(f"Key bundle not found: {bundle_id}")

        if not ctx.has_secret_key:
            raise ValueError("Secret key not available for decryption")

        return ctx.decrypt(encrypted_delta.ciphertext)

    def forward(
        self,
        encrypted_activation: EncryptedActivation,
        adapter_ids: Optional[List[str]] = None,
    ) -> List[EncryptedDelta]:
        """
        Compute encrypted deltas for multiple adapters.

        Args:
            encrypted_activation: Encrypted hidden state
            adapter_ids: Adapters to apply (default: all registered)

        Returns:
            List of EncryptedDelta objects
        """
        adapter_ids = adapter_ids or list(self._adapters.keys())

        deltas = []
        for adapter_id in adapter_ids:
            delta = self.compute_delta(encrypted_activation, adapter_id)
            deltas.append(delta)

        return deltas

    def get_metrics(self) -> Dict[str, Any]:
        """Get runtime metrics for monitoring."""
        return {
            "operations_count": self._operations_count,
            "total_compute_time_ms": self._total_compute_time_ms,
            "avg_compute_time_ms": (
                self._total_compute_time_ms / self._operations_count
                if self._operations_count > 0
                else 0.0
            ),
            "errors_count": self._errors_count,
            "adapters_registered": len(self._adapters),
            "mode": self.config.mode.value,
            "context_metrics": self._context.get_metrics() if self._context else None,
        }

    def get_audit_record(
        self,
        operation: str,
        encrypted_activation: Optional[EncryptedActivation] = None,
        encrypted_delta: Optional[EncryptedDelta] = None,
    ) -> Dict[str, Any]:
        """
        Generate audit record for an operation.

        Args:
            operation: Operation type (encrypt, compute, decrypt)
            encrypted_activation: Input activation (if applicable)
            encrypted_delta: Output delta (if applicable)

        Returns:
            Audit record dictionary
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": operation,
            "mode": self.config.mode.value,
            "key_bundle_id": self.config.key_bundle_id,
        }

        if encrypted_activation:
            record["input_metadata"] = encrypted_activation.get_metadata()

        if encrypted_delta:
            record["output_metadata"] = encrypted_delta.get_metadata()

        return record


def create_encrypted_runtime(
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
    key_manager: Optional[HEKeyManager] = None,
    tenant_id: str = "default",
) -> Tuple[EncryptedLoRARuntime, HEKeyBundle]:
    """
    Factory function to create a fully configured encrypted LoRA runtime.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha scaling
        target_modules: Modules to apply LoRA to
        key_manager: HE key manager (created if not provided)
        tenant_id: Tenant ID for key generation

    Returns:
        Tuple of (runtime, key_bundle)
    """
    key_manager = key_manager or HEKeyManager()

    # Generate keys
    bundle = key_manager.generate_key_bundle(
        tenant_id=tenant_id,
        params=HESchemeParams.default_lora_params(),
    )

    # Create config
    config = AdapterEncryptionConfig(
        mode=AdapterMode.ENCRYPTED,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules
        or ["q_proj", "v_proj", "k_proj", "o_proj"],
        he_params=bundle.params,
        key_bundle_id=bundle.bundle_id,
    )

    # Create runtime
    runtime = EncryptedLoRARuntime(config=config, key_manager=key_manager)

    logger.info(
        f"Created encrypted LoRA runtime for tenant {tenant_id}: "
        f"bundle={bundle.bundle_id}, rank={rank}, alpha={alpha}"
    )

    return runtime, bundle
