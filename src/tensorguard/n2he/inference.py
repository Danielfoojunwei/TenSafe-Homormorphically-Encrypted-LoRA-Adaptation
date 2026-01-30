"""
Private Inference Mode for TenSafe.

Implements privacy-preserving sample/evaluation using homomorphic encryption.
This is architectural option 2: "Private inference mode for sensitive evaluation/telemetry"

TenSafe already has sample functionality for generation/evaluation in the training
lifecycle. This module adds a "private sample" mode that:
    - Encrypts prompts/activations
    - Runs a constrained HE-friendly model path (or partial path)
    - Returns encrypted outputs

Use Cases:
    1. Private evaluation during training - evaluate on sensitive test data
       without exposing prompts or outputs to the server
    2. Privacy-preserving telemetry - collect model behavior metrics on
       encrypted data
    3. Confidential inference - generate completions on encrypted prompts

Architecture:
    Client                          Server (TenSafe)
    ------                          ----------------
    1. Generate/load HE keys
    2. Encrypt prompt tokens
       with public key
    3. Send encrypted batch    -->  4. Receive encrypted batch
                                    5. Run HE-friendly forward pass
                                       (attention, FFN under HE)
                                    6. Return encrypted logits/output
    7. Decrypt output with     <--
       secret key
    8. Sample tokens locally
       (argmax, top-k, etc.)
"""

import hashlib
import json
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


class PrivateInferenceProfile(Enum):
    """Inference privacy profiles."""

    # Encrypt only the prompt embeddings, process under HE
    ENCRYPTED_INPUT = "encrypted_input"

    # Encrypt activations at each layer, process under HE
    FULL_ENCRYPTED = "full_encrypted"

    # Only encrypt attention queries/keys (for privacy of what model attends to)
    ENCRYPTED_ATTENTION = "encrypted_attention"

    # Hybrid: encrypt sensitive layers, plaintext for compute-heavy layers
    HYBRID = "hybrid"


@dataclass
class EncryptedBatch:
    """
    Encrypted input batch for private inference.

    Contains encrypted token embeddings and associated metadata.
    """

    batch_id: str

    # Encrypted data
    encrypted_embeddings: List[Ciphertext]  # One per sequence position
    attention_mask: Optional[np.ndarray] = None  # Plaintext mask (positions)

    # Metadata
    batch_size: int = 1
    seq_len: int = 0
    hidden_dim: int = 0
    key_bundle_id: str = ""

    # Encryption metadata
    encrypted_at: datetime = field(default_factory=datetime.utcnow)
    params_hash: str = ""

    def __post_init__(self):
        self.seq_len = len(self.encrypted_embeddings)

    def get_metadata(self) -> Dict[str, Any]:
        """Get batch metadata for audit logging."""
        return {
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "hidden_dim": self.hidden_dim,
            "key_bundle_id": self.key_bundle_id,
            "encrypted_at": self.encrypted_at.isoformat(),
            "params_hash": self.params_hash,
        }


@dataclass
class EncryptedOutput:
    """
    Encrypted output from private inference.

    Contains encrypted logits/hidden states and associated metadata.
    """

    output_id: str

    # Encrypted data
    encrypted_logits: Optional[Ciphertext] = None  # For next-token prediction
    encrypted_hidden_states: Optional[List[Ciphertext]] = None  # Per-position states

    # Computation metadata
    key_bundle_id: str = ""
    computed_at: datetime = field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0
    layers_processed: int = 0

    # Privacy metrics
    noise_budget_remaining: Optional[float] = None
    operations_performed: int = 0

    def get_metadata(self) -> Dict[str, Any]:
        """Get output metadata for audit logging."""
        return {
            "output_id": self.output_id,
            "key_bundle_id": self.key_bundle_id,
            "computed_at": self.computed_at.isoformat(),
            "computation_time_ms": self.computation_time_ms,
            "layers_processed": self.layers_processed,
            "noise_budget_remaining": self.noise_budget_remaining,
            "operations_performed": self.operations_performed,
        }


@dataclass
class PrivateInferenceConfig:
    """Configuration for private inference mode."""

    profile: PrivateInferenceProfile = PrivateInferenceProfile.ENCRYPTED_INPUT

    # Model parameters
    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32

    # HE parameters
    he_params: Optional[HESchemeParams] = None
    key_bundle_id: Optional[str] = None

    # Processing limits (for HE budget management)
    max_seq_len: int = 128  # Shorter for HE efficiency
    max_layers_encrypted: int = 4  # Only first N layers under HE

    # Sampling parameters (applied after decryption)
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "profile": self.profile.value,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "he_params": self.he_params.to_dict() if self.he_params else None,
            "key_bundle_id": self.key_bundle_id,
            "max_seq_len": self.max_seq_len,
            "max_layers_encrypted": self.max_layers_encrypted,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }


@dataclass
class PrivateSampleResult:
    """Result from private sampling."""

    output_id: str
    encrypted_output: EncryptedOutput

    # After decryption (by client)
    decrypted_logits: Optional[np.ndarray] = None
    sampled_tokens: Optional[List[int]] = None
    completion_text: Optional[str] = None

    # Metrics
    total_time_ms: float = 0.0
    server_time_ms: float = 0.0
    privacy_preserved: bool = True


class PrivateInferenceMode:
    """
    Private inference mode for TenSafe.

    Provides privacy-preserving sample/evaluation by running model
    computations on encrypted data.

    This integrates with the existing TenSafe training lifecycle:
        - TrainingClient.sample() → PrivateInferenceMode.private_sample()
        - Evaluation metrics computed on encrypted test data
        - Telemetry collected without exposing user data
    """

    def __init__(
        self,
        config: PrivateInferenceConfig,
        key_manager: Optional[HEKeyManager] = None,
    ):
        """
        Initialize private inference mode.

        Args:
            config: Inference configuration
            key_manager: HE key manager
        """
        self.config = config
        self.key_manager = key_manager or HEKeyManager()

        # N2HE context
        self._context: Optional[N2HEContext] = None

        # Mock model weights (in production, loaded from checkpoint)
        self._embedding_weights: Optional[np.ndarray] = None
        self._layer_weights: Dict[int, Dict[str, np.ndarray]] = {}
        self._lm_head_weights: Optional[np.ndarray] = None

        # Metrics
        self._samples_processed = 0
        self._total_encrypted_ops = 0

        # Initialize
        if config.key_bundle_id:
            self._load_context(config.key_bundle_id)
        self._initialize_mock_weights()

    def _load_context(self, bundle_id: str) -> None:
        """Load N2HE context."""
        self._context = self.key_manager.get_context(bundle_id)
        if self._context is None:
            # Generate new bundle if needed
            self.key_manager.generate_key_bundle(
                tenant_id="default",
                params=self.config.he_params,
                bundle_id=bundle_id,
            )
            self._context = self.key_manager.get_context(bundle_id)

    def _initialize_mock_weights(self) -> None:
        """Initialize mock model weights for testing."""
        hidden = self.config.hidden_dim
        vocab = self.config.vocab_size

        # Embedding layer
        self._embedding_weights = np.random.randn(vocab, hidden).astype(np.float32) * 0.02

        # Transformer layers (simplified)
        for layer_idx in range(self.config.max_layers_encrypted):
            self._layer_weights[layer_idx] = {
                "q_proj": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "k_proj": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "v_proj": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "o_proj": np.random.randn(hidden, hidden).astype(np.float32) * 0.02,
                "mlp_up": np.random.randn(hidden * 4, hidden).astype(np.float32) * 0.02,
                "mlp_down": np.random.randn(hidden, hidden * 4).astype(np.float32) * 0.02,
            }

        # LM head
        self._lm_head_weights = np.random.randn(vocab, hidden).astype(np.float32) * 0.02

    def encrypt_batch(
        self,
        token_ids: List[List[int]],
        key_bundle_id: Optional[str] = None,
    ) -> EncryptedBatch:
        """
        Encrypt input tokens for private inference.

        Args:
            token_ids: Batch of token ID sequences [[token1, token2, ...], ...]
            key_bundle_id: Key bundle to use

        Returns:
            EncryptedBatch ready for processing
        """
        bundle_id = key_bundle_id or self.config.key_bundle_id
        if bundle_id is None:
            raise ValueError("No key bundle specified")

        # Load context with secret key for encryption
        ctx = self.key_manager.get_context(bundle_id, include_secret_key=True)
        if ctx is None:
            raise ValueError(f"Key bundle not found: {bundle_id}")

        batch_size = len(token_ids)
        max_len = min(max(len(seq) for seq in token_ids), self.config.max_seq_len)

        # Create attention mask
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int32)
        for i, seq in enumerate(token_ids):
            attention_mask[i, : len(seq)] = 1

        # Embed and encrypt each position
        encrypted_embeddings = []
        for pos in range(max_len):
            # Gather tokens at this position
            pos_tokens = [
                seq[pos] if pos < len(seq) else 0 for seq in token_ids
            ]

            # Look up embeddings (in real impl, this would be on client)
            embeddings = self._embedding_weights[pos_tokens]  # [batch, hidden]

            # Encrypt (simplified - encrypt first element)
            flat_emb = embeddings.flatten()[: self.config.hidden_dim]
            ct = ctx.encrypt(flat_emb.astype(np.int64))
            encrypted_embeddings.append(ct)

        batch_id = f"batch-{hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:8]}"

        return EncryptedBatch(
            batch_id=batch_id,
            encrypted_embeddings=encrypted_embeddings,
            attention_mask=attention_mask,
            batch_size=batch_size,
            hidden_dim=self.config.hidden_dim,
            key_bundle_id=bundle_id,
            params_hash=ctx.params.get_hash(),
        )

    def private_forward(
        self,
        encrypted_batch: EncryptedBatch,
    ) -> EncryptedOutput:
        """
        Run private forward pass on encrypted batch.

        This processes the encrypted embeddings through a constrained
        HE-friendly model path.

        Args:
            encrypted_batch: Encrypted input batch

        Returns:
            EncryptedOutput with encrypted logits/hidden states
        """
        start_time = time.time()

        if self._context is None:
            self._load_context(encrypted_batch.key_bundle_id)

        # Process through encrypted layers
        hidden_states = list(encrypted_batch.encrypted_embeddings)
        layers_processed = 0
        total_ops = 0

        for layer_idx in range(self.config.max_layers_encrypted):
            if layer_idx >= len(self._layer_weights):
                break

            weights = self._layer_weights[layer_idx]

            # Process each position
            new_hidden = []
            for pos, h in enumerate(hidden_states):
                # Simplified: just compute Q projection under HE
                # Real impl would do full attention under HE
                q_out = self._context.encrypted_linear(
                    h,
                    weights["q_proj"][:16, :16],  # Reduced for simulation
                )
                new_hidden.append(q_out)
                total_ops += 1

            hidden_states = new_hidden
            layers_processed += 1

        # Get last hidden state for LM head
        if hidden_states:
            # Apply LM head (simplified)
            final_hidden = hidden_states[-1]  # Last position
            encrypted_logits = self._context.encrypted_linear(
                final_hidden,
                self._lm_head_weights[:16, :16],  # Reduced
            )
            total_ops += 1
        else:
            encrypted_logits = None

        computation_time_ms = (time.time() - start_time) * 1000
        self._samples_processed += 1
        self._total_encrypted_ops += total_ops

        output_id = f"out-{encrypted_batch.batch_id}"

        return EncryptedOutput(
            output_id=output_id,
            encrypted_logits=encrypted_logits,
            encrypted_hidden_states=hidden_states,
            key_bundle_id=encrypted_batch.key_bundle_id,
            computation_time_ms=computation_time_ms,
            layers_processed=layers_processed,
            noise_budget_remaining=getattr(
                encrypted_logits, "noise_budget", None
            ) if encrypted_logits else None,
            operations_performed=total_ops,
        )

    def decrypt_output(
        self,
        encrypted_output: EncryptedOutput,
        key_bundle_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Decrypt inference output.

        This is done by the client with their secret key.

        Args:
            encrypted_output: Encrypted output from private_forward
            key_bundle_id: Key bundle (must have secret key)

        Returns:
            Decrypted logits as numpy array
        """
        bundle_id = key_bundle_id or encrypted_output.key_bundle_id

        ctx = self.key_manager.get_context(bundle_id, include_secret_key=True)
        if ctx is None:
            raise ValueError(f"Key bundle not found: {bundle_id}")

        if not ctx.has_secret_key:
            raise ValueError("Secret key required for decryption")

        if encrypted_output.encrypted_logits is None:
            raise ValueError("No encrypted logits in output")

        return ctx.decrypt(encrypted_output.encrypted_logits)

    def private_sample(
        self,
        prompts: List[str],
        tokenizer: Optional[Any] = None,
        max_new_tokens: int = 32,
    ) -> List[PrivateSampleResult]:
        """
        Generate private samples from prompts.

        This is the high-level API for privacy-preserving generation:
        1. Tokenize prompts
        2. Encrypt tokens
        3. Run private forward pass
        4. Decrypt and sample

        Args:
            prompts: List of prompt strings
            tokenizer: Tokenizer (uses mock if not provided)
            max_new_tokens: Maximum new tokens to generate

        Returns:
            List of PrivateSampleResult
        """
        start_time = time.time()

        # Mock tokenization if no tokenizer
        if tokenizer is None:
            token_ids = [
                [hash(c) % 32000 for c in prompt[:self.config.max_seq_len]]
                for prompt in prompts
            ]
        else:
            encoded = tokenizer(
                prompts,
                max_length=self.config.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
            token_ids = encoded["input_ids"].tolist()

        results = []

        for i, prompt in enumerate(prompts):
            # Encrypt this prompt
            encrypted_batch = self.encrypt_batch(
                [token_ids[i]],
                key_bundle_id=self.config.key_bundle_id,
            )

            # Run private forward
            encrypted_output = self.private_forward(encrypted_batch)

            # Decrypt (client-side)
            try:
                decrypted_logits = self.decrypt_output(encrypted_output)
            except Exception as e:
                logger.warning(f"Decryption failed: {e}")
                decrypted_logits = None

            # Sample token (simplified - just take argmax)
            sampled_tokens = None
            if decrypted_logits is not None:
                sampled_tokens = [int(np.argmax(decrypted_logits))]

            total_time_ms = (time.time() - start_time) * 1000

            results.append(
                PrivateSampleResult(
                    output_id=encrypted_output.output_id,
                    encrypted_output=encrypted_output,
                    decrypted_logits=decrypted_logits,
                    sampled_tokens=sampled_tokens,
                    completion_text=f"[Private completion {i}]",  # Mock
                    total_time_ms=total_time_ms,
                    server_time_ms=encrypted_output.computation_time_ms,
                    privacy_preserved=True,
                )
            )

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """Get inference mode metrics."""
        return {
            "samples_processed": self._samples_processed,
            "total_encrypted_ops": self._total_encrypted_ops,
            "profile": self.config.profile.value,
            "max_layers_encrypted": self.config.max_layers_encrypted,
            "context_metrics": self._context.get_metrics() if self._context else None,
        }

    def get_audit_record(
        self,
        encrypted_batch: Optional[EncryptedBatch] = None,
        encrypted_output: Optional[EncryptedOutput] = None,
    ) -> Dict[str, Any]:
        """Generate audit record for private inference."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "operation": "private_inference",
            "profile": self.config.profile.value,
        }

        if encrypted_batch:
            record["input_metadata"] = encrypted_batch.get_metadata()

        if encrypted_output:
            record["output_metadata"] = encrypted_output.get_metadata()

        return record


def create_private_inference_mode(
    profile: str = "encrypted_input",
    key_manager: Optional[HEKeyManager] = None,
    tenant_id: str = "default",
    hidden_dim: int = 4096,
    max_layers_encrypted: int = 4,
) -> Tuple[PrivateInferenceMode, HEKeyBundle]:
    """
    Factory function to create private inference mode.

    Args:
        profile: Privacy profile name
        key_manager: HE key manager
        tenant_id: Tenant ID for keys
        hidden_dim: Model hidden dimension
        max_layers_encrypted: Number of layers to process under HE

    Returns:
        Tuple of (PrivateInferenceMode, HEKeyBundle)
    """
    key_manager = key_manager or HEKeyManager()

    # Generate keys
    bundle = key_manager.generate_key_bundle(
        tenant_id=tenant_id,
        params=HESchemeParams.default_lora_params(),
    )

    # Create config
    config = PrivateInferenceConfig(
        profile=PrivateInferenceProfile(profile),
        hidden_dim=hidden_dim,
        max_layers_encrypted=max_layers_encrypted,
        he_params=bundle.params,
        key_bundle_id=bundle.bundle_id,
    )

    # Create mode
    mode = PrivateInferenceMode(config=config, key_manager=key_manager)

    logger.info(
        f"Created private inference mode for tenant {tenant_id}: "
        f"profile={profile}, bundle={bundle.bundle_id}"
    )

    return mode, bundle
