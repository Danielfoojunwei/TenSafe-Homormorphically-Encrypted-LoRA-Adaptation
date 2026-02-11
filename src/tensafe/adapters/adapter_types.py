"""
Advanced Adapter Types for TenSafe Hot-Swap System.

This module implements production-grade support for multiple adapter types
beyond standard LoRA, drawing from industry best practices:

- DoRA: Weight-Decomposed Low-Rank Adaptation (magnitude + direction)
- AdaLoRA: Adaptive Budget Allocation via SVD parameterization
- VeRA: Vector-based Random Matrix Adaptation (extreme parameter efficiency)
- LoRA-FA: Frozen-A variant for memory efficiency
- rsLoRA: Rank-Stabilized LoRA for high-rank fine-tuning
- GatedLoRA: Gated LoRA with learnable gates

References:
    - DoRA: https://arxiv.org/abs/2402.09353
    - AdaLoRA: https://arxiv.org/abs/2303.10512
    - VeRA: https://arxiv.org/abs/2310.11454
    - LoRA-FA: https://arxiv.org/abs/2308.03303
    - rsLoRA: https://arxiv.org/abs/2312.03732

Author: TenSafe Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


class AdapterType(Enum):
    """Supported adapter types."""
    LORA = "lora"                  # Standard LoRA: y = y + s*(xA)B
    DORA = "dora"                  # Weight-Decomposed: y = m * (W + sAB) / ||W + sAB||
    ADALORA = "adalora"            # SVD-based adaptive rank: y = y + PΛQ
    VERA = "vera"                  # Vector-based: Λ_b ⊙ (B @ (Λ_d ⊙ A)) with frozen A,B
    LORA_FA = "lora_fa"            # Frozen-A: y = y + xA_frozen @ B
    RS_LORA = "rs_lora"            # Rank-stabilized: scaling = α/√r
    GATED_LORA = "gated_lora"      # Gated: y = y + gate(x) * (xA)B
    GLORA = "glora"                # Generalized: supports scaling, shifting, prompts


class ScalingStrategy(Enum):
    """Scaling strategies for LoRA variants."""
    STANDARD = "standard"          # α/r
    RANK_STABILIZED = "rs"         # α/√r (rsLoRA)
    ADAPTIVE = "adaptive"          # Learned per-layer scaling
    IMPORTANCE_WEIGHTED = "importance"  # Based on layer importance scores


@dataclass
class AdapterConfig:
    """
    Unified configuration for all adapter types.

    This configuration supports all adapter variants through a common interface,
    with type-specific parameters handled via the `extra_config` field.
    """
    # Core parameters
    adapter_type: AdapterType = AdapterType.LORA
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    # Target modules
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Scaling
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD

    # DoRA specific
    use_magnitude: bool = False  # Enable magnitude component for DoRA

    # AdaLoRA specific
    initial_rank: Optional[int] = None  # Starting rank (pruned to `rank`)
    importance_beta: float = 0.85  # EMA coefficient for importance scores

    # VeRA specific
    vera_d_initial: float = 0.1  # Initial value for λ_d
    vera_b_initial: float = 0.1  # Initial value for λ_b
    projection_seed: int = 42    # Seed for reproducible random matrices

    # LoRA-FA specific
    freeze_a: bool = False  # Freeze A matrix after initialization

    # Gated LoRA specific
    gate_type: str = "sigmoid"  # sigmoid, tanh, learned

    # GLoRA specific
    use_weight_scaling: bool = False  # A tensor for W * A
    use_weight_shift: bool = False    # B tensor for W + B
    use_bias_scaling: bool = False    # D tensor for bias * D
    use_bias_shift: bool = False      # E tensor for bias + E
    use_prompt: bool = False          # C tensor for prompts

    # Layer-wise configuration
    layer_config: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Extra type-specific config
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        """Compute scaling factor based on strategy."""
        if self.scaling_strategy == ScalingStrategy.RANK_STABILIZED:
            return self.alpha / math.sqrt(self.rank)
        elif self.scaling_strategy == ScalingStrategy.STANDARD:
            return self.alpha / self.rank
        else:
            # For adaptive/importance, return base scaling (modified at runtime)
            return self.alpha / self.rank

    def get_layer_config(self, layer_idx: int) -> 'AdapterConfig':
        """Get configuration for a specific layer with overrides applied."""
        if layer_idx not in self.layer_config:
            return self

        # Create a copy with layer-specific overrides
        overrides = self.layer_config[layer_idx]
        config_dict = {
            'adapter_type': self.adapter_type,
            'rank': overrides.get('rank', self.rank),
            'alpha': overrides.get('alpha', self.alpha),
            'dropout': overrides.get('dropout', self.dropout),
            'target_modules': overrides.get('target_modules', self.target_modules),
            'scaling_strategy': overrides.get('scaling_strategy', self.scaling_strategy),
            'use_magnitude': overrides.get('use_magnitude', self.use_magnitude),
            'freeze_a': overrides.get('freeze_a', self.freeze_a),
            'gate_type': overrides.get('gate_type', self.gate_type),
            'extra_config': {**self.extra_config, **overrides.get('extra_config', {})},
        }
        return AdapterConfig(**config_dict)

    def validate(self) -> List[str]:
        """Validate configuration."""
        errors = []

        if self.rank <= 0:
            errors.append(f"rank must be positive, got {self.rank}")

        if self.alpha <= 0:
            errors.append(f"alpha must be positive, got {self.alpha}")

        if self.dropout < 0 or self.dropout >= 1:
            errors.append(f"dropout must be in [0, 1), got {self.dropout}")

        if self.adapter_type == AdapterType.ADALORA:
            if self.initial_rank is not None and self.initial_rank < self.rank:
                errors.append(f"initial_rank must be >= rank for AdaLoRA")

        if self.adapter_type == AdapterType.VERA:
            if self.vera_d_initial <= 0 or self.vera_b_initial <= 0:
                errors.append("VeRA initial values must be positive")

        return errors


class BaseAdapter(ABC):
    """
    Abstract base class for all adapter types.

    Defines the common interface that all adapters must implement for
    seamless hot-swapping and integration with the TenSafe pipeline.
    """

    def __init__(self, config: AdapterConfig, in_features: int, out_features: int):
        """
        Initialize adapter.

        Args:
            config: Adapter configuration
            in_features: Input dimension
            out_features: Output dimension
        """
        self.config = config
        self.in_features = in_features
        self.out_features = out_features
        self._is_merged = False
        self._trainable_params: Dict[str, np.ndarray] = {}
        self._frozen_params: Dict[str, np.ndarray] = {}

    @abstractmethod
    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize adapter weights."""
        pass

    @abstractmethod
    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute adapter output (delta to add to base model output).

        Args:
            x: Input activations [batch, seq, hidden] or [batch, hidden]
            original_weight: Original weight matrix (needed for DoRA)
            original_output: Original layer output (for some adapter types)

        Returns:
            Delta to add to base model output
        """
        pass

    @abstractmethod
    def get_delta_weight(self) -> np.ndarray:
        """
        Get the effective weight delta for merging.

        Returns:
            Weight delta matrix [out_features, in_features]
        """
        pass

    def can_merge(self) -> bool:
        """Check if this adapter can be merged into base weights."""
        # Most linear adapters can merge; non-linear ones cannot
        return self.config.adapter_type in {
            AdapterType.LORA,
            AdapterType.RS_LORA,
            AdapterType.LORA_FA,
            AdapterType.DORA,
            AdapterType.VERA,
        }

    def get_trainable_params(self) -> Dict[str, np.ndarray]:
        """Get trainable parameters."""
        return self._trainable_params

    def get_frozen_params(self) -> Dict[str, np.ndarray]:
        """Get frozen parameters."""
        return self._frozen_params

    def get_param_count(self) -> int:
        """Get total trainable parameter count."""
        return sum(p.size for p in self._trainable_params.values())

    def set_weights(
        self,
        weights: Dict[str, np.ndarray],
        strict: bool = True,
    ) -> List[str]:
        """
        Load weights into adapter.

        Args:
            weights: Dict of parameter name -> weight array
            strict: If True, raise error for missing/unexpected keys

        Returns:
            List of missing keys (if not strict)
        """
        missing = []
        for name, param in self._trainable_params.items():
            if name in weights:
                if weights[name].shape != param.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: "
                        f"expected {param.shape}, got {weights[name].shape}"
                    )
                self._trainable_params[name] = weights[name].copy()
            elif strict:
                raise KeyError(f"Missing weight: {name}")
            else:
                missing.append(name)
        return missing

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Get all adapter state for serialization."""
        state = {}
        for name, param in self._trainable_params.items():
            state[f"trainable.{name}"] = param.copy()
        for name, param in self._frozen_params.items():
            state[f"frozen.{name}"] = param.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        """Load adapter state from serialization."""
        for key, value in state.items():
            if key.startswith("trainable."):
                name = key[len("trainable."):]
                if name in self._trainable_params:
                    self._trainable_params[name] = value.copy()
            elif key.startswith("frozen."):
                name = key[len("frozen."):]
                if name in self._frozen_params:
                    self._frozen_params[name] = value.copy()


class LoRAAdapter(BaseAdapter):
    """
    Standard LoRA adapter.

    Implements: y = y + scaling * (x @ A.T) @ B.T
    where A ∈ R^{rank × in_features}, B ∈ R^{out_features × rank}
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize LoRA weights with Kaiming initialization for A, zeros for B."""
        if rng is None:
            rng = np.random.default_rng()

        rank = self.config.rank

        # A: Kaiming uniform initialization
        # B: Zero initialization (standard LoRA)
        bound = math.sqrt(6.0 / (self.in_features + rank))
        self._trainable_params["lora_A"] = rng.uniform(
            -bound, bound, size=(rank, self.in_features)
        ).astype(np.float32)

        self._trainable_params["lora_B"] = np.zeros(
            (self.out_features, rank), dtype=np.float32
        )

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute LoRA delta: scaling * (x @ A.T) @ B.T"""
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]

        # Apply dropout during training (if configured)
        # For inference, dropout is disabled

        # x @ A.T: [batch, hidden] @ [hidden, rank] -> [batch, rank]
        intermediate = np.matmul(x, lora_A.T)

        # intermediate @ B.T: [batch, rank] @ [rank, out] -> [batch, out]
        delta = np.matmul(intermediate, lora_B.T)

        return self.config.scaling * delta

    def get_delta_weight(self) -> np.ndarray:
        """Get merged weight delta: scaling * B @ A"""
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]
        return self.config.scaling * (lora_B @ lora_A)


class rsLoRAAdapter(LoRAAdapter):
    """
    Rank-Stabilized LoRA adapter.

    Uses scaling = α/√r instead of α/r to maintain stable gradients
    at higher ranks. Enables effective fine-tuning with larger ranks.

    Reference: https://arxiv.org/abs/2312.03732
    """

    def __init__(self, config: AdapterConfig, in_features: int, out_features: int):
        # Force rank-stabilized scaling
        config.scaling_strategy = ScalingStrategy.RANK_STABILIZED
        super().__init__(config, in_features, out_features)


class LoRAFAAdapter(BaseAdapter):
    """
    LoRA with Frozen-A adapter.

    Freezes the A matrix after initialization, only training B.
    This reduces activation memory by ~1.4x since we only need to
    store the low-rank intermediate (x @ A) for backprop, not full x.

    Reference: https://arxiv.org/abs/2308.03303
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize with frozen A (Kaiming) and trainable B (zeros)."""
        if rng is None:
            rng = np.random.default_rng()

        rank = self.config.rank

        # A: Kaiming initialization, then FROZEN
        bound = math.sqrt(6.0 / (self.in_features + rank))
        self._frozen_params["lora_A"] = rng.uniform(
            -bound, bound, size=(rank, self.in_features)
        ).astype(np.float32)

        # B: Trainable, initialized to zeros
        self._trainable_params["lora_B"] = np.zeros(
            (self.out_features, rank), dtype=np.float32
        )

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute LoRA-FA delta: scaling * (x @ A_frozen.T) @ B.T"""
        lora_A = self._frozen_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]

        intermediate = np.matmul(x, lora_A.T)
        delta = np.matmul(intermediate, lora_B.T)

        return self.config.scaling * delta

    def get_delta_weight(self) -> np.ndarray:
        """Get merged weight delta."""
        lora_A = self._frozen_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]
        return self.config.scaling * (lora_B @ lora_A)

    def get_param_count(self) -> int:
        """Only B is trainable."""
        return self._trainable_params["lora_B"].size


class DoRAAdapter(BaseAdapter):
    """
    Weight-Decomposed Low-Rank Adaptation (DoRA).

    Decomposes weight updates into magnitude and direction:
    W' = m * (W + BA) / ||W + BA||_c

    where m is a learnable magnitude vector and the direction is updated
    via standard LoRA. This provides better learning dynamics than LoRA.

    Reference: https://arxiv.org/abs/2402.09353
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize DoRA weights including magnitude vector."""
        if rng is None:
            rng = np.random.default_rng()

        rank = self.config.rank

        # LoRA components (same as standard LoRA)
        bound = math.sqrt(6.0 / (self.in_features + rank))
        self._trainable_params["lora_A"] = rng.uniform(
            -bound, bound, size=(rank, self.in_features)
        ).astype(np.float32)

        self._trainable_params["lora_B"] = np.zeros(
            (self.out_features, rank), dtype=np.float32
        )

        # Magnitude vector: initialized from pretrained weight norm
        # Will be set properly when original_weight is provided
        self._trainable_params["magnitude"] = np.ones(
            (1, self.out_features), dtype=np.float32
        )

        # Flag to track if magnitude has been initialized from pretrained weights
        self._magnitude_initialized = False

    def _initialize_magnitude_from_weight(self, weight: np.ndarray) -> None:
        """Initialize magnitude from pretrained weight matrix."""
        if not self._magnitude_initialized:
            # magnitude = ||W||_column (L2 norm per output dimension)
            self._trainable_params["magnitude"] = np.linalg.norm(
                weight, axis=1, keepdims=True
            ).T.astype(np.float32)
            self._magnitude_initialized = True

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute DoRA delta.

        DoRA modifies the effective weight: W' = m * (W + sBA) / ||W + sBA||

        For delta computation, we need:
        delta = x @ (W' - W).T = x @ (m * norm_dir - W).T

        where norm_dir = (W + sBA) / ||W + sBA||
        """
        if original_weight is None:
            # Fall back to standard LoRA if no original weight provided
            return self._forward_lora_only(x)

        # Initialize magnitude from original weight if needed
        self._initialize_magnitude_from_weight(original_weight)

        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]
        magnitude = self._trainable_params["magnitude"]

        # Compute LoRA update
        lora_update = self.config.scaling * (lora_B @ lora_A)

        # Updated weight
        updated_weight = original_weight + lora_update

        # Normalize by column
        column_norm = np.linalg.norm(updated_weight, axis=1, keepdims=True)
        column_norm = np.maximum(column_norm, 1e-8)  # Avoid division by zero
        normalized_direction = updated_weight / column_norm

        # Apply magnitude
        effective_weight = magnitude.T * normalized_direction

        # Compute delta: difference from original
        delta_weight = effective_weight - original_weight

        # Apply to input
        return np.matmul(x, delta_weight.T)

    def _forward_lora_only(self, x: np.ndarray) -> np.ndarray:
        """Fallback to standard LoRA computation."""
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]

        intermediate = np.matmul(x, lora_A.T)
        delta = np.matmul(intermediate, lora_B.T)

        return self.config.scaling * delta

    def get_delta_weight(self) -> np.ndarray:
        """
        Get merged weight delta for DoRA.

        Note: DoRA merge is more complex as it depends on the original weight.
        Returns the LoRA delta only; full merge requires original weight.
        """
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]
        return self.config.scaling * (lora_B @ lora_A)


class VeRAAdapter(BaseAdapter):
    """
    Vector-based Random Matrix Adaptation (VeRA).

    Uses shared frozen random matrices with learnable scaling vectors:
    Δ = Λ_b ⊙ (B @ (Λ_d ⊙ A))

    where A, B are FROZEN random matrices shared across all layers,
    and Λ_d, Λ_b are trainable scaling vectors per layer.

    Achieves ~10x fewer parameters than LoRA with comparable performance.

    Reference: https://arxiv.org/abs/2310.11454
    """

    # Class-level shared random matrices (created once, shared across instances)
    _shared_A: Optional[np.ndarray] = None
    _shared_B: Optional[np.ndarray] = None
    _shared_seed: Optional[int] = None
    _shared_max_in: int = 0
    _shared_max_out: int = 0
    _shared_rank: int = 0

    def __init__(self, config: AdapterConfig, in_features: int, out_features: int):
        super().__init__(config, in_features, out_features)
        self._ensure_shared_matrices()

    @classmethod
    def _ensure_shared_matrices(cls) -> None:
        """Ensure shared random matrices exist."""
        # This will be properly initialized when initialize_weights is called
        pass

    @classmethod
    def reset_shared_matrices(cls) -> None:
        """Reset shared matrices (for testing)."""
        cls._shared_A = None
        cls._shared_B = None
        cls._shared_seed = None

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """
        Initialize VeRA weights.

        Creates/extends shared random matrices if needed, and initializes
        the trainable scaling vectors Λ_d and Λ_b.
        """
        seed = self.config.projection_seed
        rank = self.config.rank

        # Check if we need to create or extend shared matrices
        need_new_matrices = (
            VeRAAdapter._shared_A is None or
            VeRAAdapter._shared_seed != seed or
            VeRAAdapter._shared_rank != rank or
            self.in_features > VeRAAdapter._shared_max_in or
            self.out_features > VeRAAdapter._shared_max_out
        )

        if need_new_matrices:
            # Create new shared matrices with maximum dimensions seen so far
            max_in = max(self.in_features, VeRAAdapter._shared_max_in)
            max_out = max(self.out_features, VeRAAdapter._shared_max_out)

            shared_rng = np.random.default_rng(seed)

            # A: [rank, max_in] - Kaiming initialization
            bound_a = math.sqrt(6.0 / (max_in + rank))
            VeRAAdapter._shared_A = shared_rng.uniform(
                -bound_a, bound_a, size=(rank, max_in)
            ).astype(np.float32)

            # B: [max_out, rank] - Kaiming initialization
            bound_b = math.sqrt(6.0 / (max_out + rank))
            VeRAAdapter._shared_B = shared_rng.uniform(
                -bound_b, bound_b, size=(max_out, rank)
            ).astype(np.float32)

            VeRAAdapter._shared_seed = seed
            VeRAAdapter._shared_rank = rank
            VeRAAdapter._shared_max_in = max_in
            VeRAAdapter._shared_max_out = max_out

        # Store references to the relevant slices (frozen)
        self._frozen_params["lora_A"] = VeRAAdapter._shared_A[:, :self.in_features]
        self._frozen_params["lora_B"] = VeRAAdapter._shared_B[:self.out_features, :]

        # Initialize trainable scaling vectors
        self._trainable_params["lambda_d"] = np.full(
            (rank,), self.config.vera_d_initial, dtype=np.float32
        )
        self._trainable_params["lambda_b"] = np.full(
            (self.out_features,), self.config.vera_b_initial, dtype=np.float32
        )

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute VeRA delta: Λ_b ⊙ (B @ (Λ_d ⊙ (A @ x.T))).T

        With proper broadcasting: Λ_b ⊙ ((Λ_d ⊙ (x @ A.T)) @ B.T)
        """
        lora_A = self._frozen_params["lora_A"]  # [rank, in]
        lora_B = self._frozen_params["lora_B"]  # [out, rank]
        lambda_d = self._trainable_params["lambda_d"]  # [rank]
        lambda_b = self._trainable_params["lambda_b"]  # [out]

        # x @ A.T: [batch, hidden] @ [hidden, rank] -> [batch, rank]
        intermediate = np.matmul(x, lora_A.T)

        # Scale by λ_d: [batch, rank] * [rank] -> [batch, rank]
        scaled = intermediate * lambda_d

        # @ B.T: [batch, rank] @ [rank, out] -> [batch, out]
        output = np.matmul(scaled, lora_B.T)

        # Scale by λ_b: [batch, out] * [out] -> [batch, out]
        delta = output * lambda_b

        return delta

    def get_delta_weight(self) -> np.ndarray:
        """Get merged weight delta for VeRA."""
        lora_A = self._frozen_params["lora_A"]
        lora_B = self._frozen_params["lora_B"]
        lambda_d = self._trainable_params["lambda_d"]
        lambda_b = self._trainable_params["lambda_b"]

        # (λ_b @ (B @ diag(λ_d) @ A))
        scaled_A = lambda_d[:, np.newaxis] * lora_A  # [rank, in]
        BA = lora_B @ scaled_A  # [out, in]
        return (lambda_b[:, np.newaxis] * BA)

    def get_param_count(self) -> int:
        """Only λ_d and λ_b are trainable."""
        return (
            self._trainable_params["lambda_d"].size +
            self._trainable_params["lambda_b"].size
        )


class AdaLoRAAdapter(BaseAdapter):
    """
    Adaptive Budget Allocation LoRA (AdaLoRA).

    Uses SVD parameterization with importance-based pruning:
    ΔW = P @ Λ @ Q

    where P ∈ R^{out × r}, Λ = diag(λ_1, ..., λ_r), Q ∈ R^{r × in}

    Importance scores guide which triplets to prune during training,
    allowing adaptive rank allocation across layers.

    Reference: https://arxiv.org/abs/2303.10512
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize AdaLoRA with SVD parameterization."""
        if rng is None:
            rng = np.random.default_rng()

        initial_rank = self.config.initial_rank or self.config.rank

        # P: Left singular vectors [out_features, initial_rank]
        # Orthogonal initialization
        P = rng.standard_normal((self.out_features, initial_rank)).astype(np.float32)
        P, _ = np.linalg.qr(P)
        if P.shape[1] < initial_rank:
            # Pad if needed
            padding = rng.standard_normal((self.out_features, initial_rank - P.shape[1])).astype(np.float32)
            P = np.concatenate([P, padding], axis=1)
        self._trainable_params["P"] = P[:, :initial_rank]

        # Λ: Singular values [initial_rank]
        # Initialize small to allow gradual growth
        self._trainable_params["Lambda"] = np.full(
            (initial_rank,), 0.01, dtype=np.float32
        )

        # Q: Right singular vectors [initial_rank, in_features]
        # Orthogonal initialization
        Q = rng.standard_normal((initial_rank, self.in_features)).astype(np.float32)
        Q, _ = np.linalg.qr(Q.T)
        Q = Q.T
        if Q.shape[0] < initial_rank:
            padding = rng.standard_normal((initial_rank - Q.shape[0], self.in_features)).astype(np.float32)
            Q = np.concatenate([Q, padding], axis=0)
        self._trainable_params["Q"] = Q[:initial_rank, :]

        # Importance scores (not trained, computed from gradients)
        self._importance_scores = np.zeros(initial_rank, dtype=np.float32)

        # Mask for pruned triplets
        self._rank_mask = np.ones(initial_rank, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute AdaLoRA delta: x @ (P @ diag(Λ * mask) @ Q).T"""
        P = self._trainable_params["P"]
        Lambda = self._trainable_params["Lambda"]
        Q = self._trainable_params["Q"]

        # Apply mask and scaling
        masked_lambda = Lambda * self._rank_mask * self.config.scaling

        # x @ Q.T: [batch, in] @ [in, rank] -> [batch, rank]
        intermediate = np.matmul(x, Q.T)

        # Scale by masked lambda: [batch, rank] * [rank] -> [batch, rank]
        scaled = intermediate * masked_lambda

        # @ P.T: [batch, rank] @ [rank, out] -> [batch, out]
        delta = np.matmul(scaled, P.T)

        return delta

    def update_importance_scores(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> None:
        """
        Update importance scores based on gradients.

        Score = |λ_i * ∇λ_i| + mean(|P_i * ∇P_i|) + mean(|Q_i * ∇Q_i|)
        """
        beta = self.config.importance_beta

        Lambda = self._trainable_params["Lambda"]
        P = self._trainable_params["P"]
        Q = self._trainable_params["Q"]

        grad_lambda = gradients.get("Lambda", np.zeros_like(Lambda))
        grad_P = gradients.get("P", np.zeros_like(P))
        grad_Q = gradients.get("Q", np.zeros_like(Q))

        # Compute sensitivity scores
        lambda_score = np.abs(Lambda * grad_lambda)
        P_score = np.mean(np.abs(P * grad_P), axis=0)
        Q_score = np.mean(np.abs(Q * grad_Q), axis=1)

        triplet_scores = lambda_score + P_score + Q_score

        # EMA update
        self._importance_scores = (
            beta * self._importance_scores +
            (1 - beta) * triplet_scores
        )

    def prune_to_budget(self, target_rank: int) -> None:
        """Prune triplets to meet target rank budget."""
        current_active = int(self._rank_mask.sum())

        if current_active <= target_rank:
            return

        # Find threshold to keep top-k triplets
        scores = self._importance_scores * self._rank_mask
        threshold_idx = np.argsort(scores)[-(target_rank):]

        new_mask = np.zeros_like(self._rank_mask)
        new_mask[threshold_idx] = 1.0
        self._rank_mask = new_mask

    def get_delta_weight(self) -> np.ndarray:
        """Get merged weight delta."""
        P = self._trainable_params["P"]
        Lambda = self._trainable_params["Lambda"]
        Q = self._trainable_params["Q"]

        masked_lambda = Lambda * self._rank_mask * self.config.scaling

        # P @ diag(Λ) @ Q
        scaled_Q = masked_lambda[:, np.newaxis] * Q
        return P @ scaled_Q

    def get_effective_rank(self) -> int:
        """Get current effective rank (non-pruned triplets)."""
        return int(self._rank_mask.sum())


class GatedLoRAAdapter(BaseAdapter):
    """
    Gated LoRA adapter with learnable gates.

    Implements: y = y + gate(x) * scaling * (x @ A.T) @ B.T

    The gate controls how much of the LoRA update to apply,
    allowing the model to dynamically adjust adapter influence.
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize gated LoRA weights."""
        if rng is None:
            rng = np.random.default_rng()

        rank = self.config.rank

        # Standard LoRA components
        bound = math.sqrt(6.0 / (self.in_features + rank))
        self._trainable_params["lora_A"] = rng.uniform(
            -bound, bound, size=(rank, self.in_features)
        ).astype(np.float32)

        self._trainable_params["lora_B"] = np.zeros(
            (self.out_features, rank), dtype=np.float32
        )

        # Gate components
        gate_type = self.config.gate_type

        if gate_type == "learned":
            # Learnable gate vector
            self._trainable_params["gate"] = np.zeros(
                (self.out_features,), dtype=np.float32
            )
        elif gate_type in ["sigmoid", "tanh"]:
            # Linear projection for gate computation
            self._trainable_params["gate_proj"] = rng.uniform(
                -0.01, 0.01, size=(1, self.in_features)
            ).astype(np.float32)
            self._trainable_params["gate_bias"] = np.zeros(
                (1,), dtype=np.float32
            )

    def _compute_gate(self, x: np.ndarray) -> np.ndarray:
        """Compute gate values from input."""
        gate_type = self.config.gate_type

        if gate_type == "learned":
            # Static learned gate
            gate = self._trainable_params["gate"]
            return 1.0 / (1.0 + np.exp(-gate))  # Sigmoid

        # Input-dependent gate
        gate_proj = self._trainable_params["gate_proj"]
        gate_bias = self._trainable_params["gate_bias"]

        # [batch, hidden] @ [hidden, 1] -> [batch, 1]
        gate_logits = np.matmul(x, gate_proj.T) + gate_bias

        if gate_type == "sigmoid":
            return 1.0 / (1.0 + np.exp(-gate_logits))
        elif gate_type == "tanh":
            return np.tanh(gate_logits)
        else:
            return np.ones_like(gate_logits)

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute gated LoRA delta."""
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]

        # Compute LoRA output
        intermediate = np.matmul(x, lora_A.T)
        lora_output = np.matmul(intermediate, lora_B.T)

        # Compute and apply gate
        gate = self._compute_gate(x)

        return gate * self.config.scaling * lora_output

    def get_delta_weight(self) -> np.ndarray:
        """
        Get merged weight delta.

        Note: Gated LoRA cannot be fully merged since the gate is input-dependent.
        Returns the LoRA delta without gating for approximate merge.
        """
        lora_A = self._trainable_params["lora_A"]
        lora_B = self._trainable_params["lora_B"]
        return self.config.scaling * (lora_B @ lora_A)

    def can_merge(self) -> bool:
        """Gated LoRA cannot be fully merged due to input-dependent gate."""
        return self.config.gate_type == "learned"


class GLoRAAdapter(BaseAdapter):
    """
    Generalized LoRA (GLoRA) adapter.

    Provides a unified framework supporting:
    - Weight scaling: W * A
    - Weight shifting: W + B
    - Bias scaling: bias * D
    - Bias shifting: bias + E
    - Prompt generation: W @ C

    The support tensors can be scalars, vectors, or low-rank matrices
    depending on configuration.

    Reference: https://arxiv.org/abs/2306.07967
    """

    def initialize_weights(self, rng: Optional[np.random.Generator] = None) -> None:
        """Initialize GLoRA support tensors."""
        if rng is None:
            rng = np.random.default_rng()

        rank = self.config.rank

        # Core LoRA for weight shifting (B tensor)
        if self.config.use_weight_shift:
            bound = math.sqrt(6.0 / (self.in_features + rank))
            self._trainable_params["shift_A"] = rng.uniform(
                -bound, bound, size=(rank, self.in_features)
            ).astype(np.float32)
            self._trainable_params["shift_B"] = np.zeros(
                (self.out_features, rank), dtype=np.float32
            )

        # Weight scaling (A tensor) - vector form
        if self.config.use_weight_scaling:
            self._trainable_params["scale_weight"] = np.ones(
                (self.out_features,), dtype=np.float32
            )

        # Bias scaling (D tensor)
        if self.config.use_bias_scaling:
            self._trainable_params["scale_bias"] = np.ones(
                (self.out_features,), dtype=np.float32
            )

        # Bias shifting (E tensor)
        if self.config.use_bias_shift:
            self._trainable_params["shift_bias"] = np.zeros(
                (self.out_features,), dtype=np.float32
            )

        # Prompt tensor (C)
        if self.config.use_prompt:
            prompt_len = self.config.extra_config.get("prompt_length", 8)
            self._trainable_params["prompt_C"] = rng.uniform(
                -0.01, 0.01, size=(prompt_len, self.in_features)
            ).astype(np.float32)

    def forward(
        self,
        x: np.ndarray,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute GLoRA output.

        optimal_weight = W + W*A + B (scaling + shifting)
        optimal_bias = bias + bias*D + E
        prompt_contribution = W @ C
        """
        if original_weight is None and original_output is None:
            raise ValueError("GLoRA requires either original_weight or original_output")

        delta = np.zeros_like(x[..., :self.out_features] if x.shape[-1] > self.out_features else x)

        # Weight shifting (LoRA-style)
        if self.config.use_weight_shift:
            shift_A = self._trainable_params["shift_A"]
            shift_B = self._trainable_params["shift_B"]
            intermediate = np.matmul(x, shift_A.T)
            shift = np.matmul(intermediate, shift_B.T)
            delta = delta + self.config.scaling * shift

        # Weight scaling (applied to original output)
        if self.config.use_weight_scaling and original_output is not None:
            scale_weight = self._trainable_params["scale_weight"]
            delta = delta + original_output * (scale_weight - 1.0)

        # Bias shifting
        if self.config.use_bias_shift:
            shift_bias = self._trainable_params["shift_bias"]
            delta = delta + shift_bias

        return delta

    def get_delta_weight(self) -> np.ndarray:
        """Get merged weight delta (shift component only)."""
        if not self.config.use_weight_shift:
            return np.zeros((self.out_features, self.in_features), dtype=np.float32)

        shift_A = self._trainable_params["shift_A"]
        shift_B = self._trainable_params["shift_B"]
        return self.config.scaling * (shift_B @ shift_A)

    def can_merge(self) -> bool:
        """GLoRA can only partially merge (the shift component)."""
        return self.config.use_weight_shift and not (
            self.config.use_weight_scaling or
            self.config.use_prompt
        )


# =============================================================================
# ADAPTER FACTORY
# =============================================================================

_ADAPTER_REGISTRY: Dict[AdapterType, type] = {
    AdapterType.LORA: LoRAAdapter,
    AdapterType.RS_LORA: rsLoRAAdapter,
    AdapterType.LORA_FA: LoRAFAAdapter,
    AdapterType.DORA: DoRAAdapter,
    AdapterType.VERA: VeRAAdapter,
    AdapterType.ADALORA: AdaLoRAAdapter,
    AdapterType.GATED_LORA: GatedLoRAAdapter,
    AdapterType.GLORA: GLoRAAdapter,
}


def create_adapter(
    config: AdapterConfig,
    in_features: int,
    out_features: int,
    rng: Optional[np.random.Generator] = None,
) -> BaseAdapter:
    """
    Factory function to create an adapter instance.

    Args:
        config: Adapter configuration
        in_features: Input dimension
        out_features: Output dimension
        rng: Optional random number generator for reproducibility

    Returns:
        Initialized adapter instance
    """
    if config.adapter_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter type: {config.adapter_type}")

    # Validate configuration
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid adapter config: {errors}")

    # Create adapter
    adapter_cls = _ADAPTER_REGISTRY[config.adapter_type]
    adapter = adapter_cls(config, in_features, out_features)

    # Initialize weights
    adapter.initialize_weights(rng)

    return adapter


def register_adapter_type(adapter_type: AdapterType, adapter_cls: type) -> None:
    """Register a custom adapter type."""
    if not issubclass(adapter_cls, BaseAdapter):
        raise TypeError("adapter_cls must be a subclass of BaseAdapter")
    _ADAPTER_REGISTRY[adapter_type] = adapter_cls


def list_adapter_types() -> List[AdapterType]:
    """List all registered adapter types."""
    return list(_ADAPTER_REGISTRY.keys())


# =============================================================================
# ADAPTER CONVERSION UTILITIES
# =============================================================================

def convert_adapter(
    source: BaseAdapter,
    target_type: AdapterType,
    target_config: Optional[AdapterConfig] = None,
) -> BaseAdapter:
    """
    Convert an adapter from one type to another.

    Not all conversions are possible. This function handles the common case
    of converting standard LoRA to other compatible types.

    Args:
        source: Source adapter
        target_type: Target adapter type
        target_config: Optional config for target (inherits from source if not provided)

    Returns:
        New adapter instance with converted weights
    """
    if target_config is None:
        target_config = AdapterConfig(
            adapter_type=target_type,
            rank=source.config.rank,
            alpha=source.config.alpha,
            target_modules=source.config.target_modules,
        )

    target = create_adapter(
        target_config,
        source.in_features,
        source.out_features,
    )

    # Transfer compatible weights
    source_params = source.get_trainable_params()
    target_params = target.get_trainable_params()

    # Common LoRA weights
    if "lora_A" in source_params and "lora_A" in target_params:
        target._trainable_params["lora_A"] = source_params["lora_A"].copy()
    if "lora_B" in source_params and "lora_B" in target_params:
        target._trainable_params["lora_B"] = source_params["lora_B"].copy()

    return target


__all__ = [
    "AdapterType",
    "ScalingStrategy",
    "AdapterConfig",
    "BaseAdapter",
    "LoRAAdapter",
    "rsLoRAAdapter",
    "LoRAFAAdapter",
    "DoRAAdapter",
    "VeRAAdapter",
    "AdaLoRAAdapter",
    "GatedLoRAAdapter",
    "GLoRAAdapter",
    "create_adapter",
    "register_adapter_type",
    "list_adapter_types",
    "convert_adapter",
]
