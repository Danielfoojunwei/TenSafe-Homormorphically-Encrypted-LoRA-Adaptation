"""
Privacy-Preserving LoRA for Homomorphic Encryption and Federated Learning

Implements privacy-preserving LoRA techniques:
- FFA-LoRA (Federated Freeze A LoRA): Freezes A matrix for HE compatibility
- HE-compatible initialization
- Federated aggregation strategies

Based on research from:
- "Improving LoRA in Privacy-preserving Federated Learning" (ICLR 2024)
- "Private LoRA Fine-tuning with Homomorphic Encryption"
- "SHE-LoRA: Selective Homomorphic Encryption"

These techniques are particularly relevant for TenSafe's HE-LoRA system.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class PrivacyMode(Enum):
    """Privacy mode for LoRA training."""

    STANDARD = "standard"
    """No privacy constraints, standard LoRA."""

    FFA_LORA = "ffa_lora"
    """Federated Freeze A LoRA: freeze A matrix, only train B."""

    HE_COMPATIBLE = "he_compatible"
    """Homomorphic encryption compatible mode."""

    DIFFERENTIAL_PRIVACY = "differential_privacy"
    """Differential privacy with noise injection."""


@dataclass
class FFALoRAConfig:
    """
    Configuration for FFA-LoRA (Federated Freeze A LoRA).

    FFA-LoRA addresses challenges in privacy-preserving federated LoRA:
    1. Data heterogeneity effects
    2. DP noise amplification through A×B
    3. Discordance between local optimization and global aggregation

    Solution: Freeze the randomly initialized A matrix, only train B.
    This halves communication costs and improves convergence.

    From "Improving LoRA in Privacy-preserving Federated Learning" (ICLR 2024)
    """

    rank: int = 32
    alpha: float = 64.0

    # A matrix initialization
    a_init_method: str = "kaiming"  # "kaiming", "xavier", "normal", "orthogonal"
    a_init_seed: Optional[int] = 42  # For reproducibility across clients

    # B matrix initialization (always zeros in standard LoRA)
    b_init_zeros: bool = True

    # Training behavior
    freeze_a: bool = True  # Core of FFA-LoRA
    train_b_only: bool = True

    # Federated settings
    local_epochs: int = 1
    aggregation_method: str = "fedavg"  # "fedavg", "fedprox", "scaffold"

    # Privacy settings
    use_differential_privacy: bool = False
    dp_epsilon: float = 8.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0

    @property
    def scaling_factor(self) -> float:
        """Compute scaling factor."""
        return self.alpha / self.rank


@dataclass
class HELoRAConfig:
    """
    Configuration for Homomorphic Encryption compatible LoRA.

    Optimized for scenarios where:
    - Client owns data and LoRA weights
    - Server performs HE computations on base model
    - Communication is encrypted
    """

    rank: int = 32
    alpha: float = 64.0

    # HE-specific settings
    use_quantization: bool = True
    quantization_bits: int = 8  # Bits for weight quantization

    # Matrix properties for HE efficiency
    use_square_matrices: bool = False  # MoRA-style for some operations
    block_size: Optional[int] = None  # For blocked computation

    # Packing strategy
    use_column_packing: bool = True  # MOAI-style packing
    use_interleaved_batching: bool = True

    # Noise budget management
    max_multiplicative_depth: int = 4
    target_precision_bits: int = 20

    # FFA-LoRA integration
    freeze_a_for_he: bool = True  # Recommended for HE scenarios


class PrivacyPreservingLoRA:
    """
    Privacy-preserving LoRA implementation.

    Provides utilities for:
    1. FFA-LoRA configuration for federated learning
    2. HE-compatible weight initialization
    3. Gradient clipping and noise injection for DP
    4. Federated aggregation strategies
    """

    def __init__(
        self,
        mode: PrivacyMode = PrivacyMode.FFA_LORA,
        ffa_config: Optional[FFALoRAConfig] = None,
        he_config: Optional[HELoRAConfig] = None,
    ):
        """
        Initialize privacy-preserving LoRA.

        Args:
            mode: Privacy mode to use
            ffa_config: FFA-LoRA configuration
            he_config: HE-LoRA configuration
        """
        self.mode = mode
        self.ffa_config = ffa_config or FFALoRAConfig()
        self.he_config = he_config or HELoRAConfig()

    def initialize_lora_weights(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        seed: Optional[int] = None,
    ) -> Tuple[Any, Any]:
        """
        Initialize LoRA A and B matrices with privacy-preserving settings.

        For FFA-LoRA: A is randomly initialized (but will be frozen)
        For HE: Uses quantization-friendly initialization

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank
            seed: Random seed for reproducibility

        Returns:
            Tuple of (A matrix, B matrix)
        """
        try:
            import torch

            if seed is not None:
                torch.manual_seed(seed)

            # Initialize A matrix
            if self.ffa_config.a_init_method == "kaiming":
                # Kaiming/He initialization
                std = math.sqrt(2.0 / in_features)
                A = torch.randn(rank, in_features) * std
            elif self.ffa_config.a_init_method == "xavier":
                # Xavier/Glorot initialization
                std = math.sqrt(2.0 / (in_features + rank))
                A = torch.randn(rank, in_features) * std
            elif self.ffa_config.a_init_method == "orthogonal":
                # Orthogonal initialization (good for preserving gradients)
                A = torch.empty(rank, in_features)
                torch.nn.init.orthogonal_(A)
            else:
                # Standard normal
                A = torch.randn(rank, in_features) / math.sqrt(rank)

            # Initialize B matrix (always zeros in standard LoRA/FFA-LoRA)
            if self.ffa_config.b_init_zeros:
                B = torch.zeros(out_features, rank)
            else:
                B = torch.randn(out_features, rank) * 0.01

            # Quantize if HE mode
            if self.mode == PrivacyMode.HE_COMPATIBLE and self.he_config.use_quantization:
                A = self._quantize_weights(A, self.he_config.quantization_bits)
                # Don't quantize B if zeros

            return A, B

        except ImportError:
            raise ImportError("PyTorch is required for weight initialization")

    def _quantize_weights(self, weights: Any, bits: int) -> Any:
        """Quantize weights to specified bit width."""
        try:
            import torch

            # Compute scale
            max_val = weights.abs().max()
            if max_val == 0:
                return weights

            scale = (2 ** (bits - 1) - 1) / max_val

            # Quantize and dequantize
            quantized = torch.round(weights * scale)
            dequantized = quantized / scale

            return dequantized
        except ImportError:
            return weights

    def get_trainable_params_mask(self, param_name: str) -> bool:
        """
        Determine if a parameter should be trainable.

        For FFA-LoRA, only B matrices are trainable.

        Args:
            param_name: Name of the parameter

        Returns:
            True if parameter should be trainable
        """
        if self.mode == PrivacyMode.FFA_LORA or self.ffa_config.freeze_a:
            # Only train B matrices
            if "lora_A" in param_name:
                return False
            if "lora_B" in param_name:
                return True
        return True

    def configure_model_for_privacy(self, model) -> None:
        """
        Configure a PEFT model for privacy-preserving training.

        Freezes A matrices if using FFA-LoRA mode.

        Args:
            model: PEFT model with LoRA adapters
        """
        if self.mode == PrivacyMode.FFA_LORA or self.ffa_config.freeze_a:
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    param.requires_grad = False
                elif "lora_B" in name:
                    param.requires_grad = True

    def clip_gradients(
        self,
        model,
        max_norm: float,
    ) -> float:
        """
        Clip gradients for differential privacy.

        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm

        Returns:
            Total gradient norm before clipping
        """
        try:
            import torch

            parameters = [p for p in model.parameters() if p.grad is not None]

            if len(parameters) == 0:
                return 0.0

            total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach()) for p in parameters])
            )

            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in parameters:
                    p.grad.detach().mul_(clip_coef)

            return total_norm.item()
        except ImportError:
            return 0.0

    def add_noise_for_dp(
        self,
        model,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> None:
        """
        Add calibrated Gaussian noise for differential privacy.

        Args:
            model: Model to add noise to
            noise_multiplier: Noise multiplier (σ in DP)
            max_grad_norm: Maximum gradient norm (for noise calibration)
        """
        try:
            import torch

            noise_std = noise_multiplier * max_grad_norm

            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * noise_std
                    param.grad.add_(noise)
        except ImportError:
            pass

    def get_communication_cost(self, model) -> Dict[str, int]:
        """
        Estimate communication cost for federated learning.

        FFA-LoRA halves the cost by only transmitting B matrices.

        Args:
            model: Model to estimate for

        Returns:
            Dictionary with parameter counts
        """
        a_params = 0
        b_params = 0
        other_params = 0

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            numel = param.numel()
            if "lora_A" in name:
                a_params += numel
            elif "lora_B" in name:
                b_params += numel
            else:
                other_params += numel

        return {
            "lora_a_params": a_params,
            "lora_b_params": b_params,
            "other_params": other_params,
            "total_trainable": a_params + b_params + other_params,
            "communication_per_round": (
                b_params + other_params if self.ffa_config.freeze_a
                else a_params + b_params + other_params
            ),
        }


def create_ffa_lora_config(
    rank: int = 32,
    use_dp: bool = False,
    dp_epsilon: float = 8.0,
) -> FFALoRAConfig:
    """
    Create FFA-LoRA configuration for federated/HE scenarios.

    Args:
        rank: LoRA rank
        use_dp: Whether to use differential privacy
        dp_epsilon: Privacy budget (lower = more private)

    Returns:
        Configured FFALoRAConfig
    """
    return FFALoRAConfig(
        rank=rank,
        alpha=2.0 * rank,  # Best practice
        freeze_a=True,
        train_b_only=True,
        use_differential_privacy=use_dp,
        dp_epsilon=dp_epsilon,
    )


def create_he_lora_config(
    rank: int = 32,
    quantization_bits: int = 8,
) -> HELoRAConfig:
    """
    Create HE-compatible LoRA configuration.

    Args:
        rank: LoRA rank
        quantization_bits: Bits for weight quantization

    Returns:
        Configured HELoRAConfig
    """
    return HELoRAConfig(
        rank=rank,
        alpha=2.0 * rank,
        use_quantization=True,
        quantization_bits=quantization_bits,
        use_column_packing=True,
        freeze_a_for_he=True,
    )


# Documentation
PRIVACY_LORA_GUIDE = """
Privacy-Preserving LoRA Guide
=============================

FFA-LoRA (Federated Freeze A LoRA)
----------------------------------
Problem: Standard LoRA in federated/HE settings has issues:
1. A and B matrices are optimized jointly locally
2. But aggregated separately on server
3. This discordance hurts convergence
4. DP noise gets amplified through A×B

Solution: Freeze A matrix (randomly initialized), only train B.

Benefits:
- 50% reduction in communication (only send B)
- Better convergence in federated settings
- DP noise not amplified through multiplication
- Reproducible across clients (same A with same seed)

Results (from paper):
- GSM-8K: 17.12% accuracy (vs 15.68% with standard LoRA)
- Consistently better under differential privacy

HE-Compatible LoRA
------------------
For homomorphic encryption scenarios:

1. Weight Quantization
   - Quantize weights to reduce HE computation cost
   - 8-bit quantization typically sufficient
   - Preserves most of the accuracy

2. Column Packing (MOAI-style)
   - Pack weights for SIMD operations
   - Minimizes rotations in HE
   - Critical for performance

3. Freeze A for HE
   - A matrix can be precomputed
   - Only B changes during training
   - Reduces encrypted computation

Integration with TenSafe
------------------------
The TenSafe HE-LoRA system uses these techniques:
1. FFA-LoRA for training (freeze A)
2. Column packing for inference
3. Quantization for efficiency
4. Every-token HE correction

Recommended Configuration:
    config = create_ffa_lora_config(
        rank=32,
        use_dp=True,
        dp_epsilon=8.0,
    )

For maximum privacy with HE:
    he_config = create_he_lora_config(
        rank=32,
        quantization_bits=8,
    )
"""
