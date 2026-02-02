"""
Adapter Configuration for N2HE/TFHE Integration

This module defines configuration options for LoRA adapters that use
TFHE programmable bootstrapping for non-linear operations.

Adapter Types:
    LINEAR_LORA: Standard LoRA (CKKS only, no activations)
    GATED_LORA: LoRA with gating mechanism (requires activation)
    NONLINEAR_LORA: LoRA with explicit non-linear transformation

The user selects adapter type at training initialization, which determines:
    - Whether TFHE bootstrapping is required
    - Where in the forward pass activations are applied
    - Memory/compute tradeoffs

Key Design Principle:
    TFHE programmable bootstrapping provides EXACT computation on discrete
    plaintexts, enabling precise non-linear function evaluation via LUTs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
import math


class AdapterType(Enum):
    """
    Types of LoRA adapters supported.

    LINEAR_LORA: Standard LoRA (y = Wx + BAx)
        - Uses CKKS with MOAI column packing
        - Fast, rotation-free Ct×Pt multiplication
        - No bootstrapping required

    GATED_LORA: Gated LoRA (y = Wx + σ(gate) * BAx)
        - Requires activation function σ for gating
        - Uses TFHE programmable bootstrap for exact gate evaluation
        - Higher latency but enables dynamic adaptation

    NONLINEAR_LORA: Non-linear LoRA (y = Wx + f(BAx))
        - Applies non-linear function f to adapter output
        - Uses TFHE programmable bootstrap for exact f evaluation
        - Enables expressive non-linear transformations
    """
    LINEAR_LORA = "linear"
    GATED_LORA = "gated"
    NONLINEAR_LORA = "nonlinear"


class NonLinearActivation(Enum):
    """
    Non-linear activation functions supported via TFHE LUT evaluation.

    All activations are computed EXACTLY on discrete message space
    via programmable bootstrapping.
    """
    RELU = "relu"           # max(0, x)
    GELU = "gelu"           # x * Φ(x) - Gaussian CDF
    SIGMOID = "sigmoid"     # 1 / (1 + exp(-x))
    TANH = "tanh"           # (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    SWISH = "swish"         # x * sigmoid(x)
    SIGN = "sign"           # -1, 0, or 1
    STEP = "step"           # 0 or 1 (Heaviside)

    def get_function(self) -> Callable[[float], float]:
        """Get the underlying mathematical function."""
        import math

        functions = {
            NonLinearActivation.RELU: lambda x: max(0.0, x),
            NonLinearActivation.GELU: lambda x: 0.5 * x * (1 + math.erf(x / math.sqrt(2))),
            NonLinearActivation.SIGMOID: lambda x: 1.0 / (1 + math.exp(-max(-500, min(500, x)))),
            NonLinearActivation.TANH: lambda x: math.tanh(x),
            NonLinearActivation.SWISH: lambda x: x / (1 + math.exp(-max(-500, min(500, x)))),
            NonLinearActivation.SIGN: lambda x: -1.0 if x < 0 else (1.0 if x > 0 else 0.0),
            NonLinearActivation.STEP: lambda x: 0.0 if x < 0 else 1.0,
        }
        return functions[self]

    def get_output_range(self, input_range: tuple) -> tuple:
        """Get expected output range given input range."""
        ranges = {
            NonLinearActivation.RELU: (0.0, max(0, input_range[1])),
            NonLinearActivation.GELU: (min(0, input_range[0] * 0.5), input_range[1]),
            NonLinearActivation.SIGMOID: (0.0, 1.0),
            NonLinearActivation.TANH: (-1.0, 1.0),
            NonLinearActivation.SWISH: (min(input_range[0] * 0.5, 0), input_range[1]),
            NonLinearActivation.SIGN: (-1.0, 1.0),
            NonLinearActivation.STEP: (0.0, 1.0),
        }
        return ranges[self]


class AdapterPlacement(Enum):
    """
    Where to apply the adapter in the transformer layer.

    ATTENTION_QKV: Apply to query, key, value projections
    ATTENTION_OUTPUT: Apply to attention output projection
    MLP_UP: Apply to MLP up-projection
    MLP_DOWN: Apply to MLP down-projection
    ALL: Apply to all projections
    """
    ATTENTION_QKV = "attention_qkv"
    ATTENTION_OUTPUT = "attention_output"
    MLP_UP = "mlp_up"
    MLP_DOWN = "mlp_down"
    ALL = "all"

    def get_target_modules(self) -> List[str]:
        """Get list of target module patterns."""
        module_map = {
            AdapterPlacement.ATTENTION_QKV: ["q_proj", "k_proj", "v_proj"],
            AdapterPlacement.ATTENTION_OUTPUT: ["o_proj"],
            AdapterPlacement.MLP_UP: ["gate_proj", "up_proj"],
            AdapterPlacement.MLP_DOWN: ["down_proj"],
            AdapterPlacement.ALL: [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
        }
        return module_map[self]


@dataclass
class N2HEAdapterConfig:
    """
    Configuration for N2HE/TFHE-enabled LoRA adapters.

    This configuration is specified at training initialization and
    determines the computation path through the HE system.

    Example:
        # Standard linear LoRA (CKKS only)
        config = N2HEAdapterConfig(
            adapter_type=AdapterType.LINEAR_LORA,
        )

        # Gated LoRA with sigmoid gate (requires TFHE)
        config = N2HEAdapterConfig(
            adapter_type=AdapterType.GATED_LORA,
            activation=NonLinearActivation.SIGMOID,
            use_tfhe_bootstrap=True,
        )

        # Non-linear LoRA with GELU (requires TFHE)
        config = N2HEAdapterConfig(
            adapter_type=AdapterType.NONLINEAR_LORA,
            activation=NonLinearActivation.GELU,
            use_tfhe_bootstrap=True,
        )
    """
    # Core adapter configuration
    adapter_type: AdapterType = AdapterType.LINEAR_LORA

    # LoRA hyperparameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0

    # Placement configuration
    placement: AdapterPlacement = AdapterPlacement.ATTENTION_QKV
    layer_indices: Optional[List[int]] = None  # None = all layers

    # Non-linear activation (for GATED_LORA and NONLINEAR_LORA)
    activation: Optional[NonLinearActivation] = None

    # TFHE configuration
    use_tfhe_bootstrap: bool = False
    tfhe_precision_bits: int = 8  # Bits of precision for LUT
    activation_range: tuple = (-10.0, 10.0)  # Expected activation value range

    # Performance options
    use_gpu_tfhe: bool = True  # GPU acceleration for TFHE
    use_cpu_fallback: bool = True  # Allow CPU fallback (FasterNTT)
    batch_bootstrap: bool = True  # Batch multiple bootstraps together

    # MOAI integration (for linear part)
    use_moai_packing: bool = True  # Column packing for rotation-free matmul

    def __post_init__(self):
        """Validate configuration."""
        self._validate()

    def _validate(self):
        """Ensure configuration is consistent."""
        # Non-linear adapters require activation and TFHE
        if self.adapter_type in (AdapterType.GATED_LORA, AdapterType.NONLINEAR_LORA):
            if self.activation is None:
                raise ValueError(
                    f"Adapter type {self.adapter_type.value} requires activation function"
                )
            if not self.use_tfhe_bootstrap:
                raise ValueError(
                    f"Adapter type {self.adapter_type.value} requires use_tfhe_bootstrap=True"
                )

        # Linear LoRA should not use TFHE
        if self.adapter_type == AdapterType.LINEAR_LORA:
            if self.use_tfhe_bootstrap:
                raise ValueError(
                    "LINEAR_LORA does not need TFHE bootstrapping. "
                    "Set use_tfhe_bootstrap=False or use a non-linear adapter type."
                )

        # Precision must fit in reasonable bounds
        if not (4 <= self.tfhe_precision_bits <= 12):
            raise ValueError(
                f"TFHE precision bits must be in [4, 12], got {self.tfhe_precision_bits}"
            )

        # Activation range must be valid
        if self.activation_range[0] >= self.activation_range[1]:
            raise ValueError(
                f"Invalid activation range: {self.activation_range}"
            )

    @property
    def requires_tfhe(self) -> bool:
        """Check if this configuration requires TFHE bootstrapping."""
        return self.use_tfhe_bootstrap and self.adapter_type != AdapterType.LINEAR_LORA

    @property
    def target_modules(self) -> List[str]:
        """Get list of target module patterns."""
        return self.placement.get_target_modules()

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank

    @property
    def message_space_size(self) -> int:
        """Size of discrete message space for TFHE."""
        return 2 ** self.tfhe_precision_bits

    def get_lut_for_activation(self) -> Optional[List[int]]:
        """
        Generate quantized LUT for the configured activation.

        Returns:
            List of discrete output values, or None if no activation.
        """
        if self.activation is None:
            return None

        p = self.message_space_size
        func = self.activation.get_function()
        input_min, input_max = self.activation_range
        output_min, output_max = self.activation.get_output_range(self.activation_range)

        lut = []
        for i in range(p):
            # Map discrete input to real value
            x = input_min + (input_max - input_min) * i / (p - 1)
            # Apply activation
            y = func(x)
            # Quantize output
            y_clamped = max(output_min, min(output_max, y))
            y_normalized = (y_clamped - output_min) / (output_max - output_min)
            y_discrete = int(round(y_normalized * (p - 1)))
            lut.append(max(0, min(p - 1, y_discrete)))

        return lut

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            'adapter_type': self.adapter_type.value,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'placement': self.placement.value,
            'layer_indices': self.layer_indices,
            'activation': self.activation.value if self.activation else None,
            'use_tfhe_bootstrap': self.use_tfhe_bootstrap,
            'tfhe_precision_bits': self.tfhe_precision_bits,
            'activation_range': self.activation_range,
            'use_gpu_tfhe': self.use_gpu_tfhe,
            'use_cpu_fallback': self.use_cpu_fallback,
            'batch_bootstrap': self.batch_bootstrap,
            'use_moai_packing': self.use_moai_packing,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'N2HEAdapterConfig':
        """Deserialize configuration from dictionary."""
        return cls(
            adapter_type=AdapterType(d['adapter_type']),
            rank=d.get('rank', 16),
            alpha=d.get('alpha', 32.0),
            dropout=d.get('dropout', 0.0),
            placement=AdapterPlacement(d.get('placement', 'attention_qkv')),
            layer_indices=d.get('layer_indices'),
            activation=NonLinearActivation(d['activation']) if d.get('activation') else None,
            use_tfhe_bootstrap=d.get('use_tfhe_bootstrap', False),
            tfhe_precision_bits=d.get('tfhe_precision_bits', 8),
            activation_range=tuple(d.get('activation_range', (-10.0, 10.0))),
            use_gpu_tfhe=d.get('use_gpu_tfhe', True),
            use_cpu_fallback=d.get('use_cpu_fallback', True),
            batch_bootstrap=d.get('batch_bootstrap', True),
            use_moai_packing=d.get('use_moai_packing', True),
        )


def validate_adapter_config(config: N2HEAdapterConfig, model_config: dict) -> List[str]:
    """
    Validate adapter configuration against model configuration.

    Args:
        config: Adapter configuration to validate
        model_config: Model configuration dict with hidden_size, num_layers, etc.

    Returns:
        List of warning/error messages (empty if valid)
    """
    messages = []

    hidden_size = model_config.get('hidden_size', 4096)
    num_layers = model_config.get('num_hidden_layers', 32)

    # Check rank vs hidden size
    if config.rank > hidden_size // 4:
        messages.append(
            f"LoRA rank {config.rank} is large relative to hidden_size {hidden_size}. "
            f"Consider rank <= {hidden_size // 8} for efficiency."
        )

    # Check layer indices
    if config.layer_indices is not None:
        invalid_layers = [l for l in config.layer_indices if l < 0 or l >= num_layers]
        if invalid_layers:
            messages.append(
                f"Invalid layer indices {invalid_layers}. "
                f"Model has {num_layers} layers (0-{num_layers-1})."
            )

    # Check TFHE precision vs memory
    if config.use_tfhe_bootstrap and config.tfhe_precision_bits > 10:
        lut_size = 2 ** config.tfhe_precision_bits
        messages.append(
            f"High TFHE precision ({config.tfhe_precision_bits} bits = {lut_size} LUT entries) "
            f"may impact memory and latency."
        )

    # Warn about activation range
    if config.activation == NonLinearActivation.RELU:
        if config.activation_range[0] > 0:
            messages.append(
                f"ReLU activation with positive minimum ({config.activation_range[0]}) "
                f"will never produce zeros. Consider range starting at negative value."
            )

    return messages


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def get_standard_linear_config(rank: int = 16, alpha: float = 32.0) -> N2HEAdapterConfig:
    """
    Get standard linear LoRA configuration (CKKS only).

    This is the default, fastest configuration using MOAI column packing.
    """
    return N2HEAdapterConfig(
        adapter_type=AdapterType.LINEAR_LORA,
        rank=rank,
        alpha=alpha,
        use_moai_packing=True,
    )


def get_gated_lora_config(
    rank: int = 16,
    alpha: float = 32.0,
    gate_activation: NonLinearActivation = NonLinearActivation.SIGMOID,
) -> N2HEAdapterConfig:
    """
    Get gated LoRA configuration with TFHE bootstrapping.

    The gate is computed using TFHE programmable bootstrapping for
    exact evaluation of the gate activation function.
    """
    return N2HEAdapterConfig(
        adapter_type=AdapterType.GATED_LORA,
        rank=rank,
        alpha=alpha,
        activation=gate_activation,
        use_tfhe_bootstrap=True,
        tfhe_precision_bits=8,
        use_moai_packing=True,
    )


def get_nonlinear_lora_config(
    rank: int = 16,
    alpha: float = 32.0,
    activation: NonLinearActivation = NonLinearActivation.GELU,
    precision_bits: int = 8,
) -> N2HEAdapterConfig:
    """
    Get non-linear LoRA configuration with TFHE bootstrapping.

    The adapter output is passed through a non-linear activation
    computed exactly via TFHE programmable bootstrapping.
    """
    return N2HEAdapterConfig(
        adapter_type=AdapterType.NONLINEAR_LORA,
        rank=rank,
        alpha=alpha,
        activation=activation,
        use_tfhe_bootstrap=True,
        tfhe_precision_bits=precision_bits,
        use_moai_packing=True,
    )
