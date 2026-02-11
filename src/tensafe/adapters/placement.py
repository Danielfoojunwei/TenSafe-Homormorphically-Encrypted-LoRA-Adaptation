"""
Architecture-Aware Adapter Placement Resolution.

This module provides automatic discovery and resolution of adapter placement
targets for all major transformer architectures. It handles:

- Automatic architecture detection from model config
- Module path resolution for attention and MLP layers
- Fused QKV projection handling (split/recombine)
- Cross-attention targeting for encoder-decoder models
- Layer-wise configuration and importance weighting

Supported Architectures:
    - LLaMA, LLaMA 2/3/4
    - Mistral, Mixtral
    - Qwen, Qwen2
    - Falcon, Falcon-40B
    - GPT-NeoX, GPT-J
    - GPT-2, GPT-BigCode
    - BLOOM
    - MPT
    - Phi, Phi-2/3
    - Gemma, Gemma-2
    - T5, mT5
    - BART, mBART
    - BERT, RoBERTa

References:
    - PEFT: https://github.com/huggingface/peft
    - S-LoRA: https://arxiv.org/abs/2311.03285

Author: TenSafe Team
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import logging
import re

logger = logging.getLogger(__name__)


class LayerType(Enum):
    """Types of layers that can be targeted."""
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"  # For encoder-decoder models
    MLP = "mlp"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"


class ProjectionType(Enum):
    """Types of projections within attention."""
    QUERY = "q"
    KEY = "k"
    VALUE = "v"
    OUTPUT = "o"
    QKV_FUSED = "qkv"      # Fused Q, K, V in single projection
    GATE_UP_FUSED = "gate_up"  # Fused gate and up projections in MLP


class TargetScope(Enum):
    """Scope of adapter targeting."""
    ATTENTION_ONLY = "attention_only"      # Q, K, V, O only
    ATTENTION_QKV = "attention_qkv"        # Q, K, V only (no O)
    ATTENTION_QV = "attention_qv"          # Q, V only (PEFT default)
    MLP_ONLY = "mlp_only"                  # MLP projections only
    ALL_LINEAR = "all_linear"              # All linear layers
    CUSTOM = "custom"                      # Custom target list


@dataclass
class ProjectionTarget:
    """
    Represents a single projection target for adapter placement.

    Attributes:
        layer_idx: Layer index in the model
        layer_type: Type of layer (attention, MLP, etc.)
        projection_type: Type of projection (Q, K, V, O, or fused)
        module_path: Full dotted path to the module
        module_name: Short name of the module (e.g., "q_proj")
        in_features: Input dimension
        out_features: Output dimension
        is_fused: Whether this is a fused projection
        fused_components: For fused projections, which components are included
    """
    layer_idx: int
    layer_type: LayerType
    projection_type: ProjectionType
    module_path: str
    module_name: str
    in_features: int = 0
    out_features: int = 0
    is_fused: bool = False
    fused_components: List[ProjectionType] = field(default_factory=list)

    def __hash__(self):
        return hash((self.layer_idx, self.module_path))

    def __eq__(self, other):
        if not isinstance(other, ProjectionTarget):
            return False
        return self.layer_idx == other.layer_idx and self.module_path == other.module_path


@dataclass
class ArchitectureConfig:
    """
    Configuration for a specific model architecture.

    Defines the module naming patterns and structure for an architecture.
    """
    # Architecture identifier
    name: str
    architectures: List[str]  # HuggingFace architecture names

    # Layer path patterns
    encoder_layer_path: Optional[str] = None  # e.g., "encoder.block"
    decoder_layer_path: Optional[str] = None  # e.g., "decoder.block"
    layer_path: str = "model.layers"          # Default for decoder-only

    # Attention module patterns
    attention_module: str = "self_attn"
    cross_attention_module: Optional[str] = None  # For encoder-decoder

    # Projection naming
    q_proj: str = "q_proj"
    k_proj: str = "k_proj"
    v_proj: str = "v_proj"
    o_proj: str = "o_proj"

    # Fused projection (if applicable)
    qkv_fused: Optional[str] = None  # e.g., "query_key_value" for Falcon

    # MLP module patterns
    mlp_module: str = "mlp"
    gate_proj: Optional[str] = "gate_proj"
    up_proj: Optional[str] = "up_proj"
    down_proj: Optional[str] = "down_proj"

    # Alternative MLP naming (GPT-2 style)
    fc1: Optional[str] = None  # First MLP projection
    fc2: Optional[str] = None  # Second MLP projection

    # Fused MLP projection
    gate_up_fused: Optional[str] = None  # e.g., "gate_up_proj"

    # Additional metadata
    has_bias: bool = False
    supports_kv_sharing: bool = False  # GQA/MQA
    num_kv_heads_attr: Optional[str] = None  # Attribute for num_kv_heads

    def get_qkv_modules(self) -> List[str]:
        """Get list of Q, K, V module names."""
        if self.qkv_fused:
            return [self.qkv_fused]
        return [self.q_proj, self.k_proj, self.v_proj]

    def get_attention_modules(self) -> List[str]:
        """Get all attention projection module names."""
        modules = self.get_qkv_modules()
        if self.o_proj:
            modules.append(self.o_proj)
        return modules

    def get_mlp_modules(self) -> List[str]:
        """Get MLP projection module names."""
        if self.gate_up_fused:
            return [self.gate_up_fused, self.down_proj]
        if self.gate_proj and self.up_proj:
            return [self.gate_proj, self.up_proj, self.down_proj]
        if self.fc1 and self.fc2:
            return [self.fc1, self.fc2]
        return []


# =============================================================================
# ARCHITECTURE REGISTRY
# =============================================================================

ARCHITECTURE_CONFIGS: Dict[str, ArchitectureConfig] = {
    # LLaMA family
    "llama": ArchitectureConfig(
        name="llama",
        architectures=["LlamaForCausalLM", "LlamaModel"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="mlp",
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
        supports_kv_sharing=True,
        num_kv_heads_attr="num_key_value_heads",
    ),

    # Mistral family
    "mistral": ArchitectureConfig(
        name="mistral",
        architectures=["MistralForCausalLM", "MistralModel"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="mlp",
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
        supports_kv_sharing=True,
        num_kv_heads_attr="num_key_value_heads",
    ),

    # Mixtral (MoE)
    "mixtral": ArchitectureConfig(
        name="mixtral",
        architectures=["MixtralForCausalLM", "MixtralModel"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="block_sparse_moe",  # Special for MoE
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
        supports_kv_sharing=True,
    ),

    # Qwen family
    "qwen": ArchitectureConfig(
        name="qwen",
        architectures=["QWenLMHeadModel", "Qwen2ForCausalLM", "Qwen2MoeForCausalLM"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="mlp",
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
        supports_kv_sharing=True,
    ),

    # Falcon family (FUSED QKV)
    "falcon": ArchitectureConfig(
        name="falcon",
        architectures=["FalconForCausalLM", "FalconModel", "RWForCausalLM"],
        layer_path="transformer.h",
        attention_module="self_attention",
        qkv_fused="query_key_value",  # Fused QKV
        q_proj="query_key_value", k_proj="query_key_value", v_proj="query_key_value",
        o_proj="dense",
        mlp_module="mlp",
        fc1="dense_h_to_4h", fc2="dense_4h_to_h",
        supports_kv_sharing=True,
    ),

    # GPT-NeoX (FUSED QKV)
    "gpt_neox": ArchitectureConfig(
        name="gpt_neox",
        architectures=["GPTNeoXForCausalLM", "GPTNeoXModel"],
        layer_path="gpt_neox.layers",
        attention_module="attention",
        qkv_fused="query_key_value",
        q_proj="query_key_value", k_proj="query_key_value", v_proj="query_key_value",
        o_proj="dense",
        mlp_module="mlp",
        fc1="dense_h_to_4h", fc2="dense_4h_to_h",
    ),

    # GPT-J
    "gpt_j": ArchitectureConfig(
        name="gpt_j",
        architectures=["GPTJForCausalLM", "GPTJModel"],
        layer_path="transformer.h",
        attention_module="attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="out_proj",
        mlp_module="mlp",
        fc1="fc_in", fc2="fc_out",
    ),

    # GPT-2 (FUSED QKV with special structure)
    "gpt2": ArchitectureConfig(
        name="gpt2",
        architectures=["GPT2LMHeadModel", "GPT2Model"],
        layer_path="transformer.h",
        attention_module="attn",
        qkv_fused="c_attn",  # Conv1D, fused QKV
        q_proj="c_attn", k_proj="c_attn", v_proj="c_attn",
        o_proj="c_proj",
        mlp_module="mlp",
        fc1="c_fc", fc2="c_proj",
    ),

    # BLOOM (FUSED QKV)
    "bloom": ArchitectureConfig(
        name="bloom",
        architectures=["BloomForCausalLM", "BloomModel"],
        layer_path="transformer.h",
        attention_module="self_attention",
        qkv_fused="query_key_value",
        q_proj="query_key_value", k_proj="query_key_value", v_proj="query_key_value",
        o_proj="dense",
        mlp_module="mlp",
        fc1="dense_h_to_4h", fc2="dense_4h_to_h",
    ),

    # MPT (FUSED QKV)
    "mpt": ArchitectureConfig(
        name="mpt",
        architectures=["MptForCausalLM", "MptModel"],
        layer_path="transformer.blocks",
        attention_module="attn",
        qkv_fused="Wqkv",
        q_proj="Wqkv", k_proj="Wqkv", v_proj="Wqkv",
        o_proj="out_proj",
        mlp_module="ffn",
        fc1="up_proj", fc2="down_proj",
    ),

    # Phi family
    "phi": ArchitectureConfig(
        name="phi",
        architectures=["PhiForCausalLM", "Phi3ForCausalLM", "PhiModel"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="dense",
        mlp_module="mlp",
        fc1="fc1", fc2="fc2",
    ),

    # Gemma family
    "gemma": ArchitectureConfig(
        name="gemma",
        architectures=["GemmaForCausalLM", "Gemma2ForCausalLM", "GemmaModel"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="mlp",
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
        supports_kv_sharing=True,
    ),

    # OPT
    "opt": ArchitectureConfig(
        name="opt",
        architectures=["OPTForCausalLM", "OPTModel"],
        layer_path="model.decoder.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="out_proj",
        mlp_module="",
        fc1="fc1", fc2="fc2",
    ),

    # T5 (Encoder-Decoder)
    "t5": ArchitectureConfig(
        name="t5",
        architectures=["T5ForConditionalGeneration", "T5Model", "MT5ForConditionalGeneration"],
        encoder_layer_path="encoder.block",
        decoder_layer_path="decoder.block",
        layer_path="encoder.block",  # Default to encoder
        attention_module="layer.0.SelfAttention",
        cross_attention_module="layer.1.EncDecAttention",
        q_proj="q", k_proj="k", v_proj="v", o_proj="o",
        mlp_module="layer.1.DenseReluDense" if True else "layer.2.DenseReluDense",
        fc1="wi", fc2="wo",
    ),

    # BART (Encoder-Decoder)
    "bart": ArchitectureConfig(
        name="bart",
        architectures=["BartForConditionalGeneration", "BartModel", "MBartForConditionalGeneration"],
        encoder_layer_path="model.encoder.layers",
        decoder_layer_path="model.decoder.layers",
        layer_path="model.encoder.layers",
        attention_module="self_attn",
        cross_attention_module="encoder_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="out_proj",
        mlp_module="",
        fc1="fc1", fc2="fc2",
    ),

    # BERT family
    "bert": ArchitectureConfig(
        name="bert",
        architectures=["BertForMaskedLM", "BertModel", "RobertaForMaskedLM", "RobertaModel"],
        layer_path="encoder.layer",
        attention_module="attention.self",
        q_proj="query", k_proj="key", v_proj="value",
        o_proj="attention.output.dense",
        mlp_module="intermediate",
        fc1="dense", fc2="output.dense",
    ),

    # StableLM
    "stablelm": ArchitectureConfig(
        name="stablelm",
        architectures=["StableLmForCausalLM", "StableLMEpochForCausalLM"],
        layer_path="model.layers",
        attention_module="self_attn",
        q_proj="q_proj", k_proj="k_proj", v_proj="v_proj", o_proj="o_proj",
        mlp_module="mlp",
        gate_proj="gate_proj", up_proj="up_proj", down_proj="down_proj",
    ),

    # ChatGLM (FUSED QKV)
    "chatglm": ArchitectureConfig(
        name="chatglm",
        architectures=["ChatGLMForConditionalGeneration", "ChatGLMModel"],
        layer_path="transformer.encoder.layers",
        attention_module="self_attention",
        qkv_fused="query_key_value",
        q_proj="query_key_value", k_proj="query_key_value", v_proj="query_key_value",
        o_proj="dense",
        mlp_module="mlp",
        fc1="dense_h_to_4h", fc2="dense_4h_to_h",
    ),

    # CodeGen (FUSED QKV)
    "codegen": ArchitectureConfig(
        name="codegen",
        architectures=["CodeGenForCausalLM", "CodeGenModel"],
        layer_path="transformer.h",
        attention_module="attn",
        qkv_fused="qkv_proj",
        q_proj="qkv_proj", k_proj="qkv_proj", v_proj="qkv_proj",
        o_proj="out_proj",
        mlp_module="mlp",
        fc1="fc_in", fc2="fc_out",
    ),
}


def get_architecture_config(model_or_config: Any) -> ArchitectureConfig:
    """
    Get architecture configuration for a model.

    Args:
        model_or_config: HuggingFace model, config, or architecture string

    Returns:
        ArchitectureConfig for the model
    """
    # Extract architecture name
    arch_name = None

    if isinstance(model_or_config, str):
        arch_name = model_or_config
    elif hasattr(model_or_config, 'config'):
        # It's a model
        config = model_or_config.config
        if hasattr(config, 'architectures') and config.architectures:
            arch_name = config.architectures[0]
        elif hasattr(config, 'model_type'):
            arch_name = config.model_type
    elif hasattr(model_or_config, 'architectures'):
        # It's a config
        if model_or_config.architectures:
            arch_name = model_or_config.architectures[0]
        elif hasattr(model_or_config, 'model_type'):
            arch_name = model_or_config.model_type

    if arch_name is None:
        raise ValueError("Could not determine model architecture")

    # Search for matching config
    for config_key, config in ARCHITECTURE_CONFIGS.items():
        if arch_name in config.architectures:
            return config
        if arch_name.lower() == config_key.lower():
            return config

    # Try partial matching
    arch_lower = arch_name.lower()
    for config_key, config in ARCHITECTURE_CONFIGS.items():
        if config_key in arch_lower:
            logger.warning(f"Using partial match: {config_key} for {arch_name}")
            return config

    # Fallback to LLaMA-style (most common)
    logger.warning(f"Unknown architecture {arch_name}, using LLaMA defaults")
    return ARCHITECTURE_CONFIGS["llama"]


# =============================================================================
# ADAPTER PLACEMENT RESOLVER
# =============================================================================

@dataclass
class PlacementConfig:
    """Configuration for adapter placement resolution."""
    # Target scope
    scope: TargetScope = TargetScope.ATTENTION_QV

    # Layer selection
    layer_indices: Optional[List[int]] = None  # None = all layers
    skip_layers: List[int] = field(default_factory=list)

    # Component selection
    include_self_attention: bool = True
    include_cross_attention: bool = False  # For encoder-decoder
    include_mlp: bool = False

    # Projection selection (for custom scope)
    target_projections: List[str] = field(default_factory=list)

    # Encoder-decoder specific
    include_encoder: bool = True
    include_decoder: bool = True

    # Layer-wise importance weights (for adaptive rank allocation)
    layer_importance: Dict[int, float] = field(default_factory=dict)


class AdapterPlacementResolver:
    """
    Resolves adapter placement targets for a model.

    This class handles automatic discovery of target modules, including:
    - Architecture-specific module naming
    - Fused QKV projection handling
    - Layer filtering and selection
    - Cross-attention targeting for encoder-decoder models
    """

    def __init__(
        self,
        model: Any,
        placement_config: Optional[PlacementConfig] = None,
    ):
        """
        Initialize the placement resolver.

        Args:
            model: HuggingFace model or model config
            placement_config: Configuration for target selection
        """
        self.model = model
        self.config = placement_config or PlacementConfig()
        self.arch_config = get_architecture_config(model)

        # Cache resolved targets
        self._targets: Optional[List[ProjectionTarget]] = None

        # Model metadata
        self._num_layers: Optional[int] = None
        self._hidden_size: Optional[int] = None
        self._num_heads: Optional[int] = None
        self._head_dim: Optional[int] = None

        # Extract model config
        self._extract_model_config()

    def _extract_model_config(self) -> None:
        """Extract configuration from model."""
        config = getattr(self.model, 'config', self.model)

        self._hidden_size = getattr(config, 'hidden_size', None)
        self._num_heads = getattr(config, 'num_attention_heads', None)

        if self._hidden_size and self._num_heads:
            self._head_dim = self._hidden_size // self._num_heads

        # Count layers
        self._num_layers = getattr(config, 'num_hidden_layers', None)
        if self._num_layers is None:
            self._num_layers = getattr(config, 'n_layer', None)
        if self._num_layers is None:
            self._num_layers = getattr(config, 'num_layers', None)

    @property
    def num_layers(self) -> int:
        """Get number of layers."""
        return self._num_layers or 0

    @property
    def hidden_size(self) -> int:
        """Get hidden size."""
        return self._hidden_size or 0

    def resolve(self) -> List[ProjectionTarget]:
        """
        Resolve all adapter placement targets.

        Returns:
            List of ProjectionTarget objects
        """
        if self._targets is not None:
            return self._targets

        targets = []

        # Determine which layers to target
        layer_indices = self.config.layer_indices
        if layer_indices is None:
            layer_indices = list(range(self.num_layers))

        # Remove skipped layers
        layer_indices = [i for i in layer_indices if i not in self.config.skip_layers]

        # Resolve targets based on scope
        if self.config.scope == TargetScope.ALL_LINEAR:
            targets = self._resolve_all_linear(layer_indices)
        elif self.config.scope == TargetScope.MLP_ONLY:
            targets = self._resolve_mlp_only(layer_indices)
        elif self.config.scope == TargetScope.CUSTOM:
            targets = self._resolve_custom(layer_indices)
        else:
            # Attention-based scopes
            targets = self._resolve_attention(layer_indices)

        self._targets = targets
        return targets

    def _resolve_attention(self, layer_indices: List[int]) -> List[ProjectionTarget]:
        """Resolve attention projection targets."""
        targets = []

        # Determine which projections to include based on scope
        if self.config.scope == TargetScope.ATTENTION_QV:
            proj_types = [ProjectionType.QUERY, ProjectionType.VALUE]
        elif self.config.scope == TargetScope.ATTENTION_QKV:
            proj_types = [ProjectionType.QUERY, ProjectionType.KEY, ProjectionType.VALUE]
        elif self.config.scope == TargetScope.ATTENTION_ONLY:
            proj_types = [ProjectionType.QUERY, ProjectionType.KEY,
                         ProjectionType.VALUE, ProjectionType.OUTPUT]
        else:
            proj_types = [ProjectionType.QUERY, ProjectionType.VALUE]

        # Check for fused QKV
        has_fused_qkv = self.arch_config.qkv_fused is not None

        for layer_idx in layer_indices:
            layer_path = f"{self.arch_config.layer_path}.{layer_idx}"

            # Self-attention
            if self.config.include_self_attention:
                attn_path = f"{layer_path}.{self.arch_config.attention_module}"

                if has_fused_qkv:
                    # Add fused QKV target
                    qkv_needed = any(pt in proj_types for pt in [
                        ProjectionType.QUERY, ProjectionType.KEY, ProjectionType.VALUE
                    ])
                    if qkv_needed:
                        targets.append(ProjectionTarget(
                            layer_idx=layer_idx,
                            layer_type=LayerType.SELF_ATTENTION,
                            projection_type=ProjectionType.QKV_FUSED,
                            module_path=f"{attn_path}.{self.arch_config.qkv_fused}",
                            module_name=self.arch_config.qkv_fused,
                            is_fused=True,
                            fused_components=[ProjectionType.QUERY, ProjectionType.KEY, ProjectionType.VALUE],
                            in_features=self._hidden_size or 0,
                            out_features=(self._hidden_size or 0) * 3,
                        ))
                else:
                    # Add separate Q, K, V targets
                    proj_map = {
                        ProjectionType.QUERY: self.arch_config.q_proj,
                        ProjectionType.KEY: self.arch_config.k_proj,
                        ProjectionType.VALUE: self.arch_config.v_proj,
                    }

                    for proj_type in [ProjectionType.QUERY, ProjectionType.KEY, ProjectionType.VALUE]:
                        if proj_type in proj_types:
                            proj_name = proj_map[proj_type]
                            targets.append(ProjectionTarget(
                                layer_idx=layer_idx,
                                layer_type=LayerType.SELF_ATTENTION,
                                projection_type=proj_type,
                                module_path=f"{attn_path}.{proj_name}",
                                module_name=proj_name,
                                in_features=self._hidden_size or 0,
                                out_features=self._hidden_size or 0,
                            ))

                # Output projection
                if ProjectionType.OUTPUT in proj_types and self.arch_config.o_proj:
                    targets.append(ProjectionTarget(
                        layer_idx=layer_idx,
                        layer_type=LayerType.SELF_ATTENTION,
                        projection_type=ProjectionType.OUTPUT,
                        module_path=f"{attn_path}.{self.arch_config.o_proj}",
                        module_name=self.arch_config.o_proj,
                        in_features=self._hidden_size or 0,
                        out_features=self._hidden_size or 0,
                    ))

            # Cross-attention (for encoder-decoder models)
            if self.config.include_cross_attention and self.arch_config.cross_attention_module:
                cross_attn_path = f"{layer_path}.{self.arch_config.cross_attention_module}"

                # Cross-attention typically has separate Q, K, V
                for proj_type in proj_types:
                    if proj_type == ProjectionType.OUTPUT:
                        continue  # Handle O separately
                    proj_map = {
                        ProjectionType.QUERY: self.arch_config.q_proj,
                        ProjectionType.KEY: self.arch_config.k_proj,
                        ProjectionType.VALUE: self.arch_config.v_proj,
                    }
                    if proj_type in proj_map:
                        targets.append(ProjectionTarget(
                            layer_idx=layer_idx,
                            layer_type=LayerType.CROSS_ATTENTION,
                            projection_type=proj_type,
                            module_path=f"{cross_attn_path}.{proj_map[proj_type]}",
                            module_name=proj_map[proj_type],
                            in_features=self._hidden_size or 0,
                            out_features=self._hidden_size or 0,
                        ))

        # Add MLP if configured
        if self.config.include_mlp:
            mlp_targets = self._resolve_mlp_only(layer_indices)
            targets.extend(mlp_targets)

        return targets

    def _resolve_mlp_only(self, layer_indices: List[int]) -> List[ProjectionTarget]:
        """Resolve MLP projection targets."""
        targets = []

        for layer_idx in layer_indices:
            layer_path = f"{self.arch_config.layer_path}.{layer_idx}"
            mlp_path = f"{layer_path}.{self.arch_config.mlp_module}" if self.arch_config.mlp_module else layer_path

            # Check for fused gate_up
            if self.arch_config.gate_up_fused:
                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.MLP,
                    projection_type=ProjectionType.GATE_UP_FUSED,
                    module_path=f"{mlp_path}.{self.arch_config.gate_up_fused}",
                    module_name=self.arch_config.gate_up_fused,
                    is_fused=True,
                    in_features=self._hidden_size or 0,
                    out_features=(self._hidden_size or 0) * 8,  # Approximate
                ))
            elif self.arch_config.gate_proj:
                # SwiGLU-style MLP
                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.MLP,
                    projection_type=ProjectionType.QUERY,  # Reuse for gate
                    module_path=f"{mlp_path}.{self.arch_config.gate_proj}",
                    module_name=self.arch_config.gate_proj,
                    in_features=self._hidden_size or 0,
                    out_features=(self._hidden_size or 0) * 4,
                ))
                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.MLP,
                    projection_type=ProjectionType.KEY,  # Reuse for up
                    module_path=f"{mlp_path}.{self.arch_config.up_proj}",
                    module_name=self.arch_config.up_proj,
                    in_features=self._hidden_size or 0,
                    out_features=(self._hidden_size or 0) * 4,
                ))
            elif self.arch_config.fc1:
                # Standard FFN
                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.MLP,
                    projection_type=ProjectionType.QUERY,
                    module_path=f"{mlp_path}.{self.arch_config.fc1}",
                    module_name=self.arch_config.fc1,
                    in_features=self._hidden_size or 0,
                    out_features=(self._hidden_size or 0) * 4,
                ))

            # Down projection (always present)
            down_name = self.arch_config.down_proj or self.arch_config.fc2
            if down_name:
                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.MLP,
                    projection_type=ProjectionType.VALUE,  # Reuse for down
                    module_path=f"{mlp_path}.{down_name}",
                    module_name=down_name,
                    in_features=(self._hidden_size or 0) * 4,
                    out_features=self._hidden_size or 0,
                ))

        return targets

    def _resolve_all_linear(self, layer_indices: List[int]) -> List[ProjectionTarget]:
        """Resolve all linear layer targets."""
        targets = []

        # Attention (all projections)
        self.config.scope = TargetScope.ATTENTION_ONLY
        targets.extend(self._resolve_attention(layer_indices))

        # MLP
        targets.extend(self._resolve_mlp_only(layer_indices))

        # Reset scope
        self.config.scope = TargetScope.ALL_LINEAR

        return targets

    def _resolve_custom(self, layer_indices: List[int]) -> List[ProjectionTarget]:
        """Resolve custom target list."""
        targets = []

        for layer_idx in layer_indices:
            layer_path = f"{self.arch_config.layer_path}.{layer_idx}"

            for target_name in self.config.target_projections:
                # Determine projection type from name
                proj_type = ProjectionType.QUERY  # Default
                if 'k' in target_name.lower() or 'key' in target_name.lower():
                    proj_type = ProjectionType.KEY
                elif 'v' in target_name.lower() or 'value' in target_name.lower():
                    proj_type = ProjectionType.VALUE
                elif 'o' in target_name.lower() or 'out' in target_name.lower():
                    proj_type = ProjectionType.OUTPUT

                # Build module path
                if '.' in target_name:
                    module_path = f"{layer_path}.{target_name}"
                else:
                    module_path = f"{layer_path}.{self.arch_config.attention_module}.{target_name}"

                targets.append(ProjectionTarget(
                    layer_idx=layer_idx,
                    layer_type=LayerType.SELF_ATTENTION,
                    projection_type=proj_type,
                    module_path=module_path,
                    module_name=target_name,
                ))

        return targets

    def get_target_modules(self) -> List[str]:
        """Get list of target module names (for PEFT compatibility)."""
        targets = self.resolve()
        return list(set(t.module_name for t in targets))

    def get_targets_by_layer(self) -> Dict[int, List[ProjectionTarget]]:
        """Get targets organized by layer index."""
        targets = self.resolve()
        by_layer: Dict[int, List[ProjectionTarget]] = {}

        for target in targets:
            if target.layer_idx not in by_layer:
                by_layer[target.layer_idx] = []
            by_layer[target.layer_idx].append(target)

        return by_layer

    def get_targets_for_layer(self, layer_idx: int) -> List[ProjectionTarget]:
        """Get targets for a specific layer."""
        by_layer = self.get_targets_by_layer()
        return by_layer.get(layer_idx, [])

    def get_fused_targets(self) -> List[ProjectionTarget]:
        """Get only fused projection targets."""
        return [t for t in self.resolve() if t.is_fused]

    def has_fused_projections(self) -> bool:
        """Check if model has fused projections."""
        return self.arch_config.qkv_fused is not None or self.arch_config.gate_up_fused is not None

    def get_module(self, target: ProjectionTarget) -> Any:
        """
        Get the actual module for a target.

        Args:
            target: ProjectionTarget to get module for

        Returns:
            The module object
        """
        module = self.model
        for part in target.module_path.split('.'):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def validate_targets(self) -> List[str]:
        """
        Validate that all targets exist in the model.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        targets = self.resolve()

        for target in targets:
            try:
                self.get_module(target)
            except (AttributeError, IndexError, KeyError) as e:
                errors.append(f"Target {target.module_path} not found: {e}")

        return errors


# =============================================================================
# FUSED PROJECTION HANDLER
# =============================================================================

class FusedProjectionHandler:
    """
    Handles fused QKV projections for adapter application.

    Provides utilities to:
    - Split fused projections into Q, K, V components
    - Recombine after individual adapter application
    - Handle different fusion patterns (stacked vs interleaved)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        fusion_pattern: str = "stacked",  # "stacked" or "interleaved"
    ):
        """
        Initialize the fused projection handler.

        Args:
            hidden_size: Model hidden size
            num_heads: Number of attention heads
            num_kv_heads: Number of KV heads (for GQA/MQA)
            fusion_pattern: How Q, K, V are combined
        """
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.fusion_pattern = fusion_pattern

        self.head_dim = hidden_size // num_heads
        self.q_size = hidden_size
        self.kv_size = self.num_kv_heads * self.head_dim

    def split_qkv(
        self,
        qkv_output: Any,  # torch.Tensor or np.ndarray
    ) -> Tuple[Any, Any, Any]:
        """
        Split fused QKV output into separate Q, K, V.

        Args:
            qkv_output: Fused output [batch, seq, qkv_dim]

        Returns:
            Tuple of (Q, K, V)
        """
        import numpy as np

        is_numpy = isinstance(qkv_output, np.ndarray)

        if self.fusion_pattern == "stacked":
            # Q, K, V are stacked: [batch, seq, q_size + k_size + v_size]
            if is_numpy:
                q = qkv_output[..., :self.q_size]
                k = qkv_output[..., self.q_size:self.q_size + self.kv_size]
                v = qkv_output[..., self.q_size + self.kv_size:]
            else:
                q, k, v = qkv_output.split(
                    [self.q_size, self.kv_size, self.kv_size],
                    dim=-1
                )
        else:
            # Interleaved pattern: Q_h1, K_h1, V_h1, Q_h2, K_h2, V_h2, ...
            # This is less common but used in some architectures
            raise NotImplementedError("Interleaved fusion not yet supported")

        return q, k, v

    def combine_qkv(
        self,
        q: Any,
        k: Any,
        v: Any,
    ) -> Any:
        """
        Combine separate Q, K, V into fused representation.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor

        Returns:
            Fused QKV tensor
        """
        import numpy as np

        is_numpy = isinstance(q, np.ndarray)

        if self.fusion_pattern == "stacked":
            if is_numpy:
                return np.concatenate([q, k, v], axis=-1)
            else:
                import torch
                return torch.cat([q, k, v], dim=-1)
        else:
            raise NotImplementedError("Interleaved fusion not yet supported")

    def apply_adapters_to_split(
        self,
        qkv_input: Any,
        qkv_output: Any,
        q_adapter: Optional[Callable] = None,
        k_adapter: Optional[Callable] = None,
        v_adapter: Optional[Callable] = None,
    ) -> Any:
        """
        Apply separate adapters to fused QKV projection.

        Splits the output, applies adapters to each component,
        and recombines.

        Args:
            qkv_input: Input to the fused projection
            qkv_output: Output from the fused projection
            q_adapter: Adapter function for Q (takes input, returns delta)
            k_adapter: Adapter function for K
            v_adapter: Adapter function for V

        Returns:
            Modified fused output
        """
        q, k, v = self.split_qkv(qkv_output)

        # Apply adapters
        if q_adapter is not None:
            q_delta = q_adapter(qkv_input)
            if q_delta is not None:
                q = q + q_delta

        if k_adapter is not None:
            k_delta = k_adapter(qkv_input)
            if k_delta is not None:
                k = k + k_delta

        if v_adapter is not None:
            v_delta = v_adapter(qkv_input)
            if v_delta is not None:
                v = v + v_delta

        return self.combine_qkv(q, k, v)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def auto_discover_targets(
    model: Any,
    scope: TargetScope = TargetScope.ATTENTION_QV,
    include_mlp: bool = False,
) -> List[str]:
    """
    Automatically discover target module names for a model.

    Args:
        model: HuggingFace model
        scope: Target scope
        include_mlp: Whether to include MLP layers

    Returns:
        List of target module names
    """
    config = PlacementConfig(
        scope=scope,
        include_mlp=include_mlp,
    )
    resolver = AdapterPlacementResolver(model, config)
    return resolver.get_target_modules()


def get_layer_importance_weights(
    num_layers: int,
    strategy: str = "uniform",
    top_heavy_ratio: float = 2.0,
) -> Dict[int, float]:
    """
    Generate layer importance weights for adaptive rank allocation.

    Args:
        num_layers: Total number of layers
        strategy: "uniform", "top_heavy", "bottom_heavy", "middle_heavy"
        top_heavy_ratio: Ratio between highest and lowest importance

    Returns:
        Dict mapping layer index to importance weight
    """
    weights = {}

    if strategy == "uniform":
        for i in range(num_layers):
            weights[i] = 1.0

    elif strategy == "top_heavy":
        # Higher layers get more importance
        for i in range(num_layers):
            ratio = i / (num_layers - 1) if num_layers > 1 else 1.0
            weights[i] = 1.0 + (top_heavy_ratio - 1.0) * ratio

    elif strategy == "bottom_heavy":
        # Lower layers get more importance
        for i in range(num_layers):
            ratio = 1.0 - (i / (num_layers - 1)) if num_layers > 1 else 1.0
            weights[i] = 1.0 + (top_heavy_ratio - 1.0) * ratio

    elif strategy == "middle_heavy":
        # Middle layers get more importance
        middle = num_layers // 2
        for i in range(num_layers):
            dist = abs(i - middle) / middle if middle > 0 else 0
            weights[i] = top_heavy_ratio - (top_heavy_ratio - 1.0) * dist

    # Normalize so mean is 1.0
    mean_weight = sum(weights.values()) / len(weights)
    for i in weights:
        weights[i] /= mean_weight

    return weights


__all__ = [
    "LayerType",
    "ProjectionType",
    "TargetScope",
    "ProjectionTarget",
    "ArchitectureConfig",
    "PlacementConfig",
    "AdapterPlacementResolver",
    "FusedProjectionHandler",
    "ARCHITECTURE_CONFIGS",
    "get_architecture_config",
    "auto_discover_targets",
    "get_layer_importance_weights",
]
