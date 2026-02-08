"""Configuration for TenSafe vLLM Backend."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class HESchemeType(str, Enum):
    """Supported HE schemes for LoRA computation."""
    CKKS = "ckks"
    TFHE = "tfhe"
    HYBRID = "hybrid"  # CKKS for linear, TFHE for gating


class CKKSProfile(str, Enum):
    """CKKS security/performance profiles."""
    FAST = "fast"      # Lower security, faster (development)
    SAFE = "safe"      # Full security (production)
    TURBO = "turbo"    # Maximum performance, minimum security (benchmarking only)


@dataclass
class TenSafeVLLMConfig:
    """Configuration for TenSafe vLLM engine.

    This configuration controls how TenSafe integrates with vLLM for
    high-throughput inference with privacy-preserving HE-LoRA.

    Attributes:
        model_path: Path to the base model (HuggingFace format or local)
        tssp_package_path: Path to TSSP package containing encrypted LoRA adapters

        # vLLM Configuration
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of GPUs for pipeline parallelism
        max_model_len: Maximum sequence length
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)

        # HE-LoRA Configuration
        enable_he_lora: Whether to enable HE-LoRA injection
        he_scheme: HE scheme type (ckks, tfhe, hybrid)
        ckks_profile: CKKS security profile (fast, safe, turbo)
        he_batch_size: Batch size for HE operations

        # Privacy Configuration
        enable_audit_logging: Log all inference operations
        enable_privacy_tracking: Track privacy budget consumption

        # Performance Configuration
        enable_prefix_caching: Cache common prefixes
        enable_chunked_prefill: Process prefill in chunks
        max_num_batched_tokens: Maximum tokens per batch
        max_num_seqs: Maximum concurrent sequences

        # API Configuration
        api_key_required: Require API key for requests
        max_tokens_per_request: Maximum tokens per request
    """

    # Model configuration
    model_path: str
    tssp_package_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

    # vLLM configuration
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"

    # HE-LoRA configuration
    enable_he_lora: bool = True
    he_scheme: HESchemeType = HESchemeType.CKKS
    ckks_profile: CKKSProfile = CKKSProfile.FAST
    he_batch_size: int = 8
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Privacy configuration
    enable_audit_logging: bool = True
    enable_privacy_tracking: bool = True
    audit_log_path: Optional[str] = None

    # Performance configuration
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256

    # API configuration
    api_key_required: bool = True
    max_tokens_per_request: int = 4096

    # Speculative decoding
    enable_speculative_decoding: bool = False
    speculative_model_path: Optional[str] = None
    num_speculative_tokens: int = 5

    # Quantization
    quantization: Optional[str] = None  # awq, gptq, squeezellm

    # Additional vLLM arguments
    extra_vllm_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")

        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")

        if self.enable_he_lora and not self.tssp_package_path:
            # HE-LoRA requires a TSSP package
            import warnings
            warnings.warn(
                "HE-LoRA is enabled but no TSSP package provided. "
                "HE-LoRA will be disabled."
            )
            self.enable_he_lora = False

        if isinstance(self.he_scheme, str):
            self.he_scheme = HESchemeType(self.he_scheme)

        if isinstance(self.ckks_profile, str):
            self.ckks_profile = CKKSProfile(self.ckks_profile)

    def to_vllm_args(self) -> Dict[str, Any]:
        """Convert to vLLM EngineArgs format."""
        args = {
            "model": self.model_path,
            "tokenizer": self.tokenizer_path or self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
        }

        if self.max_model_len:
            args["max_model_len"] = self.max_model_len

        if self.max_num_batched_tokens:
            args["max_num_batched_tokens"] = self.max_num_batched_tokens

        if self.quantization:
            args["quantization"] = self.quantization

        if self.enable_speculative_decoding and self.speculative_model_path:
            args["speculative_model"] = self.speculative_model_path
            args["num_speculative_tokens"] = self.num_speculative_tokens

        # Merge extra args
        args.update(self.extra_vllm_args)

        return args


@dataclass
class TenSafeLoRAXConfig:
    """Configuration for LoRAX multi-adapter serving.

    LoRAX enables serving 100+ LoRA adapters on a single GPU through
    dynamic adapter loading and tiered weight caching.
    """

    base_model_path: str
    adapter_registry_path: str

    # Multi-adapter settings
    max_adapters_per_gpu: int = 100
    adapter_cache_size_gb: float = 4.0
    enable_adapter_hot_swap: bool = True

    # TSSP integration
    require_tssp_verification: bool = True
    allowed_adapter_sources: List[str] = field(default_factory=list)

    # Performance
    enable_turbo_lora: bool = True  # Speculative decoding for LoRA

    def __post_init__(self):
        if self.max_adapters_per_gpu < 1:
            raise ValueError("max_adapters_per_gpu must be >= 1")
