"""
TenSafe Unified Configuration System.

This module provides a unified configuration system that:
- Combines all configuration aspects (model, LoRA, training, DP, HE, inference, RLVR)
- Supports YAML/JSON loading and environment variable overrides
- Provides validation and type safety via Pydantic
- Enables configuration composition for different training modes

Configuration Hierarchy:
    TenSafeConfig (root)
    ├── model: ModelConfig
    ├── lora: LoRAConfig
    ├── training: TrainingConfig
    ├── dp: DPConfig
    ├── he: HEConfig
    ├── inference: InferenceConfig
    └── rlvr: RLVRConfig (optional, for RLVR mode)

Usage:
    # Load from YAML
    config = load_config("config.yaml")

    # Create programmatically
    config = TenSafeConfig(
        model=ModelConfig(name="meta-llama/Llama-3.1-8B"),
        training=TrainingConfig(mode="sft", total_steps=1000),
    )

    # Environment variable overrides
    # TENSAFE_TRAINING__TOTAL_STEPS=2000 python train.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import yaml

logger = logging.getLogger(__name__)


class TrainingMode(str, Enum):
    """Supported training modes."""
    SFT = "sft"  # Supervised Fine-Tuning
    RLVR = "rlvr"  # Reinforcement Learning with Verifiable Rewards
    RLHF = "rlhf"  # RLHF (future)
    DPO = "dpo"  # Direct Preference Optimization (future)


class HEMode(str, Enum):
    """
    Homomorphic encryption modes.

    The HE system uses a unified microkernel architecture with MOAI optimizations
    (rotation-free column packing for CKKS). All production HE goes through this
    single pipeline.

    Modes:
        DISABLED: No homomorphic encryption
        PRODUCTION: Full HE with GPU acceleration (recommended for deployment)
        SIMULATION: Microkernel simulation mode (for testing, NOT cryptographically secure)

    Legacy modes (deprecated, mapped to new modes):
        TOY -> SIMULATION (with deprecation warning)
        N2HE -> PRODUCTION (N2HE path removed, uses microkernel)
        N2HE_HEXL -> PRODUCTION (HEXL now integrated into microkernel)
    """
    DISABLED = "disabled"  # No HE
    PRODUCTION = "production"  # Microkernel with GPU acceleration (recommended)
    SIMULATION = "simulation"  # Microkernel simulation mode (testing only, NOT SECURE)

    # Legacy modes - kept for backward compatibility but deprecated
    TOY = "toy"  # DEPRECATED: Maps to SIMULATION
    N2HE = "n2he"  # DEPRECATED: Maps to PRODUCTION
    N2HE_HEXL = "n2he_hexl"  # DEPRECATED: Maps to PRODUCTION

    @classmethod
    def resolve(cls, mode: "HEMode") -> "HEMode":
        """
        Resolve legacy modes to their modern equivalents.

        Args:
            mode: Any HEMode value

        Returns:
            Resolved mode (DISABLED, PRODUCTION, or SIMULATION)
        """
        import logging
        logger = logging.getLogger(__name__)

        if mode == cls.TOY:
            logger.warning(
                "HEMode.TOY is deprecated. Use HEMode.SIMULATION instead. "
                "Note: SIMULATION mode is NOT cryptographically secure."
            )
            return cls.SIMULATION
        elif mode == cls.N2HE:
            logger.warning(
                "HEMode.N2HE is deprecated. The N2HE standalone path has been removed. "
                "All HE now uses the unified microkernel with MOAI optimizations. "
                "Mapping to HEMode.PRODUCTION."
            )
            return cls.PRODUCTION
        elif mode == cls.N2HE_HEXL:
            logger.warning(
                "HEMode.N2HE_HEXL is deprecated. HEXL acceleration is now integrated "
                "into the unified microkernel. Mapping to HEMode.PRODUCTION."
            )
            return cls.PRODUCTION

        return mode

    @property
    def is_legacy(self) -> bool:
        """Check if this is a legacy/deprecated mode."""
        return self in (HEMode.TOY, HEMode.N2HE, HEMode.N2HE_HEXL)

    @property
    def is_secure(self) -> bool:
        """Check if this mode provides cryptographic security."""
        resolved = HEMode.resolve(self)
        return resolved == HEMode.PRODUCTION


class LoRATarget(str, Enum):
    """Common LoRA target modules."""
    Q_PROJ = "q_proj"
    K_PROJ = "k_proj"
    V_PROJ = "v_proj"
    O_PROJ = "o_proj"
    GATE_PROJ = "gate_proj"
    UP_PROJ = "up_proj"
    DOWN_PROJ = "down_proj"


@dataclass
class ModelConfig:
    """Model configuration."""

    # Model identification
    name: str = "meta-llama/Llama-3.1-8B"
    revision: Optional[str] = None

    # Model loading
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device_map: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Tokenizer
    tokenizer_name: Optional[str] = None  # Defaults to model name
    padding_side: str = "right"
    truncation_side: str = "right"
    max_seq_length: int = 2048

    # Trust settings
    trust_remote_code: bool = False

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.name


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    # Enable/disable
    enabled: bool = True

    # Core LoRA parameters
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05

    # Target modules
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # LoRA variant settings
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora: bool = False  # Weight-decomposed LoRA

    # Bias handling
    bias: str = "none"  # "none", "all", "lora_only"

    # Task type
    task_type: str = "CAUSAL_LM"

    @property
    def scaling(self) -> float:
        """Compute LoRA scaling factor."""
        return self.alpha / self.rank


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Training mode
    mode: TrainingMode = TrainingMode.SFT

    # Training duration
    total_steps: int = 1000
    warmup_steps: int = 100
    warmup_ratio: float = 0.1

    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 0  # Auto-computed if 0

    # Optimizer
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "cosine"  # "linear", "cosine", "constant", "constant_with_warmup"

    # Loss function (for SFT)
    loss_fn: str = "token_ce"
    loss_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500
    save_total_limit: int = 3

    # Data
    dataset_name: Optional[str] = None
    dataset_subset: Optional[str] = None
    train_split: str = "train"
    eval_split: str = "validation"
    max_samples: Optional[int] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Output
    output_dir: str = "./outputs"
    run_name: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = TrainingMode(self.mode)
        if self.effective_batch_size == 0:
            self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps


@dataclass
class DPConfig:
    """Differential Privacy configuration."""

    # Enable/disable
    enabled: bool = True

    # Privacy budget
    target_epsilon: float = 8.0
    target_delta: float = 1e-5

    # Noise mechanism
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0

    # Accounting
    accounting_mode: str = "rdp"  # "rdp", "gdp", "prv"
    accountant: str = "rdp"  # For backward compatibility

    # Sampling
    sample_rate: float = 0.01  # Batch size / dataset size

    # Validation
    strict_mode: bool = True  # Fail if privacy budget exceeded
    warn_on_budget_usage: float = 0.8  # Warn when 80% of budget used

    @property
    def is_private(self) -> bool:
        """Check if training is private."""
        return self.enabled and self.noise_multiplier > 0


@dataclass
class HEConfig:
    """Homomorphic Encryption configuration."""

    # Mode selection
    mode: HEMode = HEMode.DISABLED

    # Scheme parameters
    scheme: str = "ckks"  # "lwe", "rlwe", "ckks"

    # Security parameters
    security_level: int = 128

    # LWE/RLWE parameters
    n: int = 1024  # Lattice dimension
    q: int = 2**32  # Ciphertext modulus
    t: int = 2**16  # Plaintext modulus
    std_dev: float = 3.2

    # CKKS parameters
    poly_modulus_degree: int = 8192
    coeff_modulus_bits: List[int] = field(default_factory=lambda: [60, 40, 40, 60])
    scale_bits: int = 40

    # MOAI optimizations (for HE-LoRA)
    use_column_packing: bool = True
    use_interleaved_batching: bool = True

    # Noise management
    noise_budget_threshold: float = 5.0  # Minimum bits before refresh
    auto_rescale: bool = True

    # Performance
    use_hexl: bool = True  # Use Intel HEXL acceleration

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = HEMode(self.mode)


@dataclass
class InferenceConfig:
    """Inference configuration with TGSP enforcement support."""

    # Generation parameters
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    # Repetition control
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    # Stopping
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # LoRA mode for inference
    lora_mode: str = "plaintext"  # "none", "plaintext", "he_only", "full_he"

    # TGSP enforcement (required for HE modes)
    enforce_tgsp: bool = True

    # Batching
    batch_size: int = 1

    # Caching
    use_cache: bool = True

    def requires_tgsp(self) -> bool:
        """Check if current configuration requires TGSP format adapters."""
        if not self.enforce_tgsp:
            return False
        return self.lora_mode in ("he_only", "full_he")


@dataclass
class RLVRConfig:
    """RLVR-specific configuration."""

    # Algorithm: "reinforce", "reinforce_pp", "ppo", "grpo", "rloo"
    algorithm: str = "reinforce"

    # Pluggable advantage estimator override (optional)
    # "baseline", "grpo", "rloo", "reinforce_pp", "gae"
    advantage_estimator: Optional[str] = None

    # Pluggable policy loss override (optional)
    # "ppo_clip", "gspo", "sapo", "cispo", "clip_cov", "kl_cov",
    # "cross_entropy", "importance_sampling"
    policy_loss: Optional[str] = None

    # Rollout settings
    rollout_batch_size: int = 8
    max_new_tokens: int = 128
    num_samples_per_prompt: int = 1  # For GRPO/RLOO group statistics

    # Reward settings
    reward_fn: str = "keyword_contains"
    reward_kwargs: Dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None

    # Algorithm parameters (shared)
    gamma: float = 1.0
    use_baseline: bool = True
    baseline_decay: float = 0.99
    normalize_advantages: bool = True
    entropy_coef: float = 0.01
    kl_coef: float = 0.0

    # PPO-specific
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 4
    vf_coef: float = 0.5

    # GRPO-specific
    grpo_normalize_within_group: bool = True
    grpo_normalize_batch: bool = False
    grpo_min_group_size: int = 2

    # RLOO-specific
    rloo_fallback_to_batch_mean: bool = True

    # REINFORCE++-specific
    reinforce_pp_temporal_whitening: bool = True

    # Off-policy correction
    off_policy_enabled: bool = False
    off_policy_staleness_decay: float = 0.95

    # Async rollout
    async_rollout_enabled: bool = False
    async_max_staleness_steps: int = 5
    async_num_workers: int = 2

    # Micro-batch gradient accumulation
    micro_batch_size: int = 0  # 0 = disabled

    # Environment
    environment: Optional[str] = None
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Buffer settings
    buffer_size: int = 10000
    prioritized_replay: bool = False
    priority_alpha: float = 0.6


@dataclass
class TenSafeConfig:
    """
    Root configuration for TenSafe.

    This is the unified configuration that combines all aspects:
    - Model configuration
    - LoRA configuration
    - Training configuration
    - Differential privacy configuration
    - Homomorphic encryption configuration
    - Inference configuration
    - RLVR configuration (when training mode is RLVR)

    Example:
        config = TenSafeConfig(
            model=ModelConfig(name="meta-llama/Llama-3.1-8B"),
            lora=LoRAConfig(rank=16, alpha=32),
            training=TrainingConfig(mode="sft", total_steps=1000),
            dp=DPConfig(enabled=True, target_epsilon=8.0),
        )
    """

    # Version for config compatibility
    version: str = "1.0"

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dp: DPConfig = field(default_factory=DPConfig)
    he: HEConfig = field(default_factory=HEConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    rlvr: Optional[RLVRConfig] = None

    # API/SDK settings (for remote training)
    api_key: Optional[str] = None
    base_url: str = "https://api.tensafe.dev"
    tenant_id: Optional[str] = None

    # Runtime settings
    debug: bool = False
    verbose: bool = False
    dry_run: bool = False

    def __post_init__(self):
        # Auto-create RLVR config when mode is RLVR
        if self.training.mode == TrainingMode.RLVR and self.rlvr is None:
            self.rlvr = RLVRConfig()

        # Load API key from environment if not set
        if self.api_key is None:
            self.api_key = os.environ.get("TS_API_KEY") or os.environ.get("TENSAFE_API_KEY")

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.

        Each issue is prefixed with a severity level for structured parsing:
          "error: ..."  — configuration is invalid, training should not proceed
          "warning: ..." — configuration is risky but may be intentional

        Returns:
            List of issue messages (empty if valid)
        """
        issues = []

        # Check LoRA configuration
        if self.lora.enabled:
            if self.lora.rank > 128:
                issues.append(f"warning: LoRA rank {self.lora.rank} is very high, consider lower rank")
            if not self.lora.target_modules:
                issues.append("error: LoRA enabled but no target modules specified")

        # Check DP configuration
        if self.dp.enabled:
            if self.dp.target_epsilon < 1.0:
                issues.append(f"warning: Very tight epsilon={self.dp.target_epsilon} may severely impact utility")
            if self.dp.noise_multiplier == 0:
                issues.append("error: DP enabled but noise_multiplier is 0")

        # Check HE configuration
        if self.he.mode != HEMode.DISABLED:
            resolved_mode = HEMode.resolve(self.he.mode)
            if resolved_mode == HEMode.SIMULATION:
                issues.append(
                    "warning: HE SIMULATION mode is NOT cryptographically secure. "
                    "Use PRODUCTION mode for deployment."
                )
            if self.he.mode.is_legacy:
                issues.append(
                    f"warning: HEMode.{self.he.mode.value} is deprecated. "
                    f"Use HEMode.{resolved_mode.value} instead."
                )
            if self.he.security_level < 128:
                issues.append(f"warning: HE security level {self.he.security_level} is below recommended 128-bit")

        # Check training configuration
        if self.training.total_steps < 10:
            issues.append("warning: Very few training steps, this may be a test run")
        if self.training.learning_rate > 1e-2:
            issues.append(f"warning: Learning rate {self.training.learning_rate} is very high")

        # Check RLVR configuration
        _valid_rlvr_algorithms = ("reinforce", "reinforce_pp", "ppo", "grpo", "rloo")
        if self.training.mode == TrainingMode.RLVR:
            if self.rlvr is None:
                issues.append("error: RLVR mode but no RLVR config provided")
            elif self.rlvr.algorithm not in _valid_rlvr_algorithms:
                issues.append(f"error: Unknown RLVR algorithm: {self.rlvr.algorithm}")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        def convert(obj, seen=None):
            if seen is None:
                seen = set()

            # Prevent infinite recursion by tracking seen objects
            obj_id = id(obj)
            if obj_id in seen:
                return None

            if hasattr(obj, '__dataclass_fields__'):
                seen.add(obj_id)
                return {k: convert(v, seen) for k, v in asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [convert(v, seen) for v in obj]
            elif isinstance(obj, dict):
                return {k: convert(v, seen) for k, v in obj.items()}
            return obj

        return convert(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenSafeConfig":
        """Create from dictionary."""
        # Parse nested configs
        model = ModelConfig(**data.get("model", {})) if "model" in data else ModelConfig()
        lora = LoRAConfig(**data.get("lora", {})) if "lora" in data else LoRAConfig()

        training_data = data.get("training", {})
        training = TrainingConfig(**training_data)

        dp = DPConfig(**data.get("dp", {})) if "dp" in data else DPConfig()
        he = HEConfig(**data.get("he", {})) if "he" in data else HEConfig()
        inference = InferenceConfig(**data.get("inference", {})) if "inference" in data else InferenceConfig()

        rlvr = None
        if "rlvr" in data and data["rlvr"]:
            rlvr = RLVRConfig(**data["rlvr"])

        return cls(
            version=data.get("version", "1.0"),
            model=model,
            lora=lora,
            training=training,
            dp=dp,
            he=he,
            inference=inference,
            rlvr=rlvr,
            api_key=data.get("api_key"),
            base_url=data.get("base_url", "https://api.tensafe.dev"),
            tenant_id=data.get("tenant_id"),
            debug=data.get("debug", False),
            verbose=data.get("verbose", False),
            dry_run=data.get("dry_run", False),
        )


def load_config(
    path: Union[str, Path],
    env_override: bool = True,
    validate: bool = True,
) -> TenSafeConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file
        env_override: Allow environment variable overrides
        validate: Validate configuration after loading

    Returns:
        Loaded and optionally validated TenSafeConfig

    Environment variable format:
        TENSAFE_MODEL__NAME=... -> model.name
        TENSAFE_TRAINING__TOTAL_STEPS=... -> training.total_steps
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load file
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    # Apply environment variable overrides
    if env_override:
        data = _apply_env_overrides(data)

    # Create config
    config = TenSafeConfig.from_dict(data)

    # Validate
    if validate:
        issues = config.validate()
        for issue in issues:
            if issue.startswith("error:"):
                raise ValueError(issue)
            else:
                logger.warning(issue)

    logger.info(f"Loaded configuration from {path}")
    return config


def save_config(config: TenSafeConfig, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Args:
        config: Configuration to save
        path: Output path
    """
    path = Path(path)
    data = config.to_dict()

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        if path.suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

    logger.info(f"Saved configuration to {path}")


def _apply_env_overrides(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to config data."""
    prefix = "TENSAFE_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Parse key: TENSAFE_TRAINING__TOTAL_STEPS -> ["training", "total_steps"]
        parts = key[len(prefix):].lower().split("__")

        # Navigate and set value
        current = data
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set final value with type coercion
        final_key = parts[-1]
        current[final_key] = _coerce_value(value)

    return data


def _coerce_value(value: str) -> Any:
    """Coerce string value to appropriate type.

    Number check comes before boolean to avoid treating "0" as False
    and "1" as True when an integer is intended.
    """
    # None
    if value.lower() in ("none", "null", ""):
        return None

    # Number (checked before boolean so "0" → 0, not False)
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # Boolean (only unambiguous string forms)
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # JSON array/object
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # String
    return value


def create_default_config(
    mode: TrainingMode = TrainingMode.SFT,
    model_name: str = "meta-llama/Llama-3.1-8B",
    with_dp: bool = True,
    with_he: bool = False,
    he_simulation: bool = False,
) -> TenSafeConfig:
    """
    Create a default configuration for common use cases.

    Args:
        mode: Training mode (SFT, RLVR)
        model_name: Model to fine-tune
        with_dp: Enable differential privacy
        with_he: Enable homomorphic encryption
        he_simulation: Use HE simulation mode (for testing, not secure)

    Returns:
        Configured TenSafeConfig
    """
    # Determine HE mode
    if not with_he:
        he_mode = HEMode.DISABLED
    elif he_simulation:
        he_mode = HEMode.SIMULATION
    else:
        he_mode = HEMode.PRODUCTION

    config = TenSafeConfig(
        model=ModelConfig(name=model_name),
        lora=LoRAConfig(enabled=True, rank=16, alpha=32),
        training=TrainingConfig(
            mode=mode,
            total_steps=1000,
            learning_rate=1e-4,
        ),
        dp=DPConfig(
            enabled=with_dp,
            target_epsilon=8.0,
            noise_multiplier=1.0 if with_dp else 0.0,
        ),
        he=HEConfig(
            mode=he_mode,
        ),
    )

    return config
