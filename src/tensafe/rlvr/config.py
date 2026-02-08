"""
RLVR Configuration

Defines configuration classes for RLVR training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RLVRConfig:
    """
    Configuration for RLVR training.

    This configuration controls all aspects of RLVR training including:
    - Rollout generation
    - Reward computation
    - Policy optimization algorithm
    - Advantage estimation
    - Policy loss selection
    - Off-policy correction
    - Async rollout generation
    - Micro-batch gradient accumulation
    - Environment settings
    """

    # ==================================================================
    # Algorithm selection
    # ==================================================================

    # Algorithm: "reinforce", "reinforce_pp", "ppo", "grpo", "rloo"
    algorithm: str = "reinforce"

    # Advantage estimator: "baseline", "grpo", "rloo", "reinforce_pp", "gae"
    # When set, overrides the algorithm's built-in advantage computation
    advantage_estimator: Optional[str] = None

    # Policy loss: "ppo_clip", "gspo", "sapo", "cispo", "clip_cov", "kl_cov",
    #              "cross_entropy", "importance_sampling"
    # When set, overrides the algorithm's built-in loss computation
    policy_loss: Optional[str] = None

    # ==================================================================
    # Rollout settings
    # ==================================================================

    rollout_batch_size: int = 8
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Number of samples per prompt (for GRPO/RLOO group statistics)
    num_samples_per_prompt: int = 1

    # ==================================================================
    # Reward settings
    # ==================================================================

    reward_fn: str = "keyword_contains"
    reward_kwargs: Dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None

    # ==================================================================
    # Algorithm hyperparameters
    # ==================================================================

    learning_rate: float = 1e-5
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

    # ==================================================================
    # Policy loss settings
    # ==================================================================

    # GSPO/SAPO/CISPO specific
    policy_loss_clip_range: float = 0.2
    sapo_beta: float = 0.1
    clip_cov_threshold: float = 0.0
    kl_cov_coef: float = 0.1

    # Dual clip for PPO (optional)
    dual_clip: Optional[float] = None

    # ==================================================================
    # Off-policy correction settings
    # ==================================================================

    off_policy_enabled: bool = False
    off_policy_tis_clip_min: float = 0.1
    off_policy_tis_clip_max: float = 10.0
    off_policy_seq_ratio_max: float = 5.0
    off_policy_seq_ratio_min: float = 0.2
    off_policy_outlier_z_threshold: float = 3.0
    off_policy_drop_outlier_sequences: bool = True
    off_policy_staleness_decay: float = 0.95

    # ==================================================================
    # Async rollout settings
    # ==================================================================

    async_rollout_enabled: bool = False
    async_max_staleness_steps: int = 5
    async_max_buffer_size: int = 1000
    async_max_generation_slots: int = 4
    async_min_batch_size: int = 8
    async_batch_timeout: float = 30.0
    async_num_workers: int = 2
    async_track_consumed_uids: bool = True

    # ==================================================================
    # Micro-batch gradient accumulation
    # ==================================================================

    micro_batch_size: int = 0  # 0 = disabled (use full batch)
    effective_batch_size: int = 32
    micro_batch_scale_gradients: bool = True

    # DP-SGD coordination with micro-batching
    dp_micro_batch_enabled: bool = False
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0

    # ==================================================================
    # Environment settings
    # ==================================================================

    # Environment name (from registry), or None for raw RewardFn
    environment: Optional[str] = None
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_turns: int = 1  # 1 = single-turn (default)
    reward_shaping_scale: float = 1.0
    reward_shaping_clip: Optional[float] = None

    # ==================================================================
    # Training settings
    # ==================================================================

    total_steps: int = 1000
    eval_interval: int = 100
    save_interval: int = 500
    log_interval: int = 10

    # Buffer settings
    buffer_size: int = 10000
    prioritized_replay: bool = False
    priority_alpha: float = 0.6

    # Gradient settings
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "advantage_estimator": self.advantage_estimator,
            "policy_loss": self.policy_loss,
            "rollout": {
                "batch_size": self.rollout_batch_size,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "num_samples_per_prompt": self.num_samples_per_prompt,
            },
            "reward": {
                "fn": self.reward_fn,
                "kwargs": self.reward_kwargs,
                "scale": self.reward_scale,
                "clip": self.reward_clip,
            },
            "algorithm_params": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "use_baseline": self.use_baseline,
                "baseline_decay": self.baseline_decay,
                "normalize_advantages": self.normalize_advantages,
                "entropy_coef": self.entropy_coef,
                "kl_coef": self.kl_coef,
            },
            "ppo": {
                "clip_eps": self.ppo_clip_eps,
                "epochs": self.ppo_epochs,
                "minibatch_size": self.ppo_minibatch_size,
                "vf_coef": self.vf_coef,
                "dual_clip": self.dual_clip,
            },
            "grpo": {
                "normalize_within_group": self.grpo_normalize_within_group,
                "normalize_batch": self.grpo_normalize_batch,
                "min_group_size": self.grpo_min_group_size,
            },
            "rloo": {
                "fallback_to_batch_mean": self.rloo_fallback_to_batch_mean,
            },
            "reinforce_pp": {
                "temporal_whitening": self.reinforce_pp_temporal_whitening,
            },
            "policy_loss_params": {
                "clip_range": self.policy_loss_clip_range,
                "sapo_beta": self.sapo_beta,
                "clip_cov_threshold": self.clip_cov_threshold,
                "kl_cov_coef": self.kl_cov_coef,
            },
            "off_policy": {
                "enabled": self.off_policy_enabled,
                "tis_clip_min": self.off_policy_tis_clip_min,
                "tis_clip_max": self.off_policy_tis_clip_max,
                "seq_ratio_max": self.off_policy_seq_ratio_max,
                "seq_ratio_min": self.off_policy_seq_ratio_min,
                "outlier_z_threshold": self.off_policy_outlier_z_threshold,
                "drop_outlier_sequences": self.off_policy_drop_outlier_sequences,
                "staleness_decay": self.off_policy_staleness_decay,
            },
            "async_rollout": {
                "enabled": self.async_rollout_enabled,
                "max_staleness_steps": self.async_max_staleness_steps,
                "max_buffer_size": self.async_max_buffer_size,
                "max_generation_slots": self.async_max_generation_slots,
                "min_batch_size": self.async_min_batch_size,
                "batch_timeout": self.async_batch_timeout,
                "num_workers": self.async_num_workers,
            },
            "micro_batch": {
                "micro_batch_size": self.micro_batch_size,
                "effective_batch_size": self.effective_batch_size,
                "scale_gradients": self.micro_batch_scale_gradients,
                "dp_enabled": self.dp_micro_batch_enabled,
                "dp_max_grad_norm": self.dp_max_grad_norm,
                "dp_noise_multiplier": self.dp_noise_multiplier,
            },
            "environment": {
                "name": self.environment,
                "kwargs": self.environment_kwargs,
                "max_turns": self.max_turns,
                "reward_shaping_scale": self.reward_shaping_scale,
                "reward_shaping_clip": self.reward_shaping_clip,
            },
            "training": {
                "total_steps": self.total_steps,
                "eval_interval": self.eval_interval,
                "save_interval": self.save_interval,
                "log_interval": self.log_interval,
            },
            "buffer": {
                "size": self.buffer_size,
                "prioritized": self.prioritized_replay,
            },
            "gradient": {
                "max_norm": self.max_grad_norm,
                "accumulation_steps": self.gradient_accumulation_steps,
            },
            "seed": self.seed,
            "deterministic": self.deterministic,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RLVRConfig":
        """Create from dictionary."""
        flat = {}
        flat["algorithm"] = d.get("algorithm", "reinforce")
        flat["advantage_estimator"] = d.get("advantage_estimator")
        flat["policy_loss"] = d.get("policy_loss")

        if "rollout" in d:
            flat["rollout_batch_size"] = d["rollout"].get("batch_size", 8)
            flat["max_new_tokens"] = d["rollout"].get("max_new_tokens", 128)
            flat["temperature"] = d["rollout"].get("temperature", 0.7)
            flat["top_p"] = d["rollout"].get("top_p", 0.9)
            flat["top_k"] = d["rollout"].get("top_k", 50)
            flat["num_samples_per_prompt"] = d["rollout"].get("num_samples_per_prompt", 1)

        if "reward" in d:
            flat["reward_fn"] = d["reward"].get("fn", "keyword_contains")
            flat["reward_kwargs"] = d["reward"].get("kwargs", {})
            flat["reward_scale"] = d["reward"].get("scale", 1.0)
            flat["reward_clip"] = d["reward"].get("clip")

        if "algorithm_params" in d:
            params = d["algorithm_params"]
            flat["learning_rate"] = params.get("learning_rate", 1e-5)
            flat["gamma"] = params.get("gamma", 1.0)
            flat["use_baseline"] = params.get("use_baseline", True)
            flat["baseline_decay"] = params.get("baseline_decay", 0.99)
            flat["normalize_advantages"] = params.get("normalize_advantages", True)
            flat["entropy_coef"] = params.get("entropy_coef", 0.01)
            flat["kl_coef"] = params.get("kl_coef", 0.0)

        if "ppo" in d:
            flat["ppo_clip_eps"] = d["ppo"].get("clip_eps", 0.2)
            flat["ppo_epochs"] = d["ppo"].get("epochs", 4)
            flat["ppo_minibatch_size"] = d["ppo"].get("minibatch_size", 4)
            flat["vf_coef"] = d["ppo"].get("vf_coef", 0.5)
            flat["dual_clip"] = d["ppo"].get("dual_clip")

        if "grpo" in d:
            flat["grpo_normalize_within_group"] = d["grpo"].get("normalize_within_group", True)
            flat["grpo_normalize_batch"] = d["grpo"].get("normalize_batch", False)
            flat["grpo_min_group_size"] = d["grpo"].get("min_group_size", 2)

        if "rloo" in d:
            flat["rloo_fallback_to_batch_mean"] = d["rloo"].get("fallback_to_batch_mean", True)

        if "reinforce_pp" in d:
            flat["reinforce_pp_temporal_whitening"] = d["reinforce_pp"].get("temporal_whitening", True)

        if "policy_loss_params" in d:
            plp = d["policy_loss_params"]
            flat["policy_loss_clip_range"] = plp.get("clip_range", 0.2)
            flat["sapo_beta"] = plp.get("sapo_beta", 0.1)
            flat["clip_cov_threshold"] = plp.get("clip_cov_threshold", 0.0)
            flat["kl_cov_coef"] = plp.get("kl_cov_coef", 0.1)

        if "off_policy" in d:
            op = d["off_policy"]
            flat["off_policy_enabled"] = op.get("enabled", False)
            flat["off_policy_tis_clip_min"] = op.get("tis_clip_min", 0.1)
            flat["off_policy_tis_clip_max"] = op.get("tis_clip_max", 10.0)
            flat["off_policy_seq_ratio_max"] = op.get("seq_ratio_max", 5.0)
            flat["off_policy_seq_ratio_min"] = op.get("seq_ratio_min", 0.2)
            flat["off_policy_outlier_z_threshold"] = op.get("outlier_z_threshold", 3.0)
            flat["off_policy_drop_outlier_sequences"] = op.get("drop_outlier_sequences", True)
            flat["off_policy_staleness_decay"] = op.get("staleness_decay", 0.95)

        if "async_rollout" in d:
            ar = d["async_rollout"]
            flat["async_rollout_enabled"] = ar.get("enabled", False)
            flat["async_max_staleness_steps"] = ar.get("max_staleness_steps", 5)
            flat["async_max_buffer_size"] = ar.get("max_buffer_size", 1000)
            flat["async_max_generation_slots"] = ar.get("max_generation_slots", 4)
            flat["async_min_batch_size"] = ar.get("min_batch_size", 8)
            flat["async_batch_timeout"] = ar.get("batch_timeout", 30.0)
            flat["async_num_workers"] = ar.get("num_workers", 2)

        if "micro_batch" in d:
            mb = d["micro_batch"]
            flat["micro_batch_size"] = mb.get("micro_batch_size", 0)
            flat["effective_batch_size"] = mb.get("effective_batch_size", 32)
            flat["micro_batch_scale_gradients"] = mb.get("scale_gradients", True)
            flat["dp_micro_batch_enabled"] = mb.get("dp_enabled", False)
            flat["dp_max_grad_norm"] = mb.get("dp_max_grad_norm", 1.0)
            flat["dp_noise_multiplier"] = mb.get("dp_noise_multiplier", 1.0)

        if "environment" in d:
            env = d["environment"]
            flat["environment"] = env.get("name")
            flat["environment_kwargs"] = env.get("kwargs", {})
            flat["max_turns"] = env.get("max_turns", 1)
            flat["reward_shaping_scale"] = env.get("reward_shaping_scale", 1.0)
            flat["reward_shaping_clip"] = env.get("reward_shaping_clip")

        if "training" in d:
            flat["total_steps"] = d["training"].get("total_steps", 1000)
            flat["eval_interval"] = d["training"].get("eval_interval", 100)
            flat["save_interval"] = d["training"].get("save_interval", 500)
            flat["log_interval"] = d["training"].get("log_interval", 10)

        if "buffer" in d:
            flat["buffer_size"] = d["buffer"].get("size", 10000)
            flat["prioritized_replay"] = d["buffer"].get("prioritized", False)

        if "gradient" in d:
            flat["max_grad_norm"] = d["gradient"].get("max_norm", 1.0)
            flat["gradient_accumulation_steps"] = d["gradient"].get("accumulation_steps", 1)

        flat["seed"] = d.get("seed", 42)
        flat["deterministic"] = d.get("deterministic", False)

        # Filter out None values for optional fields
        flat = {k: v for k, v in flat.items() if v is not None or k in ("advantage_estimator", "policy_loss", "reward_clip", "dual_clip", "environment", "reward_shaping_clip")}

        return cls(**flat)
