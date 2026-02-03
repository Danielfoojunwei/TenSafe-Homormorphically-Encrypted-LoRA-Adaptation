"""
RLVR Configuration

Defines configuration classes for RLVR training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RLVRConfig:
    """
    Configuration for RLVR training.

    This configuration controls all aspects of RLVR training including:
    - Rollout generation
    - Reward computation
    - Policy optimization algorithm
    """

    # Training mode
    algorithm: str = "reinforce"  # "reinforce" or "ppo"

    # Rollout settings
    rollout_batch_size: int = 8
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50

    # Reward settings
    reward_fn: str = "keyword_contains"  # Name or dotted path
    reward_kwargs: Dict[str, Any] = field(default_factory=dict)
    reward_scale: float = 1.0
    reward_clip: Optional[float] = None  # Clip rewards to [-clip, clip]

    # Algorithm settings (REINFORCE)
    learning_rate: float = 1e-5
    gamma: float = 1.0  # Discount factor (usually 1.0 for single-turn)
    use_baseline: bool = True
    baseline_decay: float = 0.99
    normalize_advantages: bool = True
    entropy_coef: float = 0.01
    kl_coef: float = 0.0

    # PPO-specific settings (used when algorithm="ppo")
    ppo_clip_eps: float = 0.2
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 4
    vf_coef: float = 0.5  # Value function coefficient

    # Training settings
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
            "rollout": {
                "batch_size": self.rollout_batch_size,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
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
    def from_dict(cls, d: Dict[str, Any]) -> RLVRConfig:
        """Create from dictionary."""
        # Flatten nested dicts
        flat = {}
        flat["algorithm"] = d.get("algorithm", "reinforce")

        if "rollout" in d:
            flat["rollout_batch_size"] = d["rollout"].get("batch_size", 8)
            flat["max_new_tokens"] = d["rollout"].get("max_new_tokens", 128)
            flat["temperature"] = d["rollout"].get("temperature", 0.7)
            flat["top_p"] = d["rollout"].get("top_p", 0.9)
            flat["top_k"] = d["rollout"].get("top_k", 50)

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

        return cls(**flat)
