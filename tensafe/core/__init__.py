"""
TenSafe Core Module.

Provides the unified pipeline infrastructure for TenSafe:
- Configuration management
- HE backend abstraction
- Training pipeline orchestrator
- Function registry (loss/reward)
- Production gating and feature flags
"""

from tensafe.core.config import (
    DPConfig,
    HEConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    RLVRConfig,
    TenSafeConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from tensafe.core.gates import (
    FeatureGate,
    GateStatus,
    ProductionGates,
    check_gate,
    require_gate,
)
from tensafe.core.registry import (
    FunctionRegistry,
    get_loss_registry,
    get_reward_registry,
    register_function,
    resolve_function,
)

__all__ = [
    # Config
    "TenSafeConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "DPConfig",
    "HEConfig",
    "InferenceConfig",
    "RLVRConfig",
    "load_config",
    "save_config",
    # Gates
    "FeatureGate",
    "GateStatus",
    "ProductionGates",
    "require_gate",
    "check_gate",
    # Registry
    "FunctionRegistry",
    "register_function",
    "resolve_function",
    "get_loss_registry",
    "get_reward_registry",
]
