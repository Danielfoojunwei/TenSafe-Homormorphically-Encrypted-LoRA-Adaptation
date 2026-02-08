"""
TenSafe - Homomorphically Encrypted LoRA Adaptation

TenSafe is a privacy-first ML training platform that provides:

Core Features:
- Unified training pipeline for SFT and RLVR modes
- Differential privacy with RDP accounting
- Homomorphic encryption for LoRA adapters (N2HE, HEXL backends)
- Production-safe gating and validation

Components:
- core: Unified configuration, pipeline, HE interface, registry
- training: Pluggable loss functions
- rlvr: Reinforcement Learning with Verifiable Rewards
- he: Unified HE backend interface
- he_lora: MOAI-optimized HE-LoRA implementation
- inference: Unified inference with HE-LoRA support
- cookbook: Practical examples and utilities

Usage:
    # Training with unified pipeline
    from tensafe.core import TenSafeConfig, load_config
    from tensafe.core.pipeline import TenSafePipeline, train

    config = load_config("config.yaml")
    result = train(config)

    # Inference
    from tensafe.core.inference import TenSafeInference, InferenceMode

    inference = TenSafeInference.from_checkpoint("./checkpoint")
    result = inference.generate("Hello!")

    # HE backend
    from tensafe.he import get_backend

    backend = get_backend(backend_type="hexl")
    ct = backend.encrypt(data)

    # CLI
    $ tensafe train --config config.yaml
    $ tensafe inference --model ./checkpoint --prompt "Hello"
"""

from tensorguard.version import tensafe_version as _tv

__version__ = _tv()
del _tv

# Lazy imports to avoid requiring all dependencies at import time
# This allows importing tensafe even if numpy isn't installed

_LAZY_IMPORT_MAP = {
    # Config (pure Python, always available)
    "TenSafeConfig": ("tensafe.core.config", "TenSafeConfig"),
    "ModelConfig": ("tensafe.core.config", "ModelConfig"),
    "LoRAConfig": ("tensafe.core.config", "LoRAConfig"),
    "TrainingConfig": ("tensafe.core.config", "TrainingConfig"),
    "DPConfig": ("tensafe.core.config", "DPConfig"),
    "HEConfig": ("tensafe.core.config", "HEConfig"),
    "InferenceConfig": ("tensafe.core.config", "InferenceConfig"),
    "RLVRConfig": ("tensafe.core.config", "RLVRConfig"),
    "TrainingMode": ("tensafe.core.config", "TrainingMode"),
    "HEMode": ("tensafe.core.config", "HEMode"),
    "load_config": ("tensafe.core.config", "load_config"),
    "save_config": ("tensafe.core.config", "save_config"),
    "create_default_config": ("tensafe.core.config", "create_default_config"),
    # Gates (pure Python)
    "FeatureGate": ("tensafe.core.gates", "FeatureGate"),
    "GateStatus": ("tensafe.core.gates", "GateStatus"),
    "ProductionGates": ("tensafe.core.gates", "ProductionGates"),
    "require_gate": ("tensafe.core.gates", "require_gate"),
    "check_gate": ("tensafe.core.gates", "check_gate"),
    "production_check": ("tensafe.core.gates", "production_check"),
    # Registry (may require torch)
    "FunctionRegistry": ("tensafe.core.registry", "FunctionRegistry"),
    "register_function": ("tensafe.core.registry", "register_function"),
    "resolve_function": ("tensafe.core.registry", "resolve_function"),
    "get_loss_registry": ("tensafe.core.registry", "get_loss_registry"),
    "get_reward_registry": ("tensafe.core.registry", "get_reward_registry"),
    # Pipeline (requires numpy)
    "TenSafePipeline": ("tensafe.core.pipeline", "TenSafePipeline"),
    "PipelineState": ("tensafe.core.pipeline", "PipelineState"),
    "PipelineEvent": ("tensafe.core.pipeline", "PipelineEvent"),
    "TrainingResult": ("tensafe.core.pipeline", "TrainingResult"),
    "StepMetrics": ("tensafe.core.pipeline", "StepMetrics"),
    "create_pipeline": ("tensafe.core.pipeline", "create_pipeline"),
    "train": ("tensafe.core.pipeline", "train"),
    # Inference (requires numpy)
    "TenSafeInference": ("tensafe.core.inference", "TenSafeInference"),
    "InferenceMode": ("tensafe.core.inference", "InferenceMode"),
    "InferenceResult": ("tensafe.core.inference", "InferenceResult"),
    "GenerationConfig": ("tensafe.core.inference", "GenerationConfig"),
    "create_inference": ("tensafe.core.inference", "create_inference"),
    # HE interface (requires numpy)
    "HEBackendInterface": ("tensafe.core.he_interface", "HEBackendInterface"),
    "HEBackendType": ("tensafe.core.he_interface", "HEBackendType"),
    "HEParams": ("tensafe.core.he_interface", "HEParams"),
    "HEMetrics": ("tensafe.core.he_interface", "HEMetrics"),
    "get_backend": ("tensafe.core.he_interface", "get_backend"),
    "is_backend_available": ("tensafe.core.he_interface", "is_backend_available"),
    "list_available_backends": ("tensafe.core.he_interface", "list_available_backends"),
}


def __getattr__(name: str):
    """Lazy import for optional components."""
    if name in _LAZY_IMPORT_MAP:
        module_name, attr_name = _LAZY_IMPORT_MAP[name]
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module 'tensafe' has no attribute '{name}'")

__all__ = [
    # Version
    "__version__",
    # Config
    "TenSafeConfig",
    "ModelConfig",
    "LoRAConfig",
    "TrainingConfig",
    "DPConfig",
    "HEConfig",
    "InferenceConfig",
    "RLVRConfig",
    "TrainingMode",
    "HEMode",
    "load_config",
    "save_config",
    "create_default_config",
    # Gates
    "FeatureGate",
    "GateStatus",
    "ProductionGates",
    "require_gate",
    "check_gate",
    "production_check",
    # Registry
    "FunctionRegistry",
    "register_function",
    "resolve_function",
    "get_loss_registry",
    "get_reward_registry",
    # Pipeline
    "TenSafePipeline",
    "PipelineState",
    "PipelineEvent",
    "TrainingResult",
    "StepMetrics",
    "create_pipeline",
    "train",
    # Inference
    "TenSafeInference",
    "InferenceMode",
    "InferenceResult",
    "GenerationConfig",
    "create_inference",
    # HE
    "HEBackendInterface",
    "HEBackendType",
    "HEParams",
    "HEMetrics",
    "get_backend",
    "is_backend_available",
    "list_available_backends",
]
