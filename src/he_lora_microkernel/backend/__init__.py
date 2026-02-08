"""
HE-LoRA Microkernel Backend

This package provides GPU-accelerated CKKS backends for HE-LoRA.

Supported backends:
  - SIMULATION: CPU simulation for testing (always available)
  - HEONGPU: HEonGPU library (GPU, requires installation)
  - FIDESLIB: FIDESlib library (GPU, requires installation)
  - OPENFHE_GPU: OpenFHE-GPU fork (GPU, requires installation)

Usage:
    from he_lora_microkernel.backend.gpu_ckks_backend import (
        BackendType,
        create_backend,
        GPUCKKSBackend,
    )

    # Create simulation backend
    backend = create_backend(
        BackendType.SIMULATION,
        ckks_params,
        device_id=0,
    )

    # Encrypt/compute/decrypt
    ct = backend.encrypt(plaintext)
    ct_result = backend.mul_plain(ct, pt)
    result = backend.decrypt(ct_result)
"""

from .gpu_ckks_backend import (
    # Types
    BackendType,
    GPUCKKSBackend,
    GPUCiphertext,
    PlaintextPacked,
    OperationCounters,
    # Factory
    create_backend,
    get_available_backends,
    register_backend,
    # Simulation
    SimulationBackend,
)

from .base_adapter import (
    BatchConfig,
    InsertionConfig,
    ModelMetadata,
    LoRATargets,
    InsertionPoint,
    LayerDeltas,
    BaseRuntimeAdapter,
    get_adapter,
    list_available_adapters,
    register_adapter,
)

# Import adapter implementations to auto-register them
# These imports trigger the @register_adapter decorators
try:
    from .vllm_adapter import adapter as _vllm
except ImportError:
    pass

try:
    from .tensorrt_llm_adapter import adapter as _trt
except ImportError:
    pass

try:
    from .sglang_adapter import adapter as _sglang
except ImportError:
    pass

__all__ = [
    'BackendType',
    'GPUCKKSBackend',
    'GPUCiphertext',
    'PlaintextPacked',
    'OperationCounters',
    'create_backend',
    'get_available_backends',
    'register_backend',
    'SimulationBackend',
    # Adapter exports
    'BatchConfig',
    'InsertionConfig',
    'ModelMetadata',
    'LoRATargets',
    'InsertionPoint',
    'LayerDeltas',
    'BaseRuntimeAdapter',
    'get_adapter',
    'list_available_adapters',
    'register_adapter',
]

