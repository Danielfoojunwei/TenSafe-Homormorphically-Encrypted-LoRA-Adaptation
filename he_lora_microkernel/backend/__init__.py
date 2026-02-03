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
    GPUCiphertext,
    GPUCKKSBackend,
    OperationCounters,
    PlaintextPacked,
    # Simulation
    SimulationBackend,
    # Factory
    create_backend,
    get_available_backends,
    register_backend,
)

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
]
