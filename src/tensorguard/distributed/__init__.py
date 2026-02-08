"""TenSafe Distributed Training Module.

This module provides distributed training capabilities using Ray Train,
with privacy-preserving features including:
- Distributed DP-SGD with secure gradient aggregation
- Multi-node HE key management
- DeepSpeed and FSDP integration
- Privacy budget tracking across workers
"""

from .dp_distributed import DistributedDPOptimizer, SecureGradientAggregator
from .ray_trainer import TenSafeRayConfig, TenSafeRayTrainer

__all__ = [
    "TenSafeRayTrainer",
    "TenSafeRayConfig",
    "DistributedDPOptimizer",
    "SecureGradientAggregator",
]
