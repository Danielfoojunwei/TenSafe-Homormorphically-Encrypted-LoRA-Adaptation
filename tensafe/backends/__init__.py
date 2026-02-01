"""
TenSafe Backend Infrastructure.

This module provides unified interfaces for all backend components:
- ML Backends (PyTorch, JAX)
- HE Backends (N2HE, HEXL)
- Privacy Accountants

All production code paths use these interfaces.
"""

from tensafe.backends.ml_backend import (
    MLBackendInterface,
    MLBackendConfig,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
    get_ml_backend,
    list_available_ml_backends,
)

from tensafe.backends.registry import (
    BackendRegistry,
    register_ml_backend,
    register_privacy_accountant,
)

__all__ = [
    # ML Backend
    "MLBackendInterface",
    "MLBackendConfig",
    "ForwardBackwardResult",
    "OptimStepResult",
    "SampleResult",
    "get_ml_backend",
    "list_available_ml_backends",
    # Registry
    "BackendRegistry",
    "register_ml_backend",
    "register_privacy_accountant",
]
