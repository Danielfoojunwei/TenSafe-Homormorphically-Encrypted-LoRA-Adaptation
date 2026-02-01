"""
Central Backend Registry.

Provides a unified registry for all backend types:
- ML Backends (PyTorch, etc.)
- HE Backends (N2HE, HEXL)
- Privacy Accountants (RDP, PRV, Moments)

This enables:
- Runtime backend selection
- Dependency checking
- Plugin architecture for custom backends
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BackendStatus(Enum):
    """Status of a backend."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    REQUIRES_SETUP = "requires_setup"


@dataclass
class BackendInfo:
    """Information about a registered backend."""
    name: str
    backend_class: Type
    description: str
    status: BackendStatus
    dependencies: List[str]
    is_production_ready: bool


class BackendRegistry:
    """
    Central registry for all backend types.

    Usage:
        # Register a backend
        BackendRegistry.register_ml_backend(
            "torch",
            TorchMLBackend,
            description="PyTorch/Transformers backend",
            dependencies=["torch", "transformers", "peft"],
        )

        # Get a backend
        backend = BackendRegistry.get_ml_backend("torch")

        # List available backends
        backends = BackendRegistry.list_ml_backends()
    """

    # ML Backends
    _ml_backends: Dict[str, BackendInfo] = {}

    # HE Backends
    _he_backends: Dict[str, BackendInfo] = {}

    # Privacy Accountants
    _privacy_accountants: Dict[str, BackendInfo] = {}

    # ===========================================================================
    # ML Backend Registration
    # ===========================================================================

    @classmethod
    def register_ml_backend(
        cls,
        name: str,
        backend_class: Type,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        is_production_ready: bool = True,
    ) -> None:
        """
        Register an ML backend.

        Args:
            name: Backend name (e.g., "torch")
            backend_class: Backend class implementing MLBackendInterface
            description: Human-readable description
            dependencies: Required Python packages
            is_production_ready: Whether suitable for production
        """
        status = cls._check_dependencies(dependencies or [])

        info = BackendInfo(
            name=name,
            backend_class=backend_class,
            description=description,
            status=status,
            dependencies=dependencies or [],
            is_production_ready=is_production_ready,
        )

        cls._ml_backends[name.lower()] = info
        logger.debug(f"Registered ML backend: {name} (status={status.value})")

    @classmethod
    def get_ml_backend(cls, name: str, **kwargs: Any):
        """
        Get an ML backend instance.

        Args:
            name: Backend name
            **kwargs: Passed to backend constructor

        Returns:
            Backend instance

        Raises:
            ValueError: If backend not found or unavailable
        """
        info = cls._ml_backends.get(name.lower())

        if info is None:
            available = list(cls._ml_backends.keys())
            raise ValueError(f"ML backend '{name}' not found. Available: {available}")

        if info.status == BackendStatus.UNAVAILABLE:
            raise ValueError(
                f"ML backend '{name}' unavailable. "
                f"Missing dependencies: {info.dependencies}"
            )

        return info.backend_class(**kwargs)

    @classmethod
    def list_ml_backends(cls, only_available: bool = False) -> List[BackendInfo]:
        """List all registered ML backends."""
        backends = list(cls._ml_backends.values())
        if only_available:
            backends = [b for b in backends if b.status == BackendStatus.AVAILABLE]
        return backends

    # ===========================================================================
    # HE Backend Registration
    # ===========================================================================

    @classmethod
    def register_he_backend(
        cls,
        name: str,
        backend_class: Type,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        is_production_ready: bool = True,
    ) -> None:
        """Register an HE backend."""
        status = cls._check_dependencies(dependencies or [])

        info = BackendInfo(
            name=name,
            backend_class=backend_class,
            description=description,
            status=status,
            dependencies=dependencies or [],
            is_production_ready=is_production_ready,
        )

        cls._he_backends[name.lower()] = info
        logger.debug(f"Registered HE backend: {name}")

    @classmethod
    def get_he_backend(cls, name: str, **kwargs: Any):
        """Get an HE backend instance."""
        info = cls._he_backends.get(name.lower())

        if info is None:
            available = list(cls._he_backends.keys())
            raise ValueError(f"HE backend '{name}' not found. Available: {available}")

        if info.status == BackendStatus.UNAVAILABLE:
            raise ValueError(
                f"HE backend '{name}' unavailable. "
                f"Missing dependencies: {info.dependencies}"
            )

        return info.backend_class(**kwargs)

    @classmethod
    def list_he_backends(cls, only_available: bool = False) -> List[BackendInfo]:
        """List all registered HE backends."""
        backends = list(cls._he_backends.values())
        if only_available:
            backends = [b for b in backends if b.status == BackendStatus.AVAILABLE]
        return backends

    # ===========================================================================
    # Privacy Accountant Registration
    # ===========================================================================

    @classmethod
    def register_privacy_accountant(
        cls,
        name: str,
        accountant_class: Type,
        description: str = "",
        dependencies: Optional[List[str]] = None,
        is_production_ready: bool = True,
    ) -> None:
        """Register a privacy accountant."""
        status = cls._check_dependencies(dependencies or [])

        info = BackendInfo(
            name=name,
            backend_class=accountant_class,
            description=description,
            status=status,
            dependencies=dependencies or [],
            is_production_ready=is_production_ready,
        )

        cls._privacy_accountants[name.lower()] = info
        logger.debug(f"Registered privacy accountant: {name}")

    @classmethod
    def get_privacy_accountant(cls, name: str, **kwargs: Any):
        """Get a privacy accountant instance."""
        info = cls._privacy_accountants.get(name.lower())

        if info is None:
            available = list(cls._privacy_accountants.keys())
            raise ValueError(
                f"Privacy accountant '{name}' not found. Available: {available}"
            )

        if info.status == BackendStatus.UNAVAILABLE:
            raise ValueError(
                f"Privacy accountant '{name}' unavailable. "
                f"Missing dependencies: {info.dependencies}"
            )

        return info.backend_class(**kwargs)

    @classmethod
    def list_privacy_accountants(
        cls,
        only_available: bool = False,
    ) -> List[BackendInfo]:
        """List all registered privacy accountants."""
        accountants = list(cls._privacy_accountants.values())
        if only_available:
            accountants = [
                a for a in accountants if a.status == BackendStatus.AVAILABLE
            ]
        return accountants

    # ===========================================================================
    # Utility Methods
    # ===========================================================================

    @classmethod
    def _check_dependencies(cls, dependencies: List[str]) -> BackendStatus:
        """Check if all dependencies are available."""
        import importlib

        for dep in dependencies:
            try:
                importlib.import_module(dep)
            except ImportError:
                return BackendStatus.UNAVAILABLE

        return BackendStatus.AVAILABLE

    @classmethod
    def get_best_ml_backend(cls, prefer_production: bool = True):
        """Get the best available ML backend."""
        backends = cls.list_ml_backends(only_available=True)

        if not backends:
            raise RuntimeError("No ML backends available")

        if prefer_production:
            production = [b for b in backends if b.is_production_ready]
            if production:
                backends = production

        # Prefer PyTorch
        for backend in backends:
            if "torch" in backend.name.lower():
                return cls.get_ml_backend(backend.name)

        return cls.get_ml_backend(backends[0].name)

    @classmethod
    def get_best_he_backend(cls, prefer_production: bool = True):
        """Get the best available HE backend."""
        backends = cls.list_he_backends(only_available=True)

        if not backends:
            raise RuntimeError("No HE backends available")

        if prefer_production:
            production = [b for b in backends if b.is_production_ready]
            if production:
                backends = production

        # Prefer HEXL
        for backend in backends:
            if "hexl" in backend.name.lower():
                return cls.get_he_backend(backend.name)

        return cls.get_he_backend(backends[0].name)

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._ml_backends.clear()
        cls._he_backends.clear()
        cls._privacy_accountants.clear()


# ==============================================================================
# Convenience Functions
# ==============================================================================


def register_ml_backend(
    name: str,
    backend_class: Type,
    **kwargs: Any,
) -> None:
    """Register an ML backend."""
    BackendRegistry.register_ml_backend(name, backend_class, **kwargs)


def register_privacy_accountant(
    name: str,
    accountant_class: Type,
    **kwargs: Any,
) -> None:
    """Register a privacy accountant."""
    BackendRegistry.register_privacy_accountant(name, accountant_class, **kwargs)


def register_he_backend(
    name: str,
    backend_class: Type,
    **kwargs: Any,
) -> None:
    """Register an HE backend."""
    BackendRegistry.register_he_backend(name, backend_class, **kwargs)


# ==============================================================================
# Auto-Registration
# ==============================================================================


def _auto_register_backends() -> None:
    """Auto-register available backends on module load."""
    # Register ML backends
    try:
        from tensafe.backends.ml_backend import TorchMLBackend

        BackendRegistry.register_ml_backend(
            "torch",
            TorchMLBackend,
            description="PyTorch/Transformers with PEFT/LoRA",
            dependencies=["torch", "transformers"],
            is_production_ready=True,
        )
    except ImportError:
        pass

    # Register HE backends
    try:
        from tensafe.core.he_interface import ToyHEBackend

        BackendRegistry.register_he_backend(
            "toy",
            ToyHEBackend,
            description="Toy HE backend (NOT SECURE, dev only)",
            dependencies=[],
            is_production_ready=False,
        )
    except ImportError:
        pass

    try:
        from tensafe.core.he_interface import N2HEBackendWrapper

        BackendRegistry.register_he_backend(
            "n2he",
            N2HEBackendWrapper,
            description="N2HE backend (pure Python with optional native)",
            dependencies=["tensorguard.n2he"],
            is_production_ready=True,
        )
    except ImportError:
        pass

    try:
        from tensafe.core.he_interface import HEXLBackendWrapper

        BackendRegistry.register_he_backend(
            "hexl",
            HEXLBackendWrapper,
            description="N2HE-HEXL production backend",
            dependencies=["tensafe.he_lora.backend"],
            is_production_ready=True,
        )
    except ImportError:
        pass


# Run auto-registration on import
_auto_register_backends()
