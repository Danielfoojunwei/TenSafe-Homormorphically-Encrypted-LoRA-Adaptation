# Platform services module
from .adapter_registry import AdapterRegistry, AdapterRegistryError, AdapterNotFoundError, PromotionGateError

__all__ = [
    "AdapterRegistry",
    "AdapterRegistryError",
    "AdapterNotFoundError",
    "PromotionGateError",
]
