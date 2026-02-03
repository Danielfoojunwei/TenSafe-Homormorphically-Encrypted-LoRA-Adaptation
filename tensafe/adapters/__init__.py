"""
TenSafe Adapters Module - Production-Grade Hot-Swap System.

This module provides comprehensive support for adapter management with:

- Multiple adapter types (LoRA, DoRA, AdaLoRA, VeRA, rsLoRA, Gated)
- Architecture-aware automatic placement resolution
- Tiered caching (GPU -> CPU -> Disk)
- Zero-copy hot-swap mechanism
- Production lifecycle management with health monitoring

Usage:
    from tensafe.adapters import (
        AdapterType,
        AdapterConfig,
        create_adapter,
        AdapterPlacementResolver,
        HotSwapManager,
        AdapterLifecycleManager,
    )

    # Create adapter
    config = AdapterConfig(adapter_type=AdapterType.DORA, rank=16)
    adapter = create_adapter(config, in_features=4096, out_features=4096)

    # Resolve placement targets
    resolver = AdapterPlacementResolver(model)
    targets = resolver.resolve()

    # Use hot-swap manager
    manager = HotSwapManager()
    manager.register_adapter("my_adapter", weights, config)
    manager.activate("my_adapter")
    delta = manager.forward(x, "q_proj")

Author: TenSafe Team
"""

# Adapter types
from .adapter_types import (
    AdapterType,
    ScalingStrategy,
    AdapterConfig,
    BaseAdapter,
    LoRAAdapter,
    rsLoRAAdapter,
    LoRAFAAdapter,
    DoRAAdapter,
    VeRAAdapter,
    AdaLoRAAdapter,
    GatedLoRAAdapter,
    GLoRAAdapter,
    create_adapter,
    register_adapter_type,
    list_adapter_types,
    convert_adapter,
)

# Placement resolution
from .placement import (
    LayerType,
    ProjectionType,
    TargetScope,
    ProjectionTarget,
    ArchitectureConfig,
    PlacementConfig,
    AdapterPlacementResolver,
    FusedProjectionHandler,
    ARCHITECTURE_CONFIGS,
    get_architecture_config,
    auto_discover_targets,
    get_layer_importance_weights,
)

# Hot-swap and lifecycle
from .hot_swap import (
    CacheTier,
    AdapterState,
    EvictionPolicy,
    AdapterMetadata,
    PackedWeights,
    TieredCacheConfig,
    HotSwapConfig,
    LifecycleConfig,
    BaseEvictionPolicy,
    LRUEvictionPolicy,
    GDSFEvictionPolicy,
    create_eviction_policy,
    TieredAdapterCache,
    HotSwapManager,
    AdapterLifecycleManager,
)

# Hooks
from .hooks import (
    InjectionMode,
    HookConfig,
    HookStatistics,
    StandardDeltaCallback,
    ExtendedDeltaCallback,
    BaseAdapterHook,
    LinearAdapterHook,
    DoRAHook,
    GatedAdapterHook,
    FusedQKVHook,
    HookManager,
    create_hooks_for_model,
)


__all__ = [
    # === Adapter Types ===
    "AdapterType",
    "ScalingStrategy",
    "AdapterConfig",
    "BaseAdapter",
    "LoRAAdapter",
    "rsLoRAAdapter",
    "LoRAFAAdapter",
    "DoRAAdapter",
    "VeRAAdapter",
    "AdaLoRAAdapter",
    "GatedLoRAAdapter",
    "GLoRAAdapter",
    "create_adapter",
    "register_adapter_type",
    "list_adapter_types",
    "convert_adapter",

    # === Placement ===
    "LayerType",
    "ProjectionType",
    "TargetScope",
    "ProjectionTarget",
    "ArchitectureConfig",
    "PlacementConfig",
    "AdapterPlacementResolver",
    "FusedProjectionHandler",
    "ARCHITECTURE_CONFIGS",
    "get_architecture_config",
    "auto_discover_targets",
    "get_layer_importance_weights",

    # === Hot-Swap ===
    "CacheTier",
    "AdapterState",
    "EvictionPolicy",
    "AdapterMetadata",
    "PackedWeights",
    "TieredCacheConfig",
    "HotSwapConfig",
    "LifecycleConfig",
    "BaseEvictionPolicy",
    "LRUEvictionPolicy",
    "GDSFEvictionPolicy",
    "create_eviction_policy",
    "TieredAdapterCache",
    "HotSwapManager",
    "AdapterLifecycleManager",

    # === Hooks ===
    "InjectionMode",
    "HookConfig",
    "HookStatistics",
    "StandardDeltaCallback",
    "ExtendedDeltaCallback",
    "BaseAdapterHook",
    "LinearAdapterHook",
    "DoRAHook",
    "GatedAdapterHook",
    "FusedQKVHook",
    "HookManager",
    "create_hooks_for_model",
]
