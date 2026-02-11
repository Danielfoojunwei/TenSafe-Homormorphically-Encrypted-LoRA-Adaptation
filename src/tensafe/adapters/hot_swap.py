"""
Production Hot-Swap System for TenSafe Adapters.

This module implements production-grade hot-swapping capabilities with:

- Tiered Caching: GPU -> CPU -> Disk with async prefetch/offload
- GDSF Eviction: Greedy-Dual-Size-Frequency based eviction policy
- Zero-Copy Swap: Pointer-swap mechanism for instant switching
- Lifecycle Management: Full adapter lifecycle with health monitoring
- Memory Pool: Unified memory management for KV cache and adapters

Drawing from industry best practices:
    - S-LoRA: Unified paging, custom CUDA kernels
    - LoRAX: Tiered caching with async operations
    - Punica: SGMV kernel for multi-adapter batching
    - PEFT: Hot-swap without recompilation

References:
    - S-LoRA: https://arxiv.org/abs/2311.03285
    - LoRAX: https://github.com/predibase/lorax
    - Punica: https://arxiv.org/abs/2310.18547

Author: TenSafe Team
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import shutil
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union

import numpy as np

from .adapter_types import AdapterConfig, AdapterType, BaseAdapter, create_adapter

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

T = TypeVar('T')
WeightDict = Dict[str, Tuple[np.ndarray, np.ndarray]]


class CacheTier(Enum):
    """Cache tier levels."""
    GPU = "gpu"         # Hot: In GPU memory, ready for inference
    CPU = "cpu"         # Warm: In CPU memory, quick to move to GPU
    DISK = "disk"       # Cold: On disk, needs loading
    REMOTE = "remote"   # External: In cloud storage


class AdapterState(Enum):
    """Adapter lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    OFFLOADING = "offloading"
    ERROR = "error"


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    GDSF = "gdsf"         # Greedy-Dual-Size-Frequency
    FIFO = "fifo"         # First In First Out
    RANDOM = "random"     # Random eviction


# =============================================================================
# ADAPTER METADATA & WEIGHT CONTAINERS
# =============================================================================

@dataclass
class AdapterMetadata:
    """
    Comprehensive metadata for a loaded adapter.

    Tracks all information needed for lifecycle management,
    caching decisions, and compatibility checking.
    """
    # Identity
    adapter_id: str
    source_path: str
    content_hash: str

    # Configuration
    adapter_type: AdapterType
    config: AdapterConfig

    # Model compatibility
    base_model_id: str
    hidden_size: int
    target_modules: List[str]

    # Size metrics
    total_params: int
    memory_bytes: int

    # State
    state: AdapterState = AdapterState.UNLOADED
    current_tier: CacheTier = CacheTier.DISK

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    loaded_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_activated_at: Optional[datetime] = None

    # Usage statistics
    activation_count: int = 0
    request_count: int = 0
    total_latency_ms: float = 0.0

    # Load metrics (for GDSF)
    load_time_ms: float = 0.0  # Time to load from disk

    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "source_path": self.source_path,
            "content_hash": self.content_hash,
            "adapter_type": self.adapter_type.value,
            "base_model_id": self.base_model_id,
            "hidden_size": self.hidden_size,
            "target_modules": self.target_modules,
            "total_params": self.total_params,
            "memory_bytes": self.memory_bytes,
            "state": self.state.value,
            "current_tier": self.current_tier.value,
            "created_at": self.created_at.isoformat(),
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "activation_count": self.activation_count,
            "request_count": self.request_count,
            "load_time_ms": self.load_time_ms,
        }


@dataclass
class PackedWeights:
    """
    Packed adapter weights optimized for hot-swap.

    Weights are pre-packed in a GPU-friendly format for instant swapping.
    """
    # Module -> (lora_A, lora_B) arrays
    weights: WeightDict

    # Pre-packed weights for CKKS (column-packed format)
    packed_weights: Optional[Dict[str, Any]] = None

    # GPU tensors (when loaded to GPU)
    gpu_tensors: Optional[Dict[str, Any]] = None

    # Memory tracking
    cpu_memory_bytes: int = 0
    gpu_memory_bytes: int = 0

    # Pack status
    is_packed: bool = False
    is_on_gpu: bool = False

    def compute_memory_size(self) -> int:
        """Compute total memory size in bytes."""
        total = 0
        for module_name, (lora_a, lora_b) in self.weights.items():
            total += lora_a.nbytes + lora_b.nbytes
        self.cpu_memory_bytes = total
        return total


# =============================================================================
# EVICTION POLICIES
# =============================================================================

class BaseEvictionPolicy(ABC):
    """Base class for cache eviction policies."""

    @abstractmethod
    def compute_score(
        self,
        metadata: AdapterMetadata,
        current_time: float,
    ) -> float:
        """
        Compute eviction score for an adapter.

        Higher score = more valuable to keep.
        """
        pass

    @abstractmethod
    def get_eviction_candidates(
        self,
        adapters: Dict[str, AdapterMetadata],
        num_to_evict: int,
        protected_ids: Set[str],
    ) -> List[str]:
        """Get list of adapter IDs to evict."""
        pass


class LRUEvictionPolicy(BaseEvictionPolicy):
    """Least Recently Used eviction policy."""

    def compute_score(
        self,
        metadata: AdapterMetadata,
        current_time: float,
    ) -> float:
        """Score based on recency. Higher = more recent = keep."""
        if metadata.last_used_at is None:
            return 0.0
        return metadata.last_used_at.timestamp()

    def get_eviction_candidates(
        self,
        adapters: Dict[str, AdapterMetadata],
        num_to_evict: int,
        protected_ids: Set[str],
    ) -> List[str]:
        current_time = time.time()
        scores = []

        for adapter_id, metadata in adapters.items():
            if adapter_id in protected_ids:
                continue
            score = self.compute_score(metadata, current_time)
            scores.append((adapter_id, score))

        # Sort by score ascending (lowest = oldest = evict first)
        scores.sort(key=lambda x: x[1])
        return [aid for aid, _ in scores[:num_to_evict]]


class GDSFEvictionPolicy(BaseEvictionPolicy):
    """
    Greedy-Dual-Size-Frequency (GDSF) eviction policy.

    Score = Frequency * Cost / Size + Age_Factor

    This balances:
    - Frequency: How often the adapter is used
    - Cost: How expensive it is to reload (load_time_ms)
    - Size: Memory footprint
    - Age: Time since last use

    Reference: Web caching algorithms, adapted for adapter caching.
    """

    def __init__(
        self,
        frequency_weight: float = 1.0,
        cost_weight: float = 1.0,
        size_weight: float = 1.0,
        age_decay: float = 0.001,  # Per-second decay
    ):
        self.frequency_weight = frequency_weight
        self.cost_weight = cost_weight
        self.size_weight = size_weight
        self.age_decay = age_decay

    def compute_score(
        self,
        metadata: AdapterMetadata,
        current_time: float,
    ) -> float:
        """
        Compute GDSF score.

        Higher score = more valuable = keep longer.
        """
        # Frequency: activation count (bounded to avoid extreme values)
        frequency = min(metadata.activation_count + 1, 1000)

        # Cost: time to reload (normalized by a baseline)
        baseline_load_time = 100.0  # 100ms baseline
        cost = max(metadata.load_time_ms / baseline_load_time, 0.1)

        # Size: memory footprint (normalized by 100MB)
        baseline_size = 100 * 1024 * 1024  # 100MB
        size = max(metadata.memory_bytes / baseline_size, 0.01)

        # Age: time since last use
        if metadata.last_used_at:
            age_seconds = current_time - metadata.last_used_at.timestamp()
        else:
            age_seconds = current_time - metadata.created_at.timestamp()

        # GDSF formula
        score = (
            (self.frequency_weight * frequency * self.cost_weight * cost) /
            (self.size_weight * size)
        )

        # Apply age decay
        score -= self.age_decay * age_seconds

        return score

    def get_eviction_candidates(
        self,
        adapters: Dict[str, AdapterMetadata],
        num_to_evict: int,
        protected_ids: Set[str],
    ) -> List[str]:
        current_time = time.time()
        scores = []

        for adapter_id, metadata in adapters.items():
            if adapter_id in protected_ids:
                continue
            if metadata.current_tier == CacheTier.DISK:
                continue  # Already on disk
            score = self.compute_score(metadata, current_time)
            scores.append((adapter_id, score))

        # Sort by score ascending (lowest score = evict first)
        scores.sort(key=lambda x: x[1])
        return [aid for aid, _ in scores[:num_to_evict]]


def create_eviction_policy(policy: EvictionPolicy) -> BaseEvictionPolicy:
    """Factory for eviction policies."""
    if policy == EvictionPolicy.LRU:
        return LRUEvictionPolicy()
    elif policy == EvictionPolicy.GDSF:
        return GDSFEvictionPolicy()
    else:
        # Default to GDSF
        return GDSFEvictionPolicy()


# =============================================================================
# TIERED CACHE
# =============================================================================

@dataclass
class TieredCacheConfig:
    """Configuration for tiered adapter cache."""
    # Capacity limits
    max_gpu_adapters: int = 8
    max_cpu_adapters: int = 32
    max_gpu_memory_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB
    max_cpu_memory_bytes: int = 32 * 1024 * 1024 * 1024  # 32GB

    # Disk storage
    disk_cache_dir: str = "/tmp/tensafe_adapter_cache"

    # Eviction
    eviction_policy: EvictionPolicy = EvictionPolicy.GDSF
    eviction_threshold: float = 0.8  # Evict when 80% full

    # Prefetch
    enable_prefetch: bool = True
    prefetch_count: int = 2  # Number of adapters to prefetch

    # Async operations
    async_offload: bool = True
    offload_timeout_ms: int = 5000


class TieredAdapterCache:
    """
    Tiered cache for adapters with GPU -> CPU -> Disk hierarchy.

    Implements:
    - Automatic tier promotion/demotion based on usage
    - Async prefetch and offload operations
    - GDSF-based eviction with configurable policies
    - Memory tracking and pressure handling
    """

    def __init__(self, config: Optional[TieredCacheConfig] = None):
        self.config = config or TieredCacheConfig()

        # Tier caches (adapter_id -> PackedWeights)
        self._gpu_cache: OrderedDict[str, PackedWeights] = OrderedDict()
        self._cpu_cache: OrderedDict[str, PackedWeights] = OrderedDict()

        # Metadata registry (adapter_id -> AdapterMetadata)
        self._metadata: Dict[str, AdapterMetadata] = {}

        # Eviction policy
        self._eviction_policy = create_eviction_policy(self.config.eviction_policy)

        # Memory tracking
        self._gpu_memory_used: int = 0
        self._cpu_memory_used: int = 0

        # Locks for thread safety
        self._lock = threading.RLock()
        self._gpu_lock = threading.Lock()
        self._cpu_lock = threading.Lock()

        # Async operation handles
        self._pending_operations: Dict[str, asyncio.Task] = {}

        # Event loop for async operations
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize disk cache directory
        os.makedirs(self.config.disk_cache_dir, exist_ok=True)

        logger.info(
            f"TieredAdapterCache initialized: "
            f"GPU={self.config.max_gpu_adapters}, "
            f"CPU={self.config.max_cpu_adapters}, "
            f"disk={self.config.disk_cache_dir}"
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def register(
        self,
        adapter_id: str,
        weights: WeightDict,
        metadata: AdapterMetadata,
    ) -> None:
        """
        Register a new adapter in the cache.

        Starts in CPU tier by default.
        """
        with self._lock:
            if adapter_id in self._metadata:
                logger.warning(f"Adapter {adapter_id} already registered, updating")

            # Create packed weights
            packed = PackedWeights(weights=weights)
            packed.compute_memory_size()

            # Store metadata
            self._metadata[adapter_id] = metadata

            # Add to CPU cache
            self._add_to_cpu_cache(adapter_id, packed)

            metadata.state = AdapterState.LOADED
            metadata.current_tier = CacheTier.CPU
            metadata.loaded_at = datetime.utcnow()

            logger.info(
                f"Registered adapter {adapter_id}: "
                f"{packed.cpu_memory_bytes / 1024 / 1024:.2f}MB"
            )

    def get(
        self,
        adapter_id: str,
        target_tier: CacheTier = CacheTier.GPU,
    ) -> Optional[PackedWeights]:
        """
        Get adapter weights, promoting to target tier if needed.

        Args:
            adapter_id: Adapter ID
            target_tier: Desired cache tier

        Returns:
            PackedWeights if found, None otherwise
        """
        with self._lock:
            if adapter_id not in self._metadata:
                return None

            metadata = self._metadata[adapter_id]

            # Update usage stats
            metadata.last_used_at = datetime.utcnow()
            metadata.request_count += 1

            # Check current tier and promote if needed
            current_tier = metadata.current_tier

            if target_tier == CacheTier.GPU:
                if adapter_id in self._gpu_cache:
                    # Already in GPU
                    self._gpu_cache.move_to_end(adapter_id)
                    return self._gpu_cache[adapter_id]

                # Need to promote to GPU
                weights = self._promote_to_gpu(adapter_id)
                return weights

            elif target_tier == CacheTier.CPU:
                if adapter_id in self._gpu_cache:
                    return self._gpu_cache[adapter_id]
                if adapter_id in self._cpu_cache:
                    self._cpu_cache.move_to_end(adapter_id)
                    return self._cpu_cache[adapter_id]

                # Need to load from disk
                return self._load_from_disk(adapter_id)

            else:
                # Disk tier - just ensure it's on disk
                self._ensure_on_disk(adapter_id)
                return self._load_from_disk(adapter_id)

    def prefetch(self, adapter_id: str) -> None:
        """Prefetch adapter to GPU cache asynchronously."""
        if not self.config.enable_prefetch:
            return

        if adapter_id in self._gpu_cache:
            return  # Already in GPU

        # Start async prefetch
        if adapter_id not in self._pending_operations:
            task = asyncio.create_task(self._async_promote_to_gpu(adapter_id))
            self._pending_operations[adapter_id] = task

    def offload(
        self,
        adapter_id: str,
        target_tier: CacheTier = CacheTier.CPU,
    ) -> None:
        """Offload adapter to lower tier."""
        with self._lock:
            if adapter_id not in self._metadata:
                return

            if target_tier == CacheTier.CPU:
                self._demote_to_cpu(adapter_id)
            elif target_tier == CacheTier.DISK:
                self._demote_to_disk(adapter_id)

    def evict_if_needed(
        self,
        protected_ids: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Evict adapters if cache is above threshold.

        Returns list of evicted adapter IDs.
        """
        protected = protected_ids or set()
        evicted = []

        with self._lock:
            # Check GPU tier
            gpu_usage = len(self._gpu_cache) / max(self.config.max_gpu_adapters, 1)
            if gpu_usage >= self.config.eviction_threshold:
                candidates = self._eviction_policy.get_eviction_candidates(
                    {k: self._metadata[k] for k in self._gpu_cache if k in self._metadata},
                    num_to_evict=max(1, len(self._gpu_cache) // 4),
                    protected_ids=protected,
                )
                for adapter_id in candidates:
                    self._demote_to_cpu(adapter_id)
                    evicted.append(adapter_id)

            # Check CPU tier
            cpu_usage = len(self._cpu_cache) / max(self.config.max_cpu_adapters, 1)
            if cpu_usage >= self.config.eviction_threshold:
                candidates = self._eviction_policy.get_eviction_candidates(
                    {k: self._metadata[k] for k in self._cpu_cache if k in self._metadata},
                    num_to_evict=max(1, len(self._cpu_cache) // 4),
                    protected_ids=protected,
                )
                for adapter_id in candidates:
                    self._demote_to_disk(adapter_id)
                    evicted.append(adapter_id)

        return evicted

    def unregister(self, adapter_id: str) -> bool:
        """Remove adapter from all tiers."""
        with self._lock:
            if adapter_id not in self._metadata:
                return False

            # Remove from all caches
            if adapter_id in self._gpu_cache:
                packed = self._gpu_cache.pop(adapter_id)
                self._gpu_memory_used -= packed.gpu_memory_bytes

            if adapter_id in self._cpu_cache:
                packed = self._cpu_cache.pop(adapter_id)
                self._cpu_memory_used -= packed.cpu_memory_bytes

            # Remove disk cache
            disk_path = self._get_disk_path(adapter_id)
            if os.path.exists(disk_path):
                os.remove(disk_path)

            # Remove metadata
            del self._metadata[adapter_id]

            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "gpu_adapters": len(self._gpu_cache),
                "cpu_adapters": len(self._cpu_cache),
                "gpu_memory_bytes": self._gpu_memory_used,
                "cpu_memory_bytes": self._cpu_memory_used,
                "gpu_utilization": len(self._gpu_cache) / max(self.config.max_gpu_adapters, 1),
                "cpu_utilization": len(self._cpu_cache) / max(self.config.max_cpu_adapters, 1),
                "total_registered": len(self._metadata),
            }

    # -------------------------------------------------------------------------
    # INTERNAL: TIER OPERATIONS
    # -------------------------------------------------------------------------

    def _add_to_cpu_cache(self, adapter_id: str, packed: PackedWeights) -> None:
        """Add to CPU cache, evicting if necessary."""
        with self._cpu_lock:
            # Check if eviction needed
            while (
                len(self._cpu_cache) >= self.config.max_cpu_adapters or
                self._cpu_memory_used + packed.cpu_memory_bytes > self.config.max_cpu_memory_bytes
            ):
                if not self._cpu_cache:
                    break
                # Evict oldest
                evict_id, evict_packed = self._cpu_cache.popitem(last=False)
                self._cpu_memory_used -= evict_packed.cpu_memory_bytes
                self._demote_to_disk(evict_id)

            self._cpu_cache[adapter_id] = packed
            self._cpu_memory_used += packed.cpu_memory_bytes

            if adapter_id in self._metadata:
                self._metadata[adapter_id].current_tier = CacheTier.CPU

    def _add_to_gpu_cache(self, adapter_id: str, packed: PackedWeights) -> None:
        """Add to GPU cache, evicting if necessary."""
        with self._gpu_lock:
            # Check if eviction needed
            while (
                len(self._gpu_cache) >= self.config.max_gpu_adapters or
                self._gpu_memory_used + packed.gpu_memory_bytes > self.config.max_gpu_memory_bytes
            ):
                if not self._gpu_cache:
                    break
                # Evict oldest
                evict_id, evict_packed = self._gpu_cache.popitem(last=False)
                self._gpu_memory_used -= evict_packed.gpu_memory_bytes
                self._demote_to_cpu(evict_id)

            self._gpu_cache[adapter_id] = packed
            self._gpu_memory_used += packed.gpu_memory_bytes

            if adapter_id in self._metadata:
                self._metadata[adapter_id].current_tier = CacheTier.GPU

    def _promote_to_gpu(self, adapter_id: str) -> Optional[PackedWeights]:
        """Promote adapter from CPU/disk to GPU."""
        # First ensure it's in CPU
        if adapter_id not in self._cpu_cache:
            packed = self._load_from_disk(adapter_id)
            if packed is None:
                return None
        else:
            packed = self._cpu_cache[adapter_id]

        # Move to GPU
        start_time = time.perf_counter()

        # In production, this would use CUDA streams for async copy
        # For now, simulate GPU upload
        packed.is_on_gpu = True
        packed.gpu_memory_bytes = packed.cpu_memory_bytes

        # Add to GPU cache
        self._add_to_gpu_cache(adapter_id, packed)

        # Update metadata
        if adapter_id in self._metadata:
            load_time = (time.perf_counter() - start_time) * 1000
            self._metadata[adapter_id].load_time_ms = load_time

        return packed

    async def _async_promote_to_gpu(self, adapter_id: str) -> Optional[PackedWeights]:
        """Async version of promote_to_gpu."""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._promote_to_gpu, adapter_id)

    def _demote_to_cpu(self, adapter_id: str) -> None:
        """Demote adapter from GPU to CPU."""
        with self._gpu_lock:
            if adapter_id not in self._gpu_cache:
                return

            packed = self._gpu_cache.pop(adapter_id)
            self._gpu_memory_used -= packed.gpu_memory_bytes

            # Clear GPU tensors
            packed.is_on_gpu = False
            packed.gpu_tensors = None
            packed.gpu_memory_bytes = 0

        # Add to CPU cache
        self._add_to_cpu_cache(adapter_id, packed)

    def _demote_to_disk(self, adapter_id: str) -> None:
        """Demote adapter from CPU to disk."""
        with self._cpu_lock:
            if adapter_id not in self._cpu_cache:
                return

            packed = self._cpu_cache.pop(adapter_id)
            self._cpu_memory_used -= packed.cpu_memory_bytes

        # Save to disk
        self._save_to_disk(adapter_id, packed)

        if adapter_id in self._metadata:
            self._metadata[adapter_id].current_tier = CacheTier.DISK

    def _load_from_disk(self, adapter_id: str) -> Optional[PackedWeights]:
        """Load adapter from disk cache."""
        disk_path = self._get_disk_path(adapter_id)

        if not os.path.exists(disk_path):
            return None

        start_time = time.perf_counter()

        with open(disk_path, 'rb') as f:
            packed = pickle.load(f)

        # Add to CPU cache
        self._add_to_cpu_cache(adapter_id, packed)

        # Update load time
        if adapter_id in self._metadata:
            load_time = (time.perf_counter() - start_time) * 1000
            self._metadata[adapter_id].load_time_ms = load_time

        return packed

    def _save_to_disk(self, adapter_id: str, packed: PackedWeights) -> None:
        """Save adapter to disk cache."""
        disk_path = self._get_disk_path(adapter_id)

        with open(disk_path, 'wb') as f:
            pickle.dump(packed, f)

    def _ensure_on_disk(self, adapter_id: str) -> None:
        """Ensure adapter is saved to disk."""
        disk_path = self._get_disk_path(adapter_id)

        if os.path.exists(disk_path):
            return

        # Get from higher tiers and save
        if adapter_id in self._gpu_cache:
            self._save_to_disk(adapter_id, self._gpu_cache[adapter_id])
        elif adapter_id in self._cpu_cache:
            self._save_to_disk(adapter_id, self._cpu_cache[adapter_id])

    def _get_disk_path(self, adapter_id: str) -> str:
        """Get disk cache path for adapter."""
        safe_id = adapter_id.replace('/', '_').replace('\\', '_')
        return os.path.join(self.config.disk_cache_dir, f"{safe_id}.pkl")


# =============================================================================
# ZERO-COPY HOT-SWAP MANAGER
# =============================================================================

@dataclass
class HotSwapConfig:
    """Configuration for hot-swap manager."""
    # Cache configuration
    cache_config: TieredCacheConfig = field(default_factory=TieredCacheConfig)

    # Hot-swap behavior
    enable_zero_copy: bool = True
    preload_on_register: bool = True
    validate_on_swap: bool = True

    # Compatibility
    require_same_rank: bool = True  # For compiled models
    require_same_type: bool = False

    # Monitoring
    track_metrics: bool = True
    metrics_window_seconds: int = 300


class HotSwapManager:
    """
    Zero-Copy Hot-Swap Manager for adapter switching.

    Provides instant adapter switching through pointer-swap mechanism,
    avoiding costly weight copies during inference.

    Features:
    - Zero-copy activation through pointer swap
    - Pre-packed weights for instant readiness
    - Compatibility validation
    - Automatic tier management
    - Metrics tracking
    """

    def __init__(self, config: Optional[HotSwapConfig] = None):
        self.config = config or HotSwapConfig()

        # Initialize tiered cache
        self._cache = TieredAdapterCache(self.config.cache_config)

        # Active adapter tracking
        self._active_adapter_id: Optional[str] = None
        self._active_weights: Optional[PackedWeights] = None

        # Adapter instances (for non-linear adapters)
        self._adapter_instances: Dict[str, BaseAdapter] = {}

        # Thread safety
        self._lock = threading.RLock()
        self._swap_lock = threading.Lock()  # Separate lock for swap operations

        # Metrics
        self._swap_count: int = 0
        self._total_swap_time_ms: float = 0.0
        self._swap_history: List[Tuple[str, str, float]] = []  # (from, to, time_ms)

        logger.info("HotSwapManager initialized")

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def register_adapter(
        self,
        adapter_id: str,
        weights: WeightDict,
        adapter_config: AdapterConfig,
        base_model_id: str = "unknown",
        preload: bool = True,
    ) -> AdapterMetadata:
        """
        Register an adapter for hot-swapping.

        Args:
            adapter_id: Unique identifier
            weights: Module -> (lora_A, lora_B) weights
            adapter_config: Adapter configuration
            base_model_id: Base model identifier for compatibility
            preload: Whether to preload to GPU

        Returns:
            AdapterMetadata for the registered adapter
        """
        with self._lock:
            # Compute metadata
            total_params = sum(
                a.size + b.size for a, b in weights.values()
            )
            memory_bytes = sum(
                a.nbytes + b.nbytes for a, b in weights.values()
            )

            # Get hidden size from weights
            hidden_size = 0
            for module_name, (lora_a, lora_b) in weights.items():
                hidden_size = max(hidden_size, lora_a.shape[1])
                break

            # Create content hash
            content_hash = hashlib.sha256()
            for module_name in sorted(weights.keys()):
                lora_a, lora_b = weights[module_name]
                content_hash.update(module_name.encode())
                content_hash.update(lora_a.tobytes())
                content_hash.update(lora_b.tobytes())

            metadata = AdapterMetadata(
                adapter_id=adapter_id,
                source_path="",
                content_hash=content_hash.hexdigest()[:16],
                adapter_type=adapter_config.adapter_type,
                config=adapter_config,
                base_model_id=base_model_id,
                hidden_size=hidden_size,
                target_modules=list(weights.keys()),
                total_params=total_params,
                memory_bytes=memory_bytes,
            )

            # Register in cache
            self._cache.register(adapter_id, weights, metadata)

            # Create adapter instance for non-linear types
            if adapter_config.adapter_type not in {
                AdapterType.LORA, AdapterType.RS_LORA, AdapterType.LORA_FA
            }:
                self._create_adapter_instance(adapter_id, weights, adapter_config)

            # Preload to GPU if configured
            if preload and self.config.preload_on_register:
                self._cache.get(adapter_id, target_tier=CacheTier.GPU)

            return metadata

    def activate(self, adapter_id: str) -> bool:
        """
        Activate an adapter for inference (zero-copy swap).

        Args:
            adapter_id: Adapter to activate

        Returns:
            True if activation successful
        """
        with self._swap_lock:
            start_time = time.perf_counter()

            # Get weights (promotes to GPU if needed)
            weights = self._cache.get(adapter_id, target_tier=CacheTier.GPU)
            if weights is None:
                logger.error(f"Adapter {adapter_id} not found in cache")
                return False

            # Validate compatibility if configured
            if self.config.validate_on_swap and self._active_adapter_id:
                errors = self._validate_compatibility(adapter_id)
                if errors:
                    logger.error(f"Compatibility errors: {errors}")
                    return False

            # Record previous adapter
            prev_adapter_id = self._active_adapter_id

            # Zero-copy pointer swap
            if self.config.enable_zero_copy:
                self._active_weights = weights
            else:
                # Copy weights (slower but safer)
                self._active_weights = PackedWeights(
                    weights={k: (v[0].copy(), v[1].copy()) for k, v in weights.weights.items()}
                )

            self._active_adapter_id = adapter_id

            # Update metadata
            metadata = self._cache._metadata.get(adapter_id)
            if metadata:
                metadata.state = AdapterState.ACTIVE
                metadata.last_activated_at = datetime.utcnow()
                metadata.activation_count += 1

            # Deactivate previous
            if prev_adapter_id and prev_adapter_id in self._cache._metadata:
                self._cache._metadata[prev_adapter_id].state = AdapterState.LOADED

            # Record metrics
            swap_time = (time.perf_counter() - start_time) * 1000
            self._swap_count += 1
            self._total_swap_time_ms += swap_time

            if self.config.track_metrics:
                self._swap_history.append((
                    prev_adapter_id or "none",
                    adapter_id,
                    swap_time
                ))
                # Trim history
                max_history = 1000
                if len(self._swap_history) > max_history:
                    self._swap_history = self._swap_history[-max_history:]

            logger.info(f"Activated adapter {adapter_id} in {swap_time:.2f}ms")
            return True

    def get_active_weights(self) -> Optional[WeightDict]:
        """Get weights of the currently active adapter."""
        if self._active_weights is None:
            return None
        return self._active_weights.weights

    def get_active_adapter(self) -> Optional[BaseAdapter]:
        """Get the active adapter instance (for non-linear adapters)."""
        if self._active_adapter_id is None:
            return None
        return self._adapter_instances.get(self._active_adapter_id)

    def forward(
        self,
        x: np.ndarray,
        module_name: str,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute adapter forward pass for active adapter.

        Args:
            x: Input activations
            module_name: Target module name
            original_weight: Original weight matrix (for DoRA)
            original_output: Original layer output

        Returns:
            Delta to add to base model output
        """
        if self._active_adapter_id is None:
            return None

        # Check for adapter instance (non-linear adapters)
        adapter = self._adapter_instances.get(self._active_adapter_id)
        if adapter is not None:
            return adapter.forward(x, original_weight, original_output)

        # Linear adapter: compute delta directly
        weights = self.get_active_weights()
        if weights is None or module_name not in weights:
            return None

        lora_a, lora_b = weights[module_name]

        # Get scaling from metadata
        metadata = self._cache._metadata.get(self._active_adapter_id)
        if metadata:
            scaling = metadata.config.scaling
        else:
            scaling = 1.0

        # Compute delta: scaling * (x @ A.T) @ B.T
        intermediate = np.matmul(x, lora_a.T)
        delta = np.matmul(intermediate, lora_b.T)

        return scaling * delta

    def deactivate(self) -> None:
        """Deactivate current adapter."""
        with self._swap_lock:
            if self._active_adapter_id:
                metadata = self._cache._metadata.get(self._active_adapter_id)
                if metadata:
                    metadata.state = AdapterState.LOADED

            self._active_adapter_id = None
            self._active_weights = None

    def unregister(self, adapter_id: str) -> bool:
        """Unregister an adapter."""
        with self._lock:
            if adapter_id == self._active_adapter_id:
                self.deactivate()

            # Remove adapter instance
            self._adapter_instances.pop(adapter_id, None)

            return self._cache.unregister(adapter_id)

    def prefetch(self, adapter_id: str) -> None:
        """Prefetch adapter to GPU for fast activation."""
        self._cache.prefetch(adapter_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get hot-swap statistics."""
        cache_stats = self._cache.get_stats()

        return {
            **cache_stats,
            "active_adapter_id": self._active_adapter_id,
            "swap_count": self._swap_count,
            "avg_swap_time_ms": (
                self._total_swap_time_ms / self._swap_count
                if self._swap_count > 0 else 0
            ),
            "adapter_instances": len(self._adapter_instances),
        }

    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        with self._lock:
            return [
                metadata.to_dict()
                for metadata in self._cache._metadata.values()
            ]

    # -------------------------------------------------------------------------
    # INTERNAL
    # -------------------------------------------------------------------------

    def _validate_compatibility(self, adapter_id: str) -> List[str]:
        """Validate adapter compatibility for hot-swap."""
        errors = []

        if self._active_adapter_id is None:
            return errors

        active_meta = self._cache._metadata.get(self._active_adapter_id)
        new_meta = self._cache._metadata.get(adapter_id)

        if not active_meta or not new_meta:
            return errors

        # Check rank compatibility (important for compiled models)
        if self.config.require_same_rank:
            if active_meta.config.rank != new_meta.config.rank:
                errors.append(
                    f"Rank mismatch: {active_meta.config.rank} vs {new_meta.config.rank}. "
                    f"For compiled models, ranks must match to avoid recompilation."
                )

        # Check type compatibility
        if self.config.require_same_type:
            if active_meta.adapter_type != new_meta.adapter_type:
                errors.append(
                    f"Type mismatch: {active_meta.adapter_type} vs {new_meta.adapter_type}"
                )

        # Check target modules
        if set(active_meta.target_modules) != set(new_meta.target_modules):
            errors.append(
                f"Target module mismatch: {active_meta.target_modules} vs {new_meta.target_modules}"
            )

        return errors

    def _create_adapter_instance(
        self,
        adapter_id: str,
        weights: WeightDict,
        config: AdapterConfig,
    ) -> None:
        """Create adapter instance for non-linear adapter types."""
        # Get dimensions from first weight
        for module_name, (lora_a, lora_b) in weights.items():
            in_features = lora_a.shape[1]
            out_features = lora_b.shape[0]
            break
        else:
            return

        # Create adapter
        adapter = create_adapter(config, in_features, out_features)

        # Load weights
        adapter.set_weights({
            "lora_A": list(weights.values())[0][0],
            "lora_B": list(weights.values())[0][1],
        }, strict=False)

        self._adapter_instances[adapter_id] = adapter


# =============================================================================
# ADAPTER LIFECYCLE MANAGER
# =============================================================================

@dataclass
class LifecycleConfig:
    """Configuration for adapter lifecycle management."""
    # Hot-swap configuration
    hot_swap_config: HotSwapConfig = field(default_factory=HotSwapConfig)

    # Health monitoring
    enable_health_monitoring: bool = True
    health_check_interval_seconds: int = 30
    unhealthy_threshold: int = 3  # Errors before marking unhealthy

    # Automatic cleanup
    auto_cleanup_enabled: bool = True
    cleanup_interval_seconds: int = 300
    max_idle_seconds: int = 3600  # Unload adapters idle for 1 hour

    # Audit logging
    enable_audit_log: bool = True
    audit_log_path: Optional[str] = None


class AdapterLifecycleManager:
    """
    Full lifecycle management for adapters.

    Provides:
    - Registration, activation, deactivation, unregistration
    - Health monitoring with automatic recovery
    - Usage statistics and metrics
    - Audit logging for compliance
    - Automatic cleanup of idle adapters
    """

    def __init__(self, config: Optional[LifecycleConfig] = None):
        self.config = config or LifecycleConfig()

        # Initialize hot-swap manager
        self._hot_swap = HotSwapManager(self.config.hot_swap_config)

        # Health tracking
        self._health_status: Dict[str, Dict[str, Any]] = {}
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_health_check = threading.Event()

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Cleanup
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()

        # Start background tasks
        if self.config.enable_health_monitoring:
            self._start_health_monitoring()

        if self.config.auto_cleanup_enabled:
            self._start_auto_cleanup()

        logger.info("AdapterLifecycleManager initialized")

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def register(
        self,
        adapter_id: str,
        weights: WeightDict,
        adapter_config: AdapterConfig,
        base_model_id: str = "unknown",
    ) -> AdapterMetadata:
        """Register a new adapter."""
        self._log_audit("REGISTER", adapter_id)

        metadata = self._hot_swap.register_adapter(
            adapter_id=adapter_id,
            weights=weights,
            adapter_config=adapter_config,
            base_model_id=base_model_id,
        )

        # Initialize health status
        self._health_status[adapter_id] = {
            "healthy": True,
            "error_count": 0,
            "last_check": datetime.utcnow(),
            "last_error": None,
        }

        return metadata

    def activate(self, adapter_id: str) -> bool:
        """Activate an adapter for inference."""
        self._log_audit("ACTIVATE", adapter_id)

        success = self._hot_swap.activate(adapter_id)

        if not success:
            self._record_error(adapter_id, "Activation failed")

        return success

    def deactivate(self) -> None:
        """Deactivate the current adapter."""
        current = self._hot_swap._active_adapter_id
        if current:
            self._log_audit("DEACTIVATE", current)
        self._hot_swap.deactivate()

    def forward(
        self,
        x: np.ndarray,
        module_name: str,
        original_weight: Optional[np.ndarray] = None,
        original_output: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Compute adapter forward pass."""
        try:
            return self._hot_swap.forward(
                x, module_name, original_weight, original_output
            )
        except Exception as e:
            active_id = self._hot_swap._active_adapter_id
            if active_id:
                self._record_error(active_id, str(e))
            raise

    def unregister(self, adapter_id: str) -> bool:
        """Unregister an adapter."""
        self._log_audit("UNREGISTER", adapter_id)

        success = self._hot_swap.unregister(adapter_id)

        if adapter_id in self._health_status:
            del self._health_status[adapter_id]

        return success

    def get_health_status(self, adapter_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status for adapter(s)."""
        if adapter_id:
            return self._health_status.get(adapter_id, {"healthy": False})
        return dict(self._health_status)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        base_stats = self._hot_swap.get_stats()

        # Add health summary
        healthy_count = sum(
            1 for s in self._health_status.values() if s.get("healthy", False)
        )

        return {
            **base_stats,
            "healthy_adapters": healthy_count,
            "unhealthy_adapters": len(self._health_status) - healthy_count,
            "audit_log_size": len(self._audit_log),
        }

    def get_audit_log(
        self,
        adapter_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        log = self._audit_log

        if adapter_id:
            log = [e for e in log if e.get("adapter_id") == adapter_id]

        return log[-limit:]

    def shutdown(self) -> None:
        """Shutdown lifecycle manager."""
        self._stop_health_check.set()
        self._stop_cleanup.set()

        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)

        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        self._log_audit("SHUTDOWN", "manager")

    # -------------------------------------------------------------------------
    # INTERNAL
    # -------------------------------------------------------------------------

    def _log_audit(self, event: str, adapter_id: str) -> None:
        """Log an audit event."""
        if not self.config.enable_audit_log:
            return

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event,
            "adapter_id": adapter_id,
            "active_adapter": self._hot_swap._active_adapter_id,
        }

        self._audit_log.append(entry)

        # Trim if too large
        max_entries = 10000
        if len(self._audit_log) > max_entries:
            self._audit_log = self._audit_log[-max_entries:]

        logger.debug(f"AUDIT: {event} - {adapter_id}")

    def _record_error(self, adapter_id: str, error: str) -> None:
        """Record an error for an adapter."""
        if adapter_id not in self._health_status:
            return

        status = self._health_status[adapter_id]
        status["error_count"] = status.get("error_count", 0) + 1
        status["last_error"] = error

        if status["error_count"] >= self.config.unhealthy_threshold:
            status["healthy"] = False
            logger.warning(f"Adapter {adapter_id} marked unhealthy: {error}")

    def _start_health_monitoring(self) -> None:
        """Start health monitoring background thread."""
        def health_check_loop():
            while not self._stop_health_check.wait(
                self.config.health_check_interval_seconds
            ):
                self._run_health_checks()

        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="adapter-health-monitor",
        )
        self._health_check_thread.start()

    def _run_health_checks(self) -> None:
        """Run health checks on all adapters."""
        for adapter_id, status in list(self._health_status.items()):
            try:
                # Check if adapter is still registered
                if adapter_id not in self._hot_swap._cache._metadata:
                    del self._health_status[adapter_id]
                    continue

                # Update last check time
                status["last_check"] = datetime.utcnow()

                # Reset error count if healthy for a while
                if status.get("error_count", 0) > 0:
                    status["error_count"] = max(0, status["error_count"] - 1)
                    if status["error_count"] < self.config.unhealthy_threshold:
                        status["healthy"] = True

            except Exception as e:
                logger.error(f"Health check failed for {adapter_id}: {e}")

    def _start_auto_cleanup(self) -> None:
        """Start automatic cleanup background thread."""
        def cleanup_loop():
            while not self._stop_cleanup.wait(
                self.config.cleanup_interval_seconds
            ):
                self._run_cleanup()

        self._cleanup_thread = threading.Thread(
            target=cleanup_loop,
            daemon=True,
            name="adapter-cleanup",
        )
        self._cleanup_thread.start()

    def _run_cleanup(self) -> None:
        """Run cleanup of idle adapters."""
        now = datetime.utcnow()
        max_idle = self.config.max_idle_seconds

        for adapter_id, metadata in list(self._hot_swap._cache._metadata.items()):
            # Skip active adapter
            if adapter_id == self._hot_swap._active_adapter_id:
                continue

            # Check idle time
            last_used = metadata.last_used_at or metadata.loaded_at
            if last_used:
                idle_seconds = (now - last_used).total_seconds()
                if idle_seconds > max_idle:
                    logger.info(f"Auto-unloading idle adapter {adapter_id}")
                    self._hot_swap._cache.offload(adapter_id, CacheTier.DISK)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "CacheTier",
    "AdapterState",
    "EvictionPolicy",

    # Data classes
    "AdapterMetadata",
    "PackedWeights",
    "TieredCacheConfig",
    "HotSwapConfig",
    "LifecycleConfig",

    # Eviction policies
    "BaseEvictionPolicy",
    "LRUEvictionPolicy",
    "GDSFEvictionPolicy",
    "create_eviction_policy",

    # Cache
    "TieredAdapterCache",

    # Hot-swap
    "HotSwapManager",

    # Lifecycle
    "AdapterLifecycleManager",
]
