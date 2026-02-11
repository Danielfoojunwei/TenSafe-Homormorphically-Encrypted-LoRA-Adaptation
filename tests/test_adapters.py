"""
Comprehensive Tests for TenSafe Adapters Module.

Tests cover:
- All adapter types (LoRA, DoRA, AdaLoRA, VeRA, rsLoRA, Gated, GLoRA)
- Placement resolution for multiple architectures
- Fused QKV projection handling
- Tiered caching and eviction policies
- Zero-copy hot-swap mechanism
- Lifecycle management

Author: TenSafe Team
"""

import os
import shutil
import tempfile
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pytest

from tensafe.adapters import (
    ARCHITECTURE_CONFIGS,
    AdaLoRAAdapter,
    AdapterConfig,
    AdapterLifecycleManager,
    AdapterMetadata,
    AdapterPlacementResolver,
    AdapterState,
    # Adapter types
    AdapterType,
    BaseAdapter,
    # Hot-swap
    CacheTier,
    DoRAAdapter,
    DoRAHook,
    EvictionPolicy,
    FusedProjectionHandler,
    FusedQKVHook,
    GatedAdapterHook,
    GatedLoRAAdapter,
    GDSFEvictionPolicy,
    GLoRAAdapter,
    HookConfig,
    HookManager,
    HotSwapConfig,
    HotSwapManager,
    # Hooks
    InjectionMode,
    # Placement
    LayerType,
    LifecycleConfig,
    LinearAdapterHook,
    LoRAAdapter,
    LoRAFAAdapter,
    LRUEvictionPolicy,
    PackedWeights,
    PlacementConfig,
    ProjectionType,
    ScalingStrategy,
    TargetScope,
    TieredAdapterCache,
    TieredCacheConfig,
    VeRAAdapter,
    convert_adapter,
    create_adapter,
    create_eviction_policy,
    get_architecture_config,
    get_layer_importance_weights,
    list_adapter_types,
    rsLoRAAdapter,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_weights() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Create sample LoRA weights for testing."""
    rank = 16
    hidden_size = 128

    rng = np.random.default_rng(42)

    weights = {}
    for module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        lora_a = rng.standard_normal((rank, hidden_size)).astype(np.float32)
        lora_b = np.zeros((hidden_size, rank), dtype=np.float32)
        weights[module] = (lora_a, lora_b)

    return weights


@pytest.fixture
def sample_config() -> AdapterConfig:
    """Create sample adapter configuration."""
    return AdapterConfig(
        adapter_type=AdapterType.LORA,
        rank=16,
        alpha=32.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


@pytest.fixture
def temp_cache_dir():
    """Create temporary directory for cache tests."""
    dir_path = tempfile.mkdtemp(prefix="tensafe_test_cache_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


# =============================================================================
# ADAPTER TYPES TESTS
# =============================================================================

class TestAdapterTypes:
    """Tests for adapter type implementations."""

    def test_lora_adapter_creation(self):
        """Test basic LoRA adapter creation and initialization."""
        config = AdapterConfig(adapter_type=AdapterType.LORA, rank=8)
        adapter = create_adapter(config, in_features=256, out_features=256)

        assert isinstance(adapter, LoRAAdapter)
        assert adapter.config.rank == 8
        assert "lora_A" in adapter.get_trainable_params()
        assert "lora_B" in adapter.get_trainable_params()

    def test_lora_forward_pass(self):
        """Test LoRA forward computation."""
        config = AdapterConfig(adapter_type=AdapterType.LORA, rank=4, alpha=8.0)
        adapter = create_adapter(config, in_features=32, out_features=32)

        x = np.random.randn(2, 32).astype(np.float32)
        delta = adapter.forward(x)

        assert delta.shape == x.shape
        # Initially B is zeros, so delta should be zeros
        np.testing.assert_array_almost_equal(delta, np.zeros_like(delta))

    def test_rs_lora_scaling(self):
        """Test rsLoRA uses rank-stabilized scaling."""
        config = AdapterConfig(adapter_type=AdapterType.RS_LORA, rank=16, alpha=32.0)
        adapter = create_adapter(config, in_features=128, out_features=128)

        expected_scaling = 32.0 / np.sqrt(16)
        assert abs(adapter.config.scaling - expected_scaling) < 1e-6

    def test_lora_fa_frozen_a(self):
        """Test LoRA-FA freezes A matrix."""
        config = AdapterConfig(adapter_type=AdapterType.LORA_FA, rank=8)
        adapter = create_adapter(config, in_features=64, out_features=64)

        assert "lora_A" in adapter.get_frozen_params()
        assert "lora_B" in adapter.get_trainable_params()
        assert "lora_A" not in adapter.get_trainable_params()

    def test_dora_magnitude_initialization(self):
        """Test DoRA magnitude vector initialization."""
        config = AdapterConfig(adapter_type=AdapterType.DORA, rank=8)
        adapter = create_adapter(config, in_features=64, out_features=64)

        assert "magnitude" in adapter.get_trainable_params()
        assert "lora_A" in adapter.get_trainable_params()
        assert "lora_B" in adapter.get_trainable_params()

    def test_dora_forward_with_original_weight(self):
        """Test DoRA forward with original weight provided."""
        config = AdapterConfig(adapter_type=AdapterType.DORA, rank=8)
        adapter = create_adapter(config, in_features=64, out_features=64)

        x = np.random.randn(2, 64).astype(np.float32)
        original_weight = np.random.randn(64, 64).astype(np.float32)

        delta = adapter.forward(x, original_weight=original_weight)

        assert delta.shape == x.shape

    def test_vera_shared_matrices(self):
        """Test VeRA shares matrices across instances."""
        config = AdapterConfig(
            adapter_type=AdapterType.VERA,
            rank=32,
            projection_seed=42,
        )

        # Reset shared matrices
        VeRAAdapter.reset_shared_matrices()

        adapter1 = create_adapter(config, in_features=64, out_features=64)
        adapter2 = create_adapter(config, in_features=64, out_features=64)

        # Should share the same underlying matrices
        np.testing.assert_array_equal(
            adapter1._frozen_params["lora_A"],
            adapter2._frozen_params["lora_A"]
        )

    def test_vera_parameter_efficiency(self):
        """Test VeRA has fewer trainable parameters than LoRA."""
        rank = 32
        in_features = 256
        out_features = 256

        lora_config = AdapterConfig(adapter_type=AdapterType.LORA, rank=rank)
        lora = create_adapter(lora_config, in_features, out_features)

        vera_config = AdapterConfig(adapter_type=AdapterType.VERA, rank=rank)
        VeRAAdapter.reset_shared_matrices()
        vera = create_adapter(vera_config, in_features, out_features)

        # VeRA should have far fewer trainable params
        assert vera.get_param_count() < lora.get_param_count() / 5

    def test_adalora_svd_parameterization(self):
        """Test AdaLoRA SVD parameterization."""
        config = AdapterConfig(
            adapter_type=AdapterType.ADALORA,
            rank=8,
            initial_rank=16,
        )
        adapter = create_adapter(config, in_features=64, out_features=64)

        assert "P" in adapter.get_trainable_params()
        assert "Lambda" in adapter.get_trainable_params()
        assert "Q" in adapter.get_trainable_params()

        # Initial rank should be used
        assert adapter._trainable_params["Lambda"].shape[0] == 16

    def test_adalora_pruning(self):
        """Test AdaLoRA rank pruning."""
        config = AdapterConfig(
            adapter_type=AdapterType.ADALORA,
            rank=4,
            initial_rank=8,
        )
        adapter = create_adapter(config, in_features=64, out_features=64)

        # Simulate importance scores
        adapter._importance_scores = np.array([0.1, 0.5, 0.2, 0.8, 0.3, 0.9, 0.4, 0.7])

        # Prune to target rank
        adapter.prune_to_budget(4)

        assert adapter.get_effective_rank() == 4

    def test_gated_lora_gate_computation(self):
        """Test Gated LoRA gate computation."""
        config = AdapterConfig(
            adapter_type=AdapterType.GATED_LORA,
            rank=8,
            gate_type="sigmoid",
        )
        adapter = create_adapter(config, in_features=64, out_features=64)

        x = np.random.randn(2, 64).astype(np.float32)
        delta = adapter.forward(x)

        assert delta.shape == x.shape

    def test_glora_support_tensors(self):
        """Test GLoRA support tensor configuration."""
        config = AdapterConfig(
            adapter_type=AdapterType.GLORA,
            rank=8,
            use_weight_shift=True,
            use_weight_scaling=True,
        )
        adapter = create_adapter(config, in_features=64, out_features=64)

        assert "shift_A" in adapter.get_trainable_params()
        assert "shift_B" in adapter.get_trainable_params()
        assert "scale_weight" in adapter.get_trainable_params()

    def test_adapter_factory(self):
        """Test adapter factory creates correct types."""
        types_to_test = [
            (AdapterType.LORA, LoRAAdapter),
            (AdapterType.RS_LORA, rsLoRAAdapter),
            (AdapterType.LORA_FA, LoRAFAAdapter),
            (AdapterType.DORA, DoRAAdapter),
            (AdapterType.ADALORA, AdaLoRAAdapter),
            (AdapterType.GATED_LORA, GatedLoRAAdapter),
            (AdapterType.GLORA, GLoRAAdapter),
        ]

        for adapter_type, expected_class in types_to_test:
            config = AdapterConfig(adapter_type=adapter_type, rank=8)
            VeRAAdapter.reset_shared_matrices()
            adapter = create_adapter(config, in_features=64, out_features=64)
            assert isinstance(adapter, expected_class), f"Expected {expected_class} for {adapter_type}"

    def test_list_adapter_types(self):
        """Test listing available adapter types."""
        types = list_adapter_types()

        assert AdapterType.LORA in types
        assert AdapterType.DORA in types
        assert len(types) >= 7


# =============================================================================
# PLACEMENT RESOLUTION TESTS
# =============================================================================

class TestPlacementResolution:
    """Tests for adapter placement resolution."""

    def test_architecture_configs_exist(self):
        """Test that architecture configs are defined for major models."""
        expected_archs = ["llama", "mistral", "falcon", "gpt2", "t5", "bert"]

        for arch in expected_archs:
            assert arch in ARCHITECTURE_CONFIGS, f"Missing config for {arch}"

    def test_llama_config(self):
        """Test LLaMA architecture configuration."""
        config = ARCHITECTURE_CONFIGS["llama"]

        assert config.layer_path == "model.layers"
        assert config.attention_module == "self_attn"
        assert config.q_proj == "q_proj"
        assert config.qkv_fused is None  # LLaMA has separate Q, K, V

    def test_falcon_fused_qkv(self):
        """Test Falcon has fused QKV configuration."""
        config = ARCHITECTURE_CONFIGS["falcon"]

        assert config.qkv_fused == "query_key_value"

    def test_get_architecture_config_by_name(self):
        """Test getting architecture config by architecture name."""
        config = get_architecture_config("LlamaForCausalLM")

        assert config.name == "llama"

    def test_layer_importance_weights_uniform(self):
        """Test uniform layer importance weights."""
        weights = get_layer_importance_weights(32, strategy="uniform")

        assert len(weights) == 32
        assert all(abs(w - 1.0) < 1e-6 for w in weights.values())

    def test_layer_importance_weights_top_heavy(self):
        """Test top-heavy layer importance weights."""
        weights = get_layer_importance_weights(10, strategy="top_heavy", top_heavy_ratio=2.0)

        assert len(weights) == 10
        # Higher layers should have more importance
        assert weights[9] > weights[0]

    def test_fused_qkv_handler_split_combine(self):
        """Test FusedProjectionHandler split and combine."""
        handler = FusedProjectionHandler(
            hidden_size=256,
            num_heads=8,
        )

        # Create fused QKV output
        batch_size = 2
        seq_len = 10
        qkv_output = np.random.randn(batch_size, seq_len, 256 * 3).astype(np.float32)

        # Split
        q, k, v = handler.split_qkv(qkv_output)

        assert q.shape == (batch_size, seq_len, 256)
        assert k.shape == (batch_size, seq_len, 256)
        assert v.shape == (batch_size, seq_len, 256)

        # Recombine
        recombined = handler.combine_qkv(q, k, v)

        np.testing.assert_array_almost_equal(qkv_output, recombined)

    def test_fused_qkv_handler_gqa(self):
        """Test FusedProjectionHandler with GQA (fewer KV heads)."""
        handler = FusedProjectionHandler(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,  # GQA with 2 KV heads
        )

        # Q size is full, but K and V are smaller
        assert handler.q_size == 256
        assert handler.kv_size == 64  # 2 heads * 32 dim


# =============================================================================
# CACHING AND HOT-SWAP TESTS
# =============================================================================

class TestTieredCache:
    """Tests for tiered caching system."""

    def test_cache_creation(self, temp_cache_dir):
        """Test cache creation with custom config."""
        config = TieredCacheConfig(
            max_gpu_adapters=4,
            max_cpu_adapters=16,
            disk_cache_dir=temp_cache_dir,
        )
        cache = TieredAdapterCache(config)

        stats = cache.get_stats()
        assert stats["gpu_adapters"] == 0
        assert stats["cpu_adapters"] == 0

    def test_register_and_get(self, temp_cache_dir, sample_weights, sample_config):
        """Test registering and getting an adapter."""
        config = TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        cache = TieredAdapterCache(config)

        metadata = AdapterMetadata(
            adapter_id="test_adapter",
            source_path="",
            content_hash="abc123",
            adapter_type=AdapterType.LORA,
            config=sample_config,
            base_model_id="test_model",
            hidden_size=128,
            target_modules=["q_proj"],
            total_params=1000,
            memory_bytes=4000,
        )

        cache.register("test_adapter", sample_weights, metadata)

        # Should be in CPU by default
        assert cache._metadata["test_adapter"].current_tier == CacheTier.CPU

        # Get should return the weights
        packed = cache.get("test_adapter", CacheTier.CPU)
        assert packed is not None
        assert "q_proj" in packed.weights

    def test_tier_promotion(self, temp_cache_dir, sample_weights, sample_config):
        """Test promotion from CPU to GPU tier."""
        config = TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        cache = TieredAdapterCache(config)

        metadata = AdapterMetadata(
            adapter_id="test_adapter",
            source_path="",
            content_hash="abc123",
            adapter_type=AdapterType.LORA,
            config=sample_config,
            base_model_id="test_model",
            hidden_size=128,
            target_modules=["q_proj"],
            total_params=1000,
            memory_bytes=4000,
        )

        cache.register("test_adapter", sample_weights, metadata)

        # Request GPU tier
        packed = cache.get("test_adapter", CacheTier.GPU)

        assert packed is not None
        assert "test_adapter" in cache._gpu_cache

    def test_eviction(self, temp_cache_dir, sample_weights, sample_config):
        """Test cache eviction when full."""
        config = TieredCacheConfig(
            max_gpu_adapters=2,
            max_cpu_adapters=3,
            disk_cache_dir=temp_cache_dir,
            eviction_threshold=0.5,
        )
        cache = TieredAdapterCache(config)

        # Register multiple adapters
        for i in range(4):
            metadata = AdapterMetadata(
                adapter_id=f"adapter_{i}",
                source_path="",
                content_hash=f"hash_{i}",
                adapter_type=AdapterType.LORA,
                config=sample_config,
                base_model_id="test_model",
                hidden_size=128,
                target_modules=["q_proj"],
                total_params=1000,
                memory_bytes=4000,
            )
            cache.register(f"adapter_{i}", sample_weights, metadata)

        # Evict
        evicted = cache.evict_if_needed()

        # Should have evicted some
        assert len(cache._cpu_cache) <= config.max_cpu_adapters


class TestEvictionPolicies:
    """Tests for eviction policies."""

    def test_lru_policy(self):
        """Test LRU eviction policy."""
        policy = LRUEvictionPolicy()

        now = time.time()
        adapters = {
            "old": AdapterMetadata(
                adapter_id="old",
                source_path="",
                content_hash="",
                adapter_type=AdapterType.LORA,
                config=AdapterConfig(),
                base_model_id="",
                hidden_size=128,
                target_modules=[],
                total_params=1000,
                memory_bytes=4000,
                last_used_at=datetime.fromtimestamp(now - 100),
            ),
            "new": AdapterMetadata(
                adapter_id="new",
                source_path="",
                content_hash="",
                adapter_type=AdapterType.LORA,
                config=AdapterConfig(),
                base_model_id="",
                hidden_size=128,
                target_modules=[],
                total_params=1000,
                memory_bytes=4000,
                last_used_at=datetime.fromtimestamp(now - 10),
            ),
        }

        candidates = policy.get_eviction_candidates(adapters, 1, set())

        # "old" should be evicted first
        assert candidates[0] == "old"

    def test_gdsf_policy(self):
        """Test GDSF eviction policy."""
        policy = GDSFEvictionPolicy()

        now = time.time()

        # Large, rarely used adapter
        large_rare = AdapterMetadata(
            adapter_id="large_rare",
            source_path="",
            content_hash="",
            adapter_type=AdapterType.LORA,
            config=AdapterConfig(),
            base_model_id="",
            hidden_size=128,
            target_modules=[],
            total_params=1000000,
            memory_bytes=4000000,  # Large
            activation_count=1,  # Rarely used
            load_time_ms=100,
            last_used_at=datetime.fromtimestamp(now - 100),
        )

        # Small, frequently used adapter
        small_frequent = AdapterMetadata(
            adapter_id="small_frequent",
            source_path="",
            content_hash="",
            adapter_type=AdapterType.LORA,
            config=AdapterConfig(),
            base_model_id="",
            hidden_size=128,
            target_modules=[],
            total_params=1000,
            memory_bytes=4000,  # Small
            activation_count=100,  # Frequently used
            load_time_ms=500,  # Expensive to reload
            last_used_at=datetime.fromtimestamp(now - 10),
        )

        adapters = {
            "large_rare": large_rare,
            "small_frequent": small_frequent,
        }

        candidates = policy.get_eviction_candidates(adapters, 1, set())

        # Large, rarely used should be evicted first
        assert candidates[0] == "large_rare"


class TestHotSwapManager:
    """Tests for hot-swap manager."""

    def test_register_and_activate(self, temp_cache_dir, sample_weights, sample_config):
        """Test registering and activating an adapter."""
        config = HotSwapConfig(
            cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        )
        manager = HotSwapManager(config)

        manager.register_adapter(
            adapter_id="test_adapter",
            weights=sample_weights,
            adapter_config=sample_config,
        )

        success = manager.activate("test_adapter")

        assert success
        assert manager._active_adapter_id == "test_adapter"
        assert manager.get_active_weights() is not None

    def test_hot_swap(self, temp_cache_dir, sample_weights, sample_config):
        """Test hot-swapping between adapters."""
        config = HotSwapConfig(
            cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        )
        manager = HotSwapManager(config)

        # Register two adapters
        manager.register_adapter("adapter_1", sample_weights, sample_config)
        manager.register_adapter("adapter_2", sample_weights, sample_config)

        # Activate first
        manager.activate("adapter_1")
        assert manager._active_adapter_id == "adapter_1"

        # Swap to second
        manager.activate("adapter_2")
        assert manager._active_adapter_id == "adapter_2"

        # Check swap count
        stats = manager.get_stats()
        assert stats["swap_count"] == 2

    def test_forward_computation(self, temp_cache_dir, sample_weights, sample_config):
        """Test forward pass through hot-swap manager."""
        config = HotSwapConfig(
            cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        )
        manager = HotSwapManager(config)

        # Modify weights so delta is non-zero
        modified_weights = {}
        for name, (a, b) in sample_weights.items():
            # Set B to non-zero
            b_new = np.random.randn(*b.shape).astype(np.float32)
            modified_weights[name] = (a, b_new)

        manager.register_adapter("test_adapter", modified_weights, sample_config)
        manager.activate("test_adapter")

        x = np.random.randn(2, 128).astype(np.float32)
        delta = manager.forward(x, "q_proj")

        assert delta is not None
        assert delta.shape == x.shape

    def test_deactivate(self, temp_cache_dir, sample_weights, sample_config):
        """Test deactivating adapter."""
        config = HotSwapConfig(
            cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
        )
        manager = HotSwapManager(config)

        manager.register_adapter("test_adapter", sample_weights, sample_config)
        manager.activate("test_adapter")
        manager.deactivate()

        assert manager._active_adapter_id is None
        assert manager.get_active_weights() is None


class TestLifecycleManager:
    """Tests for adapter lifecycle manager."""

    def test_full_lifecycle(self, temp_cache_dir, sample_weights, sample_config):
        """Test full adapter lifecycle."""
        config = LifecycleConfig(
            hot_swap_config=HotSwapConfig(
                cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
            ),
            enable_health_monitoring=False,
            auto_cleanup_enabled=False,
        )
        manager = AdapterLifecycleManager(config)

        try:
            # Register
            metadata = manager.register("test_adapter", sample_weights, sample_config)
            assert metadata.state == AdapterState.LOADED

            # Activate
            success = manager.activate("test_adapter")
            assert success

            # Forward
            x = np.random.randn(2, 128).astype(np.float32)
            delta = manager.forward(x, "q_proj")
            assert delta is not None

            # Deactivate
            manager.deactivate()

            # Unregister
            success = manager.unregister("test_adapter")
            assert success
        finally:
            manager.shutdown()

    def test_audit_log(self, temp_cache_dir, sample_weights, sample_config):
        """Test audit logging."""
        config = LifecycleConfig(
            hot_swap_config=HotSwapConfig(
                cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
            ),
            enable_audit_log=True,
            enable_health_monitoring=False,
            auto_cleanup_enabled=False,
        )
        manager = AdapterLifecycleManager(config)

        try:
            manager.register("test_adapter", sample_weights, sample_config)
            manager.activate("test_adapter")
            manager.deactivate()

            log = manager.get_audit_log()

            events = [entry["event"] for entry in log]
            assert "REGISTER" in events
            assert "ACTIVATE" in events
            assert "DEACTIVATE" in events
        finally:
            manager.shutdown()


# =============================================================================
# HOOKS TESTS
# =============================================================================

class TestHooks:
    """Tests for adapter hooks."""

    def test_linear_hook_creation(self):
        """Test creating a linear adapter hook."""
        # Create a mock module
        class MockLinear:
            def __init__(self):
                self.weight = np.random.randn(64, 64)

            def forward(self, x):
                return x @ self.weight.T

        module = MockLinear()
        hook = LinearAdapterHook(
            layer_idx=0,
            projection_type="q",
            module=module,
        )

        assert hook.layer_idx == 0
        assert hook.projection_type == "q"
        assert not hook._installed

    def test_linear_hook_with_weights(self):
        """Test linear hook with set weights."""
        class MockLinear:
            def forward(self, x):
                return x

        module = MockLinear()

        lora_a = np.random.randn(8, 64).astype(np.float32)
        lora_b = np.random.randn(64, 8).astype(np.float32)

        hook = LinearAdapterHook(
            layer_idx=0,
            projection_type="q",
            module=module,
        )
        hook.set_weights(lora_a, lora_b, scaling=2.0)
        hook.install()

        x = np.random.randn(2, 64).astype(np.float32)
        output = module.forward(x)

        # Output should include delta
        expected_delta = 2.0 * (x @ lora_a.T @ lora_b.T)
        np.testing.assert_array_almost_equal(output, x + expected_delta, decimal=5)

        hook.remove()

    def test_hook_manager(self):
        """Test hook manager functionality."""
        manager = HookManager()

        class MockLinear:
            def forward(self, x):
                return x

        # Create mock target
        from tensafe.adapters.placement import ProjectionTarget

        target = ProjectionTarget(
            layer_idx=0,
            layer_type=LayerType.SELF_ATTENTION,
            projection_type=ProjectionType.QUERY,
            module_path="model.layers.0.self_attn.q_proj",
            module_name="q_proj",
        )

        module = MockLinear()
        hook = manager.create_hook(target, module)

        assert hook is not None
        assert manager.get_hook(0, "q") is hook

    def test_hook_statistics(self):
        """Test hook statistics tracking."""
        class MockLinear:
            def forward(self, x):
                return x

        module = MockLinear()

        config = HookConfig(enable_profiling=True)
        hook = LinearAdapterHook(
            layer_idx=0,
            projection_type="q",
            module=module,
            config=config,
        )

        hook.set_weights(
            np.random.randn(8, 64).astype(np.float32),
            np.random.randn(64, 8).astype(np.float32),
        )
        hook.install()

        # Make some calls
        for _ in range(5):
            x = np.random.randn(2, 64).astype(np.float32)
            module.forward(x)

        assert hook.stats.call_count == 5
        assert hook.stats.total_time_ms > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full adapter system."""

    def test_end_to_end_workflow(self, temp_cache_dir):
        """Test complete workflow from adapter creation to inference."""
        # 1. Create adapter configuration
        config = AdapterConfig(
            adapter_type=AdapterType.LORA,
            rank=8,
            alpha=16.0,
            target_modules=["q_proj", "v_proj"],
        )

        # 2. Create weights
        rng = np.random.default_rng(42)
        weights = {}
        for module in config.target_modules:
            lora_a = rng.standard_normal((config.rank, 64)).astype(np.float32)
            lora_b = rng.standard_normal((64, config.rank)).astype(np.float32) * 0.1
            weights[module] = (lora_a, lora_b)

        # 3. Set up lifecycle manager
        lifecycle_config = LifecycleConfig(
            hot_swap_config=HotSwapConfig(
                cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
            ),
            enable_health_monitoring=False,
            auto_cleanup_enabled=False,
        )
        manager = AdapterLifecycleManager(lifecycle_config)

        try:
            # 4. Register multiple adapters
            manager.register("task_a", weights, config)

            # Create different weights for task B
            weights_b = {}
            for module in config.target_modules:
                lora_a = rng.standard_normal((config.rank, 64)).astype(np.float32)
                lora_b = rng.standard_normal((64, config.rank)).astype(np.float32) * 0.1
                weights_b[module] = (lora_a, lora_b)
            manager.register("task_b", weights_b, config)

            # 5. Activate and run inference
            manager.activate("task_a")

            x = rng.standard_normal((4, 64)).astype(np.float32)
            delta_a = manager.forward(x, "q_proj")

            # 6. Hot-swap to task B
            manager.activate("task_b")
            delta_b = manager.forward(x, "q_proj")

            # 7. Verify different results
            assert not np.allclose(delta_a, delta_b)

            # 8. Check stats
            stats = manager.get_stats()
            assert stats["swap_count"] == 2
        finally:
            manager.shutdown()

    def test_multiple_adapter_types(self, temp_cache_dir):
        """Test using multiple adapter types together."""
        rng = np.random.default_rng(42)

        lifecycle_config = LifecycleConfig(
            hot_swap_config=HotSwapConfig(
                cache_config=TieredCacheConfig(disk_cache_dir=temp_cache_dir)
            ),
            enable_health_monitoring=False,
            auto_cleanup_enabled=False,
        )
        manager = AdapterLifecycleManager(lifecycle_config)

        try:
            # Register LoRA adapter
            lora_config = AdapterConfig(adapter_type=AdapterType.LORA, rank=8)
            lora_weights = {
                "q_proj": (
                    rng.standard_normal((8, 64)).astype(np.float32),
                    rng.standard_normal((64, 8)).astype(np.float32) * 0.1,
                )
            }
            manager.register("lora_adapter", lora_weights, lora_config)

            # Register rsLoRA adapter
            rslora_config = AdapterConfig(adapter_type=AdapterType.RS_LORA, rank=8)
            rslora_weights = {
                "q_proj": (
                    rng.standard_normal((8, 64)).astype(np.float32),
                    rng.standard_normal((64, 8)).astype(np.float32) * 0.1,
                )
            }
            manager.register("rslora_adapter", rslora_weights, rslora_config)

            # Test both
            x = rng.standard_normal((2, 64)).astype(np.float32)

            manager.activate("lora_adapter")
            delta_lora = manager.forward(x, "q_proj")

            manager.activate("rslora_adapter")
            delta_rslora = manager.forward(x, "q_proj")

            # Both should produce output
            assert delta_lora is not None
            assert delta_rslora is not None
        finally:
            manager.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
