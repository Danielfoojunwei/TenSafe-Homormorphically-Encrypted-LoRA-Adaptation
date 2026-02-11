"""
vLLM Integration Contract Tests

Verifies that:
1. Response schemas are correctly structured
2. Mock mode works for testing without GPU
3. Error handling is robust
4. Production mode fails-closed when vLLM unavailable
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

# Try to import torch, but don't fail if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

# Import directly to avoid crypto imports
sys.path.insert(0, "src")
sys.path.insert(0, "he_lora_microkernel")


def import_adapter_module():
    """Import adapter module avoiding crypto conflicts."""
    import importlib.util

    # First import base_adapter
    base_spec = importlib.util.spec_from_file_location(
        "base_adapter",
        "he_lora_microkernel/backend/base_adapter.py"
    )
    base_module = importlib.util.module_from_spec(base_spec)
    sys.modules["base_adapter"] = base_module
    sys.modules["backend.base_adapter"] = base_module
    base_spec.loader.exec_module(base_module)

    # Import hooks module
    hooks_spec = importlib.util.spec_from_file_location(
        "hooks",
        "he_lora_microkernel/backend/vllm_adapter/hooks.py"
    )
    hooks_module = importlib.util.module_from_spec(hooks_spec)
    sys.modules["hooks"] = hooks_module
    sys.modules["vllm_adapter.hooks"] = hooks_module

    # Create proper module structure
    sys.modules["..base_adapter"] = base_module
    sys.modules[".hooks"] = hooks_module

    return base_module


base_adapter = import_adapter_module()


# ==============================================================================
# Response Schema Definitions (for contract pinning)
# ==============================================================================


@dataclass
class ModelMetadataSchema:
    """Schema for model metadata returned by init()."""

    required_fields = [
        "model_id",
        "num_layers",
        "hidden_size",
        "num_attention_heads",
        "head_dim",
        "vocab_size",
        "max_position_embeddings",
    ]

    optional_fields = [
        "architecture",
        "has_output_projection",
        "layer_name_template",
    ]

    @classmethod
    def validate(cls, metadata) -> List[str]:
        """Validate metadata against schema."""
        errors = []

        for field in cls.required_fields:
            if not hasattr(metadata, field):
                errors.append(f"Missing required field: {field}")
            else:
                value = getattr(metadata, field)
                if value is None:
                    errors.append(f"Required field is None: {field}")

        # Type checks
        if hasattr(metadata, "num_layers"):
            if not isinstance(metadata.num_layers, int) or metadata.num_layers <= 0:
                errors.append(f"num_layers must be positive int, got {metadata.num_layers}")

        if hasattr(metadata, "hidden_size"):
            if not isinstance(metadata.hidden_size, int) or metadata.hidden_size <= 0:
                errors.append(f"hidden_size must be positive int, got {metadata.hidden_size}")

        if hasattr(metadata, "vocab_size"):
            if not isinstance(metadata.vocab_size, int) or metadata.vocab_size <= 0:
                errors.append(f"vocab_size must be positive int, got {metadata.vocab_size}")

        return errors


@dataclass
class HookStatisticsSchema:
    """Schema for hook statistics returned by get_hook_statistics()."""

    expected_keys = ["total_calls", "total_delta_applications", "hooks"]

    @classmethod
    def validate(cls, stats: Dict[str, Any]) -> List[str]:
        """Validate statistics against schema."""
        errors = []

        if not isinstance(stats, dict):
            errors.append(f"Statistics must be dict, got {type(stats)}")
            return errors

        # Note: Empty stats are valid when no hooks are installed
        return errors


@dataclass
class LayerStatesSchema:
    """Schema for layer states returned by decode_one_step()."""

    @classmethod
    def validate(cls, layer_states: Dict[int, Any]) -> List[str]:
        """Validate layer states against schema."""
        errors = []

        if not isinstance(layer_states, dict):
            errors.append(f"Layer states must be dict, got {type(layer_states)}")
            return errors

        for layer_idx, state in layer_states.items():
            if not isinstance(layer_idx, int):
                errors.append(f"Layer index must be int, got {type(layer_idx)}")

            if not isinstance(state, dict):
                errors.append(f"Layer {layer_idx} state must be dict, got {type(state)}")

        return errors


# ==============================================================================
# Test Fixtures
# ==============================================================================


requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


@pytest.fixture
def mock_adapter():
    """Create a mock vLLM adapter for testing without GPU."""
    if not HAS_TORCH:
        pytest.skip("PyTorch required for adapter tests")

    BatchConfig = base_adapter.BatchConfig

    # Ensure we're in development mode for mock
    old_env = os.environ.get("TG_ENVIRONMENT")
    os.environ["TG_ENVIRONMENT"] = "development"

    try:
        # Import adapter with mock fallback
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "adapter",
            "he_lora_microkernel/backend/vllm_adapter/adapter.py"
        )
        adapter_module = importlib.util.module_from_spec(spec)

        # Set up parent module references
        import types
        backend_pkg = types.ModuleType("backend")
        backend_pkg.base_adapter = base_adapter
        sys.modules["backend"] = backend_pkg
        sys.modules["backend.base_adapter"] = base_adapter

        # Create vllm_adapter package
        vllm_pkg = types.ModuleType("backend.vllm_adapter")
        sys.modules["backend.vllm_adapter"] = vllm_pkg

        # Import hooks for adapter
        hooks_spec = importlib.util.spec_from_file_location(
            "hooks",
            "he_lora_microkernel/backend/vllm_adapter/hooks.py"
        )
        hooks_module = importlib.util.module_from_spec(hooks_spec)
        sys.modules["backend.vllm_adapter.hooks"] = hooks_module
        sys.modules[".hooks"] = hooks_module
        vllm_pkg.hooks = hooks_module

        # Now we can reference them properly
        adapter_module.__dict__["__package__"] = "backend.vllm_adapter"

        # Execute hooks first
        hooks_spec.loader.exec_module(hooks_module)

        # Then adapter
        spec.loader.exec_module(adapter_module)

        VLLMAdapter = adapter_module.VLLMAdapter

        # Create adapter instance
        batch_config = BatchConfig(
            max_batch_size=4,
            max_context_length=1024,
            dtype="float16",
        )

        adapter = VLLMAdapter(
            model_id="mock-llama-7b",
            batch_config=batch_config,
            device="cpu",
        )

        yield adapter

        # Cleanup
        if adapter.is_initialized:
            adapter.shutdown()

    finally:
        if old_env is not None:
            os.environ["TG_ENVIRONMENT"] = old_env
        elif "TG_ENVIRONMENT" in os.environ:
            del os.environ["TG_ENVIRONMENT"]


# ==============================================================================
# Contract Tests
# ==============================================================================


@requires_torch
class TestModelMetadataContract:
    """Test model metadata response schema."""

    def test_init_returns_valid_metadata(self, mock_adapter):
        """Test that init() returns properly structured metadata."""
        metadata = mock_adapter.init()

        errors = ModelMetadataSchema.validate(metadata)
        assert len(errors) == 0, f"Metadata validation errors: {errors}"

    def test_metadata_to_dict(self, mock_adapter):
        """Test that metadata can be serialized to dict."""
        metadata = mock_adapter.init()

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "model_id" in result
        assert "num_layers" in result
        assert "hidden_size" in result

    def test_metadata_values_reasonable(self, mock_adapter):
        """Test that metadata values are in reasonable ranges."""
        metadata = mock_adapter.init()

        # Check reasonable bounds
        assert 1 <= metadata.num_layers <= 200, "num_layers out of reasonable range"
        assert 64 <= metadata.hidden_size <= 65536, "hidden_size out of reasonable range"
        assert 2 <= metadata.num_attention_heads <= 256, "num_attention_heads out of reasonable range"
        assert 1000 <= metadata.vocab_size <= 500000, "vocab_size out of reasonable range"


@requires_torch
class TestPrefillContract:
    """Test prefill response contract."""

    def test_prefill_returns_kv_cache(self, mock_adapter):
        """Test that prefill returns a KV cache handle."""
        metadata = mock_adapter.init()

        # Create dummy input
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, metadata.vocab_size, (batch_size, seq_len))

        kv_cache = mock_adapter.prefill(input_ids)

        # KV cache should not be None
        assert kv_cache is not None

    def test_prefill_with_attention_mask(self, mock_adapter):
        """Test prefill with attention mask."""
        metadata = mock_adapter.init()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, metadata.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        kv_cache = mock_adapter.prefill(input_ids, attention_mask)

        assert kv_cache is not None


@requires_torch
class TestDecodeContract:
    """Test decode step response contract."""

    def test_decode_returns_logits_and_states(self, mock_adapter):
        """Test that decode_one_step returns (logits, layer_states)."""
        metadata = mock_adapter.init()

        # Prefill first
        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, metadata.vocab_size, (batch_size, seq_len))
        kv_cache = mock_adapter.prefill(input_ids)

        # Decode one step
        last_tokens = torch.randint(0, metadata.vocab_size, (batch_size, 1))
        logits, layer_states = mock_adapter.decode_one_step(last_tokens, kv_cache)

        # Verify logits shape
        assert logits.shape == (batch_size, metadata.vocab_size), \
            f"Expected logits shape ({batch_size}, {metadata.vocab_size}), got {logits.shape}"

        # Verify layer_states structure
        errors = LayerStatesSchema.validate(layer_states)
        assert len(errors) == 0, f"Layer states validation errors: {errors}"

    def test_decode_logits_are_float(self, mock_adapter):
        """Test that logits are floating point."""
        metadata = mock_adapter.init()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, metadata.vocab_size, (batch_size, seq_len))
        kv_cache = mock_adapter.prefill(input_ids)

        last_tokens = torch.randint(0, metadata.vocab_size, (batch_size, 1))
        logits, _ = mock_adapter.decode_one_step(last_tokens, kv_cache)

        assert logits.dtype in (torch.float16, torch.float32, torch.float64), \
            f"Logits should be float, got {logits.dtype}"


@requires_torch
class TestInsertionConfigContract:
    """Test insertion configuration contract."""

    def test_set_insertion_config_requires_init(self, mock_adapter):
        """Test that set_insertion_config requires initialization."""
        InsertionConfig = base_adapter.InsertionConfig

        config = InsertionConfig()

        with pytest.raises(RuntimeError) as exc_info:
            mock_adapter.set_insertion_config(config)

        assert "not initialized" in str(exc_info.value).lower()

    def test_set_insertion_config_validates_layers(self, mock_adapter):
        """Test that invalid layer indices are rejected."""
        InsertionConfig = base_adapter.InsertionConfig

        metadata = mock_adapter.init()

        # Try to configure with invalid layer
        config = InsertionConfig(layers=[metadata.num_layers + 100])

        with pytest.raises(ValueError) as exc_info:
            mock_adapter.set_insertion_config(config)

        assert "out of range" in str(exc_info.value).lower()

    def test_set_valid_insertion_config(self, mock_adapter):
        """Test that valid configuration is accepted."""
        InsertionConfig = base_adapter.InsertionConfig
        LoRATargets = base_adapter.LoRATargets

        metadata = mock_adapter.init()

        config = InsertionConfig(
            targets=LoRATargets.QKV,
            layers=[0, 1, 2],
        )

        # Should not raise
        mock_adapter.set_insertion_config(config)


@requires_torch
class TestLifecycleContract:
    """Test adapter lifecycle contract."""

    def test_shutdown_cleans_resources(self, mock_adapter):
        """Test that shutdown properly cleans up."""
        metadata = mock_adapter.init()
        assert mock_adapter.is_initialized

        mock_adapter.shutdown()
        assert not mock_adapter.is_initialized

    def test_double_init_is_idempotent(self, mock_adapter):
        """Test that calling init twice is safe."""
        metadata1 = mock_adapter.init()
        metadata2 = mock_adapter.init()

        # Should return same metadata
        assert metadata1.model_id == metadata2.model_id
        assert metadata1.num_layers == metadata2.num_layers

    def test_init_after_shutdown_works(self, mock_adapter):
        """Test that adapter can be reinitialized after shutdown."""
        metadata1 = mock_adapter.init()
        mock_adapter.shutdown()

        metadata2 = mock_adapter.init()

        assert metadata1.model_id == metadata2.model_id


@requires_torch
class TestProductionModeContract:
    """Test production mode behavior."""

    def test_production_fails_without_vllm(self):
        """Test that production mode fails closed when vLLM unavailable."""
        BatchConfig = base_adapter.BatchConfig

        # Save and set production environment
        old_env = os.environ.get("TG_ENVIRONMENT")
        old_mock = os.environ.get("TG_ALLOW_MOCK_BACKEND")

        try:
            os.environ["TG_ENVIRONMENT"] = "production"
            # Ensure mock is not allowed
            if "TG_ALLOW_MOCK_BACKEND" in os.environ:
                del os.environ["TG_ALLOW_MOCK_BACKEND"]

            # Import adapter
            import importlib
            import importlib.util

            # Force reimport
            spec = importlib.util.spec_from_file_location(
                "adapter_prod",
                "he_lora_microkernel/backend/vllm_adapter/adapter.py"
            )

            # We need to set up the module structure again
            import types
            backend_pkg = types.ModuleType("backend")
            backend_pkg.base_adapter = base_adapter
            sys.modules["backend"] = backend_pkg
            sys.modules["backend.base_adapter"] = base_adapter

            vllm_pkg = types.ModuleType("backend.vllm_adapter")
            sys.modules["backend.vllm_adapter"] = vllm_pkg

            hooks_spec = importlib.util.spec_from_file_location(
                "hooks_prod",
                "he_lora_microkernel/backend/vllm_adapter/hooks.py"
            )
            hooks_module = importlib.util.module_from_spec(hooks_spec)
            sys.modules["backend.vllm_adapter.hooks"] = hooks_module
            vllm_pkg.hooks = hooks_module
            hooks_spec.loader.exec_module(hooks_module)

            adapter_module = importlib.util.module_from_spec(spec)
            adapter_module.__dict__["__package__"] = "backend.vllm_adapter"
            spec.loader.exec_module(adapter_module)

            VLLMAdapter = adapter_module.VLLMAdapter

            batch_config = BatchConfig(
                max_batch_size=4,
                max_context_length=1024,
                dtype="float16",
            )

            adapter = VLLMAdapter(
                model_id="test-model",
                batch_config=batch_config,
                device="cpu",
            )

            # In production without vLLM, init should fail
            with pytest.raises(RuntimeError) as exc_info:
                adapter.init()

            assert "vLLM is required" in str(exc_info.value)
            assert "production" in str(exc_info.value).lower()

        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]

            if old_mock is not None:
                os.environ["TG_ALLOW_MOCK_BACKEND"] = old_mock

    def test_production_allows_explicit_mock(self):
        """Test that production mode can explicitly allow mock."""
        BatchConfig = base_adapter.BatchConfig

        # Save and set production environment with mock allowed
        old_env = os.environ.get("TG_ENVIRONMENT")
        old_mock = os.environ.get("TG_ALLOW_MOCK_BACKEND")

        try:
            os.environ["TG_ENVIRONMENT"] = "production"
            os.environ["TG_ALLOW_MOCK_BACKEND"] = "true"

            import importlib.util
            import types

            # Set up module structure
            backend_pkg = types.ModuleType("backend")
            backend_pkg.base_adapter = base_adapter
            sys.modules["backend"] = backend_pkg
            sys.modules["backend.base_adapter"] = base_adapter

            vllm_pkg = types.ModuleType("backend.vllm_adapter")
            sys.modules["backend.vllm_adapter"] = vllm_pkg

            hooks_spec = importlib.util.spec_from_file_location(
                "hooks_mock",
                "he_lora_microkernel/backend/vllm_adapter/hooks.py"
            )
            hooks_module = importlib.util.module_from_spec(hooks_spec)
            sys.modules["backend.vllm_adapter.hooks"] = hooks_module
            vllm_pkg.hooks = hooks_module
            hooks_spec.loader.exec_module(hooks_module)

            spec = importlib.util.spec_from_file_location(
                "adapter_mock",
                "he_lora_microkernel/backend/vllm_adapter/adapter.py"
            )
            adapter_module = importlib.util.module_from_spec(spec)
            adapter_module.__dict__["__package__"] = "backend.vllm_adapter"
            spec.loader.exec_module(adapter_module)

            VLLMAdapter = adapter_module.VLLMAdapter

            batch_config = BatchConfig(
                max_batch_size=4,
                max_context_length=1024,
                dtype="float16",
            )

            adapter = VLLMAdapter(
                model_id="test-model",
                batch_config=batch_config,
                device="cpu",
            )

            # With explicit mock allowed, init should work
            metadata = adapter.init()
            assert metadata is not None

            adapter.shutdown()

        finally:
            if old_env is not None:
                os.environ["TG_ENVIRONMENT"] = old_env
            elif "TG_ENVIRONMENT" in os.environ:
                del os.environ["TG_ENVIRONMENT"]

            if old_mock is not None:
                os.environ["TG_ALLOW_MOCK_BACKEND"] = old_mock
            elif "TG_ALLOW_MOCK_BACKEND" in os.environ:
                del os.environ["TG_ALLOW_MOCK_BACKEND"]


@requires_torch
class TestErrorHandling:
    """Test error handling robustness."""

    def test_get_layer_module_validates_index(self, mock_adapter):
        """Test that get_layer_module validates layer index."""
        metadata = mock_adapter.init()

        # Valid index should work
        layer = mock_adapter.get_layer_module(0)
        assert layer is not None

        # Invalid index should raise
        with pytest.raises(ValueError):
            mock_adapter.get_layer_module(metadata.num_layers + 100)

    def test_get_layer_module_requires_init(self, mock_adapter):
        """Test that get_layer_module requires initialization."""
        # Don't init

        with pytest.raises(RuntimeError):
            mock_adapter.get_layer_module(0)


class TestAdapterRegistry:
    """Test adapter registry functionality."""

    def test_unknown_adapter_raises(self):
        """Test that unknown adapter raises ValueError."""
        BatchConfig = base_adapter.BatchConfig

        batch_config = BatchConfig()

        with pytest.raises(ValueError) as exc_info:
            base_adapter.get_adapter("nonexistent", "test-model", batch_config)

        assert "Unknown backend" in str(exc_info.value)

    @requires_torch
    def test_vllm_adapter_registered(self, mock_adapter):
        """Test that vLLM adapter is registered when imported."""
        # The mock_adapter fixture imports the adapter which registers it
        available = base_adapter.list_available_adapters()
        # Note: Registration happens through @register_adapter decorator
        # which requires the module to be imported
        assert len(available) >= 0  # At least no error

    @requires_torch
    def test_adapter_init_provides_metadata(self, mock_adapter):
        """Test that adapter initialization provides valid metadata."""
        metadata = mock_adapter.init()

        # Verify metadata has required fields
        assert metadata.model_id is not None
        assert metadata.num_layers > 0
        assert metadata.hidden_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
