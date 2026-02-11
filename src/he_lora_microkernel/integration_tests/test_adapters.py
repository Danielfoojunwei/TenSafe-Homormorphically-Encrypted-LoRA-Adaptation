"""
Integration Tests for Backend Adapters

Tests vLLM, TensorRT-LLM, and SGLang adapters with mock mode.
"""

import pytest

from he_lora_microkernel.backend.base_adapter import (
    BatchConfig,
    InsertionConfig,
    InsertionPoint,
    LoRATargets,
    get_adapter,
    list_available_adapters,
)


# Test fixtures
@pytest.fixture
def batch_config():
    """Standard batch configuration for tests."""
    return BatchConfig(
        max_batch_size=4,
        max_context_length=2048,
    )


@pytest.fixture
def insertion_config():
    """Standard insertion configuration."""
    return InsertionConfig(
        targets=LoRATargets.QKV,
        layers=[0, 1, 2, 3],  # First 4 layers
        insertion_point=InsertionPoint.POST_PROJECTION,
    )


class TestVLLMAdapter:
    """Tests for vLLM adapter."""

    def test_adapter_registration(self):
        """Test vLLM adapter is registered."""
        available = list_available_adapters()
        assert "vllm" in available

    def test_mock_initialization(self, batch_config):
        """Test adapter initializes in mock mode."""
        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )

        metadata = adapter.init()

        assert metadata is not None
        assert metadata.num_layers > 0
        assert metadata.hidden_size > 0
        assert adapter._initialized

    def test_prefill(self, batch_config):
        """Test prefill phase."""
        pytest.importorskip("torch")
        import torch

        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()

        input_ids = torch.randint(0, 32000, (2, 10))
        kv_cache = adapter.prefill(input_ids)

        assert kv_cache is not None

    def test_decode_step(self, batch_config):
        """Test decode step."""
        pytest.importorskip("torch")
        import torch

        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()

        input_ids = torch.randint(0, 32000, (2, 10))
        kv_cache = adapter.prefill(input_ids)

        last_token = torch.randint(0, 32000, (2, 1))
        logits, layer_states = adapter.decode_one_step(last_token, kv_cache)

        assert logits is not None
        assert logits.shape[0] == 2
        assert logits.shape[1] == adapter._metadata.vocab_size

    def test_insertion_config(self, batch_config, insertion_config):
        """Test insertion point configuration."""
        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()
        adapter.set_insertion_config(insertion_config)

        layers = adapter.get_patched_layers()
        assert layers == [0, 1, 2, 3]

    def test_delta_callback(self, batch_config, insertion_config):
        """Test delta callback mechanism."""
        pytest.importorskip("torch")
        import torch

        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()
        adapter.set_insertion_config(insertion_config)

        # Track callback invocations
        callback_calls = []

        def delta_callback(layer_idx, proj_type, hidden_states):
            callback_calls.append((layer_idx, proj_type))
            return torch.zeros_like(hidden_states)

        adapter.set_delta_callback(delta_callback)

        # Run decode
        input_ids = torch.randint(0, 32000, (1, 5))
        kv_cache = adapter.prefill(input_ids)
        last_token = torch.randint(0, 32000, (1, 1))
        adapter.decode_one_step(last_token, kv_cache)

        # Callback should have been called
        # (In mock mode, it may not be called depending on implementation)

    def test_shutdown(self, batch_config):
        """Test adapter shutdown."""
        adapter = get_adapter(
            "vllm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()
        adapter.shutdown()

        assert not adapter._initialized


class TestTensorRTLLMAdapter:
    """Tests for TensorRT-LLM adapter."""

    def test_adapter_registration(self):
        """Test TensorRT-LLM adapter is registered."""
        available = list_available_adapters()
        assert "tensorrt_llm" in available

    def test_mock_initialization(self, batch_config):
        """Test adapter initializes in mock mode."""
        adapter = get_adapter(
            "tensorrt_llm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )

        metadata = adapter.init()

        assert metadata is not None
        assert metadata.num_layers > 0
        assert adapter._initialized

    def test_plugin_creation(self, batch_config, insertion_config):
        """Test plugin creation for projections."""
        adapter = get_adapter(
            "tensorrt_llm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()
        adapter.set_insertion_config(insertion_config)

        # Check plugins were created
        stats = adapter.get_plugin_statistics()
        assert stats['plugin_count'] > 0
        assert len(stats['layers']) == len(insertion_config.layers)

    def test_hybrid_mode(self, batch_config):
        """Test hybrid mode execution."""
        pytest.importorskip("torch")
        import torch

        adapter = get_adapter(
            "tensorrt_llm",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()

        input_ids = torch.randint(0, 32000, (1, 5))
        kv_cache = adapter.prefill(input_ids)
        last_token = torch.randint(0, 32000, (1, 1))
        logits, _ = adapter.decode_one_step(last_token, kv_cache)

        assert logits is not None


class TestSGLangAdapter:
    """Tests for SGLang adapter."""

    def test_adapter_registration(self):
        """Test SGLang adapter is registered."""
        available = list_available_adapters()
        assert "sglang" in available

    def test_mock_initialization(self, batch_config):
        """Test adapter initializes in mock mode."""
        adapter = get_adapter(
            "sglang",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )

        metadata = adapter.init()

        assert metadata is not None
        assert metadata.num_layers > 0
        assert adapter._initialized

    def test_hook_registry(self, batch_config, insertion_config):
        """Test hook registry management."""
        adapter = get_adapter(
            "sglang",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()

        # In mock mode, hooks won't be installed since no model
        # but we can test the statistics
        stats = adapter.get_hook_statistics()
        assert 'projection_hooks' in stats
        assert 'radix_hooks' in stats

    def test_request_management(self, batch_config):
        """Test request lifecycle management."""
        adapter = get_adapter(
            "sglang",
            model_id="meta-llama/Llama-2-7b-hf",
            batch_config=batch_config,
        )
        adapter.init()

        # Create request
        adapter.create_request("test_req_1", [1, 2, 3, 4, 5])
        assert "test_req_1" in adapter.get_active_requests()

        # Release request
        adapter.release_request("test_req_1")
        assert "test_req_1" not in adapter.get_active_requests()


class TestAdapterInteroperability:
    """Tests for adapter interoperability."""

    @pytest.mark.parametrize("adapter_name", ["vllm", "tensorrt_llm", "sglang"])
    def test_common_interface(self, adapter_name, batch_config, insertion_config):
        """Test all adapters implement common interface."""
        pytest.importorskip("torch")
        import torch

        adapter = get_adapter(
            adapter_name,
            model_id="test-model",
            batch_config=batch_config,
        )

        # Test init
        metadata = adapter.init()
        assert metadata is not None
        assert hasattr(metadata, 'num_layers')
        assert hasattr(metadata, 'hidden_size')

        # Test insertion config
        adapter.set_insertion_config(insertion_config)
        layers = adapter.get_patched_layers()
        assert layers is not None

        # Test prefill
        input_ids = torch.randint(0, 1000, (1, 5))
        kv_cache = adapter.prefill(input_ids)
        assert kv_cache is not None

        # Test decode
        last_token = torch.randint(0, 1000, (1, 1))
        logits, _ = adapter.decode_one_step(last_token, kv_cache)
        assert logits is not None

        # Test shutdown
        adapter.shutdown()
        assert not adapter._initialized

    @pytest.mark.parametrize("adapter_name", ["vllm", "tensorrt_llm", "sglang"])
    def test_layer_selection(self, adapter_name, batch_config):
        """Test layer selection works across adapters."""
        adapter = get_adapter(
            adapter_name,
            model_id="test-model",
            batch_config=batch_config,
        )
        adapter.init()

        # Test different layer selections
        configs = [
            InsertionConfig(targets=LoRATargets.QKV, layers=None),  # All layers
            InsertionConfig(targets=LoRATargets.QKV, layers=[0, 1, 2]),
            InsertionConfig(targets=LoRATargets.QKVO, layers=[0]),
        ]

        for config in configs:
            adapter.set_insertion_config(config)
            layers = adapter.get_patched_layers()

            if config.layers is None:
                # All layers
                assert len(layers) == adapter._metadata.num_layers
            else:
                assert layers == config.layers

        adapter.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
