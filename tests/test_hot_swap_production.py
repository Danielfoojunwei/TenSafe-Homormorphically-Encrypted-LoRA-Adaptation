"""
Production Hardening Tests for Hot-Swap LoRA Architecture.

These tests verify the production readiness of the hot-swap mechanism:
1. Hot-swap with different target module configurations
2. Atomic hook reconfiguration with rollback
3. Hot-swap callback notifications
4. Hot-swap metrics tracking
5. MoE expert-level targeting
6. HE context health checks
"""

import json
import os
import struct
import tempfile
import time
from datetime import datetime
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestHotSwapDifferentTargets:
    """Test hot-swapping between adapters with different target modules."""

    def _create_mock_tgsp(
        self,
        model_name: str,
        target_modules: List[str],
        lora_rank: int = 16,
    ) -> str:
        """Create a mock TGSP file with specified target modules."""
        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f:
            f.write(b"TGSP\x01\x00")
            header = {"version": "1.0"}
            header_bytes = json.dumps(header).encode()
            f.write(struct.pack(">I", len(header_bytes)))
            f.write(header_bytes)
            manifest = {
                "model_name": model_name,
                "model_version": "1.0.0",
                "author_id": "test-author",
                "privacy": {
                    "scheme_config": {
                        "lora_rank": lora_rank,
                        "lora_alpha": lora_rank * 2.0,
                        "target_modules": target_modules,
                    }
                }
            }
            manifest_bytes = json.dumps(manifest).encode()
            f.write(struct.pack(">I", len(manifest_bytes)))
            f.write(manifest_bytes)
            recipients_bytes = json.dumps([]).encode()
            f.write(struct.pack(">I", len(recipients_bytes)))
            f.write(recipients_bytes)
            f.write(struct.pack(">Q", 0))
            return f.name

    def test_hot_swap_qkv_to_qkvo(self):
        """Test hot-swapping from QKV-only adapter to QKVO adapter."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Create adapters with different targets
        qkv_path = self._create_mock_tgsp(
            "qkv-only-adapter",
            ["q_proj", "k_proj", "v_proj"],
        )
        qkvo_path = self._create_mock_tgsp(
            "qkvo-adapter",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        try:
            # Load both adapters
            qkv_id = registry.load_tgsp_adapter(qkv_path)
            qkvo_id = registry.load_tgsp_adapter(qkvo_path)

            # Activate QKV adapter
            registry.activate_adapter(qkv_id)
            active = registry.get_active_adapter()
            assert set(active.metadata.target_modules) == {"q_proj", "k_proj", "v_proj"}

            # Hot-swap to QKVO adapter
            registry.activate_adapter(qkvo_id)
            active = registry.get_active_adapter()
            assert set(active.metadata.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

            # Verify QKV adapter is no longer active
            info = registry.get_adapter_info(qkv_id)
            assert info["is_active"] is False

        finally:
            os.unlink(qkv_path)
            os.unlink(qkvo_path)
            registry.cleanup()

    def test_hot_swap_full_to_single_projection(self):
        """Test hot-swapping from full projections to single projection."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Create adapters
        full_path = self._create_mock_tgsp(
            "full-adapter",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        q_only_path = self._create_mock_tgsp(
            "q-only-adapter",
            ["q_proj"],
        )

        try:
            full_id = registry.load_tgsp_adapter(full_path)
            q_only_id = registry.load_tgsp_adapter(q_only_path)

            registry.activate_adapter(full_id)
            registry.activate_adapter(q_only_id)

            active = registry.get_active_adapter()
            assert active.metadata.target_modules == ["q_proj"]

        finally:
            os.unlink(full_path)
            os.unlink(q_only_path)
            registry.cleanup()

    def test_hot_swap_mlp_targeting(self):
        """Test hot-swapping to MLP-targeting adapter (MoE use case)."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Create adapters
        attention_path = self._create_mock_tgsp(
            "attention-adapter",
            ["q_proj", "k_proj", "v_proj"],
        )
        mlp_path = self._create_mock_tgsp(
            "mlp-adapter",
            ["gate_proj", "up_proj", "down_proj"],
        )

        try:
            attn_id = registry.load_tgsp_adapter(attention_path)
            mlp_id = registry.load_tgsp_adapter(mlp_path)

            registry.activate_adapter(attn_id)
            registry.activate_adapter(mlp_id)

            active = registry.get_active_adapter()
            assert set(active.metadata.target_modules) == {"gate_proj", "up_proj", "down_proj"}

        finally:
            os.unlink(attention_path)
            os.unlink(mlp_path)
            registry.cleanup()


class TestHotSwapCallbacks:
    """Test hot-swap callback mechanism."""

    def _create_mock_tgsp(self, name: str, targets: List[str]) -> str:
        """Create a mock TGSP file."""
        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f:
            f.write(b"TGSP\x01\x00")
            header = {"version": "1.0"}
            header_bytes = json.dumps(header).encode()
            f.write(struct.pack(">I", len(header_bytes)))
            f.write(header_bytes)
            manifest = {
                "model_name": name,
                "model_version": "1.0.0",
                "author_id": "test",
                "privacy": {
                    "scheme_config": {
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "target_modules": targets,
                    }
                }
            }
            manifest_bytes = json.dumps(manifest).encode()
            f.write(struct.pack(">I", len(manifest_bytes)))
            f.write(manifest_bytes)
            recipients_bytes = json.dumps([]).encode()
            f.write(struct.pack(">I", len(recipients_bytes)))
            f.write(recipients_bytes)
            f.write(struct.pack(">Q", 0))
            return f.name

    def test_callback_invoked_on_hot_swap(self):
        """Test that callbacks are invoked during hot-swap."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        callback_invocations = []

        def hot_swap_callback(old_adapter, new_adapter, target_modules_changed):
            callback_invocations.append({
                "old_id": old_adapter.metadata.adapter_id if old_adapter else None,
                "new_id": new_adapter.metadata.adapter_id,
                "target_modules_changed": target_modules_changed,
            })

        registry.register_hot_swap_callback(hot_swap_callback)

        tgsp_1 = self._create_mock_tgsp("adapter-1", ["q_proj"])
        tgsp_2 = self._create_mock_tgsp("adapter-2", ["q_proj", "v_proj"])

        try:
            id_1 = registry.load_tgsp_adapter(tgsp_1)
            id_2 = registry.load_tgsp_adapter(tgsp_2)

            # First activation
            registry.activate_adapter(id_1)
            assert len(callback_invocations) == 1
            assert callback_invocations[0]["old_id"] is None
            assert callback_invocations[0]["target_modules_changed"] is False

            # Hot-swap (different targets)
            registry.activate_adapter(id_2)
            assert len(callback_invocations) == 2
            assert callback_invocations[1]["old_id"] == id_1
            assert callback_invocations[1]["target_modules_changed"] is True

        finally:
            os.unlink(tgsp_1)
            os.unlink(tgsp_2)
            registry.cleanup()

    def test_callback_unregister(self):
        """Test callback unregistration."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        callback_count = [0]

        def callback(old, new, changed):
            callback_count[0] += 1

        registry.register_hot_swap_callback(callback)

        tgsp = self._create_mock_tgsp("adapter", ["q_proj"])

        try:
            adapter_id = registry.load_tgsp_adapter(tgsp)

            registry.activate_adapter(adapter_id)
            assert callback_count[0] == 1

            # Unregister
            result = registry.unregister_hot_swap_callback(callback)
            assert result is True

            # Re-activate should not invoke callback
            registry.activate_adapter(adapter_id)
            assert callback_count[0] == 1  # Unchanged

        finally:
            os.unlink(tgsp)
            registry.cleanup()

    def test_callback_error_does_not_fail_swap(self):
        """Test that callback errors don't prevent hot-swap."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        def failing_callback(old, new, changed):
            raise RuntimeError("Callback error!")

        registry.register_hot_swap_callback(failing_callback)

        tgsp = self._create_mock_tgsp("adapter", ["q_proj"])

        try:
            adapter_id = registry.load_tgsp_adapter(tgsp)

            # Should not raise despite callback error
            registry.activate_adapter(adapter_id)

            # Adapter should still be active
            active = registry.get_active_adapter()
            assert active is not None
            assert active.metadata.adapter_id == adapter_id

        finally:
            os.unlink(tgsp)
            registry.cleanup()


class TestHotSwapMetrics:
    """Test hot-swap metrics tracking."""

    def _create_mock_tgsp(self, name: str) -> str:
        """Create a mock TGSP file."""
        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f:
            f.write(b"TGSP\x01\x00")
            header_bytes = json.dumps({"version": "1.0"}).encode()
            f.write(struct.pack(">I", len(header_bytes)))
            f.write(header_bytes)
            manifest = {
                "model_name": name,
                "model_version": "1.0.0",
                "author_id": "test",
            }
            manifest_bytes = json.dumps(manifest).encode()
            f.write(struct.pack(">I", len(manifest_bytes)))
            f.write(manifest_bytes)
            f.write(struct.pack(">I", 2))
            f.write(b"[]")
            f.write(struct.pack(">Q", 0))
            return f.name

    def test_metrics_tracked_on_swap(self):
        """Test that hot-swap metrics are tracked."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        tgsp_1 = self._create_mock_tgsp("adapter-1")
        tgsp_2 = self._create_mock_tgsp("adapter-2")

        try:
            id_1 = registry.load_tgsp_adapter(tgsp_1)
            id_2 = registry.load_tgsp_adapter(tgsp_2)

            # Check initial metrics
            metrics = registry.get_hot_swap_metrics()
            assert metrics["total_swaps"] == 0

            # Perform swaps
            registry.activate_adapter(id_1)
            registry.activate_adapter(id_2)
            registry.activate_adapter(id_1)

            # Check updated metrics
            metrics = registry.get_hot_swap_metrics()
            assert metrics["total_swaps"] == 3
            assert metrics["successful_swaps"] == 3
            assert metrics["failed_swaps"] == 0
            assert metrics["success_rate"] == 1.0
            assert metrics["avg_swap_time_ms"] > 0
            assert id_1 in metrics["swaps_by_adapter"]
            assert metrics["swaps_by_adapter"][id_1] == 2

        finally:
            os.unlink(tgsp_1)
            os.unlink(tgsp_2)
            registry.cleanup()


class TestAtomicHookReconfiguration:
    """Test atomic hook reconfiguration in vLLM adapter."""

    def test_hook_rollback_on_failure(self):
        """Test that hooks are rolled back on installation failure."""
        from he_lora_microkernel.backend.base_adapter import (
            BatchConfig,
            InsertionConfig,
            LoRATargets,
            get_adapter,
        )

        batch_config = BatchConfig(max_batch_size=4, max_context_length=2048)

        adapter_cls = get_adapter("vllm")
        adapter = adapter_cls(
            model_id="test-model",
            batch_config=batch_config,
        )
        adapter.init()

        # Set initial config
        initial_config = InsertionConfig(targets=LoRATargets.QKV, layers=[0, 1])
        adapter.set_insertion_config(initial_config)

        initial_hook_count = len(adapter._hooks)
        assert initial_hook_count > 0

        # Try to set invalid config (layer out of range)
        invalid_config = InsertionConfig(
            targets=LoRATargets.QKV,
            layers=[0, 1, 999],  # 999 is out of range
        )

        with pytest.raises((ValueError, RuntimeError)):
            adapter.set_insertion_config(invalid_config)

        # Hooks should be restored to initial state
        assert len(adapter._hooks) == initial_hook_count

        adapter.shutdown()

    def test_reconfigure_for_adapter_method(self):
        """Test the reconfigure_for_adapter convenience method."""
        from he_lora_microkernel.backend.base_adapter import (
            BatchConfig,
            get_adapter,
        )

        batch_config = BatchConfig(max_batch_size=4, max_context_length=2048)

        adapter_cls = get_adapter("vllm")
        adapter = adapter_cls(
            model_id="test-model",
            batch_config=batch_config,
        )
        adapter.init()

        # Reconfigure for QKV targets
        adapter.reconfigure_for_adapter(["q_proj", "k_proj", "v_proj"])

        # Verify hooks were created
        assert len(adapter._hooks) > 0

        # Reconfigure for QKVO targets
        adapter.reconfigure_for_adapter(
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            layer_indices=[0, 1],
        )

        # Verify hooks were updated
        assert len(adapter._hooks) > 0

        adapter.shutdown()


class TestN2HEAdapterConfigMoE:
    """Test N2HE adapter configuration for MoE models."""

    def test_adapter_placement_enum(self):
        """Test AdapterPlacement enum provides correct module lists."""
        from he_lora_microkernel.n2he.adapter_config import AdapterPlacement

        # Attention QKV
        qkv_modules = AdapterPlacement.ATTENTION_QKV.get_target_modules()
        assert set(qkv_modules) == {"q_proj", "k_proj", "v_proj"}

        # MLP up (expert gates in MoE)
        mlp_up_modules = AdapterPlacement.MLP_UP.get_target_modules()
        assert set(mlp_up_modules) == {"gate_proj", "up_proj"}

        # All modules
        all_modules = AdapterPlacement.ALL.get_target_modules()
        assert "q_proj" in all_modules
        assert "gate_proj" in all_modules
        assert "down_proj" in all_modules

    def test_nonlinear_adapter_requires_tfhe(self):
        """Test that non-linear adapters require TFHE bootstrapping."""
        from he_lora_microkernel.n2he.adapter_config import (
            N2HEAdapterConfig,
            AdapterType,
            NonLinearActivation,
        )

        # Should raise because TFHE not enabled
        with pytest.raises(ValueError) as exc_info:
            N2HEAdapterConfig(
                adapter_type=AdapterType.GATED_LORA,
                activation=NonLinearActivation.SIGMOID,
                use_tfhe_bootstrap=False,  # Invalid!
            )

        assert "use_tfhe_bootstrap=True" in str(exc_info.value)

    def test_linear_adapter_rejects_tfhe(self):
        """Test that linear adapters reject TFHE (not needed)."""
        from he_lora_microkernel.n2he.adapter_config import (
            N2HEAdapterConfig,
            AdapterType,
        )

        with pytest.raises(ValueError) as exc_info:
            N2HEAdapterConfig(
                adapter_type=AdapterType.LINEAR_LORA,
                use_tfhe_bootstrap=True,  # Invalid for linear!
            )

        assert "does not need TFHE" in str(exc_info.value)


class TestHotSwapIntegration:
    """Integration tests for hot-swap with hook manager."""

    def test_full_hot_swap_flow_with_callbacks(self):
        """Test complete hot-swap flow with callback-based hook reconfiguration."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry
        from he_lora_microkernel.backend.base_adapter import (
            BatchConfig,
            get_adapter,
        )

        # Setup registry
        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Setup vLLM adapter
        batch_config = BatchConfig(max_batch_size=4, max_context_length=2048)
        adapter_cls = get_adapter("vllm")
        vllm_adapter = adapter_cls(model_id="test-model", batch_config=batch_config)
        vllm_adapter.init()

        # Register callback to reconfigure hooks on hot-swap
        def on_hot_swap(old_adapter, new_adapter, target_modules_changed):
            if target_modules_changed:
                vllm_adapter.reconfigure_for_adapter(
                    new_adapter.metadata.target_modules
                )

        registry.register_hot_swap_callback(on_hot_swap)

        # Create mock TGSP files
        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f1:
            f1.write(b"TGSP\x01\x00")
            h1 = json.dumps({"version": "1.0"}).encode()
            f1.write(struct.pack(">I", len(h1)))
            f1.write(h1)
            m1 = {
                "model_name": "qkv-adapter",
                "model_version": "1.0.0",
                "author_id": "test",
                "privacy": {
                    "scheme_config": {
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "target_modules": ["q_proj", "k_proj", "v_proj"],
                    }
                }
            }
            m1_bytes = json.dumps(m1).encode()
            f1.write(struct.pack(">I", len(m1_bytes)))
            f1.write(m1_bytes)
            f1.write(struct.pack(">I", 2))
            f1.write(b"[]")
            f1.write(struct.pack(">Q", 0))
            tgsp_1 = f1.name

        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f2:
            f2.write(b"TGSP\x01\x00")
            h2 = json.dumps({"version": "1.0"}).encode()
            f2.write(struct.pack(">I", len(h2)))
            f2.write(h2)
            m2 = {
                "model_name": "qkvo-adapter",
                "model_version": "1.0.0",
                "author_id": "test",
                "privacy": {
                    "scheme_config": {
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    }
                }
            }
            m2_bytes = json.dumps(m2).encode()
            f2.write(struct.pack(">I", len(m2_bytes)))
            f2.write(m2_bytes)
            f2.write(struct.pack(">I", 2))
            f2.write(b"[]")
            f2.write(struct.pack(">Q", 0))
            tgsp_2 = f2.name

        try:
            # Load adapters
            id_1 = registry.load_tgsp_adapter(tgsp_1)
            id_2 = registry.load_tgsp_adapter(tgsp_2)

            # Activate first adapter
            registry.activate_adapter(id_1)

            # Hot-swap to second adapter (different targets)
            registry.activate_adapter(id_2)

            # Verify metrics show the swap
            metrics = registry.get_hot_swap_metrics()
            assert metrics["successful_swaps"] == 2

        finally:
            os.unlink(tgsp_1)
            os.unlink(tgsp_2)
            vllm_adapter.shutdown()
            registry.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
