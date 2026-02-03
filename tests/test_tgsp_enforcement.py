"""
Tests for TGSP Format Enforcement in Encrypted Inference.

These tests verify the TGSP lock-in mechanism that ensures only TGSP-format
adapters can be used with HE-encrypted inference.
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest


class TestTGSPAdapterRegistry:
    """Tests for TGSPAdapterRegistry."""

    def test_registry_initialization(self):
        """Test registry initializes with correct defaults."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry()
        assert registry.enforce_tgsp is True
        assert registry.auto_verify_signatures is True
        assert len(registry._adapters) == 0

        registry.cleanup()

    def test_registry_rejects_non_tgsp_format(self):
        """Test registry rejects non-TGSP format files."""
        from tensafe.tgsp_adapter_registry import (
            TGSPAdapterRegistry,
            TGSPFormatRequiredError,
        )

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Test various non-TGSP formats
        non_tgsp_files = [
            "model.safetensors",
            "adapter.bin",
            "weights.pt",
            "checkpoint.pth",
            "model.onnx",
        ]

        for file_path in non_tgsp_files:
            with pytest.raises(TGSPFormatRequiredError) as exc_info:
                registry.load_tgsp_adapter(file_path)

            assert "TGSP format" in str(exc_info.value)
            assert file_path.split('.')[-1] in exc_info.value.attempted_format or "." in exc_info.value.attempted_format

        registry.cleanup()

    def test_registry_accepts_tgsp_format(self):
        """Test registry accepts valid TGSP format files."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Create a mock TGSP file
        with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f:
            # Write TGSP magic bytes
            f.write(b"TGSP\x01\x00")

            # Write header
            header = {"version": "1.0"}
            header_bytes = json.dumps(header).encode()
            f.write(struct.pack(">I", len(header_bytes)))
            f.write(header_bytes)

            # Write manifest
            manifest = {
                "model_name": "test-model",
                "model_version": "1.0.0",
                "author_id": "test-author",
                "privacy": {
                    "scheme_config": {
                        "lora_rank": 16,
                        "lora_alpha": 32.0,
                        "target_modules": ["q_proj", "v_proj"],
                    }
                }
            }
            manifest_bytes = json.dumps(manifest).encode()
            f.write(struct.pack(">I", len(manifest_bytes)))
            f.write(manifest_bytes)

            # Write recipients block
            recipients = []
            recipients_bytes = json.dumps(recipients).encode()
            f.write(struct.pack(">I", len(recipients_bytes)))
            f.write(recipients_bytes)

            # Write payload
            payload = b"mock_payload_data"
            f.write(struct.pack(">Q", len(payload)))
            f.write(payload)

            tgsp_path = f.name

        try:
            adapter_id = registry.load_tgsp_adapter(tgsp_path)
            assert adapter_id is not None
            assert adapter_id.startswith("tgsp_")

            # Check adapter is loaded
            adapters = registry.list_adapters()
            assert len(adapters) == 1
            assert adapters[0]["adapter_id"] == adapter_id
        finally:
            os.unlink(tgsp_path)
            registry.cleanup()

    def test_adapter_hot_swap(self):
        """Test adapter hot-swapping functionality."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Create two mock TGSP files
        tgsp_files = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix='.tgsp', delete=False) as f:
                f.write(b"TGSP\x01\x00")
                header = {"version": "1.0"}
                header_bytes = json.dumps(header).encode()
                f.write(struct.pack(">I", len(header_bytes)))
                f.write(header_bytes)
                manifest = {
                    "model_name": f"test-model-{i}",
                    "model_version": "1.0.0",
                    "author_id": "test-author",
                }
                manifest_bytes = json.dumps(manifest).encode()
                f.write(struct.pack(">I", len(manifest_bytes)))
                f.write(manifest_bytes)
                recipients_bytes = json.dumps([]).encode()
                f.write(struct.pack(">I", len(recipients_bytes)))
                f.write(recipients_bytes)
                f.write(struct.pack(">Q", 0))
                tgsp_files.append(f.name)

        try:
            # Load both adapters
            adapter_id_1 = registry.load_tgsp_adapter(tgsp_files[0])
            adapter_id_2 = registry.load_tgsp_adapter(tgsp_files[1])

            # Activate first adapter
            registry.activate_adapter(adapter_id_1)
            active = registry.get_active_adapter()
            assert active is not None
            assert active.metadata.adapter_id == adapter_id_1

            # Hot-swap to second adapter
            registry.activate_adapter(adapter_id_2)
            active = registry.get_active_adapter()
            assert active is not None
            assert active.metadata.adapter_id == adapter_id_2

            # First adapter should no longer be active
            info = registry.get_adapter_info(adapter_id_1)
            assert info["is_active"] is False

        finally:
            for f in tgsp_files:
                os.unlink(f)
            registry.cleanup()

    def test_no_active_adapter_error(self):
        """Test error when no adapter is activated."""
        from tensafe.tgsp_adapter_registry import (
            NoActiveAdapterError,
            TGSPAdapterRegistry,
        )

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        with pytest.raises(NoActiveAdapterError):
            registry.forward_he(np.zeros(64))

        registry.cleanup()

    def test_audit_log(self):
        """Test audit log is maintained."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        registry = TGSPAdapterRegistry(enforce_tgsp=True)

        # Check initial audit log
        log = registry.get_audit_log()
        assert len(log) >= 1
        assert log[0]["event_type"] == "REGISTRY_INITIALIZED"

        registry.cleanup()


class TestTGSPInferenceEnforcement:
    """Tests for TGSP enforcement in TenSafeInference."""

    def test_inference_rejects_non_tgsp_in_he_mode(self):
        """Test inference rejects non-TGSP adapters in HE mode."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
            TGSPEnforcementError,
        )

        # Create config with HE mode and enforcement
        config = InferenceConfig(
            lora_mode=LoRAMode.HE_ONLY,
            enforce_tgsp=True,
        )

        # Create mock weights (non-TGSP)
        lora_a = np.random.randn(16, 64).astype(np.float64)
        lora_b = np.random.randn(64, 16).astype(np.float64)
        weights = {"q_proj": (lora_a, lora_b)}

        # Should raise enforcement error
        with pytest.raises(TGSPEnforcementError):
            TenSafeInference(
                lora_weights=weights,
                config=config,
            )

    def test_inference_allows_non_tgsp_in_plaintext_mode(self):
        """Test inference allows non-TGSP adapters in plaintext mode."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
        )

        # Create config with plaintext mode
        config = InferenceConfig(
            lora_mode=LoRAMode.PLAINTEXT,
            enforce_tgsp=True,  # Still enabled, but shouldn't matter
        )

        # Create mock weights (non-TGSP)
        lora_a = np.random.randn(16, 64).astype(np.float64)
        lora_b = np.random.randn(64, 16).astype(np.float64)
        weights = {"q_proj": (lora_a, lora_b)}

        # Should work fine in plaintext mode
        inference = TenSafeInference(
            lora_weights=weights,
            config=config,
        )

        assert inference.config.lora_mode == LoRAMode.PLAINTEXT

    def test_inference_allows_bypass_when_disabled(self):
        """Test inference allows non-TGSP when enforcement disabled."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
        )

        # Create config with HE mode but enforcement disabled
        config = InferenceConfig(
            lora_mode=LoRAMode.HE_ONLY,
            enforce_tgsp=False,  # Explicitly disabled
        )

        # Create mock weights (non-TGSP)
        lora_a = np.random.randn(16, 64).astype(np.float64)
        lora_b = np.random.randn(64, 16).astype(np.float64)
        weights = {"q_proj": (lora_a, lora_b)}

        # Should work when enforcement disabled
        # Note: HE adapter may fail to initialize, but enforcement check passes
        try:
            inference = TenSafeInference(
                lora_weights=weights,
                config=config,
            )
            # If we get here without TGSPEnforcementError, enforcement is bypassed
        except RuntimeError as e:
            # Expected if HE backend not available
            assert "N2HE-HEXL" in str(e)

    def test_register_weights_enforces_tgsp(self):
        """Test register_lora_weights enforces TGSP format."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
            TGSPEnforcementError,
        )

        # Create inference with HE mode (no initial weights)
        config = InferenceConfig(
            lora_mode=LoRAMode.PLAINTEXT,  # Start in plaintext
            enforce_tgsp=True,
        )

        inference = TenSafeInference(config=config)

        # Change to HE mode
        inference.config.lora_mode = LoRAMode.HE_ONLY

        # Try to register non-TGSP weights
        lora_a = np.random.randn(16, 64).astype(np.float64)
        lora_b = np.random.randn(64, 16).astype(np.float64)

        with pytest.raises(TGSPEnforcementError):
            inference.register_lora_weights("q_proj", lora_a, lora_b, from_tgsp=False)

    def test_register_weights_allows_tgsp_flag(self):
        """Test register_lora_weights accepts weights marked as from TGSP."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
        )

        config = InferenceConfig(
            lora_mode=LoRAMode.PLAINTEXT,
            enforce_tgsp=True,
        )

        inference = TenSafeInference(config=config)
        inference.config.lora_mode = LoRAMode.HE_ONLY

        lora_a = np.random.randn(16, 64).astype(np.float64)
        lora_b = np.random.randn(64, 16).astype(np.float64)

        # Should work with from_tgsp=True
        inference.register_lora_weights("q_proj", lora_a, lora_b, from_tgsp=True)
        assert "q_proj" in inference._lora_weights

    def test_metrics_include_tgsp_status(self):
        """Test inference metrics include TGSP compliance status."""
        from tensafe.inference import (
            InferenceConfig,
            LoRAMode,
            TenSafeInference,
        )

        config = InferenceConfig(
            lora_mode=LoRAMode.PLAINTEXT,
            enforce_tgsp=True,
        )

        inference = TenSafeInference(config=config)
        metrics = inference.get_metrics()

        assert "enforce_tgsp" in metrics
        assert "weights_from_tgsp" in metrics
        assert "tgsp_compliant" in metrics


class TestProductionGates:
    """Tests for TGSP production gates."""

    def test_tgsp_enforcement_gate_default(self):
        """Test TGSP enforcement gate is enabled by default."""
        from tensafe.core.gates import GateStatus, ProductionGates

        status = ProductionGates.TGSP_ENFORCEMENT.check()
        assert status in (GateStatus.ALLOWED, GateStatus.AUDIT)

    def test_tgsp_bypass_gate_denied_by_default(self):
        """Test TGSP bypass gate is denied by default."""
        from tensafe.core.gates import GateStatus, ProductionGates

        status = ProductionGates.TGSP_BYPASS.check()
        assert status == GateStatus.DENIED

    def test_tgsp_signature_skip_denied_by_default(self):
        """Test TGSP signature skip gate is denied by default."""
        from tensafe.core.gates import GateStatus, ProductionGates

        status = ProductionGates.TGSP_SIGNATURE_SKIP.check()
        assert status == GateStatus.DENIED

    def test_tgsp_gates_in_security_audit(self):
        """Test TGSP gates appear in security audit."""
        from tensafe.core.gates import security_audit

        audit = security_audit()

        assert "tgsp_enforcement" in audit["gates"]
        assert "tgsp_bypass" in audit["gates"]
        assert "tgsp_signature_skip" in audit["gates"]


class TestTGSPExceptions:
    """Tests for TGSP-related exceptions."""

    def test_tgsp_format_required_error_message(self):
        """Test TGSPFormatRequiredError has informative message."""
        from tensafe.tgsp_adapter_registry import TGSPFormatRequiredError

        error = TGSPFormatRequiredError("safetensors")
        assert "TGSP format" in str(error)
        assert "safetensors" in str(error)
        assert "tgsp build" in str(error)

    def test_adapter_not_loaded_error(self):
        """Test AdapterNotLoadedError message."""
        from tensafe.tgsp_adapter_registry import AdapterNotLoadedError

        error = AdapterNotLoadedError("Adapter 'test' not loaded")
        assert "test" in str(error)

    def test_no_active_adapter_error(self):
        """Test NoActiveAdapterError message."""
        from tensafe.tgsp_adapter_registry import NoActiveAdapterError

        error = NoActiveAdapterError("No adapter activated")
        assert "No adapter" in str(error)


class TestTGSPInferenceConfig:
    """Tests for InferenceConfig TGSP settings."""

    def test_config_requires_tgsp_for_he_modes(self):
        """Test config correctly identifies when TGSP is required."""
        from tensafe.inference import InferenceConfig, LoRAMode

        # HE_ONLY with enforcement
        config = InferenceConfig(lora_mode=LoRAMode.HE_ONLY, enforce_tgsp=True)
        assert config.requires_tgsp() is True

        # FULL_HE with enforcement
        config = InferenceConfig(lora_mode=LoRAMode.FULL_HE, enforce_tgsp=True)
        assert config.requires_tgsp() is True

        # Plaintext doesn't require TGSP
        config = InferenceConfig(lora_mode=LoRAMode.PLAINTEXT, enforce_tgsp=True)
        assert config.requires_tgsp() is False

        # HE mode without enforcement
        config = InferenceConfig(lora_mode=LoRAMode.HE_ONLY, enforce_tgsp=False)
        assert config.requires_tgsp() is False


class TestAdapterMetadata:
    """Tests for TGSPAdapterMetadata."""

    def test_metadata_serialization(self):
        """Test metadata serializes correctly."""
        from tensafe.tgsp_adapter_registry import TGSPAdapterMetadata

        metadata = TGSPAdapterMetadata(
            adapter_id="test-adapter",
            tgsp_path="/path/to/adapter.tgsp",
            model_name="test-model",
            model_version="1.0.0",
            author_id="test-author",
            manifest_hash="sha256:abc123",
            payload_hash="sha256:def456",
            signature_verified=True,
            signature_key_id="key-123",
            lora_rank=16,
            lora_alpha=32.0,
            target_modules=["q_proj", "v_proj"],
        )

        d = metadata.to_dict()

        assert d["adapter_id"] == "test-adapter"
        assert d["model_name"] == "test-model"
        assert d["signature_verified"] is True
        assert d["lora_rank"] == 16
        assert "loaded_at" in d


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_global_registry(self):
        """Test global registry accessor."""
        from tensafe.tgsp_adapter_registry import (
            get_global_registry,
            reset_global_registry,
        )

        # Reset first
        reset_global_registry()

        # Get registry
        registry1 = get_global_registry()
        registry2 = get_global_registry()

        # Should be same instance
        assert registry1 is registry2

        # Cleanup
        reset_global_registry()

    def test_reset_global_registry(self):
        """Test global registry reset."""
        from tensafe.tgsp_adapter_registry import (
            get_global_registry,
            reset_global_registry,
        )

        registry1 = get_global_registry()
        reset_global_registry()
        registry2 = get_global_registry()

        # Should be different instances
        assert registry1 is not registry2

        reset_global_registry()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
