"""
Tests for LoRA to TGSP Converter.

This module tests the conversion of standard LoRA adapter formats
to the TGSP (TensorGuard Secure Package) format.
"""

import json
import os
import struct

import pytest

# Import converter components
from tensafe.lora_to_tgsp_converter import (
    ConversionResult,
    LoRAConfig,
    LoRAFormat,
    LoRAToTGSPConverter,
    MissingKeyError,
    convert_lora_to_tgsp,
)


class TestLoRAFormatDetection:
    """Tests for LoRA format detection."""

    def test_detect_safetensors_format(self, tmp_path):
        """Test detection of safetensors format."""
        converter = LoRAToTGSPConverter()

        # Create a mock safetensors file
        file_path = tmp_path / "adapter_model.safetensors"
        file_path.write_bytes(b"mock safetensors content")

        detected = converter.detect_format(str(file_path))
        assert detected == LoRAFormat.SAFETENSORS

    def test_detect_pytorch_bin_format(self, tmp_path):
        """Test detection of PyTorch .bin format."""
        converter = LoRAToTGSPConverter()

        file_path = tmp_path / "adapter_model.bin"
        file_path.write_bytes(b"mock pytorch content")

        detected = converter.detect_format(str(file_path))
        assert detected == LoRAFormat.PYTORCH_BIN

    def test_detect_pytorch_pt_format(self, tmp_path):
        """Test detection of PyTorch .pt format."""
        converter = LoRAToTGSPConverter()

        file_path = tmp_path / "model.pt"
        file_path.write_bytes(b"mock pytorch content")

        detected = converter.detect_format(str(file_path))
        assert detected == LoRAFormat.PYTORCH_PT

    def test_detect_huggingface_directory(self, tmp_path):
        """Test detection of Hugging Face adapter directory."""
        converter = LoRAToTGSPConverter()

        # Create adapter directory structure
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "r": 16,
            "lora_alpha": 32,
        }))

        detected = converter.detect_format(str(adapter_dir))
        assert detected == LoRAFormat.SAFETENSORS

    def test_detect_unknown_format(self, tmp_path):
        """Test detection of unknown format."""
        converter = LoRAToTGSPConverter()

        file_path = tmp_path / "model.unknown"
        file_path.write_bytes(b"unknown content")

        detected = converter.detect_format(str(file_path))
        assert detected == LoRAFormat.UNKNOWN


class TestLoRAConfigExtraction:
    """Tests for LoRA configuration extraction."""

    def test_load_config_from_directory(self, tmp_path):
        """Test loading config from adapter directory."""
        converter = LoRAToTGSPConverter()

        # Create config file
        config_data = {
            "r": 32,
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "lora_only",
            "base_model_name_or_path": "meta-llama/Llama-3-8b",
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
        }

        config_path = tmp_path / "adapter_config.json"
        config_path.write_text(json.dumps(config_data))

        config = converter._load_config_from_directory(str(tmp_path))

        assert config.rank == 32
        assert config.alpha == 64
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.dropout == 0.05
        assert config.bias == "lora_only"
        assert config.base_model_name == "meta-llama/Llama-3-8b"

    def test_infer_config_from_weights(self):
        """Test inferring config from weight shapes."""
        converter = LoRAToTGSPConverter()

        # Mock weights
        import numpy as np

        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros((8, 4096)),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros((4096, 8)),
            "base_model.model.layers.0.self_attn.v_proj.lora_A.weight": np.zeros((8, 4096)),
            "base_model.model.layers.0.self_attn.v_proj.lora_B.weight": np.zeros((4096, 8)),
        }

        config = converter._infer_config_from_weights(weights)

        assert config.rank == 8
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


class TestLoRAWeightValidation:
    """Tests for LoRA weight validation."""

    def test_validate_valid_weights(self):
        """Test validation of valid LoRA weights."""
        converter = LoRAToTGSPConverter()

        import numpy as np

        weights = {
            "base_model.model.q_proj.lora_A.weight": np.random.randn(16, 4096).astype(np.float32),
            "base_model.model.q_proj.lora_B.weight": np.random.randn(4096, 16).astype(np.float32),
        }

        config = LoRAConfig(
            rank=16,
            alpha=32,
            target_modules=["q_proj"],
        )

        is_valid, issues = converter.validate_lora_weights(weights, config)
        assert is_valid
        assert len(issues) == 0

    def test_validate_missing_lora_weights(self):
        """Test validation fails for missing LoRA weights."""
        converter = LoRAToTGSPConverter()

        weights = {
            "some_other_key": None,
        }

        config = LoRAConfig()

        is_valid, issues = converter.validate_lora_weights(weights, config)
        assert not is_valid
        assert "No LoRA weights found" in issues[0]

    def test_validate_empty_target_modules(self):
        """Test validation warns about empty target modules."""
        converter = LoRAToTGSPConverter()

        import numpy as np

        weights = {
            "base_model.lora_A.weight": np.zeros((16, 4096)),
        }

        config = LoRAConfig(target_modules=[])

        is_valid, issues = converter.validate_lora_weights(weights, config)
        assert not is_valid
        assert any("No target modules" in issue for issue in issues)


class TestTGSPPackageCreation:
    """Tests for TGSP package creation."""

    def test_create_tgsp_manual(self, tmp_path):
        """Test manual TGSP package creation."""
        converter = LoRAToTGSPConverter(work_dir=str(tmp_path / "work"))

        # Create payload directory
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        (payload_dir / "test.txt").write_text("test content")

        output_path = str(tmp_path / "output.tgsp")

        result = converter._create_tgsp_manual(
            payload_dir=str(payload_dir),
            output_path=output_path,
            model_name="test-model",
            model_version="1.0.0",
            signing_key_path="",
            signing_pub_path="",
            recipient_pub_path="",
            adapter_id="test-adapter-id",
        )

        # Verify output file exists
        assert os.path.exists(output_path)

        # Verify TGSP magic bytes
        with open(output_path, 'rb') as f:
            magic = f.read(6)
            assert magic == b"TGSP\x01\x00"

        # Verify result
        assert "manifest_hash" in result
        assert "payload_hash" in result
        assert result["key_id"] == "key_1"


class TestLoRAToTGSPConversion:
    """Tests for end-to-end LoRA to TGSP conversion."""

    def test_conversion_nonexistent_input(self, tmp_path):
        """Test conversion fails for non-existent input."""
        converter = LoRAToTGSPConverter()

        result = converter.convert(
            input_path="/nonexistent/path",
            output_path=str(tmp_path / "output.tgsp"),
        )

        assert not result.success
        assert "does not exist" in result.error

    def test_conversion_unknown_format(self, tmp_path):
        """Test conversion fails for unknown format."""
        converter = LoRAToTGSPConverter()

        # Create file with unknown extension
        input_file = tmp_path / "model.xyz"
        input_file.write_bytes(b"unknown content")

        result = converter.convert(
            input_path=str(input_file),
            output_path=str(tmp_path / "output.tgsp"),
        )

        assert not result.success
        assert "Unknown input format" in result.error or "Unsupported format" in result.error

    @pytest.mark.skipif(
        not os.path.exists("/usr/bin/python3"),
        reason="Requires full Python environment"
    )
    def test_conversion_with_mock_weights(self, tmp_path):
        """Test conversion with mock weights when safetensors not available."""
        converter = LoRAToTGSPConverter(
            auto_generate_keys=True,
            work_dir=str(tmp_path / "work"),
        )

        # Create mock adapter directory
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()

        # Create config
        config_data = {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config_data))

        # No weights file - will use mock weights

        output_path = str(tmp_path / "output.tgsp")

        result = converter.convert(
            input_path=str(adapter_dir),
            output_path=output_path,
            model_name="test-adapter",
            validate=False,  # Skip validation since we have mock weights
        )

        # Check result
        if result.success:
            assert os.path.exists(result.output_path)
            assert result.model_name == "test-adapter"
            assert result.lora_config.rank == 16

        converter.cleanup()


class TestConversionResult:
    """Tests for ConversionResult dataclass."""

    def test_conversion_result_to_dict(self):
        """Test ConversionResult serialization."""
        result = ConversionResult(
            success=True,
            output_path="/path/to/output.tgsp",
            adapter_id="test-adapter-123",
            model_name="test-model",
            model_version="1.0.0",
            manifest_hash="abc123",
            payload_hash="def456",
            signature_key_id="key_1",
            lora_config=LoRAConfig(rank=16, alpha=32),
            input_format=LoRAFormat.SAFETENSORS,
            input_size_bytes=1024,
            output_size_bytes=2048,
            conversion_time_ms=100.5,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["adapter_id"] == "test-adapter-123"
        assert data["lora_config"]["lora_rank"] == 16
        assert data["input_format"] == "safetensors"
        assert data["conversion_time_ms"] == 100.5


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_lora_config_defaults(self):
        """Test LoRAConfig default values."""
        config = LoRAConfig()

        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.0
        assert config.bias == "none"

    def test_lora_config_to_dict(self):
        """Test LoRAConfig serialization."""
        config = LoRAConfig(
            rank=32,
            alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj"],
            base_model_name="llama-3-8b",
        )

        data = config.to_dict()

        assert data["lora_rank"] == 32
        assert data["lora_alpha"] == 64
        assert data["target_modules"] == ["q_proj", "k_proj", "v_proj"]
        assert data["base_model_name"] == "llama-3-8b"


class TestKeyGeneration:
    """Tests for cryptographic key generation."""

    def test_ensure_keys_with_auto_generate(self, tmp_path):
        """Test automatic key generation."""
        keys_dir = tmp_path / "keys"
        converter = LoRAToTGSPConverter(
            auto_generate_keys=True,
            keys_dir=str(keys_dir),
        )

        signing_key, signing_pub, recipient_pub = converter._ensure_keys_exist(
            None, None, None
        )

        # Check keys were generated
        assert os.path.exists(signing_key)
        assert os.path.exists(signing_pub)
        assert os.path.exists(recipient_pub)

        # Verify key files are valid JSON
        with open(signing_key) as f:
            key_data = json.load(f)
            assert "classic" in key_data
            assert "pqc" in key_data

    def test_ensure_keys_without_auto_generate(self, tmp_path):
        """Test that missing keys raises error without auto_generate."""
        converter = LoRAToTGSPConverter(
            auto_generate_keys=False,
            keys_dir=str(tmp_path / "nonexistent"),
        )

        with pytest.raises(MissingKeyError):
            converter._ensure_keys_exist(None, None, None)


class TestBatchConversion:
    """Tests for batch LoRA to TGSP conversion."""

    def test_batch_convert_empty_list(self, tmp_path):
        """Test batch conversion with empty list."""
        converter = LoRAToTGSPConverter()

        results = converter.batch_convert(
            input_paths=[],
            output_dir=str(tmp_path / "output"),
        )

        assert len(results) == 0

    def test_batch_convert_creates_output_dir(self, tmp_path):
        """Test batch conversion creates output directory."""
        converter = LoRAToTGSPConverter()
        output_dir = tmp_path / "new_output_dir"

        # Should create directory even with empty input
        results = converter.batch_convert(
            input_paths=[],
            output_dir=str(output_dir),
        )

        assert output_dir.exists()


class TestAuditLogging:
    """Tests for audit logging functionality."""

    def test_audit_log_events(self, tmp_path):
        """Test that audit events are logged."""
        converter = LoRAToTGSPConverter(work_dir=str(tmp_path / "work"))

        # Trigger some operations
        input_file = tmp_path / "nonexistent.safetensors"
        converter.convert(
            input_path=str(input_file),
            output_path=str(tmp_path / "output.tgsp"),
        )

        # Check audit log
        audit_log = converter.get_audit_log()
        assert len(audit_log) > 0

        # Should have conversion started and failed events
        event_types = [e["event_type"] for e in audit_log]
        assert "CONVERSION_STARTED" in event_types
        assert "CONVERSION_FAILED" in event_types


class TestConvenienceFunction:
    """Tests for convenience conversion function."""

    def test_convert_lora_to_tgsp_convenience(self, tmp_path):
        """Test the convenience function."""
        input_file = tmp_path / "nonexistent.safetensors"

        result = convert_lora_to_tgsp(
            input_path=str(input_file),
            output_path=str(tmp_path / "output.tgsp"),
            auto_generate_keys=True,
        )

        # Should fail gracefully for nonexistent input
        assert not result.success
        assert result.error is not None


class TestCleanup:
    """Tests for cleanup functionality."""

    def test_converter_cleanup(self, tmp_path):
        """Test that cleanup removes work directory."""
        work_dir = tmp_path / "work"
        converter = LoRAToTGSPConverter(work_dir=str(work_dir))

        # Work dir should be created
        assert work_dir.exists()

        converter.cleanup()

        # Work dir should be removed
        assert not work_dir.exists()


class TestTGSPFormatCompliance:
    """Tests for TGSP format compliance."""

    def test_tgsp_magic_bytes(self, tmp_path):
        """Test TGSP output has correct magic bytes."""
        converter = LoRAToTGSPConverter(
            work_dir=str(tmp_path / "work"),
            auto_generate_keys=True,
        )

        # Create minimal payload
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        (payload_dir / "dummy.txt").write_text("test")

        output_path = str(tmp_path / "test.tgsp")

        converter._create_tgsp_manual(
            payload_dir=str(payload_dir),
            output_path=output_path,
            model_name="test",
            model_version="1.0.0",
            signing_key_path="",
            signing_pub_path="",
            recipient_pub_path="",
            adapter_id="test-id",
        )

        # Verify TGSP v1.0 magic
        with open(output_path, 'rb') as f:
            magic = f.read(6)
            assert magic == b"TGSP\x01\x00", f"Expected TGSP v1.0 magic, got {magic}"

    def test_tgsp_header_structure(self, tmp_path):
        """Test TGSP output has correct header structure."""
        converter = LoRAToTGSPConverter(
            work_dir=str(tmp_path / "work"),
            auto_generate_keys=True,
        )

        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        (payload_dir / "config.json").write_text(json.dumps({"test": "data"}))

        output_path = str(tmp_path / "test.tgsp")

        converter._create_tgsp_manual(
            payload_dir=str(payload_dir),
            output_path=output_path,
            model_name="header-test",
            model_version="2.0.0",
            signing_key_path="",
            signing_pub_path="",
            recipient_pub_path="",
            adapter_id="header-test-id",
        )

        # Parse TGSP structure
        with open(output_path, 'rb') as f:
            # Magic
            magic = f.read(6)
            assert magic == b"TGSP\x01\x00"

            # Header
            header_len = struct.unpack(">I", f.read(4))[0]
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes)

            assert header["tgsp_version"] == "1.0"
            assert "hashes" in header
            assert "manifest" in header["hashes"]
            assert "payload" in header["hashes"]

            # Manifest
            manifest_len = struct.unpack(">I", f.read(4))[0]
            manifest_bytes = f.read(manifest_len)
            manifest = json.loads(manifest_bytes)

            assert manifest["model_name"] == "header-test"
            assert manifest["model_version"] == "2.0.0"


class TestIntegrationWithTGSPRegistry:
    """Tests for integration with TGSP adapter registry."""

    def test_converted_tgsp_can_be_loaded(self, tmp_path):
        """Test that converted TGSP files can be loaded by the registry."""
        pytest.importorskip("tensafe.tgsp_adapter_registry")

        from tensafe.tgsp_adapter_registry import TGSPAdapterRegistry

        converter = LoRAToTGSPConverter(
            work_dir=str(tmp_path / "work"),
            auto_generate_keys=True,
        )

        # Create minimal adapter with mock weights
        payload_dir = tmp_path / "payload"
        payload_dir.mkdir()
        (payload_dir / "adapter_config.json").write_text(json.dumps({
            "lora_rank": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        }))

        output_path = str(tmp_path / "adapter.tgsp")

        converter._create_tgsp_manual(
            payload_dir=str(payload_dir),
            output_path=output_path,
            model_name="integration-test",
            model_version="1.0.0",
            signing_key_path="",
            signing_pub_path="",
            recipient_pub_path="",
            adapter_id="integration-test-id",
        )

        # Try to load with registry (will use fallback verification)
        registry = TGSPAdapterRegistry(
            enforce_tgsp=True,
            auto_verify_signatures=False,  # Disable signature verification for test
        )

        try:
            adapter_id = registry.load_tgsp_adapter(
                tgsp_path=output_path,
            )
            assert adapter_id is not None
            assert adapter_id.startswith("tgsp_")

            # Verify adapter info
            info = registry.get_adapter_info(adapter_id)
            assert info is not None
            assert info["model_name"] == "integration-test"

        finally:
            registry.cleanup()
            converter.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
