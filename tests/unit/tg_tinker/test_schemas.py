"""
Unit tests for TG-Tinker SDK schemas.
"""

import sys
from pathlib import Path

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import pytest
from datetime import datetime

from tg_tinker.schemas import (
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
    TrainingConfig,
    BatchData,
    ForwardBackwardRequest,
    FutureStatus,
    OperationType,
    DPAccountantType,
    TrainingClientStatus,
    DPMetrics,
    ForwardBackwardResult,
    OptimStepResult,
    SaveStateResult,
    EncryptionInfo,
    AuditLogEntry,
)


class TestLoRAConfig:
    """Tests for LoRAConfig schema."""

    def test_defaults(self):
        """Test default values."""
        config = LoRAConfig()
        assert config.rank == 16
        assert config.alpha == 32.0
        assert config.dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj", "k_proj", "o_proj"]
        assert config.bias == "none"

    def test_custom_values(self):
        """Test custom values."""
        config = LoRAConfig(
            rank=8,
            alpha=16.0,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="all",
        )
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.1
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "all"

    def test_invalid_bias(self):
        """Test invalid bias value."""
        with pytest.raises(ValueError):
            LoRAConfig(bias="invalid")

    def test_rank_bounds(self):
        """Test rank bounds validation."""
        with pytest.raises(ValueError):
            LoRAConfig(rank=0)
        with pytest.raises(ValueError):
            LoRAConfig(rank=1000)


class TestOptimizerConfig:
    """Tests for OptimizerConfig schema."""

    def test_defaults(self):
        """Test default values."""
        config = OptimizerConfig()
        assert config.name == "adamw"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8

    def test_valid_optimizers(self):
        """Test valid optimizer names."""
        for name in ["adamw", "adam", "sgd", "adafactor"]:
            config = OptimizerConfig(name=name)
            assert config.name == name

    def test_invalid_optimizer(self):
        """Test invalid optimizer name."""
        with pytest.raises(ValueError):
            OptimizerConfig(name="invalid_optimizer")


class TestDPConfig:
    """Tests for DPConfig schema."""

    def test_defaults(self):
        """Test default values."""
        config = DPConfig()
        assert config.enabled is True
        assert config.noise_multiplier == 1.0
        assert config.max_grad_norm == 1.0
        assert config.target_epsilon == 8.0
        assert config.target_delta == 1e-5
        assert config.accountant_type == DPAccountantType.RDP

    def test_disabled(self):
        """Test disabled DP config."""
        config = DPConfig(enabled=False)
        assert config.enabled is False


class TestTrainingConfig:
    """Tests for TrainingConfig schema."""

    def test_minimal(self):
        """Test minimal config."""
        config = TrainingConfig(model_ref="test-model")
        assert config.model_ref == "test-model"
        assert config.lora_config is None
        assert config.dp_config is None
        assert config.batch_size == 8

    def test_full_config(self):
        """Test full config with all options."""
        config = TrainingConfig(
            model_ref="meta-llama/Llama-3-8B",
            lora_config=LoRAConfig(rank=16),
            optimizer=OptimizerConfig(learning_rate=2e-4),
            dp_config=DPConfig(noise_multiplier=1.5),
            batch_size=4,
            gradient_accumulation_steps=8,
            max_steps=1000,
            metadata={"experiment": "test"},
        )
        assert config.model_ref == "meta-llama/Llama-3-8B"
        assert config.lora_config.rank == 16
        assert config.optimizer.learning_rate == 2e-4
        assert config.dp_config.noise_multiplier == 1.5
        assert config.batch_size == 4
        assert config.metadata["experiment"] == "test"


class TestBatchData:
    """Tests for BatchData schema."""

    def test_valid_batch(self):
        """Test valid batch data."""
        batch = BatchData(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            attention_mask=[[1, 1, 1], [1, 1, 1]],
            labels=[[2, 3, -100], [5, 6, -100]],
        )
        assert len(batch.input_ids) == 2
        assert len(batch.attention_mask) == 2
        assert len(batch.labels) == 2

    def test_no_labels(self):
        """Test batch without labels."""
        batch = BatchData(
            input_ids=[[1, 2, 3]],
            attention_mask=[[1, 1, 1]],
        )
        assert batch.labels is None


class TestEnums:
    """Tests for enum schemas."""

    def test_future_status(self):
        """Test FutureStatus enum."""
        assert FutureStatus.PENDING == "pending"
        assert FutureStatus.RUNNING == "running"
        assert FutureStatus.COMPLETED == "completed"
        assert FutureStatus.FAILED == "failed"
        assert FutureStatus.CANCELLED == "cancelled"

    def test_operation_type(self):
        """Test OperationType enum."""
        assert OperationType.FORWARD_BACKWARD == "forward_backward"
        assert OperationType.OPTIM_STEP == "optim_step"
        assert OperationType.SAMPLE == "sample"
        assert OperationType.SAVE_STATE == "save_state"
        assert OperationType.LOAD_STATE == "load_state"


class TestResultSchemas:
    """Tests for result schemas."""

    def test_forward_backward_result(self):
        """Test ForwardBackwardResult schema."""
        result = ForwardBackwardResult(
            loss=2.5,
            grad_norm=1.2,
            tokens_processed=512,
        )
        assert result.loss == 2.5
        assert result.grad_norm == 1.2
        assert result.tokens_processed == 512
        assert result.dp_metrics is None

    def test_forward_backward_result_with_dp(self):
        """Test ForwardBackwardResult with DP metrics."""
        result = ForwardBackwardResult(
            loss=2.5,
            grad_norm=1.2,
            tokens_processed=512,
            dp_metrics=DPMetrics(
                noise_applied=True,
                epsilon_spent=0.1,
                total_epsilon=0.5,
            ),
        )
        assert result.dp_metrics.noise_applied is True
        assert result.dp_metrics.epsilon_spent == 0.1

    def test_optim_step_result(self):
        """Test OptimStepResult schema."""
        result = OptimStepResult(
            step=100,
            learning_rate=1e-4,
        )
        assert result.step == 100
        assert result.learning_rate == 1e-4

    def test_save_state_result(self):
        """Test SaveStateResult schema."""
        result = SaveStateResult(
            artifact_id="art-123",
            artifact_type="checkpoint",
            size_bytes=1024,
            encryption=EncryptionInfo(
                algorithm="AES-256-GCM",
                key_id="dek-123",
            ),
            content_hash="sha256:abc123",
            metadata={"step": 100},
            created_at=datetime.utcnow(),
        )
        assert result.artifact_id == "art-123"
        assert result.encryption.algorithm == "AES-256-GCM"


class TestAuditLogEntry:
    """Tests for AuditLogEntry schema."""

    def test_complete_entry(self):
        """Test complete audit log entry."""
        entry = AuditLogEntry(
            entry_id="aud-123",
            tenant_id="tenant-abc",
            training_client_id="tc-xyz",
            operation=OperationType.FORWARD_BACKWARD,
            request_hash="sha256:abc123",
            request_size_bytes=4096,
            artifact_ids_produced=["art-1", "art-2"],
            artifact_ids_consumed=[],
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            duration_ms=5000,
            success=True,
            prev_hash="sha256:prev123",
            record_hash="sha256:record123",
        )
        assert entry.entry_id == "aud-123"
        assert entry.success is True
        assert len(entry.artifact_ids_produced) == 2
