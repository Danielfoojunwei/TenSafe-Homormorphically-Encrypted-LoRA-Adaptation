"""
TG-Tinker SDK Pydantic schemas for request/response models.

This module defines all data structures used in the SDK and API communication.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


# ==============================================================================
# Enums
# ==============================================================================


class FutureStatus(str, Enum):
    """Status of an async future."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationType(str, Enum):
    """Types of operations that can be performed."""

    FORWARD_BACKWARD = "forward_backward"
    OPTIM_STEP = "optim_step"
    SAMPLE = "sample"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"


class DPAccountantType(str, Enum):
    """Types of differential privacy accountants."""

    RDP = "rdp"
    MOMENTS = "moments"
    PRV = "prv"


class TrainingClientStatus(str, Enum):
    """Status of a training client."""

    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


# ==============================================================================
# Configuration Schemas
# ==============================================================================


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    rank: int = Field(default=16, ge=1, le=512, description="LoRA rank")
    alpha: float = Field(default=32.0, ge=1.0, description="LoRA alpha scaling")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0, description="Dropout rate")
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Target modules for LoRA",
    )
    bias: str = Field(
        default="none",
        description="Bias handling: 'none', 'all', or 'lora_only'",
    )

    @field_validator("bias")
    @classmethod
    def validate_bias(cls, v: str) -> str:
        if v not in ("none", "all", "lora_only"):
            raise ValueError("bias must be 'none', 'all', or 'lora_only'")
        return v


class OptimizerConfig(BaseModel):
    """Configuration for the optimizer."""

    name: str = Field(default="adamw", description="Optimizer name")
    learning_rate: float = Field(
        default=1e-4, gt=0, le=1.0, description="Learning rate"
    )
    weight_decay: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Weight decay"
    )
    betas: Tuple[float, float] = Field(
        default=(0.9, 0.999), description="Adam beta parameters"
    )
    eps: float = Field(default=1e-8, gt=0, description="Epsilon for numerical stability")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        allowed = {"adamw", "adam", "sgd", "adafactor"}
        if v.lower() not in allowed:
            raise ValueError(f"optimizer must be one of {allowed}")
        return v.lower()


class DPConfig(BaseModel):
    """Configuration for Differential Privacy."""

    enabled: bool = Field(default=True, description="Enable DP")
    noise_multiplier: float = Field(
        default=1.0, ge=0.0, description="Gaussian noise multiplier"
    )
    max_grad_norm: float = Field(
        default=1.0, gt=0.0, description="Max gradient norm for clipping (per-batch)"
    )
    target_epsilon: Optional[float] = Field(
        default=8.0, ge=0.0, description="Target epsilon budget"
    )
    target_delta: Optional[float] = Field(
        default=1e-5, gt=0.0, lt=1.0, description="Target delta"
    )
    accountant_type: DPAccountantType = Field(
        default=DPAccountantType.RDP, description="Privacy accountant type"
    )


class SamplingConfig(BaseModel):
    """Configuration for text sampling/generation."""

    max_tokens: int = Field(default=128, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    top_k: int = Field(default=50, ge=0, description="Top-k sampling")
    stop_sequences: List[str] = Field(default_factory=list, description="Stop sequences")


class TrainingConfig(BaseModel):
    """Full training configuration."""

    model_ref: str = Field(..., description="Model reference (HF hub or local path)")
    lora_config: Optional[LoRAConfig] = Field(
        default=None, description="LoRA configuration (None for full fine-tuning)"
    )
    optimizer: OptimizerConfig = Field(
        default_factory=OptimizerConfig, description="Optimizer configuration"
    )
    dp_config: Optional[DPConfig] = Field(
        default=None, description="Differential privacy configuration"
    )
    batch_size: int = Field(default=8, ge=1, description="Batch size")
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Gradient accumulation steps"
    )
    max_steps: Optional[int] = Field(
        default=None, ge=1, description="Maximum training steps"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Custom metadata"
    )


# ==============================================================================
# Request Schemas
# ==============================================================================


class CreateTrainingClientRequest(BaseModel):
    """Request to create a new training client."""

    model_ref: str
    lora_config: Optional[LoRAConfig] = None
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dp_config: Optional[DPConfig] = None
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchData(BaseModel):
    """Training batch data."""

    input_ids: List[List[int]] = Field(..., description="Input token IDs")
    attention_mask: List[List[int]] = Field(..., description="Attention mask")
    labels: Optional[List[List[int]]] = Field(None, description="Label token IDs")


class ForwardBackwardRequest(BaseModel):
    """Request for forward-backward pass."""

    batch: BatchData
    batch_hash: Optional[str] = Field(
        None, description="Client-side hash for verification"
    )


class OptimStepRequest(BaseModel):
    """Request for optimizer step."""

    apply_dp_noise: bool = Field(
        default=True, description="Apply DP noise if enabled"
    )


class SampleRequest(BaseModel):
    """Request for text sampling."""

    prompts: List[str] = Field(..., min_length=1, description="Prompts to sample from")
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = Field(default_factory=list)


class SaveStateRequest(BaseModel):
    """Request to save training state."""

    include_optimizer: bool = Field(default=True, description="Include optimizer state")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class LoadStateRequest(BaseModel):
    """Request to load training state."""

    artifact_id: str = Field(..., description="Artifact ID to load")


# ==============================================================================
# Response Schemas
# ==============================================================================


class DPMetrics(BaseModel):
    """Differential privacy metrics."""

    noise_applied: bool = Field(default=False)
    epsilon_spent: float = Field(default=0.0, description="Epsilon spent this step")
    total_epsilon: float = Field(default=0.0, description="Total epsilon spent")
    delta: float = Field(default=1e-5, description="Delta value")
    grad_norm_before_clip: Optional[float] = None
    grad_norm_after_clip: Optional[float] = None
    num_clipped: Optional[int] = None


class TrainingClientInfo(BaseModel):
    """Information about a training client."""

    training_client_id: str
    tenant_id: str
    model_ref: str
    status: TrainingClientStatus
    step: int
    created_at: datetime
    config: TrainingConfig
    dp_metrics: Optional[DPMetrics] = None


class CreateTrainingClientResponse(BaseModel):
    """Response from creating a training client."""

    training_client_id: str
    tenant_id: str
    model_ref: str
    status: TrainingClientStatus = TrainingClientStatus.READY
    step: int = 0
    created_at: datetime
    config: TrainingConfig


class FutureResponse(BaseModel):
    """Response containing future information."""

    future_id: str
    status: FutureStatus
    created_at: datetime
    training_client_id: str
    operation: OperationType
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ForwardBackwardResult(BaseModel):
    """Result of forward-backward pass."""

    loss: float
    grad_norm: float
    tokens_processed: int
    dp_metrics: Optional[DPMetrics] = None


class OptimStepResult(BaseModel):
    """Result of optimizer step."""

    step: int
    learning_rate: float
    dp_metrics: Optional[DPMetrics] = None


class SampleCompletion(BaseModel):
    """A single sample completion."""

    prompt: str
    completion: str
    tokens_generated: int
    finish_reason: str  # "stop", "length", "error"


class SampleResult(BaseModel):
    """Result of sampling operation."""

    samples: List[SampleCompletion]
    model_step: int
    sampling_config: SamplingConfig


class EncryptionInfo(BaseModel):
    """Information about artifact encryption."""

    algorithm: str = "AES-256-GCM"
    key_id: str


class SaveStateResult(BaseModel):
    """Result of save state operation."""

    artifact_id: str
    artifact_type: str = "checkpoint"
    size_bytes: int
    encryption: EncryptionInfo
    content_hash: str
    metadata: Dict[str, Any]
    created_at: datetime
    dp_metrics: Optional[DPMetrics] = None


class LoadStateResult(BaseModel):
    """Result of load state operation."""

    training_client_id: str
    loaded_artifact_id: str
    step: int
    status: TrainingClientStatus


class FutureResultResponse(BaseModel):
    """Response containing future result."""

    future_id: str
    status: FutureStatus
    result: Optional[Union[ForwardBackwardResult, OptimStepResult, SampleResult, SaveStateResult, LoadStateResult]] = None
    error: Optional[str] = None


# ==============================================================================
# Audit Log Schemas
# ==============================================================================


class AuditLogEntry(BaseModel):
    """An entry in the audit log."""

    entry_id: str
    tenant_id: str
    training_client_id: str
    operation: OperationType
    request_hash: str
    request_size_bytes: int
    artifact_ids_produced: List[str] = Field(default_factory=list)
    artifact_ids_consumed: List[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    success: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    prev_hash: str
    record_hash: str
    dp_metrics: Optional[DPMetrics] = None


# ==============================================================================
# Error Schemas
# ==============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response wrapper."""

    error: ErrorDetail
