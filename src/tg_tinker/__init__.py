"""
TG-Tinker SDK - Privacy-first ML training API.

A privacy-first alternative to Thinking Machines' Tinker API,
providing encrypted artifacts, signed requests, immutable audit logs,
and optional differential privacy.

Example:
    >>> from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig
    >>>
    >>> # Initialize client (uses TG_TINKER_API_KEY env var)
    >>> service = ServiceClient()
    >>>
    >>> # Create training client
    >>> config = TrainingConfig(
    ...     model_ref="meta-llama/Llama-3-8B",
    ...     lora_config=LoRAConfig(rank=16, alpha=32)
    ... )
    >>> tc = service.create_training_client(config)
    >>>
    >>> # Training loop with async execution
    >>> for batch in dataloader:
    ...     fb_future = tc.forward_backward(batch)
    ...     opt_future = tc.optim_step()
    ...     result = fb_future.result()
    ...     print(f"Loss: {result.loss}")
    >>>
    >>> # Save checkpoint
    >>> checkpoint = tc.save_state()
    >>> print(f"Saved: {checkpoint.artifact_id}")
"""

__version__ = "1.0.0"

# Client
from .client import ServiceClient

# Training Client
from .training_client import TrainingClient

# Futures
from .futures import FutureHandle

# Configuration
from .config import TenSafeConfig, get_config

# Schemas
from .schemas import (
    # Enums
    FutureStatus,
    OperationType,
    DPAccountantType,
    TrainingClientStatus,
    # Config
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
    SamplingConfig,
    TrainingConfig,
    # Request/Response
    BatchData,
    ForwardBackwardRequest,
    OptimStepRequest,
    SampleRequest,
    SaveStateRequest,
    LoadStateRequest,
    # Results
    DPMetrics,
    TrainingClientInfo,
    ForwardBackwardResult,
    OptimStepResult,
    SampleResult,
    SampleCompletion,
    SaveStateResult,
    LoadStateResult,
    EncryptionInfo,
    # Audit
    AuditLogEntry,
)

# Exceptions
from .exceptions import (
    TGTinkerError,
    AuthenticationError,
    PermissionDeniedError,
    TrainingClientNotFoundError,
    FutureNotFoundError,
    ArtifactNotFoundError,
    ValidationError,
    RateLimitedError,
    QueueFullError,
    FutureTimeoutError,
    FutureCancelledError,
    FutureFailedError,
    DPBudgetExceededError,
    ServerError,
    ConnectionError,
)

__all__ = [
    # Version
    "__version__",
    # Client
    "ServiceClient",
    "TrainingClient",
    "FutureHandle",
    # Config
    "TenSafeConfig",
    "get_config",
    # Enums
    "FutureStatus",
    "OperationType",
    "DPAccountantType",
    "TrainingClientStatus",
    # Configuration
    "LoRAConfig",
    "OptimizerConfig",
    "DPConfig",
    "SamplingConfig",
    "TrainingConfig",
    # Request/Response
    "BatchData",
    "ForwardBackwardRequest",
    "OptimStepRequest",
    "SampleRequest",
    "SaveStateRequest",
    "LoadStateRequest",
    # Results
    "DPMetrics",
    "TrainingClientInfo",
    "ForwardBackwardResult",
    "OptimStepResult",
    "SampleResult",
    "SampleCompletion",
    "SaveStateResult",
    "LoadStateResult",
    "EncryptionInfo",
    # Audit
    "AuditLogEntry",
    # Exceptions
    "TGTinkerError",
    "AuthenticationError",
    "PermissionDeniedError",
    "TrainingClientNotFoundError",
    "FutureNotFoundError",
    "ArtifactNotFoundError",
    "ValidationError",
    "RateLimitedError",
    "QueueFullError",
    "FutureTimeoutError",
    "FutureCancelledError",
    "FutureFailedError",
    "DPBudgetExceededError",
    "ServerError",
    "ConnectionError",
]
