"""
HAS Protocol Buffer Message Definitions (Python Mock)

This module provides Python-native message classes that mirror the
protobuf message definitions for development without protoc.

For production, generate from has.proto:
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. has.proto
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# ADAPTER MANAGEMENT MESSAGES
# ============================================================================

@dataclass
class LoadAdapterRequest:
    """Request to load a LoRA adapter into HAS."""
    adapter_id: str = ""
    model_id: str = ""
    adapter_path: str = ""
    rank: int = 16
    alpha: float = 32.0
    targets: str = "qkv"
    layers: List[int] = field(default_factory=list)


@dataclass
class LoadAdapterResponse:
    """Response from LoadAdapter RPC."""
    success: bool = False
    error_message: str = ""
    num_layers: int = 0
    loaded_layers: List[int] = field(default_factory=list)
    memory_usage_mb: float = 0.0


@dataclass
class UnloadAdapterRequest:
    """Request to unload a LoRA adapter."""
    adapter_id: str = ""


@dataclass
class UnloadAdapterResponse:
    """Response from UnloadAdapter RPC."""
    success: bool = False
    error_message: str = ""


@dataclass
class GetAdapterInfoRequest:
    """Request adapter information."""
    adapter_id: str = ""


@dataclass
class AdapterInfo:
    """Information about a loaded adapter."""
    adapter_id: str = ""
    model_id: str = ""
    rank: int = 0
    alpha: float = 0.0
    targets: str = ""
    num_layers: int = 0
    loaded_layers: List[int] = field(default_factory=list)
    memory_usage_mb: float = 0.0


@dataclass
class GetAdapterInfoResponse:
    """Response from GetAdapterInfo RPC."""
    success: bool = False
    error_message: str = ""
    info: Optional[AdapterInfo] = None


# ============================================================================
# REQUEST LIFECYCLE MESSAGES
# ============================================================================

@dataclass
class PrepareRequestRequest:
    """Request to prepare HE context for inference request."""
    request_id: str = ""
    adapter_id: str = ""
    batch_size: int = 1
    seq_len: int = 0
    shm_region: str = ""


@dataclass
class PrepareRequestResponse:
    """Response from PrepareRequest RPC."""
    success: bool = False
    error_message: str = ""
    shm_region: str = ""
    shm_offset: int = 0
    buffer_size: int = 0


@dataclass
class ReleaseRequestRequest:
    """Request to release HE context."""
    request_id: str = ""


@dataclass
class ReleaseRequestResponse:
    """Response from ReleaseRequest RPC."""
    success: bool = False
    error_message: str = ""


# ============================================================================
# TOKEN STEP MESSAGES
# ============================================================================

@dataclass
class ApplyTokenStepRequest:
    """Request to apply HE-LoRA delta for a token step."""
    request_id: str = ""
    layer_idx: int = 0
    projection_type: str = ""  # "q", "k", "v", "o"
    token_idx: int = 0
    # Hidden states passed via shared memory


@dataclass
class ApplyTokenStepResponse:
    """Response from ApplyTokenStep RPC."""
    success: bool = False
    error_message: str = ""
    has_delta: bool = False
    shm_offset: int = 0  # Offset in shared memory where delta is written
    # Timing information
    encrypt_time_us: int = 0
    compute_time_us: int = 0
    decrypt_time_us: int = 0


@dataclass
class ApplyBatchedTokenStepRequest:
    """Request to apply HE-LoRA deltas for multiple layers/projections."""
    request_id: str = ""
    token_idx: int = 0
    layer_projections: List[str] = field(default_factory=list)  # ["0_q", "0_k", ...]


@dataclass
class DeltaResult:
    """Result for a single delta computation."""
    layer_idx: int = 0
    projection_type: str = ""
    has_delta: bool = False
    shm_offset: int = 0


@dataclass
class ApplyBatchedTokenStepResponse:
    """Response from ApplyBatchedTokenStep RPC."""
    success: bool = False
    error_message: str = ""
    results: List[DeltaResult] = field(default_factory=list)
    total_time_us: int = 0


# ============================================================================
# HEALTH AND STATUS MESSAGES
# ============================================================================

@dataclass
class HealthCheckRequest:
    """Health check request."""
    pass


@dataclass
class HealthCheckResponse:
    """Health check response."""
    healthy: bool = False
    message: str = ""
    uptime_seconds: float = 0.0


@dataclass
class GetStatusRequest:
    """Status request."""
    pass


@dataclass
class GetStatusResponse:
    """Detailed status response."""
    healthy: bool = False
    loaded_adapters: int = 0
    active_requests: int = 0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    gpu_utilization: float = 0.0
    # HE statistics
    total_tokens_processed: int = 0
    avg_encrypt_time_us: float = 0.0
    avg_compute_time_us: float = 0.0
    avg_decrypt_time_us: float = 0.0


# ============================================================================
# TELEMETRY MESSAGES
# ============================================================================

@dataclass
class TelemetryData:
    """Telemetry data point."""
    timestamp_us: int = 0
    request_id: str = ""
    token_idx: int = 0
    layer_idx: int = 0
    projection_type: str = ""
    encrypt_time_us: int = 0
    compute_time_us: int = 0
    decrypt_time_us: int = 0
    rotations: int = 0
    keyswitches: int = 0
    rescales: int = 0


@dataclass
class GetTelemetryRequest:
    """Request telemetry data."""
    request_id: str = ""
    start_token: int = 0
    end_token: int = -1  # -1 means all


@dataclass
class GetTelemetryResponse:
    """Telemetry response."""
    success: bool = False
    data: List[TelemetryData] = field(default_factory=list)
