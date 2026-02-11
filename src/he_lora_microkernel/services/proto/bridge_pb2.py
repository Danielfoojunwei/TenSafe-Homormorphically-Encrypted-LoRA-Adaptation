"""
Mock Protocol Buffer definitions for SchemeBridgeService.

This module provides Python-only mock implementations for development
and testing. In production, compile bridge.proto using grpc_tools.protoc.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

# =============================================================================
# Enums
# =============================================================================

class BridgeDirection(IntEnum):
    """Direction of bridge conversion."""
    CKKS_TO_TFHE = 0
    TFHE_TO_CKKS = 1


class ErrorCode(IntEnum):
    """Error codes for bridge operations."""
    ERROR_NONE = 0
    ERROR_INVALID_REQUEST = 1
    ERROR_DECRYPTION_FAILED = 2
    ERROR_ENCRYPTION_FAILED = 3
    ERROR_QUANTIZATION_OVERFLOW = 4
    ERROR_SESSION_NOT_FOUND = 5
    ERROR_DEADLINE_EXCEEDED = 6
    ERROR_INTERNAL = 99


class ResponseStatus(IntEnum):
    """Response status codes."""
    STATUS_OK = 0
    STATUS_ERROR = 1
    STATUS_PARTIAL = 2


# =============================================================================
# Messages
# =============================================================================

@dataclass
class QuantizationParams:
    """Quantization parameters for CKKS -> TFHE conversion."""
    bits: int = 8
    clip_min: float = -10.0
    clip_max: float = 10.0
    symmetric: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bits': self.bits,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'symmetric': self.symmetric,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuantizationParams':
        return cls(
            bits=d.get('bits', 8),
            clip_min=d.get('clip_min', -10.0),
            clip_max=d.get('clip_max', 10.0),
            symmetric=d.get('symmetric', True),
        )


@dataclass
class BridgeRequest:
    """Single bridge conversion request."""
    request_id: str = ""
    session_id: str = ""
    adapter_id: str = ""
    ciphertext_blob: bytes = b""
    direction: BridgeDirection = BridgeDirection.CKKS_TO_TFHE
    quant_params: Optional[QuantizationParams] = None
    layer_idx: int = 0
    token_idx: int = 0
    priority: int = 0
    deadline_ms: int = 5000

    def __post_init__(self):
        if self.quant_params is None:
            self.quant_params = QuantizationParams()


@dataclass
class BridgeMetrics:
    """Metrics for bridge operation."""
    decrypt_time_ms: float = 0.0
    quantize_time_ms: float = 0.0
    encrypt_time_ms: float = 0.0
    quantization_error: float = 0.0
    num_values: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'decrypt_time_ms': self.decrypt_time_ms,
            'quantize_time_ms': self.quantize_time_ms,
            'encrypt_time_ms': self.encrypt_time_ms,
            'quantization_error': self.quantization_error,
            'num_values': self.num_values,
            'total_time_ms': self.total_time_ms,
        }


@dataclass
class ErrorInfo:
    """Error information."""
    code: ErrorCode = ErrorCode.ERROR_NONE
    message: str = ""
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class BridgeResponse:
    """Single bridge conversion response."""
    request_id: str = ""
    ciphertext_blob: bytes = b""
    metrics: Optional[BridgeMetrics] = None
    error: Optional[ErrorInfo] = None
    status: ResponseStatus = ResponseStatus.STATUS_OK

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = BridgeMetrics()
        if self.error is None:
            self.error = ErrorInfo()


@dataclass
class BatchedBridgeRequest:
    """Batched bridge request."""
    session_id: str = ""
    adapter_id: str = ""
    requests: List[BridgeRequest] = field(default_factory=list)
    default_quant_params: Optional[QuantizationParams] = None


@dataclass
class BatchMetrics:
    """Aggregate metrics for batch."""
    successful: int = 0
    failed: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0


@dataclass
class BatchedBridgeResponse:
    """Batched bridge response."""
    responses: List[BridgeResponse] = field(default_factory=list)
    aggregate_metrics: Optional[BatchMetrics] = None


@dataclass
class HealthRequest:
    """Health check request."""
    service_id: str = ""


@dataclass
class HealthResponse:
    """Health check response."""
    healthy: bool = True
    version: str = "1.0.0"
    uptime_seconds: int = 0
    load: float = 0.0
    available_capacity: int = 100
