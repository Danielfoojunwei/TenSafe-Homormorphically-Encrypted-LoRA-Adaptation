"""
Mock gRPC Servicer and Stub for SchemeBridgeService.

This module provides Python-only mock implementations for development
and testing. In production, compile bridge.proto using grpc_tools.protoc.
"""

from typing import Any, Callable, Dict, Iterator, Optional
import logging
import time

from .bridge_pb2 import (
    BridgeRequest,
    BridgeResponse,
    BatchedBridgeRequest,
    BatchedBridgeResponse,
    HealthRequest,
    HealthResponse,
    BridgeDirection,
    BridgeMetrics,
    BatchMetrics,
    ErrorInfo,
    ErrorCode,
    ResponseStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Service Stub (Client)
# =============================================================================

class SchemeBridgeServiceStub:
    """
    Client stub for SchemeBridgeService.

    In production, this would be a gRPC stub. For testing,
    it can be connected to a mock servicer.
    """

    def __init__(self, channel: Any = None):
        """
        Initialize stub.

        Args:
            channel: gRPC channel (or mock servicer for testing)
        """
        self._channel = channel
        self._servicer: Optional['SchemeBridgeServiceServicer'] = None

        # If channel is a servicer, use it directly
        if isinstance(channel, SchemeBridgeServiceServicer):
            self._servicer = channel

    def CKKSDecryptAndEncryptToTFHE(
        self,
        request: BridgeRequest,
        timeout: Optional[float] = None,
    ) -> BridgeResponse:
        """Convert CKKS ciphertext to TFHE."""
        request.direction = BridgeDirection.CKKS_TO_TFHE

        if self._servicer is not None:
            return self._servicer.CKKSDecryptAndEncryptToTFHE(request, None)

        raise NotImplementedError("gRPC channel not implemented")

    def TFHEDecryptAndEncryptToCKKS(
        self,
        request: BridgeRequest,
        timeout: Optional[float] = None,
    ) -> BridgeResponse:
        """Convert TFHE ciphertext to CKKS."""
        request.direction = BridgeDirection.TFHE_TO_CKKS

        if self._servicer is not None:
            return self._servicer.TFHEDecryptAndEncryptToCKKS(request, None)

        raise NotImplementedError("gRPC channel not implemented")

    def BatchedBridge(
        self,
        request: BatchedBridgeRequest,
        timeout: Optional[float] = None,
    ) -> BatchedBridgeResponse:
        """Batched conversion."""
        if self._servicer is not None:
            return self._servicer.BatchedBridge(request, None)

        raise NotImplementedError("gRPC channel not implemented")

    def HealthCheck(
        self,
        request: HealthRequest,
        timeout: Optional[float] = None,
    ) -> HealthResponse:
        """Health check."""
        if self._servicer is not None:
            return self._servicer.HealthCheck(request, None)

        raise NotImplementedError("gRPC channel not implemented")

    def StreamingBridge(
        self,
        request_iterator: Iterator[BridgeRequest],
        timeout: Optional[float] = None,
    ) -> Iterator[BridgeResponse]:
        """Streaming bridge."""
        if self._servicer is not None:
            yield from self._servicer.StreamingBridge(request_iterator, None)
            return

        raise NotImplementedError("gRPC channel not implemented")


# =============================================================================
# Service Servicer (Server)
# =============================================================================

class SchemeBridgeServiceServicer:
    """
    Service implementation for SchemeBridgeService.

    This is the server-side implementation. In the interactive bridge model,
    this runs on the CLIENT side (the client holds the secret keys).

    Override methods to implement actual cryptographic operations.
    """

    def CKKSDecryptAndEncryptToTFHE(
        self,
        request: BridgeRequest,
        context: Any,
    ) -> BridgeResponse:
        """
        Convert CKKS ciphertext to TFHE.

        Steps:
        1. Decrypt CKKS ciphertext with client's CKKS secret key
        2. Quantize plaintext values
        3. Encrypt with client's TFHE secret key
        4. Return TFHE ciphertext

        Override this method for real cryptographic implementation.
        """
        raise NotImplementedError("Subclass must implement CKKSDecryptAndEncryptToTFHE")

    def TFHEDecryptAndEncryptToCKKS(
        self,
        request: BridgeRequest,
        context: Any,
    ) -> BridgeResponse:
        """
        Convert TFHE ciphertext to CKKS.

        Steps:
        1. Decrypt TFHE ciphertext with client's TFHE secret key
        2. Dequantize/convert to floating point
        3. Encrypt with client's CKKS secret key
        4. Return CKKS ciphertext

        Override this method for real cryptographic implementation.
        """
        raise NotImplementedError("Subclass must implement TFHEDecryptAndEncryptToCKKS")

    def BatchedBridge(
        self,
        request: BatchedBridgeRequest,
        context: Any,
    ) -> BatchedBridgeResponse:
        """
        Process multiple bridge requests.

        Default implementation processes sequentially.
        Override for parallel processing.
        """
        start_time = time.perf_counter()
        responses = []
        successful = 0
        failed = 0

        for req in request.requests:
            try:
                if req.direction == BridgeDirection.CKKS_TO_TFHE:
                    resp = self.CKKSDecryptAndEncryptToTFHE(req, context)
                else:
                    resp = self.TFHEDecryptAndEncryptToCKKS(req, context)

                responses.append(resp)

                if resp.status == ResponseStatus.STATUS_OK:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                failed += 1
                responses.append(BridgeResponse(
                    request_id=req.request_id,
                    status=ResponseStatus.STATUS_ERROR,
                    error=ErrorInfo(
                        code=ErrorCode.ERROR_INTERNAL,
                        message=str(e),
                    ),
                ))

        total_time = (time.perf_counter() - start_time) * 1000

        return BatchedBridgeResponse(
            responses=responses,
            aggregate_metrics=BatchMetrics(
                successful=successful,
                failed=failed,
                total_time_ms=total_time,
                avg_time_ms=total_time / len(request.requests) if request.requests else 0,
            ),
        )

    def HealthCheck(
        self,
        request: HealthRequest,
        context: Any,
    ) -> HealthResponse:
        """Health check - default implementation returns healthy."""
        return HealthResponse(
            healthy=True,
            version="1.0.0",
            uptime_seconds=0,
            load=0.0,
            available_capacity=100,
        )

    def StreamingBridge(
        self,
        request_iterator: Iterator[BridgeRequest],
        context: Any,
    ) -> Iterator[BridgeResponse]:
        """
        Streaming bridge for continuous conversion.

        Default implementation processes requests as they arrive.
        """
        for request in request_iterator:
            try:
                if request.direction == BridgeDirection.CKKS_TO_TFHE:
                    yield self.CKKSDecryptAndEncryptToTFHE(request, context)
                else:
                    yield self.TFHEDecryptAndEncryptToCKKS(request, context)
            except Exception as e:
                yield BridgeResponse(
                    request_id=request.request_id,
                    status=ResponseStatus.STATUS_ERROR,
                    error=ErrorInfo(
                        code=ErrorCode.ERROR_INTERNAL,
                        message=str(e),
                    ),
                )


# =============================================================================
# Mock Servicer for Testing
# =============================================================================

class MockSchemeBridgeServiceServicer(SchemeBridgeServiceServicer):
    """
    Mock implementation for testing.

    Simulates bridge operations without actual cryptography.
    Uses plaintext values serialized as bytes.
    """

    def __init__(
        self,
        quant_bits: int = 8,
        clip_min: float = -10.0,
        clip_max: float = 10.0,
    ):
        self._quant_bits = quant_bits
        self._clip_min = clip_min
        self._clip_max = clip_max
        self._start_time = time.time()

        # Statistics
        self._total_conversions = 0
        self._total_error = 0.0

    def CKKSDecryptAndEncryptToTFHE(
        self,
        request: BridgeRequest,
        context: Any,
    ) -> BridgeResponse:
        """Simulate CKKS -> TFHE conversion."""
        import numpy as np

        start = time.perf_counter()

        try:
            # Deserialize CKKS plaintext (simulated)
            t0 = time.perf_counter()
            ckks_values = np.frombuffer(request.ciphertext_blob, dtype=np.float64)
            decrypt_time = (time.perf_counter() - t0) * 1000

            # Quantize
            t0 = time.perf_counter()
            params = request.quant_params or QuantizationParams()
            quantized, max_error = self._quantize_values(ckks_values, params)
            quantize_time = (time.perf_counter() - t0) * 1000

            # Serialize TFHE ciphertext (simulated)
            t0 = time.perf_counter()
            tfhe_blob = quantized.astype(np.int32).tobytes()
            encrypt_time = (time.perf_counter() - t0) * 1000

            self._total_conversions += 1
            self._total_error += max_error

            return BridgeResponse(
                request_id=request.request_id,
                ciphertext_blob=tfhe_blob,
                metrics=BridgeMetrics(
                    decrypt_time_ms=decrypt_time,
                    quantize_time_ms=quantize_time,
                    encrypt_time_ms=encrypt_time,
                    quantization_error=max_error,
                    num_values=len(ckks_values),
                    total_time_ms=(time.perf_counter() - start) * 1000,
                ),
                status=ResponseStatus.STATUS_OK,
            )

        except Exception as e:
            return BridgeResponse(
                request_id=request.request_id,
                status=ResponseStatus.STATUS_ERROR,
                error=ErrorInfo(
                    code=ErrorCode.ERROR_INTERNAL,
                    message=str(e),
                ),
            )

    def TFHEDecryptAndEncryptToCKKS(
        self,
        request: BridgeRequest,
        context: Any,
    ) -> BridgeResponse:
        """Simulate TFHE -> CKKS conversion."""
        import numpy as np

        start = time.perf_counter()

        try:
            # Deserialize TFHE plaintext (simulated)
            t0 = time.perf_counter()
            tfhe_values = np.frombuffer(request.ciphertext_blob, dtype=np.int32)
            decrypt_time = (time.perf_counter() - t0) * 1000

            # Dequantize (or keep as-is for bit outputs)
            t0 = time.perf_counter()
            params = request.quant_params or QuantizationParams()

            # Check if this is a bit output (values are 0 or 1)
            if set(tfhe_values).issubset({0, 1, 2}):
                # Likely a LUT output, keep as-is
                ckks_values = tfhe_values.astype(np.float64)
            else:
                ckks_values = self._dequantize_values(tfhe_values, params)

            dequantize_time = (time.perf_counter() - t0) * 1000

            # Serialize CKKS ciphertext (simulated)
            t0 = time.perf_counter()
            ckks_blob = ckks_values.astype(np.float64).tobytes()
            encrypt_time = (time.perf_counter() - t0) * 1000

            self._total_conversions += 1

            return BridgeResponse(
                request_id=request.request_id,
                ciphertext_blob=ckks_blob,
                metrics=BridgeMetrics(
                    decrypt_time_ms=decrypt_time,
                    quantize_time_ms=dequantize_time,
                    encrypt_time_ms=encrypt_time,
                    quantization_error=0.0,
                    num_values=len(tfhe_values),
                    total_time_ms=(time.perf_counter() - start) * 1000,
                ),
                status=ResponseStatus.STATUS_OK,
            )

        except Exception as e:
            return BridgeResponse(
                request_id=request.request_id,
                status=ResponseStatus.STATUS_ERROR,
                error=ErrorInfo(
                    code=ErrorCode.ERROR_INTERNAL,
                    message=str(e),
                ),
            )

    def HealthCheck(
        self,
        request: HealthRequest,
        context: Any,
    ) -> HealthResponse:
        """Health check with stats."""
        uptime = int(time.time() - self._start_time)
        return HealthResponse(
            healthy=True,
            version="1.0.0-mock",
            uptime_seconds=uptime,
            load=0.1,
            available_capacity=100,
        )

    def _quantize_values(
        self,
        values: 'np.ndarray',
        params: 'QuantizationParams',
    ) -> tuple:
        """Quantize float values to integers."""
        import numpy as np

        # Clip
        clipped = np.clip(values, params.clip_min, params.clip_max)

        # Scale
        max_int = (1 << (params.bits - 1)) - 1
        max_abs = max(abs(params.clip_min), abs(params.clip_max))
        scale = max_int / max_abs

        quantized = np.round(clipped * scale).astype(np.int32)

        # Compute error
        dequantized = quantized / scale
        max_error = float(np.max(np.abs(values - dequantized)))

        return quantized, max_error

    def _dequantize_values(
        self,
        values: 'np.ndarray',
        params: 'QuantizationParams',
    ) -> 'np.ndarray':
        """Dequantize integers to floats."""
        import numpy as np

        max_int = (1 << (params.bits - 1)) - 1
        max_abs = max(abs(params.clip_min), abs(params.clip_max))
        scale = max_int / max_abs

        return values.astype(np.float64) / scale


# Helper to add servicer to gRPC server (mock implementation)
def add_SchemeBridgeServiceServicer_to_server(servicer, server):
    """Add servicer to gRPC server."""
    # In production, this would register the servicer with gRPC
    # For mock, we just store the reference
    if hasattr(server, '_servicers'):
        server._servicers['SchemeBridgeService'] = servicer
    else:
        server._servicers = {'SchemeBridgeService': servicer}
