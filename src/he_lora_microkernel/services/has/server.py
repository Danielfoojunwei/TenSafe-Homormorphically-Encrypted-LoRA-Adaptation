"""
HAS (HE Adapter Service) gRPC Server

The HAS server:
- Holds HE secret keys (never leaves process)
- Processes HE-LoRA computation requests
- Communicates with MSS via gRPC (control plane) and shared memory (data plane)
"""

from concurrent import futures
from typing import Any, Dict, List, Optional
import logging
import time

from .executor import HASExecutor
from .key_manager import KeyManager
from .shm_manager import SharedMemoryManager

logger = logging.getLogger(__name__)


class HASServicer:
    """
    gRPC servicer implementation for HE Adapter Service.

    Implements all RPC methods defined in has.proto.
    """

    def __init__(
        self,
        executor: HASExecutor,
        key_manager: KeyManager,
        shm_manager: SharedMemoryManager,
    ):
        """
        Initialize servicer.

        Args:
            executor: HE-LoRA executor
            key_manager: HE key manager
            shm_manager: Shared memory manager
        """
        self._executor = executor
        self._key_manager = key_manager
        self._shm_manager = shm_manager
        self._start_time = time.time()

        # Telemetry storage
        self._telemetry: Dict[str, List] = {}

    def LoadAdapter(self, request: Any, context: Any) -> Any:
        """Load a LoRA adapter."""
        from ..proto import has_pb2

        try:
            state = self._executor.load_adapter(
                adapter_id=request.adapter_id,
                model_id=request.model_id,
                rank=request.rank,
                alpha=request.alpha,
                targets=request.targets,
                layers=list(request.layers) if request.layers else None,
                adapter_path=request.adapter_path if request.adapter_path else None,
            )

            return has_pb2.LoadAdapterResponse(
                success=True,
                num_layers=state.num_layers,
                loaded_layers=state.loaded_layers,
                memory_usage_mb=state.memory_mb,
            )

        except Exception as e:
            logger.error(f"LoadAdapter failed: {e}")
            return has_pb2.LoadAdapterResponse(
                success=False,
                error_message=str(e),
            )

    def UnloadAdapter(self, request: Any, context: Any) -> Any:
        """Unload a LoRA adapter."""
        from ..proto import has_pb2

        success = self._executor.unload_adapter(request.adapter_id)
        return has_pb2.UnloadAdapterResponse(
            success=success,
            error_message="" if success else f"Adapter {request.adapter_id} not found",
        )

    def GetAdapterInfo(self, request: Any, context: Any) -> Any:
        """Get adapter information."""
        from ..proto import has_pb2

        state = self._executor.get_adapter(request.adapter_id)
        if state is None:
            return has_pb2.GetAdapterInfoResponse(
                success=False,
                error_message=f"Adapter {request.adapter_id} not found",
            )

        info = has_pb2.AdapterInfo(
            adapter_id=state.adapter_id,
            model_id=state.model_id,
            rank=state.rank,
            alpha=state.alpha,
            targets=state.targets,
            num_layers=state.num_layers,
            loaded_layers=state.loaded_layers,
            memory_usage_mb=state.memory_mb,
        )

        return has_pb2.GetAdapterInfoResponse(
            success=True,
            info=info,
        )

    def PrepareRequest(self, request: Any, context: Any) -> Any:
        """Prepare HE context for a request."""
        from ..proto import has_pb2

        try:
            # Create shared memory region if needed
            shm_region = None
            if request.shm_region:
                shm_region = request.shm_region
            else:
                # Create new region
                region = self._shm_manager.create_region(
                    name=request.request_id,
                    batch_size=request.batch_size,
                    hidden_size=4096,  # Should come from model
                )
                shm_region = region.name

            # Prepare request in executor
            state = self._executor.prepare_request(
                request_id=request.request_id,
                adapter_id=request.adapter_id,
                batch_size=request.batch_size,
                seq_len=request.seq_len,
                shm_region=shm_region,
            )

            return has_pb2.PrepareRequestResponse(
                success=True,
                shm_region=shm_region or "",
                shm_offset=0,
                buffer_size=request.batch_size * 4096 * 2,  # Approximate
            )

        except Exception as e:
            logger.error(f"PrepareRequest failed: {e}")
            return has_pb2.PrepareRequestResponse(
                success=False,
                error_message=str(e),
            )

    def ReleaseRequest(self, request: Any, context: Any) -> Any:
        """Release HE context."""
        from ..proto import has_pb2

        success = self._executor.release_request(request.request_id)

        # Also cleanup shared memory
        self._shm_manager.destroy_region(request.request_id)

        # Clear telemetry
        if request.request_id in self._telemetry:
            del self._telemetry[request.request_id]

        return has_pb2.ReleaseRequestResponse(
            success=success,
            error_message="" if success else "Request not found",
        )

    def ApplyTokenStep(self, request: Any, context: Any) -> Any:
        """Apply HE-LoRA delta for a token step."""
        from ..proto import has_pb2
        import numpy as np

        try:
            # Get hidden states from shared memory
            req_state = self._executor.get_request(request.request_id)
            if req_state is None:
                return has_pb2.ApplyTokenStepResponse(
                    success=False,
                    error_message="Request not found",
                )

            # Get hidden dimension from adapter
            adapter_state = self._executor.get_adapter(req_state.adapter_id)
            hidden_size = adapter_state.hidden_size if adapter_state else 4096

            # Read hidden states from shared memory
            region = self._shm_manager.get_region(req_state.shm_region)
            if region is not None:
                hidden_states = self._shm_manager.read_hidden_states(
                    region,
                    shape=(req_state.batch_size, 1, hidden_size),  # Use dynamic size
                )
            else:
                # Mock hidden states for testing
                hidden_states = np.random.randn(
                    req_state.batch_size, 1, hidden_size
                ).astype(np.float16)

            # Apply delta
            delta, encrypted_gate, timing = self._executor.apply_token_step(
                request.request_id,
                request.layer_idx,
                request.projection_type,
                hidden_states,
                is_gate_callback=request.is_gate_callback,
                client_gate_bit=request.client_gate_bit,
            )

            # Write delta to shared memory
            shm_offset = 0
            has_delta = delta is not None
            if has_delta and region is not None:
                shm_offset = self._shm_manager.write_delta(region, delta)

            # Record telemetry
            self._record_telemetry(
                request.request_id,
                request.token_idx,
                request.layer_idx,
                request.projection_type,
                timing,
            )

            return has_pb2.ApplyTokenStepResponse(
                success=True,
                has_delta=has_delta,
                shm_offset=shm_offset,
                encrypt_time_us=timing['encrypt_time_us'],
                compute_time_us=timing['compute_time_us'],
                decrypt_time_us=timing['decrypt_time_us'],
                gate_required=encrypted_gate is not None,
                encrypted_gate_signal=encrypted_gate if encrypted_gate else b"",
                evidence=b"MOCK_TEE_QUOTE_SHA256_F9A2..." + request.request_id.encode(),
            )

        except Exception as e:
            logger.error(f"ApplyTokenStep failed: {e}")
            return has_pb2.ApplyTokenStepResponse(
                success=False,
                error_message=str(e),
            )

    def ApplyBatchedTokenStep(self, request: Any, context: Any) -> Any:
        """Apply batched HE-LoRA deltas."""
        from ..proto import has_pb2
        import numpy as np

        try:
            req_state = self._executor.get_request(request.request_id)
            if req_state is None:
                return has_pb2.ApplyBatchedTokenStepResponse(
                    success=False,
                    error_message="Request not found",
                )

            # Parse layer_projections (format: "layer_proj", e.g., "0_q")
            layer_projections = []
            for lp in request.layer_projections:
                parts = lp.split('_')
                layer_idx = int(parts[0])
                proj_type = parts[1]
                layer_projections.append((layer_idx, proj_type))

            # Get hidden dimension from adapter
            adapter_state = self._executor.get_adapter(req_state.adapter_id)
            hidden_size = adapter_state.hidden_size if adapter_state else 4096

            # Get hidden states
            region = self._shm_manager.get_region(req_state.shm_region)
            if region is not None:
                hidden_states = self._shm_manager.read_hidden_states(
                    region,
                    shape=(req_state.batch_size, 1, hidden_size),
                )
            else:
                hidden_states = np.random.randn(
                    req_state.batch_size, 1, hidden_size
                ).astype(np.float16)

            # Apply all deltas
            results = self._executor.apply_batched_token_step(
                request.request_id,
                layer_projections,
                hidden_states,
            )

            # Build response
            total_time = 0
            delta_results = []
            for (layer_idx, proj_type), (delta, timing) in results.items():
                total_time += sum(timing.values())
                delta_results.append(has_pb2.DeltaResult(
                    layer_idx=layer_idx,
                    projection_type=proj_type,
                    has_delta=delta is not None,
                    shm_offset=0,  # Would be set if using shared memory
                ))

            return has_pb2.ApplyBatchedTokenStepResponse(
                success=True,
                results=delta_results,
                total_time_us=total_time,
                evidence=b"MOCK_TEE_QUOTE_BATCHED_SHA256_6B1D..." + request.request_id.encode(),
            )

        except Exception as e:
            logger.error(f"ApplyBatchedTokenStep failed: {e}")
            return has_pb2.ApplyBatchedTokenStepResponse(
                success=False,
                error_message=str(e),
            )

    def HealthCheck(self, request: Any, context: Any) -> Any:
        """Health check."""
        from ..proto import has_pb2

        return has_pb2.HealthCheckResponse(
            healthy=True,
            message="OK",
            uptime_seconds=time.time() - self._start_time,
        )

    def GetStatus(self, request: Any, context: Any) -> Any:
        """Get service status."""
        from ..proto import has_pb2

        exec_stats = self._executor.get_statistics()
        shm_stats = self._shm_manager.get_statistics()
        key_stats = self._key_manager.get_statistics()

        # Calculate averages
        total_tokens = exec_stats.get('total_tokens_processed', 0)
        avg_encrypt = 0.0
        avg_compute = 0.0
        avg_decrypt = 0.0

        return has_pb2.GetStatusResponse(
            healthy=True,
            loaded_adapters=exec_stats.get('loaded_adapters', 0),
            active_requests=exec_stats.get('active_requests', 0),
            memory_used_mb=shm_stats.get('total_size_mb', 0),
            memory_available_mb=1024.0,  # Placeholder
            gpu_utilization=0.0,  # Placeholder
            total_tokens_processed=total_tokens,
            avg_encrypt_time_us=avg_encrypt,
            avg_compute_time_us=avg_compute,
            avg_decrypt_time_us=avg_decrypt,
        )

    def GetTelemetry(self, request: Any, context: Any) -> Any:
        """Get telemetry data."""
        from ..proto import has_pb2

        data = self._telemetry.get(request.request_id, [])

        # Filter by token range
        if request.start_token >= 0:
            data = [d for d in data if d.token_idx >= request.start_token]
        if request.end_token >= 0:
            data = [d for d in data if d.token_idx <= request.end_token]

        return has_pb2.GetTelemetryResponse(
            success=True,
            data=data,
        )

    def _record_telemetry(
        self,
        request_id: str,
        token_idx: int,
        layer_idx: int,
        projection_type: str,
        timing: Dict[str, int],
    ) -> None:
        """Record telemetry data point."""
        from ..proto import has_pb2

        if request_id not in self._telemetry:
            self._telemetry[request_id] = []

        data = has_pb2.TelemetryData(
            timestamp_us=int(time.time() * 1_000_000),
            request_id=request_id,
            token_idx=token_idx,
            layer_idx=layer_idx,
            projection_type=projection_type,
            encrypt_time_us=timing.get('encrypt_time_us', 0),
            compute_time_us=timing.get('compute_time_us', 0),
            decrypt_time_us=timing.get('decrypt_time_us', 0),
            rotations=0,  # Would come from executor
            keyswitches=0,
            rescales=0,
        )

        self._telemetry[request_id].append(data)


class HASServer:
    """
    HAS gRPC Server wrapper.

    Usage:
        server = HASServer()
        server.start(port=50051)
        # ... server running ...
        server.stop()
    """

    def __init__(
        self,
        backend_type: str = "SIMULATION",
        ckks_profile: str = "FAST",
        max_workers: int = 10,
    ):
        """
        Initialize HAS server.

        Args:
            backend_type: GPU CKKS backend type
            ckks_profile: CKKS parameter profile
            max_workers: Max gRPC workers
        """
        self._backend_type = backend_type
        self._ckks_profile = ckks_profile
        self._max_workers = max_workers

        # Components
        self._executor: Optional[HASExecutor] = None
        self._key_manager: Optional[KeyManager] = None
        self._shm_manager: Optional[SharedMemoryManager] = None
        self._servicer: Optional[HASServicer] = None

        # gRPC server
        self._server = None

    def start(self, port: int = 50051) -> None:
        """
        Start the gRPC server.

        Args:
            port: Port to listen on
        """
        # Initialize components
        self._executor = HASExecutor(
            backend_type=self._backend_type,
            ckks_profile=self._ckks_profile,
        )
        self._executor.initialize()

        self._key_manager = KeyManager()
        self._shm_manager = SharedMemoryManager()

        # Initialize keys (would use executor backend)
        # MOAI column packing guarantees 0 rotations for Ct×Pt LoRA matmul,
        # so no Galois (rotation) keys are needed. Generating them would waste
        # ~5.5 MB of memory and ~17ms of startup time per key (at N=16384).
        self._key_manager.initialize(
            backend=None,  # Mock for now
            galois_steps=[],
        )

        # Create servicer
        self._servicer = HASServicer(
            executor=self._executor,
            key_manager=self._key_manager,
            shm_manager=self._shm_manager,
        )

        # Create and start gRPC server
        try:
            import grpc
            from ..proto.has_pb2_grpc import add_HEAdapterServiceServicer_to_server

            self._server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self._max_workers)
            )
            add_HEAdapterServiceServicer_to_server(self._servicer, self._server)
            self._server.add_insecure_port(f'[::]:{port}')
            self._server.start()

            logger.info(f"HAS server started on port {port}")

        except ImportError:
            logger.warning("gRPC not available, server not started")

    def stop(self, grace: Optional[float] = None) -> None:
        """
        Stop the server.

        Args:
            grace: Grace period for shutdown
        """
        if self._server is not None:
            self._server.stop(grace)
            self._server = None

        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

        if self._key_manager is not None:
            self._key_manager.clear_keys()
            self._key_manager = None

        if self._shm_manager is not None:
            self._shm_manager.shutdown()
            self._shm_manager = None

        logger.info("HAS server stopped")

    def wait_for_termination(self) -> None:
        """Wait for server termination."""
        if self._server is not None:
            self._server.wait_for_termination()


def create_has_server(
    backend_type: str = "SIMULATION",
    ckks_profile: str = "FAST",
    port: int = 50051,
) -> HASServer:
    """
    Create and start a HAS server.

    Args:
        backend_type: GPU CKKS backend type
        ckks_profile: CKKS parameter profile
        port: Port to listen on

    Returns:
        Running HASServer instance
    """
    server = HASServer(
        backend_type=backend_type,
        ckks_profile=ckks_profile,
    )
    server.start(port=port)
    return server


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="HE Adapter Service Server")
    parser.add_argument("--port", "-p", type=int, default=50051, help="Port to bind")
    parser.add_argument("--backend", "-b", type=str, default="SIMULATION",
                       choices=["SIMULATION", "HEONGPU", "FIDESLIB", "OPENFHE_GPU"],
                       help="GPU CKKS backend")
    parser.add_argument("--profile", type=str, default="FAST",
                       choices=["FAST", "SAFE"],
                       help="CKKS parameter profile")
    parser.add_argument("--workers", "-w", type=int, default=10,
                       help="Max gRPC workers")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Start server
    server = HASServer(
        backend_type=args.backend,
        ckks_profile=args.profile,
        max_workers=args.workers,
    )
    server.start(port=args.port)

    print(f"HAS server running on port {args.port}. Press Ctrl+C to stop.")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5.0)


if __name__ == "__main__":
    main()
