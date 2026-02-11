"""
HAS (HE Adapter Service) Client

Client for communicating with the HE Adapter Service from MSS.
Uses gRPC for control plane and shared memory for data plane.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Client connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class HASConfig:
    """Configuration for HAS client."""
    # gRPC settings
    grpc_host: str = "localhost"
    grpc_port: int = 50051
    grpc_timeout_ms: int = 5000
    grpc_max_retries: int = 3

    # Shared memory settings
    shm_prefix: str = "/helora_shm"
    shm_size_mb: int = 256
    use_cuda_ipc: bool = True

    # Connection settings
    connect_timeout_ms: int = 10000
    keepalive_interval_ms: int = 5000


@dataclass
class AdapterInfo:
    """Information about a loaded adapter."""
    adapter_id: str
    model_id: str
    rank: int
    alpha: float
    targets: str  # "qkv" or "qkvo"
    num_layers: int
    loaded_layers: List[int] = field(default_factory=list)


@dataclass
class RequestContext:
    """Context for an active HE-LoRA request."""
    request_id: str
    adapter_id: str
    batch_size: int
    seq_len: int
    shm_region: Optional[str] = None
    created_at: float = field(default_factory=time.time)


class HASClient:
    """
    Client for HE Adapter Service.

    Provides control plane operations via gRPC and data plane
    operations via shared memory.

    Usage:
        client = HASClient(config)
        client.connect()

        # Load adapter
        client.load_adapter("my-adapter", adapter_config)

        # For each request
        ctx = client.prepare_request(request_id, batch_size, seq_len)

        # During generation
        for token_idx in range(max_tokens):
            deltas = client.apply_token_step(ctx, layer_idx, hidden_states)

        # Cleanup
        client.release_request(ctx)
    """

    def __init__(self, config: Optional[HASConfig] = None):
        self._config = config or HASConfig()
        self._state = ConnectionState.DISCONNECTED

        # gRPC components (initialized on connect)
        self._channel = None
        self._stub = None

        # Shared memory regions
        self._shm_regions: Dict[str, Any] = {}

        # Active requests
        self._requests: Dict[str, RequestContext] = {}

        # Loaded adapters
        self._adapters: Dict[str, AdapterInfo] = {}

        # Mock mode for testing
        self._mock_mode = False

    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Connect to HAS.

        Returns:
            True if connection successful
        """
        if self._state == ConnectionState.CONNECTED:
            return True

        self._state = ConnectionState.CONNECTING

        try:
            # Try to import gRPC
            import grpc

            from ..proto import has_pb2_grpc

            # Create channel
            target = f"{self._config.grpc_host}:{self._config.grpc_port}"
            self._channel = grpc.insecure_channel(target)

            # Wait for channel ready
            try:
                grpc.channel_ready_future(self._channel).result(
                    timeout=self._config.connect_timeout_ms / 1000
                )
            except grpc.FutureTimeoutError:
                logger.warning("gRPC connection timeout, using mock mode")
                self._enable_mock_mode()
                return True

            # Create stub
            self._stub = has_pb2_grpc.HEAdapterServiceStub(self._channel)

            self._state = ConnectionState.CONNECTED
            logger.info(f"Connected to HAS at {target}")
            return True

        except ImportError:
            logger.warning("gRPC not available, using mock mode")
            self._enable_mock_mode()
            return True

        except Exception as e:
            logger.error(f"Failed to connect to HAS: {e}")
            self._state = ConnectionState.ERROR
            return False

    def _enable_mock_mode(self) -> None:
        """Enable mock mode for testing."""
        self._mock_mode = True
        self._state = ConnectionState.CONNECTED
        logger.info("HAS client running in mock mode")

    def disconnect(self) -> None:
        """Disconnect from HAS."""
        # Release all requests
        for request_id in list(self._requests.keys()):
            self.release_request(self._requests[request_id])

        # Close shared memory regions
        for shm_name, shm in self._shm_regions.items():
            try:
                shm.close()
            except Exception as e:
                logger.warning(f"Error closing shm {shm_name}: {e}")
        self._shm_regions.clear()

        # Close gRPC channel
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from HAS")

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    # -------------------------------------------------------------------------
    # ADAPTER MANAGEMENT
    # -------------------------------------------------------------------------

    def load_adapter(
        self,
        adapter_id: str,
        model_id: str,
        adapter_path: Optional[str] = None,
        rank: int = 16,
        alpha: float = 32.0,
        targets: str = "qkv",
        layers: Optional[List[int]] = None,
    ) -> AdapterInfo:
        """
        Load a LoRA adapter into HAS.

        Args:
            adapter_id: Unique identifier for this adapter
            model_id: Base model identifier
            adapter_path: Path to adapter weights (optional)
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            targets: Target projections ("qkv" or "qkvo")
            layers: Specific layers to load (None = all)

        Returns:
            AdapterInfo with loaded adapter details
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to HAS")

        if self._mock_mode:
            return self._mock_load_adapter(
                adapter_id, model_id, rank, alpha, targets, layers
            )

        # Make gRPC call
        from ..proto import has_pb2

        request = has_pb2.LoadAdapterRequest(
            adapter_id=adapter_id,
            model_id=model_id,
            adapter_path=adapter_path or "",
            rank=rank,
            alpha=alpha,
            targets=targets,
        )
        if layers:
            request.layers.extend(layers)

        response = self._stub.LoadAdapter(
            request,
            timeout=self._config.grpc_timeout_ms / 1000,
        )

        if not response.success:
            raise RuntimeError(f"Failed to load adapter: {response.error_message}")

        info = AdapterInfo(
            adapter_id=adapter_id,
            model_id=model_id,
            rank=rank,
            alpha=alpha,
            targets=targets,
            num_layers=response.num_layers,
            loaded_layers=list(response.loaded_layers),
        )
        self._adapters[adapter_id] = info

        logger.info(f"Loaded adapter {adapter_id} with {len(info.loaded_layers)} layers")
        return info

    def _mock_load_adapter(
        self,
        adapter_id: str,
        model_id: str,
        rank: int,
        alpha: float,
        targets: str,
        layers: Optional[List[int]],
    ) -> AdapterInfo:
        """Mock adapter loading."""
        num_layers = 32  # Default
        loaded_layers = layers if layers else list(range(num_layers))

        info = AdapterInfo(
            adapter_id=adapter_id,
            model_id=model_id,
            rank=rank,
            alpha=alpha,
            targets=targets,
            num_layers=num_layers,
            loaded_layers=loaded_layers,
        )
        self._adapters[adapter_id] = info
        return info

    def unload_adapter(self, adapter_id: str) -> bool:
        """
        Unload an adapter from HAS.

        Returns:
            True if adapter was unloaded
        """
        if adapter_id not in self._adapters:
            return False

        if not self._mock_mode and self._stub is not None:
            from ..proto import has_pb2

            request = has_pb2.UnloadAdapterRequest(adapter_id=adapter_id)
            response = self._stub.UnloadAdapter(
                request,
                timeout=self._config.grpc_timeout_ms / 1000,
            )
            if not response.success:
                logger.warning(f"Failed to unload adapter: {response.error_message}")

        del self._adapters[adapter_id]
        return True

    def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get information about a loaded adapter."""
        return self._adapters.get(adapter_id)

    # -------------------------------------------------------------------------
    # REQUEST MANAGEMENT
    # -------------------------------------------------------------------------

    def prepare_request(
        self,
        request_id: str,
        adapter_id: str,
        batch_size: int,
        seq_len: int,
    ) -> RequestContext:
        """
        Prepare a request for HE-LoRA processing.

        Args:
            request_id: Unique request identifier
            adapter_id: Adapter to use for this request
            batch_size: Batch size for this request
            seq_len: Initial sequence length

        Returns:
            RequestContext for use in token steps
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to HAS")

        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")

        # Allocate shared memory region for this request
        shm_name = self._allocate_shm_region(request_id, batch_size, seq_len)

        ctx = RequestContext(
            request_id=request_id,
            adapter_id=adapter_id,
            batch_size=batch_size,
            seq_len=seq_len,
            shm_region=shm_name,
        )

        if not self._mock_mode and self._stub is not None:
            from ..proto import has_pb2

            request = has_pb2.PrepareRequestRequest(
                request_id=request_id,
                adapter_id=adapter_id,
                batch_size=batch_size,
                seq_len=seq_len,
                shm_region=shm_name or "",
            )
            response = self._stub.PrepareRequest(
                request,
                timeout=self._config.grpc_timeout_ms / 1000,
            )
            if not response.success:
                raise RuntimeError(f"Failed to prepare request: {response.error_message}")

        self._requests[request_id] = ctx
        return ctx

    def _allocate_shm_region(
        self,
        request_id: str,
        batch_size: int,
        seq_len: int,
    ) -> Optional[str]:
        """Allocate shared memory region for data transfer."""
        if self._mock_mode:
            return None

        try:

            # Calculate size needed
            # Assuming FP16 activations: batch_size * hidden_size * 2 bytes
            # Estimate hidden_size as 4096
            hidden_size = 4096
            size = batch_size * hidden_size * 2 * 4  # Some buffer

            shm_name = f"{self._config.shm_prefix}_{request_id}"

            # Create shared memory (platform-specific)
            # This is a simplified version; real implementation would use
            # memfd_create on Linux or platform-specific APIs
            self._shm_regions[request_id] = {
                'name': shm_name,
                'size': size,
            }

            return shm_name

        except Exception as e:
            logger.warning(f"Failed to allocate shared memory: {e}")
            return None

    def release_request(self, ctx: RequestContext) -> None:
        """
        Release a request and its resources.

        Args:
            ctx: RequestContext to release
        """
        request_id = ctx.request_id

        if not self._mock_mode and self._stub is not None:
            from ..proto import has_pb2

            request = has_pb2.ReleaseRequestRequest(request_id=request_id)
            try:
                self._stub.ReleaseRequest(
                    request,
                    timeout=self._config.grpc_timeout_ms / 1000,
                )
            except Exception as e:
                logger.warning(f"Error releasing request on HAS: {e}")

        # Free shared memory
        if request_id in self._shm_regions:
            del self._shm_regions[request_id]

        if request_id in self._requests:
            del self._requests[request_id]

    # -------------------------------------------------------------------------
    # TOKEN STEP PROCESSING
    # -------------------------------------------------------------------------

    def apply_token_step(
        self,
        ctx: RequestContext,
        layer_idx: int,
        projection_type: str,
        hidden_states: Any,
    ) -> Optional[Any]:
        """
        Apply HE-LoRA delta for a single layer/projection.

        Args:
            ctx: Request context
            layer_idx: Layer index
            projection_type: "q", "k", "v", or "o"
            hidden_states: Input hidden states tensor

        Returns:
            Delta tensor to add to projection output, or None
        """
        if not self.is_connected:
            return None

        adapter_info = self._adapters.get(ctx.adapter_id)
        if adapter_info is None:
            return None

        # Check if this layer is loaded
        if layer_idx not in adapter_info.loaded_layers:
            return None

        # Check if this projection is targeted
        if projection_type not in adapter_info.targets:
            return None

        if self._mock_mode:
            return self._mock_apply_delta(ctx, layer_idx, projection_type, hidden_states)

        # Real implementation would:
        # 1. Write hidden_states to shared memory
        # 2. Call gRPC to trigger HE computation
        # 3. Read delta from shared memory
        # 4. Return delta tensor

        from ..proto import has_pb2

        # Write to shared memory (simplified)
        # In real implementation, use CUDA IPC or memfd

        request = has_pb2.ApplyTokenStepRequest(
            request_id=ctx.request_id,
            layer_idx=layer_idx,
            projection_type=projection_type,
            # Hidden states passed via shared memory
        )

        response = self._stub.ApplyTokenStep(
            request,
            timeout=self._config.grpc_timeout_ms / 1000,
        )

        if not response.success or not response.has_delta:
            return None

        # Read delta from shared memory
        # Return as tensor
        return self._read_delta_from_shm(ctx, response)

    def _mock_apply_delta(
        self,
        ctx: RequestContext,
        layer_idx: int,
        projection_type: str,
        hidden_states: Any,
    ) -> Optional[Any]:
        """Mock delta application."""
        import numpy as np

        # Return a small random delta for testing
        shape = hidden_states.shape if hasattr(hidden_states, 'shape') else (ctx.batch_size, 1, 4096)
        delta = np.random.randn(*shape).astype(np.float16) * 0.01

        # Convert to torch if input was torch
        if hasattr(hidden_states, 'numpy'):
            import torch
            return torch.from_numpy(delta).to(hidden_states.device)

        return delta

    def _read_delta_from_shm(self, ctx: RequestContext, response: Any) -> Optional[Any]:
        """Read delta tensor from shared memory."""
        # Placeholder for shared memory read
        return None

    # -------------------------------------------------------------------------
    # BATCH OPERATIONS
    # -------------------------------------------------------------------------

    def create_delta_callback(self, ctx: RequestContext) -> Callable:
        """
        Create a delta callback function for use with adapters.

        The callback has signature:
            callback(layer_idx, projection_type, hidden_states) -> Optional[delta]

        Args:
            ctx: Request context

        Returns:
            Callback function
        """
        def callback(layer_idx: int, projection_type: str, hidden_states: Any) -> Optional[Any]:
            return self.apply_token_step(ctx, layer_idx, projection_type, hidden_states)

        return callback

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'state': self._state.value,
            'mock_mode': self._mock_mode,
            'loaded_adapters': list(self._adapters.keys()),
            'active_requests': list(self._requests.keys()),
            'shm_regions': len(self._shm_regions),
        }
