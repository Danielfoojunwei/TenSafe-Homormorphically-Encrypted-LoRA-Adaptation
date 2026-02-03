"""
HAS Executor

Executes HE-LoRA computation within the HAS service.
Integrates with the HE-LoRA microkernel compiler and runtime.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AdapterState:
    """State for a loaded adapter."""
    adapter_id: str
    model_id: str
    rank: int
    alpha: float
    targets: str
    num_layers: int
    loaded_layers: List[int]

    # Compiled schedule
    schedule: Optional[Any] = None

    # Weights per layer
    weights: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Memory usage
    memory_mb: float = 0.0


@dataclass
class RequestState:
    """State for an active request."""
    request_id: str
    adapter_id: str
    batch_size: int
    seq_len: int

    # Shared memory region
    shm_region: Optional[str] = None

    # HE context (encrypted activations, etc.)
    he_context: Optional[Any] = None

    # Token counter
    tokens_processed: int = 0

    # Timing statistics
    total_encrypt_time_us: int = 0
    total_compute_time_us: int = 0
    total_decrypt_time_us: int = 0


class HASExecutor:
    """
    Executes HE-LoRA computation.

    Responsibilities:
    - Load and manage adapter weights
    - Compile HE schedules
    - Execute encryption/computation/decryption
    - Collect telemetry
    """

    def __init__(
        self,
        backend_type: str = "SIMULATION",
        ckks_profile: str = "FAST",
    ):
        """
        Initialize executor.

        Args:
            backend_type: GPU CKKS backend type
            ckks_profile: CKKS parameter profile
        """
        self._backend_type = backend_type
        self._ckks_profile = ckks_profile

        # Initialize HE-LoRA components
        self._backend = None
        self._executor = None

        # Loaded adapters
        self._adapters: Dict[str, AdapterState] = {}

        # Active requests
        self._requests: Dict[str, RequestState] = {}

        # Statistics
        self._total_tokens = 0
        self._total_operations = 0

    def initialize(self) -> bool:
        """
        Initialize the executor with HE backend.

        Returns:
            True if initialization successful
        """
        try:
            from ...backend.gpu_ckks_backend import BackendType, get_backend
            from ...compiler.ckks_params import CKKSProfile, get_profile

            # Get CKKS parameters
            profile = CKKSProfile[self._ckks_profile]
            ckks_params = get_profile(profile)

            # Initialize backend
            backend_enum = BackendType[self._backend_type]
            self._backend = get_backend(backend_enum, ckks_params)
            self._backend.initialize()

            logger.info(f"HAS executor initialized with {self._backend_type} backend")
            return True

        except ImportError as e:
            logger.warning(f"Failed to import HE-LoRA components: {e}")
            logger.warning("Using mock executor")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize executor: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown executor."""
        # Unload all adapters
        for adapter_id in list(self._adapters.keys()):
            self.unload_adapter(adapter_id)

        # Release all requests
        for request_id in list(self._requests.keys()):
            self.release_request(request_id)

        # Shutdown backend
        if self._backend is not None:
            self._backend.shutdown()
            self._backend = None

        logger.info("HAS executor shutdown")

    # -------------------------------------------------------------------------
    # ADAPTER MANAGEMENT
    # -------------------------------------------------------------------------

    def load_adapter(
        self,
        adapter_id: str,
        model_id: str,
        rank: int = 16,
        alpha: float = 32.0,
        targets: str = "qkv",
        layers: Optional[List[int]] = None,
        adapter_path: Optional[str] = None,
    ) -> AdapterState:
        """
        Load a LoRA adapter.

        Args:
            adapter_id: Unique identifier
            model_id: Base model identifier
            rank: LoRA rank
            alpha: LoRA alpha scaling
            targets: Target projections
            layers: Specific layers to load
            adapter_path: Path to adapter weights

        Returns:
            AdapterState with loaded adapter
        """
        if adapter_id in self._adapters:
            return self._adapters[adapter_id]

        # Determine number of layers (from model config or default)
        num_layers = 32  # Default, should come from model metadata

        # Determine which layers to load
        if layers is None:
            loaded_layers = list(range(num_layers))
        else:
            loaded_layers = [l for l in layers if 0 <= l < num_layers]

        # Create adapter state
        state = AdapterState(
            adapter_id=adapter_id,
            model_id=model_id,
            rank=rank,
            alpha=alpha,
            targets=targets,
            num_layers=num_layers,
            loaded_layers=loaded_layers,
        )

        # Load weights if path provided
        if adapter_path:
            self._load_adapter_weights(state, adapter_path)
        else:
            # Generate mock weights for testing
            self._generate_mock_weights(state)

        # Compile HE schedule
        state.schedule = self._compile_schedule(state)

        # Calculate memory usage
        state.memory_mb = self._calculate_memory_usage(state)

        self._adapters[adapter_id] = state
        logger.info(f"Loaded adapter {adapter_id}: {len(loaded_layers)} layers, "
                   f"{state.memory_mb:.2f} MB")

        return state

    def _load_adapter_weights(self, state: AdapterState, path: str) -> None:
        """Load adapter weights from file."""
        try:
            import torch

            # Load weights
            weights = torch.load(path, map_location='cpu')

            # Parse and store weights per layer
            for layer_idx in state.loaded_layers:
                layer_weights = {}
                for proj in state.targets:
                    a_key = f"layers.{layer_idx}.self_attn.{proj}_proj.lora_A.weight"
                    b_key = f"layers.{layer_idx}.self_attn.{proj}_proj.lora_B.weight"

                    if a_key in weights and b_key in weights:
                        layer_weights[f'{proj}_A'] = weights[a_key].numpy()
                        layer_weights[f'{proj}_B'] = weights[b_key].numpy()

                if layer_weights:
                    state.weights[layer_idx] = layer_weights

        except Exception as e:
            logger.warning(f"Failed to load weights from {path}: {e}")
            self._generate_mock_weights(state)

    def _generate_mock_weights(self, state: AdapterState) -> None:
        """Generate mock weights for testing."""
        # Assume hidden_size based on common models
        hidden_size = 4096

        for layer_idx in state.loaded_layers:
            layer_weights = {}
            for proj in state.targets:
                # A: (rank, hidden_size)
                # B: (hidden_size, rank)
                layer_weights[f'{proj}_A'] = np.random.randn(
                    state.rank, hidden_size
                ).astype(np.float16) * 0.01
                layer_weights[f'{proj}_B'] = np.random.randn(
                    hidden_size, state.rank
                ).astype(np.float16) * 0.01

            state.weights[layer_idx] = layer_weights

    def _compile_schedule(self, state: AdapterState) -> Optional[Any]:
        """Compile HE execution schedule for adapter."""
        try:
            from ...compiler.ckks_params import CKKSProfile, get_profile
            from ...compiler.lora_ir import LoRAConfig, LoRATargets
            from ...compiler.scheduler import compile_schedule

            # Create LoRA config
            targets = LoRATargets.QKV if state.targets == "qkv" else LoRATargets.QKVO

            config = LoRAConfig(
                hidden_size=4096,  # Should come from model
                rank=state.rank,
                alpha=state.alpha,
                targets=targets,
                batch_size=8,  # Will be updated per request
                max_context_length=2048,
                ckks_profile=CKKSProfile[self._ckks_profile],
            )

            # Compile schedule
            ckks_params = get_profile(config.ckks_profile)
            schedule = compile_schedule(config, ckks_params)

            return schedule

        except ImportError:
            return None

    def _calculate_memory_usage(self, state: AdapterState) -> float:
        """Calculate memory usage in MB."""
        total_bytes = 0

        for layer_weights in state.weights.values():
            for weight in layer_weights.values():
                if hasattr(weight, 'nbytes'):
                    total_bytes += weight.nbytes

        return total_bytes / (1024 * 1024)

    def unload_adapter(self, adapter_id: str) -> bool:
        """Unload an adapter."""
        if adapter_id not in self._adapters:
            return False

        state = self._adapters[adapter_id]

        # Clear weights
        state.weights.clear()
        state.schedule = None

        del self._adapters[adapter_id]
        return True

    def get_adapter(self, adapter_id: str) -> Optional[AdapterState]:
        """Get adapter state."""
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
        shm_region: Optional[str] = None,
    ) -> RequestState:
        """
        Prepare HE context for a request.

        Args:
            request_id: Unique request ID
            adapter_id: Adapter to use
            batch_size: Batch size
            seq_len: Initial sequence length
            shm_region: Shared memory region name

        Returns:
            RequestState for the request
        """
        if request_id in self._requests:
            return self._requests[request_id]

        if adapter_id not in self._adapters:
            raise ValueError(f"Adapter {adapter_id} not loaded")

        state = RequestState(
            request_id=request_id,
            adapter_id=adapter_id,
            batch_size=batch_size,
            seq_len=seq_len,
            shm_region=shm_region,
        )

        # Initialize HE context if backend available
        if self._backend is not None:
            # Context would include encrypted key material, etc.
            state.he_context = {}

        self._requests[request_id] = state
        return state

    def release_request(self, request_id: str) -> bool:
        """Release a request and its resources."""
        if request_id not in self._requests:
            return False

        state = self._requests[request_id]

        # Clean up HE context
        state.he_context = None

        del self._requests[request_id]
        return True

    def get_request(self, request_id: str) -> Optional[RequestState]:
        """Get request state."""
        return self._requests.get(request_id)

    # -------------------------------------------------------------------------
    # COMPUTATION
    # -------------------------------------------------------------------------

    def apply_token_step(
        self,
        request_id: str,
        layer_idx: int,
        projection_type: str,
        hidden_states: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
        """
        Apply HE-LoRA delta for a single layer/projection.

        Args:
            request_id: Request ID
            layer_idx: Layer index
            projection_type: "q", "k", "v", or "o"
            hidden_states: Input hidden states (batch, seq, hidden)

        Returns:
            Tuple of (delta array or None, timing stats)
        """
        timing = {
            'encrypt_time_us': 0,
            'compute_time_us': 0,
            'decrypt_time_us': 0,
        }

        request = self._requests.get(request_id)
        if request is None:
            return None, timing

        adapter = self._adapters.get(request.adapter_id)
        if adapter is None:
            return None, timing

        # Check if this layer/projection is targeted
        if layer_idx not in adapter.loaded_layers:
            return None, timing

        if projection_type not in adapter.targets:
            return None, timing

        # Get weights
        layer_weights = adapter.weights.get(layer_idx, {})
        A = layer_weights.get(f'{projection_type}_A')
        B = layer_weights.get(f'{projection_type}_B')

        if A is None or B is None:
            return None, timing

        # Compute delta: delta = alpha/r * (x @ A^T @ B^T)
        # In HE: encrypt(x), compute, decrypt

        t0 = time.perf_counter_ns()

        if self._backend is not None and hasattr(self._backend, 'encrypt'):
            # Real HE computation
            encrypted = self._backend.encrypt(hidden_states)
            timing['encrypt_time_us'] = (time.perf_counter_ns() - t0) // 1000

            t1 = time.perf_counter_ns()
            # Ct×Pt multiplication
            intermediate = self._backend.ct_pt_multiply(encrypted, A.T)
            result = self._backend.ct_pt_multiply(intermediate, B.T)
            timing['compute_time_us'] = (time.perf_counter_ns() - t1) // 1000

            t2 = time.perf_counter_ns()
            delta = self._backend.decrypt(result)
            timing['decrypt_time_us'] = (time.perf_counter_ns() - t2) // 1000

        else:
            # Mock computation (no HE)
            timing['encrypt_time_us'] = 100  # Mock timing
            timing['compute_time_us'] = 500
            timing['decrypt_time_us'] = 100

            # Plain computation
            # x: (batch, seq, hidden)
            # A: (rank, hidden) -> A^T: (hidden, rank)
            # B: (hidden, rank) -> B^T: (rank, hidden)
            # delta = x @ A^T @ B^T
            scale = adapter.alpha / adapter.rank
            intermediate = np.dot(hidden_states, A.T.astype(np.float32))
            delta = np.dot(intermediate, B.T.astype(np.float32)) * scale
            delta = delta.astype(np.float16)

        # Update statistics
        request.tokens_processed += 1
        request.total_encrypt_time_us += timing['encrypt_time_us']
        request.total_compute_time_us += timing['compute_time_us']
        request.total_decrypt_time_us += timing['decrypt_time_us']

        self._total_tokens += 1
        self._total_operations += 1

        return delta, timing

    def apply_batched_token_step(
        self,
        request_id: str,
        layer_projections: List[Tuple[int, str]],
        hidden_states: np.ndarray,
    ) -> Dict[Tuple[int, str], Tuple[Optional[np.ndarray], Dict[str, int]]]:
        """
        Apply HE-LoRA deltas for multiple layers/projections.

        Args:
            request_id: Request ID
            layer_projections: List of (layer_idx, projection_type) tuples
            hidden_states: Input hidden states

        Returns:
            Dict mapping (layer, proj) to (delta, timing)
        """
        results = {}

        for layer_idx, proj_type in layer_projections:
            delta, timing = self.apply_token_step(
                request_id, layer_idx, proj_type, hidden_states
            )
            results[(layer_idx, proj_type)] = (delta, timing)

        return results

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get executor statistics."""
        return {
            'backend_type': self._backend_type,
            'ckks_profile': self._ckks_profile,
            'loaded_adapters': len(self._adapters),
            'active_requests': len(self._requests),
            'total_tokens_processed': self._total_tokens,
            'total_operations': self._total_operations,
            'adapters': {
                aid: {
                    'model_id': state.model_id,
                    'rank': state.rank,
                    'layers': len(state.loaded_layers),
                    'memory_mb': state.memory_mb,
                }
                for aid, state in self._adapters.items()
            },
        }
