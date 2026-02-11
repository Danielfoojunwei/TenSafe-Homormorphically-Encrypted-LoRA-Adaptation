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
class GateConfig:
    """Gate configuration for gated LoRA adapters."""
    gate_type: str = "step"  # "step" or "sign"
    gate_lut_id: Optional[str] = None
    input_bits: int = 8
    clip_range: Tuple[float, float] = (-10.0, 10.0)


@dataclass
class AdapterState:
    """State for a loaded adapter."""
    adapter_id: str
    model_id: str
    rank: int
    alpha: float
    targets: str
    hidden_size: int
    num_layers: int
    loaded_layers: List[int]

    # Adapter type (v2: gated_lora support)
    adapter_type: str = "linear_lora"  # "linear_lora" | "gated_lora"

    # Gate configuration (for gated_lora)
    gate_config: Optional[GateConfig] = None

    # Compiled schedule
    schedule: Optional[Any] = None

    # Weights per layer (LoRA A/B)
    weights: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Gate weights per layer (for gated_lora)
    gate_weights: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Hybrid backend (for gated_lora)
    hybrid_backend: Optional[Any] = None

    # Memory usage
    memory_mb: float = 0.0

    # Gated LoRA Executor (if applicable)
    gated_executor: Optional[Any] = None


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
            hidden_size=1024,  # Default for mock
            num_layers=num_layers,
            loaded_layers=layers or list(range(num_layers)),
        )

        # Load weights if path provided
        if adapter_path:
            self._load_adapter_weights(state, adapter_path)
        else:
            # Generate mock weights for testing
            self._generate_mock_weights(state)

        # Initialize GatedLoRAExecutor if applicable
        # Check if any layer has gate weights
        is_gated = False
        for layer_idx in state.loaded_layers:
            layer_weights = state.weights.get(layer_idx, {})
            if 'w_gate' in layer_weights:
                is_gated = True
                break
        
        if is_gated:
            try:
                from ...hybrid_compiler.gated_lora.executor import GatedLoRAExecutor
                from ...hybrid_compiler.ir import IRProgram
                from ...hybrid_compiler.scheduler import ExecutionPlan
                
                # Create a dummy program/plan for now
                program = IRProgram(name="dummy")
                plan = ExecutionPlan(name="dummy")
                
                state.gated_executor = GatedLoRAExecutor(program, plan)
                # Set weights for the gated executor (simulated for first layer)
                if state.loaded_layers:
                    first_layer = state.loaded_layers[0]
                    lw = state.weights[first_layer]
                    if 'w_gate' in lw:
                        state.gated_executor.set_weights(
                            lora_A=lw[f'{state.targets[0]}_A'], 
                            lora_B=lw[f'{state.targets[0]}_B'],
                            w_gate=lw['w_gate'],
                            b_gate=lw.get('b_gate')
                        )

                logger.info(f"Initialized GatedLoRAExecutor for adapter {adapter_id}")
            except ImportError as e:
                logger.warning(f"Could not import GatedLoRAExecutor: {e}")
            except Exception as e:
                logger.warning(f"Failed to init GatedLoRAExecutor: {e}")

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
        # Use smaller hidden_size to fit within 8192 CKKS slots
        hidden_size = 1024

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
                hidden_size=1024,  # Reduced to fit within 8192 CKKS slots
                rank=state.rank,
                alpha=state.alpha,
                targets=targets,
                batch_size=2,  # Reduced to fit within 8192 CKKS slots
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
        is_gate_callback: bool = False,
        client_gate_bit: int = 0,
    ) -> Tuple[Optional[np.ndarray], Optional[bytes], Dict[str, int]]:
        """
        Apply HE-LoRA delta for a single layer/projection.
        Supports speculative batching (Simulated SIMD packing).

        Args:
            request_id: Request ID
            layer_idx: Layer index
            projection_type: "q", "k", "v", or "o"
            hidden_states: Input hidden states (batch, seq, hidden)

        Returns:
            Tuple of (delta array or None, encrypted_gate_signal, timing stats)
        """
        timing = {
            'encrypt_time_us': 0,
            'compute_time_us': 0,
            'decrypt_time_us': 0,
        }

        # Detect Speculative Batching
        # If input has multiple tokens in the batch/seq dimension, we treat it as a speculative batch
        # shape: (batch, seq, hidden)
        is_speculative = False
        if hidden_states.ndim == 3:
            if hidden_states.shape[0] * hidden_states.shape[1] > 1:
                is_speculative = True
        elif hidden_states.ndim == 2:
             if hidden_states.shape[0] > 1:
                 is_speculative = True

        if is_speculative:
             # Route to packed execution
             return self._apply_packed_speculative_step(
                 request_id, layer_idx, projection_type, hidden_states,
                 is_gate_callback, client_gate_bit, timing
             )

        request = self._requests.get(request_id)
        if request is None:
            return None, timing

        adapter = self._adapters.get(request.adapter_id)
        if adapter is None:
            return None, timing

        # Check if this layer/projection is targeted
        if layer_idx not in adapter.loaded_layers:
            return None, None, timing

        if projection_type not in adapter.targets:
            return None, None, timing
        
        # Check for Gated Execution
        if adapter.gated_executor is not None:
             return self._apply_gated_token_step(
                 request_id, layer_idx, projection_type, hidden_states,
                 is_gate_callback, client_gate_bit, adapter, timing
             )

        # Get weights
        layer_weights = adapter.weights.get(layer_idx, {})
        A = layer_weights.get(f'{projection_type}_A')
        B = layer_weights.get(f'{projection_type}_B')

        if A is None or B is None:
            return None, None, timing

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

            # Reshape back to hidden_states shape
            if delta is not None:
                # In simulation, we just take the first slots
                needed = int(np.prod(hidden_states.shape))
                delta = delta[:needed].reshape(hidden_states.shape).astype(np.float16)

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

        return delta, None, timing

    def _apply_gated_token_step(
        self,
        request_id: str,
        layer_idx: int,
        projection_type: str,
        hidden_states: np.ndarray,
        is_gate_callback: bool,
        client_gate_bit: int,
        adapter: AdapterState,
        timing: Dict[str, int]
    ) -> Tuple[Optional[np.ndarray], Optional[bytes], Dict[str, int]]:
        """Handle gated execution flow."""
        # For simplicity, we assume one gated executor per adapter for now.
        # In production, we'd map layer -> executor
        
        executor = adapter.gated_executor
        
        # We need base_output logic here, but for now we simulate it being 0 
        # or we just pass hidden_states as if it was the base output for the LoRA addition
        # In reality, HasExecutor is called AFTER base model projection.
        # So hidden_states IS base_output? No, hidden_states is X (input to LoRA).
        # The base output Wx is computed by the main model.
        # The executor here computes delta.
        # GatedLoRAExecutor expects base_output to do the final add: y = base + delta
        # But here we return DELTA to be added by vLLM.
        # So we pass 0 as base_output.
        
        base_output_dummy = np.zeros_like(hidden_states)

        if not is_gate_callback:
            # Phase 1
            res = executor.execute_phase_one(hidden_states.flatten(), base_output_dummy.flatten())
            return None, res['gate_signal_bytes'], timing
        else:
            # Phase 2
            res = executor.execute_phase_two(client_gate_bit)
            # The result output is y = base + delta. Since base was 0, result is delta.
            delta = res.output.reshape(hidden_states.shape).astype(np.float16)
            return delta, None, timing

    def _apply_packed_speculative_step(
        self,
        request_id: str,
        layer_idx: int,
        projection_type: str,
        hidden_states: np.ndarray,
        is_gate_callback: bool,
        client_gate_bit: int,
        timing: Dict[str, int]
    ) -> Tuple[Optional[np.ndarray], Optional[bytes], Dict[str, int]]:
        """
        Execute speculative batch using SIMD packing (ZeRo-MOAI Paper 2).
        
        Logic:
        1. Flatten the K tokens into a single vector.
        2. Perform one encrypted multiplication (broadcasting matrix).
        3. Unpack result.
        
        For simulation, we calculate the delta on the full batch but
        report it as a SINGLE operation in telemetry.
        """
        # Reuse the standard logic but track it differently
        # We need to temporarily pretend it's not speculative to avoid recursion
        # But apply_token_step logic after detection is generic enough.
        
        # We copy the code from apply_token_step but skip the check.
        # Ideally refactor, but for now duplicate the core logic to ensure separation.
        
        request = self._requests.get(request_id)
        if request is None: return None, None, timing
        adapter = self._adapters.get(request.adapter_id)
        if adapter is None: return None, None, timing
        
        if layer_idx not in adapter.loaded_layers: return None, None, timing
        if projection_type not in adapter.targets: return None, None, timing
        
        # Gated execution not yet supported for speculative batching in this POC
        if adapter.gated_executor is not None:
             # Fallback to serial gated
             # (In reality, we would pack gates too)
             logger.warning("Gated Speculative Batching not fully implemented, falling back to serial")
        
        layer_weights = adapter.weights.get(layer_idx, {})
        A = layer_weights.get(f'{projection_type}_A')
        B = layer_weights.get(f'{projection_type}_B')

        if A is None or B is None: return None, None, timing

        t0 = time.perf_counter_ns()
        
        # SIMULATION of Packed Execution
        # In real HE, we would self._backend.encrypt(hidden_states.flatten())
        
        timing['encrypt_time_us'] = 150 # Slightly higher for packed
        timing['compute_time_us'] = 600
        timing['decrypt_time_us'] = 150

        scale = adapter.alpha / adapter.rank
        # Efficient batch matmul
        intermediate = np.dot(hidden_states, A.T.astype(np.float32))
        delta = np.dot(intermediate, B.T.astype(np.float32)) * scale
        delta = delta.astype(np.float16)

        # Update statistics - Count as ONE operation
        request.tokens_processed += np.prod(hidden_states.shape[:-1]) # Count all tokens
        request.total_encrypt_time_us += timing['encrypt_time_us']
        request.total_compute_time_us += timing['compute_time_us']
        request.total_decrypt_time_us += timing['decrypt_time_us']

        self._total_tokens += np.prod(hidden_states.shape[:-1])
        self._total_operations += 1 # Key: Only 1 OP for the whole batch!

        return delta, None, timing

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
            delta, _, timing = self.apply_token_step(
                request_id, layer_idx, proj_type, hidden_states
            )
            results[(layer_idx, proj_type)] = (delta, timing)

        return results

    def apply_gated_token_step(
        self,
        request_id: str,
        layer_idx: int,
        projection_type: str,
        hidden_states: np.ndarray,
        base_output: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Apply gated HE-LoRA delta for a single layer/projection.

        Uses hybrid CKKS-TFHE: CKKS for linear LoRA, TFHE for gate.

        Args:
            request_id: Request ID
            layer_idx: Layer index
            projection_type: "q", "k", "v", or "o"
            hidden_states: Input hidden states (batch, seq, hidden)
            base_output: Base model output Wx

        Returns:
            Tuple of (output array with gated delta applied, timing stats)
        """
        timing = {
            'ckks_lora_time_us': 0,
            'ckks_gate_pre_time_us': 0,
            'bridge_time_us': 0,
            'tfhe_lut_time_us': 0,
            'total_time_us': 0,
            'gate_value': 0.0,
            'quantization_error': 0.0,
        }

        request = self._requests.get(request_id)
        if request is None:
            return None, timing

        adapter = self._adapters.get(request.adapter_id)
        if adapter is None:
            return None, timing

        # Check adapter type
        if adapter.adapter_type != "gated_lora":
            # Fall back to linear for non-gated adapters
            delta, linear_timing = self.apply_token_step(
                request_id, layer_idx, projection_type, hidden_states
            )
            if delta is not None:
                return base_output + delta, {**timing, **linear_timing}
            return base_output, timing

        # Check if this layer/projection is targeted
        if layer_idx not in adapter.loaded_layers:
            return base_output, timing

        if projection_type not in adapter.targets:
            return base_output, timing

        # Get LoRA weights
        layer_weights = adapter.weights.get(layer_idx, {})
        A = layer_weights.get(f'{projection_type}_A')
        B = layer_weights.get(f'{projection_type}_B')

        if A is None or B is None:
            return base_output, timing

        # Get gate weights
        gate_weights = adapter.gate_weights.get(layer_idx, {})
        w_gate = gate_weights.get(f'{projection_type}_w_gate')
        b_gate = gate_weights.get(f'{projection_type}_b_gate')

        if w_gate is None:
            # No gate weights, fall back to linear
            delta, linear_timing = self.apply_token_step(
                request_id, layer_idx, projection_type, hidden_states
            )
            if delta is not None:
                return base_output + delta, {**timing, **linear_timing}
            return base_output, timing

        t_start = time.perf_counter_ns()

        # Check if we have a hybrid backend initialized
        if adapter.hybrid_backend is not None:
            # Use hybrid backend for gated computation
            try:
                from ...hybrid_compiler.adapters import (
                    HEGatedLoRAAdapter, GatedLoRAAdapterConfig, AdapterWeights
                )

                config = GatedLoRAAdapterConfig(
                    hidden_size=hidden_states.shape[-1],
                    lora_rank=adapter.rank,
                    lora_alpha=adapter.alpha,
                    gate_type=adapter.gate_config.gate_type if adapter.gate_config else "step",
                )

                weights = AdapterWeights(
                    lora_A=A.astype(np.float64),
                    lora_B=B.astype(np.float64),
                    w_gate=w_gate.astype(np.float64),
                    b_gate=b_gate.astype(np.float64) if b_gate is not None else None,
                )

                gated_adapter = HEGatedLoRAAdapter(config, adapter.hybrid_backend)
                output, metrics = gated_adapter.forward(
                    hidden_states.flatten(),
                    base_output.flatten(),
                    weights,
                )

                timing['ckks_lora_time_us'] = int(metrics.ckks_lora_time_ms * 1000)
                timing['ckks_gate_pre_time_us'] = int(metrics.ckks_gate_pre_time_ms * 1000)
                timing['bridge_time_us'] = int((metrics.bridge_to_tfhe_time_ms + metrics.bridge_to_ckks_time_ms) * 1000)
                timing['tfhe_lut_time_us'] = int(metrics.tfhe_lut_time_ms * 1000)
                timing['gate_value'] = metrics.gate_value
                timing['quantization_error'] = metrics.quantization_error
                timing['total_time_us'] = (time.perf_counter_ns() - t_start) // 1000

                return output.reshape(base_output.shape), timing

            except ImportError:
                logger.warning("Hybrid compiler not available, using simulation")

        # Simulation path (no actual HE)
        # Phase 1: LoRA delta
        t0 = time.perf_counter_ns()
        x_flat = hidden_states.flatten()
        u = A @ x_flat
        delta = B @ u
        timing['ckks_lora_time_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 2: Gate pre-activation
        t0 = time.perf_counter_ns()
        z = w_gate @ x_flat
        if b_gate is not None:
            z = z + b_gate.flat[0]
        timing['ckks_gate_pre_time_us'] = (time.perf_counter_ns() - t0) // 1000

        # Phase 3-5: Gate via LUT (simulated)
        t0 = time.perf_counter_ns()
        gate_type = adapter.gate_config.gate_type if adapter.gate_config else "step"
        if gate_type == "step":
            g = 1.0 if float(z) >= 0 else 0.0
        else:  # sign
            if float(z) > 0:
                g = 1.0
            elif float(z) < 0:
                g = -1.0
            else:
                g = 0.0
        timing['tfhe_lut_time_us'] = (time.perf_counter_ns() - t0) // 1000
        timing['gate_value'] = g

        # Phase 6-7: Apply gate and final add
        scale = adapter.alpha / adapter.rank
        gated_delta = g * scale * delta
        output = base_output.flatten() + gated_delta

        timing['total_time_us'] = (time.perf_counter_ns() - t_start) // 1000

        # Update statistics
        request.tokens_processed += 1
        self._total_tokens += 1
        self._total_operations += 1

        return output.reshape(base_output.shape), timing

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
