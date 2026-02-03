"""
Gated LoRA Executor

Executes compiled gated LoRA programs with:
- Simulated CKKS operations (for testing)
- Simulated TFHE operations (for testing)
- Real backend integration hooks

The executor provides:
- Plaintext reference implementation
- Simulated encrypted execution
- Performance metrics collection
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from ..bridge import CKKSTFHEBridge
from ..ir import IRProgram
from ..scheduler import ExecutionPlan
from ..tfhe_lut import LUTLibrary

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of gated LoRA execution."""
    # Output values
    output: np.ndarray

    # Intermediate values (for debugging)
    intermediates: Dict[str, np.ndarray] = field(default_factory=dict)

    # Gate value
    gate_value: float = 0.0

    # Timing
    total_time_ms: float = 0.0
    ckks_time_ms: float = 0.0
    tfhe_time_ms: float = 0.0
    bridge_time_ms: float = 0.0

    # Operation counts
    ckks_ops: int = 0
    tfhe_ops: int = 0
    bootstraps: int = 0

    # Errors
    quantization_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate_value': self.gate_value,
            'total_time_ms': self.total_time_ms,
            'ckks_time_ms': self.ckks_time_ms,
            'tfhe_time_ms': self.tfhe_time_ms,
            'bridge_time_ms': self.bridge_time_ms,
            'ckks_ops': self.ckks_ops,
            'tfhe_ops': self.tfhe_ops,
            'bootstraps': self.bootstraps,
            'quantization_error': self.quantization_error,
        }


class GatedLoRAExecutor:
    """
    Executor for gated LoRA programs.

    Supports multiple execution modes:
    - plaintext: Direct numpy computation (reference)
    - simulated: Simulated encrypted execution
    - encrypted: Real HE backend (not yet implemented)
    """

    def __init__(
        self,
        program: IRProgram,
        plan: ExecutionPlan,
        config: Optional[Any] = None,
        mode: str = "simulated",
    ):
        self.program = program
        self.plan = plan
        self.config = config
        self.mode = mode

        # Initialize components
        self.bridge = CKKSTFHEBridge()
        self.lut_library = LUTLibrary()

        # Weight storage
        self._weights: Dict[str, np.ndarray] = {}

        # Execution state
        self._values: Dict[str, np.ndarray] = {}

    def set_weights(
        self,
        lora_A: np.ndarray,
        lora_B: np.ndarray,
        w_gate: np.ndarray,
        b_gate: Optional[np.ndarray] = None,
    ) -> None:
        """
        Set LoRA and gate weights.

        Args:
            lora_A: LoRA A matrix [rank, hidden_size]
            lora_B: LoRA B matrix [hidden_size, rank]
            w_gate: Gate weight vector [1, hidden_size]
            b_gate: Gate bias scalar (optional)
        """
        self._weights['lora_A'] = lora_A.astype(np.float64)
        self._weights['lora_B'] = lora_B.astype(np.float64)
        self._weights['w_gate'] = w_gate.astype(np.float64)
        if b_gate is not None:
            self._weights['b_gate'] = b_gate.astype(np.float64)

    def execute(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
    ) -> ExecutionResult:
        """
        Execute gated LoRA on input.

        Args:
            x: Input activation [hidden_size]
            base_output: Base model output Wx [hidden_size]

        Returns:
            ExecutionResult with output and metrics
        """
        start_time = time.perf_counter()

        # Initialize values
        self._values = {
            'x': x.astype(np.float64),
            'base_output': base_output.astype(np.float64),
        }

        result = ExecutionResult(output=np.zeros_like(x))

        # Execute by phase
        ckks_start = time.perf_counter()

        # Phase 1: LoRA Delta
        self._execute_lora_delta()
        result.intermediates['delta'] = self._values.get('delta_rs', np.zeros(1))

        # Phase 2: Gate pre-activation
        self._execute_gate_preact()

        result.ckks_time_ms = (time.perf_counter() - ckks_start) * 1000
        result.ckks_ops = 4  # 2 matmuls + 2 rescales for delta, 1+1 for gate

        # Phase 3-5: Bridge and TFHE
        bridge_start = time.perf_counter()
        self._execute_bridge_to_tfhe()
        result.bridge_time_ms = (time.perf_counter() - bridge_start) * 1000

        tfhe_start = time.perf_counter()
        self._execute_gate_lut()
        result.tfhe_time_ms = (time.perf_counter() - tfhe_start) * 1000
        result.tfhe_ops = 1
        result.bootstraps = 1

        bridge_start = time.perf_counter()
        self._execute_bridge_to_ckks()
        result.bridge_time_ms += (time.perf_counter() - bridge_start) * 1000

        # Phase 6-7: Apply gate and final
        ckks_start = time.perf_counter()
        self._execute_apply_gate()
        self._execute_final_add()
        result.ckks_time_ms += (time.perf_counter() - ckks_start) * 1000
        result.ckks_ops += 2  # multiply + add

        # Collect results
        result.output = self._values.get('y', np.zeros_like(x))
        result.gate_value = float(self._values.get('g_ckks', np.array([0]))[0])
        result.total_time_ms = (time.perf_counter() - start_time) * 1000

        return result

    def _execute_lora_delta(self) -> None:
        """Execute LoRA delta computation: delta = B(Ax)."""
        x = self._values['x']
        A = self._weights['lora_A']
        B = self._weights['lora_B']

        # u = A @ x
        u = A @ x
        self._values['u'] = u
        self._values['u_rs'] = u  # Simulated rescale

        # delta = B @ u
        delta = B @ u
        self._values['delta'] = delta
        self._values['delta_rs'] = delta  # Simulated rescale

    def _execute_gate_preact(self) -> None:
        """Execute gate pre-activation: z = w_g^T @ x + b_g."""
        x = self._values['x']
        w_g = self._weights['w_gate']

        # z = w_g @ x
        z = w_g @ x
        if 'b_gate' in self._weights:
            z = z + self._weights['b_gate']

        self._values['z_pre'] = z
        self._values['z'] = z  # Simulated rescale

    def _execute_bridge_to_tfhe(self) -> None:
        """Bridge CKKS -> TFHE with quantization."""
        z = self._values['z']

        # Quantize
        z_scalar = float(z[0]) if hasattr(z, '__len__') else float(z)
        z_q, q_error = self.bridge.quantize_ckks_value(z_scalar)

        self._values['z_q'] = np.array([z_q])

        # Convert to TFHE representation
        tfhe_ct = self.bridge.ckks_to_tfhe(np.array([z_scalar]))
        self._values['z_tfhe'] = tfhe_ct

    def _execute_gate_lut(self) -> None:
        """Execute TFHE gate LUT."""
        tfhe_input = self._values['z_tfhe']

        # Get step LUT
        step_entries = self.lut_library.get_entries('step')
        if step_entries is None:
            # Fallback: simple step function
            z_val = tfhe_input['values'][0]
            g = 1 if z_val >= 0 else 0
            self._values['g_tfhe'] = {'values': [g], 'params': {'bits': 1}}
        else:
            # Apply LUT
            result = self.bridge.apply_lut_simulation(tfhe_input, step_entries)
            self._values['g_tfhe'] = result

    def _execute_bridge_to_ckks(self) -> None:
        """Bridge TFHE -> CKKS."""
        g_tfhe = self._values['g_tfhe']

        # Convert back to CKKS
        g_values = self.bridge.tfhe_to_ckks(g_tfhe)
        self._values['g_ckks'] = g_values

    def _execute_apply_gate(self) -> None:
        """Apply gate to LoRA delta: gated_delta = g * delta."""
        g = self._values['g_ckks']
        delta = self._values['delta_rs']

        # Broadcast gate if scalar
        if np.isscalar(g) or len(g) == 1:
            g_scalar = float(g[0]) if hasattr(g, '__len__') else float(g)
            gated_delta = g_scalar * delta
        else:
            gated_delta = g * delta

        self._values['gated_delta'] = gated_delta
        self._values['gated_delta_rs'] = gated_delta  # Simulated rescale

    def _execute_final_add(self) -> None:
        """Final output: y = base_output + gated_delta."""
        base = self._values['base_output']
        gated_delta = self._values['gated_delta_rs']

        y = base + gated_delta
        self._values['y'] = y

    def execute_simulated(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        weights: Dict[str, np.ndarray],
    ) -> ExecutionResult:
        """
        Execute gated LoRA with simulated encryption.

        This is a convenience method that combines set_weights and execute.

        Args:
            x: Input activation [hidden_size]
            base_output: Base model output Wx [hidden_size]
            weights: Dictionary containing:
                - lora_A: LoRA A matrix [rank, hidden_size]
                - lora_B: LoRA B matrix [hidden_size, rank]
                - w_gate: Gate weight vector [hidden_size]
                - b_gate: Gate bias [1] (optional)

        Returns:
            ExecutionResult with output and metrics
        """
        # Set weights from dictionary
        self.set_weights(
            lora_A=weights['lora_A'],
            lora_B=weights['lora_B'],
            w_gate=weights['w_gate'],
            b_gate=weights.get('b_gate'),
        )

        # Execute
        return self.execute(x, base_output)


def execute_gated_lora(
    # Inputs
    x: np.ndarray,
    base_output: np.ndarray,

    # Weights
    lora_A: np.ndarray,
    lora_B: np.ndarray,
    w_gate: np.ndarray,
    b_gate: Optional[np.ndarray] = None,

    # Config
    hidden_size: Optional[int] = None,
    lora_rank: Optional[int] = None,
) -> ExecutionResult:
    """
    Convenience function for gated LoRA execution.

    Args:
        x: Input activation
        base_output: Base model output Wx
        lora_A: LoRA A matrix
        lora_B: LoRA B matrix
        w_gate: Gate weight
        b_gate: Gate bias (optional)
        hidden_size: Override hidden size
        lora_rank: Override LoRA rank

    Returns:
        ExecutionResult
    """
    from .compiler import compile_gated_lora

    # Infer dimensions
    if hidden_size is None:
        hidden_size = len(x)
    if lora_rank is None:
        lora_rank = lora_A.shape[0]

    # Compile
    program, plan = compile_gated_lora(
        hidden_size=hidden_size,
        lora_rank=lora_rank,
    )

    # Create executor
    executor = GatedLoRAExecutor(program, plan)
    executor.set_weights(lora_A, lora_B, w_gate, b_gate)

    # Execute
    return executor.execute(x, base_output)


def plaintext_gated_lora(
    x: np.ndarray,
    base_output: np.ndarray,
    lora_A: np.ndarray,
    lora_B: np.ndarray,
    w_gate: np.ndarray,
    b_gate: float = 0.0,
    return_gate: bool = False,
):
    """
    Plaintext reference implementation of gated LoRA.

    y = Wx + g(x) * B(Ax)
    g(x) = step(w_g^T x + b_g)

    Args:
        x: Input activation [hidden_size]
        base_output: Base model output Wx [hidden_size]
        lora_A: LoRA A matrix [rank, hidden_size]
        lora_B: LoRA B matrix [hidden_size, rank]
        w_gate: Gate weight vector [hidden_size]
        b_gate: Gate bias scalar
        return_gate: If True, return (y, z, g) instead of just y

    Returns:
        If return_gate=False: output y [hidden_size]
        If return_gate=True: tuple of (output y, gate pre-activation z, gate value g)
    """
    # LoRA delta
    u = lora_A @ x
    delta = lora_B @ u

    # Gate pre-activation
    z = w_gate @ x
    if b_gate is not None:
        z = z + b_gate

    # Gate value (step function)
    g = 1.0 if float(z) >= 0 else 0.0

    # Output
    y = base_output + g * delta

    if return_gate:
        return y, float(z), g
    else:
        return y
