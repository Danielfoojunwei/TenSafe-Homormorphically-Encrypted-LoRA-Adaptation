"""
Integration test for the Client-Aided Bridge (Two-Phase Flow).
Verifies that the HAS service correctly triggers gating and accepts client callbacks.
"""
import pytest
import numpy as np
from he_lora_microkernel.services.has.executor import HASExecutor
from he_lora_microkernel.client.gate_evaluator import GateEvaluator

class TestClientAidedFlow:
    """Tests the two-phase gated LoRA execution."""

    def test_e2e_two_phase_gating(self):
        """
        Verifies the complete cycle:
        1. Server Phase 1 -> Returns encrypted gate signal.
        2. Client Evaluation -> Decrypts and evaluates gate bit.
        3. Server Phase 2 -> Returns LoRA delta gated by client bit.
        """
        # 1. Setup Server
        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()
        
        # Load adapter with gate weights
        adapter_id = "kimi-moe"
        state = executor.load_adapter(
            adapter_id=adapter_id,
            model_id="kimi-2.5",
            rank=16,
            targets="qkv"
        )
        
        # Manually inject gate weights for testing
        # w_gate: (rank, hidden)
        weights = state.weights[0] # Layer 0
        weights['w_gate'] = np.random.randn(16, 1024).astype(np.float16)
        
        # Re-initialize gated executor to pick up new weights
        from he_lora_microkernel.hybrid_compiler.gated_lora.executor import GatedLoRAExecutor
        from he_lora_microkernel.hybrid_compiler.ir import IRProgram
        from he_lora_microkernel.hybrid_compiler.scheduler import ExecutionPlan
        
        state.gated_executor = GatedLoRAExecutor(IRProgram(name="test"), ExecutionPlan(name="test"))
        state.gated_executor.set_weights(
            lora_A=weights['q_A'],
            lora_B=weights['q_B'],
            w_gate=weights['w_gate']
        )
        
        executor.prepare_request(
            request_id="req-moai",
            adapter_id=adapter_id,
            batch_size=1,
            seq_len=1
        )

        # 2. Phase 1: Request Gated LoRA
        hidden_states = np.random.randn(1, 1, 1024).astype(np.float16)
        
        delta_p1, gate_signal, timing = executor.apply_token_step(
            request_id="req-moai",
            layer_idx=0,
            projection_type="q",
            hidden_states=hidden_states,
            is_gate_callback=False
        )
        
        assert delta_p1 is None, "Phase 1 should not return delta"
        assert gate_signal is not None, "Phase 1 should return gate signal"
        
        # 3. Client Side: Evaluate Gate
        evaluator = GateEvaluator()
        gate_bit = evaluator.decrypt_and_evaluate(gate_signal)
        assert gate_bit in [0, 1]
        
        # 4. Phase 2: Callback with Gate Bit
        delta_p2, _, _ = executor.apply_token_step(
            request_id="req-moai",
            layer_idx=0,
            projection_type="q",
            hidden_states=hidden_states,
            is_gate_callback=True,
            client_gate_bit=gate_bit
        )
        
        assert delta_p2 is not None, "Phase 2 should return gated delta"
        assert delta_p2.shape == (1, 1, 1024)
        
        # Cleanup
        executor.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
