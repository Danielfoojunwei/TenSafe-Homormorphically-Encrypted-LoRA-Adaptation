"""
Integration Tests for Research Alignment (Gap Analysis Response)

Tests:
1. Client-Aided Bridge (Paper 3)
2. Speculative Batching (Paper 2)
"""

import unittest
import numpy as np
import logging
import sys
import os

# Add parent directory to path to allow importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.has.executor import HASExecutor
from client.gate_evaluator import GateEvaluator
from hybrid_compiler.gated_lora.executor import GatedLoRAExecutor
from hybrid_compiler.ir import IRProgram
from hybrid_compiler.scheduler import ExecutionPlan

# Configure logging
logging.basicConfig(level=logging.INFO)

class TestResearchAlignment(unittest.TestCase):
    def setUp(self):
        self.executor = HASExecutor(backend_type="SIMULATION")
        self.executor.initialize()

    def tearDown(self):
        self.executor.shutdown()

    def test_client_aided_flow(self):
        """Test the 2-Phase Client-Aided Bridge flow (Paper 3)."""
        logging.info("Testing Client-Aided Bridge Flow...")
        
        # 1. Load an adapter
        adapter_state = self.executor.load_adapter("test_gated", "model_x")
        
        # 2. Inject Gate weights manually to trigger "Gated" logic on the object
        # Create a GatedLoRAExecutor
        gated_executor = GatedLoRAExecutor(IRProgram(name="dummy"), ExecutionPlan(name="dummy"))
        
        # Set weights
        hidden_size = 1024
        # Need float64 for the executor backend
        w_gate = np.random.randn(hidden_size).astype(np.float64)
        lora_A = np.random.randn(16, hidden_size).astype(np.float64)
        lora_B = np.random.randn(hidden_size, 16).astype(np.float64)
        gated_executor.set_weights(lora_A, lora_B, w_gate)
        
        # Inject into adapter state
        adapter_state.gated_executor = gated_executor
        adapter_state.targets = "q"
        # Ensure we target layer 0
        if 0 not in adapter_state.loaded_layers:
            adapter_state.loaded_layers.append(0)
        
        # 3. Phase 1: Call apply_token_step (Server -> Client)
        # Prepare request first
        self.executor.prepare_request("req_1", "test_gated", batch_size=1, seq_len=1)

        # Input: (Batch=1, Seq=1, Hidden)
        hidden_states = np.random.randn(1, 1, hidden_size).astype(np.float16)
        
        # We expect it to return encrypted_gate_signal and NO delta
        delta, encrypted_gate, timings = self.executor.apply_token_step(
            "req_1", 0, "q", hidden_states, is_gate_callback=False
        )
        
        self.assertIsNone(delta, "Phase 1 should not return delta")
        self.assertIsNotNone(encrypted_gate, "Phase 1 should return encrypted gate signal")
        
        # 4. Client Side: Decrypt and Evaluate
        evaluator = GateEvaluator()
        gate_bit = evaluator.decrypt_and_evaluate(encrypted_gate)
        logging.info(f"Client evaluated gate bit: {gate_bit}")
        self.assertIn(gate_bit, [0, 1])
        
        # 5. Phase 2: Callback with gate bit (Client -> Server)
        delta_2, encrypted_gate_2, timings_2 = self.executor.apply_token_step(
            "req_1", 0, "q", hidden_states, 
            is_gate_callback=True, client_gate_bit=gate_bit
        )
        
        self.assertIsNotNone(delta_2, "Phase 2 should return delta")
        self.assertIsNone(encrypted_gate_2, "Phase 2 should not return encrypted gate signal")
        self.assertEqual(delta_2.shape, hidden_states.shape, "Delta shape mismatch")
        
        logging.info("Client-Aided Bridge Flow Verified.")

    def test_speculative_batching(self):
        """Test Flattened SIMD Packing for Speculative Batching (Paper 2)."""
        logging.info("Testing Speculative Batching...")
        
        # 1. Load regular adapter
        adapter_state = self.executor.load_adapter("test_spec", "model_x")
        
        # 2. Create Speculative Batch Input (K=4)
        K = 4
        hidden_size = 1024
        # VLLM shape: (1, K, Hidden) for a single sequence with K speculative tokens
        input_states = np.random.randn(1, K, hidden_size).astype(np.float16)
        
        # Ensure adapter has weights
        # Force layer 0, projection 'q'
        adapter_state.targets = "q"
        if 0 not in adapter_state.loaded_layers:
            adapter_state.loaded_layers.append(0)
            
        # Re-generate mock weights to ensure coverage
        self.executor._generate_mock_weights(adapter_state)
        
        # Prepare request
        self.executor.prepare_request("req_spec", "test_spec", batch_size=1, seq_len=K)

        # Get baseline operations count
        initial_ops = self.executor._total_operations
        initial_tokens = self.executor._total_tokens
        
        # 3. Call apply_token_step
        delta, _, _ = self.executor.apply_token_step(
            "req_spec", 0, "q", input_states
        )
        
        # 4. Assertions
        self.assertIsNotNone(delta)
        self.assertEqual(delta.shape, input_states.shape)
        
        # Check operations count
        final_ops = self.executor._total_operations
        final_tokens = self.executor._total_tokens
        
        ops_diff = final_ops - initial_ops
        tokens_diff = final_tokens - initial_tokens
        
        logging.info(f"Ops increase: {ops_diff}, Tokens increase: {tokens_diff}")
        
        # Verify Packing: "Flattened SIMD Packing" means 1 operation for K tokens
        self.assertEqual(ops_diff, 1, f"Should count as 1 SIMD operation, got {ops_diff}")
        self.assertEqual(tokens_diff, K, f"Should process K={K} tokens, got {tokens_diff}")
        
        logging.info("Speculative Batching Verified.")

if __name__ == '__main__':
    unittest.main()
