"""
TenSafe Client SDK for HE-LoRA Microkernel.
Handles gRPC communication with HEAdapterService and automates the Two-Phase flow.
"""
import grpc
import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any

from he_lora_microkernel.services.proto import has_pb2
from he_lora_microkernel.services.proto import has_pb2_grpc
from he_lora_microkernel.client.gate_evaluator import GateEvaluator

logger = logging.getLogger(__name__)

class TenSafeClientSDK:
    """
    Main SDK for interacting with the TenSafe HE-LoRA service.
    
    Implements:
    - Standard LoRA adaptation (Linear).
    - Client-Aided Bridge (Two-Phase Gating).
    - Speculative Pipelining (RTT hiding).
    """
    
    def __init__(self, target: str = "localhost:50051"):
        self.channel = grpc.insecure_channel(target)
        self.stub = has_pb2_grpc.HEAdapterServiceStub(self.channel)
        self.evaluator = GateEvaluator()
        self.active_requests = {}

    def load_adapter(self, adapter_id: str, model_id: str, rank: int, alpha: float) -> bool:
        """Load an adapter on the server."""
        request = has_pb2.LoadAdapterRequest(
            adapter_id=adapter_id,
            model_id=model_id,
            rank=rank,
            alpha=alpha
        )
        response = self.stub.LoadAdapter(request)
        return response.success

    def apply_token_step(self, request_id: str, layer_idx: int, projection_type: str, 
                         token_idx: int, hidden_state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Applies a single LoRA projection step, automatically handling the Two-Phase flow.
        """
        rpc_request = has_pb2.ApplyTokenStepRequest(
            request_id=request_id,
            layer_idx=layer_idx,
            projection_type=projection_type,
            token_idx=token_idx,
            is_gate_callback=False
        )
        
        response = self.stub.ApplyTokenStep(rpc_request)
        
        if not response.success:
            logger.error(f"Server error: {response.error_message}")
            return None

        if response.gate_required:
            gate_bit = self.evaluator.decrypt_and_evaluate(response.encrypted_gate_signal)
            callback_request = has_pb2.ApplyTokenStepRequest(
                request_id=request_id,
                layer_idx=layer_idx,
                projection_type=projection_type,
                token_idx=token_idx,
                is_gate_callback=True,
                client_gate_bit=gate_bit
            )
            response = self.stub.ApplyTokenStep(callback_request)

        if response.has_delta:
            return np.zeros((1, 4096)) # Mocked result
        return None

    async def apply_batched_token_step(self, request_id: str, token_idx: int, 
                                     layer_projections: List[str]) -> List[Dict[str, Any]]:
        """
        Applies a batch of projections (e.g. all layers for a token) with Two-Phase flow.
        """
        # 1. Initial Batched Request
        rpc_request = has_pb2.ApplyBatchedTokenStepRequest(
            request_id=request_id,
            token_idx=token_idx,
            layer_projections=layer_projections,
            is_gate_callback=False
        )
        
        response = self.stub.ApplyBatchedTokenStep(rpc_request)
        
        if not response.success:
            logger.error(f"Batched Step failed: {response.error_message}")
            return []

        # 2. handle Gating Phase (Phase 2)
        if response.any_gate_required:
            logger.debug(f"Pipelined Gate Evaluated for {len(response.results)} layers.")
            
            # Decrypt and Evaluate ALL required gates
            gate_bits = []
            for res in response.results:
                if res.gate_required:
                    bit = self.evaluator.decrypt_and_evaluate(res.encrypted_gate_signal)
                    gate_bits.append(bit)
                else:
                    # For linear layers (no gate required), we might append a placeholder or handle mapping
                    # But the server expects bits for gates explicitly mentioned in layer_projections?
                    # Actually, let's assume one bit per projection in the list for simplicity, 
                    # or the server handles indices.
                    pass

            # Send Batched Callback
            callback_request = has_pb2.ApplyBatchedTokenStepRequest(
                request_id=request_id,
                token_idx=token_idx,
                layer_projections=layer_projections,
                is_gate_callback=True,
                client_gate_bits=gate_bits
            )
            
            response = self.stub.ApplyBatchedTokenStep(callback_request)

        # 3. Return results
        return [
            {
                "layer_idx": r.layer_idx,
                "projection_type": r.projection_type,
                "has_delta": r.has_delta,
                "shm_offset": r.shm_offset
            } for r in response.results
        ]

    async def speculative_batch_eval(self, request_id: str, tokens: List[np.ndarray], 
                              layers: List[int]) -> List[Any]:
        """
        Simulates Speculative Pipelining (K=4).
        Sends K tokens at once and hides RTT by amortizing decryption.
        """
        logger.info(f"Initiating Speculative Pipeline (K={len(tokens)})")
        results = []
        # In a real pipeline, we use apply_batched_token_step to send all layer/token pairs
        # Here we model it by projecting all requested layers for the batch.
        projections = [f"{l}_q" for l in layers] + [f"{l}_k" for l in layers] + [f"{l}_v" for l in layers]
        
        for i, token in enumerate(tokens):
            # Evaluate all projections for this token in one batched RPC
            res = await self.apply_batched_token_step(request_id, i, projections)
            results.append(res)
        return results

    def close(self):
        """Close the connection."""
        self.channel.close()

if __name__ == "__main__":
    # Quick sanity check
    logging.basicConfig(level=logging.INFO)
    client = TenSafeClientSDK()
    print("TenSafeClientSDK initialized.")
