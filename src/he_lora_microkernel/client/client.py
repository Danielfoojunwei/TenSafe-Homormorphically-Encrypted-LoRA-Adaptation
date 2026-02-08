"""
TenSafe Client
"""
import logging
from typing import Any

import grpc

# Import proto modules - need to be careful with imports
try:
    from ..services.proto import has_pb2, has_pb2_grpc
except ImportError:
    # Fallback for when running as script
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from he_lora_microkernel.services.proto import has_pb2, has_pb2_grpc

from .gate_evaluator import GateEvaluator

logger = logging.getLogger(__name__)

class TenSafeClient:
    """
    Client for the TenSafe HE Adapter Service.
    Handles the Client-Aided Bridge protocol.
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = has_pb2_grpc.HEAdapterServiceStub(self.channel)
        self.gate_evaluator = GateEvaluator()

    def process_token(
        self, 
        request_id: str, 
        layer_idx: int, 
        projection_type: str, 
        token_idx: int
    ) -> Any:
        """
        Process a token step, handling the interactive gate protocol if needed.
        """
        # Initial call (Phase 1)
        response = self.stub.ApplyTokenStep(has_pb2.ApplyTokenStepRequest(
            request_id=request_id,
            layer_idx=layer_idx,
            projection_type=projection_type,
            token_idx=token_idx,
            is_gate_callback=False
        ))

        # Check if Gate Evaluation is required (Phase 2)
        if response.gate_required:
            logger.debug(f"Gate required for req={request_id} layer={layer_idx}")
            
            # Client-Side Decryption & Evaluation
            gate_bit = self.gate_evaluator.decrypt_and_evaluate(response.encrypted_gate_signal)
            
            # Callback with gate bit (Phase 2)
            response = self.stub.ApplyTokenStep(has_pb2.ApplyTokenStepRequest(
                request_id=request_id,
                layer_idx=layer_idx,
                projection_type=projection_type,
                token_idx=token_idx,
                is_gate_callback=True,
                client_gate_bit=gate_bit
            ))
            
        return response
        
    def health_check(self) -> bool:
        """Check service health."""
        try:
            resp = self.stub.HealthCheck(has_pb2.HealthCheckRequest())
            return resp.healthy
        except grpc.RpcError:
             return False

    def load_adapter(self, adapter_id: str, model_id: str, adapter_path: str = ""):
        """Load an adapter."""
        return self.stub.LoadAdapter(has_pb2.LoadAdapterRequest(
            adapter_id=adapter_id,
            model_id=model_id,
            adapter_path=adapter_path
        ))
