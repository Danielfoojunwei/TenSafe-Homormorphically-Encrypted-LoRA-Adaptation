"""
Gate Evaluator for Client-Aided Bridge
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

class GateEvaluator:
    """
    Evaluates non-linear gates on the client side.
    
    In a real implementation, this would hold the HE private key
    and perform TFHE decryption.
    In this simulation, it interprets the mock encrypted bytes as float64.
    """
    
    def __init__(self, private_key: bytes = b"mock_key"):
        self.private_key = private_key

    def decrypt_and_evaluate(self, encrypted_gate_signal: bytes) -> int:
        """
        Decrypt the gate signal and evaluate the non-linear function.
        Simulated decryption: just interpret bytes as float.
        
        Args:
            encrypted_gate_signal: Encrypted bytes from server
            
        Returns:
            Gate bit (0 or 1)
        """
        try:
            if not encrypted_gate_signal:
                return 0
                
            # Simulate decryption: read float from bytes
            # The server sends np.array([gate_scalar]).tobytes()
            gate_signal = np.frombuffer(encrypted_gate_signal, dtype=np.float64)[0]
            
            # Evaluate ReLU/Threshold
            # g(x) = 1 if x >= 0 else 0
            gate_bit = 1 if gate_signal >= 0 else 0
            
            logger.debug(f"Evaluated gate: signal={gate_signal:.4f} -> bit={gate_bit}")
            return gate_bit
            
        except Exception as e:
            logger.error(f"Gate evaluation failed: {e}")
            return 0
