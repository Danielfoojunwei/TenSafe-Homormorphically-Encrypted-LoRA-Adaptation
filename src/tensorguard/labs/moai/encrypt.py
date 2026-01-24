"""
MOAI Encryption/Decryption (TenSEAL)
For client-side preprocessing and postprocessing.
"""

import numpy as np
import tenseal as ts
from typing import List, Union

from .moai_config import MoaiConfig

class MoaiEncryptor:
    """Client-side encryptor (TenSEAL)."""
    
    def __init__(self, key_id: str, context_bytes: bytes):
        self.key_id = key_id
        # Load context from bytes (includes keys)
        self.ctx = ts.context_from(context_bytes)
        
    def encrypt_vector(self, vector: np.ndarray) -> bytes:
        """
        Encrypt a numpy vector into a REAL CKKS ciphertext.
        """
        # Ensure it's a 1D vector or flatten it for CKKS packing
        if hasattr(vector, 'flatten'):
            vec_flat = vector.flatten().tolist()
        else:
            vec_flat = list(vector)
            
        enc_vec = ts.ckks_vector(self.ctx, vec_flat)
        return enc_vec.serialize()

class MoaiDecryptor:
    """Client-side decryptor (TenSEAL)."""
    
    def __init__(self, key_id: str, context_bytes: bytes):
        self.key_id = key_id
        self.ctx = ts.context_from(context_bytes)
        
    def decrypt_vector(self, ciphertext: bytes) -> np.ndarray:
        """
        Decrypt a CKKS ciphertext.
        """
        try:
            # We need to construct the CKKS vector linked to our context
            enc_vec = ts.ckks_vector_from(self.ctx, ciphertext)
            
            # Decrypt
            decrypted_list = enc_vec.decrypt()
            return np.array(decrypted_list)
            
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")
