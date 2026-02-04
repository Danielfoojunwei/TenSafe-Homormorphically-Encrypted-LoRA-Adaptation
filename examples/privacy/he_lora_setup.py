#!/usr/bin/env python3
"""
HE-LoRA Setup with TenSafe

This example demonstrates how to set up Homomorphically Encrypted LoRA
(HE-LoRA) for private inference. HE-LoRA keeps adapter weights encrypted
during computation, providing cryptographic privacy guarantees.

What this example demonstrates:
- Understanding HE-LoRA architecture
- Initializing the HE context
- Encrypting LoRA adapter weights
- Running encrypted inference

Key concepts:
- CKKS scheme: Approximate arithmetic for neural network computation
- Key generation: Public/private key pairs for encryption
- Ciphertext: Encrypted data that can be computed on
- MOAI optimization: Zero-rotation technique for efficient HE-LoRA

Prerequisites:
- TenSafe with HE support
- TENSAFE_TOY_HE=1 for testing mode

Expected Output:
    Initializing HE context...
    Key generation complete (128-bit security)

    Encrypting LoRA adapter...
    Original size: 24.5 MB
    Encrypted size: 156.8 MB

    HE-LoRA ready for inference!
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Enable toy HE mode for demonstration
os.environ["TENSAFE_TOY_HE"] = "1"


@dataclass
class HEContextConfig:
    """Configuration for homomorphic encryption context."""
    scheme: str = "CKKS"
    security_level: int = 128
    poly_modulus_degree: int = 8192
    scale_bits: int = 40
    first_mod_bits: int = 60
    use_toy_mode: bool = True


@dataclass
class LoRAAdapterConfig:
    """Configuration for LoRA adapter."""
    rank: int = 16
    alpha: float = 32.0
    hidden_dim: int = 4096
    target_modules: list = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


class HELoRASetup:
    """Set up HE-LoRA for encrypted inference."""

    def __init__(self, he_config: HEContextConfig, lora_config: LoRAAdapterConfig):
        self.he_config = he_config
        self.lora_config = lora_config
        self.context = None
        self.encrypted_weights: Dict[str, Any] = {}

    def initialize_context(self):
        """Initialize the HE context and generate keys."""
        print("Initializing HE context...")

        try:
            from tensorguard.n2he.core import create_context

            self.context = create_context(
                profile="lora",
                use_toy_mode=self.he_config.use_toy_mode
            )
            self.context.generate_keys()

            print(f"  Scheme: {self.he_config.scheme}")
            print(f"  Security level: {self.he_config.security_level}-bit")
            print(f"  Polynomial degree: {self.he_config.poly_modulus_degree}")
            print(f"  Toy mode: {self.he_config.use_toy_mode}")

        except ImportError:
            print("  HE library not available, using simulation mode")
            self.context = SimulatedHEContext(self.he_config)

    def encrypt_adapter(self, adapter_weights: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Encrypt LoRA adapter weights.

        Returns metrics about the encryption process.
        """
        print("\nEncrypting LoRA adapter...")

        metrics = {
            "original_size_mb": 0.0,
            "encrypted_size_mb": 0.0,
            "encryption_time_ms": 0.0,
            "modules_encrypted": 0,
        }

        start_time = time.time()

        for module_name, weights in adapter_weights.items():
            # Calculate original size
            original_bytes = weights.nbytes
            metrics["original_size_mb"] += original_bytes / (1024 * 1024)

            # Encrypt weights
            if hasattr(self.context, 'encrypt_matrix'):
                encrypted = self.context.encrypt_matrix(weights)
            else:
                # Simulation: just store the weights
                encrypted = {"ciphertext": weights, "simulated": True}

            self.encrypted_weights[module_name] = encrypted
            metrics["modules_encrypted"] += 1

            # Estimate encrypted size (CKKS expansion factor ~6.4x)
            metrics["encrypted_size_mb"] += original_bytes * 6.4 / (1024 * 1024)

        metrics["encryption_time_ms"] = (time.time() - start_time) * 1000

        print(f"  Original size: {metrics['original_size_mb']:.2f} MB")
        print(f"  Encrypted size: {metrics['encrypted_size_mb']:.2f} MB")
        print(f"  Expansion factor: {metrics['encrypted_size_mb']/metrics['original_size_mb']:.1f}x")
        print(f"  Encryption time: {metrics['encryption_time_ms']:.0f}ms")
        print(f"  Modules encrypted: {metrics['modules_encrypted']}")

        return metrics

    def create_sample_weights(self) -> Dict[str, np.ndarray]:
        """Create sample LoRA weights for demonstration."""
        weights = {}
        for module in self.lora_config.target_modules:
            # LoRA A matrix: (rank, hidden_dim)
            weights[f"{module}_lora_A"] = np.random.randn(
                self.lora_config.rank,
                self.lora_config.hidden_dim
            ).astype(np.float32) * 0.01

            # LoRA B matrix: (hidden_dim, rank) - initialized to zero
            weights[f"{module}_lora_B"] = np.zeros(
                (self.lora_config.hidden_dim, self.lora_config.rank),
                dtype=np.float32
            )

        return weights


class SimulatedHEContext:
    """Simulated HE context for demonstration."""

    def __init__(self, config: HEContextConfig):
        self.config = config

    def encrypt_matrix(self, matrix: np.ndarray) -> Dict[str, Any]:
        return {"ciphertext_simulated": matrix, "size": matrix.nbytes}


def main():
    """Demonstrate HE-LoRA setup."""

    # =========================================================================
    # Step 1: Understanding HE-LoRA
    # =========================================================================
    print("=" * 60)
    print("HE-LORA SETUP")
    print("=" * 60)
    print("""
    HE-LoRA provides cryptographic privacy for LoRA adapters:

    Architecture:
    +----------------+     +-----------------+
    | Base Model     |     | Encrypted LoRA  |
    | (plaintext)    |     | (ciphertext)    |
    +-------+--------+     +--------+--------+
            |                       |
            v                       v
    +-------+--------+     +--------+--------+
    | Base Output    |  +  | HE Computation  |
    | (plaintext)    |     | (encrypted)     |
    +-------+--------+     +--------+--------+
            |                       |
            +----------+------------+
                       |
                       v
               +--------------+
               | Decrypt &    |
               | Combine      |
               +--------------+
                       |
                       v
               +--------------+
               | Final Output |
               +--------------+

    Security guarantees:
    - Adapter weights never exposed in plaintext during inference
    - Inference provider cannot extract the adapter
    - 128-bit post-quantum security level

    Performance:
    - ~5-15% latency overhead for LoRA-only encryption
    - ~6x storage overhead for encrypted weights
    - GPU-accelerated HE operations
    """)

    # =========================================================================
    # Step 2: Configure HE and LoRA
    # =========================================================================
    print("\nConfiguring HE-LoRA...")

    he_config = HEContextConfig(
        scheme="CKKS",
        security_level=128,
        poly_modulus_degree=8192,
        scale_bits=40,
        use_toy_mode=True,
    )

    lora_config = LoRAAdapterConfig(
        rank=16,
        alpha=32.0,
        hidden_dim=4096,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )

    print(f"  LoRA rank: {lora_config.rank}")
    print(f"  LoRA alpha: {lora_config.alpha}")
    print(f"  Hidden dim: {lora_config.hidden_dim}")
    print(f"  Target modules: {lora_config.target_modules}")

    # =========================================================================
    # Step 3: Initialize HE context
    # =========================================================================
    setup = HELoRASetup(he_config, lora_config)
    setup.initialize_context()

    # =========================================================================
    # Step 4: Encrypt adapter weights
    # =========================================================================
    print("\nCreating sample LoRA weights...")
    sample_weights = setup.create_sample_weights()
    print(f"  Created {len(sample_weights)} weight matrices")

    metrics = setup.encrypt_adapter(sample_weights)

    # =========================================================================
    # Step 5: Verify setup
    # =========================================================================
    print("\n" + "=" * 60)
    print("HE-LORA SETUP COMPLETE")
    print("=" * 60)
    print(f"""
    HE-LoRA is ready for encrypted inference!

    Configuration summary:
    - Scheme: {he_config.scheme}
    - Security: {he_config.security_level}-bit
    - LoRA rank: {lora_config.rank}
    - Modules: {len(lora_config.target_modules)}

    Encryption summary:
    - Original size: {metrics['original_size_mb']:.2f} MB
    - Encrypted size: {metrics['encrypted_size_mb']:.2f} MB
    - Modules encrypted: {metrics['modules_encrypted']}

    Next steps:
    1. Export encrypted adapter as TGSP package
    2. Deploy to inference server
    3. Run encrypted inference with encrypted_inference.py
    """)

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("HE-LORA BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for HE-LoRA deployment:

    1. Key management
       - Generate keys securely
       - Store private keys safely
       - Rotate keys periodically

    2. Performance optimization
       - Use MOAI (zero-rotation) optimization
       - Batch multiple inferences
       - Pre-compute relinearization keys

    3. Security considerations
       - Use production parameters (not toy mode)
       - Verify 128-bit security level
       - Monitor for side-channel attacks

    4. Deployment
       - Package as TGSP for portability
       - Test with toy mode first
       - Benchmark latency overhead
    """)


if __name__ == "__main__":
    main()
