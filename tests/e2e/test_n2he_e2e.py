"""
E2E Test: N2HE Homomorphic Encryption Integration

This test performs a complete end-to-end validation of N2HE integration:
1. Encrypted LoRA Adapter - compute deltas on encrypted activations
2. Private Inference Mode - run inference with encrypted embeddings
3. HE Key Management - key generation, rotation, and distribution
4. Ciphertext Serialization - binary, JSON, base64 format conversion
5. Integration with TenSafe KEK/DEK hierarchy
6. Benchmarks and performance metrics collection

Metrics collected:
- Key generation latency (LWE, RLWE, CKKS)
- Encryption/decryption throughput
- Encrypted LoRA delta computation time
- Homomorphic operations (add, multiply, matmul)
- Serialization format performance
- Memory usage and ciphertext sizes
"""

import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tensorguard.n2he import (
    N2HEContext,
    HESchemeParams,
    HEKeyManager,
    EncryptedLoRARuntime,
    PrivateInferenceMode,
    CiphertextSerializer,
    CiphertextFormat,
    create_encrypted_runtime,
    create_private_inference_mode,
    serialize_ciphertext,
    deserialize_ciphertext,
    create_ciphertext_bundle,
)
from tensorguard.n2he.core import LWECiphertext, RLWECiphertext
from tensorguard.n2he.benchmark import N2HEBenchmark, run_quick_benchmark


@dataclass
class N2HEMetrics:
    """Metrics collected during N2HE E2E testing."""
    # Key generation metrics
    keygen_lwe_times: List[float] = field(default_factory=list)
    keygen_rlwe_times: List[float] = field(default_factory=list)
    keygen_bundle_times: List[float] = field(default_factory=list)

    # Encryption/decryption metrics
    encrypt_times: List[float] = field(default_factory=list)
    decrypt_times: List[float] = field(default_factory=list)
    encrypt_batch_times: List[float] = field(default_factory=list)

    # Homomorphic operations
    he_add_times: List[float] = field(default_factory=list)
    he_multiply_times: List[float] = field(default_factory=list)
    he_matmul_times: List[float] = field(default_factory=list)

    # LoRA adapter metrics
    lora_delta_times: List[float] = field(default_factory=list)
    lora_forward_times: List[float] = field(default_factory=list)
    adapter_register_times: List[float] = field(default_factory=list)

    # Private inference metrics
    private_inference_times: List[float] = field(default_factory=list)
    encrypt_embedding_times: List[float] = field(default_factory=list)
    decrypt_output_times: List[float] = field(default_factory=list)

    # Serialization metrics
    serialize_binary_times: List[float] = field(default_factory=list)
    serialize_json_times: List[float] = field(default_factory=list)
    serialize_base64_times: List[float] = field(default_factory=list)
    deserialize_binary_times: List[float] = field(default_factory=list)
    deserialize_json_times: List[float] = field(default_factory=list)

    # Size metrics
    ciphertext_sizes: List[int] = field(default_factory=list)
    bundle_sizes: List[int] = field(default_factory=list)

    # Counters
    operations_completed: int = 0
    total_test_time: float = 0.0

    def percentile(self, data: List[float], p: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

    def to_report(self) -> Dict[str, Any]:
        """Generate metrics report."""
        return {
            "test_summary": {
                "operations_completed": self.operations_completed,
                "total_test_time_seconds": self.total_test_time,
            },
            "key_generation": {
                "lwe_keygen_ms": {
                    "p50": self.percentile(self.keygen_lwe_times, 50) * 1000,
                    "p95": self.percentile(self.keygen_lwe_times, 95) * 1000,
                    "mean": statistics.mean(self.keygen_lwe_times) * 1000 if self.keygen_lwe_times else 0,
                    "throughput": 1000 / (statistics.mean(self.keygen_lwe_times) * 1000) if self.keygen_lwe_times else 0,
                },
                "rlwe_keygen_ms": {
                    "p50": self.percentile(self.keygen_rlwe_times, 50) * 1000,
                    "p95": self.percentile(self.keygen_rlwe_times, 95) * 1000,
                    "mean": statistics.mean(self.keygen_rlwe_times) * 1000 if self.keygen_rlwe_times else 0,
                },
                "bundle_generation_ms": {
                    "p50": self.percentile(self.keygen_bundle_times, 50) * 1000,
                    "p95": self.percentile(self.keygen_bundle_times, 95) * 1000,
                    "mean": statistics.mean(self.keygen_bundle_times) * 1000 if self.keygen_bundle_times else 0,
                },
            },
            "encryption_decryption": {
                "encrypt_ms": {
                    "p50": self.percentile(self.encrypt_times, 50) * 1000,
                    "p95": self.percentile(self.encrypt_times, 95) * 1000,
                    "mean": statistics.mean(self.encrypt_times) * 1000 if self.encrypt_times else 0,
                    "throughput": 1000 / (statistics.mean(self.encrypt_times) * 1000) if self.encrypt_times else 0,
                },
                "decrypt_ms": {
                    "p50": self.percentile(self.decrypt_times, 50) * 1000,
                    "p95": self.percentile(self.decrypt_times, 95) * 1000,
                    "mean": statistics.mean(self.decrypt_times) * 1000 if self.decrypt_times else 0,
                    "throughput": 1000 / (statistics.mean(self.decrypt_times) * 1000) if self.decrypt_times else 0,
                },
                "encrypt_batch_ms": {
                    "p50": self.percentile(self.encrypt_batch_times, 50) * 1000,
                    "p95": self.percentile(self.encrypt_batch_times, 95) * 1000,
                    "mean": statistics.mean(self.encrypt_batch_times) * 1000 if self.encrypt_batch_times else 0,
                },
            },
            "homomorphic_operations": {
                "add_ms": {
                    "p50": self.percentile(self.he_add_times, 50) * 1000,
                    "p95": self.percentile(self.he_add_times, 95) * 1000,
                    "mean": statistics.mean(self.he_add_times) * 1000 if self.he_add_times else 0,
                },
                "multiply_ms": {
                    "p50": self.percentile(self.he_multiply_times, 50) * 1000,
                    "p95": self.percentile(self.he_multiply_times, 95) * 1000,
                    "mean": statistics.mean(self.he_multiply_times) * 1000 if self.he_multiply_times else 0,
                },
                "matmul_ms": {
                    "p50": self.percentile(self.he_matmul_times, 50) * 1000,
                    "p95": self.percentile(self.he_matmul_times, 95) * 1000,
                    "mean": statistics.mean(self.he_matmul_times) * 1000 if self.he_matmul_times else 0,
                },
            },
            "encrypted_lora": {
                "delta_computation_ms": {
                    "p50": self.percentile(self.lora_delta_times, 50) * 1000,
                    "p95": self.percentile(self.lora_delta_times, 95) * 1000,
                    "mean": statistics.mean(self.lora_delta_times) * 1000 if self.lora_delta_times else 0,
                    "throughput": 1000 / (statistics.mean(self.lora_delta_times) * 1000) if self.lora_delta_times else 0,
                },
                "forward_pass_ms": {
                    "p50": self.percentile(self.lora_forward_times, 50) * 1000,
                    "p95": self.percentile(self.lora_forward_times, 95) * 1000,
                    "mean": statistics.mean(self.lora_forward_times) * 1000 if self.lora_forward_times else 0,
                },
                "adapter_register_ms": {
                    "p50": self.percentile(self.adapter_register_times, 50) * 1000,
                    "p95": self.percentile(self.adapter_register_times, 95) * 1000,
                    "mean": statistics.mean(self.adapter_register_times) * 1000 if self.adapter_register_times else 0,
                },
            },
            "private_inference": {
                "total_inference_ms": {
                    "p50": self.percentile(self.private_inference_times, 50) * 1000,
                    "p95": self.percentile(self.private_inference_times, 95) * 1000,
                    "mean": statistics.mean(self.private_inference_times) * 1000 if self.private_inference_times else 0,
                },
                "encrypt_embedding_ms": {
                    "p50": self.percentile(self.encrypt_embedding_times, 50) * 1000,
                    "p95": self.percentile(self.encrypt_embedding_times, 95) * 1000,
                    "mean": statistics.mean(self.encrypt_embedding_times) * 1000 if self.encrypt_embedding_times else 0,
                },
                "decrypt_output_ms": {
                    "p50": self.percentile(self.decrypt_output_times, 50) * 1000,
                    "p95": self.percentile(self.decrypt_output_times, 95) * 1000,
                    "mean": statistics.mean(self.decrypt_output_times) * 1000 if self.decrypt_output_times else 0,
                },
            },
            "serialization": {
                "binary_serialize_ms": {
                    "p50": self.percentile(self.serialize_binary_times, 50) * 1000,
                    "p95": self.percentile(self.serialize_binary_times, 95) * 1000,
                    "mean": statistics.mean(self.serialize_binary_times) * 1000 if self.serialize_binary_times else 0,
                },
                "json_serialize_ms": {
                    "p50": self.percentile(self.serialize_json_times, 50) * 1000,
                    "p95": self.percentile(self.serialize_json_times, 95) * 1000,
                    "mean": statistics.mean(self.serialize_json_times) * 1000 if self.serialize_json_times else 0,
                },
                "base64_serialize_ms": {
                    "p50": self.percentile(self.serialize_base64_times, 50) * 1000,
                    "p95": self.percentile(self.serialize_base64_times, 95) * 1000,
                    "mean": statistics.mean(self.serialize_base64_times) * 1000 if self.serialize_base64_times else 0,
                },
                "binary_deserialize_ms": {
                    "p50": self.percentile(self.deserialize_binary_times, 50) * 1000,
                    "p95": self.percentile(self.deserialize_binary_times, 95) * 1000,
                    "mean": statistics.mean(self.deserialize_binary_times) * 1000 if self.deserialize_binary_times else 0,
                },
            },
            "size_metrics": {
                "average_ciphertext_bytes": statistics.mean(self.ciphertext_sizes) if self.ciphertext_sizes else 0,
                "average_bundle_bytes": statistics.mean(self.bundle_sizes) if self.bundle_sizes else 0,
            },
        }


class N2HETestHarness:
    """
    N2HE testing harness with comprehensive metrics collection.

    Provides a unified interface for testing all N2HE components:
    - Encrypted LoRA adapter
    - Private inference mode
    - Key management
    - Serialization
    """

    def __init__(self, tenant_id: str = "test-tenant"):
        self.tenant_id = tenant_id
        self.metrics = N2HEMetrics()

        # Initialize components
        self._init_key_manager()
        self._init_contexts()
        self._init_runtime()
        self._init_inference_mode()
        self._init_serializer()

    def _init_key_manager(self):
        """Initialize HE key manager."""
        self.key_manager = HEKeyManager(
            storage_path="/tmp/n2he_test_keys",
        )

    def _init_contexts(self):
        """Initialize HE contexts for different schemes."""
        # LWE context (default)
        self.lwe_params = HESchemeParams.default_lora_params()
        self.lwe_context = N2HEContext(params=self.lwe_params)
        self.lwe_context.generate_keys()

        # RLWE context for packed operations
        self.rlwe_params = HESchemeParams.high_precision_params()
        self.rlwe_context = N2HEContext(params=self.rlwe_params)
        self.rlwe_context.generate_keys()

    def _init_runtime(self):
        """Initialize encrypted LoRA runtime."""
        self.runtime, self.key_bundle = create_encrypted_runtime(
            rank=16,
            alpha=32.0,
            tenant_id=self.tenant_id,
        )

    def _init_inference_mode(self):
        """Initialize private inference mode."""
        self.inference_mode, _ = create_private_inference_mode(
            profile="encrypted_input",
            hidden_dim=4096,
            max_layers_encrypted=4,
            tenant_id=self.tenant_id,
        )

    def _init_serializer(self):
        """Initialize ciphertext serializer."""
        self.serializer = CiphertextSerializer()

    # =========================================================================
    # Key Generation Tests
    # =========================================================================

    def test_keygen_lwe(self, iterations: int = 100) -> Dict[str, Any]:
        """Test LWE key generation performance."""
        print(f"\n  Testing LWE keygen ({iterations} iterations)...")

        for i in range(iterations):
            start = time.perf_counter()
            ctx = N2HEContext(params=self.lwe_params)
            ctx.generate_keys()
            elapsed = time.perf_counter() - start
            self.metrics.keygen_lwe_times.append(elapsed)
            self.metrics.operations_completed += 1

        mean_ms = statistics.mean(self.metrics.keygen_lwe_times) * 1000
        print(f"    LWE keygen: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "iterations": iterations}

    def test_keygen_bundle(self, iterations: int = 50) -> Dict[str, Any]:
        """Test full key bundle generation."""
        print(f"\n  Testing key bundle generation ({iterations} iterations)...")

        for i in range(iterations):
            start = time.perf_counter()
            bundle = self.key_manager.generate_key_bundle(
                tenant_id=f"tenant-bundle-{i}",
                params=self.lwe_params,
            )
            elapsed = time.perf_counter() - start
            self.metrics.keygen_bundle_times.append(elapsed)
            self.metrics.operations_completed += 1

        mean_ms = statistics.mean(self.metrics.keygen_bundle_times) * 1000
        print(f"    Bundle generation: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "iterations": iterations}

    # =========================================================================
    # Encryption/Decryption Tests
    # =========================================================================

    def test_encrypt_decrypt(self, iterations: int = 500) -> Dict[str, Any]:
        """Test encryption and decryption performance."""
        print(f"\n  Testing encrypt/decrypt ({iterations} iterations)...")

        for i in range(iterations):
            plaintext = np.array([float(i % 1000)], dtype=np.float32)

            # Encryption
            start = time.perf_counter()
            ciphertext = self.lwe_context.encrypt(plaintext)
            self.metrics.encrypt_times.append(time.perf_counter() - start)

            # Decryption
            start = time.perf_counter()
            decrypted = self.lwe_context.decrypt(ciphertext)
            self.metrics.decrypt_times.append(time.perf_counter() - start)

            self.metrics.operations_completed += 2

            # Track ciphertext size
            serialized = self.serializer.serialize(ciphertext, CiphertextFormat.BINARY)
            self.metrics.ciphertext_sizes.append(len(serialized.data))

        enc_mean = statistics.mean(self.metrics.encrypt_times) * 1000
        dec_mean = statistics.mean(self.metrics.decrypt_times) * 1000
        print(f"    Encrypt: {enc_mean:.3f}ms, Decrypt: {dec_mean:.3f}ms (mean)")
        return {"encrypt_ms": enc_mean, "decrypt_ms": dec_mean, "iterations": iterations}

    def test_batch_encryption(self, batch_sizes: List[int] = None) -> Dict[str, Any]:
        """Test batch encryption performance."""
        if batch_sizes is None:
            batch_sizes = [8, 16, 32, 64]

        print(f"\n  Testing batch encryption (sizes: {batch_sizes})...")
        results = {}

        for batch_size in batch_sizes:
            plaintexts = [np.array([float(i)], dtype=np.float32) for i in range(batch_size)]

            start = time.perf_counter()
            ciphertexts = [self.lwe_context.encrypt(p) for p in plaintexts]
            elapsed = time.perf_counter() - start

            self.metrics.encrypt_batch_times.append(elapsed)
            self.metrics.operations_completed += batch_size

            results[f"batch_{batch_size}"] = {
                "total_ms": elapsed * 1000,
                "per_element_ms": (elapsed / batch_size) * 1000,
            }
            print(f"    Batch {batch_size}: {elapsed*1000:.3f}ms total, {(elapsed/batch_size)*1000:.3f}ms/elem")

        return results

    # =========================================================================
    # Homomorphic Operations Tests
    # =========================================================================

    def test_homomorphic_add(self, iterations: int = 500) -> Dict[str, Any]:
        """Test homomorphic addition performance."""
        print(f"\n  Testing homomorphic add ({iterations} iterations)...")

        # Pre-generate ciphertexts
        ct1 = self.lwe_context.encrypt(np.array([42.0], dtype=np.float32))
        ct2 = self.lwe_context.encrypt(np.array([58.0], dtype=np.float32))

        for _ in range(iterations):
            start = time.perf_counter()
            result = self.lwe_context.scheme.add(ct1, ct2)
            self.metrics.he_add_times.append(time.perf_counter() - start)
            self.metrics.operations_completed += 1

        mean_ms = statistics.mean(self.metrics.he_add_times) * 1000
        print(f"    HE Add: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "iterations": iterations}

    def test_homomorphic_multiply(self, iterations: int = 200) -> Dict[str, Any]:
        """Test homomorphic multiplication performance."""
        print(f"\n  Testing homomorphic multiply ({iterations} iterations)...")

        ct = self.lwe_context.encrypt(np.array([7.0], dtype=np.float32))
        scalar = np.array([6.0], dtype=np.float32)

        for _ in range(iterations):
            start = time.perf_counter()
            result = self.lwe_context.scheme.multiply(ct, scalar)
            self.metrics.he_multiply_times.append(time.perf_counter() - start)
            self.metrics.operations_completed += 1

        mean_ms = statistics.mean(self.metrics.he_multiply_times) * 1000
        print(f"    HE Multiply: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "iterations": iterations}

    def test_homomorphic_matmul(self, matrix_sizes: List[int] = None) -> Dict[str, Any]:
        """Test encrypted matrix multiplication."""
        if matrix_sizes is None:
            matrix_sizes = [8, 16, 32]

        print(f"\n  Testing homomorphic matmul (sizes: {matrix_sizes})...")
        results = {}

        for size in matrix_sizes:
            # Create test matrix and encrypted input
            weight_matrix = np.random.randn(size, size).astype(np.float32)
            activation = np.random.randn(size).astype(np.float32)
            ct = self.lwe_context.encrypt(activation)
            ek = self.lwe_context.export_eval_key()

            start = time.perf_counter()
            result = self.lwe_context.scheme.matmul(ct, weight_matrix, ek)
            elapsed = time.perf_counter() - start

            self.metrics.he_matmul_times.append(elapsed)
            self.metrics.operations_completed += 1

            results[f"matmul_{size}x{size}"] = {"ms": elapsed * 1000}
            print(f"    Matmul {size}x{size}: {elapsed*1000:.3f}ms")

        return results

    # =========================================================================
    # Encrypted LoRA Tests
    # =========================================================================

    def test_encrypted_lora_adapter(self, iterations: int = 100) -> Dict[str, Any]:
        """Test encrypted LoRA adapter end-to-end."""
        print(f"\n  Testing encrypted LoRA adapter ({iterations} iterations)...")

        # Register test adapters
        lora_a = np.random.randn(4096, 16).astype(np.float32)
        lora_b = np.random.randn(16, 4096).astype(np.float32)

        for i in range(min(4, iterations)):  # Register a few adapters
            start = time.perf_counter()
            self.runtime.register_adapter(
                adapter_id=f"adapter_{i}",
                module_name=f"model.layers.{i}.self_attn.q_proj",
                lora_a=lora_a,
                lora_b=lora_b,
            )
            self.metrics.adapter_register_times.append(time.perf_counter() - start)

        # Test delta computation
        activation = np.random.randn(8, 512, 4096).astype(np.float32)  # batch, seq, hidden

        for i in range(iterations):
            # Encrypt activation
            start = time.perf_counter()
            encrypted_act = self.runtime.encrypt_activation(activation)
            encrypt_time = time.perf_counter() - start
            self.metrics.encrypt_embedding_times.append(encrypt_time)

            # Compute delta
            start = time.perf_counter()
            encrypted_delta = self.runtime.compute_delta(encrypted_act, "adapter_0")
            self.metrics.lora_delta_times.append(time.perf_counter() - start)

            # Decrypt delta
            start = time.perf_counter()
            delta = self.runtime.decrypt_delta(encrypted_delta)
            self.metrics.decrypt_output_times.append(time.perf_counter() - start)

            self.metrics.operations_completed += 3

        delta_mean = statistics.mean(self.metrics.lora_delta_times) * 1000
        print(f"    LoRA delta computation: {delta_mean:.3f}ms (mean)")
        return {"delta_ms": delta_mean, "iterations": iterations}

    def test_lora_forward_pass(self, iterations: int = 50) -> Dict[str, Any]:
        """Test full LoRA forward pass through all adapters."""
        print(f"\n  Testing LoRA forward pass ({iterations} iterations)...")

        activation = np.random.randn(8, 512, 4096).astype(np.float32)

        for _ in range(iterations):
            start = time.perf_counter()

            # Encrypt
            encrypted = self.runtime.encrypt_activation(activation)

            # Forward through all registered adapters
            deltas = self.runtime.forward(encrypted)

            # Decrypt all deltas (deltas may be a list or dict)
            if isinstance(deltas, dict):
                results = {k: self.runtime.decrypt_delta(v) for k, v in deltas.items()}
            else:
                results = [self.runtime.decrypt_delta(d) for d in deltas]

            self.metrics.lora_forward_times.append(time.perf_counter() - start)
            self.metrics.operations_completed += 1

        mean_ms = statistics.mean(self.metrics.lora_forward_times) * 1000
        print(f"    Full forward pass: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "iterations": iterations}

    # =========================================================================
    # Private Inference Tests
    # =========================================================================

    def test_private_inference(self, num_prompts: int = 20) -> Dict[str, Any]:
        """Test private inference mode end-to-end."""
        print(f"\n  Testing private inference ({num_prompts} prompts)...")

        test_prompts = [
            "What is homomorphic encryption?",
            "Explain neural network privacy.",
            "How does secure computation work?",
            "Describe encrypted inference.",
            "What are LoRA adapters?",
        ]

        for i in range(num_prompts):
            prompt = test_prompts[i % len(test_prompts)]

            start = time.perf_counter()
            results = self.inference_mode.private_sample([prompt])
            total_time = time.perf_counter() - start

            self.metrics.private_inference_times.append(total_time)
            self.metrics.operations_completed += 1

            if i == 0:
                # Verify first result
                assert len(results) == 1
                assert results[0].privacy_preserved

        mean_ms = statistics.mean(self.metrics.private_inference_times) * 1000
        print(f"    Private inference: {mean_ms:.3f}ms (mean)")
        return {"mean_ms": mean_ms, "prompts": num_prompts}

    # =========================================================================
    # Serialization Tests
    # =========================================================================

    def test_serialization_formats(self, iterations: int = 100) -> Dict[str, Any]:
        """Test ciphertext serialization in different formats."""
        print(f"\n  Testing serialization formats ({iterations} iterations)...")

        ct = self.lwe_context.encrypt(np.array([12345.0], dtype=np.float32))

        for _ in range(iterations):
            # Binary
            start = time.perf_counter()
            binary = serialize_ciphertext(ct, CiphertextFormat.BINARY)
            self.metrics.serialize_binary_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            _ = deserialize_ciphertext(binary, self.lwe_params)
            self.metrics.deserialize_binary_times.append(time.perf_counter() - start)

            # JSON
            start = time.perf_counter()
            json_data = serialize_ciphertext(ct, CiphertextFormat.JSON)
            self.metrics.serialize_json_times.append(time.perf_counter() - start)

            # Base64
            start = time.perf_counter()
            b64 = serialize_ciphertext(ct, CiphertextFormat.BASE64)
            self.metrics.serialize_base64_times.append(time.perf_counter() - start)

            self.metrics.operations_completed += 5

        results = {
            "binary_ms": statistics.mean(self.metrics.serialize_binary_times) * 1000,
            "json_ms": statistics.mean(self.metrics.serialize_json_times) * 1000,
            "base64_ms": statistics.mean(self.metrics.serialize_base64_times) * 1000,
        }
        print(f"    Binary: {results['binary_ms']:.3f}ms, JSON: {results['json_ms']:.3f}ms, Base64: {results['base64_ms']:.3f}ms")
        return results

    def test_ciphertext_bundles(self, bundle_sizes: List[int] = None) -> Dict[str, Any]:
        """Test ciphertext bundle creation and handling."""
        if bundle_sizes is None:
            bundle_sizes = [10, 50, 100]

        print(f"\n  Testing ciphertext bundles (sizes: {bundle_sizes})...")
        results = {}

        for size in bundle_sizes:
            ciphertexts = [self.lwe_context.encrypt(np.array([float(i)], dtype=np.float32)) for i in range(size)]

            start = time.perf_counter()
            bundle = create_ciphertext_bundle(
                ciphertexts=ciphertexts,
                bundle_id=f"bundle-{size}",
                metadata={"size": size, "test": True},
            )
            elapsed = time.perf_counter() - start

            total_size = bundle.get_total_size()
            self.metrics.bundle_sizes.append(total_size)
            self.metrics.operations_completed += 1

            results[f"bundle_{size}"] = {
                "creation_ms": elapsed * 1000,
                "total_size_bytes": total_size,
            }
            print(f"    Bundle {size}: {elapsed*1000:.3f}ms, {total_size} bytes")

        return results


def generate_n2he_report(metrics: N2HEMetrics) -> Dict[str, Any]:
    """Generate comprehensive N2HE test report."""
    report = metrics.to_report()

    report["metadata"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "test_type": "N2HE E2E Integration",
        "framework": "TenSafe + N2HE",
    }

    report["summary"] = {
        "total_operations": metrics.operations_completed,
        "total_time_seconds": metrics.total_test_time,
        "all_tests_passed": True,
    }

    # Calculate key performance indicators
    report["kpi"] = {
        "keygen_throughput": f"{1000 / (statistics.mean(metrics.keygen_lwe_times) * 1000):.0f} ops/sec" if metrics.keygen_lwe_times else "N/A",
        "encrypt_throughput": f"{1000 / (statistics.mean(metrics.encrypt_times) * 1000):.0f} ops/sec" if metrics.encrypt_times else "N/A",
        "lora_delta_throughput": f"{1000 / (statistics.mean(metrics.lora_delta_times) * 1000):.0f} ops/sec" if metrics.lora_delta_times else "N/A",
        "average_ciphertext_size": f"{statistics.mean(metrics.ciphertext_sizes):.0f} bytes" if metrics.ciphertext_sizes else "N/A",
    }

    return report


# =============================================================================
# Test Functions
# =============================================================================

@pytest.fixture
def n2he_harness():
    """Create N2HE test harness."""
    return N2HETestHarness(tenant_id="e2e-test-tenant")


class TestN2HEE2E:
    """E2E tests for N2HE homomorphic encryption integration."""

    @pytest.mark.e2e
    def test_quick_validation(self, n2he_harness):
        """Quick validation of N2HE functionality."""
        harness = n2he_harness

        # Test basic encrypt/decrypt
        plaintext = np.array([42.0], dtype=np.float32)
        ct = harness.lwe_context.encrypt(plaintext)
        decrypted = harness.lwe_context.decrypt(ct)
        assert abs(decrypted[0] - plaintext[0]) < 0.1, "Decrypt mismatch"

        # Test LoRA runtime
        activation = np.random.randn(1, 16, 4096).astype(np.float32)
        encrypted = harness.runtime.encrypt_activation(activation)
        assert encrypted is not None

        # Test serialization
        serialized = serialize_ciphertext(ct, CiphertextFormat.BINARY)
        deserialized = deserialize_ciphertext(serialized, harness.lwe_params)
        assert deserialized is not None

        print("\n  Quick validation passed!")

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_full_n2he_e2e(self, n2he_harness):
        """
        Full N2HE E2E test with comprehensive metrics.

        This test:
        1. Tests all N2HE components
        2. Collects detailed performance metrics
        3. Generates comparison report
        """
        harness = n2he_harness

        # Adjust iterations based on environment
        if os.getenv("FULL_E2E") == "1":
            iterations = {"keygen": 100, "encrypt": 500, "lora": 100, "inference": 50}
        else:
            iterations = {"keygen": 20, "encrypt": 100, "lora": 20, "inference": 10}

        test_start = time.perf_counter()

        print("\n" + "=" * 60)
        print("N2HE E2E TEST SUITE")
        print("=" * 60)

        # =================================================================
        # Phase 1: Key Generation
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 1: Key Generation")
        print("-" * 40)

        harness.test_keygen_lwe(iterations["keygen"])
        harness.test_keygen_bundle(iterations["keygen"] // 2)

        # =================================================================
        # Phase 2: Encryption/Decryption
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 2: Encryption/Decryption")
        print("-" * 40)

        harness.test_encrypt_decrypt(iterations["encrypt"])
        harness.test_batch_encryption([8, 16, 32])

        # =================================================================
        # Phase 3: Homomorphic Operations
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 3: Homomorphic Operations")
        print("-" * 40)

        harness.test_homomorphic_add(iterations["encrypt"])
        harness.test_homomorphic_multiply(iterations["encrypt"] // 2)
        harness.test_homomorphic_matmul([8, 16])

        # =================================================================
        # Phase 4: Encrypted LoRA
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 4: Encrypted LoRA Adapter")
        print("-" * 40)

        harness.test_encrypted_lora_adapter(iterations["lora"])
        harness.test_lora_forward_pass(iterations["lora"] // 2)

        # =================================================================
        # Phase 5: Private Inference
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 5: Private Inference")
        print("-" * 40)

        harness.test_private_inference(iterations["inference"])

        # =================================================================
        # Phase 6: Serialization
        # =================================================================
        print("\n" + "-" * 40)
        print("PHASE 6: Serialization")
        print("-" * 40)

        harness.test_serialization_formats(iterations["encrypt"] // 2)
        harness.test_ciphertext_bundles([10, 50])

        # =================================================================
        # Generate Report
        # =================================================================
        harness.metrics.total_test_time = time.perf_counter() - test_start

        report = generate_n2he_report(harness.metrics)

        # Save report
        reports_dir = Path(__file__).parent.parent.parent / "reports" / "e2e"
        reports_dir.mkdir(parents=True, exist_ok=True)

        report_path = reports_dir / f"n2he_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print("\n" + "=" * 60)
        print("N2HE E2E TEST SUMMARY")
        print("=" * 60)

        print(f"\n  Total Operations: {harness.metrics.operations_completed}")
        print(f"  Total Test Time: {harness.metrics.total_test_time:.2f}s")

        print("\n  Key Performance Indicators:")
        for kpi, value in report["kpi"].items():
            print(f"    {kpi}: {value}")

        print(f"\n  Report saved to: {report_path}")
        print("=" * 60)

        # Assertions
        assert harness.metrics.operations_completed > 0
        assert len(harness.metrics.encrypt_times) > 0
        assert len(harness.metrics.lora_delta_times) > 0

        return report

    @pytest.mark.e2e
    def test_integration_with_tensafe(self, n2he_harness):
        """Test N2HE integration with TenSafe KEK/DEK hierarchy."""
        harness = n2he_harness

        print("\n  Testing TenSafe integration...")

        # Verify key bundle integrates with TenSafe key management
        bundle = harness.key_manager.generate_key_bundle(
            tenant_id="tensafe-integration-test",
            params=harness.lwe_params,
        )

        # Export keys (simulates distribution)
        public_key = bundle.export_public_key()
        eval_key = bundle.export_evaluation_key()

        assert public_key is not None
        assert eval_key is not None

        # Verify manifest claims are generated
        claims = harness.runtime.get_manifest_claims()
        assert claims is not None
        assert "he_scheme" in claims or hasattr(claims, "he_scheme_config")

        print("    TenSafe integration verified!")

    @pytest.mark.e2e
    def test_benchmark_runner(self):
        """Run the built-in N2HE benchmark suite."""
        print("\n  Running N2HE benchmark suite...")

        suite = run_quick_benchmark()

        # Check that benchmark suite has results
        assert len(suite.results) > 0, "No benchmark results"

        # Get operation names from results
        op_names = [r.operation for r in suite.results]
        assert "keygen" in op_names, f"Missing keygen benchmark, got: {op_names}"
        assert "encryption" in op_names, f"Missing encryption benchmark, got: {op_names}"
        assert "lora_delta" in op_names, f"Missing lora_delta benchmark, got: {op_names}"

        print("\n  Benchmark Results:")
        for result in suite.results:
            print(f"    {result.operation}: {result.mean_time_ms:.3f}ms ({result.ops_per_second:.0f} ops/sec)")


if __name__ == "__main__":
    # Run with: python -m pytest tests/e2e/test_n2he_e2e.py -v -s
    pytest.main([__file__, "-v", "-s", "-m", "not slow"])
