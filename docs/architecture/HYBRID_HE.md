# Hybrid CKKS-TFHE Architecture for Non-Linear Adapters

**Version:** 1.0.0
**Status:** Active
**Last Updated:** 2026-02-05

## Overview

TenSafe's Hybrid HE system enables privacy-preserving inference with non-linear adapter operations (e.g., gated LoRA) by combining:

- **CKKS**: Efficient approximate arithmetic for high-dimensional linear operations (LoRA matmuls)
- **TFHE**: Exact discrete arithmetic for non-linear control flow (gates, thresholds)

This document defines the canonical architecture, APIs, trust model, and integration points.

## Supported Adapter Types

### v1: Gated LoRA

The initial release supports Gated LoRA, which adds a discrete gate to control adapter application:

```
y = Wx + g(x) * B(Ax)

Where:
- Wx          = Base model output (CKKS)
- B(Ax)       = LoRA delta computed entirely in CKKS
- g(x)        = Discrete gate computed in TFHE via LUT
- g(x) in {0, 1} for step gate, or {-1, 0, 1} for sign gate
```

**Computational Phases:**
1. `CKKS_LORA_DELTA`: u = A @ x; delta = B @ u (CKKS matmuls)
2. `CKKS_GATE_PRE`: z = w_g^T @ x + b_g (CKKS dot product)
3. `BRIDGE_TO_TFHE`: Quantize z and convert to TFHE (interactive)
4. `TFHE_GATE_EVAL`: g = LUT(z_q) with programmable bootstrap
5. `BRIDGE_TO_CKKS`: Convert g back to CKKS encoding
6. `CKKS_APPLY_GATE`: gated_delta = g * delta (CKKS scalar mult)
7. `CKKS_FINAL_ADD`: y = Wx + gated_delta (CKKS add)

### Future: Additional Non-Linear Adapters

Reserved for future extension:
- Mixture-of-LoRA with discrete routing
- Threshold-activated adapters
- Multi-gate adapters

## Trust Model (v1: Interactive Bridge)

### Choice: Interactive Scheme Conversion

For v1, we implement **Interactive Bridge** (Option A). This provides:
- Clear security guarantees (server never sees plaintext)
- Faster time-to-market
- Compatibility with existing key management

**Non-interactive scheme switching** (Option B) is deferred to v2.

### Security Properties

1. **Confidentiality**: Server never sees plaintext values during inference
   - CKKS ciphertexts: Client holds secret key (sk_ckks)
   - TFHE ciphertexts: Client holds secret key (sk_tfhe)
   - Bridge operations require client participation

2. **Integrity**: All adapter computations are verifiable
   - TGSP signature on adapter weights
   - Deterministic gate computations (LUT is public)
   - Audit trail for all operations

3. **Key Ownership**:
   ```
   Client owns:
     - sk_ckks: CKKS secret key
     - sk_tfhe: TFHE secret key
     - Derivation keys for session binding

   Server holds:
     - pk_ckks: CKKS public key
     - evk_ckks: CKKS evaluation keys (rotations, relinearization)
     - pk_tfhe: TFHE public key (for re-encryption if needed)
     - bsk_tfhe: TFHE bootstrapping key (public)
     - Encrypted adapter weights (in CKKS)
   ```

4. **Threat Model**:
   - Honest-but-curious server: May try to learn from ciphertexts
   - Malicious adapters: Mitigated by TGSP signature verification
   - Side channels: Out of scope for v1

### Bridge Protocol (Interactive)

```
                   CLIENT                                 SERVER
                      |                                      |
                      |  <-- [CKKS ciphertext: z]            |  z = gate pre-activation
                      |                                      |
  Decrypt(sk_ckks, z) |                                      |
          = z_plain   |                                      |
                      |                                      |
  Quantize(z_plain)   |                                      |
          = z_q       |                                      |
                      |                                      |
  Encrypt(sk_tfhe,    |                                      |
          z_q)        |                                      |
          = z_tfhe    |                                      |
                      |                                      |
                      |  --> [TFHE ciphertext: z_tfhe]       |
                      |                                      |  Apply LUT via bootstrap
                      |  <-- [TFHE ciphertext: g_tfhe]       |  g = step(z_q)
                      |                                      |
  Decrypt(sk_tfhe,    |                                      |
          g_tfhe)     |                                      |
          = g_plain   |                                      |
                      |                                      |
  Dequantize(g_plain) |                                      |
          = g_real    |                                      |
                      |                                      |
  Encrypt(sk_ckks,    |                                      |
          g_real)     |                                      |
          = g_ckks    |                                      |
                      |                                      |
                      |  --> [CKKS ciphertext: g_ckks]       |
                      |                                      |  Continue with gated delta
```

## API Specification

### HybridHEBackend

The unified backend for hybrid CKKS-TFHE operations:

```python
class HybridHEBackend:
    """
    Production backend for hybrid CKKS-TFHE operations.

    Responsibilities:
    - CKKS linear operations (reuses existing GPU CKKS backend)
    - TFHE LUT evaluation via programmable bootstrapping
    - Scheme bridging (interactive or library-based)
    - Telemetry collection
    """

    def __init__(
        self,
        ckks_backend: CKKSBackendInterface,
        bridge_service: SchemeBridgeServiceClient,
        lut_library: LUTLibrary,
        config: HybridHEConfig,
    ) -> None: ...

    # CKKS Operations (delegated)
    def ckks_matvec(
        self,
        ct_x: CKKSCiphertext,
        pt_matrix: np.ndarray,
    ) -> CKKSCiphertext: ...

    def ckks_matmul(
        self,
        ct_x: CKKSCiphertext,
        pt_A: np.ndarray,
        pt_B: np.ndarray,
    ) -> CKKSCiphertext: ...

    def ckks_scalar_mul(
        self,
        ct_x: CKKSCiphertext,
        ct_scalar: CKKSCiphertext,
    ) -> CKKSCiphertext: ...

    def ckks_add(
        self,
        ct_a: CKKSCiphertext,
        ct_b: CKKSCiphertext,
    ) -> CKKSCiphertext: ...

    # TFHE Operations
    def tfhe_lut_apply(
        self,
        ct_tfhe: TFHECiphertext,
        lut_id: str,
    ) -> TFHECiphertext:
        """
        Apply LUT via programmable bootstrapping.

        Args:
            ct_tfhe: TFHE ciphertext (scalar or small vector)
            lut_id: Registered LUT identifier (e.g., "step", "sign")

        Returns:
            TFHE ciphertext with LUT applied and noise refreshed
        """
        ...

    def tfhe_lut_apply_custom(
        self,
        ct_tfhe: TFHECiphertext,
        lut_data: List[int],
    ) -> TFHECiphertext: ...

    # Bridge Operations (Interactive v1)
    def bridge_ckks_to_tfhe(
        self,
        ct_ckks: CKKSCiphertext,
        request_id: str,
        quantization_bits: int = 8,
    ) -> TFHECiphertext:
        """
        Convert CKKS ciphertext to TFHE via interactive bridge.

        Requires client participation for decrypt/re-encrypt.
        Uses SchemeBridgeService for the round-trip.
        """
        ...

    def bridge_tfhe_to_ckks(
        self,
        ct_tfhe: TFHECiphertext,
        request_id: str,
    ) -> CKKSCiphertext:
        """
        Convert TFHE ciphertext back to CKKS via interactive bridge.
        """
        ...

    # Telemetry
    def get_operation_stats(self) -> HybridOperationStats: ...
```

### NonLinearAdapter Interface

```python
class NonLinearAdapter(Protocol):
    """
    Protocol for non-linear adapter implementations.

    All non-linear adapters must implement this interface to be
    usable in the hybrid HE pipeline.
    """

    @property
    def adapter_type(self) -> str:
        """Adapter type identifier (e.g., 'gated_lora')."""
        ...

    @property
    def requires_tfhe(self) -> bool:
        """Whether this adapter requires TFHE operations."""
        ...

    def forward(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        module_name: str,
        adapter_state: AdapterState,
    ) -> Tuple[np.ndarray, AdapterMetrics]:
        """
        Compute adapter output.

        Args:
            x: Input activations [batch, seq, hidden]
            base_output: Base model output Wx
            module_name: Target module (e.g., "q_proj")
            adapter_state: Loaded adapter weights and config

        Returns:
            Tuple of (output, metrics)
        """
        ...

    def forward_encrypted(
        self,
        ct_x: CKKSCiphertext,
        ct_base: CKKSCiphertext,
        module_name: str,
        adapter_state: AdapterState,
        backend: HybridHEBackend,
    ) -> Tuple[CKKSCiphertext, AdapterMetrics]:
        """
        Compute adapter output under encryption.
        """
        ...


class HEGatedLoRAAdapter(NonLinearAdapter):
    """
    Gated LoRA adapter using hybrid CKKS-TFHE encryption.

    Implements: y = Wx + g(x) * B(Ax)
    Where g(x) = LUT(w_g^T x + b_g)
    """

    def __init__(
        self,
        config: GatedLoRAConfig,
        backend: HybridHEBackend,
        lut_type: str = "step",  # "step" or "sign"
    ) -> None: ...

    def set_weights(
        self,
        lora_A: np.ndarray,
        lora_B: np.ndarray,
        w_gate: np.ndarray,
        b_gate: Optional[np.ndarray] = None,
    ) -> None: ...

    def forward(
        self,
        x: np.ndarray,
        base_output: np.ndarray,
        module_name: str,
        adapter_state: AdapterState,
    ) -> Tuple[np.ndarray, AdapterMetrics]: ...

    def forward_encrypted(
        self,
        ct_x: CKKSCiphertext,
        ct_base: CKKSCiphertext,
        module_name: str,
        adapter_state: AdapterState,
        backend: HybridHEBackend,
    ) -> Tuple[CKKSCiphertext, AdapterMetrics]: ...
```

### SchemeBridgeService (gRPC)

```protobuf
// Schema bridge service for interactive CKKS-TFHE conversion
service SchemeBridgeService {
    // Convert CKKS ciphertext to TFHE (client-side operation)
    rpc CKKSDecryptAndEncryptToTFHE(BridgeRequest) returns (BridgeResponse);

    // Convert TFHE ciphertext to CKKS (client-side operation)
    rpc TFHEDecryptAndEncryptToCKKS(BridgeRequest) returns (BridgeResponse);

    // Batched conversion (multiple scalars in one call)
    rpc BatchedBridge(BatchedBridgeRequest) returns (BatchedBridgeResponse);

    // Health check
    rpc HealthCheck(HealthRequest) returns (HealthResponse);
}

message BridgeRequest {
    string request_id = 1;              // Unique request identifier
    string session_id = 2;              // Session for key binding
    string adapter_id = 3;              // Associated adapter
    bytes ciphertext_blob = 4;          // Serialized ciphertext
    BridgeDirection direction = 5;      // CKKS_TO_TFHE or TFHE_TO_CKKS
    QuantizationParams quant_params = 6; // For CKKS->TFHE
}

message BridgeResponse {
    string request_id = 1;
    bytes ciphertext_blob = 2;          // Converted ciphertext
    BridgeMetrics metrics = 3;          // Timing and error info
    ErrorInfo error = 4;                // Error if failed
}

message QuantizationParams {
    int32 bits = 1;                     // Quantization bit width
    double clip_min = 2;                // Clipping range minimum
    double clip_max = 3;                // Clipping range maximum
    bool symmetric = 4;                 // Symmetric quantization
}

message BridgeMetrics {
    double decrypt_time_ms = 1;
    double quantize_time_ms = 2;
    double encrypt_time_ms = 3;
    double quantization_error = 4;
}

enum BridgeDirection {
    CKKS_TO_TFHE = 0;
    TFHE_TO_CKKS = 1;
}
```

## TGSP Manifest Extension

For gated LoRA adapters, the TGSP manifest must include:

```json
{
    "manifest_version": "2.0",
    "adapter_type": "gated_lora",
    "model_name": "llama3-8b",

    "lora_config": {
        "rank": 16,
        "alpha": 32.0,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },

    "gate_config": {
        "gate_type": "step",
        "gate_lut_id": "step",
        "input_bits": 8,
        "output_bits": 1,
        "signed_input": true,
        "signed_output": false,
        "clip_range": [-10.0, 10.0]
    },

    "hybrid_he_config": {
        "scheme": "hybrid_ckks_tfhe",
        "ckks_profile": "FAST",
        "tfhe_profile": "STANDARD",
        "bridge_mode": "interactive"
    },

    "weights": {
        "lora_A": "weights/lora_a.safetensors",
        "lora_B": "weights/lora_b.safetensors",
        "w_gate": "weights/w_gate.safetensors",
        "b_gate": "weights/b_gate.safetensors"
    }
}
```

## Integration Points

### HAS Executor Integration

The HAS executor (`he_lora_microkernel/services/has/executor.py`) is extended:

```python
@dataclass
class AdapterState:
    adapter_id: str
    model_id: str
    rank: int
    alpha: float
    targets: str
    num_layers: int
    loaded_layers: List[int]

    # NEW: Adapter type for hybrid support
    adapter_type: str = "linear_lora"  # "linear_lora" | "gated_lora"

    # NEW: Gate configuration (for gated_lora)
    gate_config: Optional[GateConfig] = None

    # NEW: Hybrid backend (for gated_lora)
    hybrid_backend: Optional[HybridHEBackend] = None
```

### vLLM Hook Integration

The vLLM hooks (`src/tensorguard/backends/vllm/hooks.py`) route based on scheme:

```python
class HELoRAHook:
    def __call__(self, module, input, output):
        if self.config.scheme == "hybrid" and self.adapter_type == "gated_lora":
            # Route to HAS ApplyTokenStep with gating
            return self._apply_gated_lora_via_has(module, input, output)
        else:
            # Standard CKKS-only path
            return self._apply_linear_lora(module, input, output)
```

## Telemetry

### Required Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `hybrid_conversions_total` | Counter | Total CKKS<->TFHE conversions |
| `hybrid_lut_applications_total` | Counter | Total LUT/bootstrap operations |
| `hybrid_bridge_latency_ms` | Histogram | Bridge round-trip latency |
| `hybrid_lut_latency_ms` | Histogram | LUT evaluation latency |
| `hybrid_ckks_latency_ms` | Histogram | CKKS operation latency |
| `hybrid_total_latency_ms` | Histogram | Total per-token latency |
| `hybrid_quantization_error` | Gauge | Current quantization error |
| `hybrid_gate_activation_rate` | Gauge | Fraction of tokens with g=1 |

### Structured Logging

```json
{
    "event": "hybrid_token_step",
    "request_id": "req_123",
    "adapter_id": "adapter_456",
    "layer_idx": 12,
    "module": "q_proj",
    "timings_ms": {
        "ckks_linear": 2.5,
        "bridge_to_tfhe": 15.2,
        "tfhe_lut": 8.1,
        "bridge_to_ckks": 14.8,
        "ckks_apply_gate": 1.2,
        "total": 41.8
    },
    "gate_value": 1.0,
    "quantization_error": 0.0023
}
```

## Error Handling

### Fallback Policy

If hybrid operations fail, the system **MUST NOT** silently fall back to plaintext:

```python
class HybridExecutionError(Exception):
    """Raised when hybrid HE execution fails."""
    pass

class HybridNotAvailableError(Exception):
    """Raised when hybrid mode requested but not available."""
    pass

# In adapter loading:
if adapter_type == "gated_lora" and not hybrid_available:
    raise HybridNotAvailableError(
        f"Adapter '{adapter_id}' requires hybrid CKKS-TFHE mode, "
        f"but hybrid backend is not available. "
        f"Ensure SchemeBridgeService is running."
    )
```

## Constraints and Limitations

### v1 Limitations

1. **Bootstrap Budget**: Maximum 2 bootstraps per layer per token
2. **TFHE Vector Size**: Maximum 16 elements for TFHE operations
3. **Gate Types**: Only `step` and `sign` LUTs in v1
4. **Bridge Latency**: ~30ms round-trip for interactive bridge
5. **Quantization**: 8-bit symmetric quantization for bridge

### Parameter Constraints

| Parameter | Constraint | Rationale |
|-----------|------------|-----------|
| `quantization_bits` | 4-12 | LUT size = 2^bits |
| `clip_range` | [-127, 127] for 8-bit | Avoid overflow |
| `lora_rank` | <= hidden_size/4 | MOAI packing efficiency |
| `gate_bias_range` | [-10, 10] | Quantization precision |

## Migration Guide

### From Linear LoRA to Gated LoRA

1. Update adapter manifest:
   ```diff
   - "adapter_type": "linear_lora"
   + "adapter_type": "gated_lora"
   + "gate_config": { ... }
   ```

2. Add gate weights to TGSP package:
   ```bash
   tgsp build --adapter-type gated_lora \
       --lora-weights ./lora/ \
       --gate-weights ./gate/ \
       --output adapter.tgsp
   ```

3. Deploy with hybrid mode:
   ```bash
   tensafe serve --hybrid-mode \
       --bridge-service grpc://bridge:50051 \
       --adapter adapter.tgsp
   ```

## References

- [MOAI Paper](https://eprint.iacr.org/2025/991): Rotation-minimal CKKS
- [TFHE Paper](https://eprint.iacr.org/2018/421): Programmable bootstrapping
- [TenSafe Architecture](./MODULE_MAP.md): Overall system architecture
- [TGSP Specification](../TGSP_SPEC.md): Secure adapter packaging
