# HE-LoRA Microkernel Architecture

## Overview

The HE-LoRA Microkernel is a production-grade system for secure LoRA inference
using homomorphic encryption (HE). It treats LoRA adapters as compilable secure
microkernels, inspired by MOAI's rotation-minimization philosophy.

## Design Principles

### 1. Two-Plane Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    Base Model Plane (FP16)                       │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │  Embed  │ -> │  Attn   │ -> │   FFN   │ -> │  Output │     │
│   └─────────┘    └────┬────┘    └─────────┘    └─────────┘     │
│                       │                                          │
│                       │ Activations                              │
├───────────────────────┼─────────────────────────────────────────┤
│                       v                                          │
│              ┌────────────────┐                                  │
│              │  HE-LoRA Delta │                                  │
│              │   (Encrypted)  │                                  │
│              └────────────────┘                                  │
│                                                                  │
│                   Secure Adapter Plane (CKKS)                    │
└─────────────────────────────────────────────────────────────────┘
```

The architecture separates inference into two planes:
- **Base Model Plane**: Standard FP16 inference (fast, unencrypted)
- **Secure Adapter Plane**: HE-LoRA correction (encrypted)

### 2. Ct×Pt Regime

HE-LoRA operates in the **Ciphertext × Plaintext** regime:
- Activations are encrypted (ciphertext)
- LoRA weights remain plaintext (FP16)
- No ciphertext-ciphertext multiplication (avoids expensive relinearization)

### 3. Every-Token Execution

```
for token in generation:
    base_output = base_model(token)

    # HE-LoRA on EVERY token - NO SKIPPING
    encrypted_activations = encrypt(activations)
    he_delta = execute_he_lora(encrypted_activations)
    delta = decrypt(he_delta)

    output = base_output + delta
```

## System Components

### Compiler

```
he_lora_microkernel/compiler/
├── ckks_params.py      # CKKS parameter profiles
├── packer.py           # SIMD packing strategies
├── lora_ir.py          # Intermediate representation
├── scheduler.py        # Rotation-minimal scheduling
├── cost_model.py       # Cost estimation and budgets
└── emit_artifacts.py   # Artifact serialization
```

The compiler transforms LoRA adapters into optimized execution schedules:

1. **CKKS Parameters**: Selects encryption parameters (FAST/SAFE profiles)
2. **Packing**: Computes SIMD slot layout for batch-parallel execution
3. **IR Generation**: Creates intermediate representation of LoRA computation
4. **Scheduling**: Generates rotation-minimal execution plan
5. **Cost Modeling**: Estimates and validates rotation/keyswitch budgets

### Runtime

```
he_lora_microkernel/runtime/
├── executor.py         # Main execution engine
├── batching.py         # Dynamic batch management
└── telemetry.py        # Performance monitoring
```

The runtime executes compiled schedules:

1. **Executor**: Runs HE-LoRA for each token
2. **Batching**: Handles batch size changes with recompilation
3. **Telemetry**: Tracks rotation counts for CI enforcement

### Backend

```
he_lora_microkernel/backend/
├── gpu_ckks_backend.py    # Python backend interface
├── gpu_ckks_backend.h     # C++ header
└── backend_adapter.cc     # Backend implementations
```

The backend provides GPU-accelerated CKKS operations:
- Encryption/Decryption
- Ct×Pt multiplication
- Rotation (with key switching)
- Rescale/Modswitch

## Data Flow

```
                    ┌──────────────────────────────────────────────────┐
                    │                  COMPILATION                      │
                    │                                                    │
    Config ─────────┤   LoRA Config  ──> IR ──> Schedule ──> Artifacts │
    Weights ────────┤   A, B, alpha  ──> Pack ──> Plaintexts           │
                    └──────────────────────────────────────────────────┘
                                           │
                                           v
                    ┌──────────────────────────────────────────────────┐
                    │                   RUNTIME                         │
                    │                                                    │
    Activations ────┤   Pack ──> Encrypt ──> Execute ──> Decrypt ──> Δ │
                    │              │           │                        │
                    │              └───────────┴──> Telemetry ──> CI    │
                    └──────────────────────────────────────────────────┘
```

## LoRA Computation

### Mathematical Form

```
Δy = (α/r) · A · B · x

Where:
  x: Input activations     (batch_size, hidden_size)
  B: Down-projection       (rank, hidden_size)
  A: Up-projection         (hidden_size, rank)
  α: Scaling factor
  r: Rank
```

### HE Execution

```
1. Pack activations into SIMD slots
2. Encrypt packed activations
3. For each block:
   a. Ct×Pt: encrypted_x × B_block
   b. Rescale
4. Accumulate blocks (tree reduction)
5. Ct×Pt: intermediate × A_block
6. Rescale
7. Decrypt
8. Unpack
```

## MOAI-Inspired Optimizations

### Column-Packed Matrix Multiplication (CPMM)

Traditional HE matrix-vector multiplication requires O(n) rotations.
MOAI's CPMM achieves **rotation-free** Ct×Pt by column packing:

```
Standard:    Σᵢ rotate(ct, i) × pt[i]     # O(n) rotations

CPMM:        ct × column_packed(pt)        # 0 rotations
```

### Batch-First Packing

SIMD slots process entire batches in parallel:

```
Slots: [b0_c0, b1_c0, b2_c0, ..., b0_c1, b1_c1, b2_c1, ...]
        └─────── channel 0 ───────┘ └─────── channel 1 ─────┘
```

### Blocked Packing

For large hidden_size × batch_size > slot_count:

```
Block 0: channels [0, block_size)
Block 1: channels [block_size, 2×block_size)
...
```

Cross-block accumulation uses tree reduction with log₂(blocks) rotations.

## CI Enforcement

### Rotation Budget

```python
DEFAULT_BUDGETS = {
    'rotations_per_token': 16,   # R_max
    'keyswitches_per_token': 16, # K_max
    'rescales_per_token': 8,     # S_max
}
```

CI fails if actual costs exceed budgets.

### Determinism

The compiler produces deterministic schedules:
- Same inputs → identical schedule hash
- Enables regression detection

## Cloud Portability

The microkernel is designed for general cloud GPUs:
- No H100-specific code
- Works on A100, L40, etc.
- Backend-agnostic interface

## Prohibitions

The microkernel explicitly prohibits:
- ❌ Quantization (QLoRA, INT8/INT4)
- ❌ Integer HE (TFHE)
- ❌ CPU-only HE
- ❌ Bootstrapping
- ❌ Token skipping
