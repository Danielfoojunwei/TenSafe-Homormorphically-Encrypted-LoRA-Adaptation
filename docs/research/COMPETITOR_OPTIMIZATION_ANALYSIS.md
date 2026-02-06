# Competitor Optimization Analysis: What TenSafe Can Still Adopt

> **Date:** 2026-02-06
> **Scope:** Deep-dive into 10+ competitor/peer implementations of HE-for-ML at scale, mapped against TenSafe's current architecture to identify concrete optimization gaps.

---

## Executive Summary

TenSafe already leads in one critical niche — **production-ready HE-LoRA with MOAI zero-rotation column packing** — but the broader HE-ML ecosystem has advanced rapidly (2024–2025). This analysis identifies **18 concrete optimization opportunities** across six categories, prioritized by impact and implementation complexity.

| Priority | Category | Count | Estimated Impact |
|----------|----------|-------|------------------|
| P0 (Critical) | GPU Acceleration | 3 | 40–2000x on HE primitives |
| P0 (Critical) | Memory Optimization | 3 | 2–100x memory reduction |
| P1 (High) | Bootstrapping & Noise | 3 | Unlocks deeper circuits, 2–55x bootstrap throughput |
| P1 (High) | Compiler Infrastructure | 3 | Automatic optimization, cross-scheme interop |
| P2 (Medium) | Distributed Scale-Up | 3 | Linear GPU scaling, multi-node HE |
| P2 (Medium) | LoRA-Specific HE | 3 | Selective encryption, training-time HE |

---

## 1. GPU-Accelerated HE Primitives [P0 — Critical Gap]

### 1.1 Current State in TenSafe

TenSafe has a `gpu_ckks_backend.py` stub, but the actual HE computation path is CPU-bound. The MOAI column-packing eliminates rotations, but the underlying polynomial arithmetic (NTT, key-switching, rescaling) still runs on CPU.

### 1.2 What Competitors Are Doing

**Cerium (CMU/NVIDIA/UT Austin, Dec 2025):**
- First GPU bootstrapping under 10ms (7.5ms), enabling encrypted Llama3-8B inference in 134 seconds.
- **Sparse polynomial representation** reduces memory from TB-scale to manageable levels (100x reduction).
- Compiler-driven multi-GPU parallelization with communication-aware scheduling.

**CAT Framework (March 2025):**
- Open-source GPU CKKS/BFV/BGV achieving **2173x speedup** over prior GPU work on NVIDIA 4090.
- Dual GPU pool architecture with kernel fusion and segmentation.

**HEonGPU (2025):**
- Full GPU execution with zero CPU-GPU transfers. Multi-stream architecture.
- **380x speedup** over SEAL for key-switching operations alone.

**HEngine (ACM TACO 2025):**
- **Warp-shuffle NTT optimization**: Uses CUDA warp shuffles instead of shared memory for NTT and inverse-CRT, eliminating bank conflicts.
- **Kernel fusion**: Fuses NTT + inner product into a single kernel launch, halving memory round-trips.
- CUDA streams for overlapping computation with memory transfers on large batches.

**WarpDrive (HPCA 2025):**
- First system to exploit **NVIDIA Tensor Cores** for FHE — the same hardware designed for matrix multiply-accumulate in deep learning can accelerate polynomial arithmetic in CKKS.

### 1.3 Optimization Opportunities for TenSafe

**Opportunity 1.3a — NTT Kernel Fusion on GPU**
```
Impact:    100–380x speedup on polynomial multiply and key-switching
Effort:    High (4–6 weeks for core NTT+keyswitch kernels)
Approach:  Port the NTT and key-switching inner loops to CUDA with warp-shuffle
           optimization (HEngine approach). Fuse NTT + inner product into single
           kernel. The MOAI column-packed layout is already SIMD-friendly, which
           maps well to GPU warps.
Dependency: Requires CUDA toolkit; compatible with existing PyTorch GPU context.
```

**Opportunity 1.3b — Tensor Core Exploitation for CKKS MatMul**
```
Impact:    Additional 2–5x on top of GPU NTT (WarpDrive showed significant gains)
Effort:    High (requires custom WMMA intrinsics for modular arithmetic)
Approach:  MOAI column-packed matrices are already in a layout suitable for
           matrix-multiply-accumulate. Map the encrypted matmul to Tensor Core
           WMMA operations with modular reduction interleaved.
Dependency: Requires Ampere+ GPU (SM80+). A100/H100 preferred.
```

**Opportunity 1.3c — Zero-Copy GPU HE Pipeline**
```
Impact:    Eliminates CPU-GPU transfer overhead entirely
Effort:    Medium (2–3 weeks, following HEonGPU's architecture)
Approach:  Keep ciphertexts in GPU memory throughout the entire encrypt → matmul
           → rescale → decrypt pipeline. Only move plaintexts to/from CPU.
           Use CUDA unified memory or explicit pinned memory for the
           encrypt/decrypt boundaries.
Current:   TenSafe's vLLM hooks already run on GPU; the HE computation is the
           only component that bounces back to CPU.
```

---

## 2. Memory Optimization [P0 — Critical Gap]

### 2.1 Current State in TenSafe

CKKS ciphertexts with N=16384 and 4–5 primes occupy ~1–2 MB each. For a model with 32 LoRA-adapted layers (each needing A and B matrices encrypted), this means 64+ ciphertexts (~64–128 MB) per forward pass. The expansion factor (ciphertext vs. plaintext) is roughly 50–100x.

### 2.2 What Competitors Are Doing

**Cerium — Sparse Polynomial Representation:**
- Ciphertexts are stored in sparse format (only non-zero coefficients). For typical HE workloads, this yields **100x memory reduction** because most polynomial coefficients are zero after operations.

**IBM HE-PEx — Ciphertext Pruning:**
- Prunes entire ciphertexts (not individual weights) by using permutation-based expansion. Achieves 33–65% fewer ciphertexts with <2.5% accuracy degradation.
- Compared to standard pruning adapted for HE, HE-PEx uses 70% fewer ciphertexts on average.

**OpenFHE — Selective Key Serialization:**
- Serializes only the rotation keys actually needed (vs. the full rotation key set). For LoRA with MOAI (which needs 0 rotations), this means the rotation key set can be **empty**, saving significant memory.

**Rotation Keyset Design (AICompS 2024):**
- Optimal rotation keyset selection achieves **11.29x memory reduction** and 2.55x speedup by computing the minimal set of rotation keys required for a given computation.

### 2.3 Optimization Opportunities for TenSafe

**Opportunity 2.3a — Eliminate Rotation Keys Entirely**
```
Impact:    Save ~500 MB–2 GB of key material per HE context
Effort:    Low (1–2 days; mostly configuration)
Approach:  Since MOAI column packing achieves 0 rotations, TenSafe should NOT
           generate rotation keys at all. Currently the CKKS parameter setup
           likely generates a full rotation key set by default. Generate only
           the relinearization key and the encryption/decryption key pair.
Verify:    Check ckks_params.py and the backend initialization to confirm no
           rotation keys are being generated unnecessarily.
```

**Opportunity 2.3b — Ciphertext Pruning for LoRA (HE-PEx Adaptation)**
```
Impact:    33–65% fewer ciphertexts, 17–35% memory reduction
Effort:    Medium (2–3 weeks)
Approach:  Adapt IBM's HE-PEx for LoRA matrices. Since LoRA matrices are
           low-rank (r << hidden_size), many ciphertext slots contain
           near-zero values. Prune ciphertexts whose contribution to the
           final output is below a threshold, using permutation-based
           expansion to reconstruct approximately.
           This is especially effective for larger ranks (r=32+) where
           some singular values dominate.
```

**Opportunity 2.3c — Sparse Ciphertext Representation**
```
Impact:    Up to 100x memory reduction for intermediate computations
Effort:    High (4–6 weeks; requires changes to polynomial representation)
Approach:  Adopt Cerium's sparse polynomial representation for CKKS
           ciphertexts. After rescaling, many coefficients become zero or
           near-zero. Store only non-zero coefficients with index arrays.
           Convert back to dense only for NTT operations.
Caveat:    Most beneficial for large-scale workloads (many ciphertexts
           simultaneously in memory). Less impactful for single-layer LoRA.
```

---

## 3. Bootstrapping & Noise Budget [P1 — Unlocks Deeper Circuits]

### 3.1 Current State in TenSafe

TenSafe explicitly **does not support bootstrapping** (documented architectural decision). All computations must fit within the CKKS modulus chain (max_depth=2 for FAST, max_depth=3 for SAFE). This limits TenSafe to linear LoRA (2 multiplications: A @ x, then B @ intermediate).

### 3.2 What Competitors Are Doing

**OpenFHE — Functional Bootstrapping via CKKS (Crypto 2025):**
- Evaluates arbitrary functions during CKKS bootstrapping using trigonometric Hermite interpolation.
- 0.72ms amortized time for 8-bit LUT evaluation — **1000x faster** than DM/CGGI approach.
- Enables non-linear activations (GeLU, SiLU) within the encrypted domain.

**HElib — NTT-Decomposed Bootstrapping (ASIACRYPT 2024):**
- Decomposes linear transformations in bootstrapping into NTT-like sub-transformations.
- **2.4x–55.1x improvement** in bootstrapping throughput (4096–32768 slots).

**Cerium — GPU Bootstrapping:**
- 7.5ms per bootstrap on GPU. Fast enough for practical use in inference pipelines.

**Sorted Bootstrapping (EUROCRYPT 2025):**
- Reorders ciphertexts to batch those with similar noise levels, reducing wasted noise budget.
- **1.75–8.28x speedup** over naive bootstrapping.

**Mean Compensation (TCHES 2025):**
- Reduces TFHE noise variance by 2x through statistical compensation of the rounding error.

### 3.3 Optimization Opportunities for TenSafe

**Opportunity 3.3a — Lightweight CKKS Bootstrapping for Gated LoRA**
```
Impact:    Enables GeLU/SiLU activations in encrypted LoRA (currently
           impossible), unlocking gated LoRA variants (LLaMA gate_proj)
Effort:    High (6–8 weeks; integrate OpenFHE's functional bootstrapping)
Approach:  Use OpenFHE's Crypto 2025 technique: evaluate the activation
           function DURING the bootstrap step. At 0.72ms amortized, this
           adds <1ms per non-linear activation — acceptable for TenSafe's
           current 5ms per-layer budget.
           Start with a CKKS→FHEW scheme switch for the activation, then
           switch back. OpenFHE already supports this hybrid path.
Trade-off: Increases modulus chain requirements (need larger N or more primes).
           May need to move to N=32768 for bootstrapping support.
```

**Opportunity 3.3b — Noise-Aware Scheduling**
```
Impact:    10–30% more computation within existing noise budget
Effort:    Low-Medium (1–2 weeks)
Approach:  The cost_model.py enforces hard rotation/rescale budgets but does
           not track actual noise levels. Implement real-time noise estimation
           (following OpenFHE's noise model) to:
           1. Reorder operations to minimize noise growth
           2. Fuse operations that share intermediate results
           3. Skip unnecessary rescaling when noise headroom permits
Current:   The compiler/scheduler.py has schedule optimization but it
           optimizes for latency, not noise budget.
```

**Opportunity 3.3c — Sorted Ciphertext Batching**
```
Impact:    1.75–8.28x speedup IF bootstrapping is adopted
Effort:    Low (few days, algorithmic change only)
Approach:  If/when bootstrapping is added (Opportunity 3.3a), sort
           ciphertexts by noise level before batching bootstrap operations.
           Those with similar noise levels can share parameters, reducing
           per-bootstrap overhead.
```

---

## 4. Compiler Infrastructure [P1 — Systematic Optimization]

### 4.1 Current State in TenSafe

TenSafe has a custom compiler pipeline (`he_lora_microkernel/compiler/`) with:
- `lora_ir.py`: Simple IR (Encrypt → MatMul → Rescale → Decrypt)
- `cost_model.py`: Budget enforcement (rotation/rescale limits)
- `packer.py`: MOAI column packing
- `scheduler.py`: Schedule optimization

This is functional but limited compared to production HE compilers.

### 4.2 What Competitors Are Doing

**Google HEIR (MLIR-based):**
- Multi-level IR built on LLVM's MLIR framework.
- Enables automatic optimization passes: constant folding, dead code elimination, operation fusion, automatic scheme selection, automatic parameter selection.
- Cross-scheme interoperability (CKKS ↔ TFHE lowering).
- Hardware target abstraction (CPU, GPU, FPGA, ASIC).

**Microsoft EVA:**
- Automatic rescaling insertion and relinearization placement.
- Waterline rescaling: globally optimal rescaling schedule that minimizes noise.
- Automatic encryption parameter selection based on computation graph analysis.

**Orion (ASPLOS 2025 Best Paper):**
- **Single-shot multiplexed packing** for arbitrary convolutions (multiplicative depth of 1 per convolution).
- Automatic bootstrap placement and scale management.
- Compiles PyTorch models directly to FHE programs.

### 4.3 Optimization Opportunities for TenSafe

**Opportunity 4.3a — MLIR-Based HE Compiler Backend**
```
Impact:    Enables systematic optimization passes, hardware retargeting
Effort:    Very High (8–12 weeks for MVP)
Approach:  Rewrite lora_ir.py as an MLIR dialect (following HEIR's approach).
           This enables:
           1. Standard compiler optimizations (CSE, DCE, constant folding)
           2. Automatic operation fusion (mul+rescale already done manually)
           3. Hardware-specific lowering (GPU kernels, future ASIC targets)
           4. Cross-scheme lowering (CKKS → TFHE for activations)
Trade-off: Major architectural change. Consider as a v5.0 initiative.
Alternative: Integrate with HEIR directly as a backend rather than
             rebuilding from scratch.
```

**Opportunity 4.3b — Automatic Rescaling Waterline (EVA-style)**
```
Impact:    Optimal noise budget utilization, fewer manual tuning parameters
Effort:    Medium (2–3 weeks)
Approach:  Implement EVA's waterline rescaling algorithm in scheduler.py.
           Instead of rescaling after every multiplication, analyze the
           full computation DAG and insert rescalings at the globally
           optimal points. For LoRA's simple graph (2 matmuls), the
           benefit is modest, but this becomes critical if gated LoRA
           or deeper computations are added.
Current:   TenSafe fuses mul+rescale, which is locally optimal but not
           globally optimal for multi-operation sequences.
```

**Opportunity 4.3c — PyTorch-to-HE Compilation (Orion-style)**
```
Impact:    Automates the HE-LoRA hook generation, reduces manual effort
Effort:    High (4–6 weeks)
Approach:  Instead of manually defining HE hooks in vllm/hooks.py, trace
           the LoRA forward pass with torch.fx, extract the computation
           graph, and automatically compile it to HE operations.
           Orion showed this is feasible at scale (ResNet-20, YOLO-v1).
           For LoRA (which is just 2 linear layers), this should be
           significantly simpler.
Benefit:   New LoRA variants (DoRA, VeRA, AdaLoRA) would automatically
           get HE support without manual hook implementation.
```

---

## 5. Distributed HE Scale-Up [P2 — Next-Level Scale]

### 5.1 Current State in TenSafe

TenSafe scales to 16 workers via Ray Train with pairwise masking for secure aggregation. Efficiency is 84% at 8 workers. HE operations are per-worker (not distributed).

### 5.2 What Competitors Are Doing

**ArctyrEX (NVIDIA Research):**
- Distributes a single HE computation across arbitrary numbers of GPUs with linear scaling.
- Compiler generates parallelization plans that partition the polynomial ring across GPUs.

**Cerium:**
- Multi-GPU FHE with communication-aware scheduling. The compiler minimizes inter-GPU data movement by analyzing the computation graph's data dependencies.

**FHE4DMM (IEEE TPDS 2025):**
- Distributed encrypted matrix multiplication across cloud clusters. Partitions encrypted matrices across nodes, performs local HE operations, and aggregates results with minimal communication.

**IBM Federated + HE:**
- HE-encrypted model updates in federated learning. The aggregator never sees plaintext gradients. This is architecturally similar to TenSafe's pairwise masking but uses HE instead of symmetric masking.

### 5.3 Optimization Opportunities for TenSafe

**Opportunity 5.3a — Multi-GPU HE for Large Models**
```
Impact:    Enables HE-LoRA for models that don't fit on a single GPU
           (e.g., 70B parameter models with h=8192)
Effort:    High (6–8 weeks)
Approach:  Partition the LoRA computation across GPUs along the hidden
           dimension. GPU 0 handles columns 0–4095, GPU 1 handles
           columns 4096–8191. Each GPU performs local encrypted matmul,
           then results are aggregated (plaintext addition, since the
           base model output is already distributed via tensor parallelism).
Synergy:   vLLM already supports tensor parallelism. The HE-LoRA hooks
           should be made tensor-parallelism-aware.
```

**Opportunity 5.3b — Pipelined HE Computation**
```
Impact:    Overlap encryption of layer N+1 with computation of layer N
Effort:    Medium (2–3 weeks)
Approach:  Use CUDA streams (or CPU thread pools) to pipeline:
           Stream 1: Encrypt activations for layer N+1
           Stream 2: Compute encrypted matmul for layer N
           Stream 3: Decrypt results from layer N-1
           This hides encryption/decryption latency behind computation.
Current:   TenSafe processes layers sequentially.
```

**Opportunity 5.3c — HE-Native Secure Aggregation**
```
Impact:    Stronger security guarantees than pairwise masking
           (HE aggregation is post-quantum secure)
Effort:    Medium (3–4 weeks)
Approach:  Replace pairwise masking protocol with HE-based aggregation
           (IBM's approach). Workers encrypt their LoRA gradients under
           a shared HE key, send ciphertexts to aggregator, which
           homomorphically averages them and returns the encrypted
           average. Each worker decrypts locally.
Trade-off: Higher communication cost (ciphertexts are larger than masked
           plaintexts). Only worthwhile if post-quantum secure aggregation
           is required.
```

---

## 6. LoRA-Specific HE Optimizations [P2 — Unique Niche]

### 6.1 Current State in TenSafe

TenSafe encrypts the entire LoRA computation (both A and B matrices, all slots). The MOAI packing is universal — it doesn't exploit LoRA-specific structure beyond the low-rank property.

### 6.2 What Competitors Are Doing

**SHE-LoRA (May 2025) — Selective Encryption:**
- Only encrypts a **subset** of LoRA parameters (the most important ones by magnitude/gradient).
- Uses Order-Preserving Encryption (OPE) to hide WHICH parameters are encrypted, preventing targeted attacks.
- Reduces HE overhead proportionally to the fraction of parameters encrypted.

**PrivTuner (Oct 2024) — HE-LoRA Training Protocol:**
- Clients send FHE-encrypted data to server.
- Server creates per-device LoRA adapters and runs forward passes on encrypted data.
- Clients decrypt locally, compute loss, and send gradients back (in plaintext or encrypted).
- Joint optimization of energy consumption and privacy level.

**Private LoRA Fine-tuning (May 2025):**
- Interactive protocol with orchestrator (data owner) and worker nodes.
- Workers perform encrypted forward/backward passes via HE.
- Key finding: **"cleartext and homomorphic loss curves overlap almost perfectly"** — HE noise does not materially perturb LoRA gradient computation.
- Fine-tuned Llama-3.2-1B with HE-compatible quantization on GPU.

**ReBoot Architecture (2025):**
- Minimizes multiplicative depth for training by using local error signals instead of full backpropagation.
- **8.83x latency reduction** compared to standard encrypted backprop.
- Enables encrypted training within shallow HE circuits (depth 2–3).

### 6.3 Optimization Opportunities for TenSafe

**Opportunity 6.3a — Selective Parameter Encryption (SHE-LoRA)**
```
Impact:    2–5x reduction in HE computation for LoRA inference
Effort:    Medium (2–3 weeks)
Approach:  Analyze LoRA weight magnitudes after training. Encrypt only
           the top-k% of parameters (by absolute value or gradient
           sensitivity). Use OPE to hide the selection.
           For typical LoRA matrices (rank 16, hidden 4096), many
           singular values are near-zero. Encrypting only the top 50%
           of slots halves HE computation with minimal accuracy loss.
Caveat:    Requires careful analysis of which parameters are "important"
           — this is model-specific.
```

**Opportunity 6.3b — Encrypted LoRA Training (ReBoot-style)**
```
Impact:    Enables privacy-preserving LoRA fine-tuning, not just inference
Effort:    High (6–8 weeks)
Approach:  Implement ReBoot's local error signal approach:
           1. Forward pass: encrypted (existing TenSafe pipeline)
           2. Loss computation: at the final layer only (depth 1)
           3. Backward pass: use local error signals instead of full
              backprop. Each layer computes its own gradient from the
              local prediction error, requiring only depth-1 HE operations.
           This fits within TenSafe's FAST profile (max_depth=2).
Synergy:   Combines with DP-SGD: the encrypted gradients can have DP
           noise added in the encrypted domain.
```

**Opportunity 6.3c — Quantization-Aware Encrypted LoRA**
```
Impact:    2–4x reduction in ciphertext size and computation
Effort:    Medium (2–3 weeks)
Approach:  Following Zama's quantization-aware approach and the May 2025
           "Private LoRA" paper's HE-compatible quantization:
           1. Quantize LoRA weights to 4–8 bits during training
           2. Use smaller CKKS scale parameters (20–25 bits instead of 40)
           3. Reduce the modulus chain accordingly
           This directly reduces ciphertext size and polynomial
           multiplication cost.
Caveat:    Quantization + HE noise both reduce precision. Need careful
           analysis of combined error bounds. The May 2025 paper showed
           this is feasible for Llama-3.2-1B.
```

---

## 7. Cross-Cutting: CKKS-TFHE Hybrid for Activations

### What Competitors Are Doing

OpenFHE, Zama, and the HEIR compiler all support **scheme switching** between CKKS (efficient for linear operations) and TFHE (efficient for non-linear operations via programmable bootstrapping).

The PoPETs 2025 paper demonstrates that FHEW/TFHE ciphertexts are significantly smaller than CKKS ciphertexts, and functional bootstrapping in TFHE can evaluate arbitrary lookup tables in <1ms.

### Opportunity for TenSafe

TenSafe already has a `hybrid_compiler/` directory for CKKS-TFHE hybrid compilation. This should be prioritized because:

1. **LLaMA's gated MLP** uses SiLU activation between gate_proj and up_proj. Currently, TenSafe can only encrypt the linear projections (gate_proj, up_proj, down_proj) but NOT the SiLU gate.
2. With CKKS→TFHE switching, the SiLU can be evaluated as a lookup table during TFHE bootstrapping, then switched back to CKKS for the next linear operation.
3. This enables **full MLP encryption**, not just attention-layer encryption.

```
Estimated overhead: <2ms per activation (0.72ms for CKKS functional bootstrap
per OpenFHE benchmarks + scheme switching overhead)
```

---

## 8. Priority Roadmap

### Phase 1: Quick Wins (1–4 weeks)

| # | Opportunity | Impact | Effort |
|---|-----------|--------|--------|
| 1 | **2.3a** Eliminate unnecessary rotation keys | Save 500MB–2GB memory | 1–2 days |
| 2 | **3.3b** Noise-aware scheduling | 10–30% more computation headroom | 1–2 weeks |
| 3 | **4.3b** Waterline rescaling | Optimal noise utilization | 2–3 weeks |

### Phase 2: High-Impact GPU Work (4–12 weeks)

| # | Opportunity | Impact | Effort |
|---|-----------|--------|--------|
| 4 | **1.3c** Zero-copy GPU HE pipeline | Eliminate CPU-GPU transfers | 2–3 weeks |
| 5 | **1.3a** GPU NTT kernel fusion | 100–380x on HE primitives | 4–6 weeks |
| 6 | **5.3b** Pipelined HE computation | Hide encrypt/decrypt latency | 2–3 weeks |

### Phase 3: Advanced Optimizations (8–16 weeks)

| # | Opportunity | Impact | Effort |
|---|-----------|--------|--------|
| 7 | **6.3a** Selective parameter encryption | 2–5x less HE compute | 2–3 weeks |
| 8 | **6.3c** Quantization-aware encrypted LoRA | 2–4x smaller ciphertexts | 2–3 weeks |
| 9 | **2.3b** HE-PEx ciphertext pruning | 33–65% fewer ciphertexts | 2–3 weeks |
| 10 | **3.3a** CKKS functional bootstrapping | Unlocks non-linear encrypted ops | 6–8 weeks |

### Phase 4: Architectural Evolution (12+ weeks)

| # | Opportunity | Impact | Effort |
|---|-----------|--------|--------|
| 11 | **4.3a** MLIR-based compiler (HEIR integration) | Systematic optimization | 8–12 weeks |
| 12 | **4.3c** PyTorch-to-HE compilation | Auto HE for new LoRA variants | 4–6 weeks |
| 13 | **1.3b** Tensor Core exploitation | Additional 2–5x on GPU | 4–6 weeks |
| 14 | **6.3b** Encrypted LoRA training (ReBoot) | Privacy-preserving fine-tuning | 6–8 weeks |
| 15 | **5.3a** Multi-GPU HE for large models | Scale to 70B+ models | 6–8 weeks |

---

## 9. Competitive Positioning Summary

| Capability | TenSafe (Current) | Best Competitor | Gap |
|------------|-------------------|-----------------|-----|
| Rotation elimination | **0 rotations (MOAI)** | Orion: depth-1 packing | TenSafe leads |
| GPU HE acceleration | CPU-bound | Cerium: 7.5ms bootstrap, CAT: 2173x | Critical gap |
| Memory efficiency | Standard CKKS | Cerium: 100x sparse, HE-PEx: 65% prune | Significant gap |
| Bootstrapping | None | OpenFHE: 0.72ms functional BS | Limits depth |
| Compiler sophistication | Custom IR | HEIR: MLIR-based, EVA: auto-params | Moderate gap |
| Distributed HE | Per-worker only | ArctyrEX: linear GPU scaling | Moderate gap |
| HE-LoRA production | **Production-ready** | PrivTuner/SHE-LoRA: research only | TenSafe leads |
| LoRA variant support | Standard LoRA encrypted | None | TenSafe leads |
| Encrypted training | DP-SGD (plaintext grads) | ReBoot: encrypted backprop | Future opportunity |
| Post-quantum crypto | Dilithium3 + Kyber768 | None at this level | TenSafe leads |

**TenSafe's moat**: The only production-ready system combining HE-LoRA + DP-SGD + PQC + vLLM integration. Competitors are either research prototypes or general-purpose HE libraries without LoRA specialization.

**Biggest risks**: GPU acceleration gap (competitors are 100–2000x faster on raw HE ops) and the bootstrapping limitation (blocks non-linear encrypted operations and encrypted training).

---

## References

1. Cerium: Multi-GPU FHE for LLMs — arXiv:2512.11269 (Dec 2025)
2. CAT: GPU FHE Framework — arXiv:2503.22227 (March 2025)
3. HEonGPU: Full GPU FHE — ePrint 2025/124 (2025)
4. HEngine: CUDA CKKS — ACM TACO 2025
5. WarpDrive: Tensor Cores for FHE — HPCA 2025
6. Orion: PyTorch-to-FHE — ASPLOS 2025 Best Paper
7. HEIR: MLIR-based FHE Compiler — heir.dev
8. EVA: Microsoft HE Compiler — GitHub microsoft/EVA
9. OpenFHE Functional Bootstrapping — Crypto 2025
10. HElib NTT Bootstrapping — ASIACRYPT 2024
11. IBM HE-PEx — ESORICS 2023
12. SHE-LoRA: Selective HE for LoRA — arXiv:2505.21051 (May 2025)
13. PrivTuner: HE-LoRA Training — arXiv:2410.00433 (Oct 2024)
14. Private LoRA Fine-tuning with HE — arXiv:2505.07329 (May 2025)
15. ReBoot: Efficient Encrypted Training — 2025
16. Zama Concrete ML Benchmarks — zama.org (July 2024)
17. NVIDIA FLARE HE Acceleration — NVIDIA Developer Blog (Dec 2024)
18. ArctyrEX: Multi-GPU FHE — NVIDIA Research
19. FHE4DMM: Distributed Encrypted MatMul — IEEE TPDS 2025
20. Cross-Platform HE Benchmarking — ePrint 2025/473
