# Research Paper Skeletons: Top-Tier Conference Ready

This document outlines the structural skeletons for the three core contributions of TenSafe. Each skeleton is tailored to the expectations of specific top-tier venues (USENIX Security, OSDI, NeurIPS/ICLR).

---

## Paper 1: ZeRo-MOAI (Zero-KeyGen Extension for Modular Architecture)
**Target Venue**: USENIX Security Symposium / IEEE S&P (Oakland)
**Theme**: Cryptographic Engineering & Systems Security

### 1. Title
**ZeRo-MOAI: System-Level Elimination of Rotation Keys for Private PEFT via Modular Extension**

### 2. Abstract
*   **Context**: Building upon the **MOAI** framework by Wang et al. (NTU DTC), which optimizes Transformer modules for secure inference by reducing or eliminating kernel rotations.
*   **The Problem**: While baseline MOAI optimizes *computational* overhead, general-purpose transformer inference still requires a complete evaluation key set (Galois Keys) for flexible module handling, exceeding the memory of many edge devices.
*   **Our Extension**: We introduce **ZeRo-MOAI**, a modular extension specialized for **Low-Rank Adaptation (LoRA)**. 
*   **Insight**: By exploiting the extreme rank-deficiency of LoRA ($r \ll N$), we enforce a "Zero-KeyGen" contract. We prove that the entire LoRA-extension of the transformer can be executed with **absolute zero rotations** and thus **zero Galois Keys**.
*   **Results**: Reduces initialization time from seconds (KeyGen) to milliseconds. Reduces key-storage overhead from ~1GB to **0 MB** for the adapter pathway. This enables the first "Keyless" private fine-tuning deployment for mobile LLMs.

### 3. Introduction
*   **The Baseline**: Acknowledge MOAI's [Wang et al.] breakthrough in modularizing secure inference kernels.
*   **The Deployment Gap**: Identify that "Non-Interactive" compute still requires "Interactive" key setup (large key transmissions).
*   **The ZeRo Expansion**: Frame ZeRo-MOAI as the "last mile" optimization for PEFT. It shifts the paradigm from "Minimizing Rotations" to "Eliminating Key Sets."
*   **Key Contributions**:
    1.  **Rank-Deficient Specialization**: A mathematical protocol for mapping LoRA rank $r$ to SIMD slots that guarantees Zero Rotations.
    2.  **Keyless Deployment Protocol**: Eliminating the evaluation key generation step entirely.
    3.  **Cross-Library Integration**: Implementation as a microkernel compatible with the broader MOAI scheduler.

### 4. Comparison with Baseline MOAI
| Metric | Original MOAI (Baseline) | ZeRo-MOAI (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Focus** | General Transformer Inference | **Specialized LoRA (PEFT)** | Domain Depth |
| **Rotations** | Optimized/Reduced | **Absolute Zero** | Algorithmic Guarantee |
| **Galois Key Size** | Required (1GB+) | **0 MB (Eliminated)** | 100% reduction |
| **Setup Time** | KeyGen Dependent | **Instantaneous** | Eliminates setup delay |
| **Edge Feasibility** | Computationally Viable | **Deployment Ready** | Low-bandwidth tolerant |

### 4. Background & Threat Model
*   **HE Primer**: CKKS, SIMD Slots, Rotations.
*   **LoRA Primer**: $W + BA$.
*   **Threat Model**: 
    *   Honest-but-curious server.
    *   **Goal**: Protect the Adapter Weights (Server IP) and User Inputs (if fully encrypted, though usually Adapter Privacy is the focus here).
    *   **Scope**: Semi-honest.

### 5. System Design (The Core)
*   **Client-Side Pre-Processing**:
    *   *The "Replication" Trick*: Explain `encrypt_moai_packed` where $x_j$ is replicated $r$ times.
    *   *Cost Analysis*: Plaintext operations are free.
*   **Server-Side CPMM**:
    *   *Column-Major Encoding*: Storing $W$ as interleaved columns.
    *   *The "Zero-Rotation" Dot Product*: Mathematical proof that $\sum (x_{rep} * W_{col})$ yields the correct result without shifting slots.
*   **Parameter Selection**:
    *   *Depth Analysis*: Demonstrating that $L=4$ is sufficient for the Depth-2 circuit, negating the need for Bootstrapping.

### 6. Security Analysis & Tradeoffs
*   **The Hardware-Math Frontier**: Justifying the use of TEEs for "Scheme-Switching" to maintain Zero-KeyGen.
*   **Mitigating TEE Side-Channels**: Constant-time enclave bridges and logic-padding.
*   **Model Extraction Guards**: Output-stage DP noise to prevent inversion.
*   **Comparison of Attack Surfaces**: Pure Hybrid (Mathematical noise) vs. TEE-Assisted (Timing/Hardware vulns).
*   **Adversarial Integrity**: Client-side noise checks (Risk Management deep dive).

### 7. Evaluation
*   **Setup**: Llama-3-8B LoRA adapters (Rank 8-64). $N=16384$.
*   **Baselines**: 
    *   Standard Diagonal-Method CKKS (Halevi-Shoup).
    *   SHE-LoRA (State of the Art).
*   **Metrics**:
    *   Key Size (GB vs MB).
    *   Inference Latency (ms).
    *   Throughput (Tokens/sec).

### 8. Related Work
*   *Contrast with SHE-LoRA*: Full encryption vs. Selective.
*   *Contrast with Iron*: Specialization for Rank-Deficient matrices.

---

## Paper 2: Speculative Batching
**Target Venue**: OSDI / SOSP / EuroSys
**Theme**: Systems, High-Performance Computing, OS

### 1. Title
**High-Throughput Encrypted Inference via Base-Model Speculation and SIMD Saturation**

### 2. Abstract
*   **Problem**: HE is "Throughput-Oriented" (SIMD) but LLM Inference is "Latency-Sensitive" (Autoregressive). Processing one token at a time wastes 99.9% of the crypto-processor's capacity.
*   **Insight**: The "Base Model" (Llama-3) is a highly accurate predictor of the "Fine-Tuned Model" (Llama-3 + LoRA).
*   **Solution**: We propose a hybrid TEE-HE architecture. The Base Model (in TEE) acts as a high-accuracy Speculator, generating $K$ tokens. The HE Engine packs verification of all $K$ tokens into a single SIMD Ciphertext.
*   **Result**: Amortizes the heavy Crypto-Cost by a factor of $K$. Achieves single-user latency comparable to plaintext batched inference.

### 3. Introduction
*   **The "Single-User" Gap**: Existing HE batching works for 100 users, but is slow for 1 user.
*   **The Bandwidth-Compute Inversion**: In Plaintext, Move Memory = Cost. In HE, Compute Arithmetic = Cost.
*   **Contribution**:
    1.  **Base-Model Speculation**: Using the plaintext model as the draft.
    2.  **Flattened SIMD Packing**: Mapping $[K, Hidden]$ tensors to $[N/2]$ slots.
    3.  **TEE Integration**: Solves the privacy leak of the base model.

### 4. Comparison with Baseline Speculative Decoding
| Metric | Leviathan et al. (Baseline) | Speculative Batching (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Domain** | Pure Plaintext | **Hybrid TEE-HE** | Security Upgrade |
| **Trust Model** | None | **Hardware/Math Split** | Cryptographic Privacy |
| **Batching** | Memory Bandwidth Bound | **SIMD Compute Bound** | HE Amortization |
| **Verify Cost** | $O(K)$ Plaintext | **$O(1)$ HE SIMD Pass** | $K \times$ throughut |
| **User Latency** | Baseline (Standard) | **Amortized HE (Ours)** | 80%+ Reduction |

### 4. Anatomy of the Engine
*   **The TEE Base (Plaintext)**:
    *   Role of **PagedAttention**: Continuous batching as the "Feeder".
    *   Speculative Decoding Loop.
*   **The HE Verifier (Encrypted)**:
    *   `HELoRAHook` Interception.
    *   Vector Flattening Strategy.
*   **The Hook Protocol**: Synchronizing the TEE stream with the HE co-processor.

### 5. Risk & Safety (Deep Dive)
*   **The Plaintext Leak**: Acknowledging the Base Model sees the prompt.
*   **The Fix**: Mandatory TEE (H100 Confidential Computing).
*   **Side Channels**: Fixed-Block communication to hide $K$ (Traffic Analysis).

### 6. Evaluation
*   **Microbenchmarks**: Crypto-Access Time vs. Compute Time.
*   **Macrobenchmarks**: End-to-End latency for generating 128 tokens.
*   **A/B Test**: vs. Non-Speculative HE (Standard execution).
*   **Acceptance Rates**: Empirical data showing Base Model acceptance >95% for LoRA tasks.

---

## Paper 3: Non-Linear Adapter Hot-Swapping
**Target Venue**: NeurIPS / ICLR / MLSys
**Theme**: Machine Learning Systems, Model Serving

### 1. Title
**Scalable Multi-Tenant Serving of Gated Encrypted Adapters via Hybrid Compilation**

### 2. Abstract
*   **Context**: Linear HE-LoRA (e.g., MOAI-ZeRo) established "Zero KeyGen" for secure adapters but is limited to simple linear shifts.
*   **The Problem**: Non-linear activations (ReLU, GELU) are essential for reasoning but require expensive "Scheme Switching" keys and bootstrapping in pure HE systems.
*   **Our Solution**: We propose the **Asymmetric Client-Aided Bridge**. We introduce a hybrid protocol where the heavy linear compute remains on the "Keyless" HE server, while the sparse non-linear decision is offloaded to the **Client** during the auto-regressive token round-trip.
*   **The Result**: The first **Zero-KeyGen**, non-linear encrypted adapter system that requires **Zero Evaluation Keys** on the server. It achieves the expressivity of full fine-tuning without the 1GB+ key overhead of traditional hybrid HE.

### 3. Introduction
*   **The Expressivity Wall**: Linear LoRA cannot learn XOR.
*   **The Deployment Wall**: Storing 1GB of evaluation keys per user kills multi-tenant scalability.
*   **Contribution**:
    1.  **GateLink Protocol**: An asymmetric bridge that hides scheme-switching in network IO.
    2.  **Asymmetric Gate Pass**: Reducing bridge overhead to bytes ($1 \times r$).
    3.  **Zero-KeyGen Seving**: Proving that the server can remain stateless and keyless even for non-linear gated models.

### 4. Comparison with Baseline Serving (S-LoRA/LoRAX)
| Metric | S-LoRA / vLLM (Baseline) | TenSafe Hot-Swapping (Ours) | Improvement |
| :--- | :--- | :--- | :--- |
| **Security** | None (Plaintext) | **HE-Encrypted Weights** | Privacy First |
| **Expressivity** | Non-Linear (Native) | **Non-Linear (Hybrid HE)** | Parity reached |
| **Multi-Tenancy** | Unified Paging | **Zero-Copy Crypto-Registry** | Encrypted Scaling |
| **Context Swap** | Paged KV Cache | **Encrypted Weight Slabs** | Crypto Context Switching |
| **Density** | High (Plaintext) | **High (Shared Crypto-Engine)** | GPU Efficiency |

### 4. Implementation: The "Crypto-OS"
*   **Compiler Design**:
    *   The `GatedLoRACompiler` pipeline.
    *   Bridge Operations (`CKKSQuantizeToInt`, `TFHEToCKKS`).
*   **The Registry**:
    *   Slab Allocation for Adapter Pages (Fragmentation Countermeasure).
    *   Constant-Time Lookup (Side-Channel Countermeasure).
*   **The Protocol**:
    *   Token $i \rightarrow$ Logit + Encrypted Gate Result $\rightarrow$ Client.
    *   Client $\rightarrow$ Gate Bit $\rightarrow$ Server $\rightarrow$ Next Token / Layer.

### 5. Expressivity Analysis (The ML Meat)
*   **Mathematical Proof**: Showing $W + \sigma(BAx)$ is a Universal Approximator.
*   **Task Benchmarks**: 
    *   Logic/Reasoning Tasks (GSM8K) where Linear LoRA fails.
    *   Accuracy boost from Gated Adapters.

### 6. Systems Evaluation
*   **Concurrency**: Latency vs. Number of Active Adapters (1 to 1000).
*   **Memory Footprint**: Shared Base Model + Swapped Adapters vs. Naive Replication.
*   **Cold-Start**: Time to first token for a "Cold" adapter (microsecond swapping).
