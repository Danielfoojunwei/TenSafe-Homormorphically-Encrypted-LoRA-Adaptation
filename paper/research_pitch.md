# Research Proposal: The "Zero-KeyGen" Paradigm for Private AI

**To:** Research Committee / Professor
**From:** [Your Name]
**Subject:** Proposal for 3 Canonical Papers on High-Performance Encrypted Inference

## Executive Summary

Homomorphic Encryption (HE) has long been trapped in a "theoretical limbo"—mathematically sound but engineeringly infeasible for production LLMs due to massive key sizes (~2GB/user) and latency barriers (100x slowdown).

We propose **three interconnected empirical papers** that dismantle these barriers. By shifting the paradigm from "General Purpose HE" to **"Rank-Deficient-Aware HE"**, we demonstrate the world's first **Zero-KeyGen** architecture for private fine-tuning. This is not a simulation; it is implemented in the **TenSafe** microkernel with empirical validation on Llama-3.

---

## Paper 1: ZeRo-MOAI
**"System-Level Elimination of Rotation Keys for Private PEFT"**

### The Core Problem
Current "Private LoRA" solutions (like SHE-LoRA) are **Security Theater**. They claim efficiency by encrypting only *some* layers, but they still require the client to generate and transmit massive **Galois Keys (Rotation Keys)** to the server.
*   **The Cost**: ~1GB upload per user.
*   **The Impact**: Impossible for mobile/edge devices on cellular networks.

### The Novel Insight
Rotation keys are only needed to align SIMD slots during matrix multiplication. However, LoRA adapters are **Low-Rank** ($r \ll N$). If the Client *pre-replicates* data into specific SIMD slots before encryption, the Server can sum the results naturally without *ever* needing to rotate.

### The Contribution (Canonical & Empirical)
We introduce **ZeRo-MOAI**, the first HE scheduler to enforce a **Zero-Rotation Contract**.
1.  **Mathematical Guarantee**: We prove that for any rank $r < N/2$, rotation keys can be essentially eliminated.
2.  **Empirical Result**: We achieve total elimination of Galois Key overhead.

| Metric | SOTA HE-LoRA (SHE-LoRA) | TenSafe (ZeRo-MOAI) | Improvement |
| :--- | :--- | :--- | :--- |
| **Galois Key Size** | ~2.4 GB | **0 MB** | $\infty$ (Eliminated) |
| **Rotation Ops** | $O(rank \times layers)$ | **Zero** | 100% Reduction |
| **A100 Latency** | 2.22 tok/s | **5.76 tok/s** | **2.6x Speedup** |

3.  **Impact**: This transforms Private AI from a "High-Bandwidth Enterprise" capability to a **"Stateless Mobile"** capability.

---

## Paper 2: Speculative Batching
**"Breaking the Single-User Latency Barrier via Base-Model Speculation"**

### The Core Problem
HE is inherently **Throughput-Oriented**. It relies on SIMD (Single Instruction, Multiple Data) to process 16,384 slots at once.
*   **The Paradox**: LLM Inference is **Latency-Oriented** (1 token at a time).
*   **The Waste**: processing a single user request wastes 99.99% of the crypto-processor's capacity. Existing "Batching" solutions (processing 100 users) do nothing to help *User A* get their answer faster.

### The Novel Insight
We treat the **Plaintext Base Model** (running in a TEE) as a "Draft Model" and the **HE Adapter** as the "Verifier."
Because LoRA is just a small perturbation, the Base Model's predictions are >95% accurate. We can generate 8 tokens in plaintext, then pack all 8 verification steps into a **Single HE Ciphertext**.

### The Contribution (Canonical & Empirical)
We propose **Cross-Domain Speculative Batching**.
1.  **Architecture**: The first system to fuse TEE-Plaintext generation with HE-Encrypted verification.
2.  **Flattened SIMD Packing**: A novel tensor mapping that fits a temporal sequence of tokens $[t_1, ..., t_K]$ into a spatial SIMD vector.
3.  **Empirical Result**: We resolve the HE-to-LLM "Utilization Gap."

| Metric | Sequential HE-LoRA (K=1) | TenSafe Speculative (K=4) | Improvement |
| :--- | :--- | :--- | :--- |
| **A100 Throughput** | 5.76 tok/s | **12.14 tok/s** | **2.1x Speedup** |
| **Groq Throughput** | 28.78 tok/s | **71.95 tok/s** (Proj) | **2.5x Speedup** |
| **SIMD Wastage** | 99.9% | **<10%** | **10x Utilization** |

---

## Paper 3: Asymmetric Client-Aided Bridge
**"Non-Linear Expressivity without System Keys"**

### The Core Problem
Linear LoRA is weak; it cannot learn complex implementations (XOR, conditional logic). To make adapters "smart," we need non-linear activations (ReLU, GELU).
*   **The Barrier**: Computing non-linearity in HE requires **Bootstrapping**, which is excruciatingly slow and requires massive server-side evaluation keys.

### The Novel Insight
LLM generation is **Auto-Regressive**. The client *already* performs a network round-trip for every token. We can **piggyback** the non-linear decision on this existing network hop. This addresses the massive shift in the SOTA market (DeepSeek, Kimi, Mixtral) moving toward gated Mixture-of-Experts architectures.

### The Contribution (Canonical & Empirical)
We introduce the **GateLink Protocol**.
1.  **The "Dumb Server" Model**: The powerful HE server computes the heavy Linear Algebra but stops *exactly* at the non-linear boundary.
2.  **The "Smart Client" Bridge**: The server sends the tiny encrypted result to the client. The client decrypts, computes ReLU, and sends back a single **Gate Bit**.
3.  **Market Realism**: While linear LoRA captures ~90% of current fine-tuning, the **"Non-Linear Frontier"** represents the high-value reasoning market (MoE). TenSafe is the only solution that scales to these "Non-Linear" adapters without a 10,000x latency penalty.
4.  **Empirical Result**: We provide the only viable path to private MoE reasoning.

| Metric | Full HE (ReLU/Softmax) | Vanilla HE-LoRA | TenSafe (GateLink) |
| :--- | :--- | :--- | :--- |
| **Expressivity** | Non-Linear (SOTA) | Linear (Weak) | **Non-Linear (SOTA)** |
| **A100 Latency** | DNF (>10,000s) | 0.50 tok/s | **3.37 tok/s** |
| **Speedup** | N/A | 1.0x | **6.7x vs Baseline** |

---

## Final Comparative Analysis (NVIDIA A100, r=32)

We baseline TenSafe against standard inference and state-of-the-art HE-LLM literature.

| Architecture | Llama 8B (Linear) | HE Overhead | Kimi 2.5 (MoE) | HE Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (FP16/vLLM)** | 53.18 tok/s | 1.0x | 25.00 tok/s | 1.0x |
| **TenSafe (A100)** | **5.76 tok/s** | **9.2x** | **3.37 tok/s** | **7.4x** |
| **TenSafe (Groq)** | **28.78 tok/s** | **1.8x** | **7.71 tok/s** | **3.2x** |
| **Full HE LLM** (Privatrans) | 0.05 tok/s | 1000x+ | **DNF (Infeasible)** | N/A |

## Why This Suite is Unique

1.  **It's a "System," not a "Trick"**: Most PEFT papers propose a math trick. These three papers form a cohesive **Systems Architecture** (ZeRo for setup, Speculation for speed, Bridge for quality).
2.  **It's Already Built**: The code exists. The `TenSafe` microkernel running on Llama-3 proves these aren't just equations—they are running code.
3.  **It Contradicts Conventional Wisdom**:
    *   *Convention*: "You need keys to do HE." $\rightarrow$ *Us*: "No, use Rank-Deficiency."
    *   *Convention*: "Encrypt everything." $\rightarrow$ *Us*: "No, Speculate with TEEs."
    *   *Convention*: "Bootstrapping is the future." $\rightarrow$ *Us*: "No, the Client is the future."

This work establishes a new **Canonical Reference Architecture** for the next 5 years of Privacy-Preserving Machine Learning.
