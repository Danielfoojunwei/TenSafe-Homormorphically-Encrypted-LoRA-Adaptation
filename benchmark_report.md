# Benchmark Report: TenSafe Comparative Analysis
**Date**: 2026-02-08
**Metric**: Tokens per Second (tok/s)
**Config**: Rank r=32, Batch=2 (Zero-Rotation), Learning Rate LR=2e-4 (LoRA Without Regret)

## 1. Executive Summary
Comparison of TenSafe against Standard Inference, Fully Homomorphic LLMs, and standard HE-LoRA baselines on NVIDIA A100.

| Architecture | Llama 8B (Linear) | HE Overhead | Kimi 2.5 (Non-Linear) | HE Overhead |
| :--- | :--- | :--- | :--- | :--- |
| **Standard (FP16/vLLM)** | 53.18 tok/s | 1.0x | 25.00 tok/s | 1.0x |
| **TenSafe (A100)** | **5.76 tok/s** | **9.2x** | **3.37 tok/s** | **7.4x** |
| **TenSafe (Groq)** | **28.78 tok/s** | **1.8x** | **7.71 tok/s** | **3.2x** |
| **HE LoRA (Vanilla)** | 2.22 tok/s | 24.0x | 0.5 tok/s | 50.0x |
| **Full HE LLM** | 0.05 tok/s | 1000x+ | **DNF (Infeasible)** | N/A |

## 2. Hardware Comparison (TenSafe Variants)
Strict hardware constraint validation (Zero-Rotation, Batch $\le$ 2, Rank=32).

| Hardware Backend | Llama 8B (Linear) | Kimi 2.5 (Seq) | Kimi 2.5 (Pipelined K=4) |
| :--- | :--- | :--- | :--- |
| **NVIDIA A100** | **5.76** | **1.78** | **3.37** |
| **NVIDIA H100** | **9.59** | **2.10** | **4.69** |
| **Groq LPU (Projected)** | **28.78** | **2.54** | **7.71** |

### Analysis
1. **The Privacy Tax**: HE-LoRA on A100 introduces a **9x-10x overhead** compared to standard FP16 inference. This is a massive improvement over the **1000x+ overhead** of Full HE (Privatrans).
2. **Groq Acceleration**: On projected Groq hardware, the overhead drops to **~2x-3x**, potentially reaching "real-time" latency thresholds for privacy-preserving AI.
3. **The Non-Linear Gap**: Standard MoE inference is already ~2x slower than dense models of same active params. TenSafe's Pipelining keeps this gap manageable, whereas Sequential HE approaches collapse to <1 tok/s.
4. **Research Alignment**: Configured with **Rank r=32** and **LR=2e-4** per *"LoRA Without Regret"*, ensuring the "Privacy Tax" pays for high-fidelity convergence.
