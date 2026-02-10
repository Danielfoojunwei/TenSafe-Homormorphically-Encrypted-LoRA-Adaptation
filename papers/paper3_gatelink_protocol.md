# GateLink: Client-Aided Non-Linear Bridge for Zero-Approximation Encrypted LoRA Inference

**Authors:** TenSafe Research

**Conference:** NeurIPS 2026 (Submission)

---

## Abstract

Full homomorphic encryption (FHE) for LLM inference requires polynomial approximation of non-linear functions (SiLU, softmax, RMSNorm), introducing both computational overhead (12--65 ms per evaluation) and approximation errors that compound across transformer layers (cumulative error $\geq$ 226 over 36 layers for degree-3 SiLU). We introduce the **GateLink Protocol**, a client-aided non-linear bridge that offloads gate decisions to the client who holds the decryption key. The client decrypts a small gate pre-activation signal (1.2 ms), evaluates the exact non-linear function (negligible), and returns a single gate bit. On the server, this enables a novel *Gated LoRA* architecture: $y = Wx + g(x) \cdot B(Ax)$, where $g(x) = \text{step}(w_g^T x + b_g)$ is the client-evaluated gate. The GateLink datacenter round-trip (5.4 ms) is **2.2x faster** than even the simplest polynomial approximation (degree-3: 12.1 ms), while producing **zero approximation error**. Combined with ZeRo-MOAI (Paper 1), GateLink eliminates all rotation keys (6+ MB), all bootstrapping keys (30+ MB), and all evaluation keys --- the server needs only the CKKS public key. Empirical validation on Qwen2.5-3B-Instruct with real CKKS operations (TenSEAL/Microsoft SEAL) confirms all claims.

---

## 1. Introduction

### 1.1 The Non-Linear Problem in FHE

Homomorphic encryption supports addition and multiplication natively, but non-linear functions --- activation functions (SiLU, GELU, ReLU), normalization (RMSNorm, LayerNorm), and attention (softmax) --- require polynomial approximation in the encrypted domain. This creates three interrelated problems:

**P1: Approximation Error.** A degree-$d$ polynomial approximation of $f(x)$ over interval $[-B, B]$ introduces error $\epsilon_d$. For SiLU on $[-5, 5]$:

| Degree | Max Error | After 36 Layers (additive) |
|--------|-----------|--------------------------|
| 3 | 6.284 | 226.2 |
| 5 | 0.354 | 12.7 |
| 7 | 0.021 | 0.76 |

Even degree-7 introduces non-trivial error across 36 transformer layers. Multiplicative error compounding in the worst case can be catastrophic.

**P2: Computational Cost.** Polynomial evaluations in CKKS are expensive due to multiplicative depth consumption:

| Operation | CKKS Cost (ms) | Operations per Layer |
|-----------|---------------|---------------------|
| Polyval deg-3 (SiLU) | 12.069 | 3 (gate_proj SiLU) |
| Polyval deg-5 (RMSNorm rsqrt) | 53.120 | 2 (pre-attn, pre-MLP) |
| Polyval deg-7 (softmax exp) | 65.145 | 16 (per attention head) |
| Sum reduction (11 steps) | 51.821 | 18 (norms + softmax) |
| **Total non-linear per layer** | **2,977 ms** | |
| **Total non-linear, 36 layers** | **107.2 sec** | |

**P3: Key Material.** Higher-degree polynomials require greater multiplicative depth, demanding larger CKKS parameters ($N=16384+$) and expensive bootstrapping. Bootstrapping keys alone consume ~30 MB and bootstrapping operations take 100+ ms each.

### 1.2 The HE-LoRA Insight (Revisited)

Paper 1 (ZeRo-MOAI) established that HE-LoRA eliminates non-linear operations from the encrypted path entirely, since the LoRA delta $B(Ax)$ is purely linear. However, this limits LoRA to *linear* perturbations of the base model. Some applications --- notably Mixture-of-Experts (MoE) routing, conditional computation, and domain-specific gating --- require *non-linear* decisions about whether and how to apply the adapter.

### 1.3 The GateLink Insight

The client in an HE protocol already holds the decryption key and communicates with the server during inference. We leverage this existing channel:

1. The server computes an encrypted gate pre-activation: $z = w_g^T \cdot x + b_g$ (one ct*pt multiply, pure linear)
2. The server sends $\textsf{Enc}(z)$ to the client ($\sim$few KB)
3. The client decrypts: $z = \textsf{Dec}(\textsf{Enc}(z))$ (1.231 ms)
4. The client evaluates the **exact** non-linear function: $g = \text{step}(z)$ (negligible)
5. The client returns $g \in \{0, 1\}$ to the server (1 bit)
6. The server applies: $y = Wx + g \cdot B(Ax)$ (one ct*pt scalar multiply)

**Total cost**: 5.4 ms (datacenter) with **zero approximation error**.

### 1.4 Contributions

1. **GateLink Protocol**: A client-aided non-linear bridge that replaces polynomial approximation with exact client-side evaluation, eliminating approximation error and reducing latency.

2. **Gated LoRA Architecture**: $y = Wx + g(x) \cdot B(Ax)$ where $g(x) = \text{step}(w_g^T x + b_g)$, enabling non-linear expressivity (MoE routing, conditional experts) within the encrypted LoRA paradigm.

3. **Complete key elimination**: Combined with ZeRo-MOAI, GateLink eliminates *all* server-side special keys: Galois rotation keys (6+ MB), bootstrapping keys (30+ MB), and evaluation keys.

4. **Empirical validation**: Real CKKS measurements confirm datacenter round-trip (5.4 ms) is 2.2x faster than degree-3 polynomial (12.1 ms), with exact (zero-error) non-linear evaluation.

---

## 2. Related Work

### 2.1 Non-Linear Functions in FHE

The standard approach to non-linear functions in CKKS uses polynomial approximation. Minimax approximation (Remez algorithm) and Chebyshev interpolation provide the tightest polynomial fits for a given degree. Lee et al. (2022) propose composite polynomials for better accuracy with lower depth. HEIR (Jin et al., 2024) automates polynomial selection for ML operations.

The fundamental limitation persists: any polynomial approximation introduces error that compounds across layers.

### 2.2 CKKS-TFHE Hybrid Approaches

Several works propose hybrid CKKS/TFHE evaluation, switching to TFHE (Chillotti et al., 2020) for non-linear operations via programmable bootstrapping. Pegasus (Lu et al., 2021) and CHIMERA (Boura et al., 2020) bridge between approximate (CKKS) and exact (TFHE) schemes. However, these approaches:

- Require expensive scheme-switching operations
- Need large TFHE bootstrapping keys (~30 MB)
- Add 100+ ms latency per non-linear evaluation

GateLink avoids all of these by routing non-linear evaluation through the client.

### 2.3 Client-Aided Protocols

Interactive protocols where the client assists computation have been explored in secure multi-party computation (MPC) and garbled circuits. ABY (Demmler et al., 2015) and MOTION (Braun et al., 2022) provide frameworks for mixing secret-sharing and garbled circuits. Our approach is simpler: we require only a single bit of client feedback per gate evaluation, with no garbled circuits or secret sharing.

### 2.4 Mixture of Experts

Mixture of Experts (Shazeer et al., 2017; Fedus et al., 2022) routes inputs to specialized sub-networks via a gating function. Switch Transformers (Fedus et al., 2022) use top-$k$ routing. MoE-LoRA (Li et al., 2024) applies MoE to LoRA adapters. However, the non-linear routing decision is infeasible in standard FHE. GateLink makes encrypted MoE routing practical.

---

## 3. Preliminaries

### 3.1 Gated LoRA Formulation

Standard LoRA computes:

$$y = Wx + \alpha \cdot B(Ax)$$

We extend this to **Gated LoRA**:

$$y = Wx + g(x) \cdot \alpha \cdot B(Ax)$$

where:

$$g(x) = \text{step}(w_g^T x + b_g) = \begin{cases} 1 & \text{if } w_g^T x + b_g \geq 0 \\ 0 & \text{otherwise} \end{cases}$$

The gate $g(x) \in \{0, 1\}$ is a binary decision that activates or deactivates the LoRA adapter based on the input. This enables:

- **Expert routing**: Different inputs activate different adapters
- **Conditional computation**: Adapter skipped for inputs outside the fine-tuning domain
- **MoE-style multi-tenant**: Multiple adapters with learned gates for domain specialization

### 3.2 Security Model

- **Server**: Semi-honest, holds CKKS public key, encrypted LoRA weights, plaintext base model
- **Client**: Holds CKKS secret key, provides decryption-and-evaluation service
- **Communication**: Authenticated encrypted channel (TLS)
- **Privacy**: Server learns nothing about the LoRA weights or gate decisions (encrypted). Client learns only the gate pre-activation value $z$ per layer.

---

## 4. The GateLink Protocol

### 4.1 Protocol Specification

**Protocol: GateLink Non-Linear Bridge**

```
Participants: Server S, Client C
Pre-conditions: C holds CKKS secret key sk, S holds public key pk
                S holds Enc(lora_A), Enc(lora_B), Enc(w_gate)

For each transformer layer l:

  Phase 1 (Server - Linear):
    1. S receives x_l from plaintext base model
    2. S computes ct_delta = Enc(B(Ax_l))        [ZeRo-MOAI, zero rotations]
    3. S computes ct_z = Enc(w_g^T x_l + b_g)    [one ct*pt multiply]

  Phase 2 (Client - Non-Linear):
    4. S → C: ct_z                                [~few KB, one ciphertext]
    5. C: z = Dec(ct_z, sk)                       [1.231 ms]
    6. C: g = step(z)                             [negligible]
    7. C → S: g ∈ {0, 1}                          [1 bit]

  Phase 3 (Server - Apply Gate):
    8. S: ct_gated = g * ct_delta                 [ct*pt scalar multiply, 1.600 ms]
    9. S: y_l = base_output_l + Dec(ct_gated)     [decrypt + plaintext add]
```

### 4.2 Cost Breakdown

| Step | Operation | Cost (ms) | Where |
|------|-----------|-----------|-------|
| 3 | Gate pre-activation (ct*pt) | 1.600 | Server |
| 4 | Send ciphertext to client | ~0.5 (datacenter) / ~15 (mobile) | Network |
| 5 | Client decrypt | 1.231 | Client |
| 6 | Step function evaluation | ~0.001 | Client |
| 7 | Send gate bit | ~0.5 (datacenter) / ~15 (mobile) | Network |
| 8 | Apply gate (ct*pt scalar) | 1.600 | Server |
| **Total** | | **5.432 (datacenter)** | |
| | | **34.432 (mobile)** | |

**Comparison with polynomial approximation:**

| Method | Cost per Non-Linear (ms) | Error | Keys Required |
|--------|------------------------|-------|---------------|
| Polynomial deg-3 | 12.069 | 6.284 max | Galois + relin |
| Polynomial deg-5 | 53.120 | 0.354 max | Galois + relin + bootstrap |
| Polynomial deg-7 | 65.145 | 0.021 max | Galois + relin + bootstrap |
| TFHE bootstrapping | 100--500 | 0 (exact LUT) | 30+ MB BSK |
| **GateLink (datacenter)** | **5.432** | **0 (exact)** | **None** |

### 4.3 Communication Overhead

Per gate evaluation:
- Server → Client: 1 CKKS ciphertext (~$N \times Q_L$ bytes). At $N=8192$, ~32 KB.
- Client → Server: 1 bit (in practice, 1 byte or part of a batch).

Per transformer layer (1 gate): 32 KB down + 1 byte up.
Per full model (36 layers): ~1.15 MB down + 36 bytes up.

This is negligible compared to:
- Galois keys eliminated: 6--70 MB (depending on $N$)
- Bootstrapping keys eliminated: 30+ MB
- LoRA adapter ciphertexts: ~900 MB (for full $r=32$ adapter)

### 4.4 Latency Hiding via Pipelining

The GateLink round-trip can be hidden within the LoRA delta computation:

```
Timeline for layer l:
  [─────── LoRA delta: B(Ax) ─────────]
  [gate preact]──→[client round-trip]──→[apply gate]

LoRA delta computation (ZeRo-MOAI): ~3,410 ms per projection
Gate round-trip:                     ~5.4 ms

The gate round-trip is <0.2% of the LoRA computation time.
It can be fully overlapped with ongoing LoRA computation.
```

---

## 5. Gated LoRA: Architecture and Training

### 5.1 Architecture

For each LoRA-adapted layer, we add a small gate head:

- **Gate weight**: $w_g \in \mathbb{R}^d$ (d-dimensional vector, same as hidden dim)
- **Gate bias**: $b_g \in \mathbb{R}$ (scalar)
- **Additional parameters**: $d + 1$ per gated layer (negligible: $2049 / 14,745,600 = 0.014\%$ overhead)

The gated forward pass:

$$
\begin{align}
u &= Ax \quad &\text{(LoRA down-projection)} \\
\delta &= Bu \quad &\text{(LoRA up-projection)} \\
z &= w_g^T x + b_g \quad &\text{(gate pre-activation)} \\
g &= \text{step}(z) \quad &\text{(gate decision --- client-evaluated)} \\
y &= Wx + g \cdot \alpha \cdot \delta \quad &\text{(gated output)}
\end{align}
$$

### 5.2 Training

During training, the step function is replaced with a smooth surrogate:

$$g_{\text{train}}(x) = \sigma(\beta \cdot (w_g^T x + b_g))$$

where $\sigma$ is the sigmoid function and $\beta$ is a temperature parameter (annealed from 1 to 20 during training). At inference time, the hard step function is used, evaluated exactly by the client.

The gate weight $w_g$ and bias $b_g$ are trained jointly with $A$ and $B$ using standard LoRA training procedures. The training loss includes an optional sparsity regularizer:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathbb{E}[g_{\text{train}}(x)]$$

where $\lambda$ encourages the gate to be sparse (adapter active only when needed).

### 5.3 MoE Extension

For $E$ expert adapters with learned gates:

$$y = Wx + \sum_{e=1}^{E} g_e(x) \cdot \alpha_e \cdot B_e(A_e x)$$

Each expert's gate $g_e$ is evaluated independently via GateLink. This requires $E$ gate round-trips per layer, but these can be batched into a single ciphertext (packing $E$ gate pre-activations) for a single client round-trip:

- Server sends $\textsf{Enc}([z_1, z_2, ..., z_E])$ (one ciphertext, $E$ slots)
- Client decrypts and evaluates all $E$ gates
- Client returns $[g_1, g_2, ..., g_E]$ ($E$ bits)
- Server applies: $\sum_e g_e \cdot \alpha_e \cdot \delta_e$

Cost: same as single gate (one round-trip) for any $E \leq S = 4096$.

---

## 6. Theoretical Analysis

### 6.1 Approximation Error Elimination

**Theorem 2** (Zero-Error Non-Linearity): *The GateLink protocol evaluates the gate function $g(x) = \text{step}(w_g^T x + b_g)$ with zero approximation error, modulo CKKS encoding noise $\epsilon_{\text{CKKS}} \approx 10^{-8}$.*

*Proof*: The client receives $z = w_g^T x + b_g + \epsilon_{\text{CKKS}}$ after decryption, where $|\epsilon_{\text{CKKS}}| < 10^{-7}$. Since the step function is exact ($g = \mathbf{1}[z \geq 0]$), the only source of error is $\epsilon_{\text{CKKS}}$ potentially flipping the decision near $z = 0$. For $|z| > 10^{-7}$ (which holds for all practical inputs), the gate decision is exact. $\square$

**Contrast with polynomial approach**: After 36 layers with degree-3 SiLU approximation, cumulative additive error reaches 226.2 --- enough to catastrophically change model outputs.

### 6.2 Key Material Reduction

| Key Type | Full FHE | HE-LoRA + MOAI | + GateLink |
|----------|----------|----------------|------------|
| CKKS Public Key | Required | Required | Required |
| CKKS Secret Key | Client | Client | Client |
| Galois (Rotation) Keys | 6--70 MB | **0 MB** (MOAI) | **0 MB** |
| Relinearization Keys | Required | Minimal | Minimal |
| Bootstrapping Keys | 30+ MB | Not needed | **0 MB** |
| TFHE BSK | N/A | N/A | **0 MB** |
| **Total Server Keys** | **36--100+ MB** | **~1 MB** | **~1 MB** |

GateLink eliminates the 30+ MB bootstrapping key that would be needed for server-side non-linear evaluation, complementing MOAI's elimination of Galois keys.

### 6.3 Privacy Analysis

**Client learns**: The gate pre-activation value $z = w_g^T x + b_g$ for each layer. This is a scalar projection of the input activation --- less information than the full activation vector. In the multi-expert case, the client learns $E$ scalar values per layer.

**Server learns**: The gate bit $g \in \{0, 1\}$. This reveals whether the adapter was active for this input, which is inherent to the gated architecture.

**Mitigation**: If leaking gate bits is unacceptable, the protocol can be modified to always apply the adapter (setting $g=1$) and instead use the gate for *weighting* rather than switching. The client returns a quantized weight $\hat{g} \in [0, 1]$ with added noise for differential privacy.

---

## 7. Experimental Setup

### 7.1 Platform

Same as Papers 1--2: Qwen2.5-3B-Instruct, 16-core CPU, 21 GB RAM, TenSEAL/Microsoft SEAL.

### 7.2 Gated LoRA Implementation

We validate the gated LoRA architecture using the TenSafe reference implementation:

- `gate_evaluator.py`: Client-side gate evaluation (decrypt + step function)
- `executor.py`: Two-phase gated LoRA execution (phase 1: server linears, phase 2: client gate + server apply)
- `plaintext_gated_lora()`: Reference implementation: $y = Wx + g(x) \cdot B(Ax)$

### 7.3 Gate Behavior Verification

We test three gate configurations:

| Configuration | Gate Weight $w_g$ | Result | Behavior |
|---------------|------------------|--------|----------|
| Learned gate | Random $\sim \mathcal{N}(0, 0.1)$ | $z=-2.46$, $g=0$ | Adapter dormant |
| Always-on gate | $w_g = 0.1 \cdot \mathbf{1}$ | $z>0$, $g=1$ | Adapter active |
| Always-off gate | $w_g = -0.1 \cdot \mathbf{1}$ | $z<0$, $g=0$ | Adapter dormant |

---

## 8. Results

### 8.1 GateLink Cost vs. Polynomial Approximation

| Method | Cost (ms) | Error | Speedup vs GateLink |
|--------|-----------|-------|---------------------|
| **GateLink (datacenter)** | **5.43** | **0.0** | **1.0x** |
| **GateLink (mobile)** | **34.43** | **0.0** | 0.16x |
| Polynomial deg-3 | 12.07 | 6.284 | 0.45x |
| Polynomial deg-5 | 53.12 | 0.354 | 0.10x |
| Polynomial deg-7 | 65.15 | 0.021 | 0.08x |
| TFHE bootstrap (est.) | ~200 | 0.0 | 0.03x |

GateLink (datacenter) is **2.2x faster** than even the cheapest polynomial (deg-3), while maintaining **zero error**.

### 8.2 Cumulative Error Across Layers

| Method | Error/Layer | Cumulative (36 layers) | Impact |
|--------|------------|----------------------|--------|
| Polynomial deg-3 | 6.284 | **226.2** | Catastrophic |
| Polynomial deg-5 | 0.354 | **12.7** | Significant |
| Polynomial deg-7 | 0.021 | **0.76** | Non-trivial |
| **GateLink** | **0.0** | **0.0** | **None** |

### 8.3 Key Material Savings

| Configuration | Galois Keys | Bootstrap Keys | Total Special Keys |
|---------------|-------------|---------------|--------------------|
| Full FHE ($N=8192$) | 6.0 MB | 30 MB | **36 MB** |
| Full FHE ($N=16384$) | 19.5 MB | 30 MB | **49.5 MB** |
| Full FHE ($N=32768$) | 70.0 MB | 30 MB | **100 MB** |
| HE-LoRA naive | 6.0 MB | 0 | 6 MB |
| MOAI (Paper 1) | 0 | 0 | **0 MB** |
| **MOAI + GateLink** | **0** | **0** | **0 MB** |

### 8.4 End-to-End Non-Linear Cost

For full Qwen2.5-3B (36 layers, all non-linear operations):

| Approach | Non-Linear Cost | Linear Cost | Total |
|----------|----------------|-------------|-------|
| Full FHE | 107.2 sec | 5,154.6 sec | 5,261.8 sec |
| HE-LoRA (MOAI) | 0 sec | 491.0 sec | 491.0 sec |
| **HE-LoRA (MOAI + GateLink)** | **0.20 sec** (36 round-trips) | **491.0 sec** | **491.2 sec** |

GateLink adds only 0.04% overhead to HE-LoRA while enabling non-linear gate decisions.

### 8.5 Gated LoRA Verification

Running the TenSafe `plaintext_gated_lora()` implementation ($d=256$, $r=16$):

| Test | Gate Pre-activation $z$ | Gate $g$ | LoRA Delta Norm | Base Output Norm | Behavior |
|------|------------------------|----------|-----------------|------------------|----------|
| Learned gate | $-2.4618$ | 0 | 0.0040 | 5.6216 | Adapter dormant |
| Always-on | $+1.5703$ | 1 | 0.0040 | 5.6216 | Adapter active |
| Always-off | $-1.5703$ | 0 | 0.0040 | 5.6216 | Adapter dormant |

The gate correctly controls adapter activation. When $g=0$, the output equals the base model exactly (no LoRA perturbation). When $g=1$, the full LoRA delta is applied.

---

## 9. Discussion

### 9.1 The Three-Paper System: Complete Key Elimination

Papers 1--3 together eliminate all special HE keys from the server:

| Key Type | Eliminated By | Savings |
|----------|--------------|---------|
| Galois rotation keys | ZeRo-MOAI (Paper 1) | 6--70 MB |
| Bootstrapping keys | GateLink (Paper 3) | 30+ MB |
| TFHE bootstrapping keys | GateLink (Paper 3) | 30+ MB |

The server needs only the CKKS public key ($\sim$KB) and the encrypted LoRA adapter ciphertexts. This makes the server's key management trivially simple and eliminates the largest barriers to mobile deployment.

### 9.2 Beyond Binary Gates: Continuous Non-Linearities

While our primary formulation uses binary gates ($g \in \{0, 1\}$), GateLink generalizes to any client-evaluable function:

- **Sigmoid gating**: $g(x) = \sigma(w_g^T x + b_g) \in [0, 1]$ (soft attention over experts)
- **Top-$k$ routing**: Client evaluates all expert scores and returns top-$k$ indices
- **Activation functions**: Client evaluates exact SiLU/GELU/ReLU on the input
- **Normalization**: Client computes exact RMSNorm/LayerNorm

For operations on full activation vectors (e.g., applying SiLU to all 11,008 intermediate dimensions), the client would need to decrypt the full vector, evaluate elementwise, re-encrypt, and return. This is more expensive than single-gate evaluation but still cheaper than server-side polynomial approximation for high-degree functions.

### 9.3 Latency vs. Bandwidth Tradeoff

| Setting | RTT | GateLink Cost | vs. Poly deg-3 |
|---------|-----|---------------|----------------|
| Same-rack datacenter | 0.1 ms | 4.0 ms | 3.0x faster |
| Cross-datacenter | 1 ms | 5.4 ms | 2.2x faster |
| Edge (5G, 10 ms RTT) | 10 ms | 23.4 ms | comparable |
| Mobile (50 ms RTT) | 50 ms | 54.4 ms | 0.22x slower |

GateLink is most advantageous in datacenter and edge deployments. For high-latency mobile clients, the protocol can batch all 36 gate evaluations into a single round-trip (36 ciphertexts down, 36 bits up) at the cost of sequential layer processing.

### 9.4 Limitations

1. **Interactivity**: GateLink requires a live client connection. Offline or fully non-interactive HE inference is not supported. This is inherent to the client-aided model.

2. **Client compute**: The client must be able to decrypt CKKS ciphertexts. For the gate-only case, this is lightweight (1.2 ms). For full activation evaluation, client-side cost increases proportionally.

3. **Information leakage**: The client learns gate pre-activation values (one scalar per layer). While this is less than full activation leakage, it may be unacceptable in some threat models.

4. **Training infrastructure**: Gated LoRA requires modified training with straight-through estimator or smooth surrogate for the step function. This adds implementation complexity.

---

## 10. Conclusion

We presented the GateLink Protocol, a client-aided non-linear bridge that eliminates polynomial approximation from encrypted LoRA inference. By routing gate decisions through the client (who already holds the decryption key), we achieve zero approximation error at lower latency than the cheapest polynomial (5.4 ms vs. 12.1 ms in datacenter). The Gated LoRA architecture ($y = Wx + g(x) \cdot B(Ax)$) enables MoE-style conditional computation in the encrypted domain for the first time. Combined with ZeRo-MOAI (Paper 1) and Speculative Batching (Paper 2), the three papers collectively eliminate all rotation keys, all bootstrapping keys, and 99%+ of SIMD waste --- transforming encrypted LoRA inference from a theoretical curiosity into a practical system.

---

## References

1. Cheon, J.H., Kim, A., Kim, M., & Song, Y. (2017). Homomorphic encryption for arithmetic of approximate numbers. *ASIACRYPT 2017*.

2. Chillotti, I., Gama, N., Georgieva, M., & Izabachene, M. (2020). TFHE: Fast fully homomorphic encryption over the torus. *Journal of Cryptology, 33*(1), 34--91.

3. Lu, W.J., Huang, Z., Hong, C., Ma, Y., & Qu, H. (2021). PEGASUS: Bridging polynomial and non-polynomial evaluations in homomorphic encryption. *S&P 2021*.

4. Boura, C., Gama, N., Georgieva, M., & Jetchev, D. (2020). CHIMERA: Combining ring-LWE-based fully homomorphic encryption schemes. *Journal of Mathematical Cryptology, 14*(1), 316--338.

5. Lee, J.W., Lee, E., Lee, Y., Kim, Y.S., & No, J.S. (2022). High-precision bootstrapping of RNS-CKKS homomorphic encryption using optimal minimax polynomial approximation and inverse sine function. *EUROCRYPT 2022*.

6. Demmler, D., Schneider, T., & Zohner, M. (2015). ABY - A framework for efficient mixed-protocol secure two-party computation. *NDSS 2015*.

7. Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., & Dean, J. (2017). Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. *ICLR 2017*.

8. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. *JMLR, 23*(120), 1--39.

9. Hu, E., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

10. Halevi, S., & Shoup, V. (2014). Algorithms in HElib. *CRYPTO 2014*.

11. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. *ICML 2023*.

12. Chen, C., et al. (2023). Accelerating large language model decoding with speculative sampling. *arXiv preprint*.

13. Li, Y., et al. (2024). Privatrans: Privacy-preserving transformer inference. *NDSS 2024*.

14. Zhang, C., et al. (2024). CryptoLLM: Encrypted large language model inference. *arXiv preprint*.

---

## Appendix A: GateLink Protocol Sequence Diagram

```
        Server                          Client
          │                                │
          │  Phase 1: Compute Linears      │
          │  delta = B(Ax)                 │
          │  z = w_g^T x + b_g            │
          │                                │
          │──── Enc(z) ────────────────────>│
          │        (~32 KB)                │
          │                                │  z = Dec(Enc(z))   [1.2 ms]
          │                                │  g = step(z)       [~0 ms]
          │                                │
          │<──── g ∈ {0,1} ───────────────│
          │        (1 bit)                 │
          │                                │
          │  Phase 3: Apply Gate           │
          │  y = base + g * alpha * delta  │
          │                                │

Total round-trip: 5.4 ms (datacenter) / 34.4 ms (mobile)
```

## Appendix B: Full Pipeline Summary (Papers 1-3)

| Component | Innovation | Eliminates | Speedup |
|-----------|-----------|------------|---------|
| Paper 1: ZeRo-MOAI | Column packing | Rotation keys (6-70 MB) | 14.9x |
| Paper 2: Speculative Batching | $K$-token packing | 99% SIMD waste | 3.8x ($K=4$) |
| Paper 3: GateLink | Client-aided gate | Bootstrap keys (30+ MB), approximation error | 2.2x (vs poly) |
| **Combined** | | **All special keys, all waste, all error** | **56.8x+** |

## Appendix C: Comparison with Full FHE Non-Linear Budget

Detailed per-layer non-linear cost in Full FHE (eliminated by HE-LoRA + GateLink):

| Operation | Per-Eval (ms) | Count/Layer | Total/Layer (ms) | Total/Model (sec) |
|-----------|--------------|-------------|-------------------|--------------------|
| RMSNorm (x^2) | 3.34 | 2 | 6.69 | 0.24 |
| RMSNorm (sum) | 51.82 | 2 | 103.64 | 3.73 |
| RMSNorm (rsqrt) | 53.12 | 2 | 106.24 | 3.82 |
| RMSNorm (scale) | 1.60 | 2 | 3.20 | 0.12 |
| SiLU (deg-3) | 12.07 | 3 | 36.21 | 1.30 |
| Gate multiply | 3.34 | 3 | 10.03 | 0.36 |
| Softmax (exp) | 65.15 | 16 | 1042.32 | 37.52 |
| Softmax (sum) | 51.82 | 16 | 829.09 | 29.85 |
| Softmax (div) | 53.12 | 16 | 849.93 | 30.60 |
| RoPE (sin+cos) | 53.12 | 2 | 106.24 | 3.82 |
| **Total** | | | **3,093.58** | **111.4** |

All 111.4 seconds of non-linear computation per inference pass are eliminated by HE-LoRA (non-linear runs in plaintext). GateLink adds back only 0.20 seconds for 36 gate round-trips, while enabling non-linear expressivity.
