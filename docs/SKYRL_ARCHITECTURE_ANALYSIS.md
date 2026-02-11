# SkyRL Architecture Deep-Dive & Cross-Pollination Analysis for TenSafe

> **Date**: 2026-02-08
> **Scope**: Architectural comparison between [SkyRL](https://github.com/novasky-ai/skyrl) (Berkeley Sky Computing Lab / Anyscale) and TenSafe, focused on actionable lessons that stay within TenSafe's privacy-preserving mission.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SkyRL Architecture Overview](#2-skyrl-architecture-overview)
3. [Structural Similarities](#3-structural-similarities)
4. [Key Architectural Differences](#4-key-architectural-differences)
5. [Lessons and Recommendations (In-Scope)](#5-lessons-and-recommendations-in-scope)
6. [Anti-Lessons: What NOT to Adopt](#6-anti-lessons-what-not-to-adopt)
7. [Detailed Comparison Matrix](#7-detailed-comparison-matrix)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Executive Summary

SkyRL is a modular, Ray-native RL post-training framework from Berkeley/Anyscale with ~1.6k stars and 75+ contributors. It prioritizes **throughput, async training, and algorithm breadth** across four independent sub-packages (train, agent, gym, tx).

TenSafe prioritizes **privacy-preserving fine-tuning** via HE-LoRA, DP-SGD, and production-hardened security gates. Our RLVR module (639 lines) is compact compared to SkyRL's full RL stack (~15k+ lines), but our HE and privacy infrastructure has no parallel in SkyRL.

**The core insight**: SkyRL has solved many of the "RL training at scale" problems that TenSafe's RLVR module will encounter as it matures. We can adopt their patterns for **async rollout generation, advantage estimation diversity, weight synchronization, and environment abstractions** while keeping our HE/DP core untouched.

---

## 2. SkyRL Architecture Overview

### 2.1 Four Sub-Packages

| Package | Purpose | Scale |
|---|---|---|
| **skyrl-train** | Core RL training (PPO, GRPO, async) | ~15k lines |
| **skyrl-agent** | Multi-turn agent training (ReAct, CodeAct) | ~5k lines |
| **skyrl-gym** | Gymnasium environments (math, code, SQL, search) | ~4k lines |
| **skyrl-tx** | Self-hosted Tinker API for local post-training | ~8k lines |

### 2.2 Training Pipeline (Synchronous)

```
DataLoader -> prepare_generator_input()
  -> InferenceEngineClient.generate() [vLLM/SGLang]
    -> Rollout trajectories
  -> compute rewards + loss masks
  -> forward pass: policy logprobs, ref logprobs, critic values
  -> KL penalty to rewards
  -> advantage estimation (GRPO/GAE/RLOO/REINFORCE++)
  -> mini-batch PPO update (policy + critic)
  -> weight sync to inference engines
  -> checkpoint + eval
```

### 2.3 Fully Async Pipeline

SkyRL's `FullyAsyncRayPPOTrainer` decouples generation from training:

```
[N generation workers] --async--> [shared buffer] --batch--> [training loop]
                                                               |
                                           weight sync <-------+
                                               |
staleness manager <----------------------------+
```

Key mechanism: `_AsyncStalenessManager` bounds how stale rollout data can be relative to the current training step, dynamically controlling generation capacity.

### 2.4 Distributed Architecture

- **Workers**: Ray remote actors (`PolicyWorker`, `CriticWorker`, `RefWorker`)
- **Sharding**: FSDP/FSDP2 or Megatron tensor/pipeline parallelism
- **Inference**: Separate vLLM/SGLang processes with NCCL weight sync
- **Placement**: Three GPU strategies (colocate_all, colocate_policy_ref, isolated)

### 2.5 Algorithm Registry

SkyRL uses a `BaseFunctionRegistry` pattern where custom loss functions and advantage estimators are registered, serialized with cloudpickle, and distributed to Ray workers:

- **Advantage estimators**: GAE, GRPO, RLOO, REINFORCE++
- **Policy losses**: PPO, GSPO, SAPO, CISPO, Clip-Cov, KL-Cov, cross-entropy, importance sampling
- **Off-policy correction**: Token-level and sequence-level TIS ratios, outlier masking

---

## 3. Structural Similarities

Both projects share significant architectural DNA despite different missions.

### 3.1 Registry-Based Extensibility

| Aspect | TenSafe | SkyRL |
|---|---|---|
| Pattern | `@register_reward("name")` decorator | `BaseFunctionRegistry` + cloudpickle |
| Scope | Loss, reward, metrics functions | Advantage estimators, policy losses |
| Resolution | Dotted-path import + type validation | cloudpickle serialization to Ray workers |

**Both** use registry patterns for pluggable reward/loss functions. TenSafe's `tensafe.core.registry` (834 lines) and SkyRL's `BaseFunctionRegistry` serve the same purpose: decouple algorithm definitions from the training loop.

### 3.2 RL Training Loop Structure

Both follow the same high-level RLVR pattern:

```
prompts -> rollout generation -> reward computation -> advantage estimation
  -> policy gradient update -> weight sync -> repeat
```

TenSafe's `RLVRTrainer.step()` and SkyRL's `RayPPOTrainer._training_step()` are structurally equivalent, differing in scale and sophistication.

### 3.3 Modular Training Modes

| TenSafe | SkyRL |
|---|---|
| `TrainingModeInterface` (SFT, RLVR, DPO) | Trainer configs (PPO, GRPO, SFT via cross-entropy loss) |
| `TenSafePipeline.from_config()` | Hydra config -> trainer instantiation |

Both use abstract interfaces to support multiple training paradigms.

### 3.4 Configuration Hierarchy

Both use deeply nested, typed configuration:

| TenSafe | SkyRL |
|---|---|
| `TenSafeConfig` (Pydantic dataclass) | `SkyRLConfig` (Hydra + dataclass) |
| 7 sub-configs (Model, LoRA, Training, DP, HE, Inference, RLVR) | 5 sub-configs (Data, Trainer, Generator, Algorithm, Environment) |
| YAML/JSON + env var overrides | YAML + Hydra CLI overrides |

### 3.5 LoRA Support

Both integrate with HuggingFace PEFT for LoRA:

| Aspect | TenSafe | SkyRL |
|---|---|---|
| Framework | PEFT + custom best-practices layer | PEFT (direct) |
| Features | LoRA+, rsLoRA, DoRA, adaptive rank, FFA-LoRA | Standard LoRA with save/load via PEFT |
| Encryption | HE-encrypted LoRA deltas | N/A |

TenSafe's LoRA implementation is substantially more advanced due to `lora_best_practices/` (3,029 lines).

### 3.6 vLLM Integration

Both use vLLM for high-throughput inference:

| TenSafe | SkyRL |
|---|---|
| `he_lora_microkernel/backend/vllm_adapter/` | `inference_engines/vllm_engine.py` |
| HE-LoRA token execution through vLLM | Standard vLLM with weight sync |
| Production serving focus | Training rollout focus |

### 3.7 Trajectory / Rollout Abstractions

| TenSafe | SkyRL |
|---|---|
| `Trajectory(prompt, response, reward, logprobs, tokens, metadata)` | `GeneratorOutput(token_ids, rewards, loss_masks, metadata)` |
| `TrajectoryBuffer` with optional prioritization | `TrainingInputBatch` with micro-batching |
| `RolloutSampler.sample()` | `SkyRLGymGenerator.generate()` |

---

## 4. Key Architectural Differences

### 4.1 Scale of RL Infrastructure

This is the single largest gap. SkyRL's RL stack is ~25x the size of TenSafe's:

| Component | TenSafe | SkyRL |
|---|---|---|
| RLVR module | 639 lines | ~15,000+ lines |
| Algorithms | REINFORCE, PPO | PPO, GRPO, RLOO, REINFORCE++, SAPO, CISPO, GSPO, Clip-Cov, KL-Cov |
| Advantage estimators | Manual baseline subtraction | GAE, GRPO, RLOO, REINFORCE++ (registry) |
| Off-policy correction | None | Token-level TIS, sequence-level TIS, outlier masking |
| Async training | None | Full async with staleness control |
| Multi-turn RL | None | Agent loop with environment step interleaving |

### 4.2 Distributed Training Depth

| Aspect | TenSafe | SkyRL |
|---|---|---|
| Framework | Ray (planned), single-node training | Ray-native, fully distributed |
| Parallelism | Data parallel (DP-SGD) | FSDP/FSDP2, Megatron (TP/PP/CP/EP) |
| Weight sync | Checkpoint-based | NCCL broadcast, CUDA IPC, checkpoint |
| GPU placement | Single device_map | Placement groups, colocate/isolate strategies |
| Memory mgmt | Standard PyTorch | CPU offload/onload between models |

### 4.3 Inference/Generation Architecture

SkyRL's inference layer is production-grade for RL training:

- **InferenceEngineClient**: Load-balanced, pause/resume weight sync
- **InferenceRouter**: HTTP proxy with session-aware hashing
- **Server pools**: Multiple vLLM/SGLang instances managed as a fleet
- **Async generation**: Non-blocking rollout with partial token accumulation

TenSafe's inference focuses on HE-LoRA serving rather than RL rollout throughput.

### 4.4 Environment System

SkyRL has a full Gymnasium-compatible environment system (`skyrl-gym`):

```python
class Env[ObsType, ActType]:
    def init() -> (observations, metadata)
    def step(action) -> EnvStepOutput(observations, reward, done, metadata)
    def close()
```

With concrete environments for GSM8K, AIME, SQL, search, code. TenSafe has no equivalent; reward functions are standalone callables.

### 4.5 What TenSafe Has That SkyRL Lacks

| TenSafe Unique | Description |
|---|---|
| **Homomorphic Encryption** | CKKS-based HE-LoRA with GPU acceleration, MOAI zero-rotation |
| **Differential Privacy** | DP-SGD with RDP/PRV/GDP accounting, per-sample clipping |
| **Production Security** | Feature gates, rate limiting, secrets sanitization, audit trails |
| **Privacy Accounting** | Formal (epsilon, delta) privacy budgets with composition theorems |
| **FFA-LoRA** | Federated fine-tuning with secure aggregation |
| **HE-LoRA Microkernel** | Dedicated GPU CKKS backend with compiler and runtime |
| **Multi-tenant Security** | Per-tenant DEK encryption, RBAC, input validation |
| **LoRA Best Practices** | LoRA+, rsLoRA, DoRA, adaptive rank (research-backed) |

---

## 5. Lessons and Recommendations (In-Scope)

These recommendations stay within TenSafe's privacy-preserving mission while adopting SkyRL's proven patterns.

### 5.1 GRPO Advantage Estimation (HIGH PRIORITY)

**What SkyRL does**: GRPO (Group Relative Policy Optimization) normalizes rewards within each prompt group, eliminating the need for a learned critic/value function:

```python
# SkyRL's GRPO (simplified)
for each prompt group:
    advantages = (rewards - mean(rewards)) / std(rewards)
```

**Why it matters for TenSafe**: Our RLVR module currently uses REINFORCE with a running-average baseline and PPO with a full critic. GRPO removes the critic entirely, which:
- **Reduces memory**: No critic model = half the model memory
- **Simplifies DP**: No need to privately train a separate value network
- **Better for HE**: Fewer model components to encrypt/manage
- **Proven effective**: DeepSeek-R1, Qwen, and others use GRPO

**Recommendation**: Add GRPO as a third algorithm in `tensafe/rlvr/algorithms/`, registered through the existing registry. This is a ~200-line addition that slots cleanly into the existing `RLAlgorithm` interface.

### 5.2 Advantage Estimator Registry (MEDIUM PRIORITY)

**What SkyRL does**: Advantage estimators are first-class registered functions, not hardcoded into algorithms:

```python
@AdvantageEstimatorRegistry.register("grpo")
def grpo_advantage(rewards, ...): ...

@AdvantageEstimatorRegistry.register("rloo")
def rloo_advantage(rewards, ...): ...
```

**Why it matters for TenSafe**: Currently, advantage computation is embedded inside each algorithm class. Extracting it into the existing registry system (`tensafe.core.registry`) enables:
- Mixing advantage estimators with different policy losses
- Easier experimentation (swap GRPO for RLOO without touching algorithm code)
- Cleaner separation of concerns

**Recommendation**: Extend `tensafe.core.registry` to support advantage estimator registration alongside reward functions.

### 5.3 Async Rollout Generation (MEDIUM PRIORITY)

**What SkyRL does**: The `FullyAsyncRayPPOTrainer` decouples generation from training with a staleness-bounded buffer. Generation workers submit rollouts to a shared queue; the training loop consumes batches as they arrive.

**Why it matters for TenSafe**: HE-LoRA inference is 9-24x slower than plaintext. This makes synchronous rollout-then-train extremely slow. Async generation would:
- **Hide HE latency**: Train on completed rollouts while new ones generate
- **Increase GPU utilization**: Training GPUs aren't idle during slow HE inference
- **Scale naturally**: Add more inference workers without changing the training loop

**Recommendation**: Implement an async rollout buffer in `tensafe/rlvr/buffers.py` with a simple staleness bound. The existing `TrajectoryBuffer` already has the right structure; adding async producer/consumer semantics is a natural extension.

### 5.4 Environment Abstraction for RLVR (LOW-MEDIUM PRIORITY)

**What SkyRL does**: `skyrl-gym` provides Gymnasium-compatible environments with `init()` / `step()` / `close()` lifecycle, enabling multi-turn RL and diverse task types.

**Why it matters for TenSafe**: Our reward functions are stateless callables (`RewardFn(prompt, response, meta) -> float`). This works for single-turn tasks but cannot support:
- Multi-turn reasoning chains (e.g., chain-of-thought with verification)
- Tool-augmented generation (search, code execution)
- Step-wise reward shaping

**Recommendation**: Introduce a lightweight `Environment` protocol in `tensafe/rlvr/` that wraps the existing `RewardFn` for single-turn cases but can be extended for multi-turn:

```python
class Environment(Protocol):
    def reset(self, prompt: str) -> Observation: ...
    def step(self, action: str) -> StepResult: ...  # (obs, reward, done, info)
```

Single-turn reward functions auto-wrap: `reset()` returns the prompt, `step()` calls the reward function and returns `done=True`.

### 5.5 Weight Sync Patterns for Distributed HE-LoRA (LOW PRIORITY)

**What SkyRL does**: Three-tier weight sync (NCCL broadcast, CUDA IPC, checkpoint) with pause/resume gating on inference engines during updates.

**Why it matters for TenSafe**: When TenSafe scales to multi-GPU HE-LoRA inference, we need weight sync between training workers and HE-LoRA inference engines. SkyRL's pattern of:
1. Extract weights from FSDP-sharded models
2. Batch into chunks with metadata
3. Broadcast via NCCL to inference processes
4. Gate inference during sync

...is directly applicable. The pause/resume protocol is particularly important for HE-LoRA since partially-updated encrypted parameters could produce garbage.

**Recommendation**: When implementing distributed HE-LoRA training, adopt SkyRL's `BroadcastTransferStrategy` pattern with an additional encryption step: sync encrypted LoRA weights rather than plaintext.

### 5.6 Off-Policy Correction for Stale HE Rollouts (LOW PRIORITY)

**What SkyRL does**: Token-level and sequence-level truncated importance sampling (TIS) ratios to correct for policy drift between rollout generation and training:

```python
tis_ratio = clamp(pi_new(a|s) / pi_old(a|s), 1-eps, 1+eps)
```

**Why it matters for TenSafe**: If async rollouts are adopted (5.3), HE-LoRA rollouts will be particularly stale due to slow encrypted inference. Off-policy correction becomes essential to prevent training instability.

**Recommendation**: Implement basic importance-ratio correction in `tensafe/rlvr/algorithms/` when async generation is introduced.

### 5.7 Micro-Batching with Gradient Accumulation (LOW PRIORITY)

**What SkyRL does**: Both policy and critic workers accumulate gradients across configurable micro-batches before optimizer steps, enabling large effective batch sizes.

**Why it matters for TenSafe**: HE operations consume significant GPU memory. Micro-batching allows larger effective batch sizes without OOM, which is critical for DP-SGD (privacy improves with larger batches due to amplification by subsampling).

**Recommendation**: Add configurable micro-batch gradient accumulation to `tensafe/backends/ml_backend.py`, coordinated with the DP accountant.

---

## 6. Anti-Lessons: What NOT to Adopt

### 6.1 Full Megatron Integration

SkyRL supports Megatron tensor/pipeline/context/expert parallelism. This is massive engineering (thousands of lines) designed for frontier-scale training. TenSafe's use case (LoRA fine-tuning, not full pretraining) does not need this. FSDP is sufficient for LoRA weight sharding.

### 6.2 Multiple Inference Engine Backends

SkyRL supports both vLLM and SGLang. TenSafe's HE-LoRA microkernel has deep vLLM integration. Adding SGLang support would require reimplementing the entire HE-LoRA token execution pipeline. Not worth it.

### 6.3 skyrl-agent Multi-Agent Framework

The ReAct/CodeAct agent framework is interesting but orthogonal to TenSafe's mission. Tool-augmented agents don't intersect with privacy-preserving fine-tuning in a meaningful way today.

### 6.4 Hydra Configuration

SkyRL uses Hydra+OmegaConf. TenSafe uses Pydantic+YAML. Migrating would be disruptive with no real benefit -- Pydantic gives us type safety, validation, and env var overrides already.

### 6.5 Colocation/Offload Memory Management

SkyRL's `WorkerDispatch` CPU offload/onload dance is needed because they run separate policy, critic, and reference models on shared GPUs. TenSafe's LoRA approach means we only have one model with small adapter weights. The memory pressure is fundamentally different and doesn't warrant this complexity.

---

## 7. Detailed Comparison Matrix

| Dimension | TenSafe | SkyRL | Gap | Priority |
|---|---|---|---|---|
| **Mission** | Privacy-preserving fine-tuning | High-throughput RL post-training | Different (complementary) | N/A |
| **HE Support** | CKKS, GPU-accelerated, MOAI | None | TenSafe leads | N/A |
| **DP Support** | Full DP-SGD with RDP/PRV/GDP | None | TenSafe leads | N/A |
| **RL Algorithms** | REINFORCE, PPO | PPO, GRPO, RLOO, REINFORCE++, SAPO, CISPO, GSPO | SkyRL leads | High |
| **Advantage Estimation** | Hardcoded in algorithms | Registry (GAE, GRPO, RLOO, REINFORCE++) | SkyRL leads | Medium |
| **Async Training** | None | Fully async with staleness control | SkyRL leads | Medium |
| **Environments** | Stateless reward functions | Gymnasium-compatible multi-turn | SkyRL leads | Low-Med |
| **Distributed** | Single-node + Ray (planned) | Ray-native, FSDP, Megatron | SkyRL leads | Low |
| **Weight Sync** | Checkpoint-based | NCCL, CUDA IPC, checkpoint | SkyRL leads | Low |
| **Off-Policy Correction** | None | TIS ratios, outlier masking | SkyRL leads | Low |
| **LoRA Sophistication** | LoRA+, rsLoRA, DoRA, FFA-LoRA, adaptive rank | Standard PEFT LoRA | TenSafe leads | N/A |
| **Security** | Production gates, rate limiting, audit | Minimal | TenSafe leads | N/A |
| **Configuration** | Pydantic + YAML + env vars | Hydra + dataclass + YAML | Equivalent | N/A |
| **Registry System** | Decorator-based, dotted-path | cloudpickle + Ray distribution | Equivalent | N/A |
| **Inference** | HE-LoRA via vLLM adapter | vLLM + SGLang, load-balanced | Different focus | N/A |

---

## 8. Implementation Roadmap

Ordered by impact-to-effort ratio, scoped to TenSafe's mission:

### Phase 1: Quick Wins (Low Effort, High Impact)

1. **Add GRPO algorithm** to `tensafe/rlvr/algorithms/grpo.py`
   - ~200 lines, slots into existing `RLAlgorithm` interface
   - Removes critic dependency, reduces memory, simplifies DP
   - Register via existing `tensafe.core.registry`

2. **Extract advantage estimation** into registry
   - Move advantage computation from algorithm classes to registered functions
   - Add RLOO as second estimator alongside GRPO's group normalization
   - ~150 lines of refactoring

### Phase 2: Medium-Term (Medium Effort, High Impact)

3. **Async rollout buffer** in `tensafe/rlvr/buffers.py`
   - Extend `TrajectoryBuffer` with async producer/consumer
   - Add staleness bound (max steps between generation and training)
   - Critical for hiding HE-LoRA inference latency
   - ~300-400 lines

4. **Lightweight environment protocol** in `tensafe/rlvr/env.py`
   - `Environment` protocol with `reset()` / `step()`
   - Auto-wrapper for existing `RewardFn` callables
   - Foundation for multi-turn RLVR in the future
   - ~150 lines

### Phase 3: Longer-Term (Higher Effort, Strategic)

5. **Distributed weight sync** for multi-GPU HE-LoRA
   - Adapt SkyRL's broadcast pattern with encryption
   - Pause/resume gating during sync
   - Only needed when scaling beyond single-node

6. **Off-policy correction** for async training
   - Importance ratio computation and clamping
   - Sequence-level outlier masking
   - Only needed after Phase 2 async buffer is in place

7. **Micro-batch gradient accumulation** with DP coordination
   - Configurable micro-batches in ML backend
   - Privacy amplification via subsampling accounting
   - Improves DP-utility tradeoff at scale

---

## Conclusion

SkyRL and TenSafe occupy complementary niches. SkyRL has invested heavily in the **mechanics of RL training at scale** -- async pipelines, algorithm diversity, distributed weight sync, environment abstractions. TenSafe has invested in **privacy-preserving infrastructure** -- HE-LoRA, DP-SGD, security hardening, and LoRA research.

The highest-leverage adoption path is to bring SkyRL's RLVR maturity into TenSafe's existing architecture, starting with GRPO (which directly reduces complexity in the privacy-preserving setting) and async rollouts (which directly addresses HE-LoRA's latency penalty). These improvements strengthen TenSafe's unique position rather than diluting it.
