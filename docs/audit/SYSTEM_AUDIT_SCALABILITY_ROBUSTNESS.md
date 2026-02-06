# TenSafe System Audit: Scalability & Robustness

**Auditor:** Claude (Staff-level SRE / Production Readiness Review)
**Date:** 2026-02-06
**Scope:** Full codebase audit — cryptography, distributed training, serving, SDK client, pipeline, privacy accounting
**Severity Levels:** P0 (Critical), P1 (High), P2 (Medium), P3 (Low)

---

## Executive Summary

TenSafe is an ambitious privacy-preserving ML platform combining HE-LoRA, DP-SGD, and PQC. The architecture is sound and well-documented. However, **27 issues** were identified across 7 categories that would block or degrade production reliability at scale. The most critical findings center on:

1. **Incorrect DP-SGD privacy accounting** that silently underreports epsilon (privacy budget), meaning users believe they have more privacy than they actually do
2. **Unsafe secure aggregation** with deterministic seeds that completely compromise the cryptographic protocol
3. **Memory-unbounded caches** in the HE-LoRA runtime that will OOM on long-running inference
4. **No circuit breaker** in the SDK client, leading to cascading failures under partial outages
5. **sys.path manipulation** at import time creating fragile, unreproducible module resolution

---

## Table of Contents

1. [P0 — Critical: Privacy & Cryptographic Correctness](#p0--critical-privacy--cryptographic-correctness)
2. [P1 — High: Scalability & Distributed Systems](#p1--high-scalability--distributed-systems)
3. [P1 — High: SDK Client Resilience](#p1--high-sdk-client-resilience)
4. [P2 — Medium: Memory & Resource Management](#p2--medium-memory--resource-management)
5. [P2 — Medium: Error Handling & Observability Gaps](#p2--medium-error-handling--observability-gaps)
6. [P2 — Medium: Configuration & Build System](#p2--medium-configuration--build-system)
7. [P3 — Low: Code Quality & Maintainability](#p3--low-code-quality--maintainability)

---

## P0 — Critical: Privacy & Cryptographic Correctness

### P0-1: Simplified RDP Accounting Silently Underreports Epsilon

**Files:** `src/tensorguard/distributed/dp_distributed.py:401-410`, `src/tensorguard/distributed/ray_trainer.py:479-484`

**Issue:** Both `_compute_rdp_single_order()` and `_compute_epsilon()` use drastically simplified formulas that **undercount privacy loss**, giving users false confidence about their privacy guarantees.

```python
# dp_distributed.py:410 — WRONG for subsampled Gaussian
def _compute_rdp_single_order(q, sigma, alpha):
    return alpha * q ** 2 / (2 * sigma ** 2)  # Missing higher-order terms

# ray_trainer.py:484 — Not even RDP, just a rough bound
def _compute_epsilon(steps, batch_size, dataset_size, noise_multiplier, delta):
    sampling_rate = batch_size / dataset_size
    return steps * sampling_rate ** 2 / (2 * noise_multiplier ** 2)  # No RDP→DP conversion
```

The correct subsampled Gaussian RDP involves a binomial sum and log-sum-exp computation (which *is* properly implemented in `src/tensafe/privacy/accountants.py:234-313`). But the distributed modules **don't use the correct implementation** — they have their own copy-pasted simplified versions.

**Impact:** Users training with distributed DP-SGD believe they achieved ε=8.0 when the real ε could be 2-5x higher. This is a **compliance and legal liability**.

**Fix:** Delete the simplified functions and import from `tensafe.privacy.accountants.ProductionRDPAccountant`. The production accountant already handles fallback gracefully.

---

### P0-2: DistributedDPOptimizer Uses Batch-Level Clipping Instead of Per-Sample Clipping

**File:** `src/tensorguard/distributed/dp_distributed.py:124-141`

**Issue:** The `_clip_gradients()` method clips the **total gradient norm** across the entire batch, not per-sample gradients. DP-SGD requires per-sample gradient clipping to bound sensitivity.

```python
def _clip_gradients(self):
    total_norm = 0.0
    for group in self.param_groups:
        for p in group['params']:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = math.sqrt(total_norm)
    clip_coef = self.max_grad_norm / (total_norm + 1e-6)
    # This clips the batch gradient, NOT per-sample
```

**Impact:** The DP guarantee **does not hold**. Batch-level clipping does not bound sensitivity — a single outlier sample can dominate the gradient. This fundamentally breaks the mathematical premise of DP-SGD.

**Fix:** Integrate with Opacus `GradSampleModule` for actual per-sample gradient computation, or implement the microbatch approach (process one sample at a time, clip each, then average).

---

### P0-3: Secure Aggregation Uses Deterministic Seeds Derived from Worker IDs

**File:** `src/tensorguard/distributed/dp_distributed.py:240-246`

**Issue:** The pairwise masking seeds are derived as `hash(tuple(sorted([worker_id, other_id]))) % (2**31)`. This is:

1. **Fully deterministic** — any observer who knows the worker IDs (public information) can reconstruct all masks
2. **Uses Python `hash()`** which is not cryptographically secure
3. **No key exchange protocol** — seeds should be established via Diffie-Hellman or similar

```python
def setup_pairwise_masks(self, worker_id: int):
    for other_id in range(self.num_workers):
        pair = tuple(sorted([worker_id, other_id]))
        seed = hash(pair) % (2**31)  # Fully predictable!
```

**Impact:** The "secure aggregation" provides **zero security**. An adversary (including the parameter server) can unmask any individual gradient by computing the same deterministic masks. This negates the entire purpose of secure aggregation.

**Fix:** Implement a proper Diffie-Hellman key agreement or use an existing library (e.g., `cryptography.hazmat.primitives.asymmetric.dh`). Seeds must be established via a cryptographic key exchange that the server cannot observe.

---

### P0-4: PRV Accountant Incorrectly Handles Subsampling

**File:** `src/tensafe/privacy/accountants.py:437-445`

**Issue:** The `_step_external()` method of the PRV accountant adjusts for subsampling by dividing `noise_multiplier` by `sample_rate`:

```python
def _step_external(self):
    subsampled_pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=self.config.noise_multiplier / self.config.sample_rate,
        sensitivity=1.0,
    )
```

This is mathematically incorrect. Subsampling amplification should be applied to the privacy loss distribution, not by scaling the noise. The correct approach is to use `from_randomized_response` or the proper subsampled PLD constructor from dp-accounting. Dividing sigma by q does not produce the correct privacy loss distribution for the subsampled mechanism.

**Impact:** The PRV accountant will either over- or under-report epsilon depending on the sample rate, leading to incorrect privacy budgets.

---

## P1 — High: Scalability & Distributed Systems

### P1-1: Ray Trainer `sys.path.insert(0, ...)` at Module Level

**Files:** `src/tensorguard/distributed/ray_trainer.py:34-35`, `src/tensorguard/backends/vllm/engine.py:33-34`

**Issue:** Multiple modules do `sys.path.insert(0, os.path.dirname(...))` at import time. This:

1. Creates **non-deterministic import resolution** — the same module name may resolve to different files depending on import order
2. Breaks in **containerized/wheel-installed** deployments where the source tree isn't present
3. Makes it **impossible to reliably reproduce** behavior across environments

```python
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
```

**Fix:** Remove all `sys.path` manipulation. Rely on the properly configured `pyproject.toml` package discovery (which already includes `tensorguard*`, `tg_tinker*`, `tensafe*` in `[tool.setuptools.packages.find]`).

---

### P1-2: Ray Trainer Closes Over Full Dataset Object in Worker Function

**File:** `src/tensorguard/distributed/ray_trainer.py:191-198`

**Issue:** `_create_train_func()` captures `self.train_dataset` in the closure of `train_func_per_worker`. When Ray serializes this function to ship to workers, it will **pickle the entire dataset**.

```python
def _create_train_func(self):
    train_dataset = self.train_dataset  # Captured in closure

    def train_func_per_worker(train_loop_config):
        # train_dataset is serialized and sent to each worker
        sampler = DistributedSampler(train_dataset, ...)
```

**Impact:** For a 10GB dataset with 4 workers, this means **40GB of redundant serialization** over the network. For large-scale distributed training (the stated target of 2000+ nodes), this is a fatal bottleneck.

**Fix:** Use Ray Datasets or pass a dataset factory function (e.g., `lambda: load_dataset("path")`) instead of the materialized dataset.

---

### P1-3: DynamicBatchExecutor Doesn't Propagate Weights on Recompile

**File:** `he_lora_microkernel/runtime/batching.py:402-414`

**Issue:** When `_get_executor()` recompiles for a new batch size, it creates a fresh `HELoRAExecutor` without transferring the loaded weights:

```python
def _get_executor(self, batch_size: int):
    if self._executor is None or self._executor_batch_size != batch_size:
        schedule = self._batch_manager.set_batch_size(batch_size)
        self._executor = HELoRAExecutor(schedule, backend_type)
        # Weights are NOT transferred from old executor!
        self._executor_batch_size = batch_size
    return self._executor
```

And `load_weights()` only loads to the *current* executor:

```python
def load_weights(self, A, B, alpha):
    if self._executor is not None:
        self._executor.load_weights(A, B, alpha)
    # If executor is later replaced, weights are lost
```

**Impact:** When batch size changes at runtime (which is the entire purpose of `DynamicBatchExecutor`), the HE-LoRA computation runs with **unloaded weights**, producing zero deltas or raising `ValueError("Weights not loaded")`.

**Fix:** Store `(A, B, alpha)` in the `DynamicBatchExecutor` and re-load after recompilation.

---

### P1-4: LoRAAdapterExecutor Creates Separate Backends Per Adapter

**File:** `he_lora_microkernel/runtime/executor.py:567-588`

**Issue:** Each adapter (Q, K, V, O) gets its own `HELoRAExecutor`, each of which creates its own `GPUCKKSBackend`. For CKKS with `poly_modulus_degree=8192`, each backend allocates ~100MB of key material and NTT tables.

```python
for name, schedule in schedules.items():
    self._executors[name] = HELoRAExecutor(
        schedule, backend_type, device_id, budget
    )
# 4 adapters × 100MB = 400MB of redundant key material
```

**Impact:** With 4 adapters × 32 layers × 100MB = **12.8GB** of key material for a single model. This is catastrophic for the multi-LoRA serving target (50+ adapters per GPU).

**Fix:** Share a single backend across adapters that use the same CKKS parameters. The backend is stateless for encode/encrypt operations.

---

### P1-5: vLLM Engine generate_async Has No Timeout or Cancellation

**File:** `src/tensorguard/backends/vllm/engine.py:333-385`

**Issue:** The async generation methods have no timeout mechanism:

```python
async def generate_async(self, prompts, sampling_params=None):
    # ... submit requests ...
    async for output in engine.generate(None):
        if output.finished:
            # ...
            if len(results) == len(prompts):
                break
    # If a request never finishes, this loops forever
```

**Impact:** A single stuck request (OOM, CUDA error, infinite generation) blocks the entire engine. No timeout means the coroutine never yields, starving other requests.

**Fix:** Add `asyncio.wait_for()` with configurable timeout, and implement request cancellation on timeout.

---

### P1-6: vLLM Engine Hardcoded Model Dimensions

**File:** `src/tensorguard/backends/vllm/engine.py:183, 231-239`

**Issue:** The engine hardcodes `hidden_size=4096` and `32 layers`, which are specific to Llama-3-8B:

```python
hidden_size = 4096  # Default for Llama-3-8B
rank = 16
for layer_idx in range(32):  # Typical transformer layers
```

**Impact:** Any model other than Llama-3-8B will get incorrect HE-LoRA weights. Llama-3-70B (hidden=8192, 80 layers), Mistral (4096, 32), etc., would silently produce wrong results.

**Fix:** Extract model dimensions from the loaded model configuration (`model.config.hidden_size`, `model.config.num_hidden_layers`).

---

## P1 — High: SDK Client Resilience

### P1-7: No Circuit Breaker Pattern in ServiceClient

**File:** `src/tg_tinker/client.py:349-403`

**Issue:** The retry logic retries on timeout/connection errors with exponential backoff, but has **no circuit breaker**. If the server is down, every client instance will exhaust its retry budget on every call, creating:

1. **Thundering herd** — all clients retry simultaneously after backoff
2. **Thread pool exhaustion** — `FutureHandle.result()` spins polling threads that all hit the failing server
3. **No fast-fail** — users wait through full retry cycle even when failure is certain

**Impact:** During a partial outage, the SDK amplifies load on the recovering server instead of shedding it.

**Fix:** Implement a circuit breaker (e.g., track consecutive failures, open circuit after N failures, half-open after cooldown). The `httpx` transport layer supports custom transports.

---

### P1-8: FutureHandle.add_done_callback() Leaks Threads

**File:** `src/tg_tinker/futures.py:233-252`

**Issue:** Each `add_done_callback()` spawns a new daemon thread that polls indefinitely:

```python
def add_done_callback(self, callback):
    def _poll_and_callback():
        while not self.done():
            time.sleep(self._poll_interval)
        callback(self)
    thread = threading.Thread(target=_poll_and_callback, daemon=True)
    thread.start()
```

No limit on concurrent threads. No cleanup. No exception handling in the callback.

**Impact:** A training loop with 10,000 steps, each registering a callback, creates **10,000 polling threads**. Even as daemon threads, the CPU overhead of context-switching will degrade the process.

**Fix:** Use a single polling thread with a callback registry, or use `concurrent.futures.ThreadPoolExecutor` with bounded workers.

---

### P1-9: ServiceClient Retry Sleeps on Main Thread

**File:** `src/tg_tinker/client.py:386-390`

**Issue:** `time.sleep(backoff)` blocks the calling thread during retries. In an async context or GUI, this freezes the event loop.

```python
except (httpx.TimeoutException, httpx.ConnectError) as e:
    if attempt < self._config.retry_count:
        backoff = self._config.retry_backoff * (2**attempt)
        time.sleep(backoff)  # Blocks calling thread
```

**Fix:** Provide an async variant (`AsyncServiceClient`) using `httpx.AsyncClient` and `asyncio.sleep()`. The training loop documentation suggests async usage patterns but the client is synchronous-only.

---

## P2 — Medium: Memory & Resource Management

### P2-1: BatchManager Schedule Cache Is Unbounded

**File:** `he_lora_microkernel/runtime/batching.py:106`

**Issue:** `_schedule_cache: Dict[int, ExecutionSchedule]` grows without bound. Each `ExecutionSchedule` contains packing layouts, cost estimates, and compiled data. If batch sizes vary continuously (common in vLLM's continuous batching), the cache grows indefinitely.

**Fix:** Use `functools.lru_cache` or implement an LRU eviction policy with a max size (e.g., 16 entries).

---

### P2-2: RDP Accountant Stores Full RDP Epsilon Dictionary

**File:** `src/tensafe/privacy/accountants.py:172, 361`

**Issue:** `_rdp_epsilons: Dict[float, float]` stores 153 entries (100 fractional + 52 integer + 3 power-of-2 orders). Every call to `get_privacy_spent()` returns a copy via `dict(self._rdp_epsilons)`.

For a training loop calling `get_privacy_spent()` every step (e.g., for logging), this creates **153-element dict copies per step**. Over 100K steps, that's 15M dict allocations.

**Fix:** Only return the optimal order's epsilon in the common case. Provide `get_privacy_spent(include_rdp_details=False)` as the default.

---

### P2-3: vLLM Engine Metrics Counter Overflow

**File:** `src/tensorguard/backends/vllm/engine.py:113-115`

**Issue:** `_total_requests` and `_total_tokens` are plain Python ints that grow without bound. While Python ints don't overflow, the `get_metrics()` method computes `tokens_per_second` from `_start_time`:

```python
"tokens_per_second": self._total_tokens / uptime if uptime > 0 else 0,
```

After weeks of uptime, this reports a **lifetime average** rather than a useful current throughput. No windowed metrics.

**Fix:** Use a sliding window counter (e.g., last 60 seconds) for rate metrics. Keep cumulative counters for Prometheus but expose windowed rates for dashboards.

---

### P2-4: HELoRAExecutor Token Statistics Are Not Thread-Safe

**File:** `he_lora_microkernel/runtime/executor.py:447-449`

**Issue:** `tokens_processed` and `total_he_time_ms` are incremented without locks:

```python
self._context.total_he_time_ms += elapsed_ms
self._context.tokens_processed += 1
```

In the batched decrypt path (`execute_all_adapters_batched_decrypt`), multiple executors update their contexts concurrently if called from different threads.

**Fix:** Use `threading.Lock` or `atomics` for counter updates, or document that the executor is not thread-safe.

---

## P2 — Medium: Error Handling & Observability Gaps

### P2-5: GDP Accountant Binary Search May Not Converge

**File:** `src/tensafe/privacy/accountants.py:592-598`

**Issue:** The binary search for epsilon in the GDP accountant uses `eps_hi = 100.0` as upper bound and converges when `eps_hi - eps_lo > 1e-6`. For extreme noise parameters, the true epsilon may exceed 100, causing the search to return an incorrect value.

```python
eps_lo, eps_hi = 0.0, 100.0
while eps_hi - eps_lo > 1e-6:
    eps_mid = (eps_lo + eps_hi) / 2
    if compute_delta(eps_mid) > delta:
        eps_lo = eps_mid
    else:
        eps_hi = eps_mid
```

Also, `compute_delta()` uses `math.exp(eps)` which overflows for `eps > ~709`.

**Fix:** Use a wider initial bracket and clamp `eps` before `math.exp()`. Add a convergence check with max iterations.

---

### P2-6: No Structured Logging in Core HE Runtime

**File:** `he_lora_microkernel/runtime/executor.py` (entire file)

**Issue:** The executor has zero logging statements. Failures in encryption, decryption, or level mismatches are only surfaced as exceptions with no prior warning. In production, operators need to see:

- Noise budget consumption rate
- Level chain depth approaching limit
- Cost budget violations

**Fix:** Add structured logging at key checkpoints (encrypt, compute, decrypt) with timing and noise budget info.

---

### P2-7: CORS Allows All Methods Including DELETE

**File:** `src/tensorguard/platform/main.py:205`

**Issue:** CORS configuration allows `DELETE` and `PATCH` methods from any allowed origin:

```python
allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
```

For a privacy-sensitive platform, browser-based DELETE/PATCH access to training clients and artifacts should be more restrictive.

**Fix:** Restrict to `GET, POST, OPTIONS` for browser origins. Only allow destructive methods from server-to-server calls (no CORS needed).

---

### P2-8: Health Check Queries Database on Every Probe

**File:** `src/tensorguard/platform/main.py:234-244`

**Issue:** The `/health` endpoint calls `check_db_health()` on every invocation. With the default Kubernetes health check interval of 30s and 100 pods, that's 200 DB queries/minute just for health checks.

**Fix:** Cache the DB health check result for a short TTL (e.g., 10s). Use `/live` (which is already lightweight) for liveness, and only check DB in `/ready`.

---

## P2 — Medium: Configuration & Build System

### P2-9: numpy Pinned Below 2.0 May Block PyTorch/vLLM

**File:** `pyproject.toml:53`

**Issue:** `numpy>=1.24.0,<2.0.0` pins numpy below 2.0. Modern PyTorch 2.2+ and vLLM require numpy 2.x. This creates a dependency conflict that will manifest at install time when users add `torch` or `vllm` to their environment.

**Fix:** Widen to `numpy>=1.24.0` (or `>=1.24.0,<3.0.0`) and test against numpy 2.x.

---

### P2-10: Dockerfile HEALTHCHECK Uses Shell Expansion in Exec Form

**File:** `Dockerfile:82`

**Issue:** The HEALTHCHECK uses exec form (`CMD [...]`) but contains `${PORT}` shell expansion:

```dockerfile
HEALTHCHECK CMD curl -f http://localhost:${PORT}/health || exit 1
```

This works because `CMD` without brackets is shell form. But it executes through `/bin/sh -c`, which means a shell crash or missing `/bin/sh` fails the health check even if the app is healthy.

This is actually fine as written, but the comment on line 80 says "Health check" which is insufficient documentation for the non-obvious shell form behavior.

---

### P2-11: python-jose Is Unmaintained / Has Known CVEs

**File:** `pyproject.toml:55`

**Issue:** `python-jose[cryptography]>=3.3.0` has been effectively unmaintained since 2022 and has known vulnerabilities (CVE-2024-33663, CVE-2024-33664). The project recommends `PyJWT` or `joserfc` as replacements.

**Fix:** Migrate to `PyJWT>=2.8.0` or `joserfc>=0.9.0`.

---

### P2-12: Config `_coerce_value` Treats "0" as False

**File:** `src/tensafe/core/config.py:688`

**Issue:** The environment variable coercion function treats `"0"` as `False`:

```python
if value.lower() in ("false", "no", "0"):
    return False
```

This means `TENSAFE_TRAINING__BATCH_SIZE=0` would set `batch_size=False` instead of `batch_size=0`. The integer parser below would catch `"0"` first... except `"0"` matches the boolean check first.

**Fix:** Check for boolean strings only for values that look like booleans (i.e., don't match `"0"` — use `"false"` and `"no"` only).

---

## P3 — Low: Code Quality & Maintainability

### P3-1: Duplicate DPConfig Definitions Across 3 Modules

**Files:**
- `src/tensafe/core/config.py:266` — `DPConfig` (dataclass)
- `src/tensafe/privacy/accountants.py:45` — `DPConfig` (dataclass)
- `src/tensorguard/distributed/ray_trainer.py:43` — `DPConfig` (fallback dataclass)

Three different `DPConfig` classes with different fields. The one in `ray_trainer.py` has `enabled: bool` while `accountants.py` does not. Code importing "DPConfig" may get different classes depending on import path.

**Fix:** Consolidate into a single canonical `DPConfig` in `tensafe.core.config` and import everywhere.

---

### P3-2: HE-LoRA Executor `execute_batch` Is Sequential

**File:** `he_lora_microkernel/runtime/executor.py:505-525`

**Issue:** `execute_batch()` processes tokens one at a time in a for loop:

```python
def execute_batch(self, activations_batch, positions=None):
    deltas = []
    for i, activations in enumerate(activations_batch):
        delta = self.execute_token(activations, pos)
        deltas.append(delta)
    return deltas
```

For a batch of 32 tokens, this is 32 sequential encrypt-compute-decrypt cycles. The GPU is idle between each token.

**Fix:** Implement true batch execution by concatenating activations and using a single encrypt/compute/decrypt pass with the SIMD slots in CKKS.

---

### P3-3: Training Config Validates Warnings as Errors Inconsistently

**File:** `src/tensafe/core/config.py:466-518`

**Issue:** `TenSafeConfig.validate()` returns a mixed list of `"Error: ..."` and `"Warning: ..."` strings. The caller in `load_config()` raises on errors but only logs warnings:

```python
for issue in issues:
    if issue.startswith("Error:"):
        raise ValueError(issue)
    else:
        logger.warning(issue)
```

This string-prefix-based error classification is fragile. Adding a typo like `"Erros: ..."` silently becomes a warning.

**Fix:** Return structured results (e.g., `ValidationResult(level=ERROR|WARNING, message=...)` or separate the lists.

---

### P3-4: vLLM Engine request_id Uses `time.time()` Which Is Not Unique

**File:** `src/tensorguard/backends/vllm/engine.py:362, 415`

**Issue:** Request IDs are generated as `f"req-{i}-{time.time()}"`. Two requests submitted in the same millisecond get the same ID. The async generator then uses `request_ids.index(output.request_id)` which returns the **first** match.

**Fix:** Use `uuid.uuid4()` for request IDs.

---

### P3-5: Production Server Launched via subprocess.run()

**File:** `src/tensorguard/platform/main.py:346-374`

**Issue:** `run_production()` launches gunicorn via `subprocess.run()`, which:

1. Loses the ability to handle signals properly (the Python process becomes a dumb shell wrapper)
2. Doesn't inherit environment variables correctly on all platforms
3. The Dockerfile already invokes gunicorn directly, so this codepath is likely dead code

**Fix:** Remove `run_production()` since the Dockerfile handles production execution. Or use `os.execvp()` instead of `subprocess.run()` to replace the process.

---

## Summary Table

| ID | Severity | Category | Component | Status |
|----|----------|----------|-----------|--------|
| P0-1 | Critical | Privacy | dp_distributed, ray_trainer | Open |
| P0-2 | Critical | Privacy | DistributedDPOptimizer | Open |
| P0-3 | Critical | Security | SecureGradientAggregator | Open |
| P0-4 | Critical | Privacy | PRV Accountant | Open |
| P1-1 | High | Build | ray_trainer, vllm engine | Open |
| P1-2 | High | Scalability | Ray Trainer | Open |
| P1-3 | High | Correctness | DynamicBatchExecutor | Open |
| P1-4 | High | Memory | LoRAAdapterExecutor | Open |
| P1-5 | High | Reliability | vLLM Engine | Open |
| P1-6 | High | Correctness | vLLM Engine | Open |
| P1-7 | High | Resilience | SDK ServiceClient | Open |
| P1-8 | High | Resource Leak | SDK FutureHandle | Open |
| P1-9 | High | Usability | SDK ServiceClient | Open |
| P2-1 | Medium | Memory | BatchManager | Open |
| P2-2 | Medium | Performance | RDP Accountant | Open |
| P2-3 | Medium | Observability | vLLM Engine | Open |
| P2-4 | Medium | Thread Safety | HELoRAExecutor | Open |
| P2-5 | Medium | Correctness | GDP Accountant | Open |
| P2-6 | Medium | Observability | HE Runtime | Open |
| P2-7 | Medium | Security | CORS Config | Open |
| P2-8 | Medium | Performance | Health Check | Open |
| P2-9 | Medium | Build | pyproject.toml | Open |
| P2-10 | Medium | Ops | Dockerfile | Open |
| P2-11 | Medium | Security | python-jose | Open |
| P2-12 | Medium | Correctness | Config coercion | Open |
| P3-1 | Low | Maintainability | DPConfig | Open |
| P3-2 | Low | Performance | execute_batch | Open |
| P3-3 | Low | Maintainability | Config validation | Open |
| P3-4 | Low | Correctness | request_id | Open |
| P3-5 | Low | Dead Code | run_production | Open |

---

## Recommended Fix Priority

### Week 1 (P0 — Ship Blockers)
1. Fix DP-SGD per-sample clipping (P0-2)
2. Replace simplified RDP with production accountant (P0-1)
3. Replace deterministic secure aggregation seeds with DH key exchange (P0-3)
4. Fix PRV accountant subsampling (P0-4)

### Week 2 (P1 — Scalability Blockers)
5. Remove sys.path hacks (P1-1)
6. Fix DynamicBatchExecutor weight propagation (P1-3)
7. Share HE backend across adapters (P1-4)
8. Add circuit breaker to SDK (P1-7)
9. Fix FutureHandle thread leak (P1-8)

### Week 3-4 (P2 — Production Hardening)
10. Add timeouts to vLLM async generation (P1-5)
11. Extract model dimensions dynamically (P1-6)
12. Bound schedule caches (P2-1)
13. Replace python-jose (P2-11)
14. Fix numpy version constraint (P2-9)
15. All remaining P2 items

---

## Positive Observations

Despite the issues above, the codebase demonstrates several strong engineering practices:

1. **Production RDP Accountant** (`accountants.py`) — The built-in RDP implementation is mathematically correct with proper log-sum-exp, binomial coefficients via lgamma, and optimal RDP→DP conversion
2. **Security Middleware Stack** — Rate limiting, CSP, input validation, security headers — all configurable via environment variables
3. **Multi-stage Docker Build** — Non-root user, minimal image, proper healthchecks
4. **Comprehensive Documentation** — 93 markdown files covering architecture, operations, compliance
5. **HE-LoRA Compiler** — The MOAI zero-rotation approach and cost model are well-designed
6. **Fused Execution Path** — `execute_token_fused()` eliminating intermediate allocations shows awareness of memory-bound operations
7. **Test Coverage Breadth** — Security invariants, privacy invariants, regression tests, noise budget tests
