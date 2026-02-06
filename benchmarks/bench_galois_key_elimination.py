#!/usr/bin/env python3
"""
E2E Benchmark: Galois Key Elimination — BEFORE vs AFTER vs HYBRID

Three modes compared:
  BEFORE:  galois_steps=[1,2,4,8,16,32,64,128,256,512,1024]  (11 hardcoded keys)
  AFTER:   galois_steps=[]  (0 keys, hardcoded empty — current fix)
  HYBRID:  galois_steps=scheduler.required_galois_keys  (dynamic, 0 for single-block
           MOAI, >0 only when hidden_size requires multi-block packing)

The HYBRID approach queries the compiler's scheduler for the exact Galois keys
the computation will actually consume, adapting automatically to the model config.
"""

import json
import math
import os
import sys
import time
import tracemalloc
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Benchmark config
# ---------------------------------------------------------------------------
KEYGEN_RUNS = 50
INIT_RUNS = 20
DELTA_TRIALS = 50
DELTA_WARMUP = 5
LORA_RANK = 16
LORA_ALPHA = 32.0
BATCH_SIZE = 2

# Three scenarios: single-block (h=768) and multi-block (h=4096, batch=1)
CONFIGS = {
    "single_block": {"hidden_dim": 768,  "batch_size": 2, "label": "h=768, b=2"},
    "multi_block":  {"hidden_dim": 4096, "batch_size": 1, "label": "h=4096, b=1"},
}

BEFORE_GALOIS_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Size of a realistic mock Galois key (N=16384, L=4 primes, 8 bytes each)
REALISTIC_KEY_SIZE = 16384 * 4 * 8  # 512 KB per key


class RealisticMockBackend:
    """Mock HE backend that generates realistically-sized keys."""
    def generate_secret_key(self) -> bytes:
        return np.random.bytes(REALISTIC_KEY_SIZE)
    def generate_public_key(self, sk: bytes) -> bytes:
        return np.random.bytes(REALISTIC_KEY_SIZE)
    def generate_relin_keys(self, sk: bytes) -> bytes:
        return np.random.bytes(REALISTIC_KEY_SIZE)
    def generate_galois_key(self, sk: bytes, step: int) -> bytes:
        return np.random.bytes(REALISTIC_KEY_SIZE)


@dataclass
class BenchResult:
    label: str
    mode: str = ""
    hidden_dim: int = 0
    batch_size: int = 0
    galois_key_count: int = 0
    keygen_mean_ms: float = 0.0
    keygen_std_ms: float = 0.0
    keygen_memory_bytes: int = 0
    server_init_mean_ms: float = 0.0
    server_init_std_ms: float = 0.0
    server_init_memory_bytes: int = 0
    scheduler_rotations: int = -1
    scheduler_galois_keys_needed: int = -1
    delta_mean_ms: float = 0.0
    delta_std_ms: float = 0.0
    delta_min_ms: float = 0.0
    delta_max_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Scheduler query: returns exactly the Galois keys the computation needs
# ---------------------------------------------------------------------------
def get_scheduler_galois_keys(hidden_dim: int, batch_size: int) -> List[int]:
    from he_lora_microkernel.compiler.scheduler import compile_schedule, ScheduleStrategy
    from he_lora_microkernel.compiler.ckks_params import get_fast_profile, CKKSProfile
    from he_lora_microkernel.compiler.lora_ir import LoRAConfig, LoRATargets

    ckks = get_fast_profile()
    config = LoRAConfig(
        hidden_size=hidden_dim, rank=LORA_RANK, alpha=LORA_ALPHA,
        targets=LoRATargets.QKV, batch_size=batch_size,
        max_context_length=2048, ckks_profile=CKKSProfile.FAST,
    )
    schedule = compile_schedule(config, ckks, ScheduleStrategy.MOAI_CPMM)
    return list(schedule.rotation_schedule.required_galois_keys)


def get_scheduler_info(hidden_dim: int, batch_size: int) -> Dict[str, Any]:
    from he_lora_microkernel.compiler.scheduler import compile_schedule, ScheduleStrategy
    from he_lora_microkernel.compiler.ckks_params import get_fast_profile, CKKSProfile
    from he_lora_microkernel.compiler.lora_ir import LoRAConfig, LoRATargets

    ckks = get_fast_profile()
    config = LoRAConfig(
        hidden_size=hidden_dim, rank=LORA_RANK, alpha=LORA_ALPHA,
        targets=LoRATargets.QKV, batch_size=batch_size,
        max_context_length=2048, ckks_profile=CKKSProfile.FAST,
    )
    schedule = compile_schedule(config, ckks, ScheduleStrategy.MOAI_CPMM)
    return {
        "rotations": schedule.rotation_schedule.total_rotations,
        "galois_keys_needed": len(schedule.rotation_schedule.required_galois_keys),
        "galois_keys": list(schedule.rotation_schedule.required_galois_keys),
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def bench_key_manager(galois_steps: List[int]) -> Dict[str, Any]:
    from he_lora_microkernel.services.has.key_manager import KeyManager
    times = []
    galois_count = 0
    for _ in range(KEYGEN_RUNS):
        manager = KeyManager(enable_audit_log=False, allow_mock=True)
        t0 = time.perf_counter()
        manager.initialize(backend=RealisticMockBackend(), galois_steps=galois_steps)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        galois_count = len(manager.get_available_galois_steps())
        manager.clear_keys()

    tracemalloc.start()
    s1 = tracemalloc.take_snapshot()
    manager = KeyManager(enable_audit_log=False, allow_mock=True)
    manager.initialize(backend=RealisticMockBackend(), galois_steps=galois_steps)
    s2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    m1 = sum(s.size for s in s1.statistics("filename"))
    m2 = sum(s.size for s in s2.statistics("filename"))
    manager.clear_keys()

    return {
        "keygen_mean_ms": float(np.mean(times)),
        "keygen_std_ms": float(np.std(times)),
        "keygen_memory_bytes": max(0, m2 - m1),
        "galois_key_count": galois_count,
    }


def bench_server_init(galois_steps: List[int]) -> Dict[str, Any]:
    from he_lora_microkernel.services.has.executor import HASExecutor
    from he_lora_microkernel.services.has.key_manager import KeyManager
    from he_lora_microkernel.services.has.shm_manager import SharedMemoryManager

    times = []
    for _ in range(INIT_RUNS):
        t0 = time.perf_counter()
        executor = HASExecutor(backend_type="SIMULATION")
        executor.initialize()
        km = KeyManager(enable_audit_log=False, allow_mock=True)
        km.initialize(backend=RealisticMockBackend(), galois_steps=galois_steps)
        shm = SharedMemoryManager()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        km.clear_keys(); executor.shutdown(); shm.shutdown()

    tracemalloc.start()
    s1 = tracemalloc.take_snapshot()
    executor = HASExecutor(backend_type="SIMULATION")
    executor.initialize()
    km = KeyManager(enable_audit_log=False, allow_mock=True)
    km.initialize(backend=RealisticMockBackend(), galois_steps=galois_steps)
    shm = SharedMemoryManager()
    s2 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    m1 = sum(s.size for s in s1.statistics("filename"))
    m2 = sum(s.size for s in s2.statistics("filename"))
    km.clear_keys(); executor.shutdown(); shm.shutdown()

    return {
        "server_init_mean_ms": float(np.mean(times)),
        "server_init_std_ms": float(np.std(times)),
        "server_init_memory_bytes": max(0, m2 - m1),
    }


def bench_e2e_delta(galois_steps: List[int], hidden_dim: int, batch_size: int) -> Dict[str, Any]:
    from he_lora_microkernel.services.has.key_manager import KeyManager

    km = KeyManager(enable_audit_log=False, allow_mock=True)
    km.initialize(backend=RealisticMockBackend(), galois_steps=galois_steps)

    A = np.random.randn(LORA_RANK, hidden_dim).astype(np.float64) * 0.01
    B = np.random.randn(hidden_dim, LORA_RANK).astype(np.float64) * 0.01
    scaling = LORA_ALPHA / LORA_RANK

    for _ in range(DELTA_WARMUP):
        x = np.random.randn(batch_size, hidden_dim).astype(np.float64)
        _ = scaling * (x @ A.T @ B.T)

    times = []
    for _ in range(DELTA_TRIALS):
        x = np.random.randn(batch_size, hidden_dim).astype(np.float64)
        t0 = time.perf_counter()
        ct_x = x + np.random.randn(*x.shape) * 1e-7
        ct_delta = scaling * (ct_x @ A.T @ B.T)
        _ = ct_delta + np.random.randn(*ct_delta.shape) * 1e-7
        _ = km.get_available_galois_steps()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    km.clear_keys()
    return {
        "delta_mean_ms": float(np.mean(times)),
        "delta_std_ms": float(np.std(times)),
        "delta_min_ms": float(np.min(times)),
        "delta_max_ms": float(np.max(times)),
    }


def run_benchmark(galois_steps: List[int], mode: str, hidden_dim: int, batch_size: int, label: str) -> BenchResult:
    result = BenchResult(label=label, mode=mode, hidden_dim=hidden_dim, batch_size=batch_size)

    km = bench_key_manager(galois_steps)
    result.keygen_mean_ms = km["keygen_mean_ms"]
    result.keygen_std_ms = km["keygen_std_ms"]
    result.keygen_memory_bytes = km["keygen_memory_bytes"]
    result.galois_key_count = km["galois_key_count"]

    si = bench_server_init(galois_steps)
    result.server_init_mean_ms = si["server_init_mean_ms"]
    result.server_init_std_ms = si["server_init_std_ms"]
    result.server_init_memory_bytes = si["server_init_memory_bytes"]

    sched = get_scheduler_info(hidden_dim, batch_size)
    result.scheduler_rotations = sched["rotations"]
    result.scheduler_galois_keys_needed = sched["galois_keys_needed"]

    e2e = bench_e2e_delta(galois_steps, hidden_dim, batch_size)
    result.delta_mean_ms = e2e["delta_mean_ms"]
    result.delta_std_ms = e2e["delta_std_ms"]
    result.delta_min_ms = e2e["delta_min_ms"]
    result.delta_max_ms = e2e["delta_max_ms"]

    return result


# ---------------------------------------------------------------------------
# 3-way comparison table
# ---------------------------------------------------------------------------
def print_3way(before: BenchResult, after: BenchResult, hybrid: BenchResult, config_label: str):
    W = 105
    print("\n" + "=" * W)
    print(f"  BEFORE vs AFTER vs HYBRID  [{config_label}]")
    print("=" * W)

    def fv(v, fmt=".3f"):
        return f"{v:{fmt}}" if isinstance(v, float) else str(v)

    def pct(base, val):
        if isinstance(base, (int, float)) and isinstance(val, (int, float)) and base > 0:
            p = (val - base) / base * 100
            return f"{p:+.1f}%"
        return "N/A"

    hdr = f"  {'Metric':<32} {'BEFORE':>14} {'AFTER':>14} {'HYBRID':>14} {'AFT vs BEF':>11} {'HYB vs BEF':>11}"
    print(hdr)
    print("  " + "-" * (W - 2))

    def row(name, b, a, h, unit="", fmt=".3f"):
        bv = fv(b, fmt)
        av = fv(a, fmt)
        hv = fv(h, fmt)
        pa = pct(b, a)
        ph = pct(b, h)
        print(f"  {name:<32} {bv:>12}{unit:>2} {av:>12}{unit:>2} {hv:>12}{unit:>2} {pa:>11} {ph:>11}")

    print()
    row("Galois keys generated", before.galois_key_count, after.galois_key_count, hybrid.galois_key_count, "", "d")
    row("Scheduler rotations needed", before.scheduler_rotations, after.scheduler_rotations, hybrid.scheduler_rotations, "", "d")
    row("Scheduler Galois keys needed", before.scheduler_galois_keys_needed, after.scheduler_galois_keys_needed, hybrid.scheduler_galois_keys_needed, "", "d")
    print()
    row("Key generation time", before.keygen_mean_ms, after.keygen_mean_ms, hybrid.keygen_mean_ms, "ms")
    row("Key generation memory", before.keygen_memory_bytes, after.keygen_memory_bytes, hybrid.keygen_memory_bytes, " B", "d")
    print()
    row("Server init time", before.server_init_mean_ms, after.server_init_mean_ms, hybrid.server_init_mean_ms, "ms")
    row("Server init memory", before.server_init_memory_bytes, after.server_init_memory_bytes, hybrid.server_init_memory_bytes, " B", "d")
    print()
    row("E2E delta latency", before.delta_mean_ms, after.delta_mean_ms, hybrid.delta_mean_ms, "ms")
    row("E2E delta std", before.delta_std_ms, after.delta_std_ms, hybrid.delta_std_ms, "ms")
    print()
    print("  " + "=" * (W - 2))

    # Speedup summary row
    kg_a = before.keygen_mean_ms / after.keygen_mean_ms if after.keygen_mean_ms > 0 else float('inf')
    kg_h = before.keygen_mean_ms / hybrid.keygen_mean_ms if hybrid.keygen_mean_ms > 0 else float('inf')
    si_a = before.server_init_mean_ms / after.server_init_mean_ms if after.server_init_mean_ms > 0 else float('inf')
    si_h = before.server_init_mean_ms / hybrid.server_init_mean_ms if hybrid.server_init_mean_ms > 0 else float('inf')
    ms_a = before.keygen_memory_bytes - after.keygen_memory_bytes
    ms_h = before.keygen_memory_bytes - hybrid.keygen_memory_bytes

    print(f"\n  Keygen speedup:     AFTER={kg_a:.2f}x   HYBRID={kg_h:.2f}x")
    print(f"  Server init speedup: AFTER={si_a:.2f}x   HYBRID={si_h:.2f}x")
    print(f"  Memory saved:        AFTER={ms_a:,} B   HYBRID={ms_h:,} B")

    # Characterize hybrid behavior
    if hybrid.galois_key_count == 0:
        print(f"\n  HYBRID generated 0 Galois keys (single-block: scheduler confirmed 0 rotations)")
    else:
        print(f"\n  HYBRID generated {hybrid.galois_key_count} Galois keys (multi-block: scheduler"
              f" needs {hybrid.scheduler_rotations} rotations for cross-block accumulation)")
    print()


def main():
    all_results = {}

    for cfg_name, cfg in CONFIGS.items():
        h = cfg["hidden_dim"]
        b = cfg["batch_size"]
        clabel = cfg["label"]

        print(f"\n{'='*60}")
        print(f"  CONFIG: {clabel}  (rank={LORA_RANK})")
        print(f"{'='*60}")

        # Determine hybrid Galois keys from scheduler
        hybrid_galois = get_scheduler_galois_keys(h, b)
        sched_info = get_scheduler_info(h, b)

        print(f"  Scheduler says: {sched_info['rotations']} rotations, "
              f"{sched_info['galois_keys_needed']} Galois keys needed: {hybrid_galois}")

        # --- BEFORE ---
        print(f"\n  Running BEFORE (11 hardcoded Galois keys)...")
        before = run_benchmark(BEFORE_GALOIS_STEPS, "BEFORE", h, b,
                               f"BEFORE [{clabel}]")
        print(f"    keygen={before.keygen_mean_ms:.3f}ms  init={before.server_init_mean_ms:.3f}ms  "
              f"delta={before.delta_mean_ms:.3f}ms  mem={before.keygen_memory_bytes:,}B")

        # --- AFTER ---
        print(f"  Running AFTER (0 hardcoded Galois keys)...")
        after = run_benchmark([], "AFTER", h, b,
                              f"AFTER [{clabel}]")
        print(f"    keygen={after.keygen_mean_ms:.3f}ms  init={after.server_init_mean_ms:.3f}ms  "
              f"delta={after.delta_mean_ms:.3f}ms  mem={after.keygen_memory_bytes:,}B")

        # --- HYBRID ---
        print(f"  Running HYBRID (scheduler-driven: {len(hybrid_galois)} Galois keys)...")
        hybrid = run_benchmark(hybrid_galois, "HYBRID", h, b,
                               f"HYBRID [{clabel}]")
        print(f"    keygen={hybrid.keygen_mean_ms:.3f}ms  init={hybrid.server_init_mean_ms:.3f}ms  "
              f"delta={hybrid.delta_mean_ms:.3f}ms  mem={hybrid.keygen_memory_bytes:,}B")

        print_3way(before, after, hybrid, clabel)

        all_results[cfg_name] = {
            "config": {"hidden_dim": h, "batch_size": b, "rank": LORA_RANK,
                       "alpha": LORA_ALPHA, "hybrid_galois_steps": hybrid_galois},
            "before": before.to_dict(),
            "after": after.to_dict(),
            "hybrid": hybrid.to_dict(),
        }

    # Save all results
    output_path = os.path.join(PROJECT_ROOT, "benchmarks", "galois_key_elimination_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "benchmark_params": {
                "keygen_runs": KEYGEN_RUNS, "init_runs": INIT_RUNS,
                "delta_trials": DELTA_TRIALS, "delta_warmup": DELTA_WARMUP,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n  All results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
