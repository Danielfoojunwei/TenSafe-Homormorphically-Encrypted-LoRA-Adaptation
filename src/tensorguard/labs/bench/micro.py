"""
Microbenchmarks for TensorGuard
Measures:
- N2HE Encryption Latency/Throughput
- Serialization Overhead
- Key Rotation
"""

import time
import numpy as np
import json
import os
import psutil
from typing import Dict, Any

from ..core.crypto import N2HEEncryptor
from ..core.production import UpdatePackage, ModelTargetMap

class MicroBenchmark:
    def __init__(self, output_dir: str = "artifacts/metrics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def log(self, test_name: str, metrics: Dict[str, Any]):
        """Log a benchmark result."""
        entry = {
            "timestamp": time.time(),
            "test_id": test_name,
            "metrics": metrics,
            "system_cpu_percent": psutil.cpu_percent(),
            "process_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }
        self.results.append(entry)
        print(f"[{test_name}] {json.dumps(metrics, indent=2)}")

    def run_crypto_bench(self, tensor_size_mb: float = 0.1):
        """Benchmark N2HE encryption."""
        print(f"Running Crypto Bench (Tensor Size: {tensor_size_mb}MB)...")
        
        # Setup
        # Create a dummy tensor of approx requested size (float32 = 4 bytes)
        num_elements = int((tensor_size_mb * 1024 * 1024) // 4)
        data = np.random.randn(num_elements).astype(np.float32)
        
        # Initialize Encryptor
        # Using a dummy key path for bench
        encryptor = N2HEEncryptor("bench_key.npy", security_level=128)
        
        # Measure Encrypt
        latencies = []
        for _ in range(10): # Warmup + Stats
            t0 = time.time()
            _ = encryptor.encrypt(data)
            latencies.append(time.time() - t0)
            
        # Stats
        latencies = np.array(latencies)
        self.log("n2he_encrypt", {
            "tensor_size_mb": tensor_size_mb,
            "p50_latency_sec": float(np.percentile(latencies, 50)),
            "p95_latency_sec": float(np.percentile(latencies, 95)),
            "throughput_mb_sec": tensor_size_mb / np.mean(latencies)
        })

    def run_serialization_bench(self):
        """Benchmark UpdatePackage serialization."""
        print("Running Serialization Bench...")
        
        # Create dummy package
        pkg = UpdatePackage(
            client_id="bench_client",
            delta_tensors={"grad": b"0"*1024*1024} # 1MB dummy
        )
        
        # Measure Serialize
        t0 = time.time()
        serialized = pkg.serialize()
        ser_time = time.time() - t0
        
        # Measure Deserialize
        t0 = time.time()
        _ = UpdatePackage.deserialize(serialized)
        deser_time = time.time() - t0
        
        self.log("serialization", {
            "serialize_sec": ser_time,
            "deserialize_sec": deser_time,
            "package_size_bytes": len(serialized)
        })
        
    def save(self):
        filename = os.path.join(self.output_dir, f"micro_bench_{int(time.time())}.jsonl")
        with open(filename, 'w') as f:
            for r in self.results:
                f.write(json.dumps(r) + "\n")
        print(f"Saved results to {filename}")

def run_micro(args):
    bench = MicroBenchmark()
    bench.run_crypto_bench()
    bench.run_serialization_bench()
    bench.save()
