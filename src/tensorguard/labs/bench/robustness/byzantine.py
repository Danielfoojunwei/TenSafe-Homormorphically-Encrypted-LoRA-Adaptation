"""
Byzantine Robustness & Outlier Detection Benchmark
"""

import time
import numpy as np
import json
import os
from datetime import datetime

from ...core.production import ResilientAggregator, ClientContribution, UpdatePackage

class RobustnessBench:
    def __init__(self, output_dir: str = "artifacts/robustness"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run_byzantine_test(self):
        print("Running Byzantine Robustness Test...")
        
        # Setup Aggregator
        agg = ResilientAggregator(quorum_threshold=3, max_staleness_seconds=60)
        agg.start_round()
        
        # Generate Contributions
        # 3 Good Clients, 2 Bad Clients
        
        # Good Clients (similar updates)
        good_update = np.random.randn(100).astype(np.float32)
        from ...core.crypto import N2HEEncryptor
        
        # We use a real encryptor
        encryptor = N2HEEncryptor(security_level=128)
        
        for i in range(3):
            # Small noise to make them not identical
            noise = np.random.normal(0, 0.1, 100)
            data_raw = good_update + noise
            
            # Real Encryption
            enc_data = encryptor.encrypt(data_raw.tobytes())
            
            # Create Pkg
            pkg = UpdatePackage(
                client_id=f"good_{i}", 
                delta_tensors={"grad": enc_data},
                safety_stats=UpdatePackage.safety_stats_from_gradients(data_raw) if hasattr(UpdatePackage, 'safety_stats_from_gradients') else None
            )
            # Fallback if helper not present
            if pkg.safety_stats is None:
                from ...core.production import SafetyStatistics
                pkg.safety_stats = SafetyStatistics(grad_norm_max=float(np.linalg.norm(data_raw, ord=np.inf)))

            agg.add_contribution(ClientContribution(f"good_{i}", pkg, datetime.utcnow()))
            
        # Bad Client 1: Sign Flip (Maliciously low/normal norm but bad content)
        bad_data_1 = -1 * good_update 
        enc_bad_1 = encryptor.encrypt(bad_data_1.tobytes())
        pkg1 = UpdatePackage(client_id="bad_flip", delta_tensors={"grad": enc_bad_1})
        from ...core.production import SafetyStatistics
        pkg1.safety_stats = SafetyStatistics(grad_norm_max=float(np.linalg.norm(bad_data_1, ord=np.inf)))
        agg.add_contribution(ClientContribution("bad_flip", pkg1, datetime.utcnow()))
        
        # Bad Client 2: Massive Scaling (Easy to detect via norm)
        bad_data_2 = good_update * 50.0
        enc_bad_2 = encryptor.encrypt(bad_data_2.tobytes())
        pkg2 = UpdatePackage(client_id="bad_noise", delta_tensors={"grad": enc_bad_2})
        pkg2.safety_stats = SafetyStatistics(grad_norm_max=float(np.linalg.norm(bad_data_2, ord=np.inf)))
        agg.add_contribution(ClientContribution("bad_noise", pkg2, datetime.utcnow()))
        
        # Detect Outliers
        t0 = time.time()
        outliers = agg.detect_outliers()
        dt = time.time() - t0
        
        detected_bad = [c for c in outliers if "bad" in c]
        detected_good = [c for c in outliers if "good" in c]
        
        results = {
            "total_clients": 5,
            "expected_outliers": ["bad_flip", "bad_noise"],
            "detected_outliers": outliers,
            "false_positives": len(detected_good),
            "detection_time_sec": dt,
            "success": len(detected_bad) == 2 and len(detected_good) == 0
        }
        
        print(f"  Detetected: {outliers}")
        
        with open(os.path.join(self.output_dir, "byzantine_results.json"), "w") as f:
            json.dump(results, f, indent=2)
            
        print("Robustness Benchmark Complete.")

def run_robustness(args):
    bench = RobustnessBench()
    bench.run_byzantine_test()
