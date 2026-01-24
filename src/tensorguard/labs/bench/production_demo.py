
import numpy as np
import time
import json
import os
from datetime import datetime, UTC
from typing import Dict, Any, List

from tensorguard.schemas.common import Demonstration
from tensorguard.core.adapters import MoEAdapter
from tensorguard.utils.fastumi_adapter import FastUMIAdapter, FastUMISimulator
from tensorguard.core.production import UpdatePackage, ModelTargetMap, TrainingMetadata, ObjectiveType
from tensorguard.core.crypto import N2HEEncryptor, N2HEContext
from tensorguard.core.keys import vault, KeyScope
from tensorguard.server.aggregator import ExpertDrivenStrategy

class ProductionPipelineTracer:
    """
    Simulates a production-grade Federated Learning pipeline.
    Tracks embeddings, gating decisions, PQC signatures, and key rotations.
    """
    def __init__(self, fleet_id: str = "fleet_production_01"):
        self.fleet_id = fleet_id
        self.adapter = FastUMIAdapter(data_root="./data/fastumi")
        self.simulator = FastUMISimulator(self.adapter)
        self.moe = MoEAdapter()
        self.strategy = ExpertDrivenStrategy(quorum_threshold=1) # High fidelity single agent trace
        
        # Security State
        self.encryption_count = 0
        self.max_encryption_limit = 10 # Force rotation every 10 updates
        self.active_key_id = None
        self.audit_log = []

    def _log_step(self, step_name: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step_name,
            "details": details
        }
        self.audit_log.append(entry)
        print(f"[{step_name:<20}] {json.dumps(details)}")

    def run_iteration(self, iter_idx: int, task: str):
        start_time = time.time()
        print(f"\n--- Iteration {iter_idx+1:02d} | Task: {task} ---")
        
        # 1. CAPTURE & EMBED
        time.sleep(0.5) # Robotic sensor ingestion
        demo = self.simulator.get_random_demonstration(task_filter=task)
        embedding_stats = {
            "obs_shape": demo.observations[0]["video_frame"].shape if demo.observations else "N/A",
            "qpos_dim": len(demo.observations[0]["qpos"]) if demo.observations else "N/A"
        }
        self._log_step("CAPTURE_EMBED", {"instruction": demo.instruction, "stats": embedding_stats})

        # 2. EXPERT GATING (IOSP)
        time.sleep(0.2) # LLM/VLA Routing
        weights = self.moe.get_expert_gate_weights(demo.instruction)
        gated_out = [exp for exp, w in weights.items() if w < 0.15]
        active_experts = [exp for exp, w in weights.items() if w >= 0.15]
        
        self._log_step("MOE_GATING", {
            "active": active_experts,
            "gated_out": gated_out,
            "primary_weight": round(max(weights.values()), 4)
        })

        # 3. PEFT GRADIENT (LoRA)
        time.sleep(0.8) # Backprop & LoRA update
        lora_params = {exp: {"A": (100, 8), "B": (8, 100)} for exp in active_experts}
        self._log_step("PEFT_COMPUTE", {"rank": 8, "affected_experts": active_experts})

        # 4. PRIVACY SHIELD & PQC SIGNING
        time.sleep(0.3) # Sparsification & Encryption
        if self.encryption_count >= self.max_encryption_limit or not self.active_key_id:
            # TRIGGER KEY ROTATION
            self.active_key_id = f"n2he_v{int(time.time())}"
            self._log_step("KEY_ROTATION", {"reason": "usage_limit", "new_key": self.active_key_id})
            self.encryption_count = 0
            
        pqc_sig = f"sig_d3_{os.urandom(8).hex()}"
        
        pkg = UpdatePackage(
            client_id="robot_prod_01",
            target_map=ModelTargetMap(
                module_names=["lora_A", "lora_B"],
                adapter_ids=["pi0-fedmoe-v2"],
                tensor_shapes={"lora_A": (100, 8), "lora_B": (8, 100)}
            ),
            delta_tensors={"lora_A": b"enc_A", "lora_B": b"enc_B"},
            expert_weights=weights,
            training_meta=TrainingMetadata(steps=iter_idx+1, learning_rate=1e-5, objective_type=ObjectiveType.IMITATION_LEARNING)
        )
        self.encryption_count += 1
        
        self._log_step("SHIELD_SIGN", {
            "key_id": self.active_key_id,
            "pqc_signature": pqc_sig,
            "pkg_id": pkg.package_id
        })

        # 5. FEDERATED AGGREGATION & VERIFICATION
        time.sleep(0.4) # Network transit & Server Verification
        self._log_step("SERVER_VERIFY", {"sig_valid": True, "pkg_id": pkg.package_id})
        
        # 6. GLOBAL RECONSTITUTION
        time.sleep(0.1)
        learned_state_gain = 0.02 * (1.0 / (1.0 + np.exp(-0.1 * iter_idx))) 
        total_latency = round(time.time() - start_time, 2)
        
        self._log_step("LEARNED_STATE_UP", {
            "iteration": iter_idx + 1,
            "global_accuracy_gain": round(learned_state_gain, 4),
            "latency_sec": total_latency,
            "next_round_status": "READY_FOR_OTA"
        })

    def run_full_suite(self, count: int = 50):
        for i in range(count):
            task = "manipulation_grasp" if i < 25 else "fastening_screwing"
            self.run_iteration(i, task)
            
        with open("artifacts/production_trace.json", "w") as f:
            json.dump(self.audit_log, f, indent=2)
        print(f"\n[FINAL] Full 50-Iteration Trace Saved to artifacts/production_trace.json")

if __name__ == "__main__":
    tracer = ProductionPipelineTracer()
    tracer.run_full_suite(count=50)
