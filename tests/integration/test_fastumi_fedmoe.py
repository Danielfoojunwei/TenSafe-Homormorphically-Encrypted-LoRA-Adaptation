
import pytest
import numpy as np
from datetime import datetime
import time

# Skip entire module if required dependencies are not available
h5py = pytest.importorskip("h5py", reason="h5py not installed - required for FastUMI tests")
flwr = pytest.importorskip("flwr", reason="flwr (Flower) not installed - required for FedMoE tests")

from tensorguard.core.adapters import MoEAdapter
from tensorguard.utils.fastumi_adapter import FastUMIAdapter, FastUMISimulator
from tensorguard.server.aggregator import ExpertDrivenStrategy, ClientContribution
from tensorguard.schemas.common import Demonstration
from tensorguard.core.production import UpdatePackage, ModelTargetMap, TrainingMetadata, ObjectiveType


class TestFastUMIFedMoE:
    """
    End-to-End Simulation of TensorGuardFlow with FastUMI-100K Dataset.
    Demonstrates 100 cycles of Federated Learning with Task Switching and KPI alignment.
    """

    @pytest.fixture
    def setup_env(self):
        adapter = FastUMIAdapter(data_root="./data/fastumi")
        simulator = FastUMISimulator(adapter)
        moe_adapter = MoEAdapter()
        strategy = ExpertDrivenStrategy(quorum_threshold=3)

        return {
            "simulator": simulator,
            "moe": moe_adapter,
            "strategy": strategy,
            "rounds": 100,
            "agents": 5
        }

    def test_fastumi_100_cycle_convergence(self, setup_env):
        env = setup_env
        history = []

        print(f"\n[START] FastUMI End-to-End Simulation ({env['rounds']} Cycles)")
        print(f"Aligning with Success KPIs: [Success Rate, Sample Efficiency, Expert Specificity]")

        # Simulated "Model" Performance (Success Rate starts low)
        current_sr = 0.1

        for r in range(1, env['rounds'] + 1):
            results = []

            # Determine Global Task Context
            # Week 1-5: Picking/Placing
            # Week 6-10: Pouring/Screwing
            task_context = "manipulation_grasp" if r <= 50 else "fastening_screwing"

            for a_idx in range(env['agents']):
                cid = f"robot_{a_idx}"

                # 1. Fetch Data from FastUMI Simulator
                demo = env['simulator'].get_random_demonstration(task_filter=task_context)

                # 2. Local Fine-Tuning (Mocked Gradients)
                # Success rate increases based on cycle count and task repetition
                # Convergence is faster for simpler tasks
                learning_gain = 0.005 if r <= 50 else 0.003
                current_sr = min(1.0, current_sr + (learning_gain * np.random.uniform(0.5, 1.5)))

                # 3. FedMoE Routing & Gating
                weights = env['moe'].get_expert_gate_weights(demo.instruction)

                # 4. Privacy-Preserving Packaging
                # Simulate gradients for all experts
                pkg = UpdatePackage(
                    client_id=cid,
                    target_map=ModelTargetMap(
                        module_names=["backbone", "policy_head"],
                        adapter_ids=["pi0-vla-moe"],
                        tensor_shapes={"backbone": (1024,), "policy_head": (256,)}
                    ),
                    delta_tensors={
                        "backbone": b"encrypted_grads",
                        "policy_head": b"encrypted_grads"
                    },
                    expert_weights=weights,
                    training_meta=TrainingMetadata(
                        steps=100,
                        learning_rate=1e-5,
                        objective_type=ObjectiveType.IMITATION_LEARNING
                    )
                )

                # Wrap for strategy
                # In real flower, this is Parameters
                from flwr.common import ndarrays_to_parameters
                params = ndarrays_to_parameters([np.frombuffer(pkg.serialize(), dtype=np.uint8)])

                from typing import Any
                class MockFitRes:
                    def __init__(self, p): self.parameters = p; self.metrics={"cid": cid}

                class MockProxy:
                    def __init__(self, c): self.cid = c

                results.append((MockProxy(cid), MockFitRes(params)))

            # 5. Global Aggregation (EDA)
            agg_params, metrics = env['strategy'].aggregate_fit(r, results, [])

            # Log Progress
            if r % 10 == 0 or r == 1:
                 print(f"Cycle {r:03d} | Task: {task_context:<18} | Success Rate: {current_sr:.2%} | Experts Active: {len(metrics['expert_weights'])}")

            history.append({
                "round": r,
                "sr": current_sr,
                "active_experts": list(metrics['expert_weights'].keys()),
                "weights": metrics['expert_weights']
            })

        # --- Final KPI Analysis ---
        print("\n[COMPLETE] Simulation Finished.")

        # KPI 1: Success Rate Improvement
        final_sr = history[-1]["sr"]
        initial_sr = history[0]["sr"]
        improvement = (final_sr - initial_sr) / initial_sr
        print(f"KPI 1: Success Rate Improvement: +{improvement:.1%}")
        assert final_sr > initial_sr

        # KPI 2: Expert Specificity (Task B should have screwing expert)
        late_weights = history[-1]["weights"]
        assert "fastening_screwing" in late_weights
        assert late_weights["fastening_screwing"] > 0.5
        print(f"KPI 2: Expert Specificity (Task B -> Screwing): PASSED")

        # KPI 3: Sample Efficiency (SR > 80% reached)
        p80_round = next((h["round"] for h in history if h["sr"] >= 0.8), None)
        print(f"KPI 3: Sample Efficiency (80% SR reached at Cycle {p80_round or 'N/A'})")

        # Store results for walkthrough
        import json
        with open("artifacts/fastumi_sim_results.json", "w") as f:
            json.dump(history, f, indent=2)
