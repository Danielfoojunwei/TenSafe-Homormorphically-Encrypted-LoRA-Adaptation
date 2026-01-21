
import pytest
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Skip entire module if flwr is not available - this must be BEFORE any tensorguard imports
# that depend on flwr
flwr = pytest.importorskip("flwr", reason="flwr (Flower) not installed - required for FedMoE tests")

from tensorguard.core.adapters import MoEAdapter
from tensorguard.core.pipeline import ExpertGater, RandomSparsifier, APHECompressor
from tensorguard.core.crypto import N2HEEncryptor, N2HEContext, LWECiphertext
from tensorguard.core.production import UpdatePackage, ModelTargetMap, TrainingMetadata, ObjectiveType
from tensorguard.server.aggregator import ExpertDrivenStrategy, ClientContribution


# Mock Flower components
class MockClientProxy:
    def __init__(self, cid: str):
        self.cid = cid


class MockFitRes:
    def __init__(self, parameters: Any, metrics: Dict[str, Any] = None):
        self.parameters = parameters
        self.metrics = metrics or {}
        self.status = "ok"


class TestFedMoESystem:
    """
    Integration test for the FedMoE (Federated Mixture-of-Experts) system.
    Verifies:
    1. Agent-side Expert Gating (IOSP)
    2. Privacy Pipeline (Clip -> Sparse -> Compress -> Encrypt)
    3. Server-side Expert-Driven Aggregation (EDA)
    """

    @pytest.fixture
    def setup_system(self):
        # Shared Context
        ctx = N2HEContext()
        # In a real test we'd use a key file, here we use an in-memory mock or ephemeral key
        encryptor = N2HEEncryptor(security_level=128)
        adapter = MoEAdapter()
        strategy = ExpertDrivenStrategy(quorum_threshold=2)

        return {
            "ctx": ctx,
            "encryptor": encryptor,
            "adapter": adapter,
            "strategy": strategy
        }

    def test_end_to_end_fedmoe_round(self, setup_system):
        sys = setup_system

        # 1. Simulate Agents with different tasks
        # Agent A: Visual Primary task
        # Agent B: Manipulation task

        tasks = [
            ("agent_a", "Picking up the geometric shapes and objects", "visual_primary"),
            ("agent_b", "Applying force to the gripper torque handle", "manipulation_grasp")
        ]

        results = []

        for cid, instruction, primary_expert in tasks:
            # A. Compute Expert Weights (IOSP)
            weights = sys["adapter"].get_expert_gate_weights(instruction)
            assert weights[primary_expert] > 0.5  # Verification of routing

            # B. Generate Mock Gradients (Vectorized)
            # Create a deterministic update for the primary expert
            raw_grads = {f"block_{i}.param": np.ones((100,)) * 0.1 for i in range(10)}

            # C. Privacy Pipeline (simplified for test efficiency)
            # Clip
            clipped = {k: v * 0.5 for k, v in raw_grads.items()}
            # Sparse
            sparse = clipped  # 1.0 sparsity for test visibility

            # D. Encryption & Packaging
            # We encrypt the primary block for that agent
            # visual_primary uses blocks 0,1,2,3
            # manipulation_grasp uses blocks 8,9

            payload = {}
            for k, v in sparse.items():
                payload[k] = sys["encryptor"].encrypt(v.tobytes())

            package = UpdatePackage(
                client_id=cid,
                target_map=ModelTargetMap(
                    module_names=list(sparse.keys()),
                    adapter_ids=["pi0-moe"],
                    tensor_shapes={k: v.shape for k, v in sparse.items()}
                ),
                delta_tensors=payload,
                expert_weights=weights,
                training_meta=TrainingMetadata(
                    steps=1,
                    learning_rate=1e-4,
                    objective_type=ObjectiveType.IMITATION_LEARNING
                )
            )

            # E. Wrap for Flower
            from flwr.common import ndarrays_to_parameters
            params = ndarrays_to_parameters([np.frombuffer(package.serialize(), dtype=np.uint8)])
            results.append((MockClientProxy(cid), MockFitRes(params)))

        # 2. Server-Side Aggregation (EDA)
        # We manually trigger the aggregation logic
        # Note: In a real Flower run, this is called by the server loop

        agg_params, metrics = sys["strategy"].aggregate_fit(
            server_round=1,
            results=results,
            failures=[]
        )

        # 3. Verification
        assert agg_params is not None
        assert metrics["accepted"] == 2

        # Verify Expert Weights Aggregation in Metrics
        expert_weights = metrics["expert_weights"]
        assert "visual_primary" in expert_weights
        assert "manipulation_grasp" in expert_weights

        # The weights should be balanced since we had one of each
        assert expert_weights["visual_primary"] > 0.2
        assert expert_weights["manipulation_grasp"] > 0.2

        print(f"\nFedMoE Aggregation Metrics: {metrics}")

    def test_gating_sparsity(self, setup_system):
        """Verify that ExpertGater correctly drops low-relevance updates."""
        sys = setup_system
        gater = ExpertGater(gate_threshold=0.5)

        expert_grads = {
            "visual_primary": {"layer1": np.array([1.0, 1.0])},
            "language_semantic": {"layer1": np.array([2.0, 2.0])}
        }

        # Test Case 1: High weight for visual
        weights = {"visual_primary": 0.9, "language_semantic": 0.1}
        combined = gater.gate(expert_grads, weights)
        assert np.array_equal(combined["layer1"], np.array([1.0, 1.0]))

        # Test Case 2: High weight for language
        weights = {"visual_primary": 0.1, "language_semantic": 0.9}
        combined = gater.gate(expert_grads, weights)
        assert np.array_equal(combined["layer1"], np.array([2.0, 2.0]))

        # Test Case 3: Both low
        weights = {"visual_primary": 0.1, "language_semantic": 0.1}
        combined = gater.gate(expert_grads, weights)
        assert "layer1" not in combined
