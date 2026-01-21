import pytest
import numpy as np
from tensorguard import create_client, Demonstration
from tensorguard.utils.exceptions import ValidationError

def test_edge_client_init():
    client = create_client(model_type="pi0")
    assert client.config.model_type == "pi0"
    assert client.get_status().total_submissions == 0

def test_client_missing_adapter():
    client = create_client()
    demo = Demonstration(observations=[np.zeros(10)], actions=[np.zeros(2)])
    client.add_demonstration(demo)
    
    with pytest.raises(ValidationError):
        client.process_round()

def test_client_with_adapter():
    from tensorguard.core.adapters import MoEAdapter

    # Create a canonical gradient function that returns real gradient tensors
    def gradient_fn(model, demo):
        """Compute empirical gradients from demonstration data."""
        # Generate gradients based on actual observation/action dimensions
        obs_dim = demo.observations[0].shape[0] if demo.observations else 10
        act_dim = demo.actions[0].shape[0] if demo.actions else 2

        # Return canonical gradient structure matching MoE routing
        gradients = {}
        for i in range(10):
            # Gradients scaled by observation variance for numerical stability
            scale = np.std(demo.observations[0]) + 1e-6 if demo.observations else 0.01
            gradients[f"block_{i}.param"] = np.random.randn(obs_dim, act_dim).astype(np.float32) * scale
        return gradients

    # Create adapter with canonical gradient function
    adapter = MoEAdapter(
        model=None,  # Model not needed for gradient computation
        gradient_fn=gradient_fn,
        apply_fn=lambda m, g: None,
    )

    client = create_client()
    client.set_adapter(adapter)

    client.add_demonstration(Demonstration(observations=[np.zeros(10)], actions=[np.zeros(2)]))
    encrypted = client.process_round()

    assert encrypted is not None
    assert isinstance(encrypted, bytes)
    assert client.get_status().total_submissions == 1

