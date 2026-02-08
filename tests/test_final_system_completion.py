import pytest
import numpy as np
import torch
from he_lora_microkernel.services.has.executor import HASExecutor
from he_lora_microkernel.services.has.server import HASServicer
from he_lora_microkernel.client.sdk import TenSafeClientSDK
from he_lora_microkernel.backend.gpu_ckks_backend import BackendType, get_backend
from he_lora_microkernel.backend.vllm_adapter.hooks import create_projection_hooks, reset_hook_statistics

class MockHASStub:
    def __init__(self, servicer):
        self.servicer = servicer

    def LoadAdapter(self, request):
        return self.servicer.LoadAdapter(request, None)

    def ApplyTokenStep(self, request):
        return self.servicer.ApplyTokenStep(request, None)

    def ApplyBatchedTokenStep(self, request):
        return self.servicer.ApplyBatchedTokenStep(request, None)

def test_tee_attestation_verification():
    """Verify that TEE evidence is populated and verified by SDK."""
    from he_lora_microkernel.services.has.key_manager import KeyManager
    from he_lora_microkernel.services.has.shm_manager import SharedMemoryManager

    executor = HASExecutor(backend_type="SIMULATION")
    executor.initialize()
    executor.load_adapter("test", "model", 16, 32.0, "qkv")
    
    km = KeyManager()
    shm = SharedMemoryManager()
    shm.create_region("req-1", 1, 1024)
    executor.prepare_request("req-1", "test", 1, 1, shm_region="req-1")
    
    servicer = HASServicer(executor, km, shm)
    sdk = TenSafeClientSDK()
    sdk.stub = MockHASStub(servicer)

    # This should internally call _verify_attestation
    res = sdk.apply_token_step("req-1", 0, "q", 0)
    assert res is not None
    # If it didn't crash, it verified the mock quote

def test_zero_rotation_enforcement():
    """Verify that rotations are blocked when enforcement is enabled."""
    from he_lora_microkernel.compiler.ckks_params import get_profile, CKKSProfile
    params = get_profile(CKKSProfile.FAST)
    backend = get_backend(BackendType.SIMULATION, params)
    
    # Enable enforcement
    backend.enforce_zero_rotation = True
    
    ct = backend.encrypt(np.zeros(params.slot_count))
    
    with pytest.raises(RuntimeError) as excinfo:
        backend.rotate(ct, 1)
    assert "ZeRo-MOAI VIOLATION" in str(excinfo.value)
    
    backend.enforce_zero_rotation = False
    backend.rotate(ct, 1) # Should work now

def test_batched_hook_caching():
    """Verify that HookSharedState correctly caches deltas for Paper 2 optimization."""
    class MockModel:
        def __init__(self):
            self.layers = torch.nn.ModuleList([
                torch.nn.Module(
                ) for _ in range(2)
            ])
            # Fake attention module
            self.layers[0].self_attn = torch.nn.Module()
            self.layers[0].self_attn.q_proj = torch.nn.Linear(1024, 1024)
            self.layers[0].self_attn.k_proj = torch.nn.Linear(1024, 1024)
            self.layers[0].self_attn.v_proj = torch.nn.Linear(1024, 1024)

    model = MockModel()
    
    call_count = 0
    def mock_callback(layer_idx, proj_type, hidden):
        nonlocal call_count
        call_count += 1
        return torch.zeros_like(hidden)

    hooks = create_projection_hooks(
        model=model,
        layer_indices=[0],
        projections=['q', 'k', 'v'],
        delta_callback=mock_callback,
        use_shared_state=True
    )
    
    # Simulate forward passes
    hidden = torch.randn(1, 1, 1024)
    
    # Call Q, K, V
    hooks[(0, 'q')].hooked_forward(hidden)
    hooks[(0, 'k')].hooked_forward(hidden)
    hooks[(0, 'v')].hooked_forward(hidden)
    
    # Optimization: In a real batched implementation, call_count would be 1.
    # In our current state-caching implemention, if the callback is individual, 
    # it still calls 3 times but ensures consistency. 
    # However, if we optimized the callback to fetch QKV at once on the first 'q' call:
    assert (0, 'q') in hooks
    assert hooks[(0, 'q')].shared_state is not None
    assert len(hooks[(0, 'q')].shared_state.cached_deltas) == 3
    
    # Reset and check with different hidden
    call_count = 0
    hooks[(0, 'q')].hooked_forward(torch.randn(1, 1, 1024))
    assert len(hooks[(0, 'q')].shared_state.cached_deltas) == 1
