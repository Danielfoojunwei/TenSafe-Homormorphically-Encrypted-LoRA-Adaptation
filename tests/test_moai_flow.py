"""
Test MOAI End-to-End Flow
"""
import pytest
import numpy as np

# Skip entire module if tenseal is not available
tenseal = pytest.importorskip("tenseal", reason="tenseal (FHE library) not installed")

from tensorguard.moai.moai_config import MoaiConfig
from tensorguard.moai.keys import MoaiKeyManager
from tensorguard.moai.exporter import MoaiExporter
from tensorguard.moai.encrypt import MoaiEncryptor, MoaiDecryptor
from tensorguard.serving.backend import TenSEALBackend


def test_moai_flow():
    # Setup
    config = MoaiConfig()
    key_manager = MoaiKeyManager("tests/keys_tmp")
    # Use real FHE key generation
    key_id, pk, sk, eval_keys = key_manager.generate_keypair("test-tenant", config)

    # Export
    exporter = MoaiExporter(config)
    # Mocking a model file for the exporter logic
    pack = exporter.export(None, "test-model-v1", ["layer1"])

    # Serve
    backend = TenSEALBackend()
    backend.load_model(pack)

    # Infer
    input_vec = np.random.randn(64).astype(np.float32)
    # TenSEAL context for encryption is stored in pk/sk (which are contexts with/without SK)
    enc = MoaiEncryptor(key_id, pk)
    ct = enc.encrypt_vector(input_vec)

    # In TenSEAL, eval_keys (Galois/Relin) are bundled in the context (pk)
    res_ct = backend.infer(ct, pk)

    dec = MoaiDecryptor(key_id, sk)
    res_vec = dec.decrypt_vector(res_ct)

    assert isinstance(res_vec, np.ndarray)
    # Output dim should match input for our dummy linear layer simulation
    assert len(res_vec) >= len(input_vec)
