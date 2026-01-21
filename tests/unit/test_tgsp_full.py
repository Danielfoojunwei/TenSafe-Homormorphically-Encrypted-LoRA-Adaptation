
import pytest
import os
import shutil
import json
from unittest.mock import patch, MagicMock

# Check if liboqs is available (required for PQC signing)
try:
    from tensorguard.crypto.pqc.dilithium import Dilithium3
    Dilithium3()  # Try to instantiate
    LIBOQS_AVAILABLE = True
except (ImportError, Exception):
    LIBOQS_AVAILABLE = False

# Import main only if dependencies are available
if LIBOQS_AVAILABLE:
    from tensorguard.tgsp.cli import main


@pytest.fixture
def tgsp_test_env(tmp_path):
    """Setup temp env for TGSP tests."""
    input_dir = tmp_path / "model_input"
    input_dir.mkdir()
    (input_dir / "weights.bin").write_bytes(b"Simulated Weights" * 100)
    (input_dir / "config.json").write_text('{"type":"vla"}')
    
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    
    return input_dir, out_dir, keys_dir

@pytest.mark.skipif(not LIBOQS_AVAILABLE, reason="liboqs (PQC) not installed")
def test_full_flow_v02(tgsp_test_env):
    input_dir, out_dir, keys_dir = tgsp_test_env
    
    # 1. Keygen
    with patch("sys.argv", ["tgsp", "keygen", "--type", "signing", "--out", str(keys_dir)]):
        main()
        
    with patch("sys.argv", ["tgsp", "keygen", "--type", "encryption", "--out", str(keys_dir / "fleet1")]):
        main()
        
    signing_key = str(keys_dir / "signing.priv")
    fleet_pub = str(keys_dir / "fleet1" / "encryption.pub")
    fleet_priv = str(keys_dir / "fleet1" / "encryption.priv")
    
    tgsp_file = str(out_dir / "model_v02.tgsp")
    
    # 2. Build v0.2
    # --recipients fleet:f1:<pub>
    cmd = [
        "tgsp", "build",
        "--input-dir", str(input_dir),
        "--out", tgsp_file,
        "--model-name", "robot-v1",
        "--recipients", f"fleet:f1:{fleet_pub}",
        "--signing-key", signing_key
    ]
    with patch("sys.argv", cmd):
        main()
        
    assert os.path.exists(tgsp_file)
    
    # 3. Inspect
    inspect_cmd = ["tgsp", "inspect", "--file", tgsp_file]
    with patch("sys.argv", inspect_cmd):
        main()
        
    # 4. Open
    extract_dir = out_dir / "extracted_v02"
    extract_dir.mkdir()
    open_cmd = [
        "tgsp", "open",
        "--file", tgsp_file,
        "--key", fleet_priv,
        "--out-dir", str(extract_dir)
    ]
    with patch("sys.argv", open_cmd):
        main()
        
    # Verify Content - automated extraction untars and removes payload.tar
    assert (extract_dir / "weights.bin").exists()
    assert (extract_dir / "config.json").exists()
    assert not (extract_dir / "payload.tar").exists()
    # Check Evidence exists
    # It writes to artifacts/evidence by default.
    # We can check global store or file existence if path known.
    
@pytest.mark.skipif(not LIBOQS_AVAILABLE, reason="liboqs (PQC) not installed")
def test_full_flow_v03_hpke(tgsp_test_env):
    input_dir, out_dir, keys_dir = tgsp_test_env
    
    # 1. Keygen
    with patch("sys.argv", ["tgsp", "keygen", "--type", "signing", "--out", str(keys_dir)]):
        main()
    with patch("sys.argv", ["tgsp", "keygen", "--type", "encryption", "--out", str(keys_dir / "fleet_hpke")]):
        main()

    fleet_pub = str(keys_dir / "fleet_hpke" / "encryption.pub")
    fleet_priv = str(keys_dir / "fleet_hpke" / "encryption.priv")
    
    tgsp_file = str(out_dir / "model_v03.tgsp")
    
    # 2. Build v0.3
    cmd = [
        "tgsp", "build",
        "--input-dir", str(input_dir),
        "--out", tgsp_file,
        "--recipients", f"fleet:hpke:{fleet_pub}",
        "--signing-key", str(keys_dir / "signing.priv") # HPKE flow still needs signing key in v1 logic
    ]
    with patch("sys.argv", cmd):
        main()
        
    assert os.path.exists(tgsp_file)
    
    # 3. Open v0.3 (Direct)
    extract_dir = out_dir / "extracted_v03"
    extract_dir.mkdir()
    open_cmd = [
        "tgsp", "open",
        "--file", tgsp_file,
        "--key", fleet_priv,
        "--out-dir", str(extract_dir)
    ]
    with patch("sys.argv", open_cmd):
        main()
        
    assert (extract_dir / "weights.bin").exists()
    assert (extract_dir / "config.json").exists()
    assert not (extract_dir / "payload.tar").exists()
