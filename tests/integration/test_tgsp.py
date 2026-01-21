import unittest
import os
import shutil
import base64
import json
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519, x25519

# Check if liboqs is available (required for PQC signing)
try:
    from tensorguard.crypto.pqc.dilithium import Dilithium3
    Dilithium3()
    LIBOQS_AVAILABLE = True
except (ImportError, Exception):
    LIBOQS_AVAILABLE = False


@pytest.mark.skipif(not LIBOQS_AVAILABLE, reason="liboqs (PQC) not installed")
class TestTGSP(unittest.TestCase):
    def setUp(self):
        from tensorguard.tgsp import manifest, crypto, cli, spec
        self.cli = cli

        self.test_dir = "tmp_tgsp_test"
        os.makedirs(self.test_dir, exist_ok=True)

        # Use Hybrid PQC Keys for v1.0 compatibility
        from tensorguard.crypto.sig import generate_hybrid_sig_keypair
        from tensorguard.crypto.kem import generate_hybrid_keypair

        # Signing Key (Hybrid Dilithium)
        pub_sig, priv_sig = generate_hybrid_sig_keypair()
        self.signing_key_path = os.path.join(self.test_dir, "producer.priv")
        with open(self.signing_key_path, 'w') as f:
            json.dump(priv_sig, f)

        # Recipient Key (Hybrid Kyber)
        pub_kem, priv_kem = generate_hybrid_keypair()
        self.recipient_priv_path = os.path.join(self.test_dir, "recipient.priv")
        self.recipient_pub_path = os.path.join(self.test_dir, "recipient.pub")
        with open(self.recipient_priv_path, 'w') as f:
            json.dump(priv_kem, f)
        with open(self.recipient_pub_path, 'w') as f:
            json.dump(pub_kem, f)

        # Dummy assets
        self.adapter_path = os.path.join(self.test_dir, "adapter.bin")
        with open(self.adapter_path, 'wb') as f:
            f.write(b"EXTREMELY_SECRET_ADAPTER_WEIGHTS")

        self.policy_path = os.path.join(self.test_dir, "policy.yaml")
        with open(self.policy_path, 'w') as f:
            f.write("id: test-policy\nversion: 1.0.0")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_flow(self):
        tgsp_file = os.path.join(self.test_dir, "test.tgsp")

        # 1. Create
        class Args:
            out = tgsp_file
            producer_signing_key = self.signing_key_path
            compat_base_model_id = ["llama-3-8b"]
            payload = ["adapter1:weights:" + self.adapter_path]
            policy = self.policy_path
            recipient = ["user1:" + self.recipient_pub_path]
            evidence_report = None
            evidence_html = None

        self.cli.create_tgsp(Args())
        self.assertTrue(os.path.exists(tgsp_file))

        # 2. Verify
        class VerifyArgs:
            in_file = tgsp_file
        self.assertTrue(self.cli.verify_tgsp(VerifyArgs()))

        # 3. Decrypt
        out_dir = os.path.join(self.test_dir, "out")
        class DecryptArgs:
            in_file = tgsp_file
            recipient_id = "user1"
            recipient_private_key = self.recipient_priv_path
            outdir = out_dir

        self.cli.decrypt_tgsp(DecryptArgs())

        recovered_path = os.path.join(out_dir, "adapter.bin")
        self.assertTrue(os.path.exists(recovered_path))
        with open(recovered_path, 'rb') as f:
            self.assertEqual(f.read(), b"EXTREMELY_SECRET_ADAPTER_WEIGHTS")

    def test_tamper_detection(self):
        tgsp_file = os.path.join(self.test_dir, "tamper.tgsp")

        class Args:
            out = tgsp_file
            producer_signing_key = self.signing_key_path
            compat_base_model_id = ["llama-3-8b"]
            payload = ["adapter1:weights:" + self.adapter_path]
            policy = self.policy_path
            recipient = ["user1:" + self.recipient_pub_path]
            evidence_report = None
            evidence_html = None

        self.cli.create_tgsp(Args())

        self.cli.create_tgsp(Args())

        # Tamper Binary bytes
        with open(tgsp_file, 'rb') as f:
            data = bytearray(f.read())

        # Flip a bit in the middle of the file (likely payload or recipients)
        if len(data) > 200:
            data[150] ^= 0xFF

        with open(tgsp_file, 'wb') as f:
            f.write(data)

        class VerifyArgs:
            in_file = tgsp_file
        # Verification should fail due to hash mismatch or AEAD tag error (if we decrypted, but here verify checks hash)
        self.assertFalse(self.cli.verify_tgsp(VerifyArgs()))


if __name__ == "__main__":
    unittest.main()
