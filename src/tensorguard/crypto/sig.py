
from typing import Dict, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .pqc.dilithium import Dilithium3


def generate_hybrid_sig_keypair() -> Tuple[Dict, Dict]:
    # 1. Classical
    priv_c = ed25519.Ed25519PrivateKey.generate()
    pub_c = priv_c.public_key()

    # 2. PQC
    pqc = Dilithium3()
    pk_pqc, sk_pqc = pqc.keygen()

    pub = {
        "classic": pub_c.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
        "pqc": pk_pqc.hex(),
        "alg": "Hybrid-Dilithium-v1"
    }

    priv = {
        "classic": priv_c.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex(),
        "pqc": sk_pqc.hex(),
        "alg": "Hybrid-Dilithium-v1"
    }
    return pub, priv

def sign_hybrid(hybrid_priv: Dict, message: bytes) -> Dict:
    # 1. Classical
    priv_c = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(hybrid_priv["classic"]))
    sig_c = priv_c.sign(message)

    # 2. PQC
    pqc = Dilithium3()
    sk_pqc = bytes.fromhex(hybrid_priv["pqc"])
    sig_pqc = pqc.sign(sk_pqc, message)

    return {
        "sig_classic": sig_c.hex(),
        "sig_pqc": sig_pqc.hex()
    }

def verify_hybrid(hybrid_pub: Dict, message: bytes, signature: Dict) -> bool:
    try:
        # 1. Classical
        pub_bytes = bytes.fromhex(hybrid_pub["classic"])
        pub_c = ed25519.Ed25519PublicKey.from_public_bytes(pub_bytes)
        sig_c = bytes.fromhex(signature["sig_classic"])
        pub_c.verify(sig_c, message)

        # 2. PQC
        pqc = Dilithium3()
        pk_pqc = bytes.fromhex(hybrid_pub["pqc"])
        sig_pqc = bytes.fromhex(signature["sig_pqc"])

        if not pqc.verify(pk_pqc, message, sig_pqc):
            return False

        return True
    except Exception:
        return False
