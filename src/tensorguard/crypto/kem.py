
from typing import Dict, Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from .pqc.kyber import Kyber768

# Unified KEM Interface

def generate_hybrid_keypair() -> Tuple[Dict, Dict]:
    """
    Generates a Hybrid Keypair (X25519 + Kyber768).
    Returns (pub_dict, priv_dict) serialization-ready bytes/hex.
    """
    # 1. Classical
    priv_c = x25519.X25519PrivateKey.generate()
    pub_c = priv_c.public_key()

    # 2. PQC
    pqc = Kyber768()
    pk_pqc, sk_pqc = pqc.keygen()

    pub = {
        "classic": pub_c.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
        "pqc": pk_pqc.hex(),
        "alg": "Hybrid-Kyber-v1"
    }

    priv = {
        "classic": priv_c.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()).hex(),
        "pqc": sk_pqc.hex(),
        "alg": "Hybrid-Kyber-v1"
    }

    return pub, priv

def encap_hybrid(hybrid_pub: Dict) -> Tuple[bytes, Dict]:
    """
    Hybrid Encapsulation.
    Returns (shared_secret, encapsulation_dict)
    """
    if hybrid_pub["alg"] != "Hybrid-Kyber-v1":
        raise ValueError(f"Unsupported KEM alg: {hybrid_pub.get('alg')}")

    # 1. Classical Encap (Ephemeral ECDH)
    peer_pub_c = x25519.X25519PublicKey.from_public_bytes(bytes.fromhex(hybrid_pub["classic"]))
    ephem_priv_c = x25519.X25519PrivateKey.generate()
    ephem_pub_c = ephem_priv_c.public_key()
    ss_c = ephem_priv_c.exchange(peer_pub_c)

    # 2. PQC Encap
    pqc = Kyber768()
    pk_pqc = bytes.fromhex(hybrid_pub["pqc"])
    ss_pqc, ct_pqc = pqc.encap(pk_pqc)

    # 3. Combine
    # SHA256( ss_c || ss_pqc )
    combiner = hashes.Hash(hashes.SHA256())
    combiner.update(ss_c)
    combiner.update(ss_pqc)
    ss_hybrid = combiner.finalize()

    encap_data = {
        "ek_classic": ephem_pub_c.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
        "ct_pqc": ct_pqc.hex()
    }

    return ss_hybrid, encap_data

def decap_hybrid(hybrid_priv: Dict, encap_data: Dict) -> bytes:
    """
    Hybrid Decapsulation.
    """
    # 1. Classical
    priv_c = x25519.X25519PrivateKey.from_private_bytes(bytes.fromhex(hybrid_priv["classic"]))
    peer_ek_c = x25519.X25519PublicKey.from_public_bytes(bytes.fromhex(encap_data["ek_classic"]))
    ss_c = priv_c.exchange(peer_ek_c)

    # 2. PQC
    pqc = Kyber768()
    sk_pqc = bytes.fromhex(hybrid_priv["pqc"])
    ct_pqc = bytes.fromhex(encap_data["ct_pqc"])
    ss_pqc = pqc.decap(sk_pqc, ct_pqc)

    # 3. Combine
    combiner = hashes.Hash(hashes.SHA256())
    combiner.update(ss_c)
    combiner.update(ss_pqc)
    ss_hybrid = combiner.finalize()

    return ss_hybrid
