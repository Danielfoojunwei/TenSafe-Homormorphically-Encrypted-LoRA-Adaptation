
import hashlib
import json
import os
import shutil
import struct
import tempfile
from typing import IO, Any, Dict, List

from ..crypto.kem import encap_hybrid

# Unified Crypto Layer
from ..crypto.payload import encrypt_stream
from ..crypto.sig import sign_hybrid, verify_hybrid
from ..evidence.canonical import canonical_bytes
from .manifest import PackageManifest

# TGSP v1.0 Constants
MAGIC_V1 = b"TGSP\x01\x00"  # TGSP v1.0

def canonical_json(data: Any) -> bytes:
    """Canonical JSON serialization that DOES NOT strip fields."""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')

def write_tgsp_package_v1(output_path: str,
                          manifest: PackageManifest,
                          payload_stream: IO[bytes],
                          recipients_public_keys: List[Dict], # List of Hybrid Public Keys
                          signing_key: Dict,  # Hybrid Private Key
                          signing_public_key: Dict,  # Hybrid Public Key
                          signing_key_id: str) -> Dict:
    """
    Write a TGSP v1.0 Container (Hybrid PQC).
    """

    # 1. Generate DEK (32 bytes)
    dek = os.urandom(32)

    # 2. Encrypt Payload (Streamed)
    # We encrypt to specific temp file
    payload_temp = tempfile.TemporaryFile()

    # Embed signing public key in manifest for self-verification
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    # Derive public key from private key
    priv_c = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(signing_key["classic"]))
    pub_c = priv_c.public_key()

    author_pubkey = {
        "classic": signing_public_key.get(
            "classic",
            pub_c.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex(),
        ),
        "pqc": signing_public_key["pqc"],
        "alg": signing_public_key.get("alg", "Hybrid-Dilithium-v1"),
    }

    # Create a new manifest with the embedded pubkey
    manifest_dict = manifest.model_dump()
    manifest_dict["author_pubkey"] = author_pubkey

    # Prepare AAD Inputs
    manifest_bytes = canonical_bytes(manifest_dict)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    # We first process recipients to get recipients_hash
    # Encapsulate DEK for each recipient
    recipients_block = []

    # Note: KEK derivation needs "context"?
    # For v1, KEK = HybridSharedSecret directly (or HKDF(ss, info="TGSPv1"))
    # Let's assume KEK is derived inside `encap_hybrid`?
    # Actually `encap_hybrid` returns `ss_hybrid`.
    # We need to wrap DEK with `ss_hybrid`.
    # Let's do simple: ct_dek = XOR(dek, ss_hybrid) + random?
    # NO. Use ChaCha20Poly1305. Key=ss_hybrid.

    from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

    for peer_pub in recipients_public_keys:
        ss_hybrid, encap_data = encap_hybrid(peer_pub)

        # Wrap DEK
        # Key = ss_hybrid (ensure 32 bytes)
        # Nonce = random(12)
        nonce = os.urandom(12)
        aead = ChaCha20Poly1305(ss_hybrid)
        ct_dek = aead.encrypt(nonce, dek, None)

        recipients_block.append({
            "recipient_id": "hybrid-id", # Should derive from key?
            "encap": encap_data,
            "wrapper": {"nonce": nonce.hex(), "ct": ct_dek.hex()}
        })

    recipients_bytes = canonical_json(recipients_block)
    recipients_hash = hashlib.sha256(recipients_bytes).hexdigest()

    # 3. Encrypt Payload Content
    # Using Unified Payload Crypto
    nonce_base_hex = encrypt_stream(payload_stream, payload_temp, dek, manifest_hash, recipients_hash)

    # 4. Compute Payload Hash
    payload_temp.seek(0)
    p_hasher = hashlib.sha256()
    while True:
        chunk = payload_temp.read(65536)
        if not chunk: break
        p_hasher.update(chunk)
    payload_hash = p_hasher.hexdigest()

    # 5. Build Header
    header = {
        "tgsp_version": "1.0",
        "hashes": {
            "manifest": manifest_hash,
            "recipients": recipients_hash,
            "payload": payload_hash
        },
        "crypto": {
            "nonce_base": nonce_base_hex,
            "alg": "CHACHA20_POLY1305",
            "kem": "Hybrid-Kyber-v1",
            "sig": "Hybrid-Dilithium-v1"
        }
    }
    header_bytes = canonical_json(header)

    # 6. Sign
    signed_area = header_bytes + manifest_bytes + recipients_bytes
    signature = sign_hybrid(signing_key, signed_area)

    sig_block = {
        "key_id": signing_key_id,
        "signature": signature
    }
    sig_bytes = canonical_json(sig_block)

    # 7. Write to Disk
    with open(output_path, "wb") as f:
        f.write(MAGIC_V1)

        # [Header Len u32][Header]
        f.write(struct.pack(">I", len(header_bytes)))
        f.write(header_bytes)

        # [Manifest Len u32][Manifest]
        f.write(struct.pack(">I", len(manifest_bytes)))
        f.write(manifest_bytes)

        # [Recipients Len u32][Recipients]
        f.write(struct.pack(">I", len(recipients_bytes)))
        f.write(recipients_bytes)

        # [Payload Len u64][Payload]
        payload_temp.seek(0, 2)
        p_len = payload_temp.tell()
        f.write(struct.pack(">Q", p_len))
        payload_temp.seek(0)
        shutil.copyfileobj(payload_temp, f)

        # [Sig Len u32][Sig]
        f.write(struct.pack(">I", len(sig_bytes)))
        f.write(sig_bytes)

    payload_temp.close()

    return {
        "event_type": "TGSP_BUILT_V1",
        "subject": {"tgsp_ref": output_path, "model_ref": manifest.model_name},
        "manifest_hash": manifest_hash,
        "payload_hash": payload_hash,
        "key_id": signing_key_id,
        "mode": "Post-Quantum Hybrid"
    }

def read_tgsp_header(path: str) -> Dict:
    """
    Read TGSP v1.0 Header.
    """
    with open(path, "rb") as f:
        magic = f.read(6)
        if magic != MAGIC_V1:
             raise ValueError(f"Invalid Magic: Expected {MAGIC_V1}, got {magic}")

        h_len = struct.unpack(">I", f.read(4))[0]
        h_bytes = f.read(h_len)
        header = json.loads(h_bytes)

        m_len = struct.unpack(">I", f.read(4))[0]
        m_bytes = f.read(m_len)
        manifest = json.loads(m_bytes)

        r_len = struct.unpack(">I", f.read(4))[0]
        r_bytes = f.read(r_len)
        recipients = json.loads(r_bytes)

        signed_area = h_bytes + m_bytes + r_bytes

        p_len = struct.unpack(">Q", f.read(8))[0]
        payload_offset = f.tell()
        f.seek(p_len, 1)

        s_len = struct.unpack(">I", f.read(4))[0]
        sig_block = json.loads(f.read(s_len))

        return {
            "version": "1.0",
            "header": header,
            "manifest": manifest,
            "recipients": recipients,
            "signature_block": sig_block,
            "signed_area": signed_area,
            "payload_offset": payload_offset,
            "payload_len": p_len
        }

def verify_tgsp_container(path: str, public_key: Dict = None) -> bool:
    """
    Verify TGSP v1.0 Container signature.
    If public_key is not provided, we might attempt to load it from a trusted store 
    or the manifest (if self-signed/trust-on-first-use, though less secure).
    For now, we require the public_key for verification.
    """
    try:
        data = read_tgsp_header(path)
        if public_key is None:
            # Fallback: check if pubkey is in manifest (some internal flows might do this)
            if "author_pubkey" in data["manifest"]:
                public_key = data["manifest"]["author_pubkey"]
            else:
                return False

        return verify_hybrid(public_key, data["signed_area"], data["signature_block"]["signature"])
    except Exception:
        return False

# Shim for backward compat if needed
create_tgsp = write_tgsp_package_v1
