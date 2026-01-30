
import argparse
import json
import logging
import os
import secrets
import tarfile
import tempfile
from pathlib import Path
from typing import List

from ..crypto.kem import decap_hybrid, generate_hybrid_keypair
from ..crypto.payload import PayloadDecryptor
from ..crypto.sig import generate_hybrid_sig_keypair
from .format import read_tgsp_header, write_tgsp_package_v1
from .manifest import PackageManifest
from .tar_deterministic import create_deterministic_tar

logger = logging.getLogger(__name__)


class TarExtractionError(Exception):
    """Raised when tar extraction fails due to security checks."""
    pass


def safe_extract_tar(tar: tarfile.TarFile, dest_dir: str, allow_symlinks: bool = False) -> List[str]:
    """
    Safely extract tar archive with path traversal protection.

    This function validates each tar member before extraction to prevent:
    - Path traversal attacks (../../ escapes)
    - Absolute path extraction
    - Symlink/hardlink attacks (optionally)

    Args:
        tar: Open tarfile.TarFile object
        dest_dir: Destination directory for extraction
        allow_symlinks: If False, refuse to extract symlinks/hardlinks

    Returns:
        List of extracted file paths

    Raises:
        TarExtractionError: If a security violation is detected
    """
    dest_path = Path(dest_dir).resolve()
    extracted_files = []

    for member in tar.getmembers():
        # Normalize and resolve the member path
        member_path = Path(dest_path / member.name)

        # Resolve to absolute path (following any .. components)
        try:
            resolved_path = member_path.resolve()
        except (OSError, ValueError) as e:
            raise TarExtractionError(
                f"Invalid path in tar archive: {member.name} - {e}"
            )

        # Check for path traversal: resolved path must be within dest_dir
        try:
            resolved_path.relative_to(dest_path)
        except ValueError:
            raise TarExtractionError(
                f"Path traversal detected: '{member.name}' would extract outside "
                f"destination directory '{dest_dir}'"
            )

        # Check for absolute paths in the archive
        if member.name.startswith('/') or member.name.startswith('\\'):
            raise TarExtractionError(
                f"Absolute path in tar archive: {member.name}"
            )

        # Check for symlinks and hardlinks
        if member.issym() or member.islnk():
            if not allow_symlinks:
                raise TarExtractionError(
                    f"Symlink/hardlink not allowed: {member.name} "
                    f"(type: {'symlink' if member.issym() else 'hardlink'})"
                )
            # Even if allowed, validate the link target
            if member.issym():
                link_target = Path(member_path.parent / member.linkname)
                try:
                    link_resolved = link_target.resolve()
                    link_resolved.relative_to(dest_path)
                except ValueError:
                    raise TarExtractionError(
                        f"Symlink escapes destination: {member.name} -> {member.linkname}"
                    )

        # Check for device files (block/char devices)
        if member.isdev():
            raise TarExtractionError(
                f"Device file not allowed: {member.name}"
            )

        # Safe to extract
        tar.extract(member, dest_dir, set_attrs=True, numeric_owner=False)
        extracted_files.append(str(resolved_path))

    return extracted_files
logging.basicConfig(level=logging.INFO)

def run_keygen(args):
    out = args.out
    os.makedirs(out, exist_ok=True)

    if args.type == "signing":
        pub, priv = generate_hybrid_sig_keypair()
        with open(os.path.join(out, "signing.priv"), "w") as f:
            json.dump(priv, f)
        with open(os.path.join(out, "signing.pub"), "w") as f:
            json.dump(pub, f)
        print(f"Generated Hybrid-Dilithium Signing Key in {out}")

    elif args.type == "encryption":
        pub, priv = generate_hybrid_keypair()
        with open(os.path.join(out, "encryption.priv"), "w") as f:
            json.dump(priv, f)
        with open(os.path.join(out, "encryption.pub"), "w") as f:
            json.dump(pub, f)
        print(f"Generated Hybrid-Kyber Encryption Key in {out}")

def run_build(args):
    # 1. Manifest
    manifest = PackageManifest(
        tgsp_version="1.0",
        package_id=secrets.token_hex(8),
        model_name=args.model_name,
        model_version=args.model_version,
        author_id="cli-user",
        payload_hash="pending"
    )

    # 2. Recipients
    recipients_public_keys = []

    if args.recipients:
        for r_str in args.recipients:
            # format: [optional_label:]path_to_pub_json
            path = r_str
            if ":" in r_str:
                path = r_str.split(":")[-1] # handle C:\ paths or labels
                # If the whole thing exists, use it (maybe no label)
                if not os.path.exists(path) and os.path.exists(r_str):
                    path = r_str

            if os.path.exists(path):
                with open(path) as f:
                    pk = json.load(f)
                recipients_public_keys.append(pk)
            else:
                logger.warning(f"Recipient key not found: {r_str} (resolved to {path})")

    # 3. Payload Stream (Tar)
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        create_deterministic_tar(args.input_dir, tf.name)
        tf_path = tf.name

    # 4. Signing Key
    if args.signing_key:
        with open(args.signing_key) as f:
            sk = json.load(f)
        sk_id = "key_1"
    else:
        raise ValueError("TGSP v1.0 requires signing key (Hybrid PQC)")

    if not args.signing_pub:
        raise ValueError("TGSP v1.0 requires signing public key (Hybrid PQC)")
    with open(args.signing_pub) as f:
        signing_pub = json.load(f)

    # 5. Write Container
    with open(tf_path, "rb") as payload_stream:
        evt = write_tgsp_package_v1(
            args.out,
            manifest,
            payload_stream,
            recipients_public_keys,
            sk,
            signing_pub,
            sk_id,
        )

    from ..evidence.store import get_store
    get_store().add_record("tgsp_build", evt)

    os.unlink(tf_path)
    print(f"TGSP v1.0 Built: {args.out}")
    print(json.dumps(evt, indent=2))

def run_inspect(args):
    data = read_tgsp_header(args.file)
    print(json.dumps(data["header"], indent=2))
    print(f"Manifest Version: {data['manifest'].get('model_version')}")
    print(f"PQC Mode: {data['header']['crypto'].get('kem')}")

def run_open(args):
    data = read_tgsp_header(args.file)

    if not args.key:
        print("Private key required to open")
        return

    with open(args.key) as f:
        sk = json.load(f)

    dek = None
    for rec in data["recipients"]:
        try:
            ss_hybrid = decap_hybrid(sk, rec["encap"])

            from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
            wrapper = rec["wrapper"]
            nonce = bytes.fromhex(wrapper["nonce"])
            ct = bytes.fromhex(wrapper["ct"])

            aead = ChaCha20Poly1305(ss_hybrid)
            dek = aead.decrypt(nonce, ct, None)
            break
        except Exception:
            continue

    if not dek:
        raise ValueError("Failed to decrypt (No matching recipient or invalid key)")

    h = data["header"]
    nonce_base = bytes.fromhex(h["crypto"]["nonce_base"])
    m_hash = h["hashes"]["manifest"]
    r_hash = h["hashes"]["recipients"]

    decryptor = PayloadDecryptor(dek, nonce_base, m_hash, r_hash)

    os.makedirs(args.out_dir, exist_ok=True)
    out_tar = os.path.join(args.out_dir, "payload_decrypted_temp.tar")

    with open(args.file, "rb") as f:
        f.seek(data["payload_offset"])
        total_read = 0
        with open(out_tar, "wb") as out_f:
            while total_read < data["payload_len"]:
                chunk = decryptor.decrypt_chunk_from_stream(f)
                if not chunk: break
                out_f.write(chunk)
                total_read += (4 + len(chunk) + 16)

    with tarfile.open(out_tar, "r") as tr:
        try:
            extracted = safe_extract_tar(tr, args.out_dir, allow_symlinks=False)
            logger.info(f"Safely extracted {len(extracted)} files")
        except TarExtractionError as e:
            os.remove(out_tar)
            raise ValueError(f"Security violation during extraction: {e}")
    os.remove(out_tar)
    print(f"Payload decrypted and extracted to {args.out_dir}")

# --- Compatibility Shims for QA Suite ---

def create_tgsp(args):
    """Shim for tests: maps old Args class to run_build."""
    # Ensure manifest has compat fields if provided
    if not hasattr(args, 'model_name'): args.model_name = "llama-3-8b"
    if not hasattr(args, 'model_version'): args.model_version = "1.0.0"
    if not hasattr(args, 'input_dir'):
        # Extract from payload if needed
        # args.payload filter: ["adapter1:weights:path"]
        if hasattr(args, 'payload') and args.payload:
            p0 = args.payload[0]
            if ":" in p0:
                args.input_dir = os.path.dirname(p0.split(":")[-1])
            else:
                args.input_dir = os.path.dirname(p0)
        else:
            args.input_dir = "."

    if hasattr(args, 'recipient'):
        args.recipients = args.recipient
    else:
        args.recipients = []

    if hasattr(args, 'producer_signing_key'):
        args.signing_key = args.producer_signing_key

    return run_build(args)

def verify_tgsp(args):
    """Shim for tests: maps VerifyArgs to verify_tgsp_container."""
    from .format import verify_tgsp_container
    # In compat mode, we might not have a public key passed,
    # so it fails if not self-signed.
    return verify_tgsp_container(args.in_file)

def decrypt_tgsp(args):
    """Shim for tests: maps DecryptArgs to run_open."""
    if hasattr(args, 'recipient_private_key'):
        args.key = args.recipient_private_key
    if hasattr(args, 'in_file'):
        args.file = args.in_file
    if hasattr(args, 'outdir'):
        args.out_dir = args.outdir
    return run_open(args)

def main():
    parser = argparse.ArgumentParser()
    subps = parser.add_subparsers(dest="cmd")

    kg = subps.add_parser("keygen")
    kg.add_argument("--type", choices=["signing", "encryption"], required=True)
    kg.add_argument("--out", required=True)

    bd = subps.add_parser("build")
    bd.add_argument("--input-dir", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--model-name", default="unknown")
    bd.add_argument("--model-version", default="1.0.0")
    bd.add_argument("--recipients", nargs="+", help="Paths to recipient public key JSONs")
    bd.add_argument("--signing-key", required=True, help="Path to signing private key JSON")
    bd.add_argument("--signing-pub", required=True, help="Path to signing public key JSON")

    ins = subps.add_parser("inspect")
    ins.add_argument("--file", required=True)

    op = subps.add_parser("open")
    op.add_argument("--file", required=True)
    op.add_argument("--key", required=True)
    op.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "keygen": run_keygen(args)
    elif args.cmd == "build": run_build(args)
    elif args.cmd == "inspect": run_inspect(args)
    elif args.cmd == "open": run_open(args)
    else: parser.print_help()

if __name__ == "__main__":
    main()
