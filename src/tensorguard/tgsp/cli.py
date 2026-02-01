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
            raise TarExtractionError(f"Invalid path in tar archive: {member.name} - {e}")

        # Check for path traversal: resolved path must be within dest_dir
        try:
            resolved_path.relative_to(dest_path)
        except ValueError:
            raise TarExtractionError(
                f"Path traversal detected: '{member.name}' would extract outside destination directory '{dest_dir}'"
            )

        # Check for absolute paths in the archive
        if member.name.startswith("/") or member.name.startswith("\\"):
            raise TarExtractionError(f"Absolute path in tar archive: {member.name}")

        # Check for symlinks and hardlinks
        if member.issym() or member.islnk():
            if not allow_symlinks:
                raise TarExtractionError(
                    f"Symlink/hardlink not allowed: {member.name} (type: {'symlink' if member.issym() else 'hardlink'})"
                )
            # Even if allowed, validate the link target
            if member.issym():
                link_target = Path(member_path.parent / member.linkname)
                try:
                    link_resolved = link_target.resolve()
                    link_resolved.relative_to(dest_path)
                except ValueError:
                    raise TarExtractionError(f"Symlink escapes destination: {member.name} -> {member.linkname}")

        # Check for device files (block/char devices)
        if member.isdev():
            raise TarExtractionError(f"Device file not allowed: {member.name}")

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
        payload_hash="pending",
    )

    # 2. Recipients
    recipients_public_keys = []

    if args.recipients:
        for r_str in args.recipients:
            # format: [optional_label:]path_to_pub_json
            path = r_str
            if ":" in r_str:
                path = r_str.split(":")[-1]  # handle C:\ paths or labels
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
                if not chunk:
                    break
                out_f.write(chunk)
                total_read += 4 + len(chunk) + 16

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
    if not hasattr(args, "model_name"):
        args.model_name = "llama-3-8b"
    if not hasattr(args, "model_version"):
        args.model_version = "1.0.0"
    if not hasattr(args, "input_dir"):
        # Extract from payload if needed
        # args.payload filter: ["adapter1:weights:path"]
        if hasattr(args, "payload") and args.payload:
            p0 = args.payload[0]
            if ":" in p0:
                args.input_dir = os.path.dirname(p0.split(":")[-1])
            else:
                args.input_dir = os.path.dirname(p0)
        else:
            args.input_dir = "."

    if hasattr(args, "recipient"):
        args.recipients = args.recipient
    else:
        args.recipients = []

    if hasattr(args, "producer_signing_key"):
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
    if hasattr(args, "recipient_private_key"):
        args.key = args.recipient_private_key
    if hasattr(args, "in_file"):
        args.file = args.in_file
    if hasattr(args, "outdir"):
        args.out_dir = args.outdir
    return run_open(args)


def run_convert(args):
    """Convert a LoRA adapter to TGSP format."""
    from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

    converter = LoRAToTGSPConverter(
        auto_generate_keys=args.auto_generate_keys,
        keys_dir=args.keys_dir if hasattr(args, 'keys_dir') and args.keys_dir else None,
    )

    try:
        result = converter.convert(
            input_path=args.input,
            output_path=args.out,
            model_name=args.model_name,
            model_version=args.model_version,
            signing_key_path=args.signing_key if hasattr(args, 'signing_key') and args.signing_key else None,
            signing_pub_path=args.signing_pub if hasattr(args, 'signing_pub') and args.signing_pub else None,
            recipient_pub_path=args.recipient if hasattr(args, 'recipient') and args.recipient else None,
            validate=not args.skip_validation if hasattr(args, 'skip_validation') else True,
        )

        if result.success:
            print(f"Conversion successful!")
            print(f"  Output: {result.output_path}")
            print(f"  Adapter ID: {result.adapter_id}")
            print(f"  Model: {result.model_name} v{result.model_version}")
            print(f"  Input format: {result.input_format.value}")
            print(f"  LoRA rank: {result.lora_config.rank}")
            print(f"  Target modules: {result.lora_config.target_modules}")
            print(f"  Input size: {result.input_size_bytes:,} bytes")
            print(f"  Output size: {result.output_size_bytes:,} bytes")
            print(f"  Time: {result.conversion_time_ms:.2f}ms")
            print(f"\nCryptographic info:")
            print(f"  Manifest hash: {result.manifest_hash[:16]}...")
            print(f"  Payload hash: {result.payload_hash[:16]}...")
            print(f"  Signature key: {result.signature_key_id}")
        else:
            print(f"Conversion failed: {result.error}")
            return 1

    finally:
        converter.cleanup()

    return 0


def run_batch_convert(args):
    """Convert multiple LoRA adapters to TGSP format."""
    from tensafe.lora_to_tgsp_converter import LoRAToTGSPConverter

    converter = LoRAToTGSPConverter(
        auto_generate_keys=args.auto_generate_keys,
        keys_dir=args.keys_dir if hasattr(args, 'keys_dir') and args.keys_dir else None,
    )

    try:
        results = converter.batch_convert(
            input_paths=args.inputs,
            output_dir=args.out_dir,
            model_version=args.model_version,
            signing_key_path=args.signing_key if hasattr(args, 'signing_key') and args.signing_key else None,
            signing_pub_path=args.signing_pub if hasattr(args, 'signing_pub') and args.signing_pub else None,
            recipient_pub_path=args.recipient if hasattr(args, 'recipient') and args.recipient else None,
        )

        success_count = sum(1 for r in results if r.success)
        print(f"\nBatch conversion complete: {success_count}/{len(results)} succeeded")

        for result in results:
            status = "OK" if result.success else f"FAILED: {result.error}"
            print(f"  {result.output_path}: {status}")

    finally:
        converter.cleanup()

    return 0 if all(r.success for r in results) else 1


def main():
    parser = argparse.ArgumentParser(
        description="TGSP (TensorGuard Secure Package) CLI - Manage encrypted LoRA adapters"
    )
    subps = parser.add_subparsers(dest="cmd")

    # Key generation
    kg = subps.add_parser("keygen", help="Generate cryptographic keys")
    kg.add_argument("--type", choices=["signing", "encryption"], required=True)
    kg.add_argument("--out", required=True)

    # Build TGSP from directory
    bd = subps.add_parser("build", help="Build TGSP package from input directory")
    bd.add_argument("--input-dir", required=True)
    bd.add_argument("--out", required=True)
    bd.add_argument("--model-name", default="unknown")
    bd.add_argument("--model-version", default="1.0.0")
    bd.add_argument("--recipients", nargs="+", help="Paths to recipient public key JSONs")
    bd.add_argument("--signing-key", required=True, help="Path to signing private key JSON")
    bd.add_argument("--signing-pub", required=True, help="Path to signing public key JSON")

    # Convert LoRA to TGSP
    cv = subps.add_parser("convert", help="Convert LoRA adapter to TGSP format")
    cv.add_argument("--input", "-i", required=True, help="Path to LoRA adapter file or directory")
    cv.add_argument("--out", "-o", required=True, help="Output TGSP file path")
    cv.add_argument("--model-name", help="Model name (auto-detected if not provided)")
    cv.add_argument("--model-version", default="1.0.0", help="Model version")
    cv.add_argument("--signing-key", help="Path to signing private key JSON")
    cv.add_argument("--signing-pub", help="Path to signing public key JSON")
    cv.add_argument("--recipient", help="Path to recipient encryption public key JSON")
    cv.add_argument("--keys-dir", help="Directory for keys (default: ~/.tensafe/keys)")
    cv.add_argument("--auto-generate-keys", action="store_true", default=True,
                    help="Auto-generate missing keys (default: true)")
    cv.add_argument("--no-auto-generate-keys", dest="auto_generate_keys", action="store_false",
                    help="Do not auto-generate keys")
    cv.add_argument("--skip-validation", action="store_true",
                    help="Skip LoRA weight validation")

    # Batch convert
    bcv = subps.add_parser("batch-convert", help="Convert multiple LoRA adapters to TGSP")
    bcv.add_argument("--inputs", "-i", nargs="+", required=True, help="Paths to LoRA adapters")
    bcv.add_argument("--out-dir", "-o", required=True, help="Output directory for TGSP files")
    bcv.add_argument("--model-version", default="1.0.0", help="Model version")
    bcv.add_argument("--signing-key", help="Path to signing private key JSON")
    bcv.add_argument("--signing-pub", help="Path to signing public key JSON")
    bcv.add_argument("--recipient", help="Path to recipient encryption public key JSON")
    bcv.add_argument("--keys-dir", help="Directory for keys")
    bcv.add_argument("--auto-generate-keys", action="store_true", default=True,
                    help="Auto-generate missing keys")

    # Inspect TGSP
    ins = subps.add_parser("inspect", help="Inspect TGSP package header")
    ins.add_argument("--file", required=True)

    # Open/decrypt TGSP
    op = subps.add_parser("open", help="Open and decrypt TGSP package")
    op.add_argument("--file", required=True)
    op.add_argument("--key", required=True)
    op.add_argument("--out-dir", required=True)

    args = parser.parse_args()
    if args.cmd == "keygen":
        run_keygen(args)
    elif args.cmd == "build":
        run_build(args)
    elif args.cmd == "convert":
        return run_convert(args)
    elif args.cmd == "batch-convert":
        return run_batch_convert(args)
    elif args.cmd == "inspect":
        run_inspect(args)
    elif args.cmd == "open":
        run_open(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
