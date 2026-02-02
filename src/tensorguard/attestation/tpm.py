"""
TPM 2.0 Attestation Provider.

Provides hardware-based attestation using TPM 2.0.
"""

import base64
import hashlib
import logging
import os
import secrets
import struct
import time
from datetime import datetime, timedelta
from typing import Optional

from .provider import (
    AttestationProvider,
    AttestationError,
    AttestationVerificationError,
    AttestationType,
    AttestationQuote,
    AttestationResult,
    QuoteType,
    VerificationPolicy,
)

logger = logging.getLogger(__name__)


class TPMAttestationProvider(AttestationProvider):
    """
    TPM 2.0 Attestation Provider.

    Uses the system TPM for hardware-based attestation.

    Requirements:
    - TPM 2.0 hardware or emulator (swtpm)
    - tpm2-tools installed
    - Appropriate permissions (tss group membership)

    For production:
    - Real TPM 2.0 hardware
    - TPM-backed attestation keys

    For development:
    - Software TPM (swtpm) is acceptable
    """

    # Default PCRs for attestation
    DEFAULT_PCRS = [0, 1, 2, 3, 4, 5, 6, 7]  # BIOS, firmware, boot config

    # TPM device paths
    TPM_DEVICE_PATHS = ["/dev/tpm0", "/dev/tpmrm0"]

    def __init__(
        self,
        device_path: Optional[str] = None,
        tcti: Optional[str] = None,
        use_software_tpm: bool = False,
    ):
        """
        Initialize TPM attestation provider.

        Args:
            device_path: Path to TPM device (auto-detected if None)
            tcti: TPM Command Transmission Interface (e.g., "device:/dev/tpm0")
            use_software_tpm: Use software TPM for development
        """
        self._device_path = device_path
        self._tcti = tcti
        self._use_software_tpm = use_software_tpm
        self._tpm_available = False

        # Check TPM availability
        self._check_tpm_availability()

        if self._tpm_available:
            logger.info(f"TPM attestation provider initialized (device: {self._device_path})")
        else:
            logger.warning("TPM not available - using software fallback (NOT FOR PRODUCTION)")

    @property
    def attestation_type(self) -> AttestationType:
        return AttestationType.TPM

    @property
    def is_available(self) -> bool:
        return self._tpm_available

    def _check_tpm_availability(self) -> None:
        """Check if TPM hardware is available."""
        # Try specified device path
        if self._device_path:
            if os.path.exists(self._device_path):
                self._tpm_available = True
                return

        # Try default device paths
        for path in self.TPM_DEVICE_PATHS:
            if os.path.exists(path):
                self._device_path = path
                self._tpm_available = True
                return

        # Check for software TPM
        if self._use_software_tpm:
            try:
                # Try to use swtpm via socket
                self._tcti = "mssim:host=localhost,port=2321"
                self._tpm_available = True
                logger.info("Using software TPM (swtpm)")
                return
            except Exception:
                pass

        self._tpm_available = False

    def _run_tpm_command(self, command: list[str]) -> tuple[int, bytes, bytes]:
        """
        Run a TPM command using tpm2-tools.

        Args:
            command: Command and arguments

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        import subprocess

        env = os.environ.copy()
        if self._tcti:
            env["TPM2TOOLS_TCTI"] = self._tcti

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                env=env,
                timeout=30,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            raise AttestationError("TPM command timed out")
        except FileNotFoundError:
            raise AttestationError(
                "tpm2-tools not installed. Install with: apt install tpm2-tools"
            )

    def generate_quote(
        self,
        nonce: Optional[bytes] = None,
        pcr_selection: Optional[list[int]] = None,
        extra_data: Optional[bytes] = None,
    ) -> AttestationQuote:
        """Generate a TPM attestation quote."""
        if not self._tpm_available:
            return self._generate_software_quote(nonce, pcr_selection, extra_data)

        if nonce is None:
            nonce = secrets.token_bytes(32)

        if pcr_selection is None:
            pcr_selection = self.DEFAULT_PCRS

        quote_id = f"tpm-quote-{secrets.token_hex(8)}"
        timestamp = datetime.utcnow()

        try:
            # Create PCR selection string
            pcr_list = ",".join(str(p) for p in pcr_selection)

            # Create temp files for quote data
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False) as nonce_file:
                nonce_file.write(nonce)
                nonce_path = nonce_file.name

            quote_path = tempfile.mktemp(suffix=".quote")
            sig_path = tempfile.mktemp(suffix=".sig")
            pcr_path = tempfile.mktemp(suffix=".pcr")

            try:
                # Generate quote using tpm2_quote
                cmd = [
                    "tpm2_quote",
                    "-c", "0x81010001",  # AIK handle
                    "-l", f"sha256:{pcr_list}",
                    "-q", nonce_path,
                    "-m", quote_path,
                    "-s", sig_path,
                    "-o", pcr_path,
                ]

                if extra_data:
                    extra_path = tempfile.mktemp()
                    with open(extra_path, "wb") as f:
                        f.write(extra_data)
                    cmd.extend(["-g", extra_path])

                rc, stdout, stderr = self._run_tpm_command(cmd)

                if rc != 0:
                    raise AttestationError(f"TPM quote failed: {stderr.decode()}")

                # Read quote data
                with open(quote_path, "rb") as f:
                    quote_data = f.read()
                with open(sig_path, "rb") as f:
                    signature = f.read()

                # Read PCR values
                pcr_values = self._read_pcr_values(pcr_selection)

            finally:
                # Cleanup temp files
                for path in [nonce_path, quote_path, sig_path, pcr_path]:
                    if os.path.exists(path):
                        os.unlink(path)

            return AttestationQuote(
                quote_id=quote_id,
                quote_type=QuoteType.PLATFORM,
                attestation_type=AttestationType.TPM,
                timestamp=timestamp,
                quote_data=quote_data,
                signature=signature,
                pcr_values=pcr_values,
                nonce=nonce,
                extra_data=extra_data,
            )

        except AttestationError:
            raise
        except Exception as e:
            raise AttestationError(f"Quote generation failed: {e}")

    def _generate_software_quote(
        self,
        nonce: Optional[bytes],
        pcr_selection: Optional[list[int]],
        extra_data: Optional[bytes],
    ) -> AttestationQuote:
        """Generate a software attestation quote (development only)."""
        logger.warning("Generating SOFTWARE attestation quote - NOT FOR PRODUCTION")

        if nonce is None:
            nonce = secrets.token_bytes(32)

        if pcr_selection is None:
            pcr_selection = self.DEFAULT_PCRS

        quote_id = f"sw-quote-{secrets.token_hex(8)}"
        timestamp = datetime.utcnow()

        # Generate mock PCR values
        pcr_values = {}
        for pcr in pcr_selection:
            # Simulate stable PCR values
            pcr_values[pcr] = hashlib.sha256(f"pcr{pcr}".encode()).digest()

        # Create quote structure
        quote_data = struct.pack(
            ">IIII",
            0x00000001,  # Version
            len(pcr_selection),
            len(nonce),
            len(extra_data) if extra_data else 0,
        )
        quote_data += nonce
        if extra_data:
            quote_data += extra_data
        for pcr in pcr_selection:
            quote_data += pcr_values[pcr]

        # Sign with a mock key
        signature = hashlib.sha256(quote_data).digest()

        return AttestationQuote(
            quote_id=quote_id,
            quote_type=QuoteType.PLATFORM,
            attestation_type=AttestationType.SOFTWARE,
            timestamp=timestamp,
            quote_data=quote_data,
            signature=signature,
            pcr_values=pcr_values,
            nonce=nonce,
            extra_data=extra_data,
            firmware_version="software-0.0.0",
        )

    def _read_pcr_values(self, pcr_selection: list[int]) -> dict[int, bytes]:
        """Read PCR values from TPM."""
        pcr_list = ",".join(str(p) for p in pcr_selection)

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
            output_path = f.name

        try:
            cmd = [
                "tpm2_pcrread",
                f"sha256:{pcr_list}",
                "-o", output_path,
            ]

            rc, stdout, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"PCR read failed: {stderr.decode()}")

            # Parse PCR values from output
            pcr_values = {}
            with open(output_path, "rb") as f:
                data = f.read()

            # Each PCR is 32 bytes for SHA256
            for i, pcr in enumerate(pcr_selection):
                start = i * 32
                end = start + 32
                if end <= len(data):
                    pcr_values[pcr] = data[start:end]

            return pcr_values

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def verify_quote(
        self,
        quote: AttestationQuote,
        policy: VerificationPolicy,
        expected_nonce: Optional[bytes] = None,
    ) -> AttestationResult:
        """Verify an attestation quote against a policy."""
        failure_reasons = []

        # Check attestation type
        if quote.attestation_type == AttestationType.SOFTWARE:
            if not policy.allow_debug:
                failure_reasons.append("Software attestation not allowed by policy")

        # Check timestamp
        quote_age = (datetime.utcnow() - quote.timestamp).total_seconds()
        timestamp_valid = quote_age <= policy.max_quote_age_seconds
        if not timestamp_valid:
            failure_reasons.append(
                f"Quote too old: {quote_age:.1f}s > {policy.max_quote_age_seconds}s"
            )

        # Check nonce
        nonce_valid = True
        if expected_nonce is not None and quote.nonce != expected_nonce:
            nonce_valid = False
            failure_reasons.append("Nonce mismatch")

        # Check PCR values
        pcr_match = True
        if policy.required_pcrs:
            for pcr_index, expected_value in policy.required_pcrs.items():
                actual_value = quote.pcr_values.get(pcr_index)
                if actual_value != expected_value:
                    pcr_match = False
                    failure_reasons.append(
                        f"PCR{pcr_index} mismatch: expected {expected_value.hex()[:16]}..., "
                        f"got {actual_value.hex()[:16] if actual_value else 'None'}..."
                    )

        # Check firmware version
        firmware_match = True
        if policy.allowed_firmware_versions and quote.firmware_version:
            if quote.firmware_version not in policy.allowed_firmware_versions:
                firmware_match = False
                failure_reasons.append(
                    f"Firmware version {quote.firmware_version} not in allowed list"
                )

        # Check attestation key
        if policy.allowed_ak_ids and quote.attestation_key_id:
            if quote.attestation_key_id not in policy.allowed_ak_ids:
                failure_reasons.append(
                    f"Attestation key {quote.attestation_key_id} not in allowed list"
                )

        # Verify signature (for real TPM quotes)
        signature_valid = True
        if self._tpm_available and quote.attestation_type == AttestationType.TPM:
            signature_valid = self._verify_quote_signature(quote)
            if not signature_valid:
                failure_reasons.append("Quote signature verification failed")

        # Run custom verifier if provided
        if policy.custom_verifier:
            try:
                custom_result = policy.custom_verifier(quote)
                if not custom_result:
                    failure_reasons.append("Custom verification failed")
            except Exception as e:
                failure_reasons.append(f"Custom verifier error: {e}")

        # Determine overall result
        verified = (
            len(failure_reasons) == 0
            and timestamp_valid
            and nonce_valid
            and pcr_match
            and firmware_match
            and signature_valid
        )

        return AttestationResult(
            verified=verified,
            quote=quote,
            policy=policy,
            pcr_match=pcr_match,
            firmware_match=firmware_match,
            signature_valid=signature_valid,
            nonce_valid=nonce_valid,
            timestamp_valid=timestamp_valid,
            failure_reasons=failure_reasons,
            platform_info={
                "attestation_type": quote.attestation_type.value,
                "firmware_version": quote.firmware_version,
            },
        )

    def _verify_quote_signature(self, quote: AttestationQuote) -> bool:
        """Verify quote signature using TPM."""
        import tempfile

        try:
            # Write quote data to temp files
            with tempfile.NamedTemporaryFile(delete=False) as qf:
                qf.write(quote.quote_data)
                quote_path = qf.name

            with tempfile.NamedTemporaryFile(delete=False) as sf:
                sf.write(quote.signature)
                sig_path = sf.name

            try:
                cmd = [
                    "tpm2_checkquote",
                    "-u", "ak.pub",  # AIK public key
                    "-m", quote_path,
                    "-s", sig_path,
                ]

                rc, _, _ = self._run_tpm_command(cmd)
                return rc == 0

            finally:
                os.unlink(quote_path)
                os.unlink(sig_path)

        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def get_attestation_key(self) -> tuple[bytes, str]:
        """Get the attestation identity key."""
        if not self._tpm_available:
            # Return mock key for software mode
            mock_key = hashlib.sha256(b"mock-attestation-key").digest()
            return mock_key, "software-ak"

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pub") as f:
            pub_path = f.name

        try:
            cmd = [
                "tpm2_readpublic",
                "-c", "0x81010001",  # AIK handle
                "-o", pub_path,
            ]

            rc, _, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"Failed to read AIK: {stderr.decode()}")

            with open(pub_path, "rb") as f:
                pub_key = f.read()

            # Compute key ID
            key_id = f"tpm-ak-{hashlib.sha256(pub_key).hexdigest()[:16]}"

            return pub_key, key_id

        finally:
            if os.path.exists(pub_path):
                os.unlink(pub_path)

    def get_endorsement_key_certificate(self) -> Optional[bytes]:
        """Get the EK certificate from TPM NVRAM."""
        if not self._tpm_available:
            return None

        try:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".crt") as f:
                cert_path = f.name

            try:
                # Read EK certificate from NVRAM
                cmd = [
                    "tpm2_nvread",
                    "0x01c00002",  # Standard EK cert location
                    "-o", cert_path,
                ]

                rc, _, _ = self._run_tpm_command(cmd)

                if rc != 0:
                    return None

                with open(cert_path, "rb") as f:
                    return f.read()

            finally:
                if os.path.exists(cert_path):
                    os.unlink(cert_path)

        except Exception:
            return None

    def extend_pcr(self, pcr_index: int, data: bytes) -> bytes:
        """Extend a PCR with new data."""
        if not self._tpm_available:
            # Software simulation
            current = hashlib.sha256(f"pcr{pcr_index}".encode()).digest()
            return hashlib.sha256(current + data).digest()

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            data_path = f.name

        try:
            cmd = [
                "tpm2_pcrextend",
                f"{pcr_index}:sha256={data_path}",
            ]

            rc, _, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"PCR extend failed: {stderr.decode()}")

            # Read new value
            return self._read_pcr_values([pcr_index])[pcr_index]

        finally:
            os.unlink(data_path)

    def seal_data(
        self,
        data: bytes,
        pcr_policy: Optional[dict[int, bytes]] = None,
    ) -> bytes:
        """Seal data to platform state."""
        if not self._tpm_available:
            # Software fallback - just encrypt
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = hashlib.sha256(b"software-seal-key").digest()
            nonce = secrets.token_bytes(12)
            aesgcm = AESGCM(key)
            return nonce + aesgcm.encrypt(nonce, data, None)

        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as df:
            df.write(data)
            data_path = df.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".blob") as bf:
            blob_path = bf.name

        try:
            cmd = [
                "tpm2_create",
                "-C", "0x81000001",  # Primary key handle
                "-i", data_path,
                "-u", blob_path + ".pub",
                "-r", blob_path + ".priv",
            ]

            if pcr_policy:
                pcr_list = ",".join(str(p) for p in pcr_policy.keys())
                cmd.extend(["-L", f"sha256:{pcr_list}"])

            rc, _, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"Seal failed: {stderr.decode()}")

            # Read sealed blob
            with open(blob_path + ".pub", "rb") as f:
                pub = f.read()
            with open(blob_path + ".priv", "rb") as f:
                priv = f.read()

            # Combine pub and priv
            return struct.pack(">I", len(pub)) + pub + priv

        finally:
            os.unlink(data_path)
            for suffix in [".pub", ".priv"]:
                path = blob_path + suffix
                if os.path.exists(path):
                    os.unlink(path)

    def unseal_data(self, sealed_blob: bytes) -> bytes:
        """Unseal previously sealed data."""
        if not self._tpm_available:
            # Software fallback
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            key = hashlib.sha256(b"software-seal-key").digest()
            nonce = sealed_blob[:12]
            ciphertext = sealed_blob[12:]
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ciphertext, None)

        import tempfile

        # Parse blob
        pub_len = struct.unpack(">I", sealed_blob[:4])[0]
        pub = sealed_blob[4 : 4 + pub_len]
        priv = sealed_blob[4 + pub_len :]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pub") as pf:
            pf.write(pub)
            pub_path = pf.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".priv") as rf:
            rf.write(priv)
            priv_path = rf.name

        with tempfile.NamedTemporaryFile(delete=False) as df:
            data_path = df.name

        try:
            # Load object
            cmd = [
                "tpm2_load",
                "-C", "0x81000001",
                "-u", pub_path,
                "-r", priv_path,
                "-c", "object.ctx",
            ]

            rc, _, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"Object load failed: {stderr.decode()}")

            # Unseal
            cmd = [
                "tpm2_unseal",
                "-c", "object.ctx",
                "-o", data_path,
            ]

            rc, _, stderr = self._run_tpm_command(cmd)

            if rc != 0:
                raise AttestationError(f"Unseal failed: {stderr.decode()}")

            with open(data_path, "rb") as f:
                return f.read()

        finally:
            for path in [pub_path, priv_path, data_path]:
                if os.path.exists(path):
                    os.unlink(path)
            if os.path.exists("object.ctx"):
                os.unlink("object.ctx")

    def health_check(self) -> dict:
        """Check TPM health."""
        if not self._tpm_available:
            return {
                "status": "degraded",
                "available": False,
                "warning": "TPM not available, using software fallback",
                "attestation_type": "software",
            }

        start = time.time()
        try:
            # Try to read PCR 0
            cmd = ["tpm2_pcrread", "sha256:0"]
            rc, _, _ = self._run_tpm_command(cmd)

            return {
                "status": "healthy" if rc == 0 else "unhealthy",
                "available": True,
                "device": self._device_path,
                "latency_ms": (time.time() - start) * 1000,
                "attestation_type": "tpm",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "available": True,
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000,
            }
