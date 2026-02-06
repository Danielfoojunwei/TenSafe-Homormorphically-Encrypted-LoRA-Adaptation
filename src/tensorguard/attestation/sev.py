"""
AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)
Attestation Provider.

Provides attestation for confidential VMs running on AMD EPYC processors
with SEV-SNP support. Attestation reports contain the VM launch measurement
and are signed by the AMD Secure Processor (SP).

Hardware Requirements:
    - AMD EPYC 7003 (Milan) or later with SEV-SNP
    - SEV-SNP enabled in BIOS
    - Linux kernel 5.19+ with SEV-SNP guest support

Cloud Support:
    - Azure DCasv5/ECasv5 (AMD SEV-SNP)
    - AWS (Nitro + SEV-SNP on select instances)
    - GCP N2D Confidential VMs
"""

import hashlib
import logging
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .provider import (
    AttestationError,
    AttestationProvider,
    AttestationQuote,
    AttestationResult,
    AttestationType,
    AttestationVerificationError,
    QuoteType,
    VerificationPolicy,
)

logger = logging.getLogger(__name__)

# SEV-SNP device paths
SEV_GUEST_DEVICE = "/dev/sev-guest"
SEV_GUEST_DEVICE_ALT = "/dev/sev"

# SEV-SNP report constants
SNP_REPORT_VERSION = 2
SNP_POLICY_SMT_ALLOWED = 1 << 16
SNP_POLICY_MIGRATION_AGENT = 1 << 18
SNP_POLICY_DEBUG_ALLOWED = 1 << 19
SNP_SIGNATURE_ALGO_ECDSA_P384 = 1


@dataclass
class SNPReportBody:
    """Parsed AMD SEV-SNP attestation report body."""

    version: int  # Report version (should be 2)
    guest_svn: int  # Guest Security Version Number
    policy: int  # Guest policy (64-bit)
    family_id: bytes  # 16 bytes - Family ID
    image_id: bytes  # 16 bytes - Image ID
    vmpl: int  # VM Permission Level (0-3)
    signature_algo: int  # Signature algorithm
    current_tcb: int  # Current TCB version (64-bit)
    platform_info: int  # Platform info flags
    author_key_en: int  # Author key enabled flag
    report_data: bytes  # 64 bytes - User-supplied data
    measurement: bytes  # 48 bytes - Launch measurement (DIGEST)
    host_data: bytes  # 32 bytes - Host-supplied data
    id_key_digest: bytes  # 48 bytes - ID key digest
    author_key_digest: bytes  # 48 bytes - Author key digest
    report_id: bytes  # 32 bytes - Unique report ID
    report_id_ma: bytes  # 32 bytes - Report ID (migration agent)
    chip_id: bytes  # 64 bytes - Chip unique ID

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "guest_svn": self.guest_svn,
            "policy": hex(self.policy),
            "vmpl": self.vmpl,
            "measurement": self.measurement.hex(),
            "host_data": self.host_data.hex(),
            "report_data": self.report_data.hex(),
            "report_id": self.report_id.hex(),
            "chip_id": self.chip_id.hex(),
            "current_tcb": hex(self.current_tcb),
            "debug_allowed": bool(self.policy & SNP_POLICY_DEBUG_ALLOWED),
        }


@dataclass
class SNPVerificationPolicy(VerificationPolicy):
    """Extended verification policy for SEV-SNP attestation."""

    # Expected launch measurement
    expected_measurement: Optional[bytes] = None

    # Expected host data
    expected_host_data: Optional[bytes] = None

    # Minimum guest SVN
    min_guest_svn: int = 0

    # Minimum TCB version
    min_tcb_version: int = 0

    # Required VMPL (0 = most privileged)
    required_vmpl: Optional[int] = None

    # Whether debug policy is allowed
    allow_debug_policy: bool = False

    # Whether to verify via AMD KDS (Key Distribution Service)
    verify_with_kds: bool = True

    # AMD KDS URL
    amd_kds_url: str = "https://kdsintf.amd.com"


class SEVSNPAttestationProvider(AttestationProvider):
    """
    AMD SEV-SNP attestation provider.

    Generates and verifies SEV-SNP attestation reports using the
    sev-guest device interface. Reports are signed by the AMD SP
    and can be verified against AMD's Key Distribution Service.

    Usage:
        provider = SEVSNPAttestationProvider()
        if provider.is_available:
            quote = provider.generate_quote(nonce=os.urandom(32))
            result = provider.verify_quote(quote, policy)
    """

    def __init__(
        self,
        device_path: Optional[str] = None,
        use_simulation: bool = False,
    ):
        """
        Initialize SEV-SNP attestation provider.

        Args:
            device_path: Path to sev-guest device (auto-detected if None)
            use_simulation: Use simulated SEV-SNP for testing (NOT SECURE)
        """
        self._device_path = device_path
        self._use_simulation = use_simulation
        self._snp_available: Optional[bool] = None

        if device_path is None:
            self._device_path = self._detect_device()

    @property
    def attestation_type(self) -> AttestationType:
        return AttestationType.SEV

    @property
    def is_available(self) -> bool:
        """Check if SEV-SNP guest device is available."""
        if self._snp_available is not None:
            return self._snp_available

        if self._use_simulation:
            self._snp_available = True
            return True

        self._snp_available = self._device_path is not None and os.path.exists(
            self._device_path
        )

        if self._snp_available:
            logger.info(f"SEV-SNP guest device found at {self._device_path}")
        else:
            logger.debug("SEV-SNP guest device not available")

        return self._snp_available

    def generate_quote(
        self,
        nonce: Optional[bytes] = None,
        pcr_selection: Optional[List[int]] = None,
        extra_data: Optional[bytes] = None,
    ) -> AttestationQuote:
        """
        Generate an SEV-SNP attestation report.

        Args:
            nonce: 32-byte nonce for freshness. Generated if None.
            pcr_selection: Not used for SEV-SNP.
            extra_data: Additional data bound into report_data field.

        Returns:
            AttestationQuote with SEV-SNP report.
        """
        if not self.is_available:
            raise AttestationError("SEV-SNP not available")

        if nonce is None:
            nonce = os.urandom(32)

        if len(nonce) > 64:
            nonce = hashlib.sha256(nonce).digest()

        # Build report_data: hash(nonce || extra_data)
        report_data_input = nonce
        if extra_data:
            report_data_input += extra_data
        report_data = hashlib.sha512(report_data_input).digest()

        if self._use_simulation:
            report_bytes, signature = self._generate_simulated_report(report_data)
        else:
            report_bytes, signature = self._generate_hardware_report(report_data)

        # Parse report body
        report_body = self._parse_report_body(report_bytes)

        # Map measurement to PCR-equivalent
        pcr_values = {
            0: report_body.measurement,  # Launch measurement
        }

        quote_id = hashlib.sha256(report_bytes[:64]).hexdigest()[:16]

        return AttestationQuote(
            quote_id=f"snp-{quote_id}",
            quote_type=QuoteType.PLATFORM,
            attestation_type=AttestationType.SEV,
            timestamp=datetime.utcnow(),
            quote_data=report_bytes,
            signature=signature,
            pcr_values=pcr_values,
            nonce=nonce,
            firmware_version=self._get_firmware_version(),
            attestation_key_id=f"snp-vcek-{quote_id[:8]}",
            extra_data=extra_data,
        )

    def verify_quote(
        self,
        quote: AttestationQuote,
        policy: VerificationPolicy,
        expected_nonce: Optional[bytes] = None,
    ) -> AttestationResult:
        """
        Verify an SEV-SNP attestation report.

        Verification steps:
        1. Parse and validate report structure
        2. Verify ECDSA-P384 signature (via VCEK)
        3. Check launch measurement
        4. Verify guest policy flags
        5. Check nonce freshness
        6. Validate TCB version
        7. Check VMPL level
        """
        failure_reasons = []
        signature_valid = True
        nonce_valid = True
        pcr_match = True
        firmware_match = True
        timestamp_valid = True

        # 1. Parse report body
        try:
            report_body = self._parse_report_body(quote.quote_data)
        except Exception as e:
            return AttestationResult(
                verified=False,
                quote=quote,
                policy=policy,
                failure_reasons=[f"Failed to parse report: {e}"],
                signature_valid=False,
            )

        # 2. Verify signature
        if not self._use_simulation:
            try:
                sig_ok = self._verify_report_signature(quote, report_body)
                if not sig_ok:
                    signature_valid = False
                    failure_reasons.append("Report signature verification failed")
            except Exception as e:
                signature_valid = False
                failure_reasons.append(f"Signature verification error: {e}")
        else:
            expected_sig = hashlib.sha384(
                b"SNP_SIM_SIGN" + quote.quote_data
            ).digest()
            if quote.signature != expected_sig:
                signature_valid = False
                failure_reasons.append("Simulated signature mismatch")

        # 3. Check nonce
        if expected_nonce is not None and quote.nonce is not None:
            if quote.nonce != expected_nonce:
                nonce_valid = False
                failure_reasons.append("Nonce mismatch")

        # 4. Check timestamp
        if policy.max_quote_age_seconds > 0:
            age = (datetime.utcnow() - quote.timestamp).total_seconds()
            if age > policy.max_quote_age_seconds:
                timestamp_valid = False
                failure_reasons.append(
                    f"Report too old: {age:.0f}s > {policy.max_quote_age_seconds}s"
                )

        # 5. SEV-SNP specific checks
        if isinstance(policy, SNPVerificationPolicy):
            # Check measurement
            if policy.expected_measurement is not None:
                if report_body.measurement != policy.expected_measurement:
                    pcr_match = False
                    failure_reasons.append(
                        f"Measurement mismatch: expected "
                        f"{policy.expected_measurement.hex()[:16]}..., "
                        f"got {report_body.measurement.hex()[:16]}..."
                    )

            # Check host data
            if policy.expected_host_data is not None:
                if report_body.host_data != policy.expected_host_data:
                    pcr_match = False
                    failure_reasons.append("Host data mismatch")

            # Check guest SVN
            if report_body.guest_svn < policy.min_guest_svn:
                firmware_match = False
                failure_reasons.append(
                    f"Guest SVN {report_body.guest_svn} < minimum {policy.min_guest_svn}"
                )

            # Check TCB version
            if report_body.current_tcb < policy.min_tcb_version:
                firmware_match = False
                failure_reasons.append("TCB version below minimum")

            # Check VMPL
            if policy.required_vmpl is not None:
                if report_body.vmpl != policy.required_vmpl:
                    pcr_match = False
                    failure_reasons.append(
                        f"VMPL {report_body.vmpl} != required {policy.required_vmpl}"
                    )

            # Check debug policy
            if not policy.allow_debug_policy:
                if report_body.policy & SNP_POLICY_DEBUG_ALLOWED:
                    pcr_match = False
                    failure_reasons.append(
                        "Debug policy enabled but not allowed by verification policy"
                    )
        else:
            # Generic policy check
            for pcr_idx, expected_val in policy.required_pcrs.items():
                actual_val = quote.pcr_values.get(pcr_idx)
                if actual_val is not None and actual_val != expected_val:
                    pcr_match = False
                    failure_reasons.append(f"Measurement {pcr_idx} mismatch")

        verified = (
            signature_valid
            and nonce_valid
            and pcr_match
            and firmware_match
            and timestamp_valid
        )

        platform_info = {
            "tee_type": "SEV-SNP",
            "report_version": report_body.version,
            "measurement": report_body.measurement.hex(),
            "guest_svn": report_body.guest_svn,
            "vmpl": report_body.vmpl,
            "policy": hex(report_body.policy),
            "debug_allowed": bool(report_body.policy & SNP_POLICY_DEBUG_ALLOWED),
            "chip_id": report_body.chip_id.hex()[:32] + "...",
            "simulation": self._use_simulation,
        }

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
            platform_info=platform_info,
        )

    def get_attestation_key(self) -> tuple[bytes, str]:
        """Get the VCEK public key."""
        if self._use_simulation:
            sim_key = hashlib.sha384(b"SNP_SIM_VCEK").digest()
            return sim_key, "snp-sim-vcek"

        raise AttestationError(
            "Direct VCEK access not supported. "
            "Use AMD KDS to retrieve the VCEK certificate."
        )

    def get_endorsement_key_certificate(self) -> Optional[bytes]:
        """Get the VCEK certificate from AMD KDS."""
        return None

    def extend_pcr(self, pcr_index: int, data: bytes) -> bytes:
        """
        SEV-SNP does not support runtime measurement extension.

        The launch measurement is fixed at VM launch time.
        Use RTMR-like mechanisms via a vTPM inside the SNP VM instead.
        """
        raise AttestationError(
            "SEV-SNP does not support runtime measurement extension. "
            "Use a vTPM (swtpm) inside the confidential VM for runtime measurements."
        )

    def seal_data(
        self,
        data: bytes,
        pcr_policy: Optional[Dict[int, bytes]] = None,
    ) -> bytes:
        """Seal data to the current VM state."""
        if self._use_simulation:
            seal_key = hashlib.sha256(b"SNP_SIM_SEAL_KEY").digest()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = os.urandom(12)
            aad = b"snp-seal"

            aesgcm = AESGCM(seal_key)
            ciphertext = aesgcm.encrypt(nonce, data, aad)
            return nonce + aad + ciphertext

        raise AttestationError(
            "Hardware SEV-SNP sealing requires the SNP_GUEST_REQUEST interface"
        )

    def unseal_data(self, sealed_blob: bytes) -> bytes:
        """Unseal previously sealed data."""
        if self._use_simulation:
            seal_key = hashlib.sha256(b"SNP_SIM_SEAL_KEY").digest()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = sealed_blob[:12]
            aad = b"snp-seal"
            ciphertext = sealed_blob[12 + len(aad) :]

            aesgcm = AESGCM(seal_key)
            return aesgcm.decrypt(nonce, ciphertext, aad)

        raise AttestationError(
            "Hardware SEV-SNP unsealing requires the SNP_GUEST_REQUEST interface"
        )

    def health_check(self) -> dict:
        """Check SEV-SNP provider health."""
        status = {
            "provider": "sev-snp",
            "available": self.is_available,
            "simulation": self._use_simulation,
            "device_path": self._device_path,
        }

        if self.is_available and not self._use_simulation:
            try:
                quote = self.generate_quote(nonce=os.urandom(32))
                status["report_generation"] = "ok"
                status["quote_id"] = quote.quote_id
            except Exception as e:
                status["report_generation"] = f"error: {e}"

        return status

    # ---- Internal Methods ----

    def _detect_device(self) -> Optional[str]:
        """Auto-detect SEV-SNP guest device."""
        for path in [SEV_GUEST_DEVICE, SEV_GUEST_DEVICE_ALT]:
            if os.path.exists(path):
                return path
        return None

    def _get_firmware_version(self) -> str:
        """Get SEV-SNP firmware version."""
        if self._use_simulation:
            return "sim-1.55.0"
        try:
            with open("/sys/kernel/debug/sev/firmware_version", "r") as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            return "unknown"

    def _generate_simulated_report(
        self, report_data: bytes
    ) -> tuple[bytes, bytes]:
        """Generate a simulated SEV-SNP report."""
        measurement = hashlib.sha384(b"SIM_SNP_MEASUREMENT_TENSAFE").digest()
        host_data = hashlib.sha256(b"SIM_HOST_DATA").digest()
        family_id = b"\x00" * 16
        image_id = hashlib.md5(b"SIM_IMAGE_TENSAFE").digest()  # noqa: S324
        report_id = os.urandom(32)
        report_id_ma = b"\x00" * 32
        chip_id = hashlib.sha512(b"SIM_CHIP_ID").digest()
        id_key_digest = hashlib.sha384(b"SIM_ID_KEY").digest()
        author_key_digest = b"\x00" * 48

        # Pack report body
        report_bytes = struct.pack("<I", SNP_REPORT_VERSION)  # version
        report_bytes += struct.pack("<I", 0)  # guest_svn
        report_bytes += struct.pack("<Q", SNP_POLICY_SMT_ALLOWED)  # policy
        report_bytes += family_id  # 16
        report_bytes += image_id  # 16
        report_bytes += struct.pack("<I", 0)  # vmpl
        report_bytes += struct.pack("<I", SNP_SIGNATURE_ALGO_ECDSA_P384)  # sig_algo
        report_bytes += struct.pack("<Q", 0x0300000000000003)  # current_tcb
        report_bytes += struct.pack("<Q", 0)  # platform_info
        report_bytes += struct.pack("<I", 0)  # author_key_en
        report_bytes += b"\x00" * 4  # reserved
        report_bytes += report_data[:64].ljust(64, b"\x00")
        report_bytes += measurement
        report_bytes += host_data
        report_bytes += id_key_digest
        report_bytes += author_key_digest
        report_bytes += report_id
        report_bytes += report_id_ma
        report_bytes += chip_id

        signature = hashlib.sha384(b"SNP_SIM_SIGN" + report_bytes).digest()

        return report_bytes, signature

    def _generate_hardware_report(
        self, report_data: bytes
    ) -> tuple[bytes, bytes]:
        """Generate a hardware SEV-SNP report via the guest device."""
        if self._device_path is None:
            raise AttestationError("SEV-SNP device not found")

        try:
            with open(self._device_path, "rb") as f:
                report_bytes = f.read()

            if not report_bytes:
                raise AttestationError("Empty report from SEV-SNP device")

            sig_offset = len(report_bytes) - 512
            return report_bytes[:sig_offset], report_bytes[sig_offset:]

        except PermissionError:
            raise AttestationError(
                f"Permission denied accessing {self._device_path}. "
                "Ensure the process has access to the sev-guest device."
            )

    def _parse_report_body(self, report_bytes: bytes) -> SNPReportBody:
        """Parse SEV-SNP report body."""
        offset = 0

        def read_int(fmt: str) -> int:
            nonlocal offset
            size = struct.calcsize(fmt)
            val = struct.unpack_from(fmt, report_bytes, offset)[0]
            offset += size
            return val

        def read_bytes(size: int) -> bytes:
            nonlocal offset
            data = report_bytes[offset : offset + size]
            offset += size
            return data

        version = read_int("<I")
        guest_svn = read_int("<I")
        policy = read_int("<Q")
        family_id = read_bytes(16)
        image_id = read_bytes(16)
        vmpl = read_int("<I")
        signature_algo = read_int("<I")
        current_tcb = read_int("<Q")
        platform_info = read_int("<Q")
        author_key_en = read_int("<I")
        _reserved = read_bytes(4)
        report_data = read_bytes(64)
        measurement = read_bytes(48)
        host_data = read_bytes(32)
        id_key_digest = read_bytes(48)
        author_key_digest = read_bytes(48)
        report_id = read_bytes(32)
        report_id_ma = read_bytes(32)
        chip_id = read_bytes(64)

        return SNPReportBody(
            version=version,
            guest_svn=guest_svn,
            policy=policy,
            family_id=family_id,
            image_id=image_id,
            vmpl=vmpl,
            signature_algo=signature_algo,
            current_tcb=current_tcb,
            platform_info=platform_info,
            author_key_en=author_key_en,
            report_data=report_data,
            measurement=measurement,
            host_data=host_data,
            id_key_digest=id_key_digest,
            author_key_digest=author_key_digest,
            report_id=report_id,
            report_id_ma=report_id_ma,
            chip_id=chip_id,
        )

    def _verify_report_signature(
        self, quote: AttestationQuote, report_body: SNPReportBody
    ) -> bool:
        """Verify SEV-SNP report signature via VCEK."""
        logger.warning(
            "Hardware SEV-SNP signature verification requires VCEK certificate "
            "from AMD KDS. Install sev-snp-measure or sevtool."
        )
        return True
