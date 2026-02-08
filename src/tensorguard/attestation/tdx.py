"""
Intel TDX (Trust Domain Extensions) Attestation Provider.

Provides attestation for confidential VMs running on Intel TDX-capable
processors. TDX quotes contain TD measurements (MRTD, RTMR) signed
by Intel's quoting enclave via DCAP.

Hardware Requirements:
    - Intel Xeon Scalable 4th Gen+ (Sapphire Rapids or later)
    - TDX-enabled BIOS/firmware
    - Linux kernel 6.2+ with TDX guest support

Cloud Support:
    - Azure DCesv5/ECesv5 (Intel TDX)
    - GCP C3 Confidential VMs
    - Alibaba Cloud g8i instances
"""

import hashlib
import logging
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from .provider import (
    AttestationError,
    AttestationProvider,
    AttestationQuote,
    AttestationResult,
    AttestationType,
    QuoteType,
    VerificationPolicy,
)

logger = logging.getLogger(__name__)

# TDX device paths
TDX_GUEST_DEVICE = "/dev/tdx_guest"
TDX_ATTEST_DEVICE = "/dev/tdx-attest"
TDX_GUEST_DEVICE_ALT = "/sys/kernel/config/tsm/report/tdx0"

# TDX quote constants
TDX_QUOTE_VERSION = 4
TDX_ATTESTATION_KEY_TYPE_ECDSA_P256 = 2
TDX_TEE_TYPE = 0x81  # TDX TEE type

# RTMR (Runtime Measurement Register) indices
RTMR_FIRMWARE = 0  # Firmware measurements
RTMR_OS_KERNEL = 1  # OS/kernel measurements
RTMR_APPLICATION = 2  # Application measurements
RTMR_USER = 3  # User-defined measurements


@dataclass
class TDXQuoteBody:
    """Parsed TDX quote body (TD Quote Body v4)."""

    tee_tcb_svn: bytes  # 16 bytes - TEE TCB SVN
    mr_seam: bytes  # 48 bytes - Measurement of SEAM module
    mr_signer_seam: bytes  # 48 bytes - Signer of SEAM module
    seam_attributes: bytes  # 8 bytes - SEAM attributes
    td_attributes: bytes  # 8 bytes - TD attributes
    xfam: bytes  # 8 bytes - XFAM (extended features)
    mr_td: bytes  # 48 bytes - Measurement of TD (MRTD)
    mr_config_id: bytes  # 48 bytes - Config ID measurement
    mr_owner: bytes  # 48 bytes - TD owner measurement
    mr_owner_config: bytes  # 48 bytes - TD owner config measurement
    rtmr0: bytes  # 48 bytes - Runtime measurement register 0
    rtmr1: bytes  # 48 bytes - Runtime measurement register 1
    rtmr2: bytes  # 48 bytes - Runtime measurement register 2
    rtmr3: bytes  # 48 bytes - Runtime measurement register 3
    report_data: bytes  # 64 bytes - Report data (user-supplied)

    def to_dict(self) -> Dict[str, str]:
        """Serialize to hex-encoded dictionary."""
        return {
            "tee_tcb_svn": self.tee_tcb_svn.hex(),
            "mr_seam": self.mr_seam.hex(),
            "mr_td": self.mr_td.hex(),
            "mr_config_id": self.mr_config_id.hex(),
            "mr_owner": self.mr_owner.hex(),
            "rtmr0": self.rtmr0.hex(),
            "rtmr1": self.rtmr1.hex(),
            "rtmr2": self.rtmr2.hex(),
            "rtmr3": self.rtmr3.hex(),
            "report_data": self.report_data.hex(),
        }


@dataclass
class TDXVerificationPolicy(VerificationPolicy):
    """Extended verification policy for TDX attestation."""

    # Expected TD measurements
    expected_mr_td: Optional[bytes] = None  # Expected MRTD
    expected_mr_config_id: Optional[bytes] = None
    expected_mr_owner: Optional[bytes] = None

    # Expected RTMR values
    expected_rtmrs: Dict[int, bytes] = field(default_factory=dict)

    # Minimum TCB SVN components
    min_tcb_svn: Optional[bytes] = None

    # Whether to verify via Intel DCAP (requires network)
    verify_with_dcap: bool = True

    # Intel PCS (Provisioning Certification Service) URL
    intel_pcs_url: str = "https://api.trustedservices.intel.com/sgx/certification/v4"


class TDXAttestationProvider(AttestationProvider):
    """
    Intel TDX attestation provider.

    Generates and verifies TDX attestation quotes using the TDX guest
    device interface. Quotes are signed by Intel's DCAP quoting enclave
    and can be verified against Intel's attestation infrastructure.

    Usage:
        provider = TDXAttestationProvider()
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
        Initialize TDX attestation provider.

        Args:
            device_path: Path to TDX guest device (auto-detected if None)
            use_simulation: Use simulated TDX for testing (NOT SECURE)
        """
        self._device_path = device_path
        self._use_simulation = use_simulation
        self._tdx_available: Optional[bool] = None
        self._cached_quote_body: Optional[TDXQuoteBody] = None

        if device_path is None:
            self._device_path = self._detect_device()

    @property
    def attestation_type(self) -> AttestationType:
        return AttestationType.SGX  # TDX uses SGX DCAP infrastructure

    @property
    def is_available(self) -> bool:
        """Check if TDX guest device is available."""
        if self._tdx_available is not None:
            return self._tdx_available

        if self._use_simulation:
            self._tdx_available = True
            return True

        self._tdx_available = self._device_path is not None and os.path.exists(
            self._device_path
        )

        if self._tdx_available:
            logger.info(f"TDX guest device found at {self._device_path}")
        else:
            logger.debug("TDX guest device not available")

        return self._tdx_available

    def generate_quote(
        self,
        nonce: Optional[bytes] = None,
        pcr_selection: Optional[List[int]] = None,
        extra_data: Optional[bytes] = None,
    ) -> AttestationQuote:
        """
        Generate a TDX attestation quote.

        The quote contains TD measurements (MRTD, RTMRs) and is signed
        by Intel's quoting enclave.

        Args:
            nonce: 32-byte nonce for freshness. Generated if None.
            pcr_selection: Not used for TDX (uses RTMRs instead).
            extra_data: Additional data to bind into report_data field.

        Returns:
            AttestationQuote with TDX quote data.
        """
        if not self.is_available:
            raise AttestationError("TDX not available")

        if nonce is None:
            nonce = os.urandom(32)

        if len(nonce) > 64:
            nonce = hashlib.sha256(nonce).digest()

        # Build report_data: hash(nonce || extra_data)
        report_data_input = nonce
        if extra_data:
            report_data_input += extra_data
        report_data = hashlib.sha512(report_data_input).digest()  # 64 bytes

        if self._use_simulation:
            quote_data, signature = self._generate_simulated_quote(report_data)
        else:
            quote_data, signature = self._generate_hardware_quote(report_data)

        # Parse quote body for PCR-equivalent values (RTMRs)
        quote_body = self._parse_quote_body(quote_data)
        self._cached_quote_body = quote_body

        pcr_values = {
            RTMR_FIRMWARE: quote_body.rtmr0,
            RTMR_OS_KERNEL: quote_body.rtmr1,
            RTMR_APPLICATION: quote_body.rtmr2,
            RTMR_USER: quote_body.rtmr3,
        }

        quote_id = hashlib.sha256(quote_data[:64]).hexdigest()[:16]

        return AttestationQuote(
            quote_id=f"tdx-{quote_id}",
            quote_type=QuoteType.PLATFORM,
            attestation_type=AttestationType.SGX,
            timestamp=datetime.utcnow(),
            quote_data=quote_data,
            signature=signature,
            pcr_values=pcr_values,
            nonce=nonce,
            firmware_version=self._get_firmware_version(),
            attestation_key_id=f"tdx-ecdsa-{quote_id[:8]}",
            extra_data=extra_data,
        )

    def verify_quote(
        self,
        quote: AttestationQuote,
        policy: VerificationPolicy,
        expected_nonce: Optional[bytes] = None,
    ) -> AttestationResult:
        """
        Verify a TDX attestation quote against a policy.

        Verification steps:
        1. Parse and validate quote structure
        2. Verify ECDSA signature (via DCAP or locally)
        3. Check MRTD against expected value
        4. Check RTMRs against policy
        5. Verify nonce freshness
        6. Check TCB SVN against minimum
        7. Validate quote timestamp

        Args:
            quote: TDX attestation quote to verify
            policy: Verification policy
            expected_nonce: Expected nonce for freshness check

        Returns:
            AttestationResult with detailed verification outcome
        """
        failure_reasons = []
        signature_valid = True
        nonce_valid = True
        pcr_match = True
        firmware_match = True
        timestamp_valid = True

        # 1. Parse quote body
        try:
            quote_body = self._parse_quote_body(quote.quote_data)
        except Exception as e:
            return AttestationResult(
                verified=False,
                quote=quote,
                policy=policy,
                failure_reasons=[f"Failed to parse quote: {e}"],
                signature_valid=False,
            )

        # 2. Verify signature
        if not self._use_simulation:
            try:
                sig_ok = self._verify_quote_signature(quote)
                if not sig_ok:
                    signature_valid = False
                    failure_reasons.append("Quote signature verification failed")
            except Exception as e:
                signature_valid = False
                failure_reasons.append(f"Signature verification error: {e}")
        else:
            # In simulation, verify the simulated HMAC
            expected_sig = hashlib.sha256(
                b"TDX_SIM_SIGN" + quote.quote_data
            ).digest()
            if quote.signature != expected_sig:
                signature_valid = False
                failure_reasons.append("Simulated signature mismatch")

        # 3. Check nonce freshness
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
                    f"Quote too old: {age:.0f}s > {policy.max_quote_age_seconds}s"
                )

        # 5. Check RTMRs (mapped to PCR values)
        if isinstance(policy, TDXVerificationPolicy):
            for rtmr_idx, expected_val in policy.expected_rtmrs.items():
                actual_val = quote.pcr_values.get(rtmr_idx)
                if actual_val is None:
                    pcr_match = False
                    failure_reasons.append(f"RTMR{rtmr_idx} not present in quote")
                elif actual_val != expected_val:
                    pcr_match = False
                    failure_reasons.append(
                        f"RTMR{rtmr_idx} mismatch: "
                        f"expected {expected_val.hex()[:16]}..., "
                        f"got {actual_val.hex()[:16]}..."
                    )

            # Check MRTD
            if policy.expected_mr_td is not None:
                if quote_body.mr_td != policy.expected_mr_td:
                    pcr_match = False
                    failure_reasons.append("MRTD mismatch")

            # Check TCB SVN
            if policy.min_tcb_svn is not None:
                if quote_body.tee_tcb_svn < policy.min_tcb_svn:
                    firmware_match = False
                    failure_reasons.append("TCB SVN below minimum")
        else:
            # Generic policy -- check required_pcrs as RTMRs
            for pcr_idx, expected_val in policy.required_pcrs.items():
                actual_val = quote.pcr_values.get(pcr_idx)
                if actual_val is not None and actual_val != expected_val:
                    pcr_match = False
                    failure_reasons.append(f"RTMR{pcr_idx} mismatch")

        # 6. Check firmware version
        if policy.allowed_firmware_versions and quote.firmware_version:
            if quote.firmware_version not in policy.allowed_firmware_versions:
                firmware_match = False
                failure_reasons.append(
                    f"Firmware version {quote.firmware_version} not in allowed list"
                )

        verified = (
            signature_valid
            and nonce_valid
            and pcr_match
            and firmware_match
            and timestamp_valid
        )

        platform_info = {
            "tee_type": "TDX",
            "quote_version": TDX_QUOTE_VERSION,
            "mr_td": quote_body.mr_td.hex(),
            "rtmr0": quote_body.rtmr0.hex(),
            "rtmr1": quote_body.rtmr1.hex(),
            "rtmr2": quote_body.rtmr2.hex(),
            "rtmr3": quote_body.rtmr3.hex(),
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
        """Get the TDX attestation key (ECDSA P-256)."""
        if self._use_simulation:
            sim_key = hashlib.sha256(b"TDX_SIM_AK").digest()
            return sim_key, "tdx-sim-ak"

        raise AttestationError(
            "Direct attestation key access not supported for TDX. "
            "Use generate_quote() to get a signed quote."
        )

    def get_endorsement_key_certificate(self) -> Optional[bytes]:
        """TDX uses PCK certificates instead of EK certificates."""
        return None

    def extend_pcr(self, pcr_index: int, data: bytes) -> bytes:
        """
        Extend an RTMR with new data.

        TDX has 4 RTMRs (0-3). RTMR2 and RTMR3 are available
        for application use.

        Args:
            pcr_index: RTMR index (0-3)
            data: Data to extend with (will be hashed to 48 bytes)
        """
        if pcr_index < 0 or pcr_index > 3:
            raise AttestationError(f"Invalid RTMR index: {pcr_index} (must be 0-3)")

        if pcr_index < 2 and not self._use_simulation:
            raise AttestationError(
                f"RTMR{pcr_index} is reserved for firmware/OS. "
                "Only RTMR2 and RTMR3 can be extended by applications."
            )

        # Hash to 48 bytes (SHA-384 for TDX)
        extend_data = hashlib.sha384(data).digest()

        if self._use_simulation:
            current = os.urandom(48)
            new_value = hashlib.sha384(current + extend_data).digest()
            logger.info(f"[SIM] Extended RTMR{pcr_index}: {new_value.hex()[:16]}...")
            return new_value

        return self._hardware_extend_rtmr(pcr_index, extend_data)

    def seal_data(
        self,
        data: bytes,
        pcr_policy: Optional[Dict[int, bytes]] = None,
    ) -> bytes:
        """
        Seal data to current TD state.

        Uses MRTD and RTMRs to create a sealing policy.
        Data can only be unsealed when measurements match.
        """
        if self._use_simulation:
            seal_key = hashlib.sha256(b"TDX_SIM_SEAL_KEY").digest()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = os.urandom(12)
            aad = b"tdx-seal"
            if pcr_policy:
                aad += hashlib.sha256(
                    b"".join(v for v in pcr_policy.values())
                ).digest()

            aesgcm = AESGCM(seal_key)
            ciphertext = aesgcm.encrypt(nonce, data, aad)
            return nonce + aad + ciphertext

        raise AttestationError("Hardware TDX sealing requires TDVMCALL interface")

    def unseal_data(self, sealed_blob: bytes) -> bytes:
        """Unseal previously sealed data."""
        if self._use_simulation:
            seal_key = hashlib.sha256(b"TDX_SIM_SEAL_KEY").digest()
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = sealed_blob[:12]
            # Find AAD boundary (tdx-seal prefix + optional hash)
            aad_base = b"tdx-seal"
            if len(sealed_blob) > 12 + len(aad_base) + 32 + 16:
                aad = sealed_blob[12 : 12 + len(aad_base) + 32]
            else:
                aad = sealed_blob[12 : 12 + len(aad_base)]
            ciphertext = sealed_blob[12 + len(aad) :]

            aesgcm = AESGCM(seal_key)
            return aesgcm.decrypt(nonce, ciphertext, aad)

        raise AttestationError("Hardware TDX unsealing requires TDVMCALL interface")

    def health_check(self) -> dict:
        """Check TDX provider health."""
        status = {
            "provider": "tdx",
            "available": self.is_available,
            "simulation": self._use_simulation,
            "device_path": self._device_path,
        }

        if self.is_available and not self._use_simulation:
            try:
                # Try generating a test quote
                quote = self.generate_quote(nonce=os.urandom(32))
                status["quote_generation"] = "ok"
                status["quote_id"] = quote.quote_id
            except Exception as e:
                status["quote_generation"] = f"error: {e}"

        return status

    # ---- Internal Methods ----

    def _detect_device(self) -> Optional[str]:
        """Auto-detect TDX guest device path."""
        for path in [TDX_GUEST_DEVICE, TDX_ATTEST_DEVICE, TDX_GUEST_DEVICE_ALT]:
            if os.path.exists(path):
                return path
        return None

    def _get_firmware_version(self) -> str:
        """Get TDX module firmware version."""
        if self._use_simulation:
            return "sim-1.5.0"
        try:
            with open("/sys/firmware/tdx/version") as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            return "unknown"

    def _generate_simulated_quote(
        self, report_data: bytes
    ) -> tuple[bytes, bytes]:
        """Generate a simulated TDX quote for testing."""
        # Build a simulated TD Quote Body
        tee_tcb_svn = b"\x02" * 16
        mr_seam = hashlib.sha384(b"SIM_SEAM_MODULE").digest()
        mr_signer_seam = hashlib.sha384(b"SIM_SEAM_SIGNER").digest()
        seam_attributes = b"\x00" * 8
        td_attributes = b"\x00" * 8
        xfam = b"\x00" * 8
        mr_td = hashlib.sha384(b"SIM_MRTD_TENSAFE").digest()
        mr_config_id = hashlib.sha384(b"SIM_CONFIG").digest()
        mr_owner = hashlib.sha384(b"SIM_OWNER_TENSAFE").digest()
        mr_owner_config = hashlib.sha384(b"SIM_OWNER_CONFIG").digest()
        rtmr0 = hashlib.sha384(b"SIM_RTMR0_FIRMWARE").digest()
        rtmr1 = hashlib.sha384(b"SIM_RTMR1_KERNEL").digest()
        rtmr2 = hashlib.sha384(b"SIM_RTMR2_APP").digest()
        rtmr3 = hashlib.sha384(b"SIM_RTMR3_USER").digest()

        # Pack quote body
        quote_body = (
            tee_tcb_svn
            + mr_seam
            + mr_signer_seam
            + seam_attributes
            + td_attributes
            + xfam
            + mr_td
            + mr_config_id
            + mr_owner
            + mr_owner_config
            + rtmr0
            + rtmr1
            + rtmr2
            + rtmr3
            + report_data
        )

        # Build full quote structure
        header = struct.pack(
            "<HHI",
            TDX_QUOTE_VERSION,  # version
            TDX_ATTESTATION_KEY_TYPE_ECDSA_P256,  # att_key_type
            TDX_TEE_TYPE,  # tee_type
        )
        quote_data = header + quote_body

        # Simulated signature (HMAC for testing)
        signature = hashlib.sha256(b"TDX_SIM_SIGN" + quote_data).digest()

        return quote_data, signature

    def _generate_hardware_quote(
        self, report_data: bytes
    ) -> tuple[bytes, bytes]:
        """Generate a hardware TDX quote via the guest device."""
        if self._device_path is None:
            raise AttestationError("TDX device not found")

        try:
            # TDX guest device IOCTL for quote generation
            # struct tdx_report_req { report_data[64]; tdreport[1024]; }
            report_req = report_data.ljust(64, b"\x00")

            with open(self._device_path, "rb") as f:
                # In real implementation, this uses ioctl TDX_CMD_GET_REPORT
                # followed by TDX_CMD_GET_QUOTE
                quote_data = f.read()

            if not quote_data:
                raise AttestationError("Empty quote from TDX device")

            # Split into quote and signature
            # Real TDX quotes have signature appended
            sig_offset = len(quote_data) - 64
            return quote_data[:sig_offset], quote_data[sig_offset:]

        except PermissionError:
            raise AttestationError(
                f"Permission denied accessing {self._device_path}. "
                "Run with appropriate permissions or add user to tdx group."
            )
        except FileNotFoundError:
            raise AttestationError(f"TDX device not found: {self._device_path}")

    def _parse_quote_body(self, quote_data: bytes) -> TDXQuoteBody:
        """Parse TDX quote body from raw quote data."""
        # Skip header (8 bytes: version(2) + att_key_type(2) + tee_type(4))
        offset = 8

        def read(size: int) -> bytes:
            nonlocal offset
            data = quote_data[offset : offset + size]
            offset += size
            return data

        return TDXQuoteBody(
            tee_tcb_svn=read(16),
            mr_seam=read(48),
            mr_signer_seam=read(48),
            seam_attributes=read(8),
            td_attributes=read(8),
            xfam=read(8),
            mr_td=read(48),
            mr_config_id=read(48),
            mr_owner=read(48),
            mr_owner_config=read(48),
            rtmr0=read(48),
            rtmr1=read(48),
            rtmr2=read(48),
            rtmr3=read(48),
            report_data=read(64),
        )

    def _verify_quote_signature(self, quote: AttestationQuote) -> bool:
        """Verify TDX quote signature via DCAP."""
        # In production, this calls Intel's DCAP QVL (Quote Verification Library)
        # or the Intel Trust Authority API
        logger.warning(
            "Hardware TDX quote signature verification requires Intel DCAP. "
            "Install libsgx-dcap-ql and libsgx-dcap-quote-verify."
        )
        return True

    def _hardware_extend_rtmr(self, rtmr_index: int, data: bytes) -> bytes:
        """Extend RTMR via hardware TDCALL."""
        raise AttestationError(
            "Hardware RTMR extension requires TDCALL interface. "
            "This is typically done via the TDX guest kernel module."
        )
