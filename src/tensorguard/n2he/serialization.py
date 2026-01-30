"""
Ciphertext Serialization Formats for N2HE.

Provides standardized serialization formats for HE ciphertexts, enabling:
    - Storage of encrypted compute artifacts
    - Network transmission of encrypted data
    - Reproducible cryptographic operations

This is part of architectural option 3: "Encrypted compute artifacts"

Format Design Principles:
    1. Self-describing - includes scheme params and metadata
    2. Compact - uses efficient binary encoding
    3. Versioned - supports format evolution
    4. Authenticated - includes integrity checks
    5. Interoperable - compatible with N2HE C++ library
"""

import base64
import hashlib
import json
import logging
import struct
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np

from .core import (
    Ciphertext,
    HESchemeParams,
    LWECiphertext,
    RLWECiphertext,
)

logger = logging.getLogger(__name__)


# Format magic number: "N2HE" in ASCII
MAGIC_NUMBER = b"N2HE"
FORMAT_VERSION = 1


class CiphertextFormat(Enum):
    """Supported ciphertext serialization formats."""

    BINARY = "binary"  # Compact binary format
    JSON = "json"  # Human-readable JSON
    BASE64 = "base64"  # Base64-encoded binary (for APIs)
    CBOR = "cbor"  # CBOR encoding (compact + typed)


@dataclass
class SerializedCiphertext:
    """
    Serialized ciphertext with metadata.

    Contains the serialized bytes along with metadata for
    deserialization and verification.
    """

    # Core data
    data: bytes
    format: CiphertextFormat

    # Metadata
    ciphertext_id: str = ""
    scheme_type: str = "lwe"
    params_hash: str = ""
    level: int = 0
    noise_budget: Optional[float] = None

    # Integrity
    content_hash: str = ""  # SHA-256 of data

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if not self.ciphertext_id:
            self.ciphertext_id = f"ct-{hashlib.sha256(self.data).hexdigest()[:12]}"
        if not self.content_hash:
            self.content_hash = f"sha256:{hashlib.sha256(self.data).hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "ciphertext_id": self.ciphertext_id,
            "format": self.format.value,
            "scheme_type": self.scheme_type,
            "params_hash": self.params_hash,
            "level": self.level,
            "noise_budget": self.noise_budget,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "size_bytes": len(self.data),
        }


class CiphertextSerializer:
    """
    Serializes and deserializes HE ciphertexts.

    Supports multiple formats and handles scheme-specific encoding.
    """

    def __init__(
        self,
        default_format: CiphertextFormat = CiphertextFormat.BINARY,
        compress: bool = True,
        compression_level: int = 6,
    ):
        """
        Initialize serializer.

        Args:
            default_format: Default serialization format
            compress: Whether to compress binary data
            compression_level: zlib compression level (1-9)
        """
        self.default_format = default_format
        self.compress = compress
        self.compression_level = compression_level

    def serialize(
        self,
        ciphertext: Ciphertext,
        format: Optional[CiphertextFormat] = None,
    ) -> SerializedCiphertext:
        """
        Serialize a ciphertext.

        Args:
            ciphertext: Ciphertext to serialize
            format: Output format (default: self.default_format)

        Returns:
            SerializedCiphertext
        """
        format = format or self.default_format

        if format == CiphertextFormat.BINARY:
            data = self._serialize_binary(ciphertext)
        elif format == CiphertextFormat.JSON:
            data = self._serialize_json(ciphertext)
        elif format == CiphertextFormat.BASE64:
            binary_data = self._serialize_binary(ciphertext)
            data = base64.b64encode(binary_data)
        elif format == CiphertextFormat.CBOR:
            data = self._serialize_cbor(ciphertext)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Determine scheme type
        if isinstance(ciphertext, LWECiphertext):
            scheme_type = "lwe"
        elif isinstance(ciphertext, RLWECiphertext):
            scheme_type = "rlwe"
        else:
            scheme_type = "unknown"

        return SerializedCiphertext(
            data=data,
            format=format,
            scheme_type=scheme_type,
            params_hash=ciphertext.params.get_hash(),
            level=ciphertext.level,
            noise_budget=ciphertext.noise_budget,
        )

    def deserialize(
        self,
        serialized: SerializedCiphertext,
        params: HESchemeParams,
    ) -> Ciphertext:
        """
        Deserialize a ciphertext.

        Args:
            serialized: Serialized ciphertext data
            params: HE scheme parameters

        Returns:
            Deserialized Ciphertext
        """
        # Verify params hash if present
        if serialized.params_hash and serialized.params_hash != params.get_hash():
            logger.warning(
                f"Parameter hash mismatch: expected {serialized.params_hash}, "
                f"got {params.get_hash()}"
            )

        if serialized.format == CiphertextFormat.BINARY:
            return self._deserialize_binary(serialized.data, params)
        elif serialized.format == CiphertextFormat.JSON:
            return self._deserialize_json(serialized.data, params)
        elif serialized.format == CiphertextFormat.BASE64:
            binary_data = base64.b64decode(serialized.data)
            return self._deserialize_binary(binary_data, params)
        elif serialized.format == CiphertextFormat.CBOR:
            return self._deserialize_cbor(serialized.data, params)
        else:
            raise ValueError(f"Unsupported format: {serialized.format}")

    def _serialize_binary(self, ciphertext: Ciphertext) -> bytes:
        """Serialize to compact binary format."""
        buf = BytesIO()

        # Header
        buf.write(MAGIC_NUMBER)
        buf.write(struct.pack(">B", FORMAT_VERSION))

        # Scheme type
        if isinstance(ciphertext, LWECiphertext):
            buf.write(struct.pack(">B", 0))  # LWE = 0
            self._write_lwe_binary(buf, ciphertext)
        elif isinstance(ciphertext, RLWECiphertext):
            buf.write(struct.pack(">B", 1))  # RLWE = 1
            self._write_rlwe_binary(buf, ciphertext)
        else:
            raise ValueError(f"Unknown ciphertext type: {type(ciphertext)}")

        data = buf.getvalue()

        # Optionally compress
        if self.compress:
            compressed = zlib.compress(data, self.compression_level)
            if len(compressed) < len(data):
                # Prepend compression flag
                data = b"\x01" + compressed
            else:
                data = b"\x00" + data
        else:
            data = b"\x00" + data

        return data

    def _deserialize_binary(
        self, data: bytes, params: HESchemeParams
    ) -> Ciphertext:
        """Deserialize from binary format."""
        # Check compression flag
        compressed = data[0] == 0x01
        data = data[1:]

        if compressed:
            data = zlib.decompress(data)

        buf = BytesIO(data)

        # Read header
        magic = buf.read(4)
        if magic != MAGIC_NUMBER:
            raise ValueError(f"Invalid magic number: {magic}")

        version = struct.unpack(">B", buf.read(1))[0]
        if version > FORMAT_VERSION:
            raise ValueError(f"Unsupported format version: {version}")

        # Read scheme type
        scheme_type = struct.unpack(">B", buf.read(1))[0]

        if scheme_type == 0:  # LWE
            return self._read_lwe_binary(buf, params)
        elif scheme_type == 1:  # RLWE
            return self._read_rlwe_binary(buf, params)
        else:
            raise ValueError(f"Unknown scheme type: {scheme_type}")

    def _write_lwe_binary(self, buf: BytesIO, ct: LWECiphertext) -> None:
        """Write LWE ciphertext to binary buffer."""
        # Level and noise budget
        buf.write(struct.pack(">B", ct.level))
        buf.write(struct.pack(">f", ct.noise_budget or 0.0))

        # Dimensions
        n = len(ct.a)
        buf.write(struct.pack(">I", n))

        # b value
        buf.write(struct.pack(">q", ct.b))

        # a vector
        buf.write(ct.a.astype(np.int32).tobytes())

    def _read_lwe_binary(
        self, buf: BytesIO, params: HESchemeParams
    ) -> LWECiphertext:
        """Read LWE ciphertext from binary buffer."""
        level = struct.unpack(">B", buf.read(1))[0]
        noise_budget = struct.unpack(">f", buf.read(4))[0]

        n = struct.unpack(">I", buf.read(4))[0]
        b = struct.unpack(">q", buf.read(8))[0]

        a = np.frombuffer(buf.read(n * 4), dtype=np.int32)

        return LWECiphertext(
            a=a,
            b=b,
            params=params,
            noise_budget=noise_budget if noise_budget > 0 else None,
            level=level,
        )

    def _write_rlwe_binary(self, buf: BytesIO, ct: RLWECiphertext) -> None:
        """Write RLWE ciphertext to binary buffer."""
        # Level, noise budget, scale
        buf.write(struct.pack(">B", ct.level))
        buf.write(struct.pack(">f", ct.noise_budget or 0.0))
        buf.write(struct.pack(">d", ct.scale))

        # Dimension
        n = len(ct.c0)
        buf.write(struct.pack(">I", n))

        # Polynomials
        buf.write(ct.c0.astype(np.int64).tobytes())
        buf.write(ct.c1.astype(np.int64).tobytes())

    def _read_rlwe_binary(
        self, buf: BytesIO, params: HESchemeParams
    ) -> RLWECiphertext:
        """Read RLWE ciphertext from binary buffer."""
        level = struct.unpack(">B", buf.read(1))[0]
        noise_budget = struct.unpack(">f", buf.read(4))[0]
        scale = struct.unpack(">d", buf.read(8))[0]

        n = struct.unpack(">I", buf.read(4))[0]

        c0 = np.frombuffer(buf.read(n * 8), dtype=np.int64)
        c1 = np.frombuffer(buf.read(n * 8), dtype=np.int64)

        return RLWECiphertext(
            c0=c0,
            c1=c1,
            params=params,
            noise_budget=noise_budget if noise_budget > 0 else None,
            level=level,
            scale=scale,
        )

    def _serialize_json(self, ciphertext: Ciphertext) -> bytes:
        """Serialize to JSON format."""
        data = {
            "format": "n2he-json-v1",
            "timestamp": datetime.utcnow().isoformat(),
        }

        if isinstance(ciphertext, LWECiphertext):
            data["type"] = "lwe"
            data["level"] = ciphertext.level
            data["noise_budget"] = ciphertext.noise_budget
            data["b"] = ciphertext.b
            data["a"] = base64.b64encode(
                ciphertext.a.astype(np.int32).tobytes()
            ).decode("ascii")
            data["a_len"] = len(ciphertext.a)

        elif isinstance(ciphertext, RLWECiphertext):
            data["type"] = "rlwe"
            data["level"] = ciphertext.level
            data["noise_budget"] = ciphertext.noise_budget
            data["scale"] = ciphertext.scale
            data["c0"] = base64.b64encode(
                ciphertext.c0.astype(np.int64).tobytes()
            ).decode("ascii")
            data["c1"] = base64.b64encode(
                ciphertext.c1.astype(np.int64).tobytes()
            ).decode("ascii")
            data["n"] = len(ciphertext.c0)

        return json.dumps(data, indent=2).encode("utf-8")

    def _deserialize_json(
        self, data: bytes, params: HESchemeParams
    ) -> Ciphertext:
        """Deserialize from JSON format."""
        obj = json.loads(data.decode("utf-8"))

        ct_type = obj.get("type")
        if ct_type == "lwe":
            a = np.frombuffer(
                base64.b64decode(obj["a"]), dtype=np.int32
            )
            return LWECiphertext(
                a=a,
                b=obj["b"],
                params=params,
                level=obj.get("level", 0),
                noise_budget=obj.get("noise_budget"),
            )
        elif ct_type == "rlwe":
            c0 = np.frombuffer(
                base64.b64decode(obj["c0"]), dtype=np.int64
            )
            c1 = np.frombuffer(
                base64.b64decode(obj["c1"]), dtype=np.int64
            )
            return RLWECiphertext(
                c0=c0,
                c1=c1,
                params=params,
                level=obj.get("level", 0),
                noise_budget=obj.get("noise_budget"),
                scale=obj.get("scale", 1.0),
            )
        else:
            raise ValueError(f"Unknown ciphertext type: {ct_type}")

    def _serialize_cbor(self, ciphertext: Ciphertext) -> bytes:
        """Serialize to CBOR format."""
        # Fallback to JSON if cbor2 not available
        try:
            import cbor2

            data = {}
            if isinstance(ciphertext, LWECiphertext):
                data = {
                    "t": 0,  # LWE
                    "l": ciphertext.level,
                    "n": ciphertext.noise_budget,
                    "b": ciphertext.b,
                    "a": ciphertext.a.astype(np.int32).tobytes(),
                }
            elif isinstance(ciphertext, RLWECiphertext):
                data = {
                    "t": 1,  # RLWE
                    "l": ciphertext.level,
                    "n": ciphertext.noise_budget,
                    "s": ciphertext.scale,
                    "c0": ciphertext.c0.astype(np.int64).tobytes(),
                    "c1": ciphertext.c1.astype(np.int64).tobytes(),
                }
            return cbor2.dumps(data)

        except ImportError:
            logger.warning("cbor2 not available, falling back to JSON")
            return self._serialize_json(ciphertext)

    def _deserialize_cbor(
        self, data: bytes, params: HESchemeParams
    ) -> Ciphertext:
        """Deserialize from CBOR format."""
        try:
            import cbor2

            obj = cbor2.loads(data)
            ct_type = obj.get("t")

            if ct_type == 0:  # LWE
                a = np.frombuffer(obj["a"], dtype=np.int32)
                return LWECiphertext(
                    a=a,
                    b=obj["b"],
                    params=params,
                    level=obj.get("l", 0),
                    noise_budget=obj.get("n"),
                )
            elif ct_type == 1:  # RLWE
                c0 = np.frombuffer(obj["c0"], dtype=np.int64)
                c1 = np.frombuffer(obj["c1"], dtype=np.int64)
                return RLWECiphertext(
                    c0=c0,
                    c1=c1,
                    params=params,
                    level=obj.get("l", 0),
                    noise_budget=obj.get("n"),
                    scale=obj.get("s", 1.0),
                )
            else:
                raise ValueError(f"Unknown ciphertext type: {ct_type}")

        except ImportError:
            return self._deserialize_json(data, params)


# Global serializer instance
_serializer = CiphertextSerializer()


def serialize_ciphertext(
    ciphertext: Ciphertext,
    format: CiphertextFormat = CiphertextFormat.BINARY,
) -> SerializedCiphertext:
    """
    Serialize a ciphertext.

    Args:
        ciphertext: Ciphertext to serialize
        format: Output format

    Returns:
        SerializedCiphertext
    """
    return _serializer.serialize(ciphertext, format)


def deserialize_ciphertext(
    serialized: SerializedCiphertext,
    params: HESchemeParams,
) -> Ciphertext:
    """
    Deserialize a ciphertext.

    Args:
        serialized: Serialized data
        params: HE scheme parameters

    Returns:
        Deserialized Ciphertext
    """
    return _serializer.deserialize(serialized, params)


@dataclass
class CiphertextBundle:
    """
    Bundle of related ciphertexts for batch operations.

    Useful for storing encrypted model states, gradients, etc.
    """

    bundle_id: str
    ciphertexts: List[SerializedCiphertext]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_total_size(self) -> int:
        """Get total size of all ciphertexts."""
        return sum(len(ct.data) for ct in self.ciphertexts)

    def get_content_hash(self) -> str:
        """Get hash of entire bundle."""
        h = hashlib.sha256()
        h.update(self.bundle_id.encode())
        for ct in self.ciphertexts:
            h.update(ct.data)
        return f"sha256:{h.hexdigest()}"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize bundle metadata."""
        return {
            "bundle_id": self.bundle_id,
            "num_ciphertexts": len(self.ciphertexts),
            "total_size_bytes": self.get_total_size(),
            "content_hash": self.get_content_hash(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "ciphertext_ids": [ct.ciphertext_id for ct in self.ciphertexts],
        }


def create_ciphertext_bundle(
    ciphertexts: List[Ciphertext],
    bundle_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    format: CiphertextFormat = CiphertextFormat.BINARY,
) -> CiphertextBundle:
    """
    Create a bundle of serialized ciphertexts.

    Args:
        ciphertexts: List of ciphertexts to bundle
        bundle_id: Optional bundle ID
        metadata: Optional metadata
        format: Serialization format

    Returns:
        CiphertextBundle
    """
    if bundle_id is None:
        bundle_id = f"bundle-{hashlib.sha256(str(datetime.utcnow()).encode()).hexdigest()[:12]}"

    serialized = [serialize_ciphertext(ct, format) for ct in ciphertexts]

    return CiphertextBundle(
        bundle_id=bundle_id,
        ciphertexts=serialized,
        metadata=metadata or {},
    )
