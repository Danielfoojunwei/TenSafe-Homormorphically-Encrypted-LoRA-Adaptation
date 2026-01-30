"""
Tests for N2HE ciphertext serialization.

Tests binary, JSON, and CBOR serialization formats.
"""

import numpy as np
import pytest

from tensorguard.n2he.core import (
    HESchemeParams,
    LWECiphertext,
    RLWECiphertext,
    create_context,
)
from tensorguard.n2he.serialization import (
    CiphertextFormat,
    CiphertextSerializer,
    SerializedCiphertext,
    serialize_ciphertext,
    deserialize_ciphertext,
    create_ciphertext_bundle,
)


class TestCiphertextSerializer:
    """Tests for ciphertext serializer."""

    @pytest.fixture
    def serializer(self):
        """Create a serializer."""
        return CiphertextSerializer()

    @pytest.fixture
    def lwe_ciphertext(self):
        """Create a test LWE ciphertext."""
        ctx = create_context()
        ctx.generate_keys()
        return ctx.encrypt(np.array([42], dtype=np.int64))

    def test_binary_serialization(self, serializer, lwe_ciphertext):
        """Test binary format serialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.BINARY)

        assert isinstance(serialized, SerializedCiphertext)
        assert serialized.format == CiphertextFormat.BINARY
        assert len(serialized.data) > 0
        assert serialized.content_hash.startswith("sha256:")

    def test_binary_deserialization(self, serializer, lwe_ciphertext):
        """Test binary format deserialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.BINARY)
        deserialized = serializer.deserialize(serialized, lwe_ciphertext.params)

        assert isinstance(deserialized, LWECiphertext)
        assert np.array_equal(deserialized.a, lwe_ciphertext.a)
        assert deserialized.b == lwe_ciphertext.b

    def test_json_serialization(self, serializer, lwe_ciphertext):
        """Test JSON format serialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.JSON)

        assert serialized.format == CiphertextFormat.JSON
        # Should be valid JSON
        import json
        json.loads(serialized.data.decode("utf-8"))

    def test_json_deserialization(self, serializer, lwe_ciphertext):
        """Test JSON format deserialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.JSON)
        deserialized = serializer.deserialize(serialized, lwe_ciphertext.params)

        assert isinstance(deserialized, LWECiphertext)
        assert np.array_equal(deserialized.a, lwe_ciphertext.a)

    def test_base64_serialization(self, serializer, lwe_ciphertext):
        """Test base64 format serialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.BASE64)

        assert serialized.format == CiphertextFormat.BASE64
        # Should be valid base64
        import base64
        base64.b64decode(serialized.data)

    def test_base64_deserialization(self, serializer, lwe_ciphertext):
        """Test base64 format deserialization."""
        serialized = serializer.serialize(lwe_ciphertext, CiphertextFormat.BASE64)
        deserialized = serializer.deserialize(serialized, lwe_ciphertext.params)

        assert isinstance(deserialized, LWECiphertext)

    def test_compression(self, lwe_ciphertext):
        """Test that compression reduces size for large ciphertexts."""
        serializer_compressed = CiphertextSerializer(compress=True)
        serializer_uncompressed = CiphertextSerializer(compress=False)

        compressed = serializer_compressed.serialize(
            lwe_ciphertext, CiphertextFormat.BINARY
        )
        uncompressed = serializer_uncompressed.serialize(
            lwe_ciphertext, CiphertextFormat.BINARY
        )

        # Compressed should be smaller (or equal for small data)
        assert len(compressed.data) <= len(uncompressed.data) + 10


class TestRLWESerialization:
    """Tests for RLWE ciphertext serialization."""

    @pytest.fixture
    def rlwe_ciphertext(self):
        """Create a test RLWE ciphertext."""
        params = HESchemeParams()
        n = params.poly_degree
        c0 = np.random.randint(0, 1000, size=n, dtype=np.int64)
        c1 = np.random.randint(0, 1000, size=n, dtype=np.int64)
        return RLWECiphertext(c0=c0, c1=c1, params=params, scale=2.5)

    def test_binary_roundtrip(self, rlwe_ciphertext):
        """Test RLWE binary serialization roundtrip."""
        serializer = CiphertextSerializer()

        serialized = serializer.serialize(rlwe_ciphertext, CiphertextFormat.BINARY)
        deserialized = serializer.deserialize(serialized, rlwe_ciphertext.params)

        assert isinstance(deserialized, RLWECiphertext)
        assert np.array_equal(deserialized.c0, rlwe_ciphertext.c0)
        assert np.array_equal(deserialized.c1, rlwe_ciphertext.c1)
        assert deserialized.scale == rlwe_ciphertext.scale

    def test_json_roundtrip(self, rlwe_ciphertext):
        """Test RLWE JSON serialization roundtrip."""
        serializer = CiphertextSerializer()

        serialized = serializer.serialize(rlwe_ciphertext, CiphertextFormat.JSON)
        deserialized = serializer.deserialize(serialized, rlwe_ciphertext.params)

        assert isinstance(deserialized, RLWECiphertext)
        assert np.array_equal(deserialized.c0, rlwe_ciphertext.c0)


class TestSerializedCiphertext:
    """Tests for SerializedCiphertext dataclass."""

    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        serialized = SerializedCiphertext(
            data=b"test data",
            format=CiphertextFormat.BINARY,
        )

        assert serialized.ciphertext_id.startswith("ct-")

    def test_content_hash_generation(self):
        """Test automatic content hash generation."""
        serialized = SerializedCiphertext(
            data=b"test data",
            format=CiphertextFormat.BINARY,
        )

        assert serialized.content_hash.startswith("sha256:")

    def test_to_dict(self):
        """Test metadata serialization."""
        serialized = SerializedCiphertext(
            data=b"test data",
            format=CiphertextFormat.BINARY,
            scheme_type="lwe",
            level=1,
        )

        data = serialized.to_dict()
        assert data["format"] == "binary"
        assert data["scheme_type"] == "lwe"
        assert data["level"] == 1
        assert data["size_bytes"] == len(b"test data")


class TestCiphertextBundle:
    """Tests for ciphertext bundles."""

    @pytest.fixture
    def ciphertexts(self):
        """Create test ciphertexts."""
        ctx = create_context()
        ctx.generate_keys()
        return [
            ctx.encrypt(np.array([i], dtype=np.int64))
            for i in range(3)
        ]

    def test_bundle_creation(self, ciphertexts):
        """Test bundle creation."""
        bundle = create_ciphertext_bundle(
            ciphertexts,
            bundle_id="test-bundle",
            metadata={"purpose": "testing"},
        )

        assert bundle.bundle_id == "test-bundle"
        assert len(bundle.ciphertexts) == 3
        assert bundle.metadata["purpose"] == "testing"

    def test_bundle_total_size(self, ciphertexts):
        """Test bundle size calculation."""
        bundle = create_ciphertext_bundle(ciphertexts)
        total_size = bundle.get_total_size()

        assert total_size > 0
        assert total_size == sum(len(ct.data) for ct in bundle.ciphertexts)

    def test_bundle_content_hash(self, ciphertexts):
        """Test bundle content hash."""
        bundle = create_ciphertext_bundle(ciphertexts)
        content_hash = bundle.get_content_hash()

        assert content_hash.startswith("sha256:")

    def test_bundle_to_dict(self, ciphertexts):
        """Test bundle metadata serialization."""
        bundle = create_ciphertext_bundle(
            ciphertexts,
            bundle_id="test-bundle",
        )

        data = bundle.to_dict()
        assert data["bundle_id"] == "test-bundle"
        assert data["num_ciphertexts"] == 3
        assert len(data["ciphertext_ids"]) == 3


class TestGlobalFunctions:
    """Tests for module-level convenience functions."""

    def test_serialize_ciphertext(self):
        """Test global serialize function."""
        ctx = create_context()
        ctx.generate_keys()
        ct = ctx.encrypt(np.array([42], dtype=np.int64))

        serialized = serialize_ciphertext(ct)
        assert isinstance(serialized, SerializedCiphertext)

    def test_deserialize_ciphertext(self):
        """Test global deserialize function."""
        ctx = create_context()
        ctx.generate_keys()
        ct = ctx.encrypt(np.array([42], dtype=np.int64))

        serialized = serialize_ciphertext(ct)
        deserialized = deserialize_ciphertext(serialized, ct.params)

        assert isinstance(deserialized, LWECiphertext)
