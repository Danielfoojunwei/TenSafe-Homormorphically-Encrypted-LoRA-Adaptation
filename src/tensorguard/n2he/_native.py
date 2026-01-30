"""
Native N2HE Library Integration.

This module provides Python bindings to the N2HE C++ library for
production-grade homomorphic encryption operations.

The N2HE library provides:
- LWE-based additive homomorphic encryption for weighted operations
- FHEW ciphertexts for non-polynomial activation functions
- FasterNTT-accelerated polynomial operations

Installation:
    1. Clone N2HE: git clone https://github.com/HintSight-Technology/N2HE.git
    2. Build library: cd N2HE && mkdir build && cd build && cmake .. && make
    3. Install to system or set N2HE_LIB_PATH environment variable

Usage:
    from tensorguard.n2he._native import NativeN2HEScheme

    scheme = NativeN2HEScheme(params)
    sk, pk, ek = scheme.keygen()
    ct = scheme.encrypt(pk, plaintext)
    result = scheme.matmul(ct, weights, ek)
    pt = scheme.decrypt(sk, ct)
"""

import ctypes
import logging
import os
import struct
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_double,
    c_int,
    c_int32,
    c_int64,
    c_size_t,
    c_uint64,
    c_void_p,
    byref,
)
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .core import (
    HESchemeParams,
    LWECiphertext,
    N2HEScheme,
)

logger = logging.getLogger(__name__)


# Library search paths
N2HE_LIB_PATHS = [
    os.environ.get("N2HE_LIB_PATH", ""),
    "/usr/local/lib/libn2he.so",
    "/usr/lib/libn2he.so",
    str(Path.home() / "N2HE" / "build" / "libn2he.so"),
    str(Path.home() / ".local" / "lib" / "libn2he.so"),
    "./libn2he.so",
]


class N2HELibraryNotFoundError(Exception):
    """Raised when the N2HE native library cannot be loaded."""

    pass


class N2HEError(Exception):
    """Raised when an N2HE operation fails."""

    pass


# C structure definitions for N2HE interop
class LWEKey(Structure):
    """LWE secret key structure."""

    _fields_ = [
        ("n", c_int),
        ("data", POINTER(c_int32)),
    ]


class LWEPublicKey(Structure):
    """LWE public key structure."""

    _fields_ = [
        ("n", c_int),
        ("a_data", POINTER(c_int64)),
        ("b", c_int64),
    ]


class LWECiphertextC(Structure):
    """LWE ciphertext structure for C interop."""

    _fields_ = [
        ("n", c_int),
        ("a", POINTER(c_int64)),
        ("b", c_int64),
        ("level", c_int),
    ]


class RLWECiphertextC(Structure):
    """RLWE ciphertext structure for C interop."""

    _fields_ = [
        ("n", c_int),
        ("c0", POINTER(c_int64)),
        ("c1", POINTER(c_int64)),
        ("level", c_int),
        ("scale", c_double),
    ]


class N2HEParams(Structure):
    """N2HE scheme parameters for C interop."""

    _fields_ = [
        ("n", c_int),  # Lattice dimension
        ("q", c_uint64),  # Ciphertext modulus
        ("t", c_uint64),  # Plaintext modulus
        ("std_dev", c_double),  # Noise standard deviation
        ("poly_degree", c_int),  # Polynomial ring degree
        ("security_level", c_int),  # Security bits
    ]


def _find_library() -> Optional[str]:
    """Find the N2HE shared library."""
    for path in N2HE_LIB_PATHS:
        if path and os.path.exists(path):
            return path

    # Try to find via ldconfig
    try:
        import subprocess

        result = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if "n2he" in line.lower():
                parts = line.split("=>")
                if len(parts) > 1:
                    return parts[1].strip()
    except Exception:
        pass

    return None


def _load_library() -> ctypes.CDLL:
    """Load the N2HE shared library."""
    lib_path = _find_library()

    if lib_path is None:
        raise N2HELibraryNotFoundError(
            "N2HE native library not found. Please install N2HE:\n"
            "  1. git clone https://github.com/HintSight-Technology/N2HE.git\n"
            "  2. cd N2HE && mkdir build && cd build && cmake .. && make\n"
            "  3. sudo make install  (or set N2HE_LIB_PATH)\n"
            "\n"
            "For development, use SimulatedN2HEScheme instead."
        )

    try:
        lib = ctypes.CDLL(lib_path)
        logger.info(f"Loaded N2HE library from {lib_path}")
        return lib
    except OSError as e:
        raise N2HELibraryNotFoundError(
            f"Failed to load N2HE library from {lib_path}: {e}"
        )


class NativeN2HEScheme(N2HEScheme):
    """
    Native N2HE scheme using the C++ library.

    Provides production-grade HE operations with hardware acceleration
    (HEXL, FasterNTT) when available.
    """

    def __init__(self, params: Optional[HESchemeParams] = None):
        """
        Initialize native N2HE scheme.

        Args:
            params: HE scheme parameters

        Raises:
            N2HELibraryNotFoundError: If native library not available
        """
        self.params = params or HESchemeParams.default_lora_params()
        self._lib = _load_library()
        self._setup_function_signatures()
        self._ctx = self._create_context()

        logger.info(
            f"Initialized NativeN2HEScheme: n={self.params.n}, "
            f"security_level={self.params.security_level}"
        )

    def _setup_function_signatures(self) -> None:
        """Configure ctypes function signatures for N2HE library."""
        lib = self._lib

        # Context management
        lib.n2he_create_context.argtypes = [POINTER(N2HEParams)]
        lib.n2he_create_context.restype = c_void_p

        lib.n2he_destroy_context.argtypes = [c_void_p]
        lib.n2he_destroy_context.restype = None

        # Key generation
        lib.n2he_keygen.argtypes = [
            c_void_p,  # context
            POINTER(c_void_p),  # secret_key out
            POINTER(c_void_p),  # public_key out
            POINTER(c_void_p),  # eval_key out
        ]
        lib.n2he_keygen.restype = c_int

        # Encryption
        lib.n2he_encrypt.argtypes = [
            c_void_p,  # context
            c_void_p,  # public_key
            POINTER(c_int64),  # plaintext
            c_size_t,  # plaintext_len
            POINTER(c_void_p),  # ciphertext out
        ]
        lib.n2he_encrypt.restype = c_int

        # Decryption
        lib.n2he_decrypt.argtypes = [
            c_void_p,  # context
            c_void_p,  # secret_key
            c_void_p,  # ciphertext
            POINTER(c_int64),  # plaintext out
            c_size_t,  # max_len
        ]
        lib.n2he_decrypt.restype = c_int

        # Homomorphic operations
        lib.n2he_add.argtypes = [
            c_void_p,  # context
            c_void_p,  # ct1
            c_void_p,  # ct2
            POINTER(c_void_p),  # result out
        ]
        lib.n2he_add.restype = c_int

        lib.n2he_multiply_plain.argtypes = [
            c_void_p,  # context
            c_void_p,  # ciphertext
            POINTER(c_int64),  # plaintext
            c_size_t,  # len
            POINTER(c_void_p),  # result out
        ]
        lib.n2he_multiply_plain.restype = c_int

        lib.n2he_matmul.argtypes = [
            c_void_p,  # context
            c_void_p,  # ciphertext
            POINTER(c_double),  # weight matrix
            c_int,  # rows
            c_int,  # cols
            c_void_p,  # eval_key
            POINTER(c_void_p),  # result out
        ]
        lib.n2he_matmul.restype = c_int

        # Serialization
        lib.n2he_serialize_ciphertext.argtypes = [
            c_void_p,  # ciphertext
            POINTER(c_char_p),  # data out
            POINTER(c_size_t),  # len out
        ]
        lib.n2he_serialize_ciphertext.restype = c_int

        lib.n2he_deserialize_ciphertext.argtypes = [
            c_void_p,  # context
            c_char_p,  # data
            c_size_t,  # len
            POINTER(c_void_p),  # ciphertext out
        ]
        lib.n2he_deserialize_ciphertext.restype = c_int

        # Noise budget estimation
        lib.n2he_get_noise_budget.argtypes = [
            c_void_p,  # context
            c_void_p,  # ciphertext
        ]
        lib.n2he_get_noise_budget.restype = c_double

    def _create_context(self) -> c_void_p:
        """Create N2HE context with parameters."""
        c_params = N2HEParams(
            n=self.params.n,
            q=self.params.q,
            t=self.params.t,
            std_dev=self.params.std_dev,
            poly_degree=self.params.poly_degree,
            security_level=self.params.security_level,
        )

        ctx = self._lib.n2he_create_context(byref(c_params))
        if ctx is None:
            raise N2HEError("Failed to create N2HE context")

        return ctx

    def keygen(self) -> Tuple[bytes, bytes, bytes]:
        """Generate key triple (secret_key, public_key, eval_key)."""
        sk_ptr = c_void_p()
        pk_ptr = c_void_p()
        ek_ptr = c_void_p()

        result = self._lib.n2he_keygen(
            self._ctx,
            byref(sk_ptr),
            byref(pk_ptr),
            byref(ek_ptr),
        )

        if result != 0:
            raise N2HEError(f"Key generation failed with code {result}")

        # Serialize keys to bytes
        sk = self._serialize_key(sk_ptr, "secret")
        pk = self._serialize_key(pk_ptr, "public")
        ek = self._serialize_key(ek_ptr, "eval")

        return sk, pk, ek

    def _serialize_key(self, key_ptr: c_void_p, key_type: str) -> bytes:
        """Serialize a key to bytes."""
        data_ptr = c_char_p()
        data_len = c_size_t()

        # Use appropriate serialization function based on key type
        if key_type == "secret":
            fn = self._lib.n2he_serialize_secret_key
        elif key_type == "public":
            fn = self._lib.n2he_serialize_public_key
        else:
            fn = self._lib.n2he_serialize_eval_key

        result = fn(key_ptr, byref(data_ptr), byref(data_len))
        if result != 0:
            raise N2HEError(f"Failed to serialize {key_type} key")

        return ctypes.string_at(data_ptr, data_len.value)

    def _deserialize_key(self, key_bytes: bytes, key_type: str) -> c_void_p:
        """Deserialize a key from bytes."""
        key_ptr = c_void_p()

        if key_type == "secret":
            fn = self._lib.n2he_deserialize_secret_key
        elif key_type == "public":
            fn = self._lib.n2he_deserialize_public_key
        else:
            fn = self._lib.n2he_deserialize_eval_key

        result = fn(
            self._ctx,
            key_bytes,
            len(key_bytes),
            byref(key_ptr),
        )

        if result != 0:
            raise N2HEError(f"Failed to deserialize {key_type} key")

        return key_ptr

    def encrypt(self, pk: bytes, plaintext: np.ndarray) -> LWECiphertext:
        """Encrypt a plaintext vector."""
        pk_ptr = self._deserialize_key(pk, "public")

        # Prepare plaintext
        pt_data = plaintext.astype(np.int64)
        pt_ptr = pt_data.ctypes.data_as(POINTER(c_int64))

        ct_ptr = c_void_p()
        result = self._lib.n2he_encrypt(
            self._ctx,
            pk_ptr,
            pt_ptr,
            len(pt_data),
            byref(ct_ptr),
        )

        if result != 0:
            raise N2HEError(f"Encryption failed with code {result}")

        # Get noise budget
        noise_budget = self._lib.n2he_get_noise_budget(self._ctx, ct_ptr)

        # Convert to our ciphertext type
        return self._ct_ptr_to_lwe(ct_ptr, noise_budget)

    def _ct_ptr_to_lwe(
        self, ct_ptr: c_void_p, noise_budget: float
    ) -> LWECiphertext:
        """Convert C ciphertext pointer to LWECiphertext."""
        # Serialize and parse
        data_ptr = c_char_p()
        data_len = c_size_t()

        self._lib.n2he_serialize_ciphertext(
            ct_ptr, byref(data_ptr), byref(data_len)
        )

        data = ctypes.string_at(data_ptr, data_len.value)

        # Parse: [n:4][level:4][b:8][a:n*8]
        n = struct.unpack("<I", data[:4])[0]
        level = struct.unpack("<I", data[4:8])[0]
        b = struct.unpack("<q", data[8:16])[0]
        a = np.frombuffer(data[16 : 16 + n * 8], dtype=np.int64)

        return LWECiphertext(
            a=a.astype(np.int32),
            b=b,
            params=self.params,
            noise_budget=noise_budget,
            level=level,
        )

    def _lwe_to_ct_ptr(self, ct: LWECiphertext) -> c_void_p:
        """Convert LWECiphertext to C pointer."""
        # Serialize our format
        n = len(ct.a)
        data = (
            struct.pack("<I", n)
            + struct.pack("<I", ct.level)
            + struct.pack("<q", ct.b)
            + ct.a.astype(np.int64).tobytes()
        )

        ct_ptr = c_void_p()
        result = self._lib.n2he_deserialize_ciphertext(
            self._ctx, data, len(data), byref(ct_ptr)
        )

        if result != 0:
            raise N2HEError("Failed to deserialize ciphertext")

        return ct_ptr

    def decrypt(self, sk: bytes, ciphertext: LWECiphertext) -> np.ndarray:
        """Decrypt a ciphertext."""
        sk_ptr = self._deserialize_key(sk, "secret")
        ct_ptr = self._lwe_to_ct_ptr(ciphertext)

        # Allocate output buffer
        max_len = self.params.n
        pt_buffer = (c_int64 * max_len)()

        result = self._lib.n2he_decrypt(
            self._ctx,
            sk_ptr,
            ct_ptr,
            pt_buffer,
            max_len,
        )

        if result < 0:
            raise N2HEError(f"Decryption failed with code {result}")

        return np.array(pt_buffer[:result], dtype=np.int64)

    def add(
        self, ct1: LWECiphertext, ct2: LWECiphertext
    ) -> LWECiphertext:
        """Homomorphic addition of two ciphertexts."""
        ct1_ptr = self._lwe_to_ct_ptr(ct1)
        ct2_ptr = self._lwe_to_ct_ptr(ct2)

        result_ptr = c_void_p()
        result = self._lib.n2he_add(
            self._ctx, ct1_ptr, ct2_ptr, byref(result_ptr)
        )

        if result != 0:
            raise N2HEError(f"Addition failed with code {result}")

        noise_budget = self._lib.n2he_get_noise_budget(self._ctx, result_ptr)
        return self._ct_ptr_to_lwe(result_ptr, noise_budget)

    def multiply(
        self, ct: LWECiphertext, plaintext: np.ndarray
    ) -> LWECiphertext:
        """Multiply ciphertext by plaintext."""
        ct_ptr = self._lwe_to_ct_ptr(ct)

        pt_data = plaintext.astype(np.int64)
        pt_ptr = pt_data.ctypes.data_as(POINTER(c_int64))

        result_ptr = c_void_p()
        result = self._lib.n2he_multiply_plain(
            self._ctx, ct_ptr, pt_ptr, len(pt_data), byref(result_ptr)
        )

        if result != 0:
            raise N2HEError(f"Multiplication failed with code {result}")

        noise_budget = self._lib.n2he_get_noise_budget(self._ctx, result_ptr)
        return self._ct_ptr_to_lwe(result_ptr, noise_budget)

    def matmul(
        self,
        ct: LWECiphertext,
        weight_matrix: np.ndarray,
        ek: bytes,
    ) -> LWECiphertext:
        """
        Encrypted matrix multiplication: ct @ W^T.

        This is the core operation for computing LoRA deltas.
        Uses key-switching and rotations for efficient computation.

        Args:
            ct: Encrypted activation vector
            weight_matrix: Plaintext weight matrix
            ek: Evaluation key for key-switching

        Returns:
            Encrypted result
        """
        ct_ptr = self._lwe_to_ct_ptr(ct)
        ek_ptr = self._deserialize_key(ek, "eval")

        # Prepare weight matrix (row-major, double precision)
        weights = weight_matrix.astype(np.float64)
        rows, cols = weights.shape
        w_ptr = weights.ctypes.data_as(POINTER(c_double))

        result_ptr = c_void_p()
        result = self._lib.n2he_matmul(
            self._ctx,
            ct_ptr,
            w_ptr,
            rows,
            cols,
            ek_ptr,
            byref(result_ptr),
        )

        if result != 0:
            raise N2HEError(f"Matrix multiplication failed with code {result}")

        noise_budget = self._lib.n2he_get_noise_budget(self._ctx, result_ptr)
        return self._ct_ptr_to_lwe(result_ptr, noise_budget)

    def __del__(self):
        """Clean up native resources."""
        if hasattr(self, "_ctx") and self._ctx:
            try:
                self._lib.n2he_destroy_context(self._ctx)
            except Exception:
                pass


def is_native_available() -> bool:
    """Check if native N2HE library is available."""
    try:
        _load_library()
        return True
    except N2HELibraryNotFoundError:
        return False


def get_native_version() -> Optional[str]:
    """Get native N2HE library version if available."""
    try:
        lib = _load_library()
        if hasattr(lib, "n2he_version"):
            lib.n2he_version.restype = c_char_p
            return lib.n2he_version().decode("utf-8")
        return "unknown"
    except N2HELibraryNotFoundError:
        return None


def create_native_scheme(
    params: Optional[HESchemeParams] = None,
) -> NativeN2HEScheme:
    """
    Create a native N2HE scheme.

    Args:
        params: HE scheme parameters

    Returns:
        NativeN2HEScheme instance

    Raises:
        N2HELibraryNotFoundError: If native library not available
    """
    return NativeN2HEScheme(params)
