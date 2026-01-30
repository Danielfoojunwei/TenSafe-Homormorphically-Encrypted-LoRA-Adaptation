import hashlib
import secrets
import struct
from typing import IO, Optional

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

CHUNK_SIZE = 4 * 1024 * 1024 # 4MB

def derive_nonce(base_nonce: bytes, chunk_index: int) -> bytes:
    """Derive deterministic nonce for chunk."""
    idx_bytes = struct.pack(">Q", chunk_index)
    hash_input = base_nonce + idx_bytes
    return hashlib.sha256(hash_input).digest()[:12]

class PayloadEncryptor:
    def __init__(self, key: bytes, manifest_hash: str, recipients_hash: str):
        self.aead = ChaCha20Poly1305(key)
        self.nonce_base = secrets.token_bytes(12)
        self.chunk_index = 0
        self.manifest_hash = manifest_hash
        self.recipients_hash = recipients_hash

    def encrypt_chunk(self, plaintext: bytes, is_last: bool = False) -> bytes:
        # AAD includes hashes and index for strong binding
        aad = (self.manifest_hash + self.recipients_hash).encode() + struct.pack(">Q", self.chunk_index)
        nonce = derive_nonce(self.nonce_base, self.chunk_index)

        ciphertext = self.aead.encrypt(nonce, plaintext, aad)

        # Format: [u32 plain_len][ciphertext(includes tag)]
        res = struct.pack(">I", len(plaintext)) + ciphertext
        self.chunk_index += 1
        return res

def encrypt_stream(input_stream: IO[bytes], output_stream: IO[bytes], key: bytes, manifest_hash: str, recipients_hash: str) -> str:
    """
    Encrypt input_stream to output_stream.
    Returns: nonce_base (hex)
    """
    encryptor = PayloadEncryptor(key, manifest_hash, recipients_hash)

    while True:
        chunk = input_stream.read(CHUNK_SIZE)
        if not chunk:
            break
        encrypted_chunk = encryptor.encrypt_chunk(chunk)
        output_stream.write(encrypted_chunk)

    return encryptor.nonce_base.hex()

class PayloadDecryptor:
    def __init__(self, key: bytes, nonce_base: bytes, manifest_hash: str, recipients_hash: str):
        self.aead = ChaCha20Poly1305(key)
        self.nonce_base = nonce_base
        self.chunk_index = 0
        self.manifest_hash = manifest_hash
        self.recipients_hash = recipients_hash

    def decrypt_chunk_from_stream(self, stream: IO[bytes]) -> Optional[bytes]:
        # Read u32 len
        len_bytes = stream.read(4)
        if not len_bytes:
            return None
        plain_len = struct.unpack(">I", len_bytes)[0]

        # Read ciphertext: plain_len + 16 (tag)
        ct_len = plain_len + 16
        ciphertext = stream.read(ct_len)
        if len(ciphertext) != ct_len:
            raise ValueError(f"Truncated ciphertext at chunk {self.chunk_index}")

        aad = (self.manifest_hash + self.recipients_hash).encode() + struct.pack(">Q", self.chunk_index)
        nonce = derive_nonce(self.nonce_base, self.chunk_index)

        plaintext = self.aead.decrypt(nonce, ciphertext, aad)
        self.chunk_index += 1
        return plaintext
