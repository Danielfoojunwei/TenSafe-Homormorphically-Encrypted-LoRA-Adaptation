
import abc
from typing import Tuple


class PostQuantumKEM(abc.ABC):
    """Abstract Base Class for Post-Quantum Key Encapsulation."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def public_key_size(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def ciphertext_size(self) -> int:
        pass

    @abc.abstractmethod
    def keygen(self) -> Tuple[bytes, bytes]:
        """Returns (pk, sk)"""
        pass

    @abc.abstractmethod
    def encap(self, pk: bytes) -> Tuple[bytes, bytes]:
        """Returns (shared_secret, ciphertext)"""
        pass

    @abc.abstractmethod
    def decap(self, sk: bytes, ct: bytes) -> bytes:
        """Returns shared_secret"""
        pass

class PostQuantumSig(abc.ABC):
    """Abstract Base Class for Post-Quantum Signatures."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @abc.abstractmethod
    def keygen(self) -> Tuple[bytes, bytes]:
        """Returns (pk, sk)"""
        pass

    @abc.abstractmethod
    def sign(self, sk: bytes, message: bytes) -> bytes:
        """Returns signature"""
        pass

    @abc.abstractmethod
    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        """Returns True if valid"""
        pass
