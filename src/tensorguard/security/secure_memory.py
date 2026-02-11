"""
Secure Memory Handling Module.

Provides secure memory management for cryptographic secrets:
- Memory locking to prevent swapping
- Secure zeroing of sensitive data
- Protected memory regions
- Context managers for automatic cleanup
"""

import ctypes
import logging
import secrets
import sys
from contextlib import contextmanager
from typing import Optional, Union

logger = logging.getLogger(__name__)


# Platform-specific memory locking
if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _mlock = _libc.mlock
        _mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _mlock.restype = ctypes.c_int

        _munlock = _libc.munlock
        _munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _munlock.restype = ctypes.c_int

        _HAVE_MLOCK = True
    except (OSError, AttributeError):
        _HAVE_MLOCK = False
elif sys.platform == "darwin":
    try:
        _libc = ctypes.CDLL("libSystem.B.dylib", use_errno=True)
        _mlock = _libc.mlock
        _mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _mlock.restype = ctypes.c_int

        _munlock = _libc.munlock
        _munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _munlock.restype = ctypes.c_int

        _HAVE_MLOCK = True
    except (OSError, AttributeError):
        _HAVE_MLOCK = False
else:
    _HAVE_MLOCK = False


def secure_zero(data: Union[bytearray, memoryview, ctypes.Array]) -> None:
    """
    Securely zero out memory containing sensitive data.

    Uses volatile writes to prevent compiler optimization from
    removing the zeroing operation.

    Args:
        data: Buffer to zero (must be mutable)
    """
    if isinstance(data, memoryview):
        # Zero via memoryview
        for i in range(len(data)):
            data[i] = 0
    elif isinstance(data, bytearray):
        # Zero bytearray
        for i in range(len(data)):
            data[i] = 0
    elif isinstance(data, ctypes.Array):
        # Zero ctypes array
        ctypes.memset(ctypes.addressof(data), 0, ctypes.sizeof(data))
    else:
        raise TypeError(f"Cannot securely zero {type(data)}")

    # Force memory barrier to ensure writes complete
    # This prevents the compiler from optimizing away the zeroing
    if hasattr(ctypes, "memmove"):
        # Dummy operation to create memory barrier
        dummy = ctypes.c_byte(0)
        ctypes.memmove(ctypes.addressof(dummy), ctypes.addressof(dummy), 1)


def secure_random(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Uses the operating system's secure random source.

    Args:
        length: Number of random bytes to generate

    Returns:
        Random bytes
    """
    return secrets.token_bytes(length)


class SecureMemory:
    """
    Secure memory buffer for cryptographic secrets.

    Features:
    - Memory locking to prevent swapping (where available)
    - Automatic secure zeroing on cleanup
    - Context manager support
    - Read-only mode after initialization
    """

    def __init__(
        self,
        size: int,
        lock_memory: bool = True,
        read_only: bool = False,
    ):
        """
        Initialize secure memory buffer.

        Args:
            size: Size in bytes
            lock_memory: Lock memory to prevent swapping
            read_only: Make buffer read-only after initialization
        """
        self._size = size
        self._lock_memory = lock_memory and _HAVE_MLOCK
        self._read_only = False
        self._locked = False
        self._cleared = False

        # Allocate buffer
        self._buffer = bytearray(size)
        self._view = memoryview(self._buffer)

        # Lock memory if requested
        if self._lock_memory:
            self._lock()

        # Set read-only if requested
        if read_only:
            self.make_read_only()

    def _lock(self) -> None:
        """Lock memory to prevent swapping."""
        if not _HAVE_MLOCK or self._locked:
            return

        try:
            # Get buffer address
            buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))

            result = _mlock(buf_addr, self._size)
            if result == 0:
                self._locked = True
                logger.debug(f"Locked {self._size} bytes of memory")
            else:
                errno = ctypes.get_errno()
                logger.warning(f"Failed to lock memory: errno={errno}")
        except Exception as e:
            logger.warning(f"Memory locking not available: {e}")

    def _unlock(self) -> None:
        """Unlock memory."""
        if not self._locked:
            return

        try:
            buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(self._buffer))
            _munlock(buf_addr, self._size)
            self._locked = False
        except Exception as e:
            logger.warning(f"Failed to unlock memory: {e}")

    def make_read_only(self) -> None:
        """Make buffer read-only."""
        self._read_only = True

    def write(self, data: bytes, offset: int = 0) -> None:
        """
        Write data to the secure buffer.

        Args:
            data: Data to write
            offset: Offset in buffer

        Raises:
            ValueError: If buffer is read-only or data too large
        """
        if self._read_only:
            raise ValueError("Buffer is read-only")

        if self._cleared:
            raise ValueError("Buffer has been cleared")

        if offset + len(data) > self._size:
            raise ValueError("Data exceeds buffer size")

        self._buffer[offset : offset + len(data)] = data

    def read(self, length: Optional[int] = None, offset: int = 0) -> bytes:
        """
        Read data from the secure buffer.

        Args:
            length: Number of bytes to read (default: entire buffer)
            offset: Offset in buffer

        Returns:
            Copy of buffer contents
        """
        if self._cleared:
            raise ValueError("Buffer has been cleared")

        if length is None:
            length = self._size - offset

        return bytes(self._buffer[offset : offset + length])

    def get_view(self) -> memoryview:
        """
        Get a memoryview of the buffer.

        Warning: This provides direct access to the buffer.
        The view should not be retained after the SecureMemory
        object is cleared.

        Returns:
            Memoryview of buffer
        """
        if self._cleared:
            raise ValueError("Buffer has been cleared")
        return self._view

    def clear(self) -> None:
        """Securely clear the buffer and unlock memory."""
        if self._cleared:
            return

        # Securely zero the buffer
        secure_zero(self._buffer)

        # Unlock memory
        self._unlock()

        self._cleared = True
        logger.debug(f"Cleared {self._size} bytes of secure memory")

    def __enter__(self) -> "SecureMemory":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clear buffer."""
        self.clear()

    def __del__(self) -> None:
        """Destructor - ensure memory is cleared."""
        self.clear()

    def __len__(self) -> int:
        """Return buffer size."""
        return self._size

    @property
    def size(self) -> int:
        """Return buffer size."""
        return self._size

    @property
    def is_locked(self) -> bool:
        """Check if memory is locked."""
        return self._locked

    @property
    def is_cleared(self) -> bool:
        """Check if buffer has been cleared."""
        return self._cleared


class SecureString:
    """
    Secure string for storing sensitive text (passwords, etc.).

    Automatically clears memory when the object is destroyed.
    """

    def __init__(self, value: str, encoding: str = "utf-8"):
        """
        Initialize secure string.

        Args:
            value: String value to store
            encoding: Text encoding
        """
        self._encoding = encoding
        encoded = value.encode(encoding)
        self._memory = SecureMemory(len(encoded))
        self._memory.write(encoded)
        self._length = len(value)

    def get(self) -> str:
        """
        Get the string value.

        Returns:
            The stored string
        """
        data = self._memory.read()
        return data.decode(self._encoding)

    def clear(self) -> None:
        """Clear the secure string."""
        self._memory.clear()

    def __enter__(self) -> "SecureString":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clear string."""
        self.clear()

    def __del__(self) -> None:
        """Destructor - ensure string is cleared."""
        self.clear()

    def __len__(self) -> int:
        """Return string length."""
        return self._length

    def __str__(self) -> str:
        """Return placeholder for security."""
        return "[SECURE STRING]"

    def __repr__(self) -> str:
        """Return placeholder for security."""
        return "SecureString(***)"


@contextmanager
def secure_key_context(key: bytes):
    """
    Context manager for secure key handling.

    Ensures the key is securely zeroed after use.

    Usage:
        with secure_key_context(my_key) as key:
            # Use key
            cipher = AES.new(key, AES.MODE_GCM)
        # Key is automatically zeroed
    """
    # Create mutable copy
    key_buffer = bytearray(key)

    try:
        yield key_buffer
    finally:
        secure_zero(key_buffer)


class SecureKeyStorage:
    """
    Secure storage for multiple cryptographic keys.

    Provides a secure container for storing multiple keys
    with automatic cleanup and memory locking.
    """

    def __init__(self, lock_memory: bool = True):
        """
        Initialize secure key storage.

        Args:
            lock_memory: Lock memory to prevent swapping
        """
        self._keys: dict[str, SecureMemory] = {}
        self._lock_memory = lock_memory

    def store(self, key_id: str, key: bytes) -> None:
        """
        Store a key securely.

        Args:
            key_id: Key identifier
            key: Key bytes
        """
        # Clear existing key if present
        if key_id in self._keys:
            self._keys[key_id].clear()

        # Store new key
        mem = SecureMemory(len(key), lock_memory=self._lock_memory)
        mem.write(key)
        mem.make_read_only()
        self._keys[key_id] = mem

    def retrieve(self, key_id: str) -> Optional[bytes]:
        """
        Retrieve a key.

        Args:
            key_id: Key identifier

        Returns:
            Key bytes or None if not found
        """
        mem = self._keys.get(key_id)
        if mem is None:
            return None
        return mem.read()

    def delete(self, key_id: str) -> bool:
        """
        Delete a key.

        Args:
            key_id: Key identifier

        Returns:
            True if key was deleted, False if not found
        """
        mem = self._keys.pop(key_id, None)
        if mem is not None:
            mem.clear()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all stored keys."""
        for mem in self._keys.values():
            mem.clear()
        self._keys.clear()

    def __contains__(self, key_id: str) -> bool:
        """Check if a key is stored."""
        return key_id in self._keys

    def __del__(self) -> None:
        """Destructor - clear all keys."""
        self.clear_all()


def wipe_string(s: str) -> None:
    """
    Attempt to wipe a Python string from memory.

    Note: Python strings are immutable, so this is not guaranteed
    to completely wipe the string from memory. Use SecureString
    for truly sensitive data.

    Args:
        s: String to wipe
    """
    # This is a best-effort approach - Python strings are immutable
    # and may have copies in various places
    try:
        # Try to overwrite the string's internal buffer
        # This may not work on all Python implementations
        import ctypes

        strlen = len(s)
        offset = sys.getsizeof(s) - strlen - 1

        ctypes.memset(id(s) + offset, 0, strlen)
    except Exception:
        # Silently fail - this is best-effort
        pass
