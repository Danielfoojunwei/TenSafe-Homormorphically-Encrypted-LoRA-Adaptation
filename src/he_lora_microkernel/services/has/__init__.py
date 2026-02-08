"""
HE Adapter Service (HAS)

The HAS is the secure service that:
- Holds HE secret keys (never leaves this process)
- Performs encryption/decryption
- Executes HE-LoRA computation
- Communicates with MSS via gRPC + shared memory

Components:
- server.py: gRPC server implementation
- executor.py: HE-LoRA execution engine
- key_manager.py: HE key management
- shm_manager.py: Shared memory management
"""

from .executor import HASExecutor
from .key_manager import KeyManager
from .server import HASServer, create_has_server
from .shm_manager import SharedMemoryManager

__all__ = [
    'HASServer',
    'create_has_server',
    'HASExecutor',
    'KeyManager',
    'SharedMemoryManager',
]
