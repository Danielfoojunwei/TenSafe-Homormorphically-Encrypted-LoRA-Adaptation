"""
Shared Memory Manager

Manages shared memory regions for low-latency data transfer between
MSS and HAS. Uses platform-specific APIs for optimal performance:

- Linux: memfd_create + mmap
- CUDA: cudaIpcMemHandle for GPU memory sharing
- Fallback: NumPy memory-mapped files

Data plane protocol:
1. MSS writes hidden states to shared memory
2. Signals HAS via gRPC (or eventfd)
3. HAS reads, encrypts, computes, decrypts
4. HAS writes delta to shared memory
5. Signals MSS
6. MSS reads delta and applies
"""

import logging
import mmap
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ShmType(Enum):
    """Shared memory implementation type."""
    MEMFD = "memfd"        # Linux memfd_create
    CUDA_IPC = "cuda_ipc"  # CUDA IPC handles
    POSIX = "posix"        # POSIX shared memory
    MMAP = "mmap"          # Memory-mapped file fallback


@dataclass
class BufferLayout:
    """Layout of a shared memory buffer."""
    # Header (64 bytes)
    header_offset: int = 0
    header_size: int = 64

    # Hidden states region
    hidden_states_offset: int = 64
    hidden_states_size: int = 0

    # Delta output region
    delta_offset: int = 0
    delta_size: int = 0

    # Total size
    total_size: int = 0


@dataclass
class ShmRegion:
    """A shared memory region."""
    name: str
    size: int
    shm_type: ShmType
    layout: BufferLayout

    # Implementation-specific handles
    fd: Optional[int] = None
    mmap_obj: Optional[mmap.mmap] = None
    cuda_handle: Optional[Any] = None
    buffer: Optional[Any] = None

    # State
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)


class SharedMemoryManager:
    """
    Manages shared memory regions for HAS data plane.

    Features:
    - Automatic buffer sizing based on batch size and hidden size
    - Double-buffering for overlap of compute and transfer
    - GPU-pinned memory for CUDA IPC
    - Reference counting for cleanup
    """

    def __init__(
        self,
        shm_prefix: str = "/helora",
        default_size_mb: int = 64,
        use_cuda: bool = True,
    ):
        """
        Initialize shared memory manager.

        Args:
            shm_prefix: Prefix for shared memory region names
            default_size_mb: Default region size in MB
            use_cuda: Whether to use CUDA IPC when available
        """
        self._prefix = shm_prefix
        self._default_size = default_size_mb * 1024 * 1024
        self._use_cuda = use_cuda

        # Shared memory regions
        self._regions: Dict[str, ShmRegion] = {}

        # Determine available SHM type
        self._shm_type = self._detect_shm_type()

        logger.info(f"SharedMemoryManager initialized with type: {self._shm_type.value}")

    def _detect_shm_type(self) -> ShmType:
        """Detect the best available shared memory type."""
        # Try CUDA IPC first
        if self._use_cuda:
            try:
                import torch
                if torch.cuda.is_available():
                    # Check if CUDA IPC is supported
                    return ShmType.CUDA_IPC
            except ImportError:
                pass

        # Try memfd on Linux
        if hasattr(os, 'memfd_create'):
            return ShmType.MEMFD

        # Try POSIX shared memory
        try:
            import posix_ipc
            return ShmType.POSIX
        except ImportError:
            pass

        # Fallback to mmap
        return ShmType.MMAP

    def create_region(
        self,
        name: str,
        batch_size: int,
        hidden_size: int,
        dtype_bytes: int = 2,  # FP16
    ) -> ShmRegion:
        """
        Create a shared memory region for a request.

        Args:
            name: Region name (will be prefixed)
            batch_size: Batch size for the region
            hidden_size: Hidden dimension size
            dtype_bytes: Bytes per element (2 for FP16)

        Returns:
            ShmRegion instance
        """
        full_name = f"{self._prefix}_{name}"

        # Calculate layout
        layout = self._calculate_layout(batch_size, hidden_size, dtype_bytes)

        # Check for existing region
        if full_name in self._regions:
            existing = self._regions[full_name]
            if existing.size >= layout.total_size:
                existing.last_accessed = time.time()
                return existing
            else:
                # Need larger region, destroy old one
                self.destroy_region(full_name)

        # Create region based on type
        if self._shm_type == ShmType.MEMFD:
            region = self._create_memfd_region(full_name, layout)
        elif self._shm_type == ShmType.CUDA_IPC:
            region = self._create_cuda_region(full_name, layout)
        elif self._shm_type == ShmType.POSIX:
            region = self._create_posix_region(full_name, layout)
        else:
            region = self._create_mmap_region(full_name, layout)

        self._regions[full_name] = region
        logger.debug(f"Created SHM region {full_name}: {layout.total_size} bytes")
        return region

    def _calculate_layout(
        self,
        batch_size: int,
        hidden_size: int,
        dtype_bytes: int,
    ) -> BufferLayout:
        """Calculate buffer layout for given parameters."""
        # Size for one batch of hidden states
        tensor_size = batch_size * hidden_size * dtype_bytes

        # Add padding for alignment (256-byte aligned)
        def align(size: int, alignment: int = 256) -> int:
            return (size + alignment - 1) & ~(alignment - 1)

        layout = BufferLayout()
        layout.header_offset = 0
        layout.header_size = 64

        layout.hidden_states_offset = align(layout.header_size)
        layout.hidden_states_size = align(tensor_size)

        layout.delta_offset = layout.hidden_states_offset + layout.hidden_states_size
        layout.delta_size = align(tensor_size)

        layout.total_size = layout.delta_offset + layout.delta_size

        return layout

    def _create_memfd_region(self, name: str, layout: BufferLayout) -> ShmRegion:
        """Create region using Linux memfd_create."""
        fd = os.memfd_create(name, 0)
        os.ftruncate(fd, layout.total_size)

        mmap_obj = mmap.mmap(fd, layout.total_size)

        return ShmRegion(
            name=name,
            size=layout.total_size,
            shm_type=ShmType.MEMFD,
            layout=layout,
            fd=fd,
            mmap_obj=mmap_obj,
        )

    def _create_cuda_region(self, name: str, layout: BufferLayout) -> ShmRegion:
        """Create region using CUDA IPC."""
        import torch

        # Allocate pinned host memory for CPU side
        buffer = torch.empty(
            layout.total_size,
            dtype=torch.uint8,
            device='cpu',
        ).pin_memory()

        # For full CUDA IPC, we'd also allocate GPU memory and get IPC handle
        # This is simplified for the initial implementation

        return ShmRegion(
            name=name,
            size=layout.total_size,
            shm_type=ShmType.CUDA_IPC,
            layout=layout,
            buffer=buffer,
        )

    def _create_posix_region(self, name: str, layout: BufferLayout) -> ShmRegion:
        """Create region using POSIX shared memory."""
        import posix_ipc

        shm = posix_ipc.SharedMemory(
            name,
            posix_ipc.O_CREAT,
            size=layout.total_size,
        )

        mmap_obj = mmap.mmap(shm.fd, layout.total_size)

        return ShmRegion(
            name=name,
            size=layout.total_size,
            shm_type=ShmType.POSIX,
            layout=layout,
            fd=shm.fd,
            mmap_obj=mmap_obj,
        )

    def _create_mmap_region(self, name: str, layout: BufferLayout) -> ShmRegion:
        """Create region using memory-mapped file."""
        import tempfile


        # Create temporary file
        tmp_path = os.path.join(tempfile.gettempdir(), f"{name}.shm")
        fd = os.open(tmp_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(fd, layout.total_size)

        mmap_obj = mmap.mmap(fd, layout.total_size)

        return ShmRegion(
            name=name,
            size=layout.total_size,
            shm_type=ShmType.MMAP,
            layout=layout,
            fd=fd,
            mmap_obj=mmap_obj,
        )

    def get_region(self, name: str) -> Optional[ShmRegion]:
        """Get a shared memory region by name."""
        full_name = f"{self._prefix}_{name}" if not name.startswith(self._prefix) else name
        return self._regions.get(full_name)

    def destroy_region(self, name: str) -> bool:
        """Destroy a shared memory region."""
        full_name = f"{self._prefix}_{name}" if not name.startswith(self._prefix) else name

        if full_name not in self._regions:
            return False

        region = self._regions[full_name]

        # Clean up resources
        if region.mmap_obj is not None:
            region.mmap_obj.close()

        if region.fd is not None:
            try:
                os.close(region.fd)
            except OSError:
                pass

        if region.buffer is not None:
            del region.buffer

        del self._regions[full_name]
        logger.debug(f"Destroyed SHM region {full_name}")
        return True

    def write_hidden_states(
        self,
        region: ShmRegion,
        data: Any,
    ) -> int:
        """
        Write hidden states to shared memory.

        Args:
            region: Target region
            data: Hidden states tensor/array

        Returns:
            Offset where data was written
        """
        region.last_accessed = time.time()
        offset = region.layout.hidden_states_offset

        if region.mmap_obj is not None:
            # Get bytes from data
            if hasattr(data, 'numpy'):
                data_bytes = data.numpy().tobytes()
            elif hasattr(data, 'tobytes'):
                data_bytes = data.tobytes()
            else:
                data_bytes = bytes(data)

            region.mmap_obj.seek(offset)
            region.mmap_obj.write(data_bytes)

        elif region.buffer is not None:
            # CUDA pinned buffer
            import torch
            if isinstance(data, torch.Tensor):
                region.buffer[offset:offset + data.numel() * data.element_size()].copy_(
                    data.view(-1).byte()
                )
            else:
                # NumPy or other
                import numpy as np
                data_np = np.asarray(data)
                region.buffer[offset:offset + data_np.nbytes].copy_(
                    torch.from_numpy(data_np.view(np.uint8))
                )

        return offset

    def read_hidden_states(
        self,
        region: ShmRegion,
        shape: Tuple[int, ...],
        dtype: Any = None,
    ) -> Any:
        """
        Read hidden states from shared memory.

        Args:
            region: Source region
            shape: Expected tensor shape
            dtype: Data type

        Returns:
            Tensor/array of hidden states
        """
        import numpy as np

        region.last_accessed = time.time()
        offset = region.layout.hidden_states_offset

        if dtype is None:
            dtype = np.float16

        size = int(np.prod(shape)) * np.dtype(dtype).itemsize

        if region.mmap_obj is not None:
            region.mmap_obj.seek(offset)
            data_bytes = region.mmap_obj.read(size)
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

        elif region.buffer is not None:
            import torch
            buffer_slice = region.buffer[offset:offset + size]
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            return buffer_slice.view(torch_dtype).reshape(shape)

        return None

    def write_delta(
        self,
        region: ShmRegion,
        data: Any,
    ) -> int:
        """
        Write delta output to shared memory.

        Args:
            region: Target region
            data: Delta tensor/array

        Returns:
            Offset where data was written
        """
        region.last_accessed = time.time()
        offset = region.layout.delta_offset

        if region.mmap_obj is not None:
            if hasattr(data, 'numpy'):
                data_bytes = data.numpy().tobytes()
            elif hasattr(data, 'tobytes'):
                data_bytes = data.tobytes()
            else:
                data_bytes = bytes(data)

            region.mmap_obj.seek(offset)
            region.mmap_obj.write(data_bytes)

        elif region.buffer is not None:
            import torch
            if isinstance(data, torch.Tensor):
                region.buffer[offset:offset + data.numel() * data.element_size()].copy_(
                    data.view(-1).byte()
                )

        return offset

    def read_delta(
        self,
        region: ShmRegion,
        shape: Tuple[int, ...],
        dtype: Any = None,
    ) -> Any:
        """
        Read delta from shared memory.

        Args:
            region: Source region
            shape: Expected tensor shape
            dtype: Data type

        Returns:
            Tensor/array of delta
        """
        import numpy as np

        region.last_accessed = time.time()
        offset = region.layout.delta_offset

        if dtype is None:
            dtype = np.float16

        size = int(np.prod(shape)) * np.dtype(dtype).itemsize

        if region.mmap_obj is not None:
            region.mmap_obj.seek(offset)
            data_bytes = region.mmap_obj.read(size)
            return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)

        elif region.buffer is not None:
            import torch
            buffer_slice = region.buffer[offset:offset + size]
            torch_dtype = torch.float16 if dtype == np.float16 else torch.float32
            return buffer_slice.view(torch_dtype).reshape(shape)

        return None

    def cleanup_stale_regions(self, max_age_seconds: float = 300) -> int:
        """
        Clean up regions not accessed recently.

        Args:
            max_age_seconds: Maximum age before cleanup

        Returns:
            Number of regions cleaned up
        """
        now = time.time()
        stale = [
            name for name, region in self._regions.items()
            if (now - region.last_accessed) > max_age_seconds
        ]

        for name in stale:
            self.destroy_region(name)

        return len(stale)

    def shutdown(self) -> None:
        """Clean up all regions."""
        for name in list(self._regions.keys()):
            self.destroy_region(name)
        logger.info("SharedMemoryManager shutdown complete")

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        total_size = sum(r.size for r in self._regions.values())
        return {
            'shm_type': self._shm_type.value,
            'region_count': len(self._regions),
            'total_size_mb': total_size / (1024 * 1024),
            'regions': [
                {
                    'name': r.name,
                    'size': r.size,
                    'age_seconds': time.time() - r.created_at,
                }
                for r in self._regions.values()
            ],
        }
