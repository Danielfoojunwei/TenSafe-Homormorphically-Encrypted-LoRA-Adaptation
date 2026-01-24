import logging
import time
import psutil
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_peak_cpu_mem():
    """Returns current process peak resident set size in MB."""
    process = psutil.Process()
    # rss is in bytes
    return process.memory_info().rss / (1024 * 1024)

def get_gpu_memory_mb():
    """Returns peak GPU memory allocated in MB if torch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0

from contextlib import contextmanager

@contextmanager
def step_timer():
    """Helper for orchestrator instrumentation."""
    t = StepTimer("temp")
    t.__enter__()
    try:
        yield t
    finally:
        t.__exit__(None, None, None)

class StepTimer:
    """Context manager for timing pipeline steps."""
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = 0
        self.duration_sec = 0
        self.peak_cpu_mb = 0
        self.peak_gpu_mb = 0
        
    def __getitem__(self, key):
        """Allow subscript access for backward compatibility with dict-style usage."""
        data = self.to_dict()
        return data.get(key)
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_sec = time.perf_counter() - self.start_time
        self.peak_cpu_mb = get_peak_cpu_mem()
        self.peak_gpu_mb = get_peak_gpu_mem()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "duration_sec": self.duration_sec,
            "duration_ms": self.duration_sec * 1000,
            "peak_cpu_mem_mb": self.peak_cpu_mb,
            "peak_gpu_mem_mb": self.peak_gpu_mb
        }

def compute_resource_metrics(step_timers: List["StepTimer"]):
    """Aggregates step timers into end-to-end metrics."""
    total_time = sum(t.duration_sec for t in step_timers)
    max_cpu = max((t.peak_cpu_mb for t in step_timers), default=0.0)
    max_gpu = max((t.peak_gpu_mb for t in step_timers), default=0.0)
    
    return {
        "end_to_end_time_sec": total_time,
        "peak_cpu_mem_mb": max_cpu,
        "peak_gpu_mem_mb": max_gpu
    }
