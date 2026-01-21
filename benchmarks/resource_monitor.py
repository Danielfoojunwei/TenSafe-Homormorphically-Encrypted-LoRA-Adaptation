"""
Resource Monitoring Module

Tracks CPU, memory, disk I/O, and network usage during benchmark execution.
Provides correlation between resource consumption and throughput.
"""

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, resource monitoring disabled")


@dataclass
class ResourceSample:
    """Single point-in-time resource measurement."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_read_bytes: int
    disk_write_bytes: int
    net_bytes_sent: int
    net_bytes_recv: int
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0


@dataclass
class ResourceMetrics:
    """Aggregated resource metrics."""
    # CPU
    cpu_percent_mean: float
    cpu_percent_max: float
    cpu_percent_p95: float

    # Memory
    memory_percent_mean: float
    memory_percent_max: float
    memory_used_mb_mean: float
    memory_used_mb_max: float

    # Process-specific
    process_cpu_percent_mean: float
    process_cpu_percent_max: float
    process_memory_mb_mean: float
    process_memory_mb_max: float

    # I/O
    disk_read_mb: float
    disk_write_mb: float
    net_sent_mb: float
    net_recv_mb: float

    # Duration
    duration_seconds: float
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu": {
                "mean_percent": round(self.cpu_percent_mean, 2),
                "max_percent": round(self.cpu_percent_max, 2),
                "p95_percent": round(self.cpu_percent_p95, 2),
            },
            "memory": {
                "mean_percent": round(self.memory_percent_mean, 2),
                "max_percent": round(self.memory_percent_max, 2),
                "mean_used_mb": round(self.memory_used_mb_mean, 1),
                "max_used_mb": round(self.memory_used_mb_max, 1),
            },
            "process": {
                "cpu_mean_percent": round(self.process_cpu_percent_mean, 2),
                "cpu_max_percent": round(self.process_cpu_percent_max, 2),
                "memory_mean_mb": round(self.process_memory_mb_mean, 1),
                "memory_max_mb": round(self.process_memory_mb_max, 1),
            },
            "io": {
                "disk_read_mb": round(self.disk_read_mb, 2),
                "disk_write_mb": round(self.disk_write_mb, 2),
                "net_sent_mb": round(self.net_sent_mb, 2),
                "net_recv_mb": round(self.net_recv_mb, 2),
            },
            "duration_seconds": round(self.duration_seconds, 2),
            "sample_count": self.sample_count,
        }


class ResourceMonitor:
    """
    Background resource monitor that samples system metrics periodically.
    """

    def __init__(self, sample_interval: float = 1.0, target_pid: Optional[int] = None):
        """
        Initialize resource monitor.

        Args:
            sample_interval: Seconds between samples
            target_pid: Process ID to monitor (default: current process)
        """
        self.sample_interval = sample_interval
        self.target_pid = target_pid or os.getpid()
        self.samples: List[ResourceSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        if PSUTIL_AVAILABLE:
            self._process = psutil.Process(self.target_pid)
            self._initial_disk_io = psutil.disk_io_counters()
            self._initial_net_io = psutil.net_io_counters()
        else:
            self._process = None
            self._initial_disk_io = None
            self._initial_net_io = None

    def _sample(self) -> Optional[ResourceSample]:
        """Collect a single resource sample."""
        if not PSUTIL_AVAILABLE:
            return None

        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            # Process-specific metrics
            process_cpu = self._process.cpu_percent(interval=None)
            process_memory = self._process.memory_info()

            return ResourceSample(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_read_bytes=disk_io.read_bytes if disk_io else 0,
                disk_write_bytes=disk_io.write_bytes if disk_io else 0,
                net_bytes_sent=net_io.bytes_sent if net_io else 0,
                net_bytes_recv=net_io.bytes_recv if net_io else 0,
                process_cpu_percent=process_cpu,
                process_memory_mb=process_memory.rss / (1024 * 1024),
            )

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            print(f"Resource sampling error: {e}")
            return None

    def _monitor_loop(self):
        """Background monitoring loop."""
        # Initial CPU reading to prime the counter
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
            self._process.cpu_percent(interval=None)

        while self._running:
            sample = self._sample()
            if sample:
                with self._lock:
                    self.samples.append(sample)

            time.sleep(self.sample_interval)

    def start(self):
        """Start background monitoring."""
        if self._running:
            return

        self._running = True
        self.samples = []

        if PSUTIL_AVAILABLE:
            self._initial_disk_io = psutil.disk_io_counters()
            self._initial_net_io = psutil.net_io_counters()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("Resource monitoring started")

    def stop(self) -> ResourceMetrics:
        """Stop monitoring and return aggregated metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

        print("Resource monitoring stopped")
        return self._calculate_metrics()

    def _calculate_metrics(self) -> ResourceMetrics:
        """Calculate aggregated metrics from samples."""
        with self._lock:
            samples = list(self.samples)

        if not samples or not PSUTIL_AVAILABLE:
            return ResourceMetrics(
                cpu_percent_mean=0,
                cpu_percent_max=0,
                cpu_percent_p95=0,
                memory_percent_mean=0,
                memory_percent_max=0,
                memory_used_mb_mean=0,
                memory_used_mb_max=0,
                process_cpu_percent_mean=0,
                process_cpu_percent_max=0,
                process_memory_mb_mean=0,
                process_memory_mb_max=0,
                disk_read_mb=0,
                disk_write_mb=0,
                net_sent_mb=0,
                net_recv_mb=0,
                duration_seconds=0,
                sample_count=0,
            )

        cpu_percents = [s.cpu_percent for s in samples]
        memory_percents = [s.memory_percent for s in samples]
        memory_used = [s.memory_used_mb for s in samples]
        process_cpu = [s.process_cpu_percent for s in samples]
        process_memory = [s.process_memory_mb for s in samples]

        # Calculate I/O totals
        final_disk_io = psutil.disk_io_counters()
        final_net_io = psutil.net_io_counters()

        disk_read_mb = 0
        disk_write_mb = 0
        net_sent_mb = 0
        net_recv_mb = 0

        if self._initial_disk_io and final_disk_io:
            disk_read_mb = (final_disk_io.read_bytes - self._initial_disk_io.read_bytes) / (1024 * 1024)
            disk_write_mb = (final_disk_io.write_bytes - self._initial_disk_io.write_bytes) / (1024 * 1024)

        if self._initial_net_io and final_net_io:
            net_sent_mb = (final_net_io.bytes_sent - self._initial_net_io.bytes_sent) / (1024 * 1024)
            net_recv_mb = (final_net_io.bytes_recv - self._initial_net_io.bytes_recv) / (1024 * 1024)

        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0

        def percentile(values: List[float], p: float) -> float:
            if not values:
                return 0
            sorted_values = sorted(values)
            idx = int(len(sorted_values) * p / 100)
            return sorted_values[min(idx, len(sorted_values) - 1)]

        return ResourceMetrics(
            cpu_percent_mean=statistics.mean(cpu_percents),
            cpu_percent_max=max(cpu_percents),
            cpu_percent_p95=percentile(cpu_percents, 95),
            memory_percent_mean=statistics.mean(memory_percents),
            memory_percent_max=max(memory_percents),
            memory_used_mb_mean=statistics.mean(memory_used),
            memory_used_mb_max=max(memory_used),
            process_cpu_percent_mean=statistics.mean(process_cpu),
            process_cpu_percent_max=max(process_cpu),
            process_memory_mb_mean=statistics.mean(process_memory),
            process_memory_mb_max=max(process_memory),
            disk_read_mb=disk_read_mb,
            disk_write_mb=disk_write_mb,
            net_sent_mb=net_sent_mb,
            net_recv_mb=net_recv_mb,
            duration_seconds=duration,
            sample_count=len(samples),
        )

    def get_current_sample(self) -> Optional[ResourceSample]:
        """Get the most recent sample."""
        with self._lock:
            return self.samples[-1] if self.samples else None

    def get_samples(self) -> List[ResourceSample]:
        """Get all collected samples."""
        with self._lock:
            return list(self.samples)


class BenchmarkWithResources:
    """
    Context manager for running benchmarks with resource monitoring.
    """

    def __init__(self, sample_interval: float = 0.5):
        self.monitor = ResourceMonitor(sample_interval=sample_interval)
        self.metrics: Optional[ResourceMetrics] = None

    def __enter__(self):
        self.monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.metrics = self.monitor.stop()
        return False

    def run(self, benchmark_fn: Callable) -> tuple:
        """
        Run a benchmark function while monitoring resources.

        Returns:
            Tuple of (benchmark_result, resource_metrics)
        """
        self.monitor.start()
        try:
            result = benchmark_fn()
            return result, self.monitor.stop()
        except Exception as e:
            self.monitor.stop()
            raise


def run_resource_benchmark(duration_seconds: float = 30.0) -> Dict[str, Any]:
    """
    Run a standalone resource monitoring benchmark.

    Monitors system resources for the specified duration without load.
    Useful for establishing baseline resource consumption.
    """
    print(f"Running resource baseline measurement for {duration_seconds}s...")

    monitor = ResourceMonitor(sample_interval=0.5)
    monitor.start()

    time.sleep(duration_seconds)

    metrics = monitor.stop()

    print(f"\nResource Baseline Results:")
    print(f"  CPU: mean={metrics.cpu_percent_mean:.1f}%, max={metrics.cpu_percent_max:.1f}%")
    print(f"  Memory: mean={metrics.memory_used_mb_mean:.0f}MB, max={metrics.memory_used_mb_max:.0f}MB")
    print(f"  Samples collected: {metrics.sample_count}")

    return {
        "type": "resource_baseline",
        "metrics": metrics.to_dict(),
    }
