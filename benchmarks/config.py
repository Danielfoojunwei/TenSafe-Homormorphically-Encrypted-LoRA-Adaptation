"""
Benchmark Configuration Module

Centralizes all configuration for TensorGuardFlow benchmarks including:
- Target endpoints
- Load parameters
- Threshold definitions
- Environment settings
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class LoadLevel(Enum):
    """Predefined load levels for benchmarking."""
    LIGHT = "light"      # 100 events/s
    MODERATE = "moderate"  # 500 events/s
    HEAVY = "heavy"       # 1000 events/s
    STRESS = "stress"     # 5000 events/s


@dataclass
class EndpointConfig:
    """Configuration for a single endpoint to benchmark."""
    path: str
    method: str = "GET"
    requires_auth: bool = True
    payload_template: Optional[Dict] = None
    description: str = ""


@dataclass
class LoadConfig:
    """Load test configuration."""
    concurrent_users: int = 10
    requests_per_second: int = 100
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    batch_size: int = 100  # For telemetry batches


@dataclass
class ThresholdConfig:
    """Performance thresholds for pass/fail criteria."""
    # Latency thresholds (milliseconds)
    p50_latency_ms: float = 50.0
    p95_latency_ms: float = 200.0
    p99_latency_ms: float = 500.0

    # Throughput thresholds
    min_rps: float = 100.0
    min_events_per_second: float = 1000.0

    # Error thresholds
    max_error_rate: float = 0.01  # 1%

    # Resource thresholds
    max_cpu_percent: float = 80.0
    max_memory_mb: float = 2048.0


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration."""
    # Server configuration
    base_url: str = field(default_factory=lambda: os.getenv("TG_BENCH_URL", "http://localhost:8000"))

    # Authentication
    admin_email: str = field(default_factory=lambda: os.getenv("TG_BENCH_EMAIL", "bench@tensorguard.local"))
    admin_password: str = field(default_factory=lambda: os.getenv("TG_BENCH_PASSWORD", "benchpass123"))

    # Load configuration
    load: LoadConfig = field(default_factory=LoadConfig)

    # Thresholds
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)

    # Output paths
    output_dir: str = "artifacts/benchmarks"
    results_file: str = "benchmark_results.json"

    # Test selection
    run_api_bench: bool = True
    run_ingest_bench: bool = True
    run_resource_bench: bool = True
    run_worker_bench: bool = True

    # Hardware info (auto-detected or manual)
    cpu_info: str = ""
    memory_gb: float = 0.0
    os_info: str = ""


# Predefined endpoint configurations
API_ENDPOINTS: List[EndpointConfig] = [
    # Health endpoints (no auth required)
    EndpointConfig(
        path="/health",
        method="GET",
        requires_auth=False,
        description="Health check endpoint"
    ),
    EndpointConfig(
        path="/ready",
        method="GET",
        requires_auth=False,
        description="Readiness probe"
    ),
    EndpointConfig(
        path="/live",
        method="GET",
        requires_auth=False,
        description="Liveness probe"
    ),
    EndpointConfig(
        path="/metrics",
        method="GET",
        requires_auth=False,
        description="Prometheus metrics"
    ),

    # Authenticated endpoints
    EndpointConfig(
        path="/api/v1/users/me",
        method="GET",
        requires_auth=True,
        description="Current user profile"
    ),
    EndpointConfig(
        path="/api/v1/fleets",
        method="GET",
        requires_auth=True,
        description="List fleets"
    ),
    EndpointConfig(
        path="/api/v1/fleets/extended",
        method="GET",
        requires_auth=True,
        description="Fleets with metrics"
    ),
    EndpointConfig(
        path="/api/v1/jobs",
        method="GET",
        requires_auth=True,
        description="List jobs"
    ),
    EndpointConfig(
        path="/api/v1/telemetry/pipeline",
        method="GET",
        requires_auth=True,
        description="Pipeline telemetry"
    ),
    EndpointConfig(
        path="/api/v1/identity/endpoints",
        method="GET",
        requires_auth=True,
        description="Identity endpoints"
    ),
]


# Load level presets
LOAD_PRESETS: Dict[LoadLevel, LoadConfig] = {
    LoadLevel.LIGHT: LoadConfig(
        concurrent_users=5,
        requests_per_second=100,
        duration_seconds=30,
        ramp_up_seconds=5,
        batch_size=50
    ),
    LoadLevel.MODERATE: LoadConfig(
        concurrent_users=20,
        requests_per_second=500,
        duration_seconds=60,
        ramp_up_seconds=10,
        batch_size=100
    ),
    LoadLevel.HEAVY: LoadConfig(
        concurrent_users=50,
        requests_per_second=1000,
        duration_seconds=120,
        ramp_up_seconds=20,
        batch_size=200
    ),
    LoadLevel.STRESS: LoadConfig(
        concurrent_users=100,
        requests_per_second=5000,
        duration_seconds=180,
        ramp_up_seconds=30,
        batch_size=500
    ),
}


def get_config(load_level: LoadLevel = LoadLevel.MODERATE) -> BenchmarkConfig:
    """Get benchmark configuration with specified load level."""
    config = BenchmarkConfig()
    config.load = LOAD_PRESETS[load_level]
    return config


def get_hardware_info() -> Dict[str, str]:
    """Detect hardware information for benchmark reporting."""
    import platform
    import psutil

    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)

    return {
        "cpu_info": f"{platform.processor()} ({cpu_count} cores)",
        "memory_gb": round(memory_gb, 1),
        "os_info": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }
