#!/usr/bin/env python3
"""
Generate Sample Benchmark Results

Creates realistic sample benchmark results for testing the analysis
and reporting pipeline without requiring a running server.
"""

import json
import random
import time
from datetime import datetime
from pathlib import Path


def generate_sample_results(output_dir: str = "artifacts/benchmarks") -> str:
    """Generate sample benchmark results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat(),
            "tensorguard_version": "3.0.0",
            "benchmark_version": "1.0.0",
            "config": {
                "base_url": "http://localhost:8000",
                "load_level": {
                    "concurrent_users": 10,
                    "requests_per_second": 500,
                    "duration_seconds": 60,
                    "batch_size": 100,
                },
                "thresholds": {
                    "p95_latency_ms": 200.0,
                    "min_rps": 100.0,
                    "max_error_rate": 0.01,
                },
            },
            "environment": {
                "cpu": "Intel Core i7 (8 cores)",
                "memory_gb": 16.0,
                "os": "Linux 5.15",
                "python": "3.11.0",
            },
            "total_duration_seconds": 180.5,
        },
        "benchmarks": {},
        "summary": {},
    }

    # Generate HTTP API results
    endpoints = [
        ("/health", 50, 12000),
        ("/ready", 45, 11000),
        ("/live", 40, 12500),
        ("/metrics", 80, 8000),
        ("/api/v1/users/me", 120, 3500),
        ("/api/v1/fleets", 150, 2800),
        ("/api/v1/fleets/extended", 250, 1800),
        ("/api/v1/jobs", 180, 2200),
        ("/api/v1/telemetry/pipeline", 300, 1500),
        ("/api/v1/identity/endpoints", 200, 2000),
    ]

    http_endpoints = {}
    for path, base_latency, base_rps in endpoints:
        # Add some variance
        p50 = base_latency * random.uniform(0.8, 1.0)
        p90 = p50 * random.uniform(1.5, 2.0)
        p95 = p90 * random.uniform(1.1, 1.3)
        p99 = p95 * random.uniform(1.2, 1.5)

        rps = base_rps * random.uniform(0.9, 1.1)
        error_rate = random.uniform(0, 0.005)

        http_endpoints[path] = {
            "endpoint": path,
            "method": "GET",
            "total_requests": int(rps * 60),
            "successful_requests": int(rps * 60 * (1 - error_rate)),
            "failed_requests": int(rps * 60 * error_rate),
            "error_rate": error_rate,
            "latency": {
                "min_ms": p50 * 0.5,
                "max_ms": p99 * 1.5,
                "mean_ms": p50 * 1.1,
                "p50_ms": p50,
                "p90_ms": p90,
                "p95_ms": p95,
                "p99_ms": p99,
                "std_ms": p50 * 0.3,
            },
            "throughput": {
                "requests_per_second": rps,
                "duration_seconds": 60.0,
            },
            "status_codes": {
                200: int(rps * 60 * (1 - error_rate)),
                500: int(rps * 60 * error_rate * 0.5),
                503: int(rps * 60 * error_rate * 0.5),
            },
        }

    avg_p95 = sum(e["latency"]["p95_ms"] for e in http_endpoints.values()) / len(http_endpoints)
    avg_rps = sum(e["throughput"]["requests_per_second"] for e in http_endpoints.values()) / len(http_endpoints)
    avg_error = sum(e["error_rate"] for e in http_endpoints.values()) / len(http_endpoints)

    results["benchmarks"]["http_api"] = {
        "type": "http_api",
        "endpoints": http_endpoints,
        "summary": {
            "total_endpoints": len(http_endpoints),
            "avg_p95_latency_ms": avg_p95,
            "avg_rps": avg_rps,
            "avg_error_rate": avg_error,
        },
        "resources": {
            "cpu": {
                "mean_percent": 45.5,
                "max_percent": 72.3,
                "p95_percent": 68.0,
            },
            "memory": {
                "mean_percent": 35.2,
                "max_percent": 42.1,
                "mean_used_mb": 580.5,
                "max_used_mb": 720.3,
            },
            "process": {
                "cpu_mean_percent": 25.3,
                "cpu_max_percent": 45.6,
                "memory_mean_mb": 280.5,
                "memory_max_mb": 350.2,
            },
            "io": {
                "disk_read_mb": 120.5,
                "disk_write_mb": 85.3,
                "net_sent_mb": 450.2,
                "net_recv_mb": 380.5,
            },
            "duration_seconds": 60.0,
            "sample_count": 120,
        },
    }

    # Generate Telemetry results
    events_per_second = random.uniform(8000, 12000)
    batches_per_second = events_per_second / 100

    results["benchmarks"]["telemetry_ingest"] = {
        "type": "telemetry_ingest",
        "metrics": {
            "total_batches": int(batches_per_second * 60),
            "total_events": int(events_per_second * 60),
            "successful_batches": int(batches_per_second * 60 * 0.995),
            "failed_batches": int(batches_per_second * 60 * 0.005),
            "events_accepted": int(events_per_second * 60 * 0.992),
            "events_rejected": int(events_per_second * 60 * 0.008),
            "acceptance_rate": 0.992,
            "latency": {
                "min_ms": 8.5,
                "max_ms": 450.2,
                "mean_ms": 35.5,
                "p50_ms": 28.3,
                "p90_ms": 65.2,
                "p95_ms": 95.5,
                "p99_ms": 180.3,
            },
            "throughput": {
                "batches_per_second": batches_per_second,
                "events_per_second": events_per_second,
                "duration_seconds": 60.0,
            },
            "payload": {
                "avg_batch_size": 100.0,
                "total_bytes": int(events_per_second * 60 * 500),  # ~500 bytes per event
            },
        },
        "resources": {
            "cpu": {
                "mean_percent": 55.2,
                "max_percent": 78.5,
                "p95_percent": 75.0,
            },
            "memory": {
                "mean_percent": 40.5,
                "max_percent": 52.3,
                "mean_used_mb": 650.2,
                "max_used_mb": 850.5,
            },
            "process": {
                "cpu_mean_percent": 35.5,
                "cpu_max_percent": 55.2,
                "memory_mean_mb": 320.5,
                "memory_max_mb": 420.3,
            },
            "io": {
                "disk_read_mb": 50.2,
                "disk_write_mb": 280.5,
                "net_sent_mb": 180.3,
                "net_recv_mb": 850.2,
            },
            "duration_seconds": 60.0,
            "sample_count": 120,
        },
    }

    # Generate resource baseline
    results["benchmarks"]["resource_baseline"] = {
        "type": "resource_baseline",
        "metrics": {
            "cpu": {
                "mean_percent": 5.2,
                "max_percent": 12.5,
                "p95_percent": 10.0,
            },
            "memory": {
                "mean_percent": 25.5,
                "max_percent": 28.3,
                "mean_used_mb": 420.5,
                "max_used_mb": 450.2,
            },
            "process": {
                "cpu_mean_percent": 0.5,
                "cpu_max_percent": 2.0,
                "memory_mean_mb": 150.2,
                "memory_max_mb": 180.5,
            },
            "io": {
                "disk_read_mb": 5.2,
                "disk_write_mb": 2.5,
                "net_sent_mb": 0.5,
                "net_recv_mb": 0.8,
            },
            "duration_seconds": 10.0,
            "sample_count": 20,
        },
    }

    # Generate summary
    results["summary"] = {
        "passed": True,
        "checks": [
            {
                "name": "http_p95_latency",
                "passed": avg_p95 <= 200,
                "value": avg_p95,
                "threshold": 200.0,
            },
            {
                "name": "http_throughput",
                "passed": avg_rps >= 100,
                "value": avg_rps,
                "threshold": 100.0,
            },
            {
                "name": "http_error_rate",
                "passed": avg_error <= 0.01,
                "value": avg_error,
                "threshold": 0.01,
            },
            {
                "name": "telemetry_events_per_second",
                "passed": events_per_second >= 1000,
                "value": events_per_second,
                "threshold": 1000.0,
            },
            {
                "name": "telemetry_acceptance_rate",
                "passed": True,
                "value": 0.992,
                "threshold": 0.95,
            },
        ],
    }

    # Check if all passed
    results["summary"]["passed"] = all(c["passed"] for c in results["summary"]["checks"])

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as latest
    latest_path = output_path / "benchmark_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Sample benchmark results generated:")
    print(f"  - {filepath}")
    print(f"  - {latest_path}")

    return str(filepath)


if __name__ == "__main__":
    generate_sample_results()
