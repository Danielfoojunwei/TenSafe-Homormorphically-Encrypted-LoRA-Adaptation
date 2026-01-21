"""
Telemetry Ingest Benchmarking Module

Measures telemetry ingestion throughput and latency by simulating
edge agents sending telemetry batches to the control plane.
"""

import asyncio
import hashlib
import hmac
import time
import uuid
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import random

import httpx

from .config import BenchmarkConfig


@dataclass
class IngestResult:
    """Result of a single telemetry batch ingestion."""
    batch_id: str
    batch_size: int
    latency_ms: float
    accepted: int
    rejected: int
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class IngestMetrics:
    """Aggregated metrics for telemetry ingestion."""
    total_batches: int
    total_events: int
    successful_batches: int
    failed_batches: int

    # Events metrics
    events_accepted: int
    events_rejected: int
    acceptance_rate: float

    # Latency metrics (milliseconds)
    latency_min: float
    latency_max: float
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float

    # Throughput
    batches_per_second: float
    events_per_second: float
    duration_seconds: float

    # Payload stats
    avg_batch_size: float
    total_payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_batches": self.total_batches,
            "total_events": self.total_events,
            "successful_batches": self.successful_batches,
            "failed_batches": self.failed_batches,
            "events_accepted": self.events_accepted,
            "events_rejected": self.events_rejected,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "latency": {
                "min_ms": round(self.latency_min, 2),
                "max_ms": round(self.latency_max, 2),
                "mean_ms": round(self.latency_mean, 2),
                "p50_ms": round(self.latency_p50, 2),
                "p90_ms": round(self.latency_p90, 2),
                "p95_ms": round(self.latency_p95, 2),
                "p99_ms": round(self.latency_p99, 2),
            },
            "throughput": {
                "batches_per_second": round(self.batches_per_second, 2),
                "events_per_second": round(self.events_per_second, 2),
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "payload": {
                "avg_batch_size": round(self.avg_batch_size, 1),
                "total_bytes": self.total_payload_bytes,
            }
        }


class TelemetryBenchmark:
    """Telemetry ingestion benchmarking suite."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[IngestResult] = []
        self.fleet_id: Optional[str] = None
        self.fleet_api_key: Optional[str] = None
        self.auth_token: Optional[str] = None
        self._lock = threading.Lock()
        self._total_bytes = 0

    def setup(self) -> bool:
        """Set up authentication and fleet for benchmarking."""
        try:
            with httpx.Client(base_url=self.config.base_url, timeout=30.0) as client:
                # Initialize tenant if needed
                try:
                    init_resp = client.post(
                        "/api/v1/onboarding/init",
                        params={
                            "name": "Telemetry Bench Tenant",
                            "admin_email": self.config.admin_email,
                            "admin_pass": self.config.admin_password,
                        }
                    )
                except Exception:
                    pass

                # Get auth token
                response = client.post(
                    "/api/v1/auth/token",
                    json={
                        "username": self.config.admin_email,
                        "password": self.config.admin_password,
                    }
                )

                if response.status_code == 200:
                    data = response.json()
                    self.auth_token = data["access_token"]
                    print("Authentication successful")

                    # Create fleet
                    fleet_resp = client.post(
                        "/api/v1/fleets",
                        params={"name": f"TelemetryBenchFleet_{uuid.uuid4().hex[:8]}"},
                        headers={"Authorization": f"Bearer {self.auth_token}"}
                    )
                    if fleet_resp.status_code == 200:
                        fleet_data = fleet_resp.json()
                        self.fleet_id = fleet_data.get("id")
                        self.fleet_api_key = fleet_data.get("api_key")
                        print(f"Created benchmark fleet: {self.fleet_id}")
                        return True

                return False

        except Exception as e:
            print(f"Setup error: {e}")
            return False

    def _compute_hmac(self, timestamp: str, nonce: str, body: bytes) -> str:
        """Compute HMAC signature for telemetry authentication."""
        if not self.fleet_api_key:
            return ""

        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{timestamp}:{nonce}:{body_hash}"
        signature = hmac.new(
            self.fleet_api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _generate_telemetry_batch(
        self,
        device_id: str,
        batch_size: int,
        include_varied_topics: bool = True
    ) -> Dict[str, Any]:
        """Generate a realistic telemetry batch."""
        messages = []
        timestamp_ns = int(time.time() * 1_000_000_000)

        topics = ["telemetry.stage", "telemetry.system", "telemetry.model_behavior"]
        stages = ["capture", "embed", "gate", "peft", "shield", "sync", "pull"]

        for i in range(batch_size):
            if include_varied_topics:
                topic = random.choice(topics)
            else:
                topic = "telemetry.stage"

            if topic == "telemetry.stage":
                payload = {
                    "device_id": device_id,
                    "stage": random.choice(stages),
                    "status": random.choice(["ok", "ok", "ok", "degraded"]),  # 75% ok
                    "latency_ms": random.uniform(10, 200),
                    "run_id": f"run_{uuid.uuid4().hex[:8]}",
                    "metadata": {
                        "batch_idx": i,
                        "model_version": "v1.0.0",
                    }
                }
            elif topic == "telemetry.system":
                payload = {
                    "device_id": device_id,
                    "cpu_pct": random.uniform(20, 80),
                    "mem_pct": random.uniform(30, 70),
                    "gpu_pct": random.uniform(0, 100) if random.random() > 0.5 else None,
                    "temp_c": random.uniform(40, 80),
                    "dropped_frames": random.randint(0, 10),
                }
            else:  # model_behavior
                payload = {
                    "device_id": device_id,
                    "model_version": "v1.0.0",
                    "adapter_id": f"adapter_{random.randint(1, 5)}",
                    "decision_hash": hashlib.sha256(f"{i}".encode()).hexdigest()[:16],
                    "refusal_rate": random.uniform(0, 0.1),
                    "tool_call_failures": random.randint(0, 5),
                    "is_shadow": random.random() > 0.9,
                }

            messages.append({
                "topic": topic,
                "timestamp_ns": timestamp_ns + i * 1000,
                "payload": payload,
                "priority": 0
            })

        return {
            "batch_id": f"batch_{uuid.uuid4().hex}",
            "device_info": {
                "device_id": device_id,
                "agent_version": "3.0.0",
                "runtime_version": "python3.11",
                "ros_distro": "humble",
            },
            "messages": messages
        }

    def _send_batch(
        self,
        client: httpx.Client,
        device_id: str,
        batch_size: int
    ) -> IngestResult:
        """Send a single telemetry batch and record the result."""
        batch = self._generate_telemetry_batch(device_id, batch_size)
        batch_id = batch["batch_id"]

        import json
        body = json.dumps(batch).encode()

        timestamp = str(int(time.time()))
        nonce = uuid.uuid4().hex

        headers = {
            "Content-Type": "application/json",
            "X-TG-Fleet-Id": self.fleet_id or "",
            "X-TG-Timestamp": timestamp,
            "X-TG-Nonce": nonce,
            "X-TG-Signature": self._compute_hmac(timestamp, nonce, body),
        }

        start_time = time.perf_counter()

        try:
            response = client.post(
                "/api/v1/telemetry/ingest",
                content=body,
                headers=headers,
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                with self._lock:
                    self._total_bytes += len(body)

                return IngestResult(
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    accepted=data.get("accepted", 0),
                    rejected=data.get("rejected", 0),
                    success=True,
                )
            else:
                return IngestResult(
                    batch_id=batch_id,
                    batch_size=batch_size,
                    latency_ms=latency_ms,
                    accepted=0,
                    rejected=batch_size,
                    success=False,
                    error=f"HTTP {response.status_code}",
                )

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return IngestResult(
                batch_id=batch_id,
                batch_size=batch_size,
                latency_ms=latency_ms,
                accepted=0,
                rejected=batch_size,
                success=False,
                error="Timeout",
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return IngestResult(
                batch_id=batch_id,
                batch_size=batch_size,
                latency_ms=latency_ms,
                accepted=0,
                rejected=batch_size,
                success=False,
                error=str(e),
            )

    def _worker(
        self,
        device_id: str,
        num_batches: int,
        batch_size: int,
        delay_between_batches: float
    ):
        """Worker function for concurrent batch generation."""
        with httpx.Client(
            base_url=self.config.base_url,
            timeout=30.0
        ) as client:
            for _ in range(num_batches):
                result = self._send_batch(client, device_id, batch_size)

                with self._lock:
                    self.results.append(result)

                if delay_between_batches > 0:
                    time.sleep(delay_between_batches)

    def run_benchmark(
        self,
        num_agents: int = 10,
        batches_per_agent: int = 100,
        batch_size: int = 100,
        delay_between_batches: float = 0.0
    ) -> IngestMetrics:
        """Run telemetry ingestion benchmark."""
        print(f"Running telemetry benchmark...")
        print(f"  Agents: {num_agents}")
        print(f"  Batches per agent: {batches_per_agent}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total events: {num_agents * batches_per_agent * batch_size}")

        self.results = []
        self._total_bytes = 0

        start_time = time.time()

        # Run concurrent agent workers
        with ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = [
                executor.submit(
                    self._worker,
                    f"device_{i:04d}",
                    batches_per_agent,
                    batch_size,
                    delay_between_batches
                )
                for i in range(num_agents)
            ]
            for future in futures:
                future.result()

        duration = time.time() - start_time

        return self._calculate_metrics(duration)

    def _calculate_metrics(self, duration: float) -> IngestMetrics:
        """Calculate aggregated metrics from results."""
        if not self.results:
            return IngestMetrics(
                total_batches=0,
                total_events=0,
                successful_batches=0,
                failed_batches=0,
                events_accepted=0,
                events_rejected=0,
                acceptance_rate=0,
                latency_min=0,
                latency_max=0,
                latency_mean=0,
                latency_p50=0,
                latency_p90=0,
                latency_p95=0,
                latency_p99=0,
                batches_per_second=0,
                events_per_second=0,
                duration_seconds=duration,
                avg_batch_size=0,
                total_payload_bytes=0,
            )

        latencies = [r.latency_ms for r in self.results]
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        total_events = sum(r.batch_size for r in self.results)
        events_accepted = sum(r.accepted for r in self.results)
        events_rejected = sum(r.rejected for r in self.results)

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_latencies[min(idx, n - 1)]

        return IngestMetrics(
            total_batches=len(self.results),
            total_events=total_events,
            successful_batches=len(successful),
            failed_batches=len(failed),
            events_accepted=events_accepted,
            events_rejected=events_rejected,
            acceptance_rate=events_accepted / total_events if total_events > 0 else 0,
            latency_min=min(latencies),
            latency_max=max(latencies),
            latency_mean=statistics.mean(latencies),
            latency_p50=percentile(50),
            latency_p90=percentile(90),
            latency_p95=percentile(95),
            latency_p99=percentile(99),
            batches_per_second=len(self.results) / duration if duration > 0 else 0,
            events_per_second=total_events / duration if duration > 0 else 0,
            duration_seconds=duration,
            avg_batch_size=total_events / len(self.results) if self.results else 0,
            total_payload_bytes=self._total_bytes,
        )

    def run_load_levels(self) -> Dict[str, IngestMetrics]:
        """Run benchmarks at multiple load levels."""
        results = {}

        # Light load: 5 agents, 50 batches, 50 events each
        print("\n--- Light Load ---")
        results["light"] = self.run_benchmark(
            num_agents=5,
            batches_per_agent=50,
            batch_size=50,
        )

        # Moderate load: 10 agents, 100 batches, 100 events each
        print("\n--- Moderate Load ---")
        results["moderate"] = self.run_benchmark(
            num_agents=10,
            batches_per_agent=100,
            batch_size=100,
        )

        # Heavy load: 20 agents, 100 batches, 200 events each
        print("\n--- Heavy Load ---")
        results["heavy"] = self.run_benchmark(
            num_agents=20,
            batches_per_agent=100,
            batch_size=200,
        )

        return results


def run_telemetry_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run telemetry benchmarks and return results."""
    benchmark = TelemetryBenchmark(config)

    if not benchmark.setup():
        print("Warning: Setup failed, results may be incomplete")

    # Run at configured load level
    metrics = benchmark.run_benchmark(
        num_agents=config.load.concurrent_users,
        batches_per_agent=max(10, config.load.duration_seconds),
        batch_size=config.load.batch_size,
    )

    # Print summary
    print(f"\nTelemetry Benchmark Results:")
    print(f"  Total events: {metrics.total_events}")
    print(f"  Events/sec: {metrics.events_per_second:.1f}")
    print(f"  Batches/sec: {metrics.batches_per_second:.1f}")
    print(f"  Acceptance rate: {metrics.acceptance_rate:.2%}")
    print(f"  Latency p50/p95/p99: {metrics.latency_p50:.1f}/{metrics.latency_p95:.1f}/{metrics.latency_p99:.1f}ms")

    return {
        "type": "telemetry_ingest",
        "metrics": metrics.to_dict(),
    }
