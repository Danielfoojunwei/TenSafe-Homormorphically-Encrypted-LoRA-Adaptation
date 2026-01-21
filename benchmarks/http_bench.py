"""
HTTP API Benchmarking Module

Measures API endpoint latency and throughput using concurrent HTTP requests.
Supports authentication, various HTTP methods, and detailed latency analysis.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import threading

import httpx

from .config import BenchmarkConfig, EndpointConfig, API_ENDPOINTS


@dataclass
class RequestResult:
    """Result of a single HTTP request."""
    endpoint: str
    method: str
    status_code: int
    latency_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class EndpointMetrics:
    """Aggregated metrics for an endpoint."""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_rate: float

    # Latency metrics (milliseconds)
    latency_min: float
    latency_max: float
    latency_mean: float
    latency_p50: float
    latency_p90: float
    latency_p95: float
    latency_p99: float
    latency_std: float

    # Throughput
    requests_per_second: float
    duration_seconds: float

    # Status code distribution
    status_codes: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "method": self.method,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate": round(self.error_rate, 4),
            "latency": {
                "min_ms": round(self.latency_min, 2),
                "max_ms": round(self.latency_max, 2),
                "mean_ms": round(self.latency_mean, 2),
                "p50_ms": round(self.latency_p50, 2),
                "p90_ms": round(self.latency_p90, 2),
                "p95_ms": round(self.latency_p95, 2),
                "p99_ms": round(self.latency_p99, 2),
                "std_ms": round(self.latency_std, 2),
            },
            "throughput": {
                "requests_per_second": round(self.requests_per_second, 2),
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "status_codes": self.status_codes,
        }


class HTTPBenchmark:
    """HTTP API benchmarking suite."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[RequestResult] = []
        self.auth_token: Optional[str] = None
        self.fleet_api_key: Optional[str] = None
        self.fleet_id: Optional[str] = None
        self._lock = threading.Lock()

    def authenticate(self) -> bool:
        """Obtain authentication token for protected endpoints."""
        try:
            with httpx.Client(base_url=self.config.base_url, timeout=30.0) as client:
                # First, try to initialize tenant if needed
                try:
                    init_resp = client.post(
                        "/api/v1/onboarding/init",
                        params={
                            "name": "Benchmark Tenant",
                            "admin_email": self.config.admin_email,
                            "admin_pass": self.config.admin_password,
                        }
                    )
                    if init_resp.status_code == 200:
                        print(f"Created benchmark tenant")
                except Exception:
                    pass  # Tenant might already exist

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
                    print(f"Authentication successful")

                    # Create a fleet for benchmarking
                    fleet_resp = client.post(
                        "/api/v1/fleets",
                        params={"name": "BenchmarkFleet"},
                        headers={"Authorization": f"Bearer {self.auth_token}"}
                    )
                    if fleet_resp.status_code == 200:
                        fleet_data = fleet_resp.json()
                        self.fleet_id = fleet_data.get("id")
                        self.fleet_api_key = fleet_data.get("api_key")
                        print(f"Created benchmark fleet: {self.fleet_id}")

                    return True
                else:
                    print(f"Authentication failed: {response.status_code}")
                    return False

        except Exception as e:
            print(f"Authentication error: {e}")
            return False

    def _make_request(
        self,
        client: httpx.Client,
        endpoint: EndpointConfig,
        headers: Dict[str, str]
    ) -> RequestResult:
        """Make a single HTTP request and record the result."""
        start_time = time.perf_counter()

        try:
            if endpoint.method == "GET":
                response = client.get(endpoint.path, headers=headers)
            elif endpoint.method == "POST":
                response = client.post(
                    endpoint.path,
                    headers=headers,
                    json=endpoint.payload_template or {}
                )
            else:
                raise ValueError(f"Unsupported method: {endpoint.method}")

            latency_ms = (time.perf_counter() - start_time) * 1000
            success = 200 <= response.status_code < 400

            return RequestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                status_code=response.status_code,
                latency_ms=latency_ms,
                success=success,
            )

        except httpx.TimeoutException:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                status_code=0,
                latency_ms=latency_ms,
                success=False,
                error="Timeout",
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return RequestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                status_code=0,
                latency_ms=latency_ms,
                success=False,
                error=str(e),
            )

    def _worker(
        self,
        endpoint: EndpointConfig,
        num_requests: int,
        delay_between_requests: float
    ):
        """Worker function for concurrent request generation."""
        headers = {}
        if endpoint.requires_auth and self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        with httpx.Client(
            base_url=self.config.base_url,
            timeout=10.0
        ) as client:
            for _ in range(num_requests):
                result = self._make_request(client, endpoint, headers)

                with self._lock:
                    self.results.append(result)

                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

    def benchmark_endpoint(
        self,
        endpoint: EndpointConfig,
        concurrent_users: int = 10,
        total_requests: int = 1000,
    ) -> EndpointMetrics:
        """Benchmark a single endpoint with concurrent users."""
        print(f"Benchmarking {endpoint.method} {endpoint.path}...")

        # Clear previous results for this endpoint
        self.results = []

        requests_per_worker = total_requests // concurrent_users
        delay = 0  # No artificial delay for maximum throughput

        start_time = time.time()

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(self._worker, endpoint, requests_per_worker, delay)
                for _ in range(concurrent_users)
            ]
            for future in futures:
                future.result()

        duration = time.time() - start_time

        # Calculate metrics
        return self._calculate_metrics(endpoint, duration)

    def _calculate_metrics(
        self,
        endpoint: EndpointConfig,
        duration: float
    ) -> EndpointMetrics:
        """Calculate aggregated metrics from results."""
        endpoint_results = [r for r in self.results if r.endpoint == endpoint.path]

        if not endpoint_results:
            return EndpointMetrics(
                endpoint=endpoint.path,
                method=endpoint.method,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                error_rate=1.0,
                latency_min=0,
                latency_max=0,
                latency_mean=0,
                latency_p50=0,
                latency_p90=0,
                latency_p95=0,
                latency_p99=0,
                latency_std=0,
                requests_per_second=0,
                duration_seconds=duration,
            )

        latencies = [r.latency_ms for r in endpoint_results]
        successful = [r for r in endpoint_results if r.success]
        failed = [r for r in endpoint_results if not r.success]

        # Status code distribution
        status_codes: Dict[int, int] = {}
        for r in endpoint_results:
            status_codes[r.status_code] = status_codes.get(r.status_code, 0) + 1

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_latencies[min(idx, n - 1)]

        return EndpointMetrics(
            endpoint=endpoint.path,
            method=endpoint.method,
            total_requests=len(endpoint_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            error_rate=len(failed) / len(endpoint_results) if endpoint_results else 0,
            latency_min=min(latencies),
            latency_max=max(latencies),
            latency_mean=statistics.mean(latencies),
            latency_p50=percentile(50),
            latency_p90=percentile(90),
            latency_p95=percentile(95),
            latency_p99=percentile(99),
            latency_std=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            requests_per_second=len(endpoint_results) / duration if duration > 0 else 0,
            duration_seconds=duration,
            status_codes=status_codes,
        )

    def run_all_benchmarks(
        self,
        endpoints: Optional[List[EndpointConfig]] = None
    ) -> Dict[str, EndpointMetrics]:
        """Run benchmarks on all configured endpoints."""
        if endpoints is None:
            endpoints = API_ENDPOINTS

        # Authenticate first
        if not self.authenticate():
            print("Warning: Running without authentication")

        results: Dict[str, EndpointMetrics] = {}
        total_requests = self.config.load.requests_per_second * self.config.load.duration_seconds

        for endpoint in endpoints:
            if endpoint.requires_auth and not self.auth_token:
                print(f"Skipping {endpoint.path} (requires auth)")
                continue

            metrics = self.benchmark_endpoint(
                endpoint,
                concurrent_users=self.config.load.concurrent_users,
                total_requests=min(total_requests, 10000),  # Cap at 10k per endpoint
            )
            results[endpoint.path] = metrics

            # Print summary
            print(f"  Requests: {metrics.total_requests}")
            print(f"  RPS: {metrics.requests_per_second:.1f}")
            print(f"  Latency p50/p95/p99: {metrics.latency_p50:.1f}/{metrics.latency_p95:.1f}/{metrics.latency_p99:.1f}ms")
            print(f"  Error rate: {metrics.error_rate:.2%}")
            print()

        return results


def run_http_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """Run HTTP benchmarks and return results."""
    benchmark = HTTPBenchmark(config)
    metrics = benchmark.run_all_benchmarks()

    return {
        "type": "http_api",
        "endpoints": {path: m.to_dict() for path, m in metrics.items()},
        "summary": {
            "total_endpoints": len(metrics),
            "avg_p95_latency_ms": statistics.mean([m.latency_p95 for m in metrics.values()]) if metrics else 0,
            "avg_rps": statistics.mean([m.requests_per_second for m in metrics.values()]) if metrics else 0,
            "avg_error_rate": statistics.mean([m.error_rate for m in metrics.values()]) if metrics else 0,
        }
    }
