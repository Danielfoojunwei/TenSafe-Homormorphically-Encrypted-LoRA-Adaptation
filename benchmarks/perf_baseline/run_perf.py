#!/usr/bin/env python3
"""
TensorGuardFlow Performance Baseline Runner

Executes a standard performance scenario and captures metrics.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def run_perf_baseline(output_path: str):
    """Run performance baseline tests."""

    results = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "python_version": sys.version,
            "tg_deterministic": os.environ.get("TG_DETERMINISTIC", "false"),
            "tg_simulation": os.environ.get("TG_SIMULATION", "false"),
        },
        "scenarios": [],
        "summary": {}
    }

    # Import after path setup
    try:
        from fastapi.testclient import TestClient
        from sqlmodel import SQLModel, create_engine, Session
        from sqlalchemy.pool import StaticPool

        # Setup test environment
        os.environ["TG_SIMULATION"] = "true"
        os.environ["TG_DEMO_MODE"] = "true"
        os.environ["TG_DETERMINISTIC"] = "true"

        from tensorguard.platform import database

        test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool
        )
        database.engine = test_engine

        # Import models
        from tensorguard.platform.models.continuous_models import *
        from tensorguard.platform.models.evidence_models import *
        from tensorguard.platform.models.tgflow_core_models import *
        from tensorguard.platform.models.metrics_models import *
        from tensorguard.platform.models.core import *

        SQLModel.metadata.create_all(test_engine)

        from tensorguard.platform.main import app
        from tensorguard.platform.database import get_session
        from tensorguard.platform.auth import get_current_user
        from tensorguard.platform.dependencies import require_tenant_context

        # Override dependencies
        def get_session_override():
            with Session(test_engine) as session:
                yield session

        def get_current_user_override():
            return User(id="perf-user", email="perf@test.com", is_active=True, tenant_id="perf-tenant")

        async def require_tenant_context_override():
            return "perf-tenant"

        app.dependency_overrides[get_session] = get_session_override
        app.dependency_overrides[get_current_user] = get_current_user_override
        app.dependency_overrides[require_tenant_context] = require_tenant_context_override

        client = TestClient(app)
        headers = {"X-Tenant-ID": "perf-tenant"}

        # Scenario 1: Route CRUD latency
        print("Running Scenario 1: Route CRUD latency...")
        scenario1 = {"name": "route_crud", "operations": []}

        for i in range(5):
            start = time.time()
            resp = client.post("/api/v1/tgflow/routes", headers=headers,
                              json={"route_key": f"perf-route-{i}", "base_model_ref": "m1"})
            latency = (time.time() - start) * 1000
            scenario1["operations"].append({
                "operation": "create_route",
                "iteration": i,
                "latency_ms": latency,
                "status": resp.status_code
            })

        # List routes
        start = time.time()
        resp = client.get("/api/v1/tgflow/routes", headers=headers)
        latency = (time.time() - start) * 1000
        scenario1["operations"].append({
            "operation": "list_routes",
            "latency_ms": latency,
            "status": resp.status_code,
            "count": len(resp.json())
        })

        results["scenarios"].append(scenario1)

        # Scenario 2: Dashboard bundle latency
        print("Running Scenario 2: Dashboard bundle latency...")
        scenario2 = {"name": "dashboard_bundle", "operations": []}

        # Setup a route
        route_key = "dashboard-perf-test"
        client.post("/api/v1/tgflow/routes", headers=headers,
                   json={"route_key": route_key, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/feed", headers=headers,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "off"})
        client.post(f"/api/v1/tgflow/routes/{route_key}/policy", headers=headers,
                   json={"novelty_threshold": 0.1})

        # Cold request
        start = time.time()
        resp = client.get(f"/api/v1/metrics/routes/{route_key}/dashboard_bundle",
                         headers=headers, params={"tenant_id": "perf-tenant"})
        cold_latency = (time.time() - start) * 1000
        scenario2["operations"].append({
            "operation": "dashboard_bundle_cold",
            "latency_ms": cold_latency,
            "status": resp.status_code,
            "budget_ms": 500,
            "passed": cold_latency < 500
        })

        # Warm requests
        warm_latencies = []
        for i in range(5):
            start = time.time()
            resp = client.get(f"/api/v1/metrics/routes/{route_key}/dashboard_bundle",
                             headers=headers, params={"tenant_id": "perf-tenant"})
            latency = (time.time() - start) * 1000
            warm_latencies.append(latency)

        avg_warm = sum(warm_latencies) / len(warm_latencies)
        scenario2["operations"].append({
            "operation": "dashboard_bundle_warm",
            "latency_ms": avg_warm,
            "latencies": warm_latencies,
            "budget_ms": 200,
            "passed": avg_warm < 200
        })

        results["scenarios"].append(scenario2)

        # Scenario 3: Health endpoint latency
        print("Running Scenario 3: Health endpoint latency...")
        scenario3 = {"name": "health_endpoints", "operations": []}

        for endpoint in ["/health", "/ready", "/live"]:
            latencies = []
            for _ in range(10):
                start = time.time()
                resp = client.get(endpoint)
                latencies.append((time.time() - start) * 1000)

            scenario3["operations"].append({
                "operation": endpoint,
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "status": resp.status_code
            })

        results["scenarios"].append(scenario3)

        # Scenario 4: Resolve endpoint latency (with N2HE)
        print("Running Scenario 4: Resolve endpoint latency...")
        scenario4 = {"name": "resolve_latency", "operations": []}

        # Setup N2HE route
        n2he_route = "resolve-perf-test"
        client.post("/api/v1/tgflow/routes", headers=headers,
                   json={"route_key": n2he_route, "base_model_ref": "m1"})
        client.post(f"/api/v1/tgflow/routes/{n2he_route}/feed", headers=headers,
                   json={"feed_type": "local", "feed_uri": "mock://", "privacy_mode": "n2he"})
        client.post(f"/api/v1/tgflow/routes/{n2he_route}/policy", headers=headers,
                   json={"novelty_threshold": 0.1})

        latencies = []
        for _ in range(20):
            start = time.time()
            resp = client.post("/api/v1/tgflow/resolve", headers=headers,
                              json={"route_key": n2he_route})
            latencies.append((time.time() - start) * 1000)

        scenario4["operations"].append({
            "operation": "resolve_n2he",
            "avg_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies)//2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies)*0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies)*0.99)],
            "budget_ms": 50,
            "passed": sum(latencies) / len(latencies) < 50
        })

        results["scenarios"].append(scenario4)

        # Generate summary
        print("Generating summary...")
        results["summary"] = {
            "total_scenarios": len(results["scenarios"]),
            "dashboard_bundle_cold_ms": cold_latency,
            "dashboard_bundle_warm_ms": avg_warm,
            "resolve_avg_ms": sum(latencies) / len(latencies),
            "all_budgets_passed": all(
                op.get("passed", True)
                for scenario in results["scenarios"]
                for op in scenario["operations"]
                if "passed" in op
            )
        }

        # Cleanup
        app.dependency_overrides.clear()

    except Exception as e:
        print(f"Error running performance baseline: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    # Write results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to: {output_path}")

    # Print summary
    print("\n=== Performance Summary ===")
    if "summary" in results:
        for key, value in results["summary"].items():
            print(f"  {key}: {value}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TensorGuardFlow performance baseline")
    parser.add_argument("--output", "-o", default="perf_baseline.json",
                       help="Output file path")
    args = parser.parse_args()

    run_perf_baseline(args.output)
