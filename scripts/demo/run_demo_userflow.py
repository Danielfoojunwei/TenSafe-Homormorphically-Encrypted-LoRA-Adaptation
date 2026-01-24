#!/usr/bin/env python3
"""
Phase 1: API Demo Driver

Executes the complete continuous-learning user flow via API.
Includes strict assertions and NDJSON tracing.
"""

import sys
import os
import time
import json
import logging
import requests
from datetime import datetime, UTC
import argparse

# Configuration
BASE_URL = os.getenv("TG_API_URL", "http://localhost:8000/api/v1")
DEMO_PROOF_DIR = os.getenv("TG_DEMO_PROOF_DIR", "demo_proof/latest")
TRACE_FILE = os.path.join(DEMO_PROOF_DIR, "api_trace.ndjson")
TENANT_ID = "fceac734-e672-4a0c-863b-c7bb8e28b88e" # Demo Tenant
HEADERS = {"X-Tenant-ID": TENANT_ID}

# Fixture Paths
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "../../demos/fixtures")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("demo_driver")

def trace(action: str, url: str, method: str, status: int, payload: dict = None, response: dict = None):
    """Log trace event to NDJSON."""
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "action": action,
        "method": method,
        "url": url,
        "status": status,
        "payload": payload,
        "response": response,
    }
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")

def load_fixture(name: str):
    with open(os.path.join(FIXTURES_DIR, name)) as f:
        return json.load(f)

def run_step(name: str, method: str, endpoint: str, data: dict = None, params: dict = None, expected_status: int = 200) -> dict:
    url = f"{BASE_URL}{endpoint}"
    logger.info(f"STEP: {name} | {method} {endpoint}")
    
    try:
        if method == "POST":
            resp = requests.post(url, json=data, params=params, headers=HEADERS)
        elif method == "GET":
            # If data is provided but params is not, treat data as params for convenience in GET
            query_params = params if params is not None else data
            resp = requests.get(url, headers=HEADERS, params=query_params)
        else:
            raise ValueError(f"Unsupported method {method}")
            
        try:
            resp_data = resp.json()
        except:
            resp_data = {"text": resp.text}
            
        trace(name, url, method, resp.status_code, data, resp_data)
        
        if resp.status_code != expected_status:
            logger.error(f"FAILED {name}: Expected {expected_status}, got {resp.status_code}")
            logger.error(f"Response: {resp.text}")
            sys.exit(1)
            
        return resp_data
    except Exception as e:
        logger.error(f"EXCEPTION in {name}: {str(e)}")
        sys.exit(1)

def main():
    # Ensure proof dir exists
    os.makedirs(DEMO_PROOF_DIR, exist_ok=True)
    
    # Load fixtures
    route_config = load_fixture("demo_route.json")
    route_key = route_config["route_key"]
    policy_config = load_fixture("demo_policy.json")
    feed_config = load_fixture("demo_feed.json")
    
    logger.info(">>> STARTING DEMO DRIVER <<<")

    # Step 1: Create Route
    # First, try to delete if exists to ensure determinism (optional, relying on cleanup script is better but this is safer)
    # We'll validly fail if unique constraint.
    
    try:
        run_step("Create Route", "POST", "/tgflow/routes", route_config)
    except SystemExit:
        # If it failed, check if it was 400 (already exists), if so, proceed.
        # But wait, run_step exits on failure. We need to modify run_step or wrapped call.
        # Let's verify manually or just ignore the exit if we can inspect the log.
        # Better: check existence first.
        pass

    # Check if route exists via list or get
    check = requests.get(f"{BASE_URL}/tgflow/routes/{route_key}", headers=HEADERS)
    if check.status_code == 404:
         # It didn't exist and create failed?
         # Re-try create, should work.
         run_step("Create Route", "POST", "/tgflow/routes", route_config)
    elif check.status_code == 200:
         logger.info(f"Route {route_key} already exists. Using existing.")
    else:
         logger.error(f"Cannot check route existence: {check.status_code}")
         sys.exit(1)
    
    # Step 2: Connect Feed
    run_step("Connect Feed", "POST", f"/tgflow/routes/{route_key}/feed", feed_config)
    
    # Step 3: Set Policy
    run_step("Set Policy", "POST", f"/tgflow/routes/{route_key}/policy", policy_config)
    
    # Step 4: Run Once (Initial)
    # This triggers ingestion and initial training (novelty defaults to 1.0/high on first run usually)
    run_res = run_step("Run Loop (1)", "POST", f"/tgflow/routes/{route_key}/run_once")
    logger.info(f"Run Loop Response: {json.dumps(run_res, indent=2)}")
    loop_id = run_res.get("loop_id")
    logger.info(f"Loop ID: {loop_id}")
    
    # Wait for processing? API might be async or sync. 
    # run_once in orchestrator is async but the endpoint probably waits or returns status.
    # If it returns "verdict", it likely completed the synchronous part of the loop logic. 
    # But training might be backgrounded? Check orchestrator. It awaits training.
    
    # Step 5: Fetch Timeline
    timeline = run_step("Fetch Timeline", "GET", f"/tgflow/routes/{route_key}/timeline")
    timeline_groups = timeline.get("timeline", [])
    events_flat = []
    for group in timeline_groups:
        events_flat.extend(group.get("events", []))
        
    event_types = [e.get("stage") for e in events_flat]
    logger.info(f"Events found: {event_types}")
    
    # Step 6: Verify Candidate Created
    logger.info("Verifying Candidate/Canary creation...")
    time.sleep(2) # Allow for async updates if any
    route_details = run_step("Get Route Details", "GET", f"/tgflow/routes/{route_key}")
    
    route_obj = route_details.get("route", {})
    canary_id = route_obj.get("canary_adapter_id")
    
    if not canary_id:
        # Fallback: Check if active_adapter changed (if auto-promote to stable was on?)
        # Demo policy has auto_promote_to_stable: false, auto_promote_to_canary: true
        # So canary_id should be set.
        # If run_once failed to produce adapter, we fail here.
        logger.warning("No canary_adapter_id found. Checking available adapters via registry list might be needed, or run failed.")
        # For demo proof, we must succeed.
        # Let's check timeline again for explicit error if missing.
        sys.exit(1)
        
    logger.info(f"Candidate/Canary Adapter Found: {canary_id}")
    
    # Step 7: Promote Candidate to Stable
    logger.info(f"Promoting {canary_id} to STABLE...")
    run_step("Promote to Stable", "POST", f"/tgflow/routes/{route_key}/promote", 
             params={"adapter_id": canary_id, "target": "stable"}, expected_status=200)

    # Step 8: Resolve
    resolve_res = run_step("Resolve Adapter", "POST", "/tgflow/resolve", {
        "route_key": route_key,
        "context": {"user_id": "demo-user"}
    })
    resolved_id = resolve_res.get("adapter_id")
    logger.info(f"Resolved Adapter: {resolved_id}")
    
    if resolved_id != canary_id:
        logger.error(f"Resolution mismatch! Expected {canary_id} (new stable), got {resolved_id}")
        # Note: Promote updates active/stable. Resolve should reflect that.
        sys.exit(1)
    
    # Step 9: Run Once Again
    run_res2 = run_step("Run Loop (2)", "POST", f"/tgflow/routes/{route_key}/run_once")
    
    # We need a SECOND promotion to test rollback (requires a fallback to exist)
    logger.info("Promoting SECOND candidate to STABLE to enable rollback testing...")
    route_details2 = run_step("Get Route Details (2)", "GET", f"/tgflow/routes/{route_key}")
    canary_id2 = route_details2["route"].get("canary_adapter_id")
    if not canary_id2:
        logger.error("Run Loop (2) did not produce a new candidate.")
        sys.exit(1)
        
    run_step("Promote to Stable (2)", "POST", f"/tgflow/routes/{route_key}/promote", 
             params={"adapter_id": canary_id2, "target": "stable"}, expected_status=200)

    # Step 10: Diff (What changed?)
    # Compare original fallback/base vs new stable (canary_id2)
    diff_res = run_step("Get Diff", "GET", f"/tgflow/routes/{route_key}/diff", 
                        {"from_adapter": canary_id, "to_adapter": canary_id2})
    logger.info(f"Diff Summary: {len(diff_res.get('changes', []))} changes detected.")
    
    # Step 11: Rollback
    run_step("Rollback", "POST", f"/tgflow/routes/{route_key}/rollback", {})
    
    # Verify rollback
    route_details_after = run_step("Get Route Details After Rollback", "GET", f"/tgflow/routes/{route_key}")
    active_after = route_details_after["route"]["active_adapter_id"]
    logger.info(f"Active Adapter After Rollback: {active_after}")
    
    if active_after == canary_id2:
         logger.error("Rollback failed! Active adapter did not change from canary 2.")
         sys.exit(1)
         
    if active_after != canary_id:
         logger.error(f"Rollback to unexpected adapter! Expected {canary_id}, got {active_after}")
         sys.exit(1)

    # Step 12: Export
    run_step("Export Spec", "POST", f"/tgflow/routes/{route_key}/export", params={"backend": "k8s"}, expected_status=200)

    logger.info(">>> DEMO DRIVER SUCCESS <<<")

if __name__ == "__main__":
    main()
