"""
Pipeline Orchestrator

The main entry point for running Enablement Jobs.
Connects Platform Adapter -> Data Layer -> Core Logic -> Governance -> Platform Adapter.
"""

import sys
import logging
import traceback
import uuid
import json
from pathlib import Path
from typing import Dict, Any, TypeVar, Type

# Imports from Enablement Layer
from ..external_platform.base import ExternalPlatformAdapter, RunContext
from ..external_platform.adapters.filesystem import FilesystemAdapter
from ..robotics.ros2.bag_reader import RosbagReader
from ..governance.policy import GovernanceEngine, PolicyViolation

# Centralized exception hierarchy
from ...utils.exceptions import PipelineError, InputError, ContractError, PublishError

logger = logging.getLogger(__name__)

def run_pipeline(
    adapter: ExternalPlatformAdapter,
    run_ctx: RunContext,
    governance: GovernanceEngine
):
    """
    Execute the trust pipeline.
    """
    try:
        logger.info(f"--- Starting Run {run_ctx.run_id} ---")
        adapter.update_status(run_ctx, "RUNNING", "Fetching inputs...")
        
        # 1. Fetch Inputs
        try:
            input_path = adapter.fetch_inputs(run_ctx)
        except Exception as e:
            raise InputError(f"Failed to fetch inputs: {e}")
            
        logger.info(f"Input acquired: {input_path}")
        
        # 2. Ingest / Validate (Robotics Layer)
        # Using RosbagReader just to verify/scan
        try:
            with RosbagReader(input_path) as bag:
                topic_count = len(bag.get_topics())
                msg_count = bag.get_message_count()
                logger.info(f"Bag validated: {topic_count} topics, {msg_count} messages")
                
                # Report metrics
                adapter.record_metric(run_ctx, "input_messages", msg_count)
                adapter.record_metric(run_ctx, "duration_sec", bag.get_duration(), "s")
                
        except Exception as e:
             raise ContractError(f"Invalid bag format: {e}")

        # 3. Core Logic (Placeholder for MOAI/N2HE job linkage)
        # In a real job, we'd invoke tensorguard.core here
        logger.info("Running Core Logic (Privacy Evaluation)...")
        # Simulating privacy budget usage
        dp_cost = 0.5
        
        # 4. Governance Check
        try:
            governance.check_dp_budget(current_spend=9.0, cost=dp_cost)
            governance.check_rollback_contract(["rollback_manifest.json", "patch.pt"])
        except PolicyViolation as e:
            # Policy failures are "SUCCESS" runs (job finished) but with a DENIED result
            adapter.update_status(run_ctx, "SUCCESS", f"Policy Denied: {e}")
            return

        # 5. Artifact Generation
        # (Mocking artifact creation)
        artifact_path = Path(f"report_{run_ctx.run_id}.json")
        with open(artifact_path, 'w') as f:
            json.dump({"run_id": run_ctx.run_id, "privacy_pass": True, "dp_cost": dp_cost}, f)

        # 6. Publish
        try:
            adapter.publish_artifact(run_ctx, str(artifact_path), "report")
        except Exception as e:
            # Do not delete artifacts on publish fail!
            logger.error(f"Publish failed: {e}")
            raise PublishError(f"Failed to upload artifacts: {e}")
            
        adapter.update_status(run_ctx, "SUCCESS", "Job completed successfully")
        logger.info("--- Run Completed ---")
        
    except PipelineError as e:
        logger.error(f"Pipeline Error: {e}")
        adapter.update_status(run_ctx, "FAILED", str(e))
        # Keep artifacts local for debugging
    except Exception as e:
        logger.error(f"Unexpected Runtime Error: {e}")
        traceback.print_exc()
        adapter.update_status(run_ctx, "FAILED", f"Runtime Error: {e}")

if __name__ == "__main__":
    # Minimal script entry for testing
    logging.basicConfig(level=logging.INFO)
    
    # 1. Config
    ctx = RunContext(
        run_id=str(uuid.uuid4())[:8],
        robot_id="robot_1",
        job_type="eval",
        config={"input_path": sys.argv[1] if len(sys.argv) > 1 else "data.mcap"}
    )
    
    # 2. Adapter
    fs_adapter = FilesystemAdapter("./runs")
    
    # 3. Governance
    gov = GovernanceEngine({"dp_budget_limit": 10.0})
    
    # 4. Run
    run_pipeline(fs_adapter, ctx, gov)
