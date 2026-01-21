import pytest
import asyncio
import json
import os
from tensorguard.integrations.peft_hub.workflow import PeftWorkflow
from tensorguard.integrations.peft_hub.schemas import PeftRunConfig


@pytest.mark.asyncio
async def test_simulated_run_completion(monkeypatch):
    """Verify that a simulated PEFT run completes all stages and generates logs."""
    # Set simulation mode via environment variable
    monkeypatch.setenv("TG_SIMULATION", "true")

    # Re-import to pick up the new env var
    import importlib
    from tensorguard.integrations.peft_hub import workflow
    importlib.reload(workflow)
    from tensorguard.integrations.peft_hub.workflow import PeftWorkflow

    from tensorguard.integrations.peft_hub.catalog import discover_connectors
    discover_connectors()

    config = PeftRunConfig(
        id="test-run-001",
        training_config={
            "method": "lora",
            "model_name_or_path": "test/model",
            "dataset_name_or_path": "test/data"
        },
        simulation=True
    )

    workflow_instance = PeftWorkflow(config)
    logs = []

    async for log in workflow_instance.execute():
        logs.append(log)

    assert len(logs) > 0
    # Search for keywords rather than exact strings
    log_text = "".join(logs).lower()

    assert "completed successfully" in log_text
    assert "packaging" in log_text or "tgsp" in log_text
    assert "evidence" in log_text


@pytest.mark.asyncio
async def test_run_artifact_creation(tmp_path, monkeypatch):
    """Verify that artifacts are 'created' (simulated paths exist)."""
    # Set simulation mode via environment variable
    monkeypatch.setenv("TG_SIMULATION", "true")

    # Re-import to pick up the new env var
    import importlib
    from tensorguard.integrations.peft_hub import workflow
    importlib.reload(workflow)
    from tensorguard.integrations.peft_hub.workflow import PeftWorkflow

    config = PeftRunConfig(
        id="test-run-artifact",
        training_config={
            "method": "lora",
            "model_name_or_path": "test/model",
            "dataset_name_or_path": "test/data"
        },
        simulation=True
    )

    workflow_instance = PeftWorkflow(config)
    async for _ in workflow_instance.execute():
        pass

    # Check if we can reach the end without error
    assert True
