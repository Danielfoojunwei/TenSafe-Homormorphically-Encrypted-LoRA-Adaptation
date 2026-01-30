"""
Regression Test Configuration

Provides fixtures for regression testing TenSafe privacy invariants.
Minimal setup for privacy component testing without full platform dependencies.
"""

import pytest
import os
import sys
import logging
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock


# --- Async Mock Helpers ---

class AsyncIteratorMock:
    """Helper class to create async iterators for mocking async generators."""

    def __init__(self, items=None):
        self.items = items or []
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def async_iter_mock(items=None):
    """Create a mock that returns an async iterator."""
    return AsyncIteratorMock(items or [])


def create_mock_workflow():
    """Create a properly configured mock PeftWorkflow for tests."""
    mock = MagicMock()
    mock.artifacts = {"adapter_path": "/mock/adapter", "tgsp_path": "/mock/tgsp"}
    mock.metrics = {"eval": {"accuracy": 0.95, "forgetting": 0.01, "regression": 0.01}}
    mock.diagnosis = None

    # Use async iterators for stage methods
    mock._stage_train.return_value = async_iter_mock([])
    mock._stage_eval.return_value = async_iter_mock([])
    mock._stage_pack_tgsp.return_value = async_iter_mock([])
    mock._stage_emit_evidence.return_value = async_iter_mock([])

    return mock


# Configure logging for regression tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("regression")

# Set deterministic environment
os.environ["TG_DETERMINISTIC"] = "true"
os.environ["TG_DEMO_MODE"] = "true"
os.environ["TG_SIMULATION"] = "true"
os.environ["TG_ENABLE_LABS"] = "false"
os.environ["TG_ENVIRONMENT"] = "development"
os.environ["TS_DETERMINISTIC"] = "true"
os.environ["TS_DEMO_MODE"] = "true"
os.environ["TS_ENVIRONMENT"] = "development"

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


# --- Fixtures ---

@pytest.fixture(name="temp_dir")
def temp_dir_fixture() -> Generator[Path, None, None]:
    """Provides a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(name="mock_workflow")
def mock_workflow_fixture():
    """Provides a mock workflow for testing."""
    return create_mock_workflow()


# --- Test Data Fixtures ---

@pytest.fixture(scope="session", autouse=True)
def create_test_fixtures():
    """Creates test fixture files if they don't exist."""
    fixtures_dir = os.path.join(os.path.dirname(__file__), "..", "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    sample_data_path = os.path.join(fixtures_dir, "sample_data.jsonl")
    if not os.path.exists(sample_data_path):
        with open(sample_data_path, "w") as f:
            f.write('{"input": "test input 1", "output": "test output 1"}\n')
            f.write('{"input": "test input 2", "output": "test output 2"}\n')
            f.write('{"input": "test input 3", "output": "test output 3"}\n')

    yield
