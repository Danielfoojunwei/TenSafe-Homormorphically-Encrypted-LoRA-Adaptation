import pytest
import os
from tensorguard.integrations.peft_hub.catalog import ConnectorCatalog


def test_catalog_discovery():
    """Verify that the catalog can discover built-in connectors."""
    from tensorguard.integrations.peft_hub.catalog import discover_connectors
    discover_connectors()

    catalog = ConnectorCatalog()
    connectors = catalog.list_connectors()

    assert len(connectors) > 0

    # Check for specific built-in connectors
    types = [c["id"] for c in connectors]
    assert "training_hf" in types
    assert "data_local" in types
    assert "store_local" in types


def test_connector_check_simulated():
    """Verify that connectors can be retrieved (check_installed depends on deps)."""
    from tensorguard.integrations.peft_hub.catalog import discover_connectors
    discover_connectors()

    catalog = ConnectorCatalog()

    # HF Training connector should be retrievable (but check_installed may return False if deps missing)
    connector = catalog.get_connector("training_hf")
    assert connector is not None

    # Check the method exists and returns a boolean
    result = connector.check_installed()
    assert isinstance(result, bool)
