import os
import pytest

from tensorguard.utils.production_gates import ProductionGateError, is_production

# Check if tenseal is available (required for serving.backend imports)
try:
    import tenseal
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False


def test_tpm_simulator_blocked_in_production(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "production")
    monkeypatch.delenv("TG_ALLOW_TPM_SIMULATOR", raising=False)
    is_production.cache_clear()

    from tensorguard.agent.identity.tpm_simulator import TPMSimulator

    with pytest.raises(ProductionGateError):
        TPMSimulator()


def test_tpm_simulator_override_marks_untrusted(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "production")
    monkeypatch.setenv("TG_ALLOW_TPM_SIMULATOR", "true")
    is_production.cache_clear()

    from tensorguard.agent.identity.tpm_simulator import TPMSimulator

    simulator = TPMSimulator()
    quote = simulator.get_quote("nonce")

    assert quote["message"]["attested"] is False
    assert quote["message"]["simulator"] is True


def test_key_provider_factory_blocks_unsupported_in_production(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "production")
    is_production.cache_clear()

    from tensorguard.identity.keys.provider import KeyProviderFactory

    with pytest.raises(ProductionGateError):
        KeyProviderFactory.create("pkcs11")


@pytest.mark.skipif(not TENSEAL_AVAILABLE, reason="tenseal not installed")
def test_native_backend_blocked_in_production(monkeypatch):
    monkeypatch.setenv("TG_ENVIRONMENT", "production")
    is_production.cache_clear()

    from tensorguard.serving.backend import NativeBackend

    with pytest.raises(ProductionGateError):
        NativeBackend()
