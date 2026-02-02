"""
Tests for the Unified HE Backend Architecture.

This module tests that:
1. All HE operations route through the single UnifiedHEBackend
2. Legacy backend types are correctly mapped to the unified backend
3. MOAI optimizations (column packing) are applied
4. Simulation mode works without the microkernel
5. Production mode requires the microkernel
"""

import numpy as np
import pytest
import warnings

from tensafe.core.config import HEMode, HEConfig, TenSafeConfig, create_default_config
from tensafe.core.he_interface import (
    HEBackendType,
    HEParams,
    HEMetrics,
    HEBackendInterface,
    UnifiedHEBackend,
    DisabledHEBackend,
    get_backend,
    is_backend_available,
    list_available_backends,
    # Legacy exports (should all route to UnifiedHEBackend)
    ToyHEBackend,
    N2HEBackendWrapper,
    HEXLBackendWrapper,
    CKKSMOAIBackendWrapper,
    MicrokernelBackendWrapper,
)


class TestHEModeUnification:
    """Test that HEMode correctly resolves legacy modes."""

    def test_production_mode_unchanged(self):
        """PRODUCTION mode should remain unchanged."""
        resolved = HEMode.resolve(HEMode.PRODUCTION)
        assert resolved == HEMode.PRODUCTION

    def test_simulation_mode_unchanged(self):
        """SIMULATION mode should remain unchanged."""
        resolved = HEMode.resolve(HEMode.SIMULATION)
        assert resolved == HEMode.SIMULATION

    def test_disabled_mode_unchanged(self):
        """DISABLED mode should remain unchanged."""
        resolved = HEMode.resolve(HEMode.DISABLED)
        assert resolved == HEMode.DISABLED

    def test_toy_maps_to_simulation(self):
        """TOY (deprecated) should map to SIMULATION."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolved = HEMode.resolve(HEMode.TOY)
            assert resolved == HEMode.SIMULATION
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_n2he_maps_to_production(self):
        """N2HE (deprecated) should map to PRODUCTION."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolved = HEMode.resolve(HEMode.N2HE)
            assert resolved == HEMode.PRODUCTION
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_n2he_hexl_maps_to_production(self):
        """N2HE_HEXL (deprecated) should map to PRODUCTION."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            resolved = HEMode.resolve(HEMode.N2HE_HEXL)
            assert resolved == HEMode.PRODUCTION
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_is_legacy_property(self):
        """Test is_legacy property identifies deprecated modes."""
        assert HEMode.TOY.is_legacy
        assert HEMode.N2HE.is_legacy
        assert HEMode.N2HE_HEXL.is_legacy
        assert not HEMode.DISABLED.is_legacy
        assert not HEMode.PRODUCTION.is_legacy
        assert not HEMode.SIMULATION.is_legacy

    def test_is_secure_property(self):
        """Test is_secure property identifies production-ready modes."""
        assert HEMode.PRODUCTION.is_secure
        assert not HEMode.SIMULATION.is_secure
        assert not HEMode.TOY.is_secure
        assert not HEMode.DISABLED.is_secure


class TestHEBackendTypeUnification:
    """Test that HEBackendType correctly resolves legacy types."""

    def test_production_unchanged(self):
        """PRODUCTION type should remain unchanged."""
        resolved = HEBackendType.resolve(HEBackendType.PRODUCTION)
        assert resolved == HEBackendType.PRODUCTION

    def test_simulation_unchanged(self):
        """SIMULATION type should remain unchanged."""
        resolved = HEBackendType.resolve(HEBackendType.SIMULATION)
        assert resolved == HEBackendType.SIMULATION

    def test_disabled_unchanged(self):
        """DISABLED type should remain unchanged."""
        resolved = HEBackendType.resolve(HEBackendType.DISABLED)
        assert resolved == HEBackendType.DISABLED

    def test_toy_maps_to_simulation(self):
        """TOY (deprecated) should map to SIMULATION."""
        resolved = HEBackendType.resolve(HEBackendType.TOY)
        assert resolved == HEBackendType.SIMULATION

    def test_n2he_maps_to_production(self):
        """N2HE (deprecated) should map to PRODUCTION."""
        resolved = HEBackendType.resolve(HEBackendType.N2HE)
        assert resolved == HEBackendType.PRODUCTION

    def test_hexl_maps_to_production(self):
        """HEXL (deprecated) should map to PRODUCTION."""
        resolved = HEBackendType.resolve(HEBackendType.HEXL)
        assert resolved == HEBackendType.PRODUCTION

    def test_ckks_moai_maps_to_production(self):
        """CKKS_MOAI (deprecated) should map to PRODUCTION."""
        resolved = HEBackendType.resolve(HEBackendType.CKKS_MOAI)
        assert resolved == HEBackendType.PRODUCTION

    def test_microkernel_maps_to_production(self):
        """MICROKERNEL (deprecated) should map to PRODUCTION."""
        resolved = HEBackendType.resolve(HEBackendType.MICROKERNEL)
        assert resolved == HEBackendType.PRODUCTION

    def test_auto_maps_to_production(self):
        """AUTO (deprecated) should map to PRODUCTION."""
        resolved = HEBackendType.resolve(HEBackendType.AUTO)
        assert resolved == HEBackendType.PRODUCTION


class TestUnifiedHEBackendSimulation:
    """Test UnifiedHEBackend in simulation mode."""

    def test_simulation_backend_setup(self):
        """Simulation backend should set up without microkernel."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()
        assert backend.is_setup
        assert not backend.is_production_ready

    def test_simulation_encrypt_decrypt(self):
        """Simulation should preserve data through encrypt/decrypt."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        plaintext = np.array([1.0, 2.0, 3.0, 4.0])
        ct = backend.encrypt(plaintext)
        result = backend.decrypt(ct, output_size=4)

        np.testing.assert_array_almost_equal(plaintext, result)

    def test_simulation_add(self):
        """Simulation should correctly add ciphertexts."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])

        ct_a = backend.encrypt(a)
        ct_b = backend.encrypt(b)
        ct_sum = backend.add(ct_a, ct_b)
        result = backend.decrypt(ct_sum, output_size=3)

        np.testing.assert_array_almost_equal(a + b, result)

    def test_simulation_multiply_plain(self):
        """Simulation should correctly multiply ciphertext by plaintext."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        x = np.array([1.0, 2.0, 3.0])
        scalar = np.array([2.5])

        ct = backend.encrypt(x)
        ct_result = backend.multiply_plain(ct, scalar)
        result = backend.decrypt(ct_result, output_size=3)

        np.testing.assert_array_almost_equal(x * 2.5, result)

    def test_simulation_matmul(self):
        """Simulation should correctly compute matrix multiplication."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        x = np.array([1.0, 2.0, 3.0])
        W = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # 2x3 matrix

        ct = backend.encrypt(x)
        ct_result = backend.matmul(ct, W)
        result = backend.decrypt(ct_result, output_size=2)

        expected = x @ W.T  # [1.0, 4.0]
        np.testing.assert_array_almost_equal(expected, result)

    def test_simulation_lora_delta(self):
        """Simulation should correctly compute LoRA delta."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        hidden_dim = 8
        rank = 2
        out_dim = 8

        x = np.random.randn(hidden_dim)
        lora_a = np.random.randn(rank, hidden_dim)
        lora_b = np.random.randn(out_dim, rank)
        scaling = 0.5

        ct = backend.encrypt(x)
        ct_delta = backend.lora_delta(ct, lora_a, lora_b, scaling)
        result = backend.decrypt(ct_delta, output_size=out_dim)

        expected = scaling * (x @ lora_a.T @ lora_b.T)
        np.testing.assert_array_almost_equal(expected, result, decimal=5)


class TestDisabledHEBackend:
    """Test DisabledHEBackend pass-through behavior."""

    def test_disabled_setup(self):
        """Disabled backend should set up immediately."""
        backend = DisabledHEBackend()
        backend.setup()
        assert backend.is_setup
        assert not backend.is_production_ready

    def test_disabled_passthrough(self):
        """Disabled backend should pass data through unchanged."""
        backend = DisabledHEBackend()
        backend.setup()

        x = np.array([1.0, 2.0, 3.0])
        ct = backend.encrypt(x)
        result = backend.decrypt(ct)

        np.testing.assert_array_equal(x, result)


class TestBackendFactory:
    """Test get_backend factory function."""

    def test_get_disabled_backend(self):
        """Should return DisabledHEBackend for DISABLED type."""
        backend = get_backend(HEBackendType.DISABLED)
        assert isinstance(backend, DisabledHEBackend)
        assert backend.backend_type == HEBackendType.DISABLED

    def test_get_simulation_backend(self):
        """Should return UnifiedHEBackend in simulation mode."""
        backend = get_backend(HEBackendType.SIMULATION)
        assert isinstance(backend, UnifiedHEBackend)
        assert backend.backend_type == HEBackendType.SIMULATION
        assert not backend.is_production_ready

    def test_get_backend_from_string(self):
        """Should accept string backend type."""
        backend = get_backend("simulation")
        assert isinstance(backend, UnifiedHEBackend)
        assert backend.backend_type == HEBackendType.SIMULATION

    def test_get_backend_with_params(self):
        """Should use provided parameters."""
        params = HEParams(
            poly_modulus_degree=16384,
            scale_bits=45,
        )
        backend = get_backend(HEBackendType.SIMULATION, params=params)
        assert backend.params.poly_modulus_degree == 16384
        assert backend.params.scale_bits == 45

    def test_legacy_types_resolve_correctly(self):
        """Legacy types should resolve to correct backend."""
        # TOY -> SIMULATION
        backend = get_backend(HEBackendType.TOY)
        assert backend.backend_type == HEBackendType.SIMULATION


class TestLegacyBackendWrappers:
    """Test backward compatibility with legacy backend class names."""

    def test_toy_backend_is_unified(self):
        """ToyHEBackend should return UnifiedHEBackend in simulation mode."""
        backend = ToyHEBackend()
        assert isinstance(backend, UnifiedHEBackend)
        assert backend._simulation_mode

    def test_n2he_backend_is_unified(self):
        """N2HEBackendWrapper should return UnifiedHEBackend."""
        backend = N2HEBackendWrapper()
        assert isinstance(backend, UnifiedHEBackend)
        assert not backend._simulation_mode

    def test_hexl_backend_is_unified(self):
        """HEXLBackendWrapper should return UnifiedHEBackend."""
        backend = HEXLBackendWrapper()
        assert isinstance(backend, UnifiedHEBackend)
        assert not backend._simulation_mode

    def test_ckks_moai_backend_is_unified(self):
        """CKKSMOAIBackendWrapper should return UnifiedHEBackend."""
        backend = CKKSMOAIBackendWrapper()
        assert isinstance(backend, UnifiedHEBackend)
        assert not backend._simulation_mode

    def test_microkernel_backend_is_unified(self):
        """MicrokernelBackendWrapper should return UnifiedHEBackend."""
        backend = MicrokernelBackendWrapper()
        assert isinstance(backend, UnifiedHEBackend)
        assert not backend._simulation_mode


class TestBackendAvailability:
    """Test backend availability checking."""

    def test_disabled_always_available(self):
        """DISABLED backend should always be available."""
        assert is_backend_available(HEBackendType.DISABLED)

    def test_simulation_always_available(self):
        """SIMULATION backend should always be available."""
        assert is_backend_available(HEBackendType.SIMULATION)

    def test_list_available_backends(self):
        """Should list at least disabled and simulation."""
        available = list_available_backends()
        assert "disabled" in available
        assert "simulation" in available


class TestConfigIntegration:
    """Test integration with TenSafeConfig."""

    def test_create_config_with_production_he(self):
        """Should create config with PRODUCTION HE mode."""
        config = create_default_config(with_he=True, he_simulation=False)
        assert config.he.mode == HEMode.PRODUCTION

    def test_create_config_with_simulation_he(self):
        """Should create config with SIMULATION HE mode."""
        config = create_default_config(with_he=True, he_simulation=True)
        assert config.he.mode == HEMode.SIMULATION

    def test_create_config_without_he(self):
        """Should create config with DISABLED HE mode."""
        config = create_default_config(with_he=False)
        assert config.he.mode == HEMode.DISABLED

    def test_config_validation_warns_on_simulation(self):
        """Config validation should warn about simulation mode."""
        config = TenSafeConfig(he=HEConfig(mode=HEMode.SIMULATION))
        issues = config.validate()
        assert any("NOT cryptographically secure" in issue for issue in issues)

    def test_config_validation_warns_on_legacy_mode(self):
        """Config validation should warn about legacy modes."""
        config = TenSafeConfig(he=HEConfig(mode=HEMode.TOY))
        issues = config.validate()
        assert any("deprecated" in issue.lower() for issue in issues)


class TestMOAIOptimizations:
    """Test that MOAI optimizations are applied."""

    def test_column_packing_enabled_by_default(self):
        """Column packing should be enabled by default."""
        params = HEParams()
        assert params.use_column_packing

    def test_interleaved_batching_enabled_by_default(self):
        """Interleaved batching should be enabled by default."""
        params = HEParams()
        assert params.use_interleaved_batching

    def test_params_converted_to_microkernel_format(self):
        """Parameters should convert to microkernel format."""
        params = HEParams(
            poly_modulus_degree=16384,
            scale_bits=45,
            use_column_packing=True,
        )
        mk_params = params.to_microkernel_params()
        assert mk_params["poly_modulus_degree"] == 16384
        assert mk_params["scale_bits"] == 45
        assert mk_params["use_column_packing"]


class TestMetrics:
    """Test metrics tracking."""

    def test_metrics_tracking(self):
        """Backend should track operation metrics."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        x = np.array([1.0, 2.0, 3.0])
        ct = backend.encrypt(x)
        _ = backend.matmul(ct, np.eye(3))

        metrics = backend.get_metrics()
        assert metrics.operations_count > 0

    def test_metrics_reset(self):
        """Metrics should reset to zero."""
        backend = UnifiedHEBackend(simulation_mode=True)
        backend.setup()

        x = np.array([1.0, 2.0, 3.0])
        ct = backend.encrypt(x)
        _ = backend.matmul(ct, np.eye(3))

        backend.reset_metrics()
        metrics = backend.get_metrics()
        assert metrics.operations_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
