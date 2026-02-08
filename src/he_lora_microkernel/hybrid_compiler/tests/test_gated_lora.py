"""
Gated LoRA Functional Correctness Tests

Tests for the gated LoRA compiler and executor including:
1. Compilation correctness
2. Execution vs plaintext reference
3. Gate behavior verification
4. Scheme transition correctness
"""

import numpy as np

from ..gated_lora import (
    GatedLoRACompiler,
    GatedLoRAConfig,
    GatedLoRAExecutor,
    compile_gated_lora,
    plaintext_gated_lora,
)
from ..ir import validate_program
from ..tfhe_lut import LUTLibrary, sign_lut


class TestGatedLoRAConfig:
    """Tests for GatedLoRAConfig validation."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = GatedLoRAConfig(
            hidden_size=256,
            lora_rank=16,
            gate_type="step",
        )
        errors = config.validate()
        assert len(errors) == 0

    def test_invalid_hidden_size(self):
        """Test invalid hidden_size is caught."""
        config = GatedLoRAConfig(
            hidden_size=0,
            lora_rank=16,
        )
        errors = config.validate()
        assert any("hidden_size" in e for e in errors)

    def test_invalid_lora_rank(self):
        """Test invalid lora_rank is caught."""
        config = GatedLoRAConfig(
            hidden_size=256,
            lora_rank=0,
        )
        errors = config.validate()
        assert any("lora_rank" in e for e in errors)

    def test_lora_rank_exceeds_hidden(self):
        """Test lora_rank > hidden_size is caught."""
        config = GatedLoRAConfig(
            hidden_size=64,
            lora_rank=128,
        )
        errors = config.validate()
        assert any("lora_rank" in e for e in errors)

    def test_invalid_quantization_bits(self):
        """Test invalid quantization_bits is caught."""
        config = GatedLoRAConfig(
            hidden_size=256,
            lora_rank=16,
            quantization_bits=2,  # Too low
        )
        errors = config.validate()
        assert any("quantization" in e.lower() for e in errors)


class TestGatedLoRACompiler:
    """Tests for GatedLoRACompiler."""

    def test_basic_compilation(self):
        """Test basic compilation succeeds."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
            gate_type="step",
        )
        compiler = GatedLoRACompiler(config)
        program, plan = compiler.compile()

        assert program is not None
        assert plan is not None
        assert program.name == "gated_lora"

    def test_ir_has_required_phases(self):
        """Test compiled IR has all required computational phases."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        # Check for presence of key operations by node type
        node_types = [type(n).__name__ for n in program.nodes]

        # Should have CKKS operations for LoRA
        assert any("CKKSMatMul" in nt for nt in node_types)

        # Should have quantization for bridge
        assert any("Quantize" in nt for nt in node_types)

        # Should have TFHE LUT for gate
        assert any("TFHELUT" in nt for nt in node_types)

        # Should have bridge operations
        assert any("ToTFHE" in nt for nt in node_types)
        assert any("ToCKKS" in nt for nt in node_types)

    def test_ir_validates(self):
        """Test compiled IR passes validation."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        result = validate_program(program)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_moai_packing_enabled(self):
        """Test MOAI packing is present when enabled."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
            use_moai_packing=True,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        node_types = [type(n).__name__ for n in program.nodes]
        assert any("PackMOAI" in nt for nt in node_types)

    def test_moai_packing_disabled(self):
        """Test MOAI packing is absent when disabled."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
            use_moai_packing=False,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        node_types = [type(n).__name__ for n in program.nodes]
        assert not any("PackMOAI" in nt for nt in node_types)

    def test_convenience_function(self):
        """Test compile_gated_lora convenience function."""
        program, plan = compile_gated_lora(
            hidden_size=128,
            lora_rank=8,
            gate_type="step",
        )
        assert program is not None
        assert plan is not None


class TestPlaintextReference:
    """Tests for plaintext reference implementation."""

    def test_plaintext_gated_lora_gate_on(self):
        """Test plaintext gated LoRA when gate is ON (z >= 0)."""
        hidden_size = 64
        lora_rank = 8

        # Random weights and input
        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.random.randn(hidden_size).astype(np.float32)
        lora_A = np.random.randn(lora_rank, hidden_size).astype(np.float32) * 0.1
        lora_B = np.random.randn(hidden_size, lora_rank).astype(np.float32) * 0.1
        w_gate = np.random.randn(hidden_size).astype(np.float32) * 0.1

        # Ensure gate is ON by adding positive bias
        b_gate = 5.0  # Large positive bias

        result = plaintext_gated_lora(
            x=x,
            base_output=base_output,
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=b_gate,
        )

        # Compute expected: y = base_output + 1.0 * (B @ (A @ x))
        delta = lora_B @ (lora_A @ x)
        expected = base_output + delta

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_plaintext_gated_lora_gate_off(self):
        """Test plaintext gated LoRA when gate is OFF (z < 0)."""
        hidden_size = 64
        lora_rank = 8

        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.random.randn(hidden_size).astype(np.float32)
        lora_A = np.random.randn(lora_rank, hidden_size).astype(np.float32) * 0.1
        lora_B = np.random.randn(hidden_size, lora_rank).astype(np.float32) * 0.1
        w_gate = np.random.randn(hidden_size).astype(np.float32) * 0.1

        # Ensure gate is OFF by adding negative bias
        b_gate = -100.0  # Large negative bias

        result = plaintext_gated_lora(
            x=x,
            base_output=base_output,
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=b_gate,
        )

        # Gate is OFF, so y = base_output + 0 * delta = base_output
        expected = base_output

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_plaintext_returns_gate_value(self):
        """Test plaintext reference returns gate value."""
        hidden_size = 32
        lora_rank = 4

        np.random.seed(123)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.zeros(hidden_size, dtype=np.float32)
        lora_A = np.eye(lora_rank, hidden_size, dtype=np.float32)
        lora_B = np.eye(hidden_size, lora_rank, dtype=np.float32)
        w_gate = np.zeros(hidden_size, dtype=np.float32)

        # Test with positive bias (gate ON)
        result, gate_z, gate_val = plaintext_gated_lora(
            x=x,
            base_output=base_output,
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=1.0,
            return_gate=True,
        )
        assert gate_val == 1.0
        assert gate_z > 0

        # Test with negative bias (gate OFF)
        result, gate_z, gate_val = plaintext_gated_lora(
            x=x,
            base_output=base_output,
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=-1.0,
            return_gate=True,
        )
        assert gate_val == 0.0
        assert gate_z < 0


class TestGatedLoRAExecutor:
    """Tests for GatedLoRAExecutor."""

    def test_executor_initialization(self):
        """Test executor can be initialized."""
        config = GatedLoRAConfig(
            hidden_size=64,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, plan = compiler.compile()

        executor = GatedLoRAExecutor(program, plan, config)
        assert executor is not None

    def test_simulated_execution(self):
        """Test simulated execution produces output."""
        hidden_size = 64
        lora_rank = 8

        config = GatedLoRAConfig(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
        )
        compiler = GatedLoRACompiler(config)
        program, plan = compiler.compile()
        executor = GatedLoRAExecutor(program, plan, config)

        # Create test inputs
        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.random.randn(hidden_size).astype(np.float32)

        # Create weights
        weights = {
            "lora_A": np.random.randn(lora_rank, hidden_size).astype(np.float32) * 0.1,
            "lora_B": np.random.randn(hidden_size, lora_rank).astype(np.float32) * 0.1,
            "w_gate": np.random.randn(hidden_size).astype(np.float32) * 0.1,
            "b_gate": np.array([1.0], dtype=np.float32),  # Gate ON
        }

        result = executor.execute_simulated(
            x=x,
            base_output=base_output,
            weights=weights,
        )

        assert result is not None
        assert result.output is not None
        assert result.output.shape == (hidden_size,)

    def test_execution_matches_reference(self):
        """Test simulated execution matches plaintext reference."""
        hidden_size = 64
        lora_rank = 8

        config = GatedLoRAConfig(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
            quantization_bits=8,
        )
        compiler = GatedLoRACompiler(config)
        program, plan = compiler.compile()
        executor = GatedLoRAExecutor(program, plan, config)

        # Create test inputs
        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.random.randn(hidden_size).astype(np.float32)

        lora_A = np.random.randn(lora_rank, hidden_size).astype(np.float32) * 0.1
        lora_B = np.random.randn(hidden_size, lora_rank).astype(np.float32) * 0.1
        w_gate = np.random.randn(hidden_size).astype(np.float32) * 0.1
        b_gate = 2.0  # Gate ON

        weights = {
            "lora_A": lora_A,
            "lora_B": lora_B,
            "w_gate": w_gate,
            "b_gate": np.array([b_gate], dtype=np.float32),
        }

        # Execute simulated
        sim_result = executor.execute_simulated(
            x=x,
            base_output=base_output,
            weights=weights,
        )

        # Compute plaintext reference
        ref_output = plaintext_gated_lora(
            x=x,
            base_output=base_output,
            lora_A=lora_A,
            lora_B=lora_B,
            w_gate=w_gate,
            b_gate=b_gate,
        )

        # Should match closely (some error from quantization simulation)
        np.testing.assert_allclose(
            sim_result.output,
            ref_output,
            rtol=0.1,  # 10% tolerance for simulation
            atol=0.01,
        )


class TestSchemeTransitions:
    """Tests for correct scheme transitions."""

    def test_no_invalid_scheme_mixing(self):
        """Test that compiled IR has no invalid scheme mixing."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        # Validate catches scheme mixing errors
        result = validate_program(program)
        scheme_errors = [e for e in result.errors if "scheme" in str(e).lower()]
        assert len(scheme_errors) == 0, f"Scheme mixing errors: {scheme_errors}"

    def test_bridge_ops_present(self):
        """Test that bridge ops are present for scheme transitions."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        # Count bridge operations
        bridge_to_tfhe = sum(
            1 for n in program.nodes if "ToTFHE" in type(n).__name__
        )
        bridge_to_ckks = sum(
            1 for n in program.nodes if "TFHEToCKKS" in type(n).__name__
        )

        # Should have at least one of each for gated LoRA
        assert bridge_to_tfhe >= 1, "Missing CKKSToTFHE bridge"
        assert bridge_to_ckks >= 1, "Missing TFHEToCKKS bridge"

    def test_bootstrap_count(self):
        """Test bootstrap count is within budget."""
        config = GatedLoRAConfig(
            hidden_size=128,
            lora_rank=8,
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        # Count TFHE LUT operations (each implies a bootstrap)
        bootstrap_count = sum(
            1 for n in program.nodes if "TFHELUT" in type(n).__name__
        )

        # Gated LoRA should have exactly 1 bootstrap for the gate
        assert bootstrap_count == 1, f"Expected 1 bootstrap, got {bootstrap_count}"


class TestGateTypes:
    """Tests for different gate types."""

    def test_step_gate(self):
        """Test step gate compilation."""
        config = GatedLoRAConfig(
            hidden_size=64,
            lora_rank=8,
            gate_type="step",
        )
        compiler = GatedLoRACompiler(config)
        program, _ = compiler.compile()

        # Find the LUT node
        lut_nodes = [n for n in program.nodes if "TFHELUT" in type(n).__name__]
        assert len(lut_nodes) == 1
        assert lut_nodes[0].lut_name == "step"

    def test_sign_gate(self):
        """Test sign gate compilation."""
        # Create library with sign LUT
        library = LUTLibrary(bits=8)
        library.register(sign_lut(8))

        config = GatedLoRAConfig(
            hidden_size=64,
            lora_rank=8,
            gate_type="sign",
        )
        compiler = GatedLoRACompiler(config, lut_library=library)
        program, _ = compiler.compile()

        lut_nodes = [n for n in program.nodes if "TFHELUT" in type(n).__name__]
        assert len(lut_nodes) == 1
        assert lut_nodes[0].lut_name == "sign"


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end gated LoRA."""

    def test_full_pipeline_gate_on(self):
        """Test full pipeline with gate ON."""
        hidden_size = 32
        lora_rank = 4

        # Compile
        config = GatedLoRAConfig(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
        )
        program, plan = compile_gated_lora(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
        )

        # Validate
        result = validate_program(program)
        assert result.valid

        # Execute (simulated)
        executor = GatedLoRAExecutor(program, plan, config)

        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.zeros(hidden_size, dtype=np.float32)

        weights = {
            "lora_A": np.eye(lora_rank, hidden_size, dtype=np.float32),
            "lora_B": np.eye(hidden_size, lora_rank, dtype=np.float32),
            "w_gate": np.zeros(hidden_size, dtype=np.float32),
            "b_gate": np.array([5.0], dtype=np.float32),  # Gate ON
        }

        result = executor.execute_simulated(
            x=x,
            base_output=base_output,
            weights=weights,
        )

        # With identity matrices and gate ON, output ≈ x[:lora_rank]
        # (padded with zeros for remaining dimensions)
        assert result.gate_value == 1.0 or result.gate_value is None

    def test_full_pipeline_gate_off(self):
        """Test full pipeline with gate OFF."""
        hidden_size = 32
        lora_rank = 4

        config = GatedLoRAConfig(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
        )
        program, plan = compile_gated_lora(
            hidden_size=hidden_size,
            lora_rank=lora_rank,
        )

        executor = GatedLoRAExecutor(program, plan, config)

        np.random.seed(42)
        x = np.random.randn(hidden_size).astype(np.float32)
        base_output = np.random.randn(hidden_size).astype(np.float32)

        weights = {
            "lora_A": np.random.randn(lora_rank, hidden_size).astype(np.float32),
            "lora_B": np.random.randn(hidden_size, lora_rank).astype(np.float32),
            "w_gate": np.zeros(hidden_size, dtype=np.float32),
            "b_gate": np.array([-10.0], dtype=np.float32),  # Gate OFF
        }

        result = executor.execute_simulated(
            x=x,
            base_output=base_output,
            weights=weights,
        )

        # Gate is OFF, output should equal base_output
        np.testing.assert_allclose(
            result.output,
            base_output,
            rtol=0.1,
            atol=0.01,
        )


# =============================================================================
# Test Runner
# =============================================================================

def run_gated_lora_tests():
    """Run all gated LoRA tests and report results."""
    test_classes = [
        TestGatedLoRAConfig,
        TestGatedLoRACompiler,
        TestPlaintextReference,
        TestGatedLoRAExecutor,
        TestSchemeTransitions,
        TestGateTypes,
        TestIntegration,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(instance, method_name)
                    method()
                    passed += 1
                    print(f"  ✓ {test_class.__name__}.{method_name}")
                except AssertionError as e:
                    failed += 1
                    errors.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"  ✗ {test_class.__name__}.{method_name}: {e}")
                except Exception as e:
                    failed += 1
                    errors.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"  ✗ {test_class.__name__}.{method_name}: ERROR - {e}")

    print(f"\nGated LoRA Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_gated_lora_tests()
    exit(0 if success else 1)
