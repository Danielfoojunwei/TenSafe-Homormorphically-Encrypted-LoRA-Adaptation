"""
IR Validation Tests

Tests for the scheme-aware IR system including:
1. Type system correctness
2. Validation rule enforcement
3. Scheme mixing detection
4. Bootstrap budget enforcement
"""

import pytest
import numpy as np
from typing import List

from ..ir import (
    IRProgram, IRValue, Shape, Scheme, ValueType,
    CKKSMetadata, TFHEMetadata,
    create_ckks_value, create_tfhe_value,
    CKKSMatMul, CKKSAdd, CKKSMul, CKKSRescale, CKKSPackMOAI, CKKSApplyMask,
    TFHELUT, CKKSQuantizeToInt, CKKSToTFHE, TFHEToCKKS,
    validate_program, ValidationResult,
)


class TestShape:
    """Tests for Shape class."""

    def test_scalar_shape(self):
        """Test scalar shape creation."""
        shape = Shape.scalar()
        assert shape.dims == (1,)
        assert shape.is_scalar
        assert shape.size == 1
        assert shape.is_tfhe_compatible

    def test_vector_shape(self):
        """Test vector shape creation."""
        shape = Shape.vector(128)
        assert shape.dims == (128,)
        assert not shape.is_scalar
        assert shape.size == 128
        assert not shape.is_tfhe_compatible  # > MAX_TFHE_ELEMENTS

    def test_small_vector_tfhe_compatible(self):
        """Test small vectors are TFHE compatible."""
        shape = Shape.vector(8)
        assert shape.is_tfhe_compatible
        assert shape.size == 8

    def test_matrix_shape(self):
        """Test matrix shape creation."""
        shape = Shape.matrix(64, 128)
        assert shape.dims == (64, 128)
        assert shape.size == 64 * 128

    def test_tfhe_max_elements_boundary(self):
        """Test TFHE compatibility at boundary."""
        # At boundary - should be compatible
        shape_at_limit = Shape.vector(Shape.MAX_TFHE_ELEMENTS)
        assert shape_at_limit.is_tfhe_compatible

        # Over boundary - should not be compatible
        shape_over_limit = Shape.vector(Shape.MAX_TFHE_ELEMENTS + 1)
        assert not shape_over_limit.is_tfhe_compatible


class TestValueType:
    """Tests for ValueType and scheme compatibility."""

    def test_ckks_value_types(self):
        """Test CKKS-compatible value types."""
        assert ValueType.REAL_APPROX.is_ckks_compatible
        assert not ValueType.REAL_APPROX.is_tfhe_compatible

    def test_tfhe_value_types(self):
        """Test TFHE-compatible value types."""
        for vt in [ValueType.BIT, ValueType.INT_4, ValueType.INT_8, ValueType.INT_16]:
            assert vt.is_tfhe_compatible
            assert not vt.is_ckks_compatible

    def test_bit_width(self):
        """Test bit width retrieval."""
        assert ValueType.BIT.bit_width == 1
        assert ValueType.INT_4.bit_width == 4
        assert ValueType.INT_8.bit_width == 8
        assert ValueType.INT_16.bit_width == 16


class TestIRValue:
    """Tests for IRValue creation and validation."""

    def test_create_ckks_value(self):
        """Test CKKS value creation."""
        val = create_ckks_value(
            name="x",
            shape=Shape.vector(512),
            precision_budget=40.0,
            scale_bits=40,
        )
        assert val.name == "x"
        assert val.scheme == Scheme.CKKS
        assert val.value_type == ValueType.REAL_APPROX
        assert val.shape.size == 512
        assert val.metadata.precision_budget == 40.0

    def test_create_tfhe_value(self):
        """Test TFHE value creation."""
        val = create_tfhe_value(
            name="g",
            shape=Shape.scalar(),
            value_type=ValueType.BIT,
            noise_budget=100,
        )
        assert val.name == "g"
        assert val.scheme == Scheme.TFHE
        assert val.value_type == ValueType.BIT
        assert val.shape.is_scalar
        assert val.metadata.noise_budget == 100

    def test_tfhe_value_large_shape_warning(self):
        """Test that large TFHE shapes are flagged."""
        val = create_tfhe_value(
            name="large",
            shape=Shape.vector(100),  # Over TFHE limit
            value_type=ValueType.INT_8,
        )
        # Value creation should succeed but validation should warn
        assert val.scheme == Scheme.TFHE
        assert not val.shape.is_tfhe_compatible


class TestIRProgram:
    """Tests for IRProgram structure."""

    def test_empty_program(self):
        """Test empty program creation."""
        prog = IRProgram(name="test")
        assert prog.name == "test"
        assert len(prog.inputs) == 0
        assert len(prog.nodes) == 0
        assert len(prog.outputs) == 0

    def test_add_input(self):
        """Test adding inputs to program."""
        prog = IRProgram(name="test")
        x = create_ckks_value("x", Shape.vector(64))
        prog.add_input(x)
        assert len(prog.inputs) == 1
        assert prog.inputs[0].name == "x"

    def test_add_node(self):
        """Test adding nodes to program."""
        prog = IRProgram(name="test")
        x = create_ckks_value("x", Shape.vector(64))
        prog.add_input(x)

        matmul = CKKSMatMul(
            node_id="matmul_1",
            input_name="x",
            weight_name="W",
            output_name="y",
            output_shape=Shape.vector(64),
            weight_shape=(64, 64),
        )
        prog.add_node(matmul)
        assert len(prog.nodes) == 1

    def test_get_value(self):
        """Test value lookup by name."""
        prog = IRProgram(name="test")
        x = create_ckks_value("x", Shape.vector(64))
        prog.add_input(x)

        # Should be able to find the value
        found = prog.get_value("x")
        assert found is not None
        assert found.name == "x"


class TestValidation:
    """Tests for IR validation rules."""

    def test_valid_ckks_only_program(self):
        """Test valid CKKS-only program passes validation."""
        prog = IRProgram(name="ckks_only")

        x = create_ckks_value("x", Shape.vector(64), precision_budget=40.0)
        prog.add_input(x)

        matmul = CKKSMatMul(
            node_id="matmul_1",
            input_name="x",
            weight_name="W",
            output_name="y",
            output_shape=Shape.vector(64),
            weight_shape=(64, 64),
        )
        prog.add_node(matmul)
        prog.add_output("y")

        result = validate_program(prog)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_missing_bridge_error(self):
        """Test that using CKKS value in TFHE op without bridge fails."""
        prog = IRProgram(name="invalid_mix")

        # CKKS input
        x = create_ckks_value("x", Shape.scalar())
        prog.add_input(x)

        # Try to use directly in TFHE LUT (invalid - needs bridge)
        lut = TFHELUT(
            node_id="lut_1",
            input_name="x",  # Still CKKS!
            output_name="g",
            lut_name="step",
        )
        prog.add_node(lut)
        prog.add_output("g")

        result = validate_program(prog)
        # Should fail because CKKS value used in TFHE op
        assert not result.valid
        assert any("scheme" in str(e).lower() for e in result.errors)

    def test_valid_bridge_usage(self):
        """Test valid bridge operation sequence."""
        prog = IRProgram(name="valid_bridge")

        # CKKS input
        z = create_ckks_value("z", Shape.scalar())
        prog.add_input(z)

        # Quantize (CKKS → discrete)
        quantize = CKKSQuantizeToInt(
            node_id="quant_1",
            input_name="z",
            output_name="z_q",
            bits=8,
        )
        prog.add_node(quantize)

        # Bridge to TFHE
        to_tfhe = CKKSToTFHE(
            node_id="bridge_1",
            input_name="z_q",
            output_name="z_tfhe",
            output_type=ValueType.INT_8,
        )
        prog.add_node(to_tfhe)

        # TFHE LUT
        lut = TFHELUT(
            node_id="lut_1",
            input_name="z_tfhe",
            output_name="g_tfhe",
            lut_name="step",
            output_type=ValueType.BIT,
        )
        prog.add_node(lut)

        # Bridge back to CKKS
        to_ckks = TFHEToCKKS(
            node_id="bridge_2",
            input_name="g_tfhe",
            output_name="g_ckks",
        )
        prog.add_node(to_ckks)

        prog.add_output("g_ckks")

        result = validate_program(prog)
        assert result.valid, f"Validation failed: {result.errors}"

    def test_bootstrap_budget_exceeded(self):
        """Test that exceeding bootstrap budget fails validation."""
        prog = IRProgram(name="too_many_bootstraps")

        # Create inputs
        for i in range(5):  # More than MAX_BOOTSTRAPS_PER_LAYER (2)
            z = create_ckks_value(f"z{i}", Shape.scalar())
            prog.add_input(z)

            # Each one goes through TFHE
            quantize = CKKSQuantizeToInt(
                node_id=f"quant_{i}",
                input_name=f"z{i}",
                output_name=f"z_q{i}",
                bits=8,
            )
            prog.add_node(quantize)

            to_tfhe = CKKSToTFHE(
                node_id=f"to_tfhe_{i}",
                input_name=f"z_q{i}",
                output_name=f"z_tfhe{i}",
            )
            prog.add_node(to_tfhe)

            lut = TFHELUT(
                node_id=f"lut_{i}",
                input_name=f"z_tfhe{i}",
                output_name=f"g{i}",
                lut_name="step",
            )
            prog.add_node(lut)

        prog.add_output("g0")

        result = validate_program(prog)
        # Should fail or warn due to bootstrap budget
        assert not result.valid or len(result.warnings) > 0

    def test_tfhe_large_vector_error(self):
        """Test that large TFHE vectors fail validation."""
        prog = IRProgram(name="large_tfhe")

        # Create TFHE value with large shape
        x_tfhe = create_tfhe_value(
            "x_tfhe",
            Shape.vector(1000),  # Way over limit
            ValueType.INT_8,
        )
        prog.add_input(x_tfhe)

        lut = TFHELUT(
            node_id="lut_1",
            input_name="x_tfhe",
            output_name="y_tfhe",
            lut_name="step",
        )
        prog.add_node(lut)
        prog.add_output("y_tfhe")

        result = validate_program(prog)
        # Should warn about TFHE on large vector
        assert len(result.warnings) > 0 or not result.valid


class TestCKKSNodes:
    """Tests for CKKS operation nodes."""

    def test_matmul_node(self):
        """Test CKKSMatMul node creation."""
        node = CKKSMatMul(
            node_id="mm_1",
            input_name="x",
            weight_name="W",
            output_name="y",
            output_shape=Shape.vector(128),
            weight_shape=(128, 64),
        )
        assert node.scheme == Scheme.CKKS
        assert node.output_shape.size == 128

    def test_add_node(self):
        """Test CKKSAdd node creation."""
        node = CKKSAdd(
            node_id="add_1",
            lhs_name="a",
            rhs_name="b",
            output_name="c",
        )
        assert node.scheme == Scheme.CKKS

    def test_rescale_node(self):
        """Test CKKSRescale node creation."""
        node = CKKSRescale(
            node_id="rs_1",
            input_name="x",
            output_name="x_rs",
            scale_bits=40,
        )
        assert node.scheme == Scheme.CKKS
        assert node.scale_bits == 40


class TestTFHENodes:
    """Tests for TFHE operation nodes."""

    def test_lut_node(self):
        """Test TFHELUT node creation."""
        node = TFHELUT(
            node_id="lut_1",
            input_name="x",
            output_name="y",
            lut_name="step",
            output_type=ValueType.BIT,
        )
        assert node.scheme == Scheme.TFHE
        assert node.output_type == ValueType.BIT

    def test_lut_node_custom_lut(self):
        """Test TFHELUT with custom LUT data."""
        custom_lut = [0, 0, 0, 1, 1, 1, 1, 1]  # 3-bit step at 3
        node = TFHELUT(
            node_id="lut_2",
            input_name="x",
            output_name="y",
            lut_name="custom",
            lut_data=custom_lut,
            output_type=ValueType.INT_4,
        )
        assert node.lut_data == custom_lut


class TestBridgeNodes:
    """Tests for bridge operation nodes."""

    def test_quantize_node(self):
        """Test CKKSQuantizeToInt node."""
        node = CKKSQuantizeToInt(
            node_id="q_1",
            input_name="z",
            output_name="z_q",
            bits=8,
            clip_min=-10.0,
            clip_max=10.0,
        )
        assert node.bits == 8
        assert node.clip_min == -10.0
        assert node.clip_max == 10.0

    def test_ckks_to_tfhe_node(self):
        """Test CKKSToTFHE bridge node."""
        node = CKKSToTFHE(
            node_id="b_1",
            input_name="z_q",
            output_name="z_tfhe",
            output_type=ValueType.INT_8,
        )
        assert node.output_type == ValueType.INT_8

    def test_tfhe_to_ckks_node(self):
        """Test TFHEToCKKS bridge node."""
        node = TFHEToCKKS(
            node_id="b_2",
            input_name="g_tfhe",
            output_name="g_ckks",
            scale_bits=40,
        )
        assert node.scale_bits == 40


# =============================================================================
# Test Runner
# =============================================================================

def run_ir_tests():
    """Run all IR tests and report results."""
    import sys

    test_classes = [
        TestShape,
        TestValueType,
        TestIRValue,
        TestIRProgram,
        TestValidation,
        TestCKKSNodes,
        TestTFHENodes,
        TestBridgeNodes,
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

    print(f"\nIR Tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_ir_tests()
    exit(0 if success else 1)
