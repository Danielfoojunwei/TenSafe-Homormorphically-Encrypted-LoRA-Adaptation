"""
Test that no BaseModel or SQLModel subclass uses bare mutable defaults.

Bare `= []` or `= {}` on class-level annotations in Pydantic models
create shared mutable state across instances.  The canonical fix is
`Field(default_factory=list)` or `Field(default_factory=dict)`.

This test uses AST parsing so it works without importing every module
(which may have heavy deps like torch).
"""

import ast
import sys
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src"

# Collect all Python files under src/
_PY_FILES = sorted(SRC_ROOT.rglob("*.py"))

# Known BaseModel / SQLModel parent class names
_MODEL_BASES = {
    "BaseModel",
    "SQLModel",
}


def _find_violations(filepath: Path) -> list:
    """
    Parse a single file and return a list of (line, field_name, default_repr)
    tuples for any class-level annotated assignment with a bare `[]` or `{}`
    inside a BaseModel/SQLModel subclass.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return []

    violations = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if any base looks like a known model class
        is_model = False
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = base.attr
            if base_name in _MODEL_BASES:
                is_model = True
                break

        if not is_model:
            continue

        # Walk direct class body for AnnAssign with bare [] or {}
        for stmt in node.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if stmt.value is None:
                continue

            target_name = ""
            if isinstance(stmt.target, ast.Name):
                target_name = stmt.target.id

            val = stmt.value
            # Bare list: `= []`
            if isinstance(val, ast.List) and len(val.elts) == 0:
                violations.append((stmt.lineno, target_name, "[]"))
            # Bare dict: `= {}`
            elif isinstance(val, ast.Dict) and len(val.keys) == 0:
                violations.append((stmt.lineno, target_name, "{}"))

    return violations


def test_no_bare_mutable_defaults_in_models():
    """Ensure no BaseModel/SQLModel subclass has bare [] or {} defaults."""
    all_violations = []

    for pyfile in _PY_FILES:
        violations = _find_violations(pyfile)
        for lineno, field_name, default_repr in violations:
            rel = pyfile.relative_to(SRC_ROOT)
            all_violations.append(f"  {rel}:{lineno}  {field_name} = {default_repr}")

    if all_violations:
        msg = (
            "Found BaseModel/SQLModel fields with bare mutable defaults.\n"
            "Replace `= []` with `Field(default_factory=list)` and "
            "`= {}` with `Field(default_factory=dict)`.\n\n"
            + "\n".join(all_violations)
        )
        pytest.fail(msg)
