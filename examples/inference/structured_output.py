#!/usr/bin/env python3
"""
Structured Output / JSON Mode with TenSafe

This example demonstrates how to generate structured outputs from LLMs,
including JSON mode, schema validation, and function calling patterns.
Structured outputs are essential for integrating LLMs into applications.

What this example demonstrates:
- Enabling JSON mode for structured responses
- Defining and validating output schemas
- Using dataclasses for type-safe outputs
- Function calling and tool use patterns

Key concepts:
- JSON mode: Force valid JSON output
- Schema validation: Ensure output matches expected format
- Response format: Specify output structure
- Constrained generation: Grammar-based output control

Prerequisites:
- TenSafe server running
- Trained LoRA adapter (optional)

Expected Output:
    JSON Mode Example:
    {
      "name": "Paris",
      "country": "France",
      "population": 2161000,
      "landmarks": ["Eiffel Tower", "Louvre", "Notre-Dame"]
    }

    Schema Validation: PASSED
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any
from enum import Enum

# Add project root to path for development
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class OutputFormat(Enum):
    """Supported output formats."""
    TEXT = "text"
    JSON = "json"
    JSON_SCHEMA = "json_schema"


@dataclass
class City:
    """Schema for city information."""
    name: str
    country: str
    population: int
    landmarks: List[str]
    description: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "City":
        return cls(
            name=data["name"],
            country=data["country"],
            population=data["population"],
            landmarks=data["landmarks"],
            description=data.get("description"),
        )

    @classmethod
    def json_schema(cls) -> Dict[str, Any]:
        """Return JSON schema for this model."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "City name"},
                "country": {"type": "string", "description": "Country name"},
                "population": {"type": "integer", "description": "Population count"},
                "landmarks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Famous landmarks"
                },
                "description": {"type": "string", "description": "Optional description"},
            },
            "required": ["name", "country", "population", "landmarks"],
        }


@dataclass
class StructuredOutputConfig:
    """Configuration for structured output generation."""
    format: OutputFormat = OutputFormat.JSON
    schema: Optional[Dict[str, Any]] = None
    strict: bool = True
    max_tokens: int = 1024


class StructuredOutputProcessor:
    """Process and validate structured outputs from LLMs."""

    def __init__(self, config: StructuredOutputConfig):
        self.config = config

    def validate_json(self, output: str) -> tuple[bool, Optional[Dict], Optional[str]]:
        """Validate that output is valid JSON."""
        try:
            parsed = json.loads(output)
            return True, parsed, None
        except json.JSONDecodeError as e:
            return False, None, str(e)

    def validate_schema(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """Validate data against JSON schema."""
        errors = []

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check field types
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    errors.append(f"Field '{field}' has wrong type")

        return len(errors) == 0, errors

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True
        return isinstance(value, expected)

    def process(self, raw_output: str) -> tuple[Any, bool, List[str]]:
        """Process raw output and return validated result."""
        # Validate JSON
        is_valid_json, parsed, json_error = self.validate_json(raw_output)
        if not is_valid_json:
            return None, False, [f"Invalid JSON: {json_error}"]

        # Validate schema if provided
        if self.config.schema:
            is_valid_schema, schema_errors = self.validate_schema(
                parsed, self.config.schema
            )
            if not is_valid_schema:
                return parsed, False, schema_errors

        return parsed, True, []


def simulate_structured_generation(prompt: str) -> str:
    """Simulate structured output generation."""
    if "Paris" in prompt or "city" in prompt.lower():
        return json.dumps({
            "name": "Paris",
            "country": "France",
            "population": 2161000,
            "landmarks": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral"],
            "description": "The capital and largest city of France."
        }, indent=2)
    else:
        return json.dumps({
            "response": "Structured response",
            "confidence": 0.95,
        }, indent=2)


def main():
    """Demonstrate structured output generation."""

    # =========================================================================
    # Step 1: Understanding structured outputs
    # =========================================================================
    print("=" * 60)
    print("STRUCTURED OUTPUT / JSON MODE")
    print("=" * 60)
    print("""
    Structured outputs ensure LLM responses follow a specific format:

    Traditional text output:
      "Paris is the capital of France with about 2.1 million people."

    Structured JSON output:
      {
        "name": "Paris",
        "country": "France",
        "population": 2161000,
        "landmarks": ["Eiffel Tower", "Louvre"]
      }

    Benefits:
    - Predictable, parseable outputs
    - Easy integration with applications
    - Type-safe data extraction
    - Reduced post-processing errors
    """)

    # =========================================================================
    # Step 2: Basic JSON mode
    # =========================================================================
    print("\n" + "=" * 60)
    print("BASIC JSON MODE")
    print("=" * 60)

    prompt = "Provide information about Paris as JSON"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating structured output...")

    raw_output = simulate_structured_generation(prompt)
    print(f"\nJSON Output:\n{raw_output}")

    # Parse and use the output
    data = json.loads(raw_output)
    print(f"\nParsed data:")
    print(f"  City: {data['name']}, {data['country']}")
    print(f"  Population: {data['population']:,}")
    print(f"  Landmarks: {', '.join(data['landmarks'])}")

    # =========================================================================
    # Step 3: Schema validation
    # =========================================================================
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)

    config = StructuredOutputConfig(
        format=OutputFormat.JSON_SCHEMA,
        schema=City.json_schema(),
        strict=True,
    )

    processor = StructuredOutputProcessor(config)

    print("\nJSON Schema:")
    print(json.dumps(City.json_schema(), indent=2))

    print("\nValidating output against schema...")
    parsed, is_valid, errors = processor.process(raw_output)

    if is_valid:
        print("Validation: PASSED")
        city = City.from_dict(parsed)
        print(f"Parsed to model: City(name='{city.name}', ...)")
    else:
        print(f"Validation: FAILED")
        for error in errors:
            print(f"  - {error}")

    # =========================================================================
    # Step 4: Function calling
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUNCTION CALLING / TOOL USE")
    print("=" * 60)
    print("""
    Function calling enables LLMs to invoke external tools:

    1. Define available functions with schemas
    2. LLM decides which function to call
    3. LLM generates structured arguments
    4. Application executes function
    5. Result fed back to LLM
    """)

    functions = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    ]

    print("\nAvailable functions:")
    for func in functions:
        print(f"  - {func['name']}: {func['description']}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BEST PRACTICES")
    print("=" * 60)
    print("""
    Tips for structured output generation:

    1. Schema design
       - Keep schemas simple and flat when possible
       - Use descriptive field names
       - Include field descriptions

    2. Prompt engineering
       - Explicitly request JSON format
       - Include example output in prompt

    3. Validation
       - Always validate outputs before use
       - Handle validation failures gracefully

    4. Error handling
       - Catch JSON parse errors
       - Validate against schema
       - Have fallback for invalid outputs
    """)


if __name__ == "__main__":
    main()
