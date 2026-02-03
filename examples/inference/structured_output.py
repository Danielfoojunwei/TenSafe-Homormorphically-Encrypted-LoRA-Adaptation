"""
Structured Output Example

Demonstrates generating structured JSON outputs with schema validation.

Requirements:
- TenSafe account and API key

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python structured_output.py
"""

import os
import json
from tensafe import TenSafeClient


def main():
    client = TenSafeClient(api_key=os.environ.get("TENSAFE_API_KEY"))
    tc = client.create_training_client(model_ref="meta-llama/Llama-3-8B")

    # Define output schema
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "topics": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string", "maxLength": 100},
        },
        "required": ["sentiment", "confidence", "topics", "summary"],
    }

    # Analyze text with structured output
    text = "I absolutely love TenSafe! The privacy features are incredible and the API is so easy to use."

    result = tc.sample(
        prompts=[f"Analyze this text and return JSON: {text}"],
        max_tokens=200,
        temperature=0.3,
        response_format={"type": "json_object", "schema": schema},
    )

    # Parse and display result
    output = json.loads(result.samples[0].completion)
    print("Analysis Result:")
    print(f"  Sentiment: {output['sentiment']}")
    print(f"  Confidence: {output['confidence']:.2f}")
    print(f"  Topics: {', '.join(output['topics'])}")
    print(f"  Summary: {output['summary']}")


if __name__ == "__main__":
    main()
