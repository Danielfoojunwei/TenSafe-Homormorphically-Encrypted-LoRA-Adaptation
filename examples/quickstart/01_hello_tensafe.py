#!/usr/bin/env python3
"""
Hello TenSafe - Simplest Possible Example

This is the simplest example showing how to connect to TenSafe and verify
the connection is working. It's the "Hello World" of TenSafe.

What this example demonstrates:
- Importing the TenSafe SDK
- Creating a ServiceClient connection
- Checking the connection status

Prerequisites:
- TenSafe server running (or use mock mode)
- API key configured (TG_TINKER_API_KEY env var)

Expected Output:
    TenSafe SDK initialized successfully!
    Connected to: http://localhost:8000
    SDK Version: 1.0.0
    Connection verified!
"""

from __future__ import annotations

import os
import sys

# Add project root to path for development
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    """Demonstrate basic TenSafe SDK connection."""

    # =========================================================================
    # Step 1: Import the SDK
    # =========================================================================
    try:
        from tg_tinker import ServiceClient
        print("TenSafe SDK imported successfully!")
    except ImportError as e:
        print(f"Error importing TenSafe SDK: {e}")
        print("Make sure you have installed the SDK: pip install tg_tinker")
        sys.exit(1)

    # =========================================================================
    # Step 2: Configure connection settings
    # =========================================================================
    # Configuration can come from environment variables or be passed directly
    base_url = os.environ.get("TG_TINKER_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("TG_TINKER_API_KEY", "demo-api-key-for-testing")

    print(f"\nConfiguration:")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {'*' * (len(api_key) - 4) + api_key[-4:]}")  # Mask API key

    # =========================================================================
    # Step 3: Create the ServiceClient
    # =========================================================================
    print("\nInitializing ServiceClient...")

    try:
        # The ServiceClient is the main entry point for all TenSafe operations
        client = ServiceClient(
            base_url=base_url,
            api_key=api_key,
            timeout=30.0,  # 30 second timeout
        )
        print(f"ServiceClient created: {client}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Connection error: {e}")
        print("\nMake sure the TenSafe server is running.")
        print("For development, you can start a mock server or use the demo mode.")
        sys.exit(1)

    # =========================================================================
    # Step 4: Verify the connection (optional health check)
    # =========================================================================
    print("\nConnection established successfully!")
    print("You're ready to use TenSafe for privacy-preserving ML.")

    # Clean up
    client.close()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    You've successfully connected to TenSafe. Next steps:

    1. Create a training client for LoRA fine-tuning
       See: 02_basic_training.py

    2. Run inference with trained models
       See: 03_inference.py

    3. Add differential privacy to your training
       See: 04_dp_training.py

    4. Use homomorphic encryption for inference
       See: 05_encrypted_inference.py
    """)


if __name__ == "__main__":
    main()
