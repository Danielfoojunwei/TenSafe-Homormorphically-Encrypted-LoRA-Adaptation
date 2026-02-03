"""
LoRA Adapter Merging Example

Demonstrates merging multiple LoRA adapters for combined capabilities
while maintaining privacy metadata.

Requirements:
- TenSafe account and API key
- Multiple trained LoRA adapters

Usage:
    export TENSAFE_API_KEY="your-api-key"
    python model_merging.py
"""

import os
from tensafe import TenSafeClient
from tensafe.merging import MergeConfig, MergeMethod


def main():
    client = TenSafeClient(
        api_key=os.environ.get("TENSAFE_API_KEY"),
    )

    # Load multiple adapters
    adapters = [
        client.load_adapter("adapter-coding-v1"),
        client.load_adapter("adapter-writing-v1"),
        client.load_adapter("adapter-math-v1"),
    ]

    print("Loaded adapters:")
    for a in adapters:
        print(f"  - {a.id}: {a.name}")
    print()

    # Method 1: Linear combination (weighted average)
    print("Method 1: Linear Combination")
    linear_config = MergeConfig(
        method=MergeMethod.LINEAR,
        weights=[0.4, 0.3, 0.3],  # Coding-focused blend
        normalize=True,
    )

    merged_linear = client.merge_adapters(
        adapters=adapters,
        config=linear_config,
        output_name="merged-linear-v1",
    )
    print(f"  Created: {merged_linear.id}")

    # Method 2: TIES merging (Trim, Elect, and Merge)
    print("\nMethod 2: TIES Merging")
    ties_config = MergeConfig(
        method=MergeMethod.TIES,
        density=0.5,  # Keep top 50% of parameters
        majority_sign=True,  # Use majority sign for ties
    )

    merged_ties = client.merge_adapters(
        adapters=adapters,
        config=ties_config,
        output_name="merged-ties-v1",
    )
    print(f"  Created: {merged_ties.id}")

    # Method 3: DARE (Drop And REscale)
    print("\nMethod 3: DARE Merging")
    dare_config = MergeConfig(
        method=MergeMethod.DARE,
        drop_rate=0.9,  # Drop 90% randomly
        rescale=True,  # Rescale remaining
    )

    merged_dare = client.merge_adapters(
        adapters=adapters,
        config=dare_config,
        output_name="merged-dare-v1",
    )
    print(f"  Created: {merged_dare.id}")

    # Verify privacy metadata preserved
    print("\nPrivacy Metadata Check:")
    for merged in [merged_linear, merged_ties, merged_dare]:
        meta = merged.get_privacy_metadata()
        print(f"  {merged.id}:")
        print(f"    Combined epsilon: {meta.get('combined_epsilon', 'N/A')}")
        print(f"    Source adapters: {len(meta.get('source_adapters', []))}")

    # Export merged adapter as TGSP
    print("\nExporting as TGSP...")
    tgsp_path = client.tgsp.export_adapter(
        adapter=merged_linear,
        output_path="/tmp/merged-linear.tgsp",
    )
    print(f"Exported to: {tgsp_path}")


if __name__ == "__main__":
    main()
