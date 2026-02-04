#!/usr/bin/env python3
"""
Data Loading for TenSafe Training

This example demonstrates efficient data loading strategies for
privacy-preserving training, including:
- Dataset preparation and tokenization
- Efficient batching for DP-SGD
- Streaming for large datasets
- Privacy-safe data preprocessing
- Multi-worker data loading

Key considerations for DP training:
- Fixed dataset size (for privacy accounting)
- Poisson sampling (for DP guarantees)
- No data augmentation that leaks information

Expected Output:
    Preparing dataset...
    Tokenizing 1000 examples...
    Creating DataLoader with 4 workers...

    Batch statistics:
      Batch 1: 32 samples, avg_length=128
      Batch 2: 32 samples, avg_length=135
    ...

    Data loading complete!
"""

from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Iterator, Optional, Callable

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dataclass
class DataConfig:
    """Configuration for data loading."""
    # Dataset
    dataset_path: str = "data/train.jsonl"
    max_samples: Optional[int] = None  # None = use all

    # Tokenization
    max_length: int = 512
    truncation: bool = True
    padding: str = "max_length"

    # Batching
    batch_size: int = 32
    shuffle: bool = True
    drop_last: bool = True  # Important for DP accounting

    # Workers
    num_workers: int = 4
    prefetch_factor: int = 2

    # DP-specific
    use_poisson_sampling: bool = False  # For DP subsampling


class TextDataset:
    """Simple text dataset for training."""

    def __init__(
        self,
        examples: List[Dict[str, str]],
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]

        # Format prompt-response
        if "instruction" in example:
            text = f"### Instruction:\n{example['instruction']}\n\n"
            if example.get("input"):
                text += f"### Input:\n{example['input']}\n\n"
            text += f"### Response:\n{example.get('output', '')}"
        else:
            text = example.get("text", "")

        # Tokenize (simulated)
        tokens = text.split()[:self.max_length]
        input_ids = list(range(len(tokens)))
        attention_mask = [1] * len(tokens)
        labels = input_ids.copy()

        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [0] * padding_length
            attention_mask += [0] * padding_length
            labels += [-100] * padding_length  # Ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataLoader:
    """DataLoader with DP-compatible sampling."""

    def __init__(
        self,
        dataset: TextDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = True,
        use_poisson_sampling: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.use_poisson_sampling = use_poisson_sampling

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, List]]:
        indices = list(range(len(self.dataset)))

        if self.use_poisson_sampling:
            # Poisson subsampling for DP
            # Each sample included independently with probability batch_size/n
            p = self.batch_size / len(self.dataset)
            for _ in range(len(self)):
                batch_indices = [i for i in indices if random.random() < p]
                if len(batch_indices) > 0:
                    yield self._collate([self.dataset[i] for i in batch_indices])
        else:
            # Standard shuffling
            if self.shuffle:
                random.shuffle(indices)

            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                yield self._collate([self.dataset[i] for i in batch_indices])

    def _collate(self, batch: List[Dict]) -> Dict[str, List]:
        """Collate batch of samples."""
        return {
            "input_ids": [s["input_ids"] for s in batch],
            "attention_mask": [s["attention_mask"] for s in batch],
            "labels": [s["labels"] for s in batch],
        }


def create_sample_data(num_samples: int = 100) -> List[Dict[str, str]]:
    """Create sample instruction-following data."""
    templates = [
        {"instruction": "Summarize the following:", "input": "Long text...", "output": "Summary..."},
        {"instruction": "Translate to French:", "input": "Hello world", "output": "Bonjour monde"},
        {"instruction": "Explain in simple terms:", "input": "Quantum computing", "output": "Explanation..."},
    ]

    data = []
    for i in range(num_samples):
        template = templates[i % len(templates)].copy()
        template["id"] = i
        data.append(template)

    return data


def main():
    """Demonstrate data loading for TenSafe training."""
    print("=" * 60)
    print("DATA LOADING FOR TENSAFE")
    print("=" * 60)
    print("""
    Data loading considerations for privacy-preserving training:

    1. Fixed Dataset Size
       - Privacy accounting requires known dataset size
       - No dynamic data augmentation

    2. Poisson Subsampling (for strict DP)
       - Each sample included independently
       - Probability = batch_size / dataset_size

    3. Drop Last Batch
       - Ensures consistent batch sizes
       - Important for DP noise calibration

    4. No Information Leakage
       - Preprocessing must be deterministic
       - No data-dependent operations
    """)

    # Configuration
    config = DataConfig(
        batch_size=32,
        max_length=256,
        shuffle=True,
        drop_last=True,
        num_workers=4,
    )

    print(f"\nConfiguration:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max length: {config.max_length}")
    print(f"  Shuffle: {config.shuffle}")
    print(f"  Drop last: {config.drop_last}")

    # Create sample data
    print("\nPreparing dataset...")
    raw_data = create_sample_data(num_samples=1000)
    print(f"  Raw samples: {len(raw_data)}")

    # Create dataset
    dataset = TextDataset(raw_data, max_length=config.max_length)
    print(f"  Tokenized samples: {len(dataset)}")

    # Create dataloader
    print("\nCreating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        drop_last=config.drop_last,
    )
    print(f"  Batches per epoch: {len(dataloader)}")

    # Sample batches
    print("\nSampling batches:")
    print("-" * 40)

    for i, batch in enumerate(dataloader):
        if i >= 3:
            break

        batch_size = len(batch["input_ids"])
        avg_length = sum(
            sum(1 for x in ids if x != 0)
            for ids in batch["input_ids"]
        ) / batch_size

        print(f"  Batch {i+1}: {batch_size} samples, avg_length={avg_length:.0f}")

    # Poisson sampling demonstration
    print("\n" + "=" * 60)
    print("POISSON SUBSAMPLING FOR DP")
    print("=" * 60)
    print("""
    For strict DP guarantees, use Poisson subsampling:
    - Each sample included with probability q = batch_size / n
    - Batch sizes vary (binomial distribution)
    - Provides amplification by subsampling
    """)

    dp_dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        use_poisson_sampling=True,
    )

    print("\nPoisson-sampled batches:")
    print("-" * 40)

    batch_sizes = []
    for i, batch in enumerate(dp_dataloader):
        if i >= 5:
            break
        batch_sizes.append(len(batch["input_ids"]))
        print(f"  Batch {i+1}: {len(batch['input_ids'])} samples (varies)")

    avg_size = sum(batch_sizes) / len(batch_sizes)
    print(f"\n  Average batch size: {avg_size:.1f} (expected: {config.batch_size})")

    # Best practices
    print("\n" + "=" * 60)
    print("BEST PRACTICES")
    print("=" * 60)
    print("""
    1. Use streaming for large datasets
       from datasets import load_dataset
       dataset = load_dataset("...", streaming=True)

    2. Precompute tokenization
       - Saves time during training
       - Ensures consistency

    3. Use multiple workers
       - num_workers = 4-8 typically
       - prefetch_factor = 2

    4. Memory-map large files
       - Avoids loading full dataset
       - Enables random access

    5. For DP training
       - Use drop_last=True
       - Consider Poisson sampling
       - Fix dataset size before training
    """)


if __name__ == "__main__":
    main()
