#!/usr/bin/env python3
"""
OASST1 Dataset Loader and Preprocessor

Loads the OpenAssistant/oasst1 dataset (Apache-2.0 license) and converts it
to Llama-3 Instruct chat template format.

Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
License: Apache-2.0

Filtering:
- English language only (lang == "en")
- Assistant turns only (role == "assistant")
- Paired with user prompts from conversation tree
"""

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Llama 3 Instruct chat template
LLAMA3_CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_message}<|eot_id|>"""

DEFAULT_SYSTEM_MESSAGE = "You are a helpful, harmless, and honest AI assistant."


@dataclass
class ConversationTurn:
    """A single conversation turn."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatExample:
    """A complete chat example for training."""
    system: str
    user: str
    assistant: str
    source_id: str

    def to_llama3_format(self) -> str:
        """Convert to Llama 3 Instruct format."""
        return LLAMA3_CHAT_TEMPLATE.format(
            system_message=self.system,
            user_message=self.user,
            assistant_message=self.assistant,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    dataset_name: str = "OpenAssistant/oasst1"
    language: str = "en"
    max_train_samples: Optional[int] = None
    eval_samples: int = 1000
    seed: int = 42
    system_message: str = DEFAULT_SYSTEM_MESSAGE
    max_seq_length: int = 2048

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OASST1Loader:
    """
    Loads and preprocesses the OpenAssistant/oasst1 dataset.

    Filtering logic:
    1. Filter to English messages (lang == "en")
    2. Build conversation trees from message_id -> parent_id relationships
    3. Extract user-assistant pairs where the assistant response is high quality
    4. Convert to Llama 3 Instruct chat template format
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self._dataset = None
        self._message_tree = {}

    def _check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import datasets
            return True
        except ImportError:
            logger.warning("datasets library not available. Install with: pip install datasets")
            return False

    def load_raw_dataset(self):
        """Load the raw OASST1 dataset from Hugging Face."""
        if not self._check_dependencies():
            return None

        from datasets import load_dataset

        logger.info(f"Loading dataset: {self.config.dataset_name}")
        try:
            self._dataset = load_dataset(self.config.dataset_name)
            logger.info(f"Dataset loaded: {len(self._dataset['train'])} train samples")
            return self._dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None

    def _build_message_tree(self, messages: List[Dict]) -> Dict[str, Dict]:
        """Build a tree structure from messages using parent_id relationships."""
        tree = {}
        for msg in messages:
            msg_id = msg.get("message_id")
            if msg_id:
                tree[msg_id] = msg
        return tree

    def _get_parent_message(self, message: Dict, tree: Dict) -> Optional[Dict]:
        """Get the parent message in the conversation tree."""
        parent_id = message.get("parent_id")
        if parent_id and parent_id in tree:
            return tree[parent_id]
        return None

    def _extract_user_assistant_pairs(
        self,
        messages: List[Dict],
    ) -> List[ChatExample]:
        """
        Extract user-assistant pairs from the message tree.

        For each assistant message:
        1. Find its parent (should be a user message)
        2. Create a ChatExample with user prompt and assistant response
        """
        examples = []

        # Build message tree
        tree = self._build_message_tree(messages)

        for msg in messages:
            # Filter: English only
            if msg.get("lang") != self.config.language:
                continue

            # Filter: Assistant messages only
            if msg.get("role") != "assistant":
                continue

            # Get parent (user) message
            parent = self._get_parent_message(msg, tree)
            if not parent or parent.get("role") != "prompter":
                continue

            # Filter: Parent must also be English
            if parent.get("lang") != self.config.language:
                continue

            # Extract text
            user_text = parent.get("text", "").strip()
            assistant_text = msg.get("text", "").strip()

            # Skip empty messages
            if not user_text or not assistant_text:
                continue

            # Create example
            example = ChatExample(
                system=self.config.system_message,
                user=user_text,
                assistant=assistant_text,
                source_id=msg.get("message_id", "unknown"),
            )
            examples.append(example)

        return examples

    def preprocess(self) -> Tuple[List[ChatExample], List[ChatExample]]:
        """
        Preprocess the dataset into train and eval splits.

        Returns:
            Tuple of (train_examples, eval_examples)
        """
        if self._dataset is None:
            self.load_raw_dataset()

        if self._dataset is None:
            logger.warning("Dataset not available, returning empty lists")
            return [], []

        # Convert to list of dicts
        train_messages = [dict(m) for m in self._dataset["train"]]
        val_messages = [dict(m) for m in self._dataset.get("validation", [])]

        # Combine all messages for tree building
        all_messages = train_messages + val_messages

        # Extract user-assistant pairs
        logger.info("Extracting user-assistant pairs...")
        examples = self._extract_user_assistant_pairs(all_messages)
        logger.info(f"Extracted {len(examples)} conversation pairs")

        # Shuffle with seed for reproducibility
        random.seed(self.config.seed)
        random.shuffle(examples)

        # Split into train and eval
        eval_examples = examples[:self.config.eval_samples]
        train_examples = examples[self.config.eval_samples:]

        # Apply max train samples limit if specified
        if self.config.max_train_samples:
            train_examples = train_examples[:self.config.max_train_samples]

        logger.info(f"Train split: {len(train_examples)} examples")
        logger.info(f"Eval split: {len(eval_examples)} examples")

        return train_examples, eval_examples

    def get_smoke_subset(
        self,
        n_train: int = 100,
        n_eval: int = 20,
    ) -> Tuple[List[ChatExample], List[ChatExample]]:
        """
        Get a tiny subset for smoke testing (CPU-friendly).

        Uses deterministic sampling with fixed seed.
        """
        train_examples, eval_examples = self.preprocess()

        # Take small subsets
        random.seed(self.config.seed)

        if len(train_examples) > n_train:
            train_subset = random.sample(train_examples, n_train)
        else:
            train_subset = train_examples

        if len(eval_examples) > n_eval:
            eval_subset = random.sample(eval_examples, n_eval)
        else:
            eval_subset = eval_examples

        logger.info(f"Smoke subset - Train: {len(train_subset)}, Eval: {len(eval_subset)}")
        return train_subset, eval_subset

    def save_preprocessed(
        self,
        train_examples: List[ChatExample],
        eval_examples: List[ChatExample],
        output_dir: Path,
    ) -> Dict[str, str]:
        """Save preprocessed examples to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save train
        train_path = output_dir / "train.jsonl"
        with open(train_path, "w") as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Save eval
        eval_path = output_dir / "eval.jsonl"
        with open(eval_path, "w") as f:
            for ex in eval_examples:
                f.write(json.dumps(ex.to_dict()) + "\n")

        # Save config
        config_path = output_dir / "dataset_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Compute dataset hash for reproducibility
        dataset_hash = self._compute_dataset_hash(train_examples, eval_examples)

        # Save metadata
        metadata = {
            "dataset_name": self.config.dataset_name,
            "license": "Apache-2.0",
            "source_url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
            "language": self.config.language,
            "train_samples": len(train_examples),
            "eval_samples": len(eval_examples),
            "seed": self.config.seed,
            "dataset_hash": dataset_hash,
        }
        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved preprocessed data to {output_dir}")

        return {
            "train": str(train_path),
            "eval": str(eval_path),
            "config": str(config_path),
            "metadata": str(metadata_path),
        }

    def _compute_dataset_hash(
        self,
        train_examples: List[ChatExample],
        eval_examples: List[ChatExample],
    ) -> str:
        """Compute a hash of the dataset for reproducibility tracking."""
        hasher = hashlib.sha256()

        for ex in train_examples[:100]:  # Sample for efficiency
            hasher.update(ex.source_id.encode())
        for ex in eval_examples[:100]:
            hasher.update(ex.source_id.encode())

        return hasher.hexdigest()[:16]


def create_synthetic_dataset(
    n_train: int = 50,
    n_eval: int = 10,
    seed: int = 42,
) -> Tuple[List[ChatExample], List[ChatExample]]:
    """
    Create a synthetic dataset for smoke testing when the real dataset
    is not available (e.g., in CI without network access).
    """
    random.seed(seed)

    prompts = [
        "What is machine learning?",
        "Explain how neural networks work.",
        "What is the difference between supervised and unsupervised learning?",
        "How does gradient descent work?",
        "What is a loss function?",
        "Explain backpropagation.",
        "What is overfitting?",
        "How do you prevent overfitting?",
        "What is cross-validation?",
        "Explain regularization techniques.",
        "What is a convolutional neural network?",
        "How do transformers work?",
        "What is attention in deep learning?",
        "Explain the concept of embeddings.",
        "What is transfer learning?",
        "How does fine-tuning work?",
        "What is LoRA?",
        "Explain parameter-efficient fine-tuning.",
        "What are the benefits of quantization?",
        "How do you evaluate language models?",
    ]

    responses = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Neural networks are computational models inspired by biological neurons.",
        "Supervised learning uses labeled data, while unsupervised learning finds patterns.",
        "Gradient descent optimizes by iteratively moving toward the minimum of a function.",
        "A loss function measures how well the model's predictions match the actual values.",
        "Backpropagation computes gradients by propagating errors backward through the network.",
        "Overfitting occurs when a model learns noise instead of the underlying pattern.",
        "Regularization, dropout, and early stopping help prevent overfitting.",
        "Cross-validation assesses model performance by training on different data subsets.",
        "Regularization adds penalties to the loss function to prevent overfitting.",
        "CNNs use convolutional layers to automatically learn spatial hierarchies.",
        "Transformers use self-attention to process sequences in parallel.",
        "Attention allows models to focus on relevant parts of the input.",
        "Embeddings are dense vector representations of discrete entities.",
        "Transfer learning applies knowledge from one task to another.",
        "Fine-tuning adapts a pre-trained model to a specific task.",
        "LoRA is a parameter-efficient method that trains low-rank adapters.",
        "PEFT methods train only a small subset of model parameters.",
        "Quantization reduces model size and improves inference speed.",
        "Language models are evaluated using perplexity, accuracy, and human evaluation.",
    ]

    examples = []
    for i in range(max(n_train + n_eval, len(prompts))):
        idx = i % len(prompts)
        example = ChatExample(
            system=DEFAULT_SYSTEM_MESSAGE,
            user=prompts[idx] + f" (variant {i})" if i >= len(prompts) else prompts[idx],
            assistant=responses[idx],
            source_id=f"synthetic_{i}",
        )
        examples.append(example)

    random.shuffle(examples)

    return examples[:n_train], examples[n_train:n_train + n_eval]


def load_from_jsonl(path: Path) -> List[ChatExample]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            examples.append(ChatExample(**data))
    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess OASST1 dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/oasst1"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Create smoke test subset")
    parser.add_argument("--synthetic", action="store_true", help="Create synthetic data")

    args = parser.parse_args()

    if args.synthetic:
        train, eval_data = create_synthetic_dataset(
            n_train=100 if args.smoke else 1000,
            n_eval=20 if args.smoke else 100,
            seed=args.seed,
        )
        output_dir = args.output_dir / "synthetic"
    else:
        config = DatasetConfig(
            max_train_samples=args.max_train_samples,
            eval_samples=args.eval_samples,
            seed=args.seed,
        )
        loader = OASST1Loader(config)

        if args.smoke:
            train, eval_data = loader.get_smoke_subset()
            output_dir = args.output_dir / "smoke"
        else:
            train, eval_data = loader.preprocess()
            output_dir = args.output_dir / "full"

    # Save (using a simple implementation if loader not available)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex.to_dict()) + "\n")

    eval_path = output_dir / "eval.jsonl"
    with open(eval_path, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex.to_dict()) + "\n")

    print(f"Saved {len(train)} train and {len(eval_data)} eval examples to {output_dir}")
