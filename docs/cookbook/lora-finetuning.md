# Cookbook: LoRA Fine-Tuning

Complete walkthrough for fine-tuning Llama with LoRA on TenSafe.

## Overview

This tutorial covers:
- Setting up a LoRA fine-tuning job
- Training with privacy guarantees
- Evaluating and saving checkpoints
- Deploying for inference

## Prerequisites

```bash
pip install tg-tinker datasets torch
export TS_API_KEY=ts-your-key
```

## Step 1: Prepare Data

```python
from datasets import load_dataset
import torch

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def format_instruction(example):
    """Format as instruction-following."""
    if example["input"]:
        text = f"""### Instruction:
{example["instruction"]}

### Input:
{example["input"]}

### Response:
{example["output"]}"""
    else:
        text = f"""### Instruction:
{example["instruction"]}

### Response:
{example["output"]}"""
    return {"text": text}

dataset = dataset.map(format_instruction)
```

## Step 2: Tokenize

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding="max_length",
    )

tokenized = dataset.map(tokenize, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask"])
```

## Step 3: Create DataLoader

```python
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    # Labels are input_ids shifted by 1 (handled server-side)
    return {
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "labels": input_ids.tolist(),
    }

dataloader = DataLoader(
    tokenized,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)
```

## Step 4: Configure Training

```python
from tg_tinker import (
    ServiceClient,
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
)

# LoRA configuration
lora_config = LoRAConfig(
    rank=16,
    alpha=32,
    dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# Optimizer with warmup
optimizer = OptimizerConfig(
    name="adamw",
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler="cosine",
)

# Optional: Differential Privacy
dp_config = DPConfig(
    enabled=True,
    target_epsilon=8.0,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# Full training config
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=lora_config,
    optimizer=optimizer,
    dp_config=dp_config,
    batch_size=8,
    gradient_accumulation_steps=4,
    max_sequence_length=2048,
    mixed_precision="bf16",
)
```

## Step 5: Training Loop

```python
from tqdm import tqdm

# Initialize client
service = ServiceClient()
tc = service.create_training_client(config)

print(f"Training client ID: {tc.id}")

# Training settings
num_epochs = 3
eval_every = 100
save_every = 500

# Track metrics
best_val_loss = float("inf")
train_losses = []

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    tc.train()
    epoch_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
    for batch in pbar:
        # Forward-backward
        fb_future = tc.forward_backward(batch)
        fb_result = fb_future.result()

        # Optimizer step (with DP noise if enabled)
        opt_future = tc.optim_step(apply_dp_noise=True)
        opt_result = opt_future.result()

        # Track metrics
        epoch_loss += fb_result.loss
        num_batches += 1
        train_losses.append(fb_result.loss)

        # Update progress bar
        pbar.set_postfix({
            "loss": f"{fb_result.loss:.4f}",
            "step": tc.step,
        })

        # Log DP metrics
        if opt_result.dp_metrics and tc.step % 50 == 0:
            print(f"\n  DP epsilon: {opt_result.dp_metrics.total_epsilon:.4f}")

        # Periodic save
        if tc.step % save_every == 0:
            checkpoint = tc.save_state(
                metadata={"epoch": epoch, "step": tc.step}
            )
            print(f"\n  Saved checkpoint: {checkpoint.artifact_id}")

    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
```

## Step 6: Evaluation

```python
def evaluate(tc, eval_dataloader):
    """Evaluate model on validation set."""
    tc.eval()

    total_loss = 0
    num_batches = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        fb_result = tc.forward_backward(batch).result()
        total_loss += fb_result.loss
        num_batches += 1

    return total_loss / num_batches

# Run evaluation
val_loss = evaluate(tc, val_dataloader)
print(f"Validation loss: {val_loss:.4f}")

# Save best model
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_checkpoint = tc.save_state(
        metadata={
            "best": True,
            "val_loss": val_loss,
            "epoch": epoch,
        }
    )
    print(f"New best model: {best_checkpoint.artifact_id}")
```

## Step 7: Generate Text

```python
# Test generation
tc.eval()

prompts = [
    "### Instruction:\nExplain what machine learning is.\n\n### Response:",
    "### Instruction:\nWrite a Python function to calculate factorial.\n\n### Response:",
]

result = tc.sample(
    prompts=prompts,
    max_tokens=256,
    temperature=0.7,
    top_p=0.9,
    stop_sequences=["###", "\n\n\n"],
)

for sample in result.samples:
    print(f"=== Prompt ===\n{sample.prompt}\n")
    print(f"=== Completion ===\n{sample.completion}\n")
    print("-" * 50)
```

## Step 8: Final Checkpoint

```python
# Save final model
final_checkpoint = tc.save_state(
    include_optimizer=True,
    metadata={
        "final": True,
        "epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "total_steps": tc.step,
    }
)

print(f"\nTraining complete!")
print(f"Final checkpoint: {final_checkpoint.artifact_id}")
print(f"Size: {final_checkpoint.size_bytes / 1024 / 1024:.2f} MB")
```

## Complete Script

```python
#!/usr/bin/env python
"""Complete LoRA fine-tuning script."""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tg_tinker import (
    ServiceClient,
    TrainingConfig,
    LoRAConfig,
    OptimizerConfig,
    DPConfig,
    DPBudgetExceededError,
)


def main():
    # Load and prepare data
    dataset = load_dataset("tatsu-lab/alpaca", split="train[:5000]")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token

    def format_and_tokenize(example):
        if example["input"]:
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"

        tokens = tokenizer(text, truncation=True, max_length=2048, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = dataset.map(format_and_tokenize)
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Split
    train_size = int(0.9 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, len(dataset) - train_size])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    # Configure training
    config = TrainingConfig(
        model_ref="meta-llama/Llama-3-8B",
        lora_config=LoRAConfig(
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        optimizer=OptimizerConfig(
            learning_rate=2e-4,
            warmup_steps=100,
            lr_scheduler="cosine",
        ),
        dp_config=DPConfig(target_epsilon=8.0),
        batch_size=8,
        gradient_accumulation_steps=4,
    )

    # Train
    service = ServiceClient()
    tc = service.create_training_client(config)

    try:
        for epoch in range(3):
            tc.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                batch_dict = {k: v.tolist() for k, v in batch.items()}
                tc.forward_backward(batch_dict).result()
                tc.optim_step().result()

            # Evaluate
            tc.eval()
            val_loss = sum(
                tc.forward_backward({k: v.tolist() for k, v in b.items()}).result().loss
                for b in val_loader
            ) / len(val_loader)
            print(f"Epoch {epoch + 1} val_loss: {val_loss:.4f}")

    except DPBudgetExceededError:
        print("DP budget exceeded, saving checkpoint...")

    finally:
        checkpoint = tc.save_state()
        print(f"Saved: {checkpoint.artifact_id}")


if __name__ == "__main__":
    main()
```

## Using Custom Loss Functions

### Built-in Losses

```python
from tensafe.training.losses import resolve_loss

# Token cross-entropy with label smoothing
loss_fn = resolve_loss("token_ce", ignore_index=-100, label_smoothing=0.1)
```

### Custom Focal Loss for Imbalanced Data

```python
from tensafe.training.losses import register_loss
import torch.nn.functional as F

@register_loss("focal_ce")
def focal_cross_entropy(outputs, batch, gamma=2.0, alpha=0.25, **kwargs):
    """Focal loss to handle class imbalance."""
    logits = outputs.logits.view(-1, outputs.logits.size(-1))
    labels = batch["labels"].view(-1)

    # Filter out ignored tokens
    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]

    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * ((1 - pt) ** gamma) * ce_loss

    return {
        "loss": focal.mean(),
        "metrics": {"focal_weight": ((1 - pt) ** gamma).mean().item()}
    }

# Configure training with custom loss
config = TrainingConfig(
    model_ref="meta-llama/Llama-3-8B",
    lora_config=LoRAConfig(rank=16, alpha=32),
    loss_config={"type": "focal_ce", "kwargs": {"gamma": 2.0}},
)
```

### Contrastive Loss for Embeddings

```python
from tensafe.training.losses import resolve_loss

# Useful for embedding fine-tuning
loss_fn = resolve_loss("contrastive", temperature=0.07, reduction="mean")
```

## Tips

### Memory Optimization

```python
config = TrainingConfig(
    gradient_checkpointing=True,  # Trade compute for memory
    mixed_precision="bf16",       # Reduce memory
    batch_size=4,                 # Smaller batches
    gradient_accumulation_steps=8, # Maintain effective batch size
)
```

### Hyperparameter Search

```python
for rank in [8, 16, 32]:
    for lr in [1e-4, 2e-4, 5e-4]:
        config = TrainingConfig(
            lora_config=LoRAConfig(rank=rank),
            optimizer=OptimizerConfig(learning_rate=lr),
        )
        tc = service.create_training_client(config)
        # Train and evaluate...
```

### Resume Training

```python
# Resume from checkpoint
tc = service.create_training_client(
    config,
    checkpoint_id="artifact-id-here"
)
print(f"Resumed at step: {tc.step}")
```

## Next Steps

- [Privacy Budget Management](privacy-budget.md)
- [Encrypted Inference](encrypted-inference.md)
- [API Reference](../api-reference/training-client.md)
