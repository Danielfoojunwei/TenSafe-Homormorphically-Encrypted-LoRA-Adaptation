#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Llama 3

Trains a LoRA adapter on the Llama 3 model using the OASST1 dataset.

Features:
- LoRA with configurable rank, alpha, dropout
- QLoRA support for memory-constrained GPUs
- FlashAttention 2 support when available
- Comprehensive logging and metrics
- Reproducible training with seed control

Base model: meta-llama/Meta-Llama-3-8B-Instruct
Dataset: OpenAssistant/oasst1 (Apache-2.0)

Usage:
    # Full training (requires GPU + HF token)
    python scripts/lora/train_lora_llama3.py --model meta-llama/Meta-Llama-3-8B-Instruct

    # Smoke test (CPU-friendly, tiny model)
    python scripts/lora/train_lora_llama3.py --smoke

    # QLoRA for limited VRAM
    python scripts/lora/train_lora_llama3.py --qlora
"""

import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts/lora"))


@dataclass
class LoRAConfig:
    """LoRA hyperparameter configuration."""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha (scaling)
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])
    # Optional: include gate/up/down projections for more capacity
    include_mlp: bool = False
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training hyperparameter configuration."""
    # Model
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    use_flash_attention: bool = True
    load_in_4bit: bool = False  # QLoRA
    load_in_8bit: bool = False

    # Data
    max_seq_length: int = 2048
    packing: bool = False  # Sequence packing

    # Training
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    # Precision
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3

    # Reproducibility
    seed: int = 42
    data_seed: int = 42

    # Output
    output_dir: str = "outputs/lora_llama3"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    total_steps: int = 0
    train_loss: float = 0.0
    eval_loss: Optional[float] = None
    tokens_per_second: float = 0.0
    step_time_ms: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    train_samples: int = 0
    eval_samples: int = 0
    total_train_time_seconds: float = 0.0
    loss_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return "unknown"


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware and environment information."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "python_version": sys.version,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "gpu_memory_total_gb": None,
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    except ImportError:
        info["torch_version"] = "not installed"

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "not installed"

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        info["peft_version"] = "not installed"

    return info


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    required = ["torch", "transformers", "peft", "datasets", "trl"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        logger.error(f"Missing required packages: {missing}")
        logger.error("Install with: pip install torch transformers peft datasets trl")
        return False

    return True


def load_model_and_tokenizer(
    config: TrainingConfig,
    lora_config: LoRAConfig,
):
    """Load the base model and tokenizer with optional quantization."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {config.model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config for QLoRA
    quantization_config = None
    if config.load_in_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif config.load_in_8bit:
        logger.info("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Model loading kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32),
    }

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    # FlashAttention 2 - only use if explicitly enabled AND package is available
    if config.use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention 2")
        except ImportError:
            logger.warning("FlashAttention 2 not installed, using default attention")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs,
    )

    # Prepare for k-bit training if quantized
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    # Get target modules
    target_modules = list(lora_config.target_modules)
    if lora_config.include_mlp:
        target_modules.extend(["gate_proj", "up_proj", "down_proj"])

    # LoRA config
    peft_config = LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=target_modules,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    # Apply LoRA
    model = get_peft_model(model, peft_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%) "
        f"| Total: {total_params:,}"
    )

    return model, tokenizer


def prepare_dataset(
    tokenizer,
    config: TrainingConfig,
    smoke: bool = False,
    synthetic: bool = False,
):
    """Prepare the training and evaluation datasets."""
    from dataset_loader import (
        ChatExample,
        DatasetConfig,
        OASST1Loader,
        create_synthetic_dataset,
    )
    from datasets import Dataset

    if synthetic or smoke:
        logger.info("Creating synthetic dataset for smoke test")
        n_train = 50 if smoke else 500
        n_eval = 10 if smoke else 50
        train_examples, eval_examples = create_synthetic_dataset(
            n_train=n_train,
            n_eval=n_eval,
            seed=config.data_seed,
        )
    else:
        logger.info("Loading OASST1 dataset")
        dataset_config = DatasetConfig(
            seed=config.data_seed,
            max_seq_length=config.max_seq_length,
        )
        loader = OASST1Loader(dataset_config)

        if smoke:
            train_examples, eval_examples = loader.get_smoke_subset(
                n_train=100,
                n_eval=20,
            )
        else:
            train_examples, eval_examples = loader.preprocess()

    # Convert to HF dataset format
    def format_example(example: ChatExample) -> Dict[str, str]:
        return {"text": example.to_llama3_format()}

    train_data = [format_example(ex) for ex in train_examples]
    eval_data = [format_example(ex) for ex in eval_examples]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    logger.info(f"Train dataset: {len(train_dataset)} examples")
    logger.info(f"Eval dataset: {len(eval_dataset)} examples")

    return train_dataset, eval_dataset


try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object


class TrainingMetricsCallback(TrainerCallback):
    """Callback to collect training metrics (inherits from TrainerCallback)."""

    def __init__(self):
        super().__init__()
        self.metrics = TrainingMetrics()
        self.step_times = []
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.metrics.loss_history.append(logs["loss"])
            if "train_loss" in logs:
                self.metrics.train_loss = logs["train_loss"]

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time:
            self.metrics.total_train_time_seconds = time.time() - self.start_time

        # Calculate average step time
        if self.step_times:
            self.metrics.step_time_ms = sum(self.step_times) / len(self.step_times) * 1000

        self.metrics.total_steps = state.global_step if state else 0


def train(
    config: TrainingConfig,
    lora_config: LoRAConfig,
    smoke: bool = False,
    synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Run LoRA training.

    Returns:
        Dictionary with training results and artifact paths
    """
    import torch
    from trl import SFTConfig, SFTTrainer

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect hardware info
    hw_info = get_hardware_info()

    # Save run config
    run_config = {
        "run_id": run_id,
        "training_config": config.to_dict(),
        "lora_config": lora_config.to_dict(),
        "hardware_info": hw_info,
        "smoke_mode": smoke,
        "synthetic_data": synthetic,
    }

    config_path = output_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"Run ID: {run_id}")
    logger.info(f"Output directory: {output_dir}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config, lora_config)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_dataset(
        tokenizer, config, smoke=smoke, synthetic=synthetic
    )

    # SFT Config (TRL 0.27+ uses SFTConfig instead of TrainingArguments)
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and torch.cuda.is_available() and not config.bf16,
        tf32=config.tf32 and torch.cuda.is_available(),
        logging_steps=config.logging_steps,
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        data_seed=config.data_seed,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
        optim="adamw_torch",
        # SFT-specific settings
        max_length=config.max_seq_length,
        packing=config.packing,
        dataset_text_field="text",
    )

    # Metrics callback
    metrics_callback = TrainingMetricsCallback()

    # Start timing
    train_start = time.time()

    # Create trainer (TRL 0.27+ uses simplified API)
    # SFTTrainer handles tokenization and data collation internally
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        processing_class=tokenizer,
    )

    # Add callback
    trainer.add_callback(metrics_callback)

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Calculate metrics
    train_time = time.time() - train_start
    metrics = TrainingMetrics(
        total_steps=trainer.state.global_step,
        train_loss=train_result.training_loss,
        train_samples=len(train_dataset),
        eval_samples=len(eval_dataset),
        total_train_time_seconds=train_time,
        loss_history=metrics_callback.metrics.loss_history,
    )

    # Calculate tokens/sec
    total_tokens = (
        len(train_dataset) *
        config.max_seq_length *
        config.num_train_epochs
    )
    metrics.tokens_per_second = total_tokens / train_time if train_time > 0 else 0

    # GPU memory
    if torch.cuda.is_available():
        metrics.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1e6

    # Run evaluation if dataset available
    if len(eval_dataset) > 0:
        logger.info("Running evaluation...")
        eval_result = trainer.evaluate()
        metrics.eval_loss = eval_result.get("eval_loss")

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info(f"Saved adapter to {adapter_dir}")

    # Save metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)

    # Save trainer state
    trainer_state_path = output_dir / "trainer_state.json"
    with open(trainer_state_path, "w") as f:
        json.dump({
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "best_metric": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
        }, f, indent=2)

    # Cleanup
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "trainer_state_path": str(trainer_state_path),
        "metrics": metrics.to_dict(),
    }

    logger.info(f"Training complete! Results saved to {output_dir}")
    logger.info(f"  Train loss: {metrics.train_loss:.4f}")
    logger.info(f"  Tokens/sec: {metrics.tokens_per_second:.2f}")
    logger.info(f"  Total time: {metrics.total_train_time_seconds:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Llama 3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test with tiny model and synthetic data",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic dataset instead of OASST1",
    )

    # LoRA args
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--include-mlp", action="store_true", help="Include MLP layers in LoRA")

    # QLoRA args
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA")
    parser.add_argument("--load-in-8bit", action="store_true", help="Use 8-bit quantization")

    # Training args
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length")

    # Precision args
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 precision")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable FlashAttention")

    # Output args
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/lora_llama3",
        help="Output directory",
    )

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Configure smoke mode
    if args.smoke:
        logger.info("Running in SMOKE mode with synthetic data")
        args.synthetic = True
        args.epochs = 1
        args.batch_size = 2
        args.grad_accum = 1
        args.max_seq_length = 512
        # Use a tiny open model for smoke tests
        if args.model == "meta-llama/Meta-Llama-3-8B-Instruct":
            args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    # Build configs
    lora_config = LoRAConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        include_mlp=args.include_mlp,
    )

    training_config = TrainingConfig(
        model_name=args.model,
        use_flash_attention=not args.no_flash_attn,
        load_in_4bit=args.qlora,
        load_in_8bit=args.load_in_8bit,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=not args.no_bf16 and not args.fp16,
        fp16=args.fp16,
        seed=args.seed,
        data_seed=args.seed,
        output_dir=args.output_dir,
    )

    # Run training
    results = train(
        config=training_config,
        lora_config=lora_config,
        smoke=args.smoke,
        synthetic=args.synthetic,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Run ID: {results['run_id']}")
    print(f"Output: {results['output_dir']}")
    print(f"Adapter: {results['adapter_dir']}")
    print(f"Train Loss: {results['metrics']['train_loss']:.4f}")
    print(f"Tokens/sec: {results['metrics']['tokens_per_second']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
