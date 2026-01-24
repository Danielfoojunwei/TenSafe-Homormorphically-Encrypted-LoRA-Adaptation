"""
HuggingFace Training Connector - Production Hardened

Provides PEFT/LoRA fine-tuning via HuggingFace Transformers.
In production mode, requires torch/transformers/peft to be installed.
"""

import logging
import os
import json
from typing import Dict, Any, Optional, Callable
from ..contracts import Connector, ConnectorValidationResult
from ..catalog import ConnectorCatalog, is_package_installed
from ....utils.production_gates import is_production, ProductionGateError

logger = logging.getLogger(__name__)


class HuggingFaceTrainingConnector(Connector):
    """Connector for HuggingFace PEFT training."""

    @property
    def id(self) -> str:
        return "training_hf"

    @property
    def name(self) -> str:
        return "HF Transformers + PEFT (Native)"

    @property
    def category(self) -> str:
        return "training"

    def check_installed(self) -> bool:
        """Check for torch, transformers, and peft."""
        return (
            is_package_installed("torch")
            and is_package_installed("transformers")
            and is_package_installed("peft")
        )

    def validate_config(self, config: Dict[str, Any]) -> ConnectorValidationResult:
        """Validate training configuration."""
        model_path = config.get("model_name_or_path")
        if not model_path:
            return ConnectorValidationResult(
                ok=False,
                details="Missing model_name_or_path",
                remediation="Specify a model ID or local path.",
            )

        # Check if local path exists if it's not a HF hub ID
        if "/" in model_path and os.path.exists(model_path):
            # Local path - verify it's a valid model directory
            if not os.path.isdir(model_path):
                return ConnectorValidationResult(
                    ok=False,
                    details=f"Model path is not a directory: {model_path}",
                    remediation="Provide a valid model directory path.",
                )

        return ConnectorValidationResult(ok=True, details="Config valid.")

    def to_runtime(self, config: Dict[str, Any]) -> Any:
        """
        Returns a Trainer instance.

        In production: requires torch/transformers/peft and returns RealTrainer.
        In development: falls back to DemoTrainer if deps missing.
        """
        if self.check_installed():
            # Dependencies available - use real trainer
            return RealTrainer(config)

        # Dependencies missing
        if is_production():
            raise ProductionGateError(
                gate_name="PEFT_DEPENDENCIES",
                message="PEFT training requires torch, transformers, and peft packages in production.",
                remediation="Install required packages: pip install torch transformers peft",
            )

        # Development mode only
        logger.warning(
            "PEFT dependencies (torch/transformers/peft) missing. "
            "Using DemoTrainer - NOT FOR PRODUCTION USE."
        )
        return DemoTrainer(config)


class RealTrainer:
    """
    Real PEFT trainer using HuggingFace Transformers and PEFT library.

    Performs actual model loading, fine-tuning, and artifact generation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration at initialization."""
        required = ["model_name_or_path", "output_dir"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def run(self, log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Execute PEFT training.

        Args:
            log_callback: Optional callback for progress logging

        Returns:
            Dict with status and metrics from actual training
        """
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig, get_peft_model, TaskType

        output_dir = self.config.get("output_dir", "./runs/latest/adapters")
        os.makedirs(output_dir, exist_ok=True)

        def _log(stage: str, progress: float, message: str) -> None:
            if log_callback:
                log_callback(
                    json.dumps(
                        {"stage": stage, "progress": progress, "message": message}
                    )
                )

        try:
            # Stage 1: Load model and tokenizer
            _log("LOADING_MODEL", 10, "Loading base model and tokenizer...")
            model_name = self.config["model_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            # Stage 2: Configure LoRA
            _log("CONFIGURING_LORA", 20, "Configuring LoRA adapter...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.get("lora_r", 8),
                lora_alpha=self.config.get("lora_alpha", 16),
                lora_dropout=self.config.get("lora_dropout", 0.05),
                target_modules=self.config.get(
                    "target_modules", ["q_proj", "v_proj"]
                ),
                bias="none",
            )

            model = get_peft_model(model, lora_config)
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in model.parameters())
            _log(
                "CONFIGURING_LORA",
                25,
                f"LoRA configured: {trainable_params:,} trainable / {total_params:,} total params",
            )

            # Stage 3: Prepare dataset
            _log("PREPARING_DATA", 30, "Preparing training dataset...")
            dataset_path = self.config.get("dataset_path")
            dataset_name = self.config.get("dataset_name")

            if dataset_path and os.path.exists(dataset_path):
                if os.path.isdir(dataset_path):
                    from datasets import load_from_disk
                    dataset = load_from_disk(dataset_path)
                else:
                    # Support single file datasets (json, jsonl, csv)
                    from datasets import load_dataset
                    ext = dataset_path.rsplit(".", 1)[-1]
                    if ext == "jsonl": ext = "json"
                    dataset = load_dataset(ext, data_files=dataset_path, split="train")
            elif dataset_name:
                from datasets import load_dataset

                dataset = load_dataset(dataset_name, split="train")
            else:
                raise ValueError(
                    "Either dataset_path or dataset_name must be provided"
                )

            # Tokenize dataset
            def tokenize_function(examples):
                text_column = self.config.get("text_column", "text")
                return tokenizer(
                    examples[text_column],
                    truncation=True,
                    max_length=self.config.get("max_length", 512),
                    padding="max_length",
                )

            tokenized_dataset = dataset.map(
                tokenize_function, batched=True, remove_columns=dataset.column_names
            )

            # Stage 4: Training
            _log("FINE_TUNING", 40, "Starting fine-tuning...")
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=self.config.get("num_epochs", 3),
                per_device_train_batch_size=self.config.get("batch_size", 4),
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps", 4
                ),
                learning_rate=self.config.get("learning_rate", 2e-4),
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                save_strategy="epoch",
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                report_to=[],  # Disable wandb/tensorboard unless configured
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )

            # Run training
            train_result = trainer.train()
            _log("FINE_TUNING", 80, "Training completed.")

            # Stage 5: Save adapter
            _log("SAVING", 90, "Saving LoRA adapter...")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Verify adapter files exist
            adapter_config_path = os.path.join(output_dir, "adapter_config.json")
            adapter_model_path = os.path.join(output_dir, "adapter_model.safetensors")
            if not os.path.exists(adapter_model_path):
                adapter_model_path = os.path.join(output_dir, "adapter_model.bin")

            if not os.path.exists(adapter_config_path):
                raise RuntimeError(
                    f"Adapter config not saved: {adapter_config_path}"
                )

            _log("SAVING", 100, "Adapter saved successfully.")

            # Extract real metrics
            metrics = {
                "loss": train_result.training_loss,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "param_efficiency": f"{100 * trainable_params / total_params:.2f}%",
                "epochs_completed": training_args.num_train_epochs,
                "samples_trained": len(tokenized_dataset),
            }

            return {"status": "ok", "metrics": metrics}

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {"status": "failed", "error": str(e)}


class DemoTrainer:
    """
    Demo trainer for development/testing when PEFT dependencies are unavailable.

    IMPORTANT: This trainer is BLOCKED in production mode.
    It creates placeholder artifacts and returns mock metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        # Block in production at initialization
        if is_production():
            raise ProductionGateError(
                gate_name="DEMO_TRAINER",
                message="DemoTrainer cannot be used in production mode.",
                remediation="Install PEFT dependencies: pip install torch transformers peft",
            )

        self.config = config

    def run(self, log_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Execute demo training (development only).

        Creates placeholder files for pipeline testing.
        """
        # Double-check production gate
        if is_production():
            raise ProductionGateError(
                gate_name="DEMO_TRAINER_RUN",
                message="DemoTrainer.run() cannot be executed in production.",
                remediation="Install PEFT dependencies: pip install torch transformers peft",
            )

        import time

        def _log(stage: str, progress: float, message: str) -> None:
            if log_callback:
                log_callback(
                    json.dumps(
                        {"stage": stage, "progress": progress, "message": message}
                    )
                )

        stages = [
            ("INIT", 10, "[DEMO] Initializing demo trainer..."),
            ("LOADING_MODEL", 25, "[DEMO] Simulating model load..."),
            ("CONFIGURING_LORA", 35, "[DEMO] Simulating LoRA config..."),
            ("PREPARING_DATA", 50, "[DEMO] Simulating data preparation..."),
            ("FINE_TUNING", 75, "[DEMO] Simulating fine-tuning..."),
            ("SAVING", 100, "[DEMO] Saving demo artifacts..."),
        ]

        output_dir = self.config.get("output_dir", "./runs/demo/adapters")
        os.makedirs(output_dir, exist_ok=True)

        for stage, progress, message in stages:
            _log(stage, progress, message)
            time.sleep(0.5)  # Brief pause for demo effect

        # Create demo artifacts (clearly marked as demo)
        demo_config = {
            "base_model": self.config.get("model_name_or_path", "unknown"),
            "peft_type": "LORA",
            "mode": "DEMO_MODE",
            "warning": "This is a demo artifact - not from real training",
        }

        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(demo_config, f, indent=2)

        # Create a placeholder adapter file (clearly marked)
        with open(os.path.join(output_dir, "adapter_model.demo.bin"), "w") as f:
            f.write("DEMO_ADAPTER_PLACEHOLDER_NOT_FOR_PRODUCTION")

        return {
            "status": "ok",
            "warning": "This was a demo run - not real training",
            "metrics": {
                "mode": "demo",
                "note": "No actual training was performed",
            },
        }


# Auto-register connector
ConnectorCatalog.register(HuggingFaceTrainingConnector())
