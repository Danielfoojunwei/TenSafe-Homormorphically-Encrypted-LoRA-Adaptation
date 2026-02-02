#!/usr/bin/env python3
"""
TenSafe Unified CLI.

Provides a single entry point for all TenSafe operations:
- Training (SFT, RLVR)
- Inference
- Configuration management
- Backend verification
- Health checks

Usage:
    # Training
    tensafe train --config config.yaml
    tensafe train --mode sft --model meta-llama/Llama-3.1-8B

    # Inference
    tensafe inference --model ./checkpoint --prompt "Hello"

    # Configuration
    tensafe config create --output config.yaml
    tensafe config validate config.yaml

    # Backend verification
    tensafe verify --backend hexl

    # Production check
    tensafe production-check --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tensafe")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tensafe",
        description="TenSafe: Privacy-first ML Training Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SFT training
  tensafe train --config train_config.yaml

  # Run RLVR training
  tensafe train --mode rlvr --config rlvr_config.yaml

  # Create default configuration
  tensafe config create --mode sft --output my_config.yaml

  # Validate configuration
  tensafe config validate my_config.yaml

  # Verify HE backend
  tensafe verify --backend hexl

  # Run production checks
  tensafe production-check --config prod_config.yaml
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 3.0.0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be repeated)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Run training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_train_args(train_parser)

    # Inference command
    inference_parser = subparsers.add_parser(
        "inference",
        help="Run inference",
    )
    _add_inference_args(inference_parser)

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")

    config_create = config_subparsers.add_parser("create", help="Create configuration")
    _add_config_create_args(config_create)

    config_validate = config_subparsers.add_parser("validate", help="Validate configuration")
    config_validate.add_argument("config_path", type=str, help="Path to configuration file")

    config_show = config_subparsers.add_parser("show", help="Show configuration")
    config_show.add_argument("config_path", type=str, help="Path to configuration file")

    # Verify command
    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify backends and dependencies",
    )
    _add_verify_args(verify_parser)

    # Production check command
    prod_check_parser = subparsers.add_parser(
        "production-check",
        help="Run production readiness checks",
    )
    prod_check_parser.add_argument(
        "--config",
        type=str,
        help="Configuration file to check",
    )

    # Parse args
    args = parser.parse_args()

    # Set verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Route to command handler
    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "train": _handle_train,
        "inference": _handle_inference,
        "config": _handle_config,
        "verify": _handle_verify,
        "production-check": _handle_production_check,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Command failed: {e}")
        return 1


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training arguments."""
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "rlvr", "dpo"],
        default="sft",
        help="Training mode (default: sft)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Total training steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--dp-epsilon",
        type=float,
        help="Target differential privacy epsilon",
    )
    parser.add_argument(
        "--he-mode",
        type=str,
        choices=["disabled", "production"],
        default="disabled",
        help="Homomorphic encryption mode (production requires HE-LoRA microkernel)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training",
    )


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Add inference arguments."""
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or checkpoint path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Input prompt",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File with prompts (one per line)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for generations",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--lora-mode",
        type=str,
        choices=["none", "plaintext", "he_only"],
        default="plaintext",
        help="LoRA inference mode",
    )


def _add_config_create_args(parser: argparse.ArgumentParser) -> None:
    """Add config create arguments."""
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "rlvr"],
        default="sft",
        help="Training mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name",
    )
    parser.add_argument(
        "--with-dp",
        action="store_true",
        help="Enable differential privacy",
    )
    parser.add_argument(
        "--with-he",
        action="store_true",
        help="Enable homomorphic encryption",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tensafe_config.yaml",
        help="Output file path",
    )


def _add_verify_args(parser: argparse.ArgumentParser) -> None:
    """Add verify arguments."""
    parser.add_argument(
        "--backend",
        type=str,
        choices=["all", "production"],
        default="all",
        help="Backend to verify (production uses HE-LoRA microkernel)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification (skip benchmarks)",
    )


def _handle_train(args: argparse.Namespace) -> int:
    """Handle train command."""
    from tensafe.core.config import (
        TenSafeConfig,
        TrainingConfig,
        ModelConfig,
        LoRAConfig,
        DPConfig,
        HEConfig,
        TrainingMode,
        HEMode,
        load_config,
    )
    from tensafe.core.pipeline import TenSafePipeline

    # Load or create config
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        # Create config from CLI args
        config = TenSafeConfig(
            model=ModelConfig(name=args.model or "meta-llama/Llama-3.1-8B"),
            lora=LoRAConfig(rank=args.lora_rank),
            training=TrainingConfig(
                mode=TrainingMode(args.mode),
                total_steps=args.steps or 1000,
                learning_rate=args.lr or 1e-4,
                batch_size=args.batch_size or 4,
                output_dir=args.output_dir,
            ),
            dp=DPConfig(
                enabled=args.dp_epsilon is not None,
                target_epsilon=args.dp_epsilon or 8.0,
            ),
            he=HEConfig(mode=HEMode(args.he_mode)),
            dry_run=args.dry_run,
        )

    # Validate config
    issues = config.validate()
    for issue in issues:
        if issue.startswith("Error:"):
            logger.error(issue)
            return 1
        logger.warning(issue)

    if args.dry_run:
        logger.info("Dry run: configuration is valid")
        print("\nConfiguration:")
        print(json.dumps(config.to_dict(), indent=2))
        return 0

    # Create and run pipeline
    logger.info(f"Starting {args.mode} training...")
    pipeline = TenSafePipeline(config)
    pipeline.setup()

    result = pipeline.train()

    if result.success:
        logger.info(f"Training completed: {result.total_steps} steps, final loss: {result.final_loss:.4f}")
        logger.info(f"Training time: {result.training_time_seconds:.1f}s")
        return 0
    else:
        logger.error(f"Training failed: {result.errors}")
        return 1


def _handle_inference(args: argparse.Namespace) -> int:
    """Handle inference command."""
    from tensafe.core.inference import (
        TenSafeInference,
        InferenceMode,
        GenerationConfig,
    )

    logger.info(f"Loading model: {args.model}")

    # Read prompts
    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.input_file:
        with open(args.input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        logger.error("Either --prompt or --input-file is required")
        return 1

    logger.info(f"Running inference on {len(prompts)} prompts...")

    # Determine inference mode
    mode = InferenceMode(args.lora_mode)

    # Create generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    # Load inference engine
    try:
        model_path = Path(args.model)
        if model_path.exists() and model_path.is_dir():
            # Load from checkpoint
            inference = TenSafeInference.from_checkpoint(
                checkpoint_path=model_path,
                mode=mode,
            )
        else:
            # Load from model name (HuggingFace)
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                logger.info(f"Loading model from HuggingFace: {args.model}")
                tokenizer = AutoTokenizer.from_pretrained(args.model)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )

                inference = TenSafeInference(
                    model=model,
                    tokenizer=tokenizer,
                    mode=mode,
                )
            except ImportError as e:
                logger.error(f"PyTorch/Transformers required for model loading: {e}")
                return 1
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Run inference
    outputs = []
    for prompt in prompts:
        try:
            result = inference.generate(prompt, generation_config=gen_config)
            output = result.text
            outputs.append(output)
            print(f"\nPrompt: {prompt}")
            print(f"Output: {output}")
            if result.tokens_per_second > 0:
                logger.info(f"Generated {len(result.tokens)} tokens at {result.tokens_per_second:.1f} tok/s")
        except RuntimeError as e:
            logger.error(f"Generation failed for prompt: {e}")
            outputs.append(f"[Error: {e}]")

    # Write outputs if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            for output in outputs:
                f.write(output + "\n")
        logger.info(f"Outputs written to {args.output_file}")

    return 0


def _handle_config(args: argparse.Namespace) -> int:
    """Handle config command."""
    if args.config_command == "create":
        return _handle_config_create(args)
    elif args.config_command == "validate":
        return _handle_config_validate(args)
    elif args.config_command == "show":
        return _handle_config_show(args)
    else:
        logger.error("Subcommand required: create, validate, or show")
        return 1


def _handle_config_create(args: argparse.Namespace) -> int:
    """Handle config create command."""
    from tensafe.core.config import (
        TenSafeConfig,
        TrainingMode,
        create_default_config,
        save_config,
    )

    config = create_default_config(
        mode=TrainingMode(args.mode),
        model_name=args.model,
        with_dp=args.with_dp,
        with_he=args.with_he,
    )

    save_config(config, args.output)
    logger.info(f"Configuration saved to {args.output}")

    return 0


def _handle_config_validate(args: argparse.Namespace) -> int:
    """Handle config validate command."""
    from tensafe.core.config import load_config
    from tensafe.core.gates import production_check

    try:
        config = load_config(args.config_path, validate=False)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Run validation
    issues = config.validate()
    result = production_check(config)

    has_errors = False

    for issue in issues:
        if issue.startswith("Error:"):
            logger.error(issue)
            has_errors = True
        else:
            logger.warning(issue)

    for error in result.errors:
        logger.error(f"Production error: {error}")
        has_errors = True

    for warning in result.warnings:
        logger.warning(f"Production warning: {warning}")

    if has_errors:
        logger.error("Configuration validation FAILED")
        return 1

    logger.info("Configuration is valid")
    return 0


def _handle_config_show(args: argparse.Namespace) -> int:
    """Handle config show command."""
    from tensafe.core.config import load_config

    try:
        config = load_config(args.config_path, validate=False)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    print(json.dumps(config.to_dict(), indent=2))
    return 0


def _handle_verify(args: argparse.Namespace) -> int:
    """Handle verify command."""
    import numpy as np  # Import here to avoid requiring numpy for other commands

    print("TenSafe Backend Verification")
    print("=" * 50)

    results = {}

    if args.backend in ("all", "production"):
        print("\nVerifying HE-LoRA Microkernel (Production) backend...")
        try:
            from he_lora_microkernel.compat import HEBackend
            backend = HEBackend()
            backend.setup()
            print(f"  [OK] Production backend operational")
            print(f"       Slot count: {backend.get_slot_count()}")
            results["production"] = "OK"
        except ImportError as e:
            print(f"  [SKIP] Production backend not installed: {e}")
            print("         Install with: pip install he_lora_microkernel")
            results["production"] = "NOT_INSTALLED"
        except Exception as e:
            print(f"  [FAIL] Production backend: {e}")
            results["production"] = f"FAIL: {e}"

    print("\n" + "=" * 50)
    print("Summary:")
    for name, status in results.items():
        emoji = "[OK]" if status == "OK" else "[--]" if "NOT_INSTALLED" in status else "[XX]"
        print(f"  {emoji} {name}: {status}")

    return 0 if all(s == "OK" or "NOT_INSTALLED" in s for s in results.values()) else 1


def _handle_production_check(args: argparse.Namespace) -> int:
    """Handle production check command."""
    from tensafe.core.gates import production_check, ProductionGates

    print("TenSafe Production Readiness Check")
    print("=" * 50)

    # Load config if provided
    config = None
    if args.config:
        from tensafe.core.config import load_config
        try:
            config = load_config(args.config, validate=False)
            print(f"\nConfiguration: {args.config}")
        except Exception as e:
            print(f"\n[FAIL] Cannot load config: {e}")
            return 1

    # Run production check
    result = production_check(config)

    # Check all gates
    print("\nFeature Gates:")
    gate_statuses = ProductionGates.check_all()
    for name, status in gate_statuses.items():
        emoji = "[OK]" if status.value == "allowed" else "[--]" if status.value == "denied" else "[!!]"
        print(f"  {emoji} {name}: {status.value}")

    # Show errors/warnings
    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  [XX] {error}")

    if result.warnings:
        print("\nWarnings:")
        for warning in result.warnings:
            print(f"  [!!] {warning}")

    print("\n" + "=" * 50)
    if result.valid:
        print("Result: PRODUCTION READY")
        return 0
    else:
        print("Result: NOT PRODUCTION READY")
        return 1


if __name__ == "__main__":
    sys.exit(main())
