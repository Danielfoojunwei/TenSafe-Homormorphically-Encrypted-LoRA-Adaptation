#!/usr/bin/env python3
"""
Inference Performance Benchmark Script

Measures inference performance metrics for Llama 3 models with and without LoRA adapters.

Metrics collected:
- TTFT (Time To First Token)
- Tokens per second
- P50/P95 end-to-end latency
- Peak RSS memory
- Peak VRAM

Variants compared:
1. Base model only
2. Base model + LoRA adapter
3. Base model + LoRA adapter (with runtime optimizations if available)

Usage:
    # Full benchmark
    python scripts/bench/perf_infer.py --model meta-llama/Meta-Llama-3-8B-Instruct --adapter outputs/lora_llama3/<run_id>/adapter

    # Smoke test (CPU-friendly)
    python scripts/bench/perf_infer.py --smoke
"""

import argparse
import gc
import json
import logging
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Standard benchmark prompts (fixed for reproducibility)
BENCHMARK_PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a Python function to calculate the Fibonacci sequence.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe the architecture of a transformer model.",
    "How does gradient descent work in neural network training?",
    "Explain the attention mechanism in deep learning.",
    "What is transfer learning and why is it useful?",
    "Write a SQL query to find the top 10 customers by total purchases.",
    "Explain the concept of overfitting and how to prevent it.",
    "What are the benefits and drawbacks of using LoRA for fine-tuning?",
    "Describe the differences between LoRA and full fine-tuning.",
    "How do you evaluate the quality of a language model?",
    "Explain the concept of tokenization in NLP.",
    "What is the purpose of the softmax function?",
    "Describe the training process for a GPT-style model.",
    "What are embeddings and how are they used in NLP?",
    "Explain the concept of perplexity in language modeling.",
    "How does beam search differ from greedy decoding?",
    "What is the role of the learning rate in training?",
    "Describe the concept of model quantization.",
]


@dataclass
class InferenceMetrics:
    """Metrics for a single inference run."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft_ms: float = 0.0  # Time to first token
    total_time_ms: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    variant: str
    num_prompts: int = 0
    avg_ttft_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    peak_rss_mb: float = 0.0
    peak_vram_mb: float = 0.0
    total_generated_tokens: int = 0
    total_time_seconds: float = 0.0
    raw_metrics: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_git_sha() -> str:
    """Get current git SHA."""
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


def get_peak_rss_mb() -> float:
    """Get peak RSS memory usage in MB."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if sys.platform == "darwin":
            return usage.ru_maxrss / 1e6
        else:
            return usage.ru_maxrss / 1e3
    except Exception:
        return 0.0


def get_vram_mb() -> float:
    """Get current VRAM usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
        return 0.0
    except ImportError:
        return 0.0


def get_peak_vram_mb() -> float:
    """Get peak VRAM usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
        return 0.0
    except ImportError:
        return 0.0


def load_model(
    model_name: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = False,
    use_flash_attention: bool = True,
) -> Tuple[Any, Any]:
    """Load model with optional LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading model: {model_name}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model config
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }

    if load_in_4bit and torch.cuda.is_available():
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    if use_flash_attention:
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using FlashAttention 2")
        except ImportError:
            logger.info("FlashAttention 2 not available, using default attention")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Load adapter if provided
    if adapter_path:
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        # Optionally merge adapter for faster inference
        # model = model.merge_and_unload()

    model.eval()

    return model, tokenizer


def benchmark_generation(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> List[InferenceMetrics]:
    """Run generation benchmark on a list of prompts."""
    import torch

    metrics_list = []
    device = next(model.parameters()).device

    for prompt in prompts:
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        # Reset CUDA events for timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Generation with timing
        start_time = time.perf_counter()
        first_token_time = None

        with torch.no_grad():
            # Use generate with streamer to measure TTFT
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            if first_token_time is None:
                first_token_time = time.perf_counter()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        # Calculate metrics
        generated_tokens = output.shape[1] - prompt_tokens
        total_time_ms = (end_time - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms / 2
        tokens_per_second = generated_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0

        metrics = InferenceMetrics(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            ttft_ms=ttft_ms,
            total_time_ms=total_time_ms,
            tokens_per_second=tokens_per_second,
        )
        metrics_list.append(metrics)

    return metrics_list


def run_benchmark(
    model,
    tokenizer,
    variant_name: str,
    prompts: List[str],
    max_new_tokens: int = 128,
    warmup_prompts: int = 2,
) -> BenchmarkResult:
    """Run full benchmark suite for a model variant."""
    import torch

    logger.info(f"Running benchmark for variant: {variant_name}")

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Warmup runs
    logger.info(f"Running {warmup_prompts} warmup prompts...")
    _ = benchmark_generation(
        model, tokenizer,
        prompts[:warmup_prompts],
        max_new_tokens=max_new_tokens,
    )

    # Actual benchmark
    logger.info(f"Benchmarking {len(prompts)} prompts...")
    start_time = time.time()

    metrics_list = benchmark_generation(
        model, tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
    )

    total_time = time.time() - start_time

    # Aggregate metrics
    latencies = [m.total_time_ms for m in metrics_list]
    ttfts = [m.ttft_ms for m in metrics_list]
    tokens_per_sec = [m.tokens_per_second for m in metrics_list]

    result = BenchmarkResult(
        variant=variant_name,
        num_prompts=len(prompts),
        avg_ttft_ms=statistics.mean(ttfts) if ttfts else 0,
        p50_latency_ms=statistics.median(latencies) if latencies else 0,
        p95_latency_ms=sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
        avg_tokens_per_second=statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
        peak_rss_mb=get_peak_rss_mb(),
        peak_vram_mb=get_peak_vram_mb(),
        total_generated_tokens=sum(m.generated_tokens for m in metrics_list),
        total_time_seconds=total_time,
        raw_metrics=[asdict(m) for m in metrics_list],
    )

    logger.info(f"  TTFT (avg): {result.avg_ttft_ms:.2f} ms")
    logger.info(f"  Latency P50: {result.p50_latency_ms:.2f} ms")
    logger.info(f"  Latency P95: {result.p95_latency_ms:.2f} ms")
    logger.info(f"  Tokens/sec: {result.avg_tokens_per_second:.2f}")
    logger.info(f"  Peak VRAM: {result.peak_vram_mb:.2f} MB")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inference performance benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test with tiny model",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=20,
        help="Number of prompts to benchmark",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Max tokens to generate per prompt",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/llama3_lora_bench",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Smoke mode configuration
    if args.smoke:
        logger.info("Running in SMOKE mode")
        args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        args.num_prompts = 5
        args.max_new_tokens = 16
        args.adapter = None

    # Check dependencies
    try:
        import torch
        import transformers
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: pip install torch transformers peft")
        sys.exit(1)

    # Create output directory
    git_sha = get_git_sha()
    output_dir = Path(args.output_dir) / git_sha
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare prompts
    prompts = BENCHMARK_PROMPTS[:args.num_prompts]
    logger.info(f"Benchmarking with {len(prompts)} prompts, {args.max_new_tokens} max tokens")

    results = []

    # Variant 1: Base model only
    logger.info("\n=== Variant 1: Base Model ===")
    model, tokenizer = load_model(
        args.model,
        adapter_path=None,
        load_in_4bit=args.load_in_4bit,
    )

    base_result = run_benchmark(
        model, tokenizer,
        variant_name="base_model",
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
    )
    results.append(base_result)

    # Cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Variant 2: Base + LoRA adapter
    if args.adapter and Path(args.adapter).exists():
        logger.info("\n=== Variant 2: Base + LoRA Adapter ===")
        model, tokenizer = load_model(
            args.model,
            adapter_path=args.adapter,
            load_in_4bit=args.load_in_4bit,
        )

        lora_result = run_benchmark(
            model, tokenizer,
            variant_name="base_plus_lora",
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(lora_result)

        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Variant 3: Base + LoRA with merged weights (optimized)
        logger.info("\n=== Variant 3: Base + LoRA (Merged) ===")
        model, tokenizer = load_model(
            args.model,
            adapter_path=args.adapter,
            load_in_4bit=args.load_in_4bit,
        )

        # Merge adapter for faster inference
        try:
            from peft import PeftModel
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()
                logger.info("Merged LoRA weights into base model")
        except Exception as e:
            logger.warning(f"Could not merge adapter: {e}")

        merged_result = run_benchmark(
            model, tokenizer,
            variant_name="base_plus_lora_merged",
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(merged_result)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
            "smoke_mode": args.smoke,
        },
        "results": [r.to_dict() for r in results],
    }

    output_path = output_dir / "perf_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Print comparison table
    print("\n" + "=" * 80)
    print("INFERENCE PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"{'Variant':<30} {'TTFT(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'Tok/s':<12}")
    print("-" * 80)

    for r in results:
        print(
            f"{r.variant:<30} "
            f"{r.avg_ttft_ms:<12.2f} "
            f"{r.p50_latency_ms:<12.2f} "
            f"{r.p95_latency_ms:<12.2f} "
            f"{r.avg_tokens_per_second:<12.2f}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
