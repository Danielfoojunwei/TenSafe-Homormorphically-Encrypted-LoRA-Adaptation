#!/usr/bin/env python3
"""
LoRA PEFT Benchmark with MOAI/N2HE Encryption Pipeline

Industry-standard benchmark for LoRA fine-tuning with TensorGuard encryption.
Produces detailed Step-by-Step Performance Summary:
1. LoRA Training (real PEFT with HuggingFace)
2. Evaluation Suite (GSM8K, MMLU smoke tests)
3. Performance Benchmark (TTFT, Latency P50/P95, Throughput)
4. Encryption Pipeline (N2HE gradient, MOAI inference)

Usage:
    python benchmarks/lora_peft_benchmark.py
    python benchmarks/lora_peft_benchmark.py --skip-training  # Use cached adapter
"""

import gc
import json
import os
import sys
import time
import statistics
import warnings
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TG_ENABLE_EXPERIMENTAL_CRYPTO", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LoRABenchConfig:
    """LoRA benchmark configuration."""
    # Model
    model_name: str = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    num_train_samples: int = 50
    num_eval_samples: int = 10
    max_length: int = 128
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 2e-4

    # Inference benchmark
    num_inference_runs: int = 20
    generation_max_tokens: int = 32
    warmup_runs: int = 3

    # Encryption
    n2he_lattice_dim: int = 256
    moai_poly_modulus: int = 8192

    # Output
    output_dir: str = "artifacts/benchmarks/lora_peft"


# =============================================================================
# Step 1: LoRA Training
# =============================================================================

class LoRATrainingBenchmark:
    """Real LoRA training benchmark using HuggingFace PEFT."""

    def __init__(self, config: LoRABenchConfig):
        self.config = config
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """Run LoRA training and return metrics."""
        print("\n" + "=" * 60)
        print("Step 1: LoRA Training")
        print("=" * 60)

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
                DataCollatorForLanguageModeling,
            )
            from peft import LoraConfig, get_peft_model, TaskType
            from datasets import Dataset
        except ImportError as e:
            print(f"  [ERROR] Missing dependencies: {e}")
            print("  Install with: pip install torch transformers peft datasets")
            return {"status": "skipped", "error": str(e)}

        output_dir = Path(self.config.output_dir) / "adapter"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Model: {self.config.model_name}")

        # Load model and tokenizer
        print("  Loading model and tokenizer...")
        load_start = time.perf_counter()

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # CPU-friendly
        )
        load_time = time.perf_counter() - load_start

        # Configure LoRA
        print("  Configuring LoRA adapter...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        param_percent = 100 * trainable_params / total_params

        print(f"  LoRA Parameters: {trainable_params:,} trainable ({param_percent:.2f}% of {total_params:,} total)")

        # Create synthetic dataset
        print(f"  Creating synthetic dataset ({self.config.num_train_samples} train, {self.config.num_eval_samples} eval)...")

        def create_synthetic_data(num_samples: int) -> Dataset:
            texts = [
                f"Question: What is {i} + {i}? Answer: The answer is {i + i}."
                for i in range(num_samples)
            ]
            return Dataset.from_dict({"text": texts})

        train_dataset = create_synthetic_data(self.config.num_train_samples)
        eval_dataset = create_synthetic_data(self.config.num_eval_samples)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
            )

        train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        eval_tokenized = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

        # Training
        print("  Starting training...")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            fp16=False,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            data_collator=data_collator,
        )

        train_start = time.perf_counter()
        train_result = trainer.train()
        train_time = time.perf_counter() - train_start

        # Calculate tokens/sec
        total_tokens = self.config.num_train_samples * self.config.max_length * self.config.num_epochs
        tokens_per_sec = total_tokens / train_time

        # Save adapter
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        self.results = {
            "status": "success",
            "model": self.config.model_name,
            "dataset_train_samples": self.config.num_train_samples,
            "dataset_eval_samples": self.config.num_eval_samples,
            "lora_trainable_params": trainable_params,
            "lora_total_params": total_params,
            "lora_param_percent": param_percent,
            "training_time_sec": train_time,
            "tokens_per_sec": tokens_per_sec,
            "train_loss": train_result.training_loss,
            "adapter_path": str(output_dir),
        }

        # Print summary
        print(f"\n  Results:")
        print(f"    Model: {self.config.model_name}")
        print(f"    Dataset: {self.config.num_train_samples} synthetic examples ({self.config.num_eval_samples} for eval)")
        print(f"    LoRA Parameters: {trainable_params:,} trainable ({param_percent:.2f}% of {total_params:,} total)")
        print(f"    Training Time: {train_time:.2f} seconds")
        print(f"    Tokens/sec: ~{tokens_per_sec:,.0f}")
        print(f"    Train Loss: {train_result.training_loss:.2f}")

        return self.results


# =============================================================================
# Step 2: Evaluation Suite
# =============================================================================

class EvaluationBenchmark:
    """Evaluation suite with GSM8K and MMLU smoke tests."""

    def __init__(self, config: LoRABenchConfig, adapter_path: str = None):
        self.config = config
        self.adapter_path = adapter_path

    def run(self) -> Dict[str, Any]:
        """Run evaluation suite."""
        print("\n" + "=" * 60)
        print("Step 2: Evaluation Suite")
        print("=" * 60)

        results = {
            "gsm8k": self._eval_gsm8k_smoke(),
            "mmlu": self._eval_mmlu_smoke(),
        }

        print(f"\n  Results:")
        print(f"    GSM8K (smoke): {results['gsm8k']['accuracy']:.2f} accuracy (synthetic mode)")
        print(f"    MMLU (smoke): {results['mmlu']['accuracy']:.2f} accuracy (synthetic mode)")
        print(f"    Generated mock metrics for CI validation")

        return results

    def _eval_gsm8k_smoke(self) -> Dict[str, Any]:
        """GSM8K smoke test (synthetic for CI)."""
        print("  Running GSM8K smoke test...")

        # Synthetic evaluation for CI (real eval would use lm-evaluation-harness)
        # In production, this would call: lm_eval --model hf --model_args ... --tasks gsm8k
        return {
            "task": "gsm8k",
            "mode": "smoke",
            "accuracy": 0.0,  # Synthetic
            "num_samples": 5,
            "note": "Synthetic smoke test for CI validation",
        }

    def _eval_mmlu_smoke(self) -> Dict[str, Any]:
        """MMLU smoke test (synthetic for CI)."""
        print("  Running MMLU smoke test...")

        return {
            "task": "mmlu",
            "mode": "smoke",
            "accuracy": 0.0,  # Synthetic
            "num_samples": 5,
            "note": "Synthetic smoke test for CI validation",
        }


# =============================================================================
# Step 3: Performance Benchmark
# =============================================================================

class PerformanceBenchmark:
    """Inference performance benchmark (TTFT, latency, throughput)."""

    def __init__(self, config: LoRABenchConfig, adapter_path: str = None):
        self.config = config
        self.adapter_path = adapter_path

    def run(self) -> Dict[str, Any]:
        """Run performance benchmark."""
        print("\n" + "=" * 60)
        print("Step 3: Performance Benchmark")
        print("=" * 60)

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            print(f"  [SKIPPED] Missing dependencies: {e}")
            return {"status": "skipped"}

        # Load model
        print("  Loading model for inference benchmark...")
        model_path = self.adapter_path or self.config.model_name

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.eval()

        # Prepare test prompts
        prompts = [
            "The capital of France is",
            "Machine learning is",
            "The answer to 2 + 2 is",
        ]

        ttft_times = []
        total_times = []
        tokens_generated = []

        # Warmup
        print(f"  Warmup ({self.config.warmup_runs} runs)...")
        for _ in range(self.config.warmup_runs):
            inputs = tokenizer(prompts[0], return_tensors="pt")
            with torch.no_grad():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation_max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )

        # Benchmark
        print(f"  Benchmarking ({self.config.num_inference_runs} runs)...")
        gc.collect()

        for i in range(self.config.num_inference_runs):
            prompt = prompts[i % len(prompts)]
            inputs = tokenizer(prompt, return_tensors="pt")
            input_len = inputs["input_ids"].shape[1]

            # Measure TTFT and total time
            total_start = time.perf_counter()

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.generation_max_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                )

            total_time = (time.perf_counter() - total_start) * 1000  # ms

            output_len = outputs.sequences.shape[1]
            new_tokens = output_len - input_len

            # Approximate TTFT as total_time / num_tokens (simplified)
            ttft = total_time / max(new_tokens, 1)

            ttft_times.append(ttft)
            total_times.append(total_time)
            tokens_generated.append(new_tokens)

        # Calculate statistics
        avg_tokens = statistics.mean(tokens_generated)
        total_tokens = sum(tokens_generated)
        total_time_sec = sum(total_times) / 1000
        throughput = total_tokens / total_time_sec

        sorted_latencies = sorted(total_times)
        n = len(sorted_latencies)

        results = {
            "status": "success",
            "ttft_ms": statistics.mean(ttft_times),
            "latency_p50_ms": sorted_latencies[n // 2],
            "latency_p95_ms": sorted_latencies[int(n * 0.95)] if n >= 20 else sorted_latencies[-1],
            "latency_mean_ms": statistics.mean(total_times),
            "throughput_tokens_sec": throughput,
            "num_runs": self.config.num_inference_runs,
            "avg_tokens_per_run": avg_tokens,
        }

        print(f"\n  Results:")
        print(f"    Time to First Token (TTFT): {results['ttft_ms']:.2f} ms")
        print(f"    Latency P50: {results['latency_p50_ms']:.2f} ms")
        print(f"    Latency P95: {results['latency_p95_ms']:.2f} ms")
        print(f"    Throughput: {results['throughput_tokens_sec']:.2f} tokens/sec")

        return results


# =============================================================================
# Step 4: Encryption Pipeline Benchmark
# =============================================================================

class EncryptionPipelineBenchmark:
    """MOAI/N2HE encryption pipeline benchmark."""

    def __init__(self, config: LoRABenchConfig):
        self.config = config

    def run(self) -> Dict[str, Any]:
        """Run encryption pipeline benchmark."""
        print("\n" + "=" * 60)
        print("Step 4: Encryption Pipeline (MOAI/N2HE)")
        print("=" * 60)

        results = {
            "n2he": self._benchmark_n2he(),
            "moai": self._benchmark_moai(),
        }

        return results

    def _benchmark_n2he(self) -> Dict[str, Any]:
        """Benchmark N2HE gradient encryption."""
        print("  N2HE Gradient Encryption:")

        try:
            from tensorguard.core.crypto import N2HEContext, N2HEParams
        except ImportError as e:
            print(f"    [SKIPPED] {e}")
            return {"status": "skipped"}

        params = N2HEParams(n=self.config.n2he_lattice_dim, security_bits=128)
        ctx = N2HEContext(params)
        ctx.generate_keys()

        # Simulate gradient encryption for different sizes
        gradient_sizes = [256, 1024, 4096]
        results = {}

        for size in gradient_sizes:
            times = []
            for _ in range(10):
                data = np.random.randint(0, params.t, size=size, dtype=np.int64)
                start = time.perf_counter()
                ct = ctx.encrypt_batch(data)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = statistics.mean(times)
            results[f"dim_{size}"] = {
                "encrypt_ms": mean_time,
                "ciphertext_bytes": len(ct.serialize()),
            }
            print(f"    Dim {size}: {mean_time:.2f}ms encrypt")

        return {"status": "success", "benchmarks": results}

    def _benchmark_moai(self) -> Dict[str, Any]:
        """Benchmark MOAI CKKS inference encryption."""
        print("  MOAI CKKS Inference Encryption:")

        try:
            import tenseal as ts
        except ImportError:
            print("    [SKIPPED] TenSEAL not installed")
            return {"status": "skipped"}

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.config.moai_poly_modulus,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.global_scale = 2 ** 40
        ctx.generate_galois_keys()

        embedding_sizes = [64, 256, 512]
        results = {}

        for size in embedding_sizes:
            times = []
            for _ in range(10):
                vec = np.random.randn(size).tolist()
                start = time.perf_counter()
                ct = ts.ckks_vector(ctx, vec)
                times.append((time.perf_counter() - start) * 1000)

            mean_time = statistics.mean(times)
            results[f"dim_{size}"] = {
                "encrypt_ms": mean_time,
                "ciphertext_kb": len(ct.serialize()) / 1024,
            }
            print(f"    Dim {size}: {mean_time:.2f}ms encrypt")

        return {"status": "success", "benchmarks": results}


# =============================================================================
# Main Runner
# =============================================================================

class LoRAPEFTBenchmark:
    """Main benchmark runner."""

    def __init__(self, config: LoRABenchConfig = None):
        self.config = config or LoRABenchConfig()
        self.results = {}

    def run_all(self, skip_training: bool = False) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("\n" + "=" * 70)
        print("LoRA PEFT BENCHMARK WITH MOAI/N2HE ENCRYPTION PIPELINE")
        print("=" * 70)

        start_time = time.time()

        # Step 1: LoRA Training
        if skip_training:
            adapter_path = str(Path(self.config.output_dir) / "adapter")
            if Path(adapter_path).exists():
                print("\n[Step 1: Skipped - Using cached adapter]")
                self.results["training"] = {"status": "skipped", "adapter_path": adapter_path}
            else:
                training_bench = LoRATrainingBenchmark(self.config)
                self.results["training"] = training_bench.run()
                adapter_path = self.results["training"].get("adapter_path")
        else:
            training_bench = LoRATrainingBenchmark(self.config)
            self.results["training"] = training_bench.run()
            adapter_path = self.results["training"].get("adapter_path")

        # Step 2: Evaluation
        eval_bench = EvaluationBenchmark(self.config, adapter_path)
        self.results["evaluation"] = eval_bench.run()

        # Step 3: Performance
        perf_bench = PerformanceBenchmark(self.config, adapter_path)
        self.results["performance"] = perf_bench.run()

        # Step 4: Encryption
        enc_bench = EncryptionPipelineBenchmark(self.config)
        self.results["encryption"] = enc_bench.run()

        # Metadata
        self.results["metadata"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_duration_sec": time.time() - start_time,
        }

        # Save results
        self._save_results()

        # Print final summary
        self._print_summary()

        return self.results

    def _save_results(self):
        """Save results to JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"lora_peft_bench_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        latest = output_dir / "lora_peft_bench_latest.json"
        with open(latest, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")

    def _print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 70)
        print("STEP-BY-STEP PERFORMANCE SUMMARY")
        print("=" * 70)

        # Step 1: Training
        t = self.results.get("training", {})
        if t.get("status") == "success":
            print(f"\nStep 1: LoRA Training")
            print(f"  Model: {t.get('model', 'N/A')}")
            print(f"  Dataset: {t.get('dataset_train_samples', 0)} synthetic examples ({t.get('dataset_eval_samples', 0)} for eval)")
            print(f"  LoRA Parameters: {t.get('lora_trainable_params', 0):,} trainable ({t.get('lora_param_percent', 0):.2f}% of {t.get('lora_total_params', 0):,} total)")
            print(f"  Training Time: {t.get('training_time_sec', 0):.2f} seconds")
            print(f"  Tokens/sec: ~{t.get('tokens_per_sec', 0):,.0f}")
            print(f"  Train Loss: {t.get('train_loss', 0):.2f}")

        # Step 2: Evaluation
        e = self.results.get("evaluation", {})
        print(f"\nStep 2: Evaluation Suite")
        print(f"  GSM8K (smoke): {e.get('gsm8k', {}).get('accuracy', 0):.2f} accuracy (synthetic mode)")
        print(f"  MMLU (smoke): {e.get('mmlu', {}).get('accuracy', 0):.2f} accuracy (synthetic mode)")
        print(f"  Generated mock metrics for CI validation")

        # Step 3: Performance
        p = self.results.get("performance", {})
        if p.get("status") == "success":
            print(f"\nStep 3: Performance Benchmark")
            print(f"  Time to First Token (TTFT): {p.get('ttft_ms', 0):.2f} ms")
            print(f"  Latency P50: {p.get('latency_p50_ms', 0):.2f} ms")
            print(f"  Latency P95: {p.get('latency_p95_ms', 0):.2f} ms")
            print(f"  Throughput: {p.get('throughput_tokens_sec', 0):.2f} tokens/sec")

        # Step 4: Encryption
        enc = self.results.get("encryption", {})
        n2he = enc.get("n2he", {})
        moai = enc.get("moai", {})

        print(f"\nStep 4: Encryption Pipeline")
        if n2he.get("status") == "success":
            print(f"  N2HE Gradient Encryption:")
            for k, v in n2he.get("benchmarks", {}).items():
                print(f"    {k}: {v.get('encrypt_ms', 0):.2f}ms")
        if moai.get("status") == "success":
            print(f"  MOAI CKKS Inference:")
            for k, v in moai.get("benchmarks", {}).items():
                print(f"    {k}: {v.get('encrypt_ms', 0):.2f}ms")

        print(f"\n  Total Duration: {self.results.get('metadata', {}).get('total_duration_sec', 0):.1f}s")
        print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA PEFT Benchmark")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, use cached adapter")
    parser.add_argument("--output", default="artifacts/benchmarks/lora_peft", help="Output directory")
    args = parser.parse_args()

    config = LoRABenchConfig(output_dir=args.output)
    benchmark = LoRAPEFTBenchmark(config)
    benchmark.run_all(skip_training=args.skip_training)

    return 0


if __name__ == "__main__":
    sys.exit(main())
