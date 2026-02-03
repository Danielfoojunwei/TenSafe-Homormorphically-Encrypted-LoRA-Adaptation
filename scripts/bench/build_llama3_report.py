#!/usr/bin/env python3
"""
Llama3 LoRA Benchmark Report Builder

Generates comprehensive benchmark reports for Llama 3 LoRA fine-tuning including:
- Training configuration and metrics
- Industry-standard evaluation results (GSM8K, MMLU, MT-Bench, AlpacaEval)
- Inference performance benchmarks
- Reproducibility information
- Benchmark citations

Usage:
    python scripts/bench/build_llama3_report.py --output-dir reports/llama3_lora_bench

Outputs:
    reports/llama3_lora_bench/<git_sha>/report.md
    reports/llama3_lora_bench/<git_sha>/report.json
"""

import argparse
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Benchmark citations
CITATIONS = {
    "gsm8k": {
        "name": "GSM8K",
        "full_name": "Grade School Math 8K",
        "description": "8.5K grade school math word problems requiring multi-step reasoning",
        "url": "https://github.com/openai/grade-school-math",
        "paper": "Cobbe et al., 2021",
    },
    "mmlu": {
        "name": "MMLU",
        "full_name": "Massive Multitask Language Understanding",
        "description": "57 subjects spanning STEM, humanities, social sciences, and more",
        "url": "https://crfm.stanford.edu/helm/mmlu/latest/",
        "paper": "Hendrycks et al., 2021",
    },
    "mt_bench": {
        "name": "MT-Bench",
        "full_name": "Multi-Turn Benchmark",
        "description": "80 multi-turn questions across 8 categories for conversation ability",
        "url": "https://lmsys.org/blog/2023-06-22-leaderboard/",
        "paper": "Zheng et al., 2023",
    },
    "alpaca_eval": {
        "name": "AlpacaEval 2.0",
        "full_name": "AlpacaEval 2.0",
        "description": "LLM-based instruction-following evaluator with length-controlled win rate",
        "url": "https://tatsu-lab.github.io/alpaca_eval/",
        "paper": "Li et al., 2023",
    },
    "lm_eval_harness": {
        "name": "lm-evaluation-harness",
        "full_name": "Language Model Evaluation Harness",
        "description": "Unified framework for evaluating language models",
        "url": "https://github.com/EleutherAI/lm-evaluation-harness",
        "paper": "Gao et al., 2021",
    },
    "oasst1": {
        "name": "OASST1",
        "full_name": "OpenAssistant Conversations",
        "description": "Human-annotated assistant-style conversation dataset",
        "url": "https://huggingface.co/datasets/OpenAssistant/oasst1",
        "license": "Apache-2.0",
    },
}


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


def get_hardware_info() -> Dict[str, Any]:
    """Collect hardware and environment information."""
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    except ImportError:
        info["torch_version"] = "not installed"

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import peft
        info["peft_version"] = peft.__version__
    except ImportError:
        pass

    return info


def load_json_safe(path: Path) -> Optional[Dict]:
    """Safely load a JSON file."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None
    return None


@dataclass
class Llama3BenchReport:
    """Complete Llama3 LoRA benchmark report."""
    # Metadata
    timestamp: str = ""
    git_sha: str = ""
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    model_name: str = ""
    dataset_name: str = ""
    dataset_license: str = ""
    lora_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)

    # Training Results
    training_metrics: Dict[str, Any] = field(default_factory=dict)

    # Evaluation Results
    eval_results: Dict[str, Any] = field(default_factory=dict)

    # Performance Results
    perf_results: Dict[str, Any] = field(default_factory=dict)

    # Reproducibility
    seeds: Dict[str, int] = field(default_factory=dict)
    commands: List[str] = field(default_factory=list)

    # Compliance
    compliance_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Llama3ReportBuilder:
    """Builds comprehensive Llama3 LoRA benchmark reports."""

    def __init__(
        self,
        git_sha: str,
        lora_output_dir: Path,
        bench_dir: Path,
        compliance_dir: Path,
        output_dir: Path,
    ):
        self.git_sha = git_sha
        self.lora_output_dir = lora_output_dir
        self.bench_dir = bench_dir
        self.compliance_dir = compliance_dir
        self.output_dir = output_dir

        # Find latest LoRA run
        self.lora_run = self._find_latest_lora_run()

        # Load all data
        self.training_data = self._load_training_data()
        self.eval_data = self._load_eval_data()
        self.perf_data = self._load_perf_data()
        self.compliance_data = self._load_compliance_data()

    def _find_latest_lora_run(self) -> Optional[Path]:
        """Find the most recent LoRA training run."""
        if not self.lora_output_dir.exists():
            return None

        runs = sorted([d for d in self.lora_output_dir.iterdir() if d.is_dir()], reverse=True)
        return runs[0] if runs else None

    def _load_training_data(self) -> Optional[Dict]:
        """Load training data from LoRA run."""
        if not self.lora_run:
            return None

        # Try different possible files
        candidates = [
            self.lora_run / "train_metrics.json",
            self.lora_run / "run_config.json",
        ]

        data = {}
        for path in candidates:
            loaded = load_json_safe(path)
            if loaded:
                data.update(loaded)

        return data if data else None

    def _load_eval_data(self) -> Optional[Dict]:
        """Load evaluation results."""
        eval_path = self.bench_dir / self.git_sha / "eval_results.json"
        if not eval_path.exists():
            # Try finding any eval results
            eval_files = list(self.bench_dir.glob("*/eval_results.json"))
            if eval_files:
                eval_path = eval_files[-1]

        return load_json_safe(eval_path)

    def _load_perf_data(self) -> Optional[Dict]:
        """Load performance benchmark results."""
        perf_path = self.bench_dir / self.git_sha / "perf_results.json"
        if not perf_path.exists():
            perf_files = list(self.bench_dir.glob("*/perf_results.json"))
            if perf_files:
                perf_path = perf_files[-1]

        return load_json_safe(perf_path)

    def _load_compliance_data(self) -> Optional[Dict]:
        """Load compliance metrics."""
        metrics_path = self.compliance_dir / self.git_sha / "metrics.json"
        if not metrics_path.exists():
            metrics_files = list(self.compliance_dir.glob("*/metrics.json"))
            if metrics_files:
                metrics_path = metrics_files[-1]

        return load_json_safe(metrics_path)

    def build_report(self) -> Llama3BenchReport:
        """Build the complete report."""
        report = Llama3BenchReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_sha=self.git_sha,
            hardware_info=get_hardware_info(),
        )

        # Extract configuration
        if self.training_data:
            report.model_name = self.training_data.get("training_config", {}).get("model_name", "unknown")
            report.lora_config = self.training_data.get("lora_config", {})
            report.training_config = self.training_data.get("training_config", {})
            report.training_metrics = {
                k: v for k, v in self.training_data.items()
                if k not in ["training_config", "lora_config", "hardware_info"]
            }
            report.seeds = {
                "seed": self.training_data.get("training_config", {}).get("seed", 42),
                "data_seed": self.training_data.get("training_config", {}).get("data_seed", 42),
            }

        # Dataset info
        report.dataset_name = CITATIONS["oasst1"]["full_name"]
        report.dataset_license = CITATIONS["oasst1"]["license"]

        # Evaluation results
        if self.eval_data:
            report.eval_results = self.eval_data

        # Performance results
        if self.perf_data:
            report.perf_results = self.perf_data

        # Compliance summary
        if self.compliance_data:
            report.compliance_summary = {
                "pii_found": self.compliance_data.get("pii_scan", {}).get("total_pii_found", 0),
                "secrets_found": self.compliance_data.get("secrets_hygiene", {}).get("secrets_found", 0),
                "audit_enabled": self.compliance_data.get("audit_logging", {}).get("audit_log_enabled", False),
            }

        # Commands for reproducibility
        report.commands = [
            f"python scripts/lora/train_lora_llama3.py --model {report.model_name} --seed {report.seeds.get('seed', 42)}",
            "python scripts/bench/eval_suite.py --tasks gsm8k,mmlu",
            "python scripts/bench/perf_infer.py",
            "python scripts/bench/build_llama3_report.py",
        ]

        return report

    def generate_markdown(self, report: Llama3BenchReport) -> str:
        """Generate Markdown report."""
        lines = []

        # Header
        lines.append("# Llama 3 LoRA Benchmark Report")
        lines.append("")
        lines.append(f"> **Git SHA**: `{report.git_sha}`")
        lines.append(f"> **Generated**: {report.timestamp}")
        lines.append(f"> **Model**: {report.model_name}")
        lines.append("")

        # Disclaimer
        lines.append("## Disclaimer")
        lines.append("")
        lines.append("This report presents benchmark results for research purposes. ")
        lines.append("Results are specific to the configuration, hardware, and data used. ")
        lines.append("Do not interpret these as claims of general model capability.")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("1. [Dataset & Licensing](#dataset--licensing)")
        lines.append("2. [Training Configuration](#training-configuration)")
        lines.append("3. [Training Results](#training-results)")
        lines.append("4. [Evaluation Results](#evaluation-results)")
        lines.append("5. [Performance Benchmarks](#performance-benchmarks)")
        lines.append("6. [Reproducibility](#reproducibility)")
        lines.append("7. [Benchmark Citations](#benchmark-citations)")
        lines.append("")

        # Dataset & Licensing
        lines.append("## Dataset & Licensing")
        lines.append("")
        lines.append(f"**Dataset**: [{report.dataset_name}]({CITATIONS['oasst1']['url']})")
        lines.append(f"**License**: {report.dataset_license}")
        lines.append(f"**Description**: {CITATIONS['oasst1']['description']}")
        lines.append("")

        # Training Configuration
        lines.append("## Training Configuration")
        lines.append("")
        lines.append("### LoRA Configuration")
        lines.append("")
        if report.lora_config:
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            for key, value in report.lora_config.items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        lines.append("### Training Hyperparameters")
        lines.append("")
        if report.training_config:
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            important_keys = ["num_train_epochs", "per_device_train_batch_size",
                           "gradient_accumulation_steps", "learning_rate",
                           "max_seq_length", "bf16", "load_in_4bit"]
            for key in important_keys:
                if key in report.training_config:
                    lines.append(f"| {key} | {report.training_config[key]} |")
            lines.append("")

        # Training Results
        lines.append("## Training Results")
        lines.append("")
        if report.training_metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            metrics_to_show = ["train_loss", "tokens_per_second", "total_train_time_seconds",
                             "peak_gpu_memory_mb", "total_steps"]
            for key in metrics_to_show:
                if key in report.training_metrics:
                    value = report.training_metrics[key]
                    if isinstance(value, float):
                        lines.append(f"| {key.replace('_', ' ').title()} | {value:.4f} |")
                    else:
                        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
            lines.append("")
        else:
            lines.append("_Training metrics not available._")
            lines.append("")

        # Evaluation Results
        lines.append("## Evaluation Results")
        lines.append("")
        lines.append("### Industry-Standard Benchmarks")
        lines.append("")

        if report.eval_results and "results" in report.eval_results:
            lines.append("| Benchmark | Metric | Base Model | + LoRA |")
            lines.append("|-----------|--------|------------|--------|")

            for result in report.eval_results.get("results", []):
                task = result.get("task_name", "unknown")
                metric = result.get("metric_name", "acc")
                score = result.get("score", 0.0)
                lines.append(f"| {task} | {metric} | - | {score:.4f} |")

            lines.append("")

            # Note about judge-based benchmarks
            has_mt_bench = any(r.get("task_name") == "mt_bench" for r in report.eval_results.get("results", []))
            has_alpaca = any(r.get("task_name") == "alpaca_eval" for r in report.eval_results.get("results", []))

            if has_mt_bench or has_alpaca:
                lines.append("**Note**: MT-Bench and AlpacaEval scores require LLM-as-judge. ")
                lines.append("Results marked with '-' indicate generations were saved but not judged.")
                lines.append("")
        else:
            lines.append("_Evaluation results not available. Run `make eval-llama3` to generate._")
            lines.append("")

        # Performance Benchmarks
        lines.append("## Performance Benchmarks")
        lines.append("")

        if report.perf_results and "results" in report.perf_results:
            lines.append("### Inference Performance")
            lines.append("")
            lines.append("| Variant | TTFT (ms) | P50 Latency (ms) | P95 Latency (ms) | Tokens/sec |")
            lines.append("|---------|-----------|------------------|------------------|------------|")

            for result in report.perf_results.get("results", []):
                variant = result.get("variant", "unknown")
                ttft = result.get("avg_ttft_ms", 0)
                p50 = result.get("p50_latency_ms", 0)
                p95 = result.get("p95_latency_ms", 0)
                tps = result.get("avg_tokens_per_second", 0)
                lines.append(f"| {variant} | {ttft:.2f} | {p50:.2f} | {p95:.2f} | {tps:.2f} |")

            lines.append("")

            # Resource usage
            if report.perf_results.get("results"):
                last_result = report.perf_results["results"][-1]
                lines.append("### Resource Usage")
                lines.append("")
                lines.append(f"- **Peak RSS Memory**: {last_result.get('peak_rss_mb', 0):.2f} MB")
                lines.append(f"- **Peak VRAM**: {last_result.get('peak_vram_mb', 0):.2f} MB")
                lines.append("")
        else:
            lines.append("_Performance results not available. Run `make perf-llama3` to generate._")
            lines.append("")

        # Reproducibility
        lines.append("## Reproducibility")
        lines.append("")
        lines.append("### Seeds")
        lines.append("")
        for name, seed in report.seeds.items():
            lines.append(f"- **{name}**: {seed}")
        lines.append("")

        lines.append("### Commands")
        lines.append("")
        lines.append("```bash")
        for cmd in report.commands:
            lines.append(cmd)
        lines.append("```")
        lines.append("")

        lines.append("### Hardware")
        lines.append("")
        if report.hardware_info:
            lines.append("| Component | Value |")
            lines.append("|-----------|-------|")
            for key in ["platform", "gpu_name", "cuda_version", "torch_version"]:
                if key in report.hardware_info:
                    lines.append(f"| {key.replace('_', ' ').title()} | {report.hardware_info[key]} |")
            lines.append("")

        # Benchmark Citations
        lines.append("## Benchmark Citations")
        lines.append("")
        for key, info in CITATIONS.items():
            lines.append(f"### {info['name']}")
            lines.append("")
            lines.append(f"- **Full Name**: {info['full_name']}")
            lines.append(f"- **Description**: {info['description']}")
            lines.append(f"- **URL**: {info['url']}")
            if "paper" in info:
                lines.append(f"- **Reference**: {info['paper']}")
            if "license" in info:
                lines.append(f"- **License**: {info['license']}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("_Generated by TensorGuard Llama3 LoRA Benchmark Suite_")
        lines.append("")

        return "\n".join(lines)

    def save_report(self) -> tuple:
        """Save both Markdown and JSON reports."""
        report = self.build_report()

        output_path = self.output_dir / self.git_sha
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_path / "report.json"
        with open(json_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        # Save Markdown
        md_path = output_path / "report.md"
        md_content = self.generate_markdown(report)
        with open(md_path, "w") as f:
            f.write(md_content)

        return str(md_path), str(json_path)


def main():
    parser = argparse.ArgumentParser(
        description="Build Llama3 LoRA benchmark report"
    )
    parser.add_argument(
        "--git-sha",
        type=str,
        default=None,
        help="Git SHA (default: current HEAD)",
    )
    parser.add_argument(
        "--lora-dir",
        type=Path,
        default=Path("outputs/lora_llama3"),
        help="LoRA outputs directory",
    )
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("reports/llama3_lora_bench"),
        help="Benchmark results directory",
    )
    parser.add_argument(
        "--compliance-dir",
        type=Path,
        default=Path("reports/compliance"),
        help="Compliance reports directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/llama3_lora_bench"),
        help="Output directory",
    )

    args = parser.parse_args()

    git_sha = args.git_sha or get_git_sha()

    builder = Llama3ReportBuilder(
        git_sha=git_sha,
        lora_output_dir=args.lora_dir.resolve(),
        bench_dir=args.bench_dir.resolve(),
        compliance_dir=args.compliance_dir.resolve(),
        output_dir=args.output_dir.resolve(),
    )

    md_path, json_path = builder.save_report()

    print("\nLlama3 LoRA Benchmark Report Generated!")
    print(f"  Markdown: {md_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
