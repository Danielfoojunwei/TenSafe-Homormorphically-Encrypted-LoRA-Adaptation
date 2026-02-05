"""
Benchmark implementations for common evaluation tasks.

Includes evaluators for:
- MMLU (Massive Multitask Language Understanding)
- GSM8K (Grade School Math)
- HumanEval (Code generation)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from .base import BenchmarkConfig, BenchmarkEvaluator, BenchmarkResult, Evaluator

logger = logging.getLogger(__name__)


class MMLUEvaluator(BenchmarkEvaluator):
    """
    Evaluator for MMLU (Massive Multitask Language Understanding).

    Tests knowledge across 57 subjects from STEM to humanities.
    """

    name = "mmlu"

    SUBJECTS = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load MMLU dataset."""
        try:
            from datasets import load_dataset

            # Load a subset of subjects if specified
            examples = []

            subjects = self.SUBJECTS[:5]  # Limit for demo

            for subject in subjects:
                try:
                    ds = load_dataset(
                        "cais/mmlu",
                        subject,
                        split=self.config.split,
                    )

                    for item in ds:
                        examples.append({
                            "subject": subject,
                            "question": item["question"],
                            "choices": item["choices"],
                            "answer": item["answer"],
                        })
                except Exception as e:
                    logger.warning(f"Failed to load {subject}: {e}")

            # Limit samples if specified
            if self.config.num_samples:
                examples = examples[: self.config.num_samples]

            return examples

        except ImportError:
            logger.warning("datasets not available, using mock data")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock MMLU data."""
        return [
            {
                "subject": "mathematics",
                "question": f"What is {i} + {i}?",
                "choices": [str(i), str(i * 2), str(i * 3), str(i * 4)],
                "answer": 1,  # Second choice (0-indexed)
            }
            for i in range(1, 11)
        ]

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format MMLU example as multiple choice prompt."""
        question = example["question"]
        choices = example["choices"]

        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            letter = chr(65 + i)  # A, B, C, D
            prompt += f"{letter}. {choice}\n"

        prompt += "\nAnswer with just the letter (A, B, C, or D):"

        return prompt

    def check_answer(
        self,
        prediction: str,
        example: Dict[str, Any],
    ) -> bool:
        """Check if MMLU answer is correct."""
        # Extract letter from prediction
        pred_clean = prediction.strip().upper()

        # Try to find a letter
        match = re.search(r"[ABCD]", pred_clean)
        if match:
            pred_letter = match.group()
            pred_idx = ord(pred_letter) - ord("A")
            return pred_idx == example["answer"]

        return False

    async def run(self) -> BenchmarkResult:
        """Run MMLU evaluation."""
        examples = self.load_dataset()
        prompts = [self.format_prompt(ex) for ex in examples]

        # Generate responses
        responses = await self.generate_responses(prompts)

        # Score
        correct = 0
        result_examples = []

        for ex, resp in zip(examples, responses):
            is_correct = self.check_answer(resp, ex)
            if is_correct:
                correct += 1

            result_examples.append({
                "question": ex["question"],
                "prediction": resp,
                "correct_answer": ex["answer"],
                "is_correct": is_correct,
            })

        accuracy = correct / len(examples) if examples else 0.0

        return BenchmarkResult(
            name=self.name,
            metrics={"accuracy": accuracy},
            num_samples=len(examples),
            num_correct=correct,
            examples=result_examples[:10],  # Log first 10
        )


class GSM8KEvaluator(BenchmarkEvaluator):
    """
    Evaluator for GSM8K (Grade School Math).

    Tests mathematical reasoning on word problems.
    """

    name = "gsm8k"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load GSM8K dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("gsm8k", "main", split=self.config.split)

            examples = []
            for item in ds:
                # Extract numeric answer
                answer = item["answer"]
                match = re.search(r"####\s*([\d,.-]+)", answer)
                if match:
                    numeric_answer = match.group(1).replace(",", "")
                else:
                    numeric_answer = ""

                examples.append({
                    "question": item["question"],
                    "full_answer": answer,
                    "numeric_answer": numeric_answer,
                })

            if self.config.num_samples:
                examples = examples[: self.config.num_samples]

            return examples

        except ImportError:
            return self._generate_mock_data()

    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock GSM8K data."""
        return [
            {
                "question": f"John has {i} apples. He buys {i * 2} more. How many apples does he have?",
                "full_answer": f"John starts with {i} apples and buys {i * 2} more. {i} + {i * 2} = {i * 3}. #### {i * 3}",
                "numeric_answer": str(i * 3),
            }
            for i in range(1, 11)
        ]

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format GSM8K example."""
        return (
            f"Solve this math problem step by step. "
            f"End your answer with the final numeric answer after '#### '.\n\n"
            f"Problem: {example['question']}\n\n"
            f"Solution:"
        )

    def check_answer(
        self,
        prediction: str,
        example: Dict[str, Any],
    ) -> bool:
        """Check if GSM8K answer is correct."""
        # Extract answer from prediction
        match = re.search(r"####\s*([\d,.-]+)", prediction)
        if not match:
            # Try to find any number at the end
            match = re.search(r"([\d,.-]+)\s*$", prediction.strip())

        if match:
            pred_answer = match.group(1).replace(",", "")
            try:
                pred_val = float(pred_answer)
                true_val = float(example["numeric_answer"])
                return abs(pred_val - true_val) < 0.01
            except ValueError:
                return False

        return False

    async def run(self) -> BenchmarkResult:
        """Run GSM8K evaluation."""
        examples = self.load_dataset()
        prompts = [self.format_prompt(ex) for ex in examples]

        responses = await self.generate_responses(prompts)

        correct = 0
        result_examples = []

        for ex, resp in zip(examples, responses):
            is_correct = self.check_answer(resp, ex)
            if is_correct:
                correct += 1

            result_examples.append({
                "question": ex["question"],
                "prediction": resp[:200],  # Truncate for logging
                "correct_answer": ex["numeric_answer"],
                "is_correct": is_correct,
            })

        accuracy = correct / len(examples) if examples else 0.0

        return BenchmarkResult(
            name=self.name,
            metrics={"accuracy": accuracy},
            num_samples=len(examples),
            num_correct=correct,
            examples=result_examples[:10],
        )


class HumanEvalEvaluator(BenchmarkEvaluator):
    """
    Evaluator for HumanEval (code generation).

    Tests code generation ability with functional correctness.
    """

    name = "humaneval"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load HumanEval dataset."""
        try:
            from datasets import load_dataset

            ds = load_dataset("openai_humaneval", split="test")

            examples = []
            for item in ds:
                examples.append({
                    "task_id": item["task_id"],
                    "prompt": item["prompt"],
                    "canonical_solution": item["canonical_solution"],
                    "test": item["test"],
                    "entry_point": item["entry_point"],
                })

            if self.config.num_samples:
                examples = examples[: self.config.num_samples]

            return examples

        except ImportError:
            return self._generate_mock_data()

    def _generate_mock_data(self) -> List[Dict[str, Any]]:
        """Generate mock HumanEval data."""
        return [
            {
                "task_id": f"HumanEval/{i}",
                "prompt": f"def add_{i}(a, b):\n    '''Add two numbers.'''\n",
                "canonical_solution": "    return a + b",
                "test": f"assert add_{i}(1, 2) == 3",
                "entry_point": f"add_{i}",
            }
            for i in range(10)
        ]

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """Format HumanEval example."""
        return (
            f"Complete the following Python function:\n\n"
            f"{example['prompt']}"
        )

    def check_answer(
        self,
        prediction: str,
        example: Dict[str, Any],
    ) -> bool:
        """
        Check if code solution passes tests.

        Note: In production, this would use sandboxed execution.
        Here we do a simplified check.
        """
        # Extract code from prediction
        code = prediction.strip()

        # Simple heuristic: check if it looks like valid Python
        # and contains a return statement
        if "return" not in code and "print" not in code:
            return False

        # For safety, we don't actually execute the code
        # In production, use a sandbox like code-eval
        return True  # Placeholder

    async def run(self) -> BenchmarkResult:
        """Run HumanEval evaluation."""
        examples = self.load_dataset()
        prompts = [self.format_prompt(ex) for ex in examples]

        responses = await self.generate_responses(prompts)

        correct = 0
        result_examples = []

        for ex, resp in zip(examples, responses):
            is_correct = self.check_answer(resp, ex)
            if is_correct:
                correct += 1

            result_examples.append({
                "task_id": ex["task_id"],
                "prediction": resp[:200],
                "is_correct": is_correct,
            })

        pass_at_1 = correct / len(examples) if examples else 0.0

        return BenchmarkResult(
            name=self.name,
            metrics={"pass@1": pass_at_1, "accuracy": pass_at_1},
            num_samples=len(examples),
            num_correct=correct,
            examples=result_examples[:10],
        )


# Register benchmarks
Evaluator.register_benchmark("mmlu", MMLUEvaluator)
Evaluator.register_benchmark("gsm8k", GSM8KEvaluator)
Evaluator.register_benchmark("humaneval", HumanEvalEvaluator)
