"""
Tool Use Training Recipe.

Train language models to effectively use tools and APIs. This recipe
supports function calling patterns and retrieval tool integration.

Example usage:
    from tensafe.cookbook.recipes import ToolUseConfig, run_tool_use

    config = ToolUseConfig(
        model_name="meta-llama/Llama-3.1-8B",
        tools=[
            {"name": "search", "description": "Search the web"},
            {"name": "calculator", "description": "Perform calculations"},
        ],
    )
    await run_tool_use(config, client)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

from ..hyperparam_utils import LoRAConfig
from ..model_info import get_recommended_renderer_name

logger = logging.getLogger(__name__)


class TrainingClient(Protocol):
    """Protocol for training clients."""

    def forward_backward(self, batch: Dict[str, Any]) -> Any:
        ...

    def optim_step(self) -> Any:
        ...

    def save_state(self, **kwargs) -> Any:
        ...


@dataclass
class ToolDefinition:
    """Definition of a tool/function that the model can call."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required_params,
                },
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            parameters=data.get("parameters", {}),
            required_params=data.get("required", data.get("required_params", [])),
        )


@dataclass
class ToolCall:
    """A tool call made by the model."""

    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }

    @classmethod
    def from_string(cls, s: str) -> Optional["ToolCall"]:
        """Parse tool call from string format."""
        try:
            # Try JSON format first
            data = json.loads(s)
            return cls(
                name=data.get("name", ""),
                arguments=data.get("arguments", {}),
            )
        except json.JSONDecodeError:
            pass

        # Try function call format: function_name(arg1=val1, arg2=val2)
        import re

        match = re.match(r"(\w+)\((.*)\)", s.strip())
        if match:
            name = match.group(1)
            args_str = match.group(2)

            # Parse arguments
            args = {}
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    try:
                        args[key.strip()] = json.loads(value.strip())
                    except json.JSONDecodeError:
                        args[key.strip()] = value.strip().strip("\"'")

            return cls(name=name, arguments=args)

        return None


@dataclass
class ToolUseConfig:
    """Configuration for tool use training."""

    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    renderer_name: Optional[str] = None

    # LoRA settings
    lora_rank: int = 32
    lora_alpha: float = 64.0

    # Tool definitions
    tools: List[Dict[str, Any]] = field(default_factory=list)

    # Dataset settings
    dataset: Optional[str] = None  # Custom dataset path
    dataset_split: str = "train"
    max_seq_length: int = 4096

    # Training settings
    batch_size: int = 64
    learning_rate: float = 2e-4
    num_epochs: int = 1
    max_steps: Optional[int] = None

    # RL settings (for reward-based training)
    use_rl: bool = False
    reward_correct_call: float = 1.0
    reward_incorrect_call: float = -0.5
    reward_no_call: float = 0.0

    # Checkpointing
    checkpoint_dir: str = "/tmp/tensafe-tool-use"
    save_steps: int = 500

    # Logging
    log_steps: int = 10

    def __post_init__(self):
        if self.renderer_name is None:
            self.renderer_name = get_recommended_renderer_name(self.model_name)

    @property
    def tool_definitions(self) -> List[ToolDefinition]:
        """Get tool definitions as objects."""
        return [ToolDefinition.from_dict(t) for t in self.tools]

    @property
    def lora_config(self) -> LoRAConfig:
        return LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha)


@dataclass
class ToolUseExample:
    """A tool use training example."""

    query: str  # User query
    expected_tool: Optional[str] = None  # Expected tool to call
    expected_args: Optional[Dict[str, Any]] = None  # Expected arguments
    tool_result: Optional[str] = None  # Result from tool execution
    final_response: Optional[str] = None  # Final response after tool use


class ToolExecutor:
    """
    Executes tools and returns results.

    Can be configured with mock implementations for training.
    """

    def __init__(self, tools: Dict[str, Callable] = None):
        """
        Initialize executor.

        Args:
            tools: Mapping of tool names to implementations
        """
        self.tools = tools or {}
        self._mock_results: Dict[str, str] = {}

    def register_tool(self, name: str, func: Callable) -> None:
        """Register a tool implementation."""
        self.tools[name] = func

    def set_mock_result(self, name: str, result: str) -> None:
        """Set mock result for a tool."""
        self._mock_results[name] = result

    def execute(self, call: ToolCall) -> str:
        """
        Execute a tool call.

        Args:
            call: Tool call to execute

        Returns:
            Result string from tool execution
        """
        # Check for mock result first
        if call.name in self._mock_results:
            return self._mock_results[call.name]

        # Check for real implementation
        if call.name in self.tools:
            try:
                result = self.tools[call.name](**call.arguments)
                return str(result)
            except Exception as e:
                return f"Error: {e}"

        return f"Unknown tool: {call.name}"


class ToolUseDataset:
    """
    Dataset for tool use training.

    Generates or loads examples of correct tool usage.
    """

    def __init__(
        self,
        tools: List[ToolDefinition],
        dataset_path: Optional[str] = None,
        max_examples: int = 1000,
    ):
        """
        Initialize dataset.

        Args:
            tools: Available tool definitions
            dataset_path: Path to custom dataset (None = generate synthetic)
            max_examples: Maximum examples to generate
        """
        self.tools = tools
        self.dataset_path = dataset_path
        self.max_examples = max_examples
        self._data: Optional[List[ToolUseExample]] = None
        self._index = 0

    def _load_data(self) -> None:
        """Load or generate dataset."""
        if self.dataset_path:
            self._load_from_path()
        else:
            self._generate_synthetic()

    def _load_from_path(self) -> None:
        """Load dataset from file."""
        try:
            with open(self.dataset_path, "r") as f:
                data = json.load(f)

            self._data = [
                ToolUseExample(
                    query=item["query"],
                    expected_tool=item.get("tool"),
                    expected_args=item.get("args"),
                    tool_result=item.get("result"),
                    final_response=item.get("response"),
                )
                for item in data
            ]
        except Exception as e:
            logger.warning(f"Failed to load dataset: {e}")
            self._generate_synthetic()

    def _generate_synthetic(self) -> None:
        """Generate synthetic tool use examples."""
        self._data = []

        # Generate examples for each tool
        for tool in self.tools:
            examples = self._generate_for_tool(tool)
            self._data.extend(examples)

        # Limit to max examples
        self._data = self._data[: self.max_examples]

    def _generate_for_tool(self, tool: ToolDefinition) -> List[ToolUseExample]:
        """Generate examples for a specific tool."""
        examples = []

        # Generate based on tool type
        if "search" in tool.name.lower():
            examples.extend(self._generate_search_examples(tool))
        elif "calculator" in tool.name.lower() or "math" in tool.name.lower():
            examples.extend(self._generate_calculator_examples(tool))
        elif "weather" in tool.name.lower():
            examples.extend(self._generate_weather_examples(tool))
        else:
            examples.extend(self._generate_generic_examples(tool))

        return examples

    def _generate_search_examples(self, tool: ToolDefinition) -> List[ToolUseExample]:
        """Generate search tool examples."""
        queries = [
            ("What is the capital of France?", {"query": "capital of France"}),
            ("Find information about Python programming", {"query": "Python programming"}),
            ("Who won the World Cup in 2022?", {"query": "World Cup 2022 winner"}),
        ]

        return [
            ToolUseExample(
                query=q,
                expected_tool=tool.name,
                expected_args=args,
                tool_result="Search results...",
                final_response="Based on the search results...",
            )
            for q, args in queries
        ]

    def _generate_calculator_examples(
        self, tool: ToolDefinition
    ) -> List[ToolUseExample]:
        """Generate calculator tool examples."""
        calculations = [
            ("What is 15 + 27?", {"expression": "15 + 27"}, "42"),
            ("Calculate 100 / 4", {"expression": "100 / 4"}, "25"),
            ("What is 7 times 8?", {"expression": "7 * 8"}, "56"),
        ]

        return [
            ToolUseExample(
                query=q,
                expected_tool=tool.name,
                expected_args=args,
                tool_result=result,
                final_response=f"The result is {result}.",
            )
            for q, args, result in calculations
        ]

    def _generate_weather_examples(self, tool: ToolDefinition) -> List[ToolUseExample]:
        """Generate weather tool examples."""
        queries = [
            ("What's the weather in New York?", {"location": "New York"}),
            ("Is it raining in London?", {"location": "London"}),
            ("Temperature in Tokyo", {"location": "Tokyo"}),
        ]

        return [
            ToolUseExample(
                query=q,
                expected_tool=tool.name,
                expected_args=args,
                tool_result="Sunny, 72°F",
                final_response="The weather is sunny with a temperature of 72°F.",
            )
            for q, args in queries
        ]

    def _generate_generic_examples(
        self, tool: ToolDefinition
    ) -> List[ToolUseExample]:
        """Generate generic tool examples."""
        return [
            ToolUseExample(
                query=f"Use the {tool.name} tool for a task",
                expected_tool=tool.name,
                expected_args={},
                tool_result="Task completed",
                final_response="I've completed the task using the tool.",
            )
            for _ in range(3)
        ]

    def __len__(self) -> int:
        if self._data is None:
            self._load_data()
        return len(self._data)

    def get_batch(self, batch_size: int) -> List[ToolUseExample]:
        """Get a batch of examples."""
        if self._data is None:
            self._load_data()

        batch = []
        for _ in range(batch_size):
            if self._index >= len(self._data):
                self._index = 0

            batch.append(self._data[self._index])
            self._index += 1

        return batch


class ToolUseTrainer:
    """
    Trainer for tool use capability.

    Trains models to correctly identify when and how to use tools.
    """

    def __init__(
        self,
        config: ToolUseConfig,
        client: TrainingClient,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration
            client: Training client
        """
        self.config = config
        self.client = client

        # Initialize components
        self.dataset = ToolUseDataset(
            tools=config.tool_definitions,
            dataset_path=config.dataset,
        )

        self.executor = ToolExecutor()

        # Build system prompt with tool descriptions
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = []
        for tool in self.config.tool_definitions:
            params = ", ".join(
                f"{name}: {info.get('type', 'string')}"
                for name, info in tool.parameters.items()
            )
            tools_desc.append(f"- {tool.name}({params}): {tool.description}")

        return (
            "You are a helpful assistant with access to the following tools:\n\n"
            + "\n".join(tools_desc)
            + "\n\nWhen a tool is needed, respond with a function call in the format: "
            "tool_name(arg1=value1, arg2=value2)"
        )

    def build_training_example(
        self, example: ToolUseExample
    ) -> Dict[str, Any]:
        """
        Build training tokens from an example.

        Args:
            example: Tool use example

        Returns:
            Training batch item
        """
        from ..tokenizer_utils import get_tokenizer
        from ..renderers import get_renderer

        tokenizer = get_tokenizer(self.config.model_name)
        renderer = get_renderer(self.config.renderer_name, tokenizer)

        from ..renderers.base import Message

        # Build conversation
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=example.query),
        ]

        # Add tool call if expected
        if example.expected_tool:
            call_str = f"{example.expected_tool}({json.dumps(example.expected_args)})"
            messages.append(Message(role="assistant", content=call_str))

            # Add tool result
            if example.tool_result:
                messages.append(
                    Message(
                        role="tool",
                        content=example.tool_result,
                        name=example.expected_tool,
                    )
                )

            # Add final response
            if example.final_response:
                messages.append(
                    Message(role="assistant", content=example.final_response)
                )
        else:
            # No tool needed, direct response
            if example.final_response:
                messages.append(
                    Message(role="assistant", content=example.final_response)
                )

        # Build supervised example
        input_ids, weights = renderer.build_supervised_example(messages)

        # Truncate if needed
        max_len = self.config.max_seq_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            weights = weights[:max_len]

        attention_mask = [1] * len(input_ids)

        # Build labels
        labels = []
        for i, (token, weight) in enumerate(zip(input_ids, weights)):
            if weight > 0 and i > 0:
                labels.append(input_ids[i])
            else:
                labels.append(-100)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    async def train(self) -> Dict[str, Any]:
        """
        Run tool use training.

        Returns:
            Training metrics
        """
        logger.info("Starting tool use training")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Tools: {[t.name for t in self.config.tool_definitions]}")

        total_loss = 0.0
        num_steps = self.config.max_steps or (
            len(self.dataset) * self.config.num_epochs // self.config.batch_size
        )

        for step in range(num_steps):
            # Get batch
            examples = self.dataset.get_batch(self.config.batch_size)

            # Build training batch
            batch = {"input_ids": [], "attention_mask": [], "labels": []}
            for example in examples:
                item = self.build_training_example(example)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["labels"].append(item["labels"])

            # Train step
            fb_future = self.client.forward_backward(batch)
            fb_result = fb_future.result()

            opt_future = self.client.optim_step()
            opt_future.result()

            total_loss += fb_result.loss

            # Log
            if step % self.config.log_steps == 0:
                avg_loss = total_loss / (step + 1)
                logger.info(f"Step {step} | Loss: {avg_loss:.4f}")

            # Save checkpoint
            if step % self.config.save_steps == 0:
                self.client.save_state(metadata={"step": step})

        final_loss = total_loss / num_steps
        logger.info(f"Training complete. Final loss: {final_loss:.4f}")

        return {"final_loss": final_loss, "num_steps": num_steps}


async def run_tool_use(
    config: ToolUseConfig,
    client: TrainingClient,
) -> Dict[str, Any]:
    """
    Run tool use training.

    Args:
        config: Training configuration
        client: Training client

    Returns:
        Training metrics
    """
    trainer = ToolUseTrainer(config, client)
    return await trainer.train()
