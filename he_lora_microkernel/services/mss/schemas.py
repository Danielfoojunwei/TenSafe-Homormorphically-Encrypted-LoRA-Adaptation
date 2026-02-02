"""
MSS API Schemas

OpenAI-compatible request/response schemas with HE-LoRA extensions
for insertion point configuration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class LoRATargetType(str, Enum):
    """LoRA target projection types."""
    QKV = "qkv"
    QKVO = "qkvo"


class InsertionPointType(str, Enum):
    """Where to inject deltas relative to projection."""
    PRE_PROJECTION = "pre_projection"
    POST_PROJECTION = "post_projection"


class LayerSelectionMode(str, Enum):
    """How to select layers for LoRA application."""
    ALL = "all"
    RANGE = "range"
    LIST = "list"
    PATTERN = "pattern"  # e.g., "every_other", "last_n"


@dataclass
class LayerSelection:
    """
    Configuration for which layers receive LoRA deltas.

    Attributes:
        mode: Selection mode (all, range, list, pattern)
        start: Start layer for range mode
        end: End layer for range mode (exclusive)
        layers: Explicit list for list mode
        pattern: Pattern name for pattern mode
        pattern_arg: Pattern argument (e.g., n for last_n)
    """
    mode: LayerSelectionMode = LayerSelectionMode.ALL
    start: Optional[int] = None
    end: Optional[int] = None
    layers: Optional[List[int]] = None
    pattern: Optional[str] = None
    pattern_arg: Optional[int] = None

    def get_layers(self, num_layers: int) -> List[int]:
        """Compute the list of layer indices."""
        if self.mode == LayerSelectionMode.ALL:
            return list(range(num_layers))

        elif self.mode == LayerSelectionMode.RANGE:
            start = self.start or 0
            end = self.end or num_layers
            return list(range(start, min(end, num_layers)))

        elif self.mode == LayerSelectionMode.LIST:
            if self.layers is None:
                return list(range(num_layers))
            return [l for l in self.layers if 0 <= l < num_layers]

        elif self.mode == LayerSelectionMode.PATTERN:
            if self.pattern == "every_other":
                return list(range(0, num_layers, 2))
            elif self.pattern == "last_n":
                n = self.pattern_arg or 8
                return list(range(max(0, num_layers - n), num_layers))
            elif self.pattern == "first_n":
                n = self.pattern_arg or 8
                return list(range(min(n, num_layers)))
            elif self.pattern == "middle":
                quarter = num_layers // 4
                return list(range(quarter, num_layers - quarter))

        return list(range(num_layers))

    def to_dict(self) -> Dict:
        return {
            'mode': self.mode.value,
            'start': self.start,
            'end': self.end,
            'layers': self.layers,
            'pattern': self.pattern,
            'pattern_arg': self.pattern_arg,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LayerSelection':
        return cls(
            mode=LayerSelectionMode(data.get('mode', 'all')),
            start=data.get('start'),
            end=data.get('end'),
            layers=data.get('layers'),
            pattern=data.get('pattern'),
            pattern_arg=data.get('pattern_arg'),
        )


@dataclass
class InsertionPointSchema:
    """
    Schema for configuring HE-LoRA insertion points.

    This schema defines where and how LoRA deltas are injected
    during inference.

    Attributes:
        adapter_id: Identifier for the LoRA adapter to use
        targets: Which projections to target (QKV or QKVO)
        layer_selection: Configuration for layer selection
        insertion_point: Pre or post projection injection
        per_layer_config: Optional per-layer target overrides
        enabled: Whether HE-LoRA is enabled for this request
    """
    adapter_id: str
    targets: LoRATargetType = LoRATargetType.QKV
    layer_selection: LayerSelection = field(default_factory=LayerSelection)
    insertion_point: InsertionPointType = InsertionPointType.POST_PROJECTION
    per_layer_config: Optional[Dict[int, LoRATargetType]] = None
    enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            'adapter_id': self.adapter_id,
            'targets': self.targets.value,
            'layer_selection': self.layer_selection.to_dict(),
            'insertion_point': self.insertion_point.value,
            'per_layer_config': {
                str(k): v.value for k, v in (self.per_layer_config or {}).items()
            },
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'InsertionPointSchema':
        per_layer = None
        if 'per_layer_config' in data and data['per_layer_config']:
            per_layer = {
                int(k): LoRATargetType(v)
                for k, v in data['per_layer_config'].items()
            }

        layer_sel = data.get('layer_selection', {})
        if isinstance(layer_sel, dict):
            layer_sel = LayerSelection.from_dict(layer_sel)
        elif isinstance(layer_sel, LayerSelection):
            pass
        else:
            layer_sel = LayerSelection()

        return cls(
            adapter_id=data['adapter_id'],
            targets=LoRATargetType(data.get('targets', 'qkv')),
            layer_selection=layer_sel,
            insertion_point=InsertionPointType(
                data.get('insertion_point', 'post_projection')
            ),
            per_layer_config=per_layer,
            enabled=data.get('enabled', True),
        )


@dataclass
class CompletionRequest:
    """
    OpenAI-compatible completion request with HE-LoRA extensions.

    Standard fields follow OpenAI API spec.
    Extended with 'helora_config' for insertion point configuration.
    """
    # Required
    model: str
    prompt: Union[str, List[str], List[int], List[List[int]]]

    # Optional standard fields
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: int = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # HE-LoRA extension
    helora_config: Optional[InsertionPointSchema] = None

    def to_dict(self) -> Dict:
        result = {
            'model': self.model,
            'prompt': self.prompt,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'stream': self.stream,
            'echo': self.echo,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'best_of': self.best_of,
        }
        if self.logprobs is not None:
            result['logprobs'] = self.logprobs
        if self.stop is not None:
            result['stop'] = self.stop
        if self.logit_bias is not None:
            result['logit_bias'] = self.logit_bias
        if self.user is not None:
            result['user'] = self.user
        if self.helora_config is not None:
            result['helora_config'] = self.helora_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'CompletionRequest':
        helora_config = None
        if 'helora_config' in data and data['helora_config']:
            helora_config = InsertionPointSchema.from_dict(data['helora_config'])

        return cls(
            model=data['model'],
            prompt=data['prompt'],
            max_tokens=data.get('max_tokens', 16),
            temperature=data.get('temperature', 1.0),
            top_p=data.get('top_p', 1.0),
            n=data.get('n', 1),
            stream=data.get('stream', False),
            logprobs=data.get('logprobs'),
            echo=data.get('echo', False),
            stop=data.get('stop'),
            presence_penalty=data.get('presence_penalty', 0.0),
            frequency_penalty=data.get('frequency_penalty', 0.0),
            best_of=data.get('best_of', 1),
            logit_bias=data.get('logit_bias'),
            user=data.get('user'),
            helora_config=helora_config,
        )


@dataclass
class ChatMessage:
    """A message in a chat conversation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict:
        result = {'role': self.role, 'content': self.content}
        if self.name is not None:
            result['name'] = self.name
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        return cls(
            role=data['role'],
            content=data['content'],
            name=data.get('name'),
        )


@dataclass
class ChatCompletionRequest:
    """
    OpenAI-compatible chat completion request with HE-LoRA extensions.
    """
    # Required
    model: str
    messages: List[ChatMessage]

    # Optional standard fields
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # HE-LoRA extension
    helora_config: Optional[InsertionPointSchema] = None

    def to_dict(self) -> Dict:
        result = {
            'model': self.model,
            'messages': [m.to_dict() for m in self.messages],
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'stream': self.stream,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
        }
        if self.max_tokens is not None:
            result['max_tokens'] = self.max_tokens
        if self.stop is not None:
            result['stop'] = self.stop
        if self.logit_bias is not None:
            result['logit_bias'] = self.logit_bias
        if self.user is not None:
            result['user'] = self.user
        if self.helora_config is not None:
            result['helora_config'] = self.helora_config.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatCompletionRequest':
        messages = [ChatMessage.from_dict(m) for m in data['messages']]

        helora_config = None
        if 'helora_config' in data and data['helora_config']:
            helora_config = InsertionPointSchema.from_dict(data['helora_config'])

        return cls(
            model=data['model'],
            messages=messages,
            max_tokens=data.get('max_tokens'),
            temperature=data.get('temperature', 1.0),
            top_p=data.get('top_p', 1.0),
            n=data.get('n', 1),
            stream=data.get('stream', False),
            stop=data.get('stop'),
            presence_penalty=data.get('presence_penalty', 0.0),
            frequency_penalty=data.get('frequency_penalty', 0.0),
            logit_bias=data.get('logit_bias'),
            user=data.get('user'),
            helora_config=helora_config,
        )


@dataclass
class Choice:
    """A completion choice."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'text': self.text,
            'index': self.index,
            'logprobs': self.logprobs,
            'finish_reason': self.finish_reason,
        }


@dataclass
class ChatChoice:
    """A chat completion choice."""
    message: ChatMessage
    index: int
    finish_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'message': self.message.to_dict(),
            'index': self.index,
            'finish_reason': self.finish_reason,
        }


@dataclass
class Usage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    # HE-LoRA extension
    helora_stats: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        result = {
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
        }
        if self.helora_stats is not None:
            result['helora_stats'] = self.helora_stats
        return result


@dataclass
class CompletionResponse:
    """OpenAI-compatible completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None

    def to_dict(self) -> Dict:
        result = {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [c.to_dict() for c in self.choices],
        }
        if self.usage is not None:
            result['usage'] = self.usage.to_dict()
        return result


@dataclass
class ChatCompletionResponse:
    """OpenAI-compatible chat completion response."""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None

    def to_dict(self) -> Dict:
        result = {
            'id': self.id,
            'object': self.object,
            'created': self.created,
            'model': self.model,
            'choices': [c.to_dict() for c in self.choices],
        }
        if self.usage is not None:
            result['usage'] = self.usage.to_dict()
        return result


@dataclass
class ErrorResponse:
    """API error response."""
    error: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {'error': self.error}

    @classmethod
    def create(
        cls,
        message: str,
        type: str = "invalid_request_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
    ) -> 'ErrorResponse':
        error = {'message': message, 'type': type}
        if param is not None:
            error['param'] = param
        if code is not None:
            error['code'] = code
        return cls(error=error)


# Validation helpers

def validate_insertion_config(config: InsertionPointSchema, num_layers: int) -> List[str]:
    """
    Validate insertion point configuration.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    if not config.adapter_id:
        errors.append("adapter_id is required")

    # Validate layer selection
    layers = config.layer_selection.get_layers(num_layers)
    if not layers:
        errors.append("layer_selection results in empty layer list")

    for layer in layers:
        if layer < 0 or layer >= num_layers:
            errors.append(f"layer {layer} is out of bounds [0, {num_layers})")

    # Validate per-layer config
    if config.per_layer_config:
        for layer_idx in config.per_layer_config.keys():
            if layer_idx not in layers:
                errors.append(
                    f"per_layer_config layer {layer_idx} not in selected layers"
                )

    return errors
