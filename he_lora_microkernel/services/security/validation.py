"""
Request Validation for MSS

Validates incoming requests for security and correctness.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ValidationSeverity(str, Enum):
    """Severity of validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """A validation issue found in a request."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None


@dataclass
class ValidationResult:
    """Result of request validation."""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    sanitized_request: Optional[Any] = None

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL)]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]


class RequestValidator:
    """
    Validates incoming API requests.

    Checks:
    - Required fields
    - Field types and ranges
    - Injection prevention
    - Rate limiting (placeholder)
    """

    # Allowed model ID patterns
    MODEL_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-./]+$')

    # Allowed adapter ID patterns
    ADAPTER_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_\-]+$')

    # Maximum lengths
    MAX_PROMPT_LENGTH = 100000
    MAX_MODEL_ID_LENGTH = 256
    MAX_ADAPTER_ID_LENGTH = 128

    def __init__(self):
        self._validators = {}

    def validate_completion_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate a completion request."""
        issues = []

        # Required fields
        if 'model' not in request:
            issues.append(ValidationIssue(
                field='model',
                message='model is required',
                severity=ValidationSeverity.ERROR,
            ))

        if 'prompt' not in request:
            issues.append(ValidationIssue(
                field='prompt',
                message='prompt is required',
                severity=ValidationSeverity.ERROR,
            ))

        # Validate model ID
        if 'model' in request:
            model_issues = self._validate_model_id(request['model'])
            issues.extend(model_issues)

        # Validate prompt
        if 'prompt' in request:
            prompt_issues = self._validate_prompt(request['prompt'])
            issues.extend(prompt_issues)

        # Validate max_tokens
        if 'max_tokens' in request:
            if not isinstance(request['max_tokens'], int) or request['max_tokens'] < 1:
                issues.append(ValidationIssue(
                    field='max_tokens',
                    message='max_tokens must be a positive integer',
                    severity=ValidationSeverity.ERROR,
                    value=request['max_tokens'],
                ))
            elif request['max_tokens'] > 4096:
                issues.append(ValidationIssue(
                    field='max_tokens',
                    message='max_tokens exceeds maximum (4096)',
                    severity=ValidationSeverity.WARNING,
                    value=request['max_tokens'],
                ))

        # Validate temperature
        if 'temperature' in request:
            temp = request['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                issues.append(ValidationIssue(
                    field='temperature',
                    message='temperature must be between 0 and 2',
                    severity=ValidationSeverity.ERROR,
                    value=temp,
                ))

        # Validate HE-LoRA config
        if 'helora_config' in request and request['helora_config']:
            helora_issues = self._validate_helora_config(request['helora_config'])
            issues.extend(helora_issues)

        valid = not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues)

        return ValidationResult(
            valid=valid,
            issues=issues,
            sanitized_request=self._sanitize_request(request) if valid else None,
        )

    def validate_chat_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate a chat completion request."""
        issues = []

        # Required fields
        if 'model' not in request:
            issues.append(ValidationIssue(
                field='model',
                message='model is required',
                severity=ValidationSeverity.ERROR,
            ))

        if 'messages' not in request:
            issues.append(ValidationIssue(
                field='messages',
                message='messages is required',
                severity=ValidationSeverity.ERROR,
            ))

        # Validate model ID
        if 'model' in request:
            model_issues = self._validate_model_id(request['model'])
            issues.extend(model_issues)

        # Validate messages
        if 'messages' in request:
            msg_issues = self._validate_messages(request['messages'])
            issues.extend(msg_issues)

        # Validate HE-LoRA config
        if 'helora_config' in request and request['helora_config']:
            helora_issues = self._validate_helora_config(request['helora_config'])
            issues.extend(helora_issues)

        valid = not any(i.severity in (ValidationSeverity.ERROR, ValidationSeverity.CRITICAL) for i in issues)

        return ValidationResult(
            valid=valid,
            issues=issues,
            sanitized_request=self._sanitize_request(request) if valid else None,
        )

    def _validate_model_id(self, model_id: Any) -> List[ValidationIssue]:
        """Validate model ID."""
        issues = []

        if not isinstance(model_id, str):
            issues.append(ValidationIssue(
                field='model',
                message='model must be a string',
                severity=ValidationSeverity.ERROR,
                value=model_id,
            ))
            return issues

        if len(model_id) > self.MAX_MODEL_ID_LENGTH:
            issues.append(ValidationIssue(
                field='model',
                message=f'model ID too long (max {self.MAX_MODEL_ID_LENGTH})',
                severity=ValidationSeverity.ERROR,
                value=model_id,
            ))

        if not self.MODEL_ID_PATTERN.match(model_id):
            issues.append(ValidationIssue(
                field='model',
                message='model ID contains invalid characters',
                severity=ValidationSeverity.ERROR,
                value=model_id,
            ))

        return issues

    def _validate_prompt(self, prompt: Any) -> List[ValidationIssue]:
        """Validate prompt field."""
        issues = []

        if isinstance(prompt, str):
            if len(prompt) > self.MAX_PROMPT_LENGTH:
                issues.append(ValidationIssue(
                    field='prompt',
                    message=f'prompt too long (max {self.MAX_PROMPT_LENGTH})',
                    severity=ValidationSeverity.ERROR,
                ))

        elif isinstance(prompt, list):
            if not prompt:
                issues.append(ValidationIssue(
                    field='prompt',
                    message='prompt list cannot be empty',
                    severity=ValidationSeverity.ERROR,
                ))

        else:
            issues.append(ValidationIssue(
                field='prompt',
                message='prompt must be string or list',
                severity=ValidationSeverity.ERROR,
            ))

        return issues

    def _validate_messages(self, messages: Any) -> List[ValidationIssue]:
        """Validate messages field."""
        issues = []

        if not isinstance(messages, list):
            issues.append(ValidationIssue(
                field='messages',
                message='messages must be a list',
                severity=ValidationSeverity.ERROR,
            ))
            return issues

        if not messages:
            issues.append(ValidationIssue(
                field='messages',
                message='messages cannot be empty',
                severity=ValidationSeverity.ERROR,
            ))
            return issues

        valid_roles = {'system', 'user', 'assistant'}

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                issues.append(ValidationIssue(
                    field=f'messages[{i}]',
                    message='message must be a dict',
                    severity=ValidationSeverity.ERROR,
                ))
                continue

            if 'role' not in msg:
                issues.append(ValidationIssue(
                    field=f'messages[{i}].role',
                    message='role is required',
                    severity=ValidationSeverity.ERROR,
                ))
            elif msg['role'] not in valid_roles:
                issues.append(ValidationIssue(
                    field=f'messages[{i}].role',
                    message=f'invalid role: {msg["role"]}',
                    severity=ValidationSeverity.ERROR,
                ))

            if 'content' not in msg:
                issues.append(ValidationIssue(
                    field=f'messages[{i}].content',
                    message='content is required',
                    severity=ValidationSeverity.ERROR,
                ))

        return issues

    def _validate_helora_config(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate HE-LoRA configuration."""
        issues = []

        # Validate adapter_id
        if 'adapter_id' not in config:
            issues.append(ValidationIssue(
                field='helora_config.adapter_id',
                message='adapter_id is required',
                severity=ValidationSeverity.ERROR,
            ))
        elif not self.ADAPTER_ID_PATTERN.match(config['adapter_id']):
            issues.append(ValidationIssue(
                field='helora_config.adapter_id',
                message='adapter_id contains invalid characters',
                severity=ValidationSeverity.ERROR,
                value=config['adapter_id'],
            ))

        # Validate targets
        if 'targets' in config:
            if config['targets'] not in ('qkv', 'qkvo'):
                issues.append(ValidationIssue(
                    field='helora_config.targets',
                    message='targets must be "qkv" or "qkvo"',
                    severity=ValidationSeverity.ERROR,
                    value=config['targets'],
                ))

        # Validate layer_selection
        if 'layer_selection' in config:
            layer_sel = config['layer_selection']
            if isinstance(layer_sel, dict):
                if 'mode' in layer_sel:
                    valid_modes = ['all', 'range', 'list', 'pattern']
                    if layer_sel['mode'] not in valid_modes:
                        issues.append(ValidationIssue(
                            field='helora_config.layer_selection.mode',
                            message=f'invalid mode: {layer_sel["mode"]}',
                            severity=ValidationSeverity.ERROR,
                        ))

        return issues

    def _sanitize_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request by removing potentially dangerous fields."""
        sanitized = dict(request)

        # Remove any unexpected fields that could be injection attempts
        allowed_fields = {
            'model', 'prompt', 'messages', 'max_tokens', 'temperature',
            'top_p', 'n', 'stream', 'stop', 'presence_penalty',
            'frequency_penalty', 'logit_bias', 'user', 'helora_config',
        }

        for key in list(sanitized.keys()):
            if key not in allowed_fields:
                del sanitized[key]

        return sanitized
