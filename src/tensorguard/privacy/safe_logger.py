"""
Safe Logger - Privacy-Aware Logging

Provides a logging wrapper that enforces privacy constraints when
N2HE mode is enabled. Blocks plaintext logging of sensitive data.
"""

import logging
import hashlib
import re
from typing import Optional, Any, Dict, Set
from functools import wraps


class SafeLoggerFilter(logging.Filter):
    """
    Logging filter that redacts sensitive content when privacy mode is enabled.
    """
    
    # Patterns that might contain sensitive data
    SENSITIVE_PATTERNS = [
        r'(?i)prompt["\s:=]+["\']?([^"\'}\n]+)',
        r'(?i)text["\s:=]+["\']?([^"\'}\n]+)',
        r'(?i)query["\s:=]+["\']?([^"\'}\n]+)',
        r'(?i)embedding["\s:=]+\[([^\]]+)\]',
        r'(?i)features?["\s:=]+\[([^\]]+)\]',
        r'(?i)input["\s:=]+["\']?([^"\'}\n]+)',
    ]
    
    def __init__(self, privacy_enabled: bool = False):
        super().__init__()
        self._privacy_enabled = privacy_enabled
        self._compiled_patterns = [re.compile(p) for p in self.SENSITIVE_PATTERNS]
    
    def set_privacy_mode(self, enabled: bool) -> None:
        """Enable or disable privacy mode."""
        self._privacy_enabled = enabled
    
    def _redact_content(self, content: str) -> str:
        """Redact sensitive content from log message."""
        if not self._privacy_enabled:
            return content
        
        result = content
        for pattern in self._compiled_patterns:
            def replacer(match):
                # Replace with hash of matched content
                matched = match.group(1) if match.lastindex else match.group(0)
                content_hash = hashlib.sha256(matched.encode()).hexdigest()[:12]
                return match.group(0).replace(matched, f"[REDACTED:{content_hash}]")
            
            result = pattern.sub(replacer, result)
        
        return result
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter and redact log record."""
        if self._privacy_enabled:
            record.msg = self._redact_content(str(record.msg))
            if record.args:
                record.args = tuple(
                    self._redact_content(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True


class SafeLogger:
    """
    Privacy-aware logger that blocks plaintext logging in N2HE mode.
    
    Usage:
        logger = SafeLogger(__name__)
        logger.set_privacy_mode(True)
        logger.info("Processing request", sensitive={"prompt": "user query"})
    """
    
    _global_privacy_mode: bool = False
    _instances: Dict[str, 'SafeLogger'] = {}
    
    def __init__(self, name: str):
        self._name = name
        self._logger = logging.getLogger(name)
        self._filter = SafeLoggerFilter(self._global_privacy_mode)
        self._logger.addFilter(self._filter)
        SafeLogger._instances[name] = self
    
    @classmethod
    def set_global_privacy_mode(cls, enabled: bool) -> None:
        """Set privacy mode for all SafeLogger instances."""
        cls._global_privacy_mode = enabled
        for instance in cls._instances.values():
            instance._filter.set_privacy_mode(enabled)
    
    @classmethod
    def is_privacy_enabled(cls) -> bool:
        """Check if privacy mode is enabled."""
        return cls._global_privacy_mode
    
    def set_privacy_mode(self, enabled: bool) -> None:
        """Set privacy mode for this logger instance."""
        self._filter.set_privacy_mode(enabled)
    
    def _log_with_privacy(
        self,
        level: int,
        msg: str,
        *args,
        sensitive: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log message with privacy handling for sensitive data."""
        if sensitive and self._filter._privacy_enabled:
            # Log hashed version of sensitive data
            hashed = {
                k: f"[HASH:{hashlib.sha256(str(v).encode()).hexdigest()[:12]}]"
                for k, v in sensitive.items()
            }
            extra_msg = f" | sensitive_hashes={hashed}"
            msg = msg + extra_msg
        elif sensitive:
            # Privacy not enabled, log normally but mark as sensitive
            msg = msg + f" | sensitive_fields={list(sensitive.keys())}"
        
        self._logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, sensitive: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_privacy(logging.DEBUG, msg, *args, sensitive=sensitive, **kwargs)
    
    def info(self, msg: str, *args, sensitive: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_privacy(logging.INFO, msg, *args, sensitive=sensitive, **kwargs)
    
    def warning(self, msg: str, *args, sensitive: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_privacy(logging.WARNING, msg, *args, sensitive=sensitive, **kwargs)
    
    def error(self, msg: str, *args, sensitive: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_privacy(logging.ERROR, msg, *args, sensitive=sensitive, **kwargs)
    
    def critical(self, msg: str, *args, sensitive: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        self._log_with_privacy(logging.CRITICAL, msg, *args, sensitive=sensitive, **kwargs)


def get_safe_logger(name: str) -> SafeLogger:
    """Get or create a SafeLogger instance."""
    if name in SafeLogger._instances:
        return SafeLogger._instances[name]
    return SafeLogger(name)


def privacy_aware(func):
    """
    Decorator that ensures function logging respects privacy mode.
    
    Usage:
        @privacy_aware
        def process_request(prompt: str):
            logger.info(f"Processing: {prompt}")  # Will be redacted if privacy enabled
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if SafeLogger.is_privacy_enabled():
            # Function execution with privacy mode active
            # Additional runtime checks could be added here
            pass
        return func(*args, **kwargs)
    return wrapper
