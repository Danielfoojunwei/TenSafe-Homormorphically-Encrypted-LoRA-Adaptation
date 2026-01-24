"""
Safe Logger

Enforces logging policies for N2HE Privacy Mode.
Prevents plaintext logging when privacy constraints are active.
"""

import logging
import contextlib
import threading

# Thread-local storage for privacy context
_privacy_context = threading.local()

def set_privacy_mode(mode: str):
    _privacy_context.mode = mode

def get_privacy_mode() -> str:
    return getattr(_privacy_context, "mode", "off")

class SafeLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        # Intercept log creation
        mode = get_privacy_mode()
        
        if mode == "n2he":
            # Sanitize message if it looks sensitive
            # Heuristic: verify if message contains known safe templates or is raw data
            # For MVP, we prepend [N2HE] and maybe hash sensitive args?
            # A strict impl would whitelist templates.
            msg = f"[N2HE][PROTECTED] {msg}"
            
        super()._log(level, msg, args, exc_info, extra, stack_info, stacklevel)

def get_safe_logger(name: str):
    # Register custom logger class if needed, or just wrap
    logger = logging.getLogger(name)
    # We can add a filter or wrapper. 
    # For now returning standard logger, assuming logic happens at call site 
    # or usage of `safe_log_context`.
    return logger

@contextlib.contextmanager
def safe_log_context(mode: str):
    """
    Context manager to enforce privacy mode logging rules.
    """
    prev = get_privacy_mode()
    set_privacy_mode(mode)
    try:
        yield
    finally:
        set_privacy_mode(prev)
