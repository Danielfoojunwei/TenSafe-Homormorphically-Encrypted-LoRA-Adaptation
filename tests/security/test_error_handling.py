"""
Error Handling Security Tests.

These tests verify that:
1. All errors have machine-readable codes
2. Error messages don't leak sensitive information
3. Error codes are registered in the taxonomy
4. Structured logging filters sensitive data
"""

import json
import logging
from io import StringIO

import pytest

from tensorguard.errors import (
    ERROR_CODES,
    ConfigError,
    ConfigMissingError,
    ConfigValidationError,
    CryptoDecryptionError,
    CryptoError,
    CryptoKeyError,
    CryptoNonceReuseError,
    HEDecryptionError,
    HEEncryptionError,
    HEError,
    HEKeygenError,
    HELibraryNotFoundError,
    HEParameterMismatchError,
    HEToyModeError,
    TensorGuardError,
    TGSPError,
    TGSPFormatError,
    TGSPIntegrityError,
    TGSPRecipientError,
    TGSPVersionError,
    validate_error_code,
)
from tensorguard.logging import (
    DevelopmentFormatter,
    LogContext,
    StructuredFormatter,
    get_logger,
)


class TestErrorTaxonomy:
    """Test error code taxonomy."""

    def test_all_errors_have_registered_codes(self):
        """All error classes should use registered error codes."""
        errors_to_test = [
            TensorGuardError("test"),
            HEKeygenError("test"),
            HEEncryptionError("test"),
            HEDecryptionError("test"),
            HEParameterMismatchError("abc123", "def456"),
            HELibraryNotFoundError(),
            HEToyModeError("encrypt"),
            CryptoDecryptionError(),
            CryptoKeyError("test"),
            CryptoNonceReuseError(),
            TGSPFormatError("test"),
            TGSPVersionError("2.0", ["1.0"]),
            TGSPIntegrityError(),
            TGSPRecipientError("test"),
            ConfigMissingError("key"),
            ConfigValidationError("key", "reason"),
        ]

        for error in errors_to_test:
            assert error.code in ERROR_CODES, f"Error code {error.code} not registered"

    def test_error_codes_are_prefixed(self):
        """All error codes should have TG_ prefix."""
        for code in ERROR_CODES:
            assert code.startswith("TG_"), f"Code {code} missing TG_ prefix"

    def test_validate_error_code(self):
        """validate_error_code should work correctly."""
        assert validate_error_code("TG_HE_KEYGEN_FAILED") is True
        assert validate_error_code("INVALID_CODE") is False

    def test_error_to_dict(self):
        """Errors should serialize to dict correctly."""
        error = HEKeygenError(
            reason="test failure",
            params_hash="abc123",
            request_id="req-456",
        )
        d = error.to_dict()

        assert d["code"] == "TG_HE_KEYGEN_FAILED"
        assert "test failure" in d["message"]
        assert d["details"]["params_hash"] == "abc123"
        assert d["request_id"] == "req-456"


class TestErrorSecurityInvariants:
    """Test that errors don't leak sensitive information."""

    def test_he_param_mismatch_truncates_hashes(self):
        """Parameter hashes should be truncated to prevent full key exposure."""
        long_hash = "a" * 64
        error = HEParameterMismatchError(long_hash, long_hash)

        # Should truncate and add ...
        assert len(error.details["expected_hash"]) < 64
        assert "..." in error.details["expected_hash"]

    def test_library_not_found_hides_full_paths(self):
        """Library not found should not expose full search paths."""
        paths = ["/home/user/.secret/lib", "/etc/sensitive/lib"]
        error = HELibraryNotFoundError(search_paths=paths)

        # Should only show count, not actual paths
        assert "searched_paths_count" in error.details
        assert error.details["searched_paths_count"] == 2
        assert "/home/user" not in str(error)
        assert "/etc/sensitive" not in str(error)

    def test_error_str_format(self):
        """Error string should include code and request_id."""
        error = HEKeygenError("test", request_id="req-123")
        s = str(error)

        assert "[TG_HE_KEYGEN_FAILED]" in s
        assert "req-123" in s

    def test_crypto_errors_no_key_material(self):
        """Crypto errors should never include key material."""
        error = CryptoKeyError(
            reason="Invalid key format",
            key_type="AES-256",
        )

        # Key type is OK, but actual key bytes should never be in error
        assert "AES-256" in str(error) or "AES-256" in str(error.details)
        # Make sure we can't accidentally pass key bytes
        assert "key_bytes" not in error.details
        assert "key_data" not in error.details


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_structured_formatter_json_output(self):
        """StructuredFormatter should produce valid JSON."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("test.structured")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("Test message", extra={"request_id": "abc123"})

        output = stream.getvalue()
        data = json.loads(output)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert "extra" in data
        assert data["extra"]["request_id"] == "abc123"

    def test_sensitive_data_filtering(self):
        """Structured logs should filter sensitive fields."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("test.sensitive")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        # Log with sensitive data
        logger.info(
            "Auth attempt",
            extra={
                "user_id": "user-123",
                "password": "secret123",
                "api_key": "key-456",
                "access_token": "token-789",
            },
        )

        output = stream.getvalue()
        data = json.loads(output)

        # Non-sensitive should be present
        assert data["extra"]["user_id"] == "user-123"

        # Sensitive should be redacted
        assert data["extra"]["password"] == "[REDACTED]"
        assert data["extra"]["api_key"] == "[REDACTED]"
        assert data["extra"]["access_token"] == "[REDACTED]"

    def test_nested_sensitive_data_filtering(self):
        """Nested sensitive data should also be filtered."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = logging.getLogger("test.nested")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info(
            "Nested data",
            extra={
                "config": {
                    "endpoint": "https://api.example.com",
                    "secret_key": "mysecret",
                }
            },
        )

        output = stream.getvalue()
        data = json.loads(output)

        assert data["extra"]["config"]["endpoint"] == "https://api.example.com"
        assert data["extra"]["config"]["secret_key"] == "[REDACTED]"

    def test_development_formatter_readable(self):
        """DevelopmentFormatter should produce human-readable output."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(DevelopmentFormatter())

        logger = logging.getLogger("test.dev")
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)

        logger.info("Test message", extra={"request_id": "abc123"})

        output = stream.getvalue()

        assert "INFO" in output
        assert "Test message" in output
        assert "abc123" in output  # Request ID should be included

    def test_get_logger_namespacing(self):
        """get_logger should namespace under tensorguard."""
        logger1 = get_logger("mymodule")
        assert logger1.name == "tensorguard.mymodule"

        logger2 = get_logger("tensorguard.existing")
        assert logger2.name == "tensorguard.existing"


class TestLogContext:
    """Test logging context manager."""

    def test_log_context_basic(self):
        """LogContext should store context."""
        with LogContext(request_id="abc", user="test") as ctx:
            context = LogContext.get_current()
            assert context["request_id"] == "abc"
            assert context["user"] == "test"

        # After context, should be empty
        assert LogContext.get_current() == {}

    def test_log_context_nested(self):
        """Nested LogContext should work correctly."""
        with LogContext(outer="value1"):
            assert LogContext.get_current()["outer"] == "value1"

            with LogContext(inner="value2"):
                ctx = LogContext.get_current()
                assert ctx["inner"] == "value2"
                assert "outer" not in ctx  # Inner context replaces outer

            # Back to outer context
            assert LogContext.get_current()["outer"] == "value1"


class TestErrorHierarchy:
    """Test error class hierarchy."""

    def test_all_errors_inherit_from_base(self):
        """All errors should inherit from TensorGuardError."""
        assert issubclass(HEError, TensorGuardError)
        assert issubclass(CryptoError, TensorGuardError)
        assert issubclass(TGSPError, TensorGuardError)
        assert issubclass(ConfigError, TensorGuardError)

    def test_specific_errors_inherit_from_category(self):
        """Specific errors should inherit from category base."""
        assert issubclass(HEKeygenError, HEError)
        assert issubclass(HEEncryptionError, HEError)
        assert issubclass(CryptoDecryptionError, CryptoError)
        assert issubclass(TGSPFormatError, TGSPError)
        assert issubclass(ConfigMissingError, ConfigError)

    def test_errors_are_catchable_by_base(self):
        """Should be able to catch all errors by base class."""
        errors = [
            HEKeygenError("test"),
            CryptoDecryptionError(),
            TGSPFormatError("test"),
            ConfigMissingError("key"),
        ]

        for error in errors:
            try:
                raise error
            except TensorGuardError as e:
                assert e.code is not None
            except Exception:
                pytest.fail(f"Error {type(error)} not caught by TensorGuardError")
