"""
TG-Tinker SDK exceptions.

This module defines all custom exceptions used in the SDK.
"""

from typing import Any, Dict, Optional


class TGTinkerError(Exception):
    """Base exception for TG-Tinker SDK errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or "TG_TINKER_ERROR"
        self.details = details or {}
        self.request_id = request_id

    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.request_id:
            parts.append(f"(request_id: {self.request_id})")
        return " ".join(parts)


class AuthenticationError(TGTinkerError):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="AUTHENTICATION_REQUIRED",
            details=details,
            request_id=request_id,
        )


class PermissionDeniedError(TGTinkerError):
    """Raised when permission is denied."""

    def __init__(
        self,
        message: str = "Permission denied",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="PERMISSION_DENIED",
            details=details,
            request_id=request_id,
        )


class TrainingClientNotFoundError(TGTinkerError):
    """Raised when a training client is not found."""

    def __init__(
        self,
        training_client_id: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Training client with ID '{training_client_id}' not found",
            code="TRAINING_CLIENT_NOT_FOUND",
            details={"training_client_id": training_client_id},
            request_id=request_id,
        )


class FutureNotFoundError(TGTinkerError):
    """Raised when a future is not found."""

    def __init__(
        self,
        future_id: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Future with ID '{future_id}' not found",
            code="FUTURE_NOT_FOUND",
            details={"future_id": future_id},
            request_id=request_id,
        )


class ArtifactNotFoundError(TGTinkerError):
    """Raised when an artifact is not found."""

    def __init__(
        self,
        artifact_id: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Artifact with ID '{artifact_id}' not found",
            code="ARTIFACT_NOT_FOUND",
            details={"artifact_id": artifact_id},
            request_id=request_id,
        )


class ValidationError(TGTinkerError):
    """Raised when request validation fails."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            request_id=request_id,
        )


class RateLimitedError(TGTinkerError):
    """Raised when rate limited."""

    def __init__(
        self,
        retry_after: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message="Rate limit exceeded",
            code="RATE_LIMITED",
            details={"retry_after": retry_after} if retry_after else {},
            request_id=request_id,
        )
        self.retry_after = retry_after


class QueueFullError(TGTinkerError):
    """Raised when the operation queue is full."""

    def __init__(
        self,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message="Operation queue is full, please try again later",
            code="QUEUE_FULL",
            details={},
            request_id=request_id,
        )


class FutureTimeoutError(TGTinkerError):
    """Raised when a future times out."""

    def __init__(
        self,
        future_id: str,
        timeout: float,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Future '{future_id}' did not complete within {timeout}s",
            code="FUTURE_TIMEOUT",
            details={"future_id": future_id, "timeout": timeout},
            request_id=request_id,
        )


class FutureCancelledError(TGTinkerError):
    """Raised when a future was cancelled."""

    def __init__(
        self,
        future_id: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Future '{future_id}' was cancelled",
            code="FUTURE_CANCELLED",
            details={"future_id": future_id},
            request_id=request_id,
        )


class FutureFailedError(TGTinkerError):
    """Raised when a future failed with an error."""

    def __init__(
        self,
        future_id: str,
        error_message: str,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"Future '{future_id}' failed: {error_message}",
            code="FUTURE_FAILED",
            details={"future_id": future_id, "error_message": error_message},
            request_id=request_id,
        )


class DPBudgetExceededError(TGTinkerError):
    """Raised when the differential privacy budget is exceeded."""

    def __init__(
        self,
        current_epsilon: float,
        max_epsilon: float,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=f"DP budget exceeded: current epsilon {current_epsilon} >= max {max_epsilon}",
            code="DP_BUDGET_EXCEEDED",
            details={"current_epsilon": current_epsilon, "max_epsilon": max_epsilon},
            request_id=request_id,
        )


class ServerError(TGTinkerError):
    """Raised for internal server errors."""

    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            code="INTERNAL_ERROR",
            details=details,
            request_id=request_id,
        )


class ConnectionError(TGTinkerError):
    """Raised when connection to server fails."""

    def __init__(
        self,
        message: str = "Failed to connect to server",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code="CONNECTION_ERROR",
            details=details,
        )
