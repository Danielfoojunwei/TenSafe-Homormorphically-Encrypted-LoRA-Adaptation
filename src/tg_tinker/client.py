"""
TG-Tinker SDK ServiceClient module.

This module provides the ServiceClient class, the primary entry point
for interacting with the TG-Tinker API.
"""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import httpx

from .config import TenSafeConfig, get_config, validate_api_key
from .exceptions import (
    ArtifactNotFoundError,
    AuthenticationError,
    FutureNotFoundError,
    PermissionDeniedError,
    QueueFullError,
    RateLimitedError,
    ServerError,
    TrainingClientNotFoundError,
    ValidationError,
)
from .schemas import (
    AuditLogEntry,
    CreateTrainingClientRequest,
    CreateTrainingClientResponse,
    ForwardBackwardRequest,
    FutureResponse,
    FutureResultResponse,
    LoadStateRequest,
    LoadStateResult,
    LoRAConfig,
    OptimStepRequest,
    SampleRequest,
    SampleResult,
    SaveStateRequest,
    SaveStateResult,
    TrainingClientInfo,
    TrainingConfig,
)
from .training_client import TrainingClient


class ServiceClient:
    """
    Primary entry point for the TG-Tinker SDK.

    ServiceClient manages authentication, HTTP communication, and provides
    methods to create and manage TrainingClients.

    Example:
        >>> from tg_tinker import ServiceClient, TrainingConfig, LoRAConfig
        >>> service = ServiceClient()  # Uses env vars
        >>> config = TrainingConfig(
        ...     model_ref="meta-llama/Llama-3-8B",
        ...     lora_config=LoRAConfig(rank=16)
        ... )
        >>> tc = service.create_training_client(config)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        """
        Initialize the ServiceClient.

        Configuration is loaded from environment variables and can be
        overridden with explicit parameters.

        Args:
            base_url: API base URL (overrides TG_TINKER_BASE_URL)
            api_key: API key (overrides TG_TINKER_API_KEY)
            tenant_id: Tenant ID (overrides TG_TINKER_TENANT_ID)
            timeout: Request timeout in seconds (overrides TG_TINKER_TIMEOUT)
            **kwargs: Additional configuration options

        Raises:
            ValueError: If no API key is provided
        """
        self._config = get_config(
            api_key=api_key,
            base_url=base_url,
            tenant_id=tenant_id,
            timeout=timeout,
            **kwargs,
        )

        # Validate API key
        self._api_key = validate_api_key(self._config.api_key)
        self._base_url = self._config.base_url.rstrip("/")
        self._tenant_id = self._config.tenant_id

        # Initialize HTTP client
        self._http_client = httpx.Client(
            base_url=f"{self._base_url}/v1",
            timeout=self._config.timeout,
            verify=self._config.verify_ssl,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "tg-tinker-sdk/1.0.0",
            },
        )

    def __enter__(self) -> "ServiceClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._http_client.close()

    # ==========================================================================
    # Public API Methods
    # ==========================================================================

    def create_training_client(
        self,
        config: TrainingConfig,
    ) -> TrainingClient:
        """
        Create a new training client.

        Args:
            config: Training configuration

        Returns:
            TrainingClient instance

        Example:
            >>> config = TrainingConfig(
            ...     model_ref="meta-llama/Llama-3-8B",
            ...     lora_config=LoRAConfig(rank=16, alpha=32)
            ... )
            >>> tc = service.create_training_client(config)
        """
        request = CreateTrainingClientRequest(
            model_ref=config.model_ref,
            lora_config=config.lora_config,
            optimizer=config.optimizer,
            dp_config=config.dp_config,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_steps=config.max_steps,
            metadata=config.metadata,
        )

        response = self._request(
            "POST",
            "/training_clients",
            json=request.model_dump(mode="json"),
        )

        tc_response = CreateTrainingClientResponse.model_validate(response)

        return TrainingClient(
            training_client_id=tc_response.training_client_id,
            client=self,
            config=tc_response.config,
            step=tc_response.step,
            dp_metrics=None,
        )

    def get_training_client(self, training_client_id: str) -> TrainingClientInfo:
        """
        Get information about a training client.

        Args:
            training_client_id: ID of the training client

        Returns:
            TrainingClientInfo with current state

        Raises:
            TrainingClientNotFoundError: If training client doesn't exist
        """
        response = self._request("GET", f"/training_clients/{training_client_id}")
        return TrainingClientInfo.model_validate(response)

    def list_training_clients(self) -> List[TrainingClientInfo]:
        """
        List all training clients for the tenant.

        Returns:
            List of TrainingClientInfo objects
        """
        response = self._request("GET", "/training_clients")
        return [TrainingClientInfo.model_validate(item) for item in response]

    def get_future(self, future_id: str) -> FutureResponse:
        """
        Get the status of a future.

        Args:
            future_id: ID of the future

        Returns:
            FutureResponse with current status

        Raises:
            FutureNotFoundError: If future doesn't exist
        """
        return self._get_future_status(future_id)

    def get_audit_logs(
        self,
        training_client_id: Optional[str] = None,
        operation: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit logs.

        Args:
            training_client_id: Filter by training client ID
            operation: Filter by operation type
            limit: Maximum number of entries to return
            offset: Number of entries to skip

        Returns:
            List of AuditLogEntry objects
        """
        params = {"limit": limit, "offset": offset}
        if training_client_id:
            params["training_client_id"] = training_client_id
        if operation:
            params["operation"] = operation

        response = self._request("GET", "/audit_logs", params=params)
        return [AuditLogEntry.model_validate(item) for item in response]

    def pull_artifact(self, artifact_id: str) -> bytes:
        """
        Download an artifact's encrypted content.

        Args:
            artifact_id: ID of the artifact to download

        Returns:
            Encrypted artifact bytes

        Raises:
            ArtifactNotFoundError: If artifact doesn't exist
        """
        response = self._http_client.get(f"/artifacts/{artifact_id}/content")
        self._check_response(response)
        return response.content

    # ==========================================================================
    # Internal Methods (used by TrainingClient and FutureHandle)
    # ==========================================================================

    def _post_forward_backward(
        self,
        training_client_id: str,
        request: ForwardBackwardRequest,
    ) -> FutureResponse:
        """Post a forward-backward request."""
        response = self._request(
            "POST",
            f"/training_clients/{training_client_id}/forward_backward",
            json=request.model_dump(mode="json"),
        )
        return FutureResponse.model_validate(response)

    def _post_optim_step(
        self,
        training_client_id: str,
        request: OptimStepRequest,
    ) -> FutureResponse:
        """Post an optim step request."""
        response = self._request(
            "POST",
            f"/training_clients/{training_client_id}/optim_step",
            json=request.model_dump(mode="json"),
        )
        return FutureResponse.model_validate(response)

    def _post_sample(
        self,
        training_client_id: str,
        request: SampleRequest,
    ) -> SampleResult:
        """Post a sample request."""
        response = self._request(
            "POST",
            f"/training_clients/{training_client_id}/sample",
            json=request.model_dump(mode="json"),
        )
        return SampleResult.model_validate(response)

    def _post_save_state(
        self,
        training_client_id: str,
        request: SaveStateRequest,
    ) -> SaveStateResult:
        """Post a save state request."""
        response = self._request(
            "POST",
            f"/training_clients/{training_client_id}/save_state",
            json=request.model_dump(mode="json"),
        )
        return SaveStateResult.model_validate(response)

    def _post_load_state(
        self,
        training_client_id: str,
        request: LoadStateRequest,
    ) -> LoadStateResult:
        """Post a load state request."""
        response = self._request(
            "POST",
            f"/training_clients/{training_client_id}/load_state",
            json=request.model_dump(mode="json"),
        )
        return LoadStateResult.model_validate(response)

    def _get_future_status(self, future_id: str) -> FutureResponse:
        """Get the status of a future."""
        response = self._request("GET", f"/futures/{future_id}")
        return FutureResponse.model_validate(response)

    def _get_future_result(self, future_id: str) -> FutureResultResponse:
        """Get the result of a future."""
        response = self._request("GET", f"/futures/{future_id}/result")
        return FutureResultResponse.model_validate(response)

    def _cancel_future(self, future_id: str) -> bool:
        """Cancel a future."""
        try:
            self._request("POST", f"/futures/{future_id}/cancel")
            return True
        except Exception:
            return False

    # ==========================================================================
    # HTTP Request Handling
    # ==========================================================================

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method
            path: API path
            json: JSON body
            params: Query parameters

        Returns:
            Parsed JSON response

        Raises:
            Various TGTinkerError subclasses based on response status
        """
        last_error = None

        for attempt in range(self._config.retry_count + 1):
            try:
                response = self._http_client.request(
                    method=method,
                    url=path,
                    json=json,
                    params=params,
                )

                self._check_response(response)
                return response.json()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if attempt < self._config.retry_count:
                    backoff = self._config.retry_backoff * (2 ** attempt)
                    time.sleep(backoff)
                    continue
                raise

            except RateLimitedError as e:
                last_error = e
                if attempt < self._config.retry_count and e.retry_after:
                    time.sleep(min(e.retry_after, 60))
                    continue
                raise

        # Should not reach here, but just in case
        raise last_error

    def _check_response(self, response: httpx.Response) -> None:
        """
        Check HTTP response for errors.

        Args:
            response: HTTP response

        Raises:
            Appropriate exception based on status code
        """
        if response.is_success:
            return

        # Try to parse error response
        try:
            error_data = response.json()
            error = error_data.get("error", {})
            code = error.get("code", "UNKNOWN_ERROR")
            message = error.get("message", response.text)
            details = error.get("details", {})
            request_id = error.get("request_id")
        except Exception:
            code = "UNKNOWN_ERROR"
            message = response.text
            details = {}
            request_id = None

        # Map status codes to exceptions
        if response.status_code == 401:
            raise AuthenticationError(message, details, request_id)
        elif response.status_code == 403:
            raise PermissionDeniedError(message, details, request_id)
        elif response.status_code == 404:
            if "training_client" in code.lower():
                raise TrainingClientNotFoundError(
                    details.get("training_client_id", "unknown"),
                    request_id,
                )
            elif "future" in code.lower():
                raise FutureNotFoundError(
                    details.get("future_id", "unknown"),
                    request_id,
                )
            elif "artifact" in code.lower():
                raise ArtifactNotFoundError(
                    details.get("artifact_id", "unknown"),
                    request_id,
                )
            else:
                raise ServerError(message, details, request_id)
        elif response.status_code == 422:
            raise ValidationError(message, details, request_id)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitedError(
                int(retry_after) if retry_after else None,
                request_id,
            )
        elif response.status_code == 503:
            if "queue" in code.lower():
                raise QueueFullError(request_id)
            raise ServerError(message, details, request_id)
        else:
            raise ServerError(message, details, request_id)

    def __repr__(self) -> str:
        return f"ServiceClient(base_url={self._base_url!r})"
