"""
TG-Tinker SDK configuration.

This module handles environment variables and SDK configuration.
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class TGTinkerConfig(BaseSettings):
    """Configuration for the TG-Tinker SDK."""

    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication",
    )
    base_url: str = Field(
        default="https://api.tensorguard.io",
        description="Base URL for TG-Tinker API",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant ID (if not derived from API key)",
    )
    timeout: float = Field(
        default=300.0,
        ge=1.0,
        description="Default request timeout in seconds",
    )
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retries for failed requests",
    )
    retry_backoff: float = Field(
        default=1.0,
        ge=0.1,
        description="Base backoff time for retries in seconds",
    )
    verify_ssl: bool = Field(
        default=True,
        description="Verify SSL certificates",
    )
    poll_interval: float = Field(
        default=1.0,
        ge=0.1,
        description="Interval between future status polls in seconds",
    )

    model_config = {
        "env_prefix": "TG_TINKER_",
        "env_file": ".env",
        "extra": "ignore",
    }


def get_config(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    tenant_id: Optional[str] = None,
    **kwargs,
) -> TGTinkerConfig:
    """
    Get TG-Tinker configuration.

    Explicit parameters override environment variables.

    Args:
        api_key: API key (overrides TG_TINKER_API_KEY env var)
        base_url: Base URL (overrides TG_TINKER_BASE_URL env var)
        tenant_id: Tenant ID (overrides TG_TINKER_TENANT_ID env var)
        **kwargs: Additional configuration options

    Returns:
        TGTinkerConfig instance
    """
    # Build config from environment first
    config = TGTinkerConfig()

    # Override with explicit parameters
    if api_key is not None:
        config = config.model_copy(update={"api_key": api_key})
    if base_url is not None:
        config = config.model_copy(update={"base_url": base_url})
    if tenant_id is not None:
        config = config.model_copy(update={"tenant_id": tenant_id})

    # Apply any additional kwargs
    if kwargs:
        valid_fields = set(TGTinkerConfig.model_fields.keys())
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if updates:
            config = config.model_copy(update=updates)

    return config


def validate_api_key(api_key: Optional[str]) -> str:
    """
    Validate that an API key is provided.

    Args:
        api_key: The API key to validate

    Returns:
        The validated API key

    Raises:
        ValueError: If no API key is provided
    """
    if not api_key:
        raise ValueError(
            "API key is required. Set TG_TINKER_API_KEY environment variable "
            "or pass api_key parameter to ServiceClient."
        )
    if not api_key.startswith("tg-"):
        # Warning only, don't fail for flexibility
        pass
    return api_key
