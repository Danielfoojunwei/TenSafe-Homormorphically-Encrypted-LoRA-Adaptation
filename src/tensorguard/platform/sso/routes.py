"""
SSO/OIDC FastAPI Routes.

Provides HTTP endpoints for SSO authentication:
- GET /auth/sso/{provider}/login - Initiate SSO flow
- GET /auth/sso/{provider}/callback - Handle OIDC callback
- GET /auth/saml/metadata - SAML SP metadata
- POST /auth/saml/acs - SAML Assertion Consumer Service
- GET /auth/sso/providers - List configured providers
"""

import hashlib
import json
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Cookie, Depends, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ..auth import create_access_token, create_refresh_token, ACCESS_TOKEN_EXPIRE_MINUTES
from ...admin.permissions import AdminUserContext, require_super_admin
from .models import (
    OIDCConfig,
    SAMLConfig,
    SSOCallbackRequest,
    SSOCallbackResponse,
    SSOLoginInitRequest,
    SSOLoginInitResponse,
    SSOProvider,
    SSOProviderInfo,
    SSOProviderType,
    SSOProvidersListResponse,
    SSOSession,
    SSOUser,
)
from .oidc import OIDCClient, OIDCError
from .providers import ProviderRegistry, get_provider_registry
from .saml import SAMLError, SAMLServiceProvider

logger = logging.getLogger(__name__)

# Router
router = APIRouter(prefix="/auth/sso", tags=["sso"])

# Environment configuration
TG_BASE_URL = os.getenv("TG_BASE_URL", "http://localhost:8000")
TG_SSO_SESSION_SECRET = os.getenv("TG_SSO_SESSION_SECRET")
if not TG_SSO_SESSION_SECRET:
    logger.warning("TG_SSO_SESSION_SECRET not set - generating ephemeral key")
    TG_SSO_SESSION_SECRET = secrets.token_hex(32)

# In-memory session store (use Redis/database in production)
_sso_sessions: Dict[str, SSOSession] = {}

# OIDC client cache
_oidc_clients: Dict[str, OIDCClient] = {}

# SAML SP cache
_saml_sps: Dict[str, SAMLServiceProvider] = {}


def _get_or_create_oidc_client(provider: SSOProvider) -> OIDCClient:
    """Get or create OIDC client for provider."""
    if provider.provider_id not in _oidc_clients:
        if not provider.oidc_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Provider missing OIDC configuration",
            )

        redirect_uri = f"{TG_BASE_URL}/auth/sso/{provider.provider_id}/callback"

        _oidc_clients[provider.provider_id] = OIDCClient(
            config=provider.oidc_config,
            provider_id=provider.provider_id,
            tenant_id=provider.tenant_id,
            redirect_uri=redirect_uri,
        )

    return _oidc_clients[provider.provider_id]


def _get_or_create_saml_sp(provider: SSOProvider) -> SAMLServiceProvider:
    """Get or create SAML SP for provider."""
    if provider.provider_id not in _saml_sps:
        if not provider.saml_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Provider missing SAML configuration",
            )

        _saml_sps[provider.provider_id] = SAMLServiceProvider(
            config=provider.saml_config,
            provider_id=provider.provider_id,
            tenant_id=provider.tenant_id,
        )

    return _saml_sps[provider.provider_id]


def _store_session(session: SSOSession) -> str:
    """Store SSO session and return session cookie value."""
    # Generate session token
    session_token = secrets.token_urlsafe(32)

    # Hash for storage lookup
    session_key = hashlib.sha256(session_token.encode()).hexdigest()

    _sso_sessions[session_key] = session

    # Clean up expired sessions periodically
    _cleanup_expired_sessions()

    return session_token


def _get_session(session_token: str) -> Optional[SSOSession]:
    """Retrieve SSO session by token."""
    session_key = hashlib.sha256(session_token.encode()).hexdigest()
    session = _sso_sessions.get(session_key)

    if session and datetime.now(timezone.utc) > session.expires_at:
        del _sso_sessions[session_key]
        return None

    return session


def _consume_session(session_token: str) -> Optional[SSOSession]:
    """Retrieve and mark session as used."""
    session_key = hashlib.sha256(session_token.encode()).hexdigest()
    session = _sso_sessions.get(session_key)

    if session:
        if session.used or datetime.now(timezone.utc) > session.expires_at:
            del _sso_sessions[session_key]
            return None

        session.used = True
        return session

    return None


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions from store."""
    now = datetime.now(timezone.utc)
    expired = [k for k, v in _sso_sessions.items() if now > v.expires_at]
    for k in expired:
        del _sso_sessions[k]


def _create_tensafe_tokens(user: SSOUser, provider: SSOProvider) -> Dict[str, Any]:
    """Create TenSafe JWT tokens for authenticated user."""
    # Determine role
    role = user.mapped_role or provider.default_role

    # Build token payload
    token_data = {
        "sub": user.email,
        "external_id": user.external_id,
        "tenant_id": user.tenant_id,
        "provider_id": user.provider_id,
        "role": role,
        "name": user.name,
        "sso_auth": True,
    }

    # Create tokens
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    }


# ==============================================================================
# Provider Discovery
# ==============================================================================


@router.get("/providers", response_model=SSOProvidersListResponse)
async def list_sso_providers(
    tenant_id: str = Query(..., description="Tenant ID"),
) -> SSOProvidersListResponse:
    """
    List available SSO providers for a tenant.

    Returns configured SSO providers that the user can authenticate with.
    """
    registry = get_provider_registry()
    providers = registry.get_by_tenant(tenant_id)

    default_provider = None
    default = registry.get_default(tenant_id)
    if default:
        default_provider = default.provider_id

    provider_infos = [
        SSOProviderInfo(
            provider_id=p.provider_id,
            name=p.name,
            display_name=p.display_name or f"Sign in with {p.name}",
            provider_type=p.provider_type,
            icon_url=p.icon_url,
            button_color=p.button_color,
            is_default=p.is_default,
        )
        for p in providers
    ]

    return SSOProvidersListResponse(
        providers=provider_infos,
        default_provider=default_provider,
    )


# ==============================================================================
# OIDC Flow
# ==============================================================================


@router.get("/{provider_id}/login", response_model=SSOLoginInitResponse)
async def initiate_sso_login(
    provider_id: str,
    response: Response,
    return_url: Optional[str] = Query(None, description="URL to redirect after login"),
    prompt: Optional[str] = Query(None, description="OIDC prompt parameter"),
    login_hint: Optional[str] = Query(None, description="Pre-fill login identifier"),
) -> SSOLoginInitResponse:
    """
    Initiate SSO login flow.

    For OIDC providers, returns the authorization URL to redirect the user.
    For SAML providers, returns the IdP SSO URL with AuthnRequest.
    """
    registry = get_provider_registry()
    provider = registry.get(provider_id)

    if not provider or not provider.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"Provider '{provider_id}' not found or disabled"},
        )

    # Validate return URL (prevent open redirect)
    if return_url:
        return_url = _validate_return_url(return_url)

    if provider.provider_type == SSOProviderType.OIDC:
        client = _get_or_create_oidc_client(provider)

        auth_url, session = await client.create_authorization_url(
            return_url=return_url,
            prompt=prompt,
            login_hint=login_hint,
        )

        # Store session and set cookie
        session_token = _store_session(session)
        response.set_cookie(
            key="sso_session",
            value=session_token,
            httponly=True,
            secure=os.getenv("TG_ENVIRONMENT") == "production",
            samesite="lax",
            max_age=600,  # 10 minutes
        )

        return SSOLoginInitResponse(
            authorization_url=auth_url,
            state=session.state,
            expires_in=600,
        )

    elif provider.provider_type == SSOProviderType.SAML:
        sp = _get_or_create_saml_sp(provider)

        sso_url, session = sp.create_authn_request(
            relay_state=return_url,
            force_authn=provider.force_reauthentication,
        )

        # Store session
        session_token = _store_session(session)
        response.set_cookie(
            key="sso_session",
            value=session_token,
            httponly=True,
            secure=os.getenv("TG_ENVIRONMENT") == "production",
            samesite="lax",
            max_age=600,
        )

        return SSOLoginInitResponse(
            authorization_url=sso_url,
            state=session.state,
            expires_in=600,
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsupported_provider", "message": f"Provider type '{provider.provider_type}' not supported"},
        )


@router.get("/{provider_id}/callback")
async def handle_oidc_callback(
    provider_id: str,
    request: Request,
    response: Response,
    code: Optional[str] = Query(None),
    state: str = Query(...),
    error: Optional[str] = Query(None),
    error_description: Optional[str] = Query(None),
    sso_session: Optional[str] = Cookie(None),
):
    """
    Handle OIDC authorization callback.

    Exchanges the authorization code for tokens and creates a TenSafe session.
    """
    # Check for OAuth error
    if error:
        logger.warning(f"OIDC error callback: {error} - {error_description}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": error,
                "error_description": error_description,
            },
        )

    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "missing_code", "message": "Authorization code not provided"},
        )

    # Validate session
    if not sso_session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "missing_session", "message": "SSO session not found"},
        )

    session = _consume_session(sso_session)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_session", "message": "SSO session expired or invalid"},
        )

    # Validate state (CSRF protection)
    if session.state != state:
        logger.warning(f"State mismatch: expected {session.state}, got {state}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "state_mismatch", "message": "Invalid state parameter"},
        )

    # Get provider
    registry = get_provider_registry()
    provider = registry.get(provider_id)

    if not provider or provider.provider_type != SSOProviderType.OIDC:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"OIDC provider '{provider_id}' not found"},
        )

    # Exchange code and validate tokens
    client = _get_or_create_oidc_client(provider)

    try:
        user = await client.process_callback(code, session)
    except OIDCError as e:
        logger.error(f"OIDC authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": e.error, "error_description": e.description},
        )

    # Validate user domain restriction
    if provider.allowed_domains:
        email_domain = user.email.split("@")[-1] if "@" in user.email else ""
        if email_domain not in provider.allowed_domains:
            logger.warning(f"User domain {email_domain} not in allowed list")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "domain_not_allowed", "message": "Email domain not authorized for this provider"},
            )

    # Create TenSafe tokens
    tokens = _create_tensafe_tokens(user, provider)

    # Clear SSO session cookie
    response.delete_cookie("sso_session")

    # Audit log
    logger.info(f"SSO login successful: {user.email} via {provider_id}")

    # Return tokens or redirect
    if session.return_url:
        # Redirect with tokens in fragment (for SPA)
        redirect_url = f"{session.return_url}#access_token={tokens['access_token']}&token_type=Bearer"
        return RedirectResponse(url=redirect_url, status_code=302)
    else:
        return SSOCallbackResponse(
            success=True,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="Bearer",
            expires_in=tokens["expires_in"],
            user=user,
        )


# ==============================================================================
# SAML Flow
# ==============================================================================


@router.get("/saml/metadata")
async def get_saml_metadata(
    provider_id: str = Query(..., description="SAML provider ID"),
) -> Response:
    """
    Get SAML Service Provider metadata.

    Returns the SP metadata XML for configuring the IdP.
    """
    registry = get_provider_registry()
    provider = registry.get(provider_id)

    if not provider or provider.provider_type != SSOProviderType.SAML:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"SAML provider '{provider_id}' not found"},
        )

    sp = _get_or_create_saml_sp(provider)
    metadata = sp.generate_metadata()

    return Response(
        content=metadata,
        media_type="application/samlmetadata+xml",
        headers={"Content-Disposition": f'attachment; filename="metadata-{provider_id}.xml"'},
    )


@router.post("/saml/acs")
async def handle_saml_acs(
    request: Request,
    response: Response,
    SAMLResponse: str = Form(...),
    RelayState: Optional[str] = Form(None),
    sso_session: Optional[str] = Cookie(None),
):
    """
    SAML Assertion Consumer Service endpoint.

    Handles SAML Response from IdP (both SP-initiated and IdP-initiated flows).
    """
    # Determine provider from RelayState or session
    provider_id = None
    expected_request_id = None

    if sso_session:
        session = _consume_session(sso_session)
        if session:
            provider_id = session.provider_id
            expected_request_id = session.state
            return_url = session.return_url or RelayState

    # Try to extract provider from RelayState if not found
    if not provider_id and RelayState:
        # RelayState might encode provider info
        try:
            relay_data = json.loads(RelayState)
            provider_id = relay_data.get("provider_id")
            return_url = relay_data.get("return_url")
        except (json.JSONDecodeError, TypeError):
            # RelayState might just be a URL
            return_url = RelayState

    if not provider_id:
        # Try to find the first SAML provider for IdP-initiated flow
        registry = get_provider_registry()
        for p in registry.list_all():
            if p.provider_type == SSOProviderType.SAML and p.is_enabled:
                provider_id = p.provider_id
                break

    if not provider_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "no_provider", "message": "Could not determine SAML provider"},
        )

    # Get provider
    registry = get_provider_registry()
    provider = registry.get(provider_id)

    if not provider or provider.provider_type != SSOProviderType.SAML:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"SAML provider '{provider_id}' not found"},
        )

    # Process SAML Response
    sp = _get_or_create_saml_sp(provider)

    try:
        user = sp.process_response(
            saml_response=SAMLResponse,
            relay_state=RelayState,
            expected_request_id=expected_request_id,
        )
    except SAMLError as e:
        logger.error(f"SAML authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": e.error, "error_description": e.description},
        )

    # Validate user domain restriction
    if provider.allowed_domains:
        email_domain = user.email.split("@")[-1] if "@" in user.email else ""
        if email_domain not in provider.allowed_domains:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "domain_not_allowed", "message": "Email domain not authorized"},
            )

    # Create TenSafe tokens
    tokens = _create_tensafe_tokens(user, provider)

    # Clear SSO session cookie
    response.delete_cookie("sso_session")

    # Audit log
    logger.info(f"SAML login successful: {user.email} via {provider_id}")

    # Return or redirect
    return_url = locals().get("return_url")
    if return_url:
        redirect_url = f"{return_url}#access_token={tokens['access_token']}&token_type=Bearer"
        return RedirectResponse(url=redirect_url, status_code=302)
    else:
        return SSOCallbackResponse(
            success=True,
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type="Bearer",
            expires_in=tokens["expires_in"],
            user=user,
        )


# ==============================================================================
# Logout
# ==============================================================================


@router.get("/{provider_id}/logout")
async def initiate_sso_logout(
    provider_id: str,
    request: Request,
    id_token_hint: Optional[str] = Query(None),
    post_logout_redirect_uri: Optional[str] = Query(None),
):
    """
    Initiate SSO logout (RP-initiated).

    For OIDC: Redirects to end_session_endpoint.
    For SAML: Creates LogoutRequest.
    """
    registry = get_provider_registry()
    provider = registry.get(provider_id)

    if not provider or not provider.is_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"Provider '{provider_id}' not found"},
        )

    # Validate redirect URI
    if post_logout_redirect_uri:
        post_logout_redirect_uri = _validate_return_url(post_logout_redirect_uri)

    if provider.provider_type == SSOProviderType.OIDC:
        client = _get_or_create_oidc_client(provider)

        logout_url = await client.create_logout_url(
            id_token_hint=id_token_hint,
            post_logout_redirect_uri=post_logout_redirect_uri,
            state=secrets.token_urlsafe(16),
        )

        if logout_url:
            return RedirectResponse(url=logout_url, status_code=302)
        else:
            # Provider doesn't support logout, just return success
            if post_logout_redirect_uri:
                return RedirectResponse(url=post_logout_redirect_uri, status_code=302)
            return {"message": "Logged out locally"}

    elif provider.provider_type == SSOProviderType.SAML:
        sp = _get_or_create_saml_sp(provider)

        # Get NameID from session/token (simplified)
        name_id = request.query_params.get("name_id", "")

        logout_url = sp.create_logout_request(name_id=name_id)

        if logout_url:
            return RedirectResponse(url=logout_url, status_code=302)
        else:
            if post_logout_redirect_uri:
                return RedirectResponse(url=post_logout_redirect_uri, status_code=302)
            return {"message": "Logged out locally"}

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "unsupported_provider", "message": "Provider type not supported"},
        )


# ==============================================================================
# Utilities
# ==============================================================================


def _validate_return_url(url: str) -> str:
    """
    Validate return URL to prevent open redirect.

    Only allows:
    - Relative URLs
    - URLs to the same domain
    - Explicitly allowed domains
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)

    # Allow relative URLs
    if not parsed.netloc:
        return url

    # Allow same domain
    base_parsed = urlparse(TG_BASE_URL)
    if parsed.netloc == base_parsed.netloc:
        return url

    # Allow configured domains
    allowed_redirect_domains = os.getenv("TG_SSO_ALLOWED_REDIRECT_DOMAINS", "").split(",")
    allowed_redirect_domains = [d.strip() for d in allowed_redirect_domains if d.strip()]

    if parsed.netloc in allowed_redirect_domains:
        return url

    logger.warning(f"Rejected redirect URL: {url}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"error": "invalid_redirect", "message": "Return URL not allowed"},
    )


# ==============================================================================
# Provider Management (Admin)
# ==============================================================================


class CreateSSOProviderRequest(BaseModel):
    """Request to create an SSO provider."""

    provider_id: str = Field(..., description="Unique provider identifier")
    name: str = Field(..., description="Display name")
    provider_type: SSOProviderType
    tenant_id: str
    is_enabled: bool = True
    is_default: bool = False
    oidc_config: Optional[OIDCConfig] = None
    saml_config: Optional[SAMLConfig] = None
    display_name: Optional[str] = None
    icon_url: Optional[str] = None
    button_color: Optional[str] = None
    auto_create_users: bool = True
    default_role: str = "user"
    allowed_domains: List[str] = []
    session_duration_hours: int = 8


@router.post("/providers", status_code=status.HTTP_201_CREATED)
async def create_sso_provider(
    request: CreateSSOProviderRequest,
    admin_user: AdminUserContext = Depends(require_super_admin),
) -> SSOProviderInfo:
    """
    Create a new SSO provider configuration.

    Admin endpoint for configuring SSO providers.
    """
    # Validate configuration
    if request.provider_type == SSOProviderType.OIDC and not request.oidc_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "missing_config", "message": "OIDC configuration required"},
        )

    if request.provider_type == SSOProviderType.SAML and not request.saml_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "missing_config", "message": "SAML configuration required"},
        )

    provider = SSOProvider(
        provider_id=request.provider_id,
        name=request.name,
        provider_type=request.provider_type,
        tenant_id=request.tenant_id,
        is_enabled=request.is_enabled,
        is_default=request.is_default,
        oidc_config=request.oidc_config,
        saml_config=request.saml_config,
        display_name=request.display_name,
        icon_url=request.icon_url,
        button_color=request.button_color,
        auto_create_users=request.auto_create_users,
        default_role=request.default_role,
        allowed_domains=request.allowed_domains,
        session_duration_hours=request.session_duration_hours,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    registry = get_provider_registry()
    registry.register(provider)

    logger.info(f"Created SSO provider: {provider.provider_id}")

    return SSOProviderInfo(
        provider_id=provider.provider_id,
        name=provider.name,
        display_name=provider.display_name or f"Sign in with {provider.name}",
        provider_type=provider.provider_type,
        icon_url=provider.icon_url,
        button_color=provider.button_color,
        is_default=provider.is_default,
    )


@router.delete("/providers/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sso_provider(
    provider_id: str,
    admin_user: AdminUserContext = Depends(require_super_admin),
):
    """Delete an SSO provider configuration."""
    registry = get_provider_registry()

    if not registry.unregister(provider_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "provider_not_found", "message": f"Provider '{provider_id}' not found"},
        )

    # Clean up cached clients
    if provider_id in _oidc_clients:
        await _oidc_clients[provider_id].close()
        del _oidc_clients[provider_id]

    if provider_id in _saml_sps:
        del _saml_sps[provider_id]

    logger.info(f"Deleted SSO provider: {provider_id}")
