"""
SSO/OIDC Pydantic Models.

Defines data models for SSO configuration and session management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from sqlmodel import Field as SQLField, SQLModel


class SSOProviderType(str, Enum):
    """Supported SSO provider types."""

    OIDC = "oidc"
    SAML = "saml"


class OIDCConfig(BaseModel):
    """OpenID Connect provider configuration."""

    # Discovery and endpoints
    issuer: str = Field(..., description="OIDC issuer URL (e.g., https://example.okta.com)")
    authorization_endpoint: Optional[str] = Field(None, description="Override authorization endpoint")
    token_endpoint: Optional[str] = Field(None, description="Override token endpoint")
    userinfo_endpoint: Optional[str] = Field(None, description="Override userinfo endpoint")
    jwks_uri: Optional[str] = Field(None, description="Override JWKS URI")
    end_session_endpoint: Optional[str] = Field(None, description="Logout endpoint")

    # Client credentials
    client_id: str = Field(..., description="OAuth2 client ID")
    client_secret: Optional[str] = Field(None, description="OAuth2 client secret (not needed for PKCE public clients)")

    # Scopes and claims
    scopes: List[str] = Field(
        default=["openid", "profile", "email"],
        description="OAuth2 scopes to request",
    )
    additional_claims: List[str] = Field(
        default=[],
        description="Additional claims to request",
    )

    # Security settings
    use_pkce: bool = Field(default=True, description="Use PKCE for authorization code flow")
    token_endpoint_auth_method: str = Field(
        default="client_secret_post",
        description="Token endpoint authentication method",
    )

    # Claim mappings (IdP claim -> TenSafe field)
    claim_mappings: Dict[str, str] = Field(
        default={
            "sub": "external_id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "groups": "groups",
            "roles": "roles",
        },
        description="Map OIDC claims to TenSafe user fields",
    )

    # Role mappings (IdP role -> TenSafe role)
    role_mappings: Dict[str, str] = Field(
        default={},
        description="Map IdP roles/groups to TenSafe roles",
    )

    # Timeouts
    request_timeout_seconds: int = Field(default=30, ge=5, le=120)

    @field_validator("issuer")
    @classmethod
    def validate_issuer(cls, v: str) -> str:
        """Ensure issuer is a valid HTTPS URL."""
        if not v.startswith("https://") and not v.startswith("http://localhost"):
            raise ValueError("Issuer must use HTTPS (except localhost for development)")
        return v.rstrip("/")


class SAMLConfig(BaseModel):
    """SAML 2.0 Service Provider configuration."""

    # Service Provider settings
    entity_id: str = Field(..., description="SP entity ID (usually your app URL)")
    acs_url: str = Field(..., description="Assertion Consumer Service URL")
    slo_url: Optional[str] = Field(None, description="Single Logout URL")
    metadata_url: Optional[str] = Field(None, description="SP metadata URL")

    # Identity Provider settings
    idp_entity_id: str = Field(..., description="IdP entity ID")
    idp_sso_url: str = Field(..., description="IdP SSO URL")
    idp_slo_url: Optional[str] = Field(None, description="IdP SLO URL")
    idp_x509_cert: str = Field(..., description="IdP X.509 certificate (PEM format)")
    idp_metadata_url: Optional[str] = Field(None, description="IdP metadata URL for auto-config")

    # SP certificate (for signing/encryption)
    sp_x509_cert: Optional[str] = Field(None, description="SP X.509 certificate (PEM)")
    sp_private_key: Optional[str] = Field(None, description="SP private key (PEM)")

    # Security settings
    want_assertions_signed: bool = Field(default=True)
    want_response_signed: bool = Field(default=True)
    want_assertions_encrypted: bool = Field(default=False)
    sign_authn_requests: bool = Field(default=True)
    signature_algorithm: str = Field(default="http://www.w3.org/2001/04/xmldsig-more#rsa-sha256")
    digest_algorithm: str = Field(default="http://www.w3.org/2001/04/xmlenc#sha256")

    # Attribute mappings (SAML attribute -> TenSafe field)
    attribute_mappings: Dict[str, str] = Field(
        default={
            "urn:oid:0.9.2342.19200300.100.1.1": "external_id",  # uid
            "urn:oid:0.9.2342.19200300.100.1.3": "email",  # mail
            "urn:oid:2.5.4.42": "first_name",  # givenName
            "urn:oid:2.5.4.4": "last_name",  # sn
            "urn:oid:2.16.840.1.113730.3.1.241": "name",  # displayName
        },
        description="Map SAML attributes to TenSafe user fields",
    )

    # Role mappings
    role_attribute: str = Field(default="groups", description="SAML attribute containing roles/groups")
    role_mappings: Dict[str, str] = Field(default={}, description="Map SAML roles to TenSafe roles")

    # Allowed clock skew for assertion validation
    allowed_clock_skew_seconds: int = Field(default=120, ge=0, le=600)


class SSOProvider(BaseModel):
    """SSO provider configuration."""

    provider_id: str = Field(..., description="Unique provider identifier")
    name: str = Field(..., description="Display name")
    provider_type: SSOProviderType = Field(..., description="Provider type (oidc or saml)")
    tenant_id: str = Field(..., description="TenSafe tenant ID")
    is_enabled: bool = Field(default=True)
    is_default: bool = Field(default=False, description="Use as default SSO provider for tenant")

    # Type-specific configuration
    oidc_config: Optional[OIDCConfig] = Field(None, description="OIDC configuration")
    saml_config: Optional[SAMLConfig] = Field(None, description="SAML configuration")

    # UI customization
    display_name: Optional[str] = Field(None, description="Button text for SSO login")
    icon_url: Optional[str] = Field(None, description="Provider icon URL")
    button_color: Optional[str] = Field(None, description="Button background color")

    # Auto-provisioning
    auto_create_users: bool = Field(default=True, description="Auto-create users on first login")
    default_role: str = Field(default="user", description="Default role for auto-created users")
    allowed_domains: List[str] = Field(default=[], description="Restrict to specific email domains")

    # Session settings
    session_duration_hours: int = Field(default=8, ge=1, le=720)
    force_reauthentication: bool = Field(default=False, description="Require re-auth on each login")

    # Audit
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, v: str) -> str:
        """Ensure provider_id is URL-safe."""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9_-]{2,62}$", v):
            raise ValueError(
                "provider_id must be 3-63 chars, start with alphanumeric, "
                "contain only lowercase letters, numbers, hyphens, underscores"
            )
        return v


class SSOUser(BaseModel):
    """User data extracted from SSO assertion/token."""

    external_id: str = Field(..., description="User ID from IdP (sub claim)")
    email: str = Field(..., description="User email")
    name: Optional[str] = Field(None, description="Full name")
    first_name: Optional[str] = None
    last_name: Optional[str] = None

    # Role/group information
    groups: List[str] = Field(default=[])
    roles: List[str] = Field(default=[])
    mapped_role: Optional[str] = Field(None, description="TenSafe role after mapping")

    # Provider information
    provider_id: str = Field(..., description="SSO provider ID")
    provider_type: SSOProviderType = Field(..., description="Provider type")
    tenant_id: str = Field(..., description="TenSafe tenant ID")

    # Token information
    id_token: Optional[str] = Field(None, description="OIDC ID token (for logout)")
    access_token: Optional[str] = Field(None, description="OAuth2 access token")
    refresh_token: Optional[str] = Field(None, description="OAuth2 refresh token")

    # Raw claims (for debugging/audit)
    raw_claims: Dict[str, Any] = Field(default={})

    # Timestamps
    authenticated_at: datetime = Field(default_factory=datetime.utcnow)
    token_expires_at: Optional[datetime] = None


class SSOSession(BaseModel):
    """SSO session state for CSRF and state validation."""

    session_id: str = Field(..., description="Unique session identifier")
    state: str = Field(..., description="OAuth2 state parameter")
    nonce: Optional[str] = Field(None, description="OIDC nonce for ID token validation")
    code_verifier: Optional[str] = Field(None, description="PKCE code verifier")
    provider_id: str = Field(..., description="SSO provider being used")
    redirect_uri: str = Field(..., description="OAuth2 redirect URI")
    return_url: Optional[str] = Field(None, description="URL to return after authentication")

    # Anti-replay
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Session expiration time")
    used: bool = Field(default=False, description="Whether state has been consumed")

    # Client fingerprint (for binding)
    client_ip: Optional[str] = None
    user_agent_hash: Optional[str] = None


# Database models for persistent storage
class SSOProviderDB(SQLModel, table=True):
    """Database model for SSO providers."""

    __tablename__ = "sso_providers"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    provider_id: str = SQLField(index=True, unique=True)
    tenant_id: str = SQLField(index=True)
    name: str
    provider_type: str
    is_enabled: bool = SQLField(default=True)
    is_default: bool = SQLField(default=False)
    config_json: str  # Encrypted JSON blob
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    updated_at: datetime = SQLField(default_factory=datetime.utcnow)


class SSOSessionDB(SQLModel, table=True):
    """Database model for SSO sessions (state tracking)."""

    __tablename__ = "sso_sessions"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    session_id: str = SQLField(index=True, unique=True)
    state: str = SQLField(index=True)
    provider_id: str = SQLField(index=True)
    data_json: str  # Encrypted session data
    created_at: datetime = SQLField(default_factory=datetime.utcnow)
    expires_at: datetime = SQLField(index=True)
    used: bool = SQLField(default=False)


class SSOAuditLog(SQLModel, table=True):
    """Audit log for SSO events."""

    __tablename__ = "sso_audit_logs"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    entry_id: str = SQLField(index=True, unique=True)
    tenant_id: str = SQLField(index=True)
    provider_id: str = SQLField(index=True)
    event_type: str  # login, logout, token_refresh, error
    user_external_id: Optional[str] = SQLField(index=True)
    user_email: Optional[str]
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    client_ip: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime = SQLField(default_factory=datetime.utcnow)
    details_json: Optional[str]


# API Request/Response Models
class SSOLoginInitRequest(BaseModel):
    """Request to initiate SSO login."""

    return_url: Optional[str] = Field(None, description="URL to redirect after login")
    prompt: Optional[str] = Field(None, description="OIDC prompt parameter (login, consent, none)")
    login_hint: Optional[str] = Field(None, description="Pre-fill login identifier")


class SSOLoginInitResponse(BaseModel):
    """Response with SSO login URL."""

    authorization_url: str = Field(..., description="URL to redirect user to")
    state: str = Field(..., description="State parameter (for verification)")
    expires_in: int = Field(..., description="Seconds until state expires")


class SSOCallbackRequest(BaseModel):
    """OAuth2/OIDC callback parameters."""

    code: Optional[str] = Field(None, description="Authorization code")
    state: str = Field(..., description="State parameter")
    error: Optional[str] = Field(None, description="Error code")
    error_description: Optional[str] = Field(None, description="Error description")


class SSOCallbackResponse(BaseModel):
    """Response from SSO callback processing."""

    success: bool
    access_token: Optional[str] = Field(None, description="TenSafe JWT access token")
    refresh_token: Optional[str] = Field(None, description="TenSafe JWT refresh token")
    token_type: str = Field(default="Bearer")
    expires_in: int = Field(..., description="Access token expiration (seconds)")
    user: Optional[SSOUser] = Field(None, description="Authenticated user info")
    error: Optional[str] = None
    error_description: Optional[str] = None


class SSOProviderInfo(BaseModel):
    """Public information about an SSO provider."""

    provider_id: str
    name: str
    display_name: Optional[str]
    provider_type: SSOProviderType
    icon_url: Optional[str]
    button_color: Optional[str]
    is_default: bool


class SSOProvidersListResponse(BaseModel):
    """List of available SSO providers."""

    providers: List[SSOProviderInfo]
    default_provider: Optional[str] = Field(None, description="Default provider ID")
