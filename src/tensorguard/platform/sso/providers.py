"""
Pre-configured SSO Provider Templates.

Provides configuration templates for popular identity providers:
- Okta
- Auth0
- Azure AD (Microsoft Entra ID)
- Google Workspace
- Generic OIDC

These templates simplify SSO setup by pre-filling provider-specific
endpoints and claim mappings.
"""

import logging
import os
from typing import Dict, List, Optional

from .models import OIDCConfig, SAMLConfig, SSOProvider, SSOProviderType

logger = logging.getLogger(__name__)


def get_okta_config(
    okta_domain: str,
    client_id: str,
    client_secret: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> OIDCConfig:
    """
    Create OIDC configuration for Okta.

    Args:
        okta_domain: Your Okta domain (e.g., "example.okta.com" or "example.oktapreview.com")
        client_id: OAuth2 client ID from Okta
        client_secret: OAuth2 client secret (optional if using PKCE)
        scopes: Additional scopes to request

    Returns:
        OIDCConfig configured for Okta
    """
    # Normalize domain
    if okta_domain.startswith("https://"):
        okta_domain = okta_domain.replace("https://", "")
    okta_domain = okta_domain.rstrip("/")

    issuer = f"https://{okta_domain}"

    default_scopes = ["openid", "profile", "email"]
    if scopes:
        default_scopes.extend(scopes)

    return OIDCConfig(
        issuer=issuer,
        authorization_endpoint=f"{issuer}/oauth2/v1/authorize",
        token_endpoint=f"{issuer}/oauth2/v1/token",
        userinfo_endpoint=f"{issuer}/oauth2/v1/userinfo",
        jwks_uri=f"{issuer}/oauth2/v1/keys",
        end_session_endpoint=f"{issuer}/oauth2/v1/logout",
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(set(default_scopes)),  # Deduplicate
        use_pkce=True,
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        claim_mappings={
            "sub": "external_id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "groups": "groups",
            "preferred_username": "username",
        },
        role_mappings={
            "TenSafe-Admin": "org_admin",
            "TenSafe-Operator": "operator",
            "TenSafe-User": "user",
        },
    )


def get_okta_saml_config(
    okta_domain: str,
    okta_app_id: str,
    sp_entity_id: str,
    sp_acs_url: str,
    idp_x509_cert: str,
    sp_x509_cert: Optional[str] = None,
    sp_private_key: Optional[str] = None,
) -> SAMLConfig:
    """
    Create SAML configuration for Okta.

    Args:
        okta_domain: Your Okta domain
        okta_app_id: Okta SAML app ID
        sp_entity_id: Service Provider entity ID
        sp_acs_url: Assertion Consumer Service URL
        idp_x509_cert: Okta's X.509 signing certificate
        sp_x509_cert: SP certificate for signing (optional)
        sp_private_key: SP private key for signing (optional)

    Returns:
        SAMLConfig configured for Okta
    """
    if okta_domain.startswith("https://"):
        okta_domain = okta_domain.replace("https://", "")

    return SAMLConfig(
        entity_id=sp_entity_id,
        acs_url=sp_acs_url,
        idp_entity_id=f"http://www.okta.com/{okta_app_id}",
        idp_sso_url=f"https://{okta_domain}/app/{okta_app_id}/sso/saml",
        idp_x509_cert=idp_x509_cert,
        sp_x509_cert=sp_x509_cert,
        sp_private_key=sp_private_key,
        want_assertions_signed=True,
        want_response_signed=True,
        sign_authn_requests=bool(sp_x509_cert and sp_private_key),
        attribute_mappings={
            "urn:oid:1.3.6.1.4.1.5923.1.1.1.6": "external_id",  # eduPersonPrincipalName
            "urn:oid:0.9.2342.19200300.100.1.3": "email",  # mail
            "urn:oid:2.5.4.42": "first_name",  # givenName
            "urn:oid:2.5.4.4": "last_name",  # sn
            "urn:oid:2.16.840.1.113730.3.1.241": "name",  # displayName
        },
        role_attribute="groups",
    )


def get_auth0_config(
    auth0_domain: str,
    client_id: str,
    client_secret: Optional[str] = None,
    audience: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> OIDCConfig:
    """
    Create OIDC configuration for Auth0.

    Args:
        auth0_domain: Your Auth0 domain (e.g., "example.auth0.com" or "example.us.auth0.com")
        client_id: Auth0 application client ID
        client_secret: Auth0 application client secret (optional for SPA/native apps)
        audience: API audience for access token (optional)
        scopes: Additional scopes to request

    Returns:
        OIDCConfig configured for Auth0
    """
    if auth0_domain.startswith("https://"):
        auth0_domain = auth0_domain.replace("https://", "")
    auth0_domain = auth0_domain.rstrip("/")

    issuer = f"https://{auth0_domain}/"  # Auth0 issuer includes trailing slash

    default_scopes = ["openid", "profile", "email"]
    if scopes:
        default_scopes.extend(scopes)

    # Add audience to additional claims if provided
    additional_claims = []
    if audience:
        additional_claims.append(audience)

    return OIDCConfig(
        issuer=issuer,
        authorization_endpoint=f"https://{auth0_domain}/authorize",
        token_endpoint=f"https://{auth0_domain}/oauth/token",
        userinfo_endpoint=f"https://{auth0_domain}/userinfo",
        jwks_uri=f"https://{auth0_domain}/.well-known/jwks.json",
        end_session_endpoint=f"https://{auth0_domain}/v2/logout",
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(set(default_scopes)),
        additional_claims=additional_claims,
        use_pkce=True,
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        claim_mappings={
            "sub": "external_id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "https://tensafe.io/roles": "roles",  # Custom Auth0 claim namespace
            "https://tensafe.io/groups": "groups",
        },
        role_mappings={
            "admin": "org_admin",
            "operator": "operator",
            "user": "user",
        },
    )


def get_azure_ad_config(
    tenant_id: str,
    client_id: str,
    client_secret: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    use_v2_endpoint: bool = True,
) -> OIDCConfig:
    """
    Create OIDC configuration for Azure AD (Microsoft Entra ID).

    Args:
        tenant_id: Azure AD tenant ID (GUID) or domain name, or 'common' for multi-tenant
        client_id: Azure AD application (client) ID
        client_secret: Application client secret (optional for public clients)
        scopes: Additional scopes to request
        use_v2_endpoint: Use v2.0 endpoints (recommended)

    Returns:
        OIDCConfig configured for Azure AD
    """
    version = "v2.0" if use_v2_endpoint else ""
    base_url = f"https://login.microsoftonline.com/{tenant_id}"

    if use_v2_endpoint:
        issuer = f"{base_url}/v2.0"
        authorization_endpoint = f"{base_url}/oauth2/v2.0/authorize"
        token_endpoint = f"{base_url}/oauth2/v2.0/token"
        jwks_uri = f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys"
        end_session_endpoint = f"{base_url}/oauth2/v2.0/logout"
    else:
        issuer = f"https://sts.windows.net/{tenant_id}/"
        authorization_endpoint = f"{base_url}/oauth2/authorize"
        token_endpoint = f"{base_url}/oauth2/token"
        jwks_uri = f"https://login.microsoftonline.com/{tenant_id}/discovery/keys"
        end_session_endpoint = f"{base_url}/oauth2/logout"

    # Azure AD requires explicit Graph API scopes for user info
    default_scopes = ["openid", "profile", "email", "User.Read"]
    if scopes:
        default_scopes.extend(scopes)

    return OIDCConfig(
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        userinfo_endpoint="https://graph.microsoft.com/oidc/userinfo",
        jwks_uri=jwks_uri,
        end_session_endpoint=end_session_endpoint,
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(set(default_scopes)),
        use_pkce=True,
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        claim_mappings={
            "sub": "external_id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "preferred_username": "username",
            "groups": "groups",
            "roles": "roles",
        },
        role_mappings={
            # Map Azure AD app roles to TenSafe roles
            "TenSafe.Admin": "org_admin",
            "TenSafe.Operator": "operator",
            "TenSafe.User": "user",
        },
    )


def get_google_config(
    client_id: str,
    client_secret: Optional[str] = None,
    hosted_domain: Optional[str] = None,
    scopes: Optional[List[str]] = None,
) -> OIDCConfig:
    """
    Create OIDC configuration for Google Workspace / Google Identity.

    Args:
        client_id: Google OAuth2 client ID
        client_secret: Google OAuth2 client secret (optional for native apps)
        hosted_domain: Restrict to specific Google Workspace domain (hd parameter)
        scopes: Additional scopes to request

    Returns:
        OIDCConfig configured for Google
    """
    issuer = "https://accounts.google.com"

    default_scopes = ["openid", "profile", "email"]
    if scopes:
        default_scopes.extend(scopes)

    config = OIDCConfig(
        issuer=issuer,
        authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
        token_endpoint="https://oauth2.googleapis.com/token",
        userinfo_endpoint="https://openidconnect.googleapis.com/v1/userinfo",
        jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
        end_session_endpoint=None,  # Google doesn't support RP-initiated logout
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(set(default_scopes)),
        use_pkce=True,
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        claim_mappings={
            "sub": "external_id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "picture": "avatar_url",
            "hd": "hosted_domain",  # Google Workspace domain
        },
        # Google doesn't provide group/role claims by default
        # Use Google Groups API separately or define static mappings
        role_mappings={},
    )

    # Store hosted domain restriction in additional claims
    if hosted_domain:
        config.additional_claims = [f"hd:{hosted_domain}"]

    return config


def get_generic_oidc_config(
    issuer: str,
    client_id: str,
    client_secret: Optional[str] = None,
    scopes: Optional[List[str]] = None,
    claim_mappings: Optional[Dict[str, str]] = None,
) -> OIDCConfig:
    """
    Create a generic OIDC configuration for any compliant provider.

    Endpoint URLs will be discovered from the provider's .well-known/openid-configuration.

    Args:
        issuer: OIDC issuer URL
        client_id: OAuth2 client ID
        client_secret: OAuth2 client secret (optional)
        scopes: Scopes to request (defaults to openid, profile, email)
        claim_mappings: Custom claim mappings

    Returns:
        OIDCConfig with minimal configuration (uses discovery)
    """
    default_scopes = ["openid", "profile", "email"]
    if scopes:
        default_scopes.extend(scopes)

    default_claim_mappings = {
        "sub": "external_id",
        "email": "email",
        "name": "name",
        "given_name": "first_name",
        "family_name": "last_name",
        "groups": "groups",
        "roles": "roles",
    }
    if claim_mappings:
        default_claim_mappings.update(claim_mappings)

    return OIDCConfig(
        issuer=issuer,
        client_id=client_id,
        client_secret=client_secret,
        scopes=list(set(default_scopes)),
        use_pkce=True,
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        claim_mappings=default_claim_mappings,
    )


class ProviderRegistry:
    """
    Registry for managing SSO providers.

    Provides in-memory caching with optional database persistence.
    """

    def __init__(self):
        self._providers: Dict[str, SSOProvider] = {}
        self._tenant_defaults: Dict[str, str] = {}  # tenant_id -> provider_id

    def register(self, provider: SSOProvider) -> None:
        """Register an SSO provider."""
        if provider.provider_id in self._providers:
            logger.warning(f"Overwriting existing provider: {provider.provider_id}")

        self._providers[provider.provider_id] = provider

        if provider.is_default:
            self._tenant_defaults[provider.tenant_id] = provider.provider_id
            logger.info(f"Set default provider for tenant {provider.tenant_id}: {provider.provider_id}")

        logger.info(f"Registered SSO provider: {provider.provider_id} ({provider.provider_type.value})")

    def unregister(self, provider_id: str) -> bool:
        """Unregister an SSO provider."""
        if provider_id not in self._providers:
            return False

        provider = self._providers.pop(provider_id)

        # Clear default if this was the default provider
        if self._tenant_defaults.get(provider.tenant_id) == provider_id:
            del self._tenant_defaults[provider.tenant_id]

        logger.info(f"Unregistered SSO provider: {provider_id}")
        return True

    def get(self, provider_id: str) -> Optional[SSOProvider]:
        """Get a provider by ID."""
        return self._providers.get(provider_id)

    def get_by_tenant(self, tenant_id: str) -> List[SSOProvider]:
        """Get all providers for a tenant."""
        return [p for p in self._providers.values() if p.tenant_id == tenant_id and p.is_enabled]

    def get_default(self, tenant_id: str) -> Optional[SSOProvider]:
        """Get the default provider for a tenant."""
        provider_id = self._tenant_defaults.get(tenant_id)
        if provider_id:
            return self._providers.get(provider_id)

        # Fall back to first enabled provider for tenant
        tenant_providers = self.get_by_tenant(tenant_id)
        return tenant_providers[0] if tenant_providers else None

    def list_all(self) -> List[SSOProvider]:
        """List all registered providers."""
        return list(self._providers.values())


# Global registry instance
_registry: Optional[ProviderRegistry] = None


def get_provider_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _registry
    if _registry is None:
        _registry = ProviderRegistry()
        _configure_from_environment()
    return _registry


def _configure_from_environment() -> None:
    """Configure providers from environment variables."""
    global _registry

    # Okta
    okta_domain = os.getenv("TG_SSO_OKTA_DOMAIN")
    okta_client_id = os.getenv("TG_SSO_OKTA_CLIENT_ID")
    if okta_domain and okta_client_id:
        okta_config = get_okta_config(
            okta_domain=okta_domain,
            client_id=okta_client_id,
            client_secret=os.getenv("TG_SSO_OKTA_CLIENT_SECRET"),
        )
        provider = SSOProvider(
            provider_id="okta",
            name="Okta",
            provider_type=SSOProviderType.OIDC,
            tenant_id=os.getenv("TG_SSO_OKTA_TENANT_ID", "default"),
            oidc_config=okta_config,
            display_name="Sign in with Okta",
            is_default=os.getenv("TG_SSO_DEFAULT_PROVIDER") == "okta",
        )
        _registry.register(provider)
        logger.info("Configured Okta SSO from environment")

    # Auth0
    auth0_domain = os.getenv("TG_SSO_AUTH0_DOMAIN")
    auth0_client_id = os.getenv("TG_SSO_AUTH0_CLIENT_ID")
    if auth0_domain and auth0_client_id:
        auth0_config = get_auth0_config(
            auth0_domain=auth0_domain,
            client_id=auth0_client_id,
            client_secret=os.getenv("TG_SSO_AUTH0_CLIENT_SECRET"),
            audience=os.getenv("TG_SSO_AUTH0_AUDIENCE"),
        )
        provider = SSOProvider(
            provider_id="auth0",
            name="Auth0",
            provider_type=SSOProviderType.OIDC,
            tenant_id=os.getenv("TG_SSO_AUTH0_TENANT_ID", "default"),
            oidc_config=auth0_config,
            display_name="Sign in with Auth0",
            is_default=os.getenv("TG_SSO_DEFAULT_PROVIDER") == "auth0",
        )
        _registry.register(provider)
        logger.info("Configured Auth0 SSO from environment")

    # Azure AD
    azure_tenant = os.getenv("TG_SSO_AZURE_TENANT_ID")
    azure_client_id = os.getenv("TG_SSO_AZURE_CLIENT_ID")
    if azure_tenant and azure_client_id:
        azure_config = get_azure_ad_config(
            tenant_id=azure_tenant,
            client_id=azure_client_id,
            client_secret=os.getenv("TG_SSO_AZURE_CLIENT_SECRET"),
        )
        provider = SSOProvider(
            provider_id="azure-ad",
            name="Microsoft",
            provider_type=SSOProviderType.OIDC,
            tenant_id=os.getenv("TG_SSO_AZURE_APP_TENANT_ID", "default"),
            oidc_config=azure_config,
            display_name="Sign in with Microsoft",
            button_color="#0078d4",
            is_default=os.getenv("TG_SSO_DEFAULT_PROVIDER") == "azure-ad",
        )
        _registry.register(provider)
        logger.info("Configured Azure AD SSO from environment")

    # Google
    google_client_id = os.getenv("TG_SSO_GOOGLE_CLIENT_ID")
    if google_client_id:
        google_config = get_google_config(
            client_id=google_client_id,
            client_secret=os.getenv("TG_SSO_GOOGLE_CLIENT_SECRET"),
            hosted_domain=os.getenv("TG_SSO_GOOGLE_HOSTED_DOMAIN"),
        )
        provider = SSOProvider(
            provider_id="google",
            name="Google",
            provider_type=SSOProviderType.OIDC,
            tenant_id=os.getenv("TG_SSO_GOOGLE_TENANT_ID", "default"),
            oidc_config=google_config,
            display_name="Sign in with Google",
            button_color="#4285f4",
            is_default=os.getenv("TG_SSO_DEFAULT_PROVIDER") == "google",
        )
        _registry.register(provider)
        logger.info("Configured Google SSO from environment")
