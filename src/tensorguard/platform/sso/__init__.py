"""
TensorGuard SSO/OIDC Authentication Module.

Provides enterprise Single Sign-On (SSO) support via:
- OpenID Connect (OIDC) with PKCE
- SAML 2.0 Service Provider
- Pre-configured provider templates (Okta, Auth0, Azure AD, Google)

Security Features:
- CSRF protection via state parameter
- PKCE for authorization code flow
- ID token validation with signature verification
- Secure session management
- Audit logging for compliance
"""

from .models import (
    OIDCConfig,
    SAMLConfig,
    SSOProvider,
    SSOProviderType,
    SSOSession,
    SSOUser,
)
from .oidc import OIDCClient
from .providers import (
    ProviderRegistry,
    get_auth0_config,
    get_azure_ad_config,
    get_google_config,
    get_okta_config,
)
from .saml import SAMLServiceProvider

__all__ = [
    # Models
    "SSOProvider",
    "SSOProviderType",
    "OIDCConfig",
    "SAMLConfig",
    "SSOUser",
    "SSOSession",
    # OIDC
    "OIDCClient",
    # SAML
    "SAMLServiceProvider",
    # Providers
    "ProviderRegistry",
    "get_okta_config",
    "get_auth0_config",
    "get_azure_ad_config",
    "get_google_config",
]
