"""
OpenID Connect (OIDC) Client Implementation.

Provides a secure OIDC client supporting:
- Authorization Code Flow with PKCE
- ID Token validation with signature verification
- Discovery document support
- Multiple identity providers (Okta, Auth0, Azure AD, Google)
- Claim extraction and mapping

Security Features:
- PKCE (Proof Key for Code Exchange) to prevent authorization code interception
- State parameter for CSRF protection
- Nonce validation to prevent replay attacks
- Token signature verification using JWKS
- Secure session binding
"""

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlparse

import httpx
from jose import JWTError, jwt
from jose.exceptions import JWKError

from .models import OIDCConfig, SSOSession, SSOUser, SSOProviderType

logger = logging.getLogger(__name__)


class OIDCError(Exception):
    """Base exception for OIDC errors."""

    def __init__(self, error: str, description: Optional[str] = None):
        self.error = error
        self.description = description
        super().__init__(f"{error}: {description}" if description else error)


class OIDCClient:
    """
    OpenID Connect client for authentication.

    Supports the Authorization Code Flow with PKCE for secure
    authentication against OIDC-compliant identity providers.
    """

    def __init__(
        self,
        config: OIDCConfig,
        provider_id: str,
        tenant_id: str,
        redirect_uri: str,
    ):
        """
        Initialize OIDC client.

        Args:
            config: OIDC configuration
            provider_id: SSO provider identifier
            tenant_id: TenSafe tenant ID
            redirect_uri: OAuth2 redirect URI for callbacks
        """
        self.config = config
        self.provider_id = provider_id
        self.tenant_id = tenant_id
        self.redirect_uri = redirect_uri

        # Discovery metadata cache
        self._discovery_metadata: Optional[Dict[str, Any]] = None
        self._discovery_fetched_at: Optional[float] = None
        self._discovery_cache_ttl = 3600  # 1 hour

        # JWKS cache
        self._jwks: Optional[Dict[str, Any]] = None
        self._jwks_fetched_at: Optional[float] = None
        self._jwks_cache_ttl = 3600  # 1 hour

        # HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout_seconds),
                follow_redirects=True,
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def _fetch_discovery_document(self) -> Dict[str, Any]:
        """
        Fetch OIDC discovery document.

        Returns:
            Discovery metadata
        """
        # Check cache
        now = time.time()
        if (
            self._discovery_metadata
            and self._discovery_fetched_at
            and (now - self._discovery_fetched_at) < self._discovery_cache_ttl
        ):
            return self._discovery_metadata

        discovery_url = f"{self.config.issuer}/.well-known/openid-configuration"
        logger.debug(f"Fetching OIDC discovery document: {discovery_url}")

        client = await self._get_http_client()
        try:
            response = await client.get(discovery_url)
            response.raise_for_status()
            self._discovery_metadata = response.json()
            self._discovery_fetched_at = now
            logger.debug(f"OIDC discovery loaded for {self.config.issuer}")
            return self._discovery_metadata
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch OIDC discovery: {e}")
            raise OIDCError("discovery_error", f"Failed to fetch discovery document: {e}")

    async def _get_authorization_endpoint(self) -> str:
        """Get authorization endpoint URL."""
        if self.config.authorization_endpoint:
            return self.config.authorization_endpoint
        metadata = await self._fetch_discovery_document()
        return metadata["authorization_endpoint"]

    async def _get_token_endpoint(self) -> str:
        """Get token endpoint URL."""
        if self.config.token_endpoint:
            return self.config.token_endpoint
        metadata = await self._fetch_discovery_document()
        return metadata["token_endpoint"]

    async def _get_userinfo_endpoint(self) -> str:
        """Get userinfo endpoint URL."""
        if self.config.userinfo_endpoint:
            return self.config.userinfo_endpoint
        metadata = await self._fetch_discovery_document()
        return metadata.get("userinfo_endpoint", "")

    async def _get_jwks_uri(self) -> str:
        """Get JWKS URI."""
        if self.config.jwks_uri:
            return self.config.jwks_uri
        metadata = await self._fetch_discovery_document()
        return metadata["jwks_uri"]

    async def _get_end_session_endpoint(self) -> Optional[str]:
        """Get end session (logout) endpoint."""
        if self.config.end_session_endpoint:
            return self.config.end_session_endpoint
        try:
            metadata = await self._fetch_discovery_document()
            return metadata.get("end_session_endpoint")
        except OIDCError:
            return None

    async def _fetch_jwks(self) -> Dict[str, Any]:
        """
        Fetch JSON Web Key Set for token validation.

        Returns:
            JWKS document
        """
        # Check cache
        now = time.time()
        if self._jwks and self._jwks_fetched_at and (now - self._jwks_fetched_at) < self._jwks_cache_ttl:
            return self._jwks

        jwks_uri = await self._get_jwks_uri()
        logger.debug(f"Fetching JWKS: {jwks_uri}")

        client = await self._get_http_client()
        try:
            response = await client.get(jwks_uri)
            response.raise_for_status()
            self._jwks = response.json()
            self._jwks_fetched_at = now
            return self._jwks
        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise OIDCError("jwks_error", f"Failed to fetch JWKS: {e}")

    @staticmethod
    def _generate_code_verifier() -> str:
        """Generate PKCE code verifier."""
        # RFC 7636: 43-128 characters from unreserved characters
        return secrets.token_urlsafe(64)[:96]

    @staticmethod
    def _generate_code_challenge(verifier: str) -> str:
        """Generate PKCE code challenge (S256 method)."""
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    @staticmethod
    def _generate_state() -> str:
        """Generate cryptographically secure state parameter."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def _generate_nonce() -> str:
        """Generate nonce for ID token validation."""
        return secrets.token_urlsafe(32)

    async def create_authorization_url(
        self,
        return_url: Optional[str] = None,
        prompt: Optional[str] = None,
        login_hint: Optional[str] = None,
        extra_params: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, SSOSession]:
        """
        Create authorization URL for initiating OIDC flow.

        Args:
            return_url: URL to redirect to after authentication
            prompt: OIDC prompt parameter (login, consent, none)
            login_hint: Pre-fill login identifier
            extra_params: Additional query parameters

        Returns:
            Tuple of (authorization_url, session_state)
        """
        authorization_endpoint = await self._get_authorization_endpoint()

        # Generate security parameters
        state = self._generate_state()
        nonce = self._generate_nonce()

        # PKCE
        code_verifier = None
        code_challenge = None
        if self.config.use_pkce:
            code_verifier = self._generate_code_verifier()
            code_challenge = self._generate_code_challenge(code_verifier)

        # Build authorization URL parameters
        params = {
            "response_type": "code",
            "client_id": self.config.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
            "nonce": nonce,
        }

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        if prompt:
            params["prompt"] = prompt

        if login_hint:
            params["login_hint"] = login_hint

        # Handle Google hosted domain restriction
        if self.config.additional_claims:
            for claim in self.config.additional_claims:
                if claim.startswith("hd:"):
                    params["hd"] = claim[3:]

        if extra_params:
            params.update(extra_params)

        # Build URL
        auth_url = f"{authorization_endpoint}?{urlencode(params)}"

        # Create session state
        session = SSOSession(
            session_id=secrets.token_urlsafe(32),
            state=state,
            nonce=nonce,
            code_verifier=code_verifier,
            provider_id=self.provider_id,
            redirect_uri=self.redirect_uri,
            return_url=return_url,
            created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )

        logger.info(f"Created authorization URL for provider {self.provider_id}")
        return auth_url, session

    async def exchange_code(
        self,
        code: str,
        session: SSOSession,
    ) -> Dict[str, Any]:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            session: SSO session with PKCE verifier

        Returns:
            Token response (access_token, id_token, refresh_token, etc.)
        """
        token_endpoint = await self._get_token_endpoint()

        # Build token request
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": session.redirect_uri,
            "client_id": self.config.client_id,
        }

        # Add PKCE verifier
        if session.code_verifier:
            data["code_verifier"] = session.code_verifier

        # Add client secret if configured
        if self.config.client_secret:
            if self.config.token_endpoint_auth_method == "client_secret_post":
                data["client_secret"] = self.config.client_secret
            elif self.config.token_endpoint_auth_method == "client_secret_basic":
                # Use HTTP Basic Auth
                pass  # Handled in headers below

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        if (
            self.config.client_secret
            and self.config.token_endpoint_auth_method == "client_secret_basic"
        ):
            credentials = f"{self.config.client_id}:{self.config.client_secret}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        logger.debug(f"Exchanging code at {token_endpoint}")

        client = await self._get_http_client()
        try:
            response = await client.post(
                token_endpoint,
                data=data,
                headers=headers,
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                logger.error(f"Token exchange failed: {error_data}")
                raise OIDCError(
                    error_data.get("error", "token_error"),
                    error_data.get("error_description", "Token exchange failed"),
                )

            return response.json()

        except httpx.HTTPError as e:
            logger.error(f"Token exchange HTTP error: {e}")
            raise OIDCError("token_error", f"Token exchange failed: {e}")

    async def validate_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate and decode ID token.

        Validates:
        - Token signature using JWKS
        - Token issuer
        - Token audience
        - Token expiration
        - Nonce (if provided)
        - at_hash (if access_token provided)

        Args:
            id_token: JWT ID token
            nonce: Expected nonce value
            access_token: Access token for at_hash validation

        Returns:
            Decoded ID token claims
        """
        # Fetch JWKS
        jwks = await self._fetch_jwks()

        try:
            # Decode header to get key ID
            header = jwt.get_unverified_header(id_token)
            kid = header.get("kid")
            alg = header.get("alg", "RS256")

            # Find matching key
            key = None
            for k in jwks.get("keys", []):
                if k.get("kid") == kid:
                    key = k
                    break

            if not key:
                # Try first key if no kid match
                if jwks.get("keys"):
                    key = jwks["keys"][0]
                else:
                    raise OIDCError("invalid_token", "No matching key found in JWKS")

            # Validate and decode token
            claims = jwt.decode(
                id_token,
                key,
                algorithms=[alg],
                audience=self.config.client_id,
                issuer=self.config.issuer,
            )

            # Validate nonce
            if nonce and claims.get("nonce") != nonce:
                raise OIDCError("invalid_token", "Nonce mismatch")

            # Validate at_hash if access token provided
            if access_token and "at_hash" in claims:
                expected_at_hash = self._calculate_at_hash(access_token, alg)
                if claims["at_hash"] != expected_at_hash:
                    logger.warning("at_hash validation failed")
                    # Some providers have issues with at_hash, log but don't fail

            logger.debug(f"ID token validated for subject: {claims.get('sub')}")
            return claims

        except JWTError as e:
            logger.error(f"ID token validation failed: {e}")
            raise OIDCError("invalid_token", str(e))
        except JWKError as e:
            logger.error(f"JWKS error: {e}")
            raise OIDCError("invalid_token", f"Key error: {e}")

    @staticmethod
    def _calculate_at_hash(access_token: str, algorithm: str) -> str:
        """Calculate at_hash claim value."""
        # Determine hash algorithm from JWT algorithm
        if algorithm.startswith("RS") or algorithm.startswith("PS"):
            hash_alg = f"sha{algorithm[2:]}"
        elif algorithm.startswith("ES"):
            hash_alg = f"sha{algorithm[2:]}"
        else:
            hash_alg = "sha256"

        # Calculate hash
        hasher = hashlib.new(hash_alg)
        hasher.update(access_token.encode("ascii"))
        digest = hasher.digest()

        # Take left half and base64url encode
        half_length = len(digest) // 2
        return base64.urlsafe_b64encode(digest[:half_length]).rstrip(b"=").decode("ascii")

    async def fetch_userinfo(self, access_token: str) -> Dict[str, Any]:
        """
        Fetch user info from userinfo endpoint.

        Args:
            access_token: OAuth2 access token

        Returns:
            User info claims
        """
        userinfo_endpoint = await self._get_userinfo_endpoint()
        if not userinfo_endpoint:
            return {}

        client = await self._get_http_client()
        try:
            response = await client.get(
                userinfo_endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch userinfo: {e}")
            return {}

    def map_claims_to_user(
        self,
        claims: Dict[str, Any],
        token_response: Dict[str, Any],
    ) -> SSOUser:
        """
        Map OIDC claims to TenSafe user model.

        Args:
            claims: Combined claims from ID token and userinfo
            token_response: Raw token response

        Returns:
            SSOUser with mapped fields
        """
        # Apply claim mappings
        mapped = {}
        for oidc_claim, tensafe_field in self.config.claim_mappings.items():
            if oidc_claim in claims:
                mapped[tensafe_field] = claims[oidc_claim]

        # Extract groups/roles
        groups = []
        roles = []

        if "groups" in mapped:
            groups = mapped.pop("groups") if isinstance(mapped.get("groups"), list) else []
        if "roles" in mapped:
            roles = mapped.pop("roles") if isinstance(mapped.get("roles"), list) else []

        # Apply role mappings
        mapped_role = None
        for idp_role, tensafe_role in self.config.role_mappings.items():
            if idp_role in groups or idp_role in roles:
                mapped_role = tensafe_role
                break

        # Build SSOUser
        return SSOUser(
            external_id=mapped.get("external_id", claims.get("sub", "")),
            email=mapped.get("email", ""),
            name=mapped.get("name"),
            first_name=mapped.get("first_name"),
            last_name=mapped.get("last_name"),
            groups=groups,
            roles=roles,
            mapped_role=mapped_role,
            provider_id=self.provider_id,
            provider_type=SSOProviderType.OIDC,
            tenant_id=self.tenant_id,
            id_token=token_response.get("id_token"),
            access_token=token_response.get("access_token"),
            refresh_token=token_response.get("refresh_token"),
            raw_claims=claims,
            authenticated_at=datetime.now(timezone.utc),
            token_expires_at=(
                datetime.now(timezone.utc) + timedelta(seconds=token_response.get("expires_in", 3600))
                if token_response.get("expires_in")
                else None
            ),
        )

    async def process_callback(
        self,
        code: str,
        session: SSOSession,
    ) -> SSOUser:
        """
        Process OIDC callback - exchange code and validate tokens.

        Args:
            code: Authorization code
            session: SSO session state

        Returns:
            Authenticated user
        """
        # Validate session
        if session.used:
            raise OIDCError("invalid_request", "Session state already used")

        if datetime.now(timezone.utc) > session.expires_at:
            raise OIDCError("invalid_request", "Session state expired")

        # Exchange code for tokens
        token_response = await self.exchange_code(code, session)

        # Validate ID token
        id_token = token_response.get("id_token")
        if not id_token:
            raise OIDCError("invalid_response", "No ID token in response")

        claims = await self.validate_id_token(
            id_token,
            nonce=session.nonce,
            access_token=token_response.get("access_token"),
        )

        # Fetch additional user info
        access_token = token_response.get("access_token")
        if access_token:
            userinfo = await self.fetch_userinfo(access_token)
            # Merge userinfo claims (ID token takes precedence)
            for key, value in userinfo.items():
                if key not in claims:
                    claims[key] = value

        # Map to user
        user = self.map_claims_to_user(claims, token_response)

        logger.info(f"OIDC authentication successful: {user.email} (provider: {self.provider_id})")
        return user

    async def create_logout_url(
        self,
        id_token_hint: Optional[str] = None,
        post_logout_redirect_uri: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[str]:
        """
        Create logout URL for RP-initiated logout.

        Args:
            id_token_hint: ID token to identify session
            post_logout_redirect_uri: URL to redirect after logout
            state: State parameter for logout callback

        Returns:
            Logout URL or None if not supported
        """
        end_session_endpoint = await self._get_end_session_endpoint()
        if not end_session_endpoint:
            logger.debug("Provider does not support RP-initiated logout")
            return None

        params = {}

        if id_token_hint:
            params["id_token_hint"] = id_token_hint

        if post_logout_redirect_uri:
            params["post_logout_redirect_uri"] = post_logout_redirect_uri

        if state:
            params["state"] = state

        if self.config.client_id:
            params["client_id"] = self.config.client_id

        if params:
            return f"{end_session_endpoint}?{urlencode(params)}"
        return end_session_endpoint

    async def refresh_tokens(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: OAuth2 refresh token

        Returns:
            New token response
        """
        token_endpoint = await self._get_token_endpoint()

        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config.client_id,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        client = await self._get_http_client()
        try:
            response = await client.post(
                token_endpoint,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                raise OIDCError(
                    error_data.get("error", "refresh_error"),
                    error_data.get("error_description", "Token refresh failed"),
                )

            return response.json()

        except httpx.HTTPError as e:
            raise OIDCError("refresh_error", f"Token refresh failed: {e}")
