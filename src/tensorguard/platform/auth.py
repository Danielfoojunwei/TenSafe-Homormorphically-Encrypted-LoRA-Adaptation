"""
TensorGuard Platform Authentication Module

Provides JWT-based authentication with enterprise security defaults:
- Argon2id password hashing (memory-hard, GPU-resistant)
- Configurable token expiration with secure defaults
- Role-based access control (RBAC)
- Password strength validation
- Rate limiting support via Redis (optional)
- Token issuer/audience validation
"""

import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session

from .database import get_session
from .models.core import User, UserRole

logger = logging.getLogger(__name__)

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

# JWT Configuration - MUST be set in production
from ..utils.production_gates import is_production, require_env

SECRET_KEY = os.getenv("TG_SECRET_KEY")
if not SECRET_KEY:
    if is_production():
        require_env(
            "TG_SECRET_KEY",
            remediation="Set TG_SECRET_KEY environment variable: export TG_SECRET_KEY=$(python -c \"import secrets; print(secrets.token_hex(32))\")",
            min_length=32,
        )
    else:
        logger.warning(
            "SECURITY WARNING: TG_SECRET_KEY not set. "
            "Generating ephemeral key - tokens will be invalid after restart. "
            "Set TG_SECRET_KEY environment variable for production."
        )
        SECRET_KEY = secrets.token_hex(32)

# Use HS256 for simplicity, but ensure key is at least 256 bits
ALGORITHM = os.getenv("TG_JWT_ALGORITHM", "HS256")

# Token expiration - short-lived for security
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("TG_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("TG_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Token validation
TOKEN_ISSUER = os.getenv("TG_TOKEN_ISSUER", "tensorguard-platform")
TOKEN_AUDIENCE = os.getenv("TG_TOKEN_AUDIENCE", "tensorguard-api")

# Password policy
MIN_PASSWORD_LENGTH = int(os.getenv("TG_MIN_PASSWORD_LENGTH", "12"))
REQUIRE_PASSWORD_COMPLEXITY = os.getenv("TG_REQUIRE_PASSWORD_COMPLEXITY", "true").lower() == "true"

# Rate limiting (requires Redis)
ENABLE_RATE_LIMITING = os.getenv("TG_ENABLE_RATE_LIMITING", "false").lower() == "true"
MAX_LOGIN_ATTEMPTS = int(os.getenv("TG_MAX_LOGIN_ATTEMPTS", "5"))
LOCKOUT_DURATION_MINUTES = int(os.getenv("TG_LOCKOUT_DURATION_MINUTES", "15"))

# ============================================================================
# PASSWORD HASHING
# ============================================================================

# Argon2id with OWASP-recommended parameters
pwd_context = CryptContext(
    schemes=["argon2"],
    deprecated="auto",
    argon2__memory_cost=65536,  # 64 MiB
    argon2__time_cost=3,        # 3 iterations
    argon2__parallelism=4,      # 4 threads
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


class PasswordValidationError(ValueError):
    """Raised when password doesn't meet security requirements."""
    pass


def validate_password_strength(password: str) -> None:
    """
    Validate password meets security requirements.

    Raises:
        PasswordValidationError: If password doesn't meet requirements
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
        )

    if REQUIRE_PASSWORD_COMPLEXITY:
        checks = [
            (r'[a-z]', "lowercase letter"),
            (r'[A-Z]', "uppercase letter"),
            (r'\d', "digit"),
            (r'[!@#$%^&*(),.?":{}|<>]', "special character"),
        ]
        missing = []
        for pattern, name in checks:
            if not re.search(pattern, password):
                missing.append(name)

        if missing:
            raise PasswordValidationError(
                f"Password must contain: {', '.join(missing)}"
            )


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str, validate: bool = True) -> str:
    """
    Hash a password using Argon2id.

    Args:
        password: Plain text password
        validate: If True, validate password strength before hashing

    Returns:
        Hashed password

    Raises:
        PasswordValidationError: If validation enabled and password is weak
    """
    if validate:
        validate_password_strength(password)
    return pwd_context.hash(password)


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
    token_type: str = "access"
) -> str:
    """
    Create a JWT access token with security claims.

    Args:
        data: Token payload data
        expires_delta: Optional custom expiration
        token_type: Token type (access or refresh)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    # Use timezone-aware datetime (Python 3.11+ compatible)
    now = datetime.now(timezone.utc)

    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # Add standard JWT claims
    to_encode.update({
        "exp": expire,
        "iat": now,
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "type": token_type,
        "jti": secrets.token_hex(16),  # Unique token ID for revocation
    })

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a long-lived refresh token."""
    return create_access_token(
        data,
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        token_type="refresh"
    )

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    session: Session = Depends(get_session)
) -> User:
    """
    Validate JWT token and return the authenticated user.

    Validates:
    - Token signature and expiration
    - Token issuer and audience
    - Token type (must be 'access')
    - User existence in database

    Raises:
        HTTPException: 401 if authentication fails
    """
    # --- DEMO MODE BYPASS ---
    # SECURITY: Demo mode is OFF by default. Set TG_DEMO_MODE=true ONLY in dev/test.
    # In production, this should NEVER be enabled.
    DEMO_MODE = os.getenv("TG_DEMO_MODE", "false").lower() == "true"
    if DEMO_MODE:
        if os.getenv("TG_ENVIRONMENT", "development") == "production":
            logger.critical("SECURITY VIOLATION: TG_DEMO_MODE=true in production environment!")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Demo mode is not allowed in production"
            )
        if not token:
            logger.warning("DEMO MODE: Returning demo user (no token required) - NOT FOR PRODUCTION")
            demo_user = User(
                id="demo-user-001",
                email="demo@tensorguard.local",
                name="Demo User",
                role=UserRole.ORG_ADMIN,
                tenant_id="fceac734-e672-4a0c-863b-c7bb8e28b88e",
                hashed_password="N/A"
            )
            return demo_user
    # --- END DEMO MODE ---

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
        )

        # Validate token type
        token_type = payload.get("type")
        if token_type != "access":
            logger.warning(f"Invalid token type: {token_type}")
            raise credentials_exception

        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception

    except JWTError as e:
        logger.debug(f"JWT validation failed: {e}")
        raise credentials_exception

    user = session.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    # Check if user is active (if such field exists)
    if hasattr(user, 'is_active') and not user.is_active:
        logger.warning(f"Inactive user attempted access: {email}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user

class RoleChecker:
    """
    RBAC dependency for FastAPI routes.

    Usage:
        @app.get("/admin", dependencies=[Depends(RoleChecker([UserRole.ADMIN]))])
        async def admin_endpoint():
            ...
    """

    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles

    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        if user.role not in self.allowed_roles:
            logger.warning(
                f"RBAC denied: user={user.email} role={user.role} "
                f"required={[r.value for r in self.allowed_roles]}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operation not permitted for your role"
            )
        return user


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def require_roles(*roles: UserRole):
    """
    Decorator-style dependency for role checking.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_roles(UserRole.ADMIN))):
            ...
    """
    return RoleChecker(list(roles))


# Pre-configured role checkers for common use cases
require_org_admin = RoleChecker([UserRole.ORG_ADMIN])
require_site_admin = RoleChecker([UserRole.ORG_ADMIN, UserRole.SITE_ADMIN])
require_operator = RoleChecker([UserRole.ORG_ADMIN, UserRole.SITE_ADMIN, UserRole.OPERATOR])


def get_token_info(token: str) -> Optional[Dict]:
    """
    Decode a token without full validation (for debugging/logging).

    WARNING: This does NOT validate the token. Use only for logging/debugging.
    """
    try:
        # Decode without verification for inspection
        return jwt.decode(token, options={"verify_signature": False})
    except JWTError:
        return None
