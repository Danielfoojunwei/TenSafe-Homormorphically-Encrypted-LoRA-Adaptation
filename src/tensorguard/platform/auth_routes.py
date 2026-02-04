"""
TensorGuard Authentication Routes

Provides HTTP endpoints for user authentication:
- POST /auth/signup - Register a new user
- POST /auth/token - Login and get access token
- POST /auth/refresh - Refresh access token
- POST /auth/password/reset-request - Request password reset
- POST /auth/password/reset - Reset password with token
- POST /auth/email/verify - Verify email address
- POST /auth/email/resend - Resend verification email
- GET /auth/me - Get current user info
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlmodel import Session, select

from .auth import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    get_password_hash,
    verify_password,
    PasswordValidationError,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from .database import get_session
from .models.core import User, Tenant, UserRole

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Configuration
REQUIRE_EMAIL_VERIFICATION = os.getenv("TG_REQUIRE_EMAIL_VERIFICATION", "true").lower() == "true"
PASSWORD_RESET_EXPIRE_HOURS = int(os.getenv("TG_PASSWORD_RESET_EXPIRE_HOURS", "24"))
DEFAULT_TENANT_ID = os.getenv("TG_DEFAULT_TENANT_ID", "default")


# ==============================================================================
# Request/Response Models
# ==============================================================================


class SignupRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8, description="Password (min 8 chars)")
    name: str = Field(..., min_length=1, max_length=100)
    company: Optional[str] = Field(None, max_length=100, description="Company/organization name")

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class SignupResponse(BaseModel):
    """User registration response."""

    user_id: str
    email: str
    name: str
    message: str
    requires_verification: bool


class TokenResponse(BaseModel):
    """OAuth2 token response."""

    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""

    token: str
    new_password: str = Field(..., min_length=8)


class EmailVerifyRequest(BaseModel):
    """Email verification request."""

    token: str


class ResendVerificationRequest(BaseModel):
    """Resend verification email request."""

    email: EmailStr


class UserResponse(BaseModel):
    """User info response."""

    user_id: str
    email: str
    name: str
    role: str
    tenant_id: str
    email_verified: bool
    created_at: datetime


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


# ==============================================================================
# Email Functions (stub - integrate with your email provider)
# ==============================================================================


async def send_verification_email(email: str, token: str, name: str) -> None:
    """
    Send email verification link.

    TODO: Integrate with your email provider (SendGrid, SES, etc.)
    """
    verification_url = f"{os.getenv('TG_BASE_URL', 'http://localhost:8000')}/verify-email?token={token}"

    logger.info(f"Sending verification email to {email}")
    logger.info(f"Verification URL: {verification_url}")

    # Example SendGrid integration:
    # from sendgrid import SendGridAPIClient
    # from sendgrid.helpers.mail import Mail
    # message = Mail(
    #     from_email='noreply@tensafe.io',
    #     to_emails=email,
    #     subject='Verify your TenSafe account',
    #     html_content=f'<p>Hi {name},</p><p>Click <a href="{verification_url}">here</a> to verify your email.</p>'
    # )
    # sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
    # sg.send(message)


async def send_password_reset_email(email: str, token: str, name: str) -> None:
    """
    Send password reset link.

    TODO: Integrate with your email provider (SendGrid, SES, etc.)
    """
    reset_url = f"{os.getenv('TG_BASE_URL', 'http://localhost:8000')}/reset-password?token={token}"

    logger.info(f"Sending password reset email to {email}")
    logger.info(f"Reset URL: {reset_url}")


# ==============================================================================
# Routes
# ==============================================================================


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    request: SignupRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> SignupResponse:
    """
    Register a new user.

    Creates a new user account with the provided email and password.
    If email verification is enabled, sends a verification email.
    """
    # Check if email already exists
    existing = session.exec(select(User).where(User.email == request.email)).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "email_exists", "message": "An account with this email already exists"},
        )

    # Hash password
    try:
        hashed_password = get_password_hash(request.password)
    except PasswordValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "weak_password", "message": str(e)},
        )

    # Create or get tenant
    tenant_id = DEFAULT_TENANT_ID
    if request.company:
        # Create a new tenant for the company
        tenant_id = str(uuid4())
        tenant = Tenant(
            tenant_id=tenant_id,
            name=request.company,
        )
        session.add(tenant)

    # Generate user ID and verification token
    user_id = str(uuid4())
    verification_token = secrets.token_urlsafe(32) if REQUIRE_EMAIL_VERIFICATION else None

    # Create user
    user = User(
        user_id=user_id,
        tenant_id=tenant_id,
        email=request.email,
        name=request.name,
        hashed_password=hashed_password,
        role=UserRole.ORG_ADMIN.value if request.company else UserRole.USER.value,
        email_verification_token=verification_token,
        email_verified=not REQUIRE_EMAIL_VERIFICATION,
    )
    session.add(user)
    session.commit()

    logger.info(f"New user registered: {request.email} (user_id={user_id})")

    # Send verification email
    if REQUIRE_EMAIL_VERIFICATION and verification_token:
        background_tasks.add_task(send_verification_email, request.email, verification_token, request.name)

    return SignupResponse(
        user_id=user_id,
        email=request.email,
        name=request.name,
        message="Account created successfully" + (". Please check your email to verify your account." if REQUIRE_EMAIL_VERIFICATION else ""),
        requires_verification=REQUIRE_EMAIL_VERIFICATION,
    )


@router.post("/token", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: Session = Depends(get_session),
) -> TokenResponse:
    """
    Authenticate user and return access token.

    Uses OAuth2 password flow. Send form data with:
    - username: email address
    - password: password
    """
    # Find user by email
    user = session.exec(select(User).where(User.email == form_data.username)).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        logger.warning(f"Failed login attempt for: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_credentials", "message": "Invalid email or password"},
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "account_disabled", "message": "Your account has been disabled"},
        )

    # Check email verification
    if REQUIRE_EMAIL_VERIFICATION and not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "email_not_verified", "message": "Please verify your email before logging in"},
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    session.add(user)
    session.commit()

    # Create tokens
    token_data = {
        "sub": user.email,
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "role": user.role,
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    logger.info(f"User logged in: {user.email}")

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="Bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    session: Session = Depends(get_session),
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    """
    from jose import JWTError, jwt
    from .auth import SECRET_KEY, ALGORITHM, TOKEN_AUDIENCE, TOKEN_ISSUER

    try:
        payload = jwt.decode(
            request.refresh_token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=TOKEN_AUDIENCE,
            issuer=TOKEN_ISSUER,
        )

        # Validate token type
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_token", "message": "Invalid refresh token"},
            )

        email = payload.get("sub")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"error": "invalid_token", "message": "Invalid refresh token"},
            )

    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_token", "message": "Invalid or expired refresh token"},
        )

    # Verify user still exists and is active
    user = session.exec(select(User).where(User.email == email)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_token", "message": "User not found or disabled"},
        )

    # Create new tokens
    token_data = {
        "sub": user.email,
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "role": user.role,
    }

    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="Bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/password/reset-request", response_model=MessageResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> MessageResponse:
    """
    Request password reset email.

    Always returns success to prevent email enumeration.
    """
    user = session.exec(select(User).where(User.email == request.email)).first()

    if user and user.is_active:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        user.password_reset_token = reset_token
        user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=PASSWORD_RESET_EXPIRE_HOURS)
        session.add(user)
        session.commit()

        # Send email
        background_tasks.add_task(send_password_reset_email, user.email, reset_token, user.name)
        logger.info(f"Password reset requested for: {request.email}")

    # Always return success to prevent enumeration
    return MessageResponse(message="If an account exists with this email, you will receive a password reset link")


@router.post("/password/reset", response_model=MessageResponse)
async def reset_password(
    request: PasswordResetConfirm,
    session: Session = Depends(get_session),
) -> MessageResponse:
    """
    Reset password using reset token.
    """
    user = session.exec(
        select(User).where(User.password_reset_token == request.token)
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_token", "message": "Invalid or expired reset token"},
        )

    # Check expiration
    if user.password_reset_expires and datetime.now(timezone.utc) > user.password_reset_expires:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "expired_token", "message": "Reset token has expired"},
        )

    # Hash new password
    try:
        hashed_password = get_password_hash(request.new_password)
    except PasswordValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "weak_password", "message": str(e)},
        )

    # Update password and clear reset token
    user.hashed_password = hashed_password
    user.password_reset_token = None
    user.password_reset_expires = None
    user.updated_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()

    logger.info(f"Password reset completed for: {user.email}")

    return MessageResponse(message="Password has been reset successfully")


@router.post("/email/verify", response_model=MessageResponse)
async def verify_email(
    request: EmailVerifyRequest,
    session: Session = Depends(get_session),
) -> MessageResponse:
    """
    Verify email address using verification token.
    """
    user = session.exec(
        select(User).where(User.email_verification_token == request.token)
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "invalid_token", "message": "Invalid verification token"},
        )

    if user.email_verified:
        return MessageResponse(message="Email already verified")

    # Mark as verified
    user.email_verified = True
    user.email_verification_token = None
    user.updated_at = datetime.now(timezone.utc)
    session.add(user)
    session.commit()

    logger.info(f"Email verified for: {user.email}")

    return MessageResponse(message="Email verified successfully")


@router.post("/email/resend", response_model=MessageResponse)
async def resend_verification(
    request: ResendVerificationRequest,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session),
) -> MessageResponse:
    """
    Resend verification email.
    """
    user = session.exec(select(User).where(User.email == request.email)).first()

    if user and not user.email_verified:
        # Generate new token
        verification_token = secrets.token_urlsafe(32)
        user.email_verification_token = verification_token
        session.add(user)
        session.commit()

        # Send email
        background_tasks.add_task(send_verification_email, user.email, verification_token, user.name)

    # Always return success to prevent enumeration
    return MessageResponse(message="If your email is registered and not verified, you will receive a verification email")


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Get current authenticated user info.
    """
    return UserResponse(
        user_id=current_user.user_id,
        email=current_user.email,
        name=current_user.name,
        role=current_user.role,
        tenant_id=current_user.tenant_id,
        email_verified=current_user.email_verified,
        created_at=current_user.created_at,
    )


@router.post("/logout", response_model=MessageResponse)
async def logout() -> MessageResponse:
    """
    Logout (client-side token invalidation).

    Note: JWT tokens cannot be truly invalidated server-side without a blocklist.
    The client should delete the tokens. For enterprise use, consider:
    - Token blocklist in Redis
    - Short token expiration with refresh tokens
    - Token versioning per user
    """
    return MessageResponse(message="Logged out successfully. Please delete your tokens.")
