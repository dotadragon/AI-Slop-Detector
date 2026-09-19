import pyotp
from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import current_user
from app.db.session import get_session
from app.models.user import EmailVerificationToken, User
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserOut,
    Verify2FARequest,
    VerifyTokenRequest,
)
from app.services.auth_service import enable_2fa, forgot_password, login_user, refresh_session, register_user, reset_password
from app.core.security import sha256_token, utcnow
from app.services.token_store import token_store
from app.core.config import settings
from app.core.security import generate_csrf_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/csrf")
async def csrf(response: Response):
    token = generate_csrf_token()
    secure = settings.app_env == "production"
    response.set_cookie("csrf_token", token, httponly=False, secure=secure, samesite="lax", max_age=3600)
    return {"csrf_token": token}


@router.post("/register")
async def register(payload: RegisterRequest, request: Request, session: AsyncSession = Depends(get_session)):
    user = await register_user(session, payload, request)
    await session.commit()
    return {"user_id": str(user.id), "message": "registered"}


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, request: Request, response: Response, session: AsyncSession = Depends(get_session)):
    access_token = await login_user(session, payload, request, response)
    await session.commit()
    return TokenResponse(access_token=access_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(request: Request, response: Response):
    return TokenResponse(access_token=await refresh_session(request.cookies.get("refresh_token"), response))


@router.post("/logout")
async def logout(response: Response, user: User = Depends(current_user)):
    token_store.revoke_user(str(user.id))
    response.delete_cookie("refresh_token")
    response.delete_cookie("csrf_token")
    return {"message": "logged_out"}


@router.post("/logout-all")
async def logout_all(response: Response, user: User = Depends(current_user)):
    token_store.revoke_user(str(user.id))
    response.delete_cookie("refresh_token")
    response.delete_cookie("csrf_token")
    return {"message": "all_sessions_revoked"}


@router.post("/forgot-password")
async def forgot_password_route(payload: ForgotPasswordRequest, request: Request, session: AsyncSession = Depends(get_session)):
    if not payload.is_honeypot:
        await forgot_password(session, payload.email, request)
        await session.commit()
    return {"message": "If the email exists, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password_route(payload: ResetPasswordRequest, request: Request, session: AsyncSession = Depends(get_session)):
    await reset_password(session, payload, request)
    await session.commit()
    return {"message": "password_reset_complete"}


@router.post("/verify-email")
async def verify_email(payload: VerifyTokenRequest, session: AsyncSession = Depends(get_session)):
    token = await session.scalar(select(EmailVerificationToken).where(EmailVerificationToken.token_hash == sha256_token(payload.token)))
    if not token or token.used or token.expires_at <= utcnow():
        return {"message": "invalid_or_expired_token"}
    token.used = True
    user = await session.get(User, token.user_id)
    if user:
        user.is_verified = True
    await session.commit()
    return {"message": "email_verified"}


@router.post("/enable-2fa")
async def enable_two_factor(session: AsyncSession = Depends(get_session), user: User = Depends(current_user)):
    result = await enable_2fa(session, user)
    await session.commit()
    return result


@router.post("/verify-2fa")
async def verify_two_factor(payload: Verify2FARequest, session: AsyncSession = Depends(get_session), user: User = Depends(current_user)):
    if user.totp_secret and pyotp.TOTP(user.totp_secret).verify(payload.code, valid_window=1):
        user.totp_enabled = True
        await session.commit()
        return {"enabled": True}
    return {"enabled": False}


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(current_user)):
    return UserOut(
        id=str(user.id),
        email=user.email,
        is_active=user.is_active,
        is_verified=user.is_verified,
        totp_enabled=user.totp_enabled,
    )
