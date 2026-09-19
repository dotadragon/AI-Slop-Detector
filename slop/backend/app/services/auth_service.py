import base64
import io
import secrets
from datetime import timedelta

import pyotp
import qrcode
from fastapi import HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    generate_raw_token,
    hash_password,
    sha256_token,
    utcnow,
    verify_password,
)
from app.models.user import EmailVerificationToken, PasswordResetToken, TotpBackupCode, User
from app.schemas.auth import LoginRequest, RegisterRequest, ResetPasswordRequest
from app.services.audit import audit
from app.services.email import send_email
from app.services.token_store import token_store


def request_ip(request: Request) -> str | None:
    return request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (request.client.host if request.client else None)


async def register_user(session: AsyncSession, payload: RegisterRequest, request: Request) -> User:
    if payload.is_honeypot:
        raise HTTPException(status_code=status.HTTP_200_OK, detail="registered")
    existing = await session.scalar(select(User).where(User.email == payload.email.lower()))
    if existing:
        raise HTTPException(status_code=409, detail="email_already_registered")
    user = User(
        email=payload.email.lower(),
        hashed_password=hash_password(payload.password),
        totp_secret=pyotp.random_base32() if payload.enable_2fa else None,
        totp_enabled=False,
    )
    session.add(user)
    await session.flush()
    raw = generate_raw_token()
    session.add(
        EmailVerificationToken(
            token_hash=sha256_token(raw),
            user_id=user.id,
            expires_at=utcnow() + timedelta(hours=settings.verification_token_expire_hours),
        )
    )
    await audit(session, "register", request_ip(request), request.headers.get("user-agent"), user.id)
    await send_email(user.email, "Verify your SLOP account", f"{settings.frontend_url}/verify-email?token={raw}")
    await send_email(
        user.email,
        "Welcome to SLOP",
        f"Start with your first scan: {settings.frontend_url}/detector\n\nUnsubscribe: {settings.marketing_unsubscribe_url}",
    )
    return user


def _set_auth_cookies(response: Response, refresh_token: str, csrf_token: str) -> None:
    secure = settings.app_env == "production"
    response.set_cookie(
        "refresh_token",
        refresh_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=settings.refresh_token_expire_days * 86400,
    )
    response.set_cookie("csrf_token", csrf_token, httponly=False, secure=secure, samesite="lax", max_age=86400)


async def login_user(session: AsyncSession, payload: LoginRequest, request: Request, response: Response) -> str:
    if payload.is_honeypot:
        raise HTTPException(status_code=status.HTTP_200_OK, detail="ok")
    user = await session.scalar(select(User).where(User.email == payload.email.lower()))
    ip = request_ip(request)
    if not user:
        await audit(session, "login_failed", ip, request.headers.get("user-agent"), details={"reason": "unknown_email"})
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if user.locked_until and user.locked_until > utcnow():
        await audit(session, "login_locked", ip, request.headers.get("user-agent"), user.id)
        raise HTTPException(status_code=423, detail="account_locked")
    if not verify_password(payload.password, user.hashed_password):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= 5:
            user.lockout_count += 1
            minutes = min(60, 15 * (2 ** max(user.lockout_count - 1, 0)))
            user.locked_until = utcnow() + timedelta(minutes=minutes)
        await audit(session, "login_failed", ip, request.headers.get("user-agent"), user.id, {"reason": "bad_password"})
        raise HTTPException(status_code=401, detail="invalid_credentials")
    if user.totp_enabled:
        totp_ok = bool(payload.totp_code and pyotp.TOTP(user.totp_secret).verify(payload.totp_code, valid_window=1))
        if not totp_ok:
            await audit(session, "totp_failed", ip, request.headers.get("user-agent"), user.id)
            raise HTTPException(status_code=401, detail="totp_required")

    user.failed_login_attempts = 0
    user.locked_until = None
    family_id = secrets.token_urlsafe(12)
    refresh_token = create_refresh_token(str(user.id), family_id)
    token_store.store_refresh(refresh_token, str(user.id), family_id, settings.refresh_token_expire_days * 86400)
    csrf_token = token_store.new_session_csrf(str(user.id))
    _set_auth_cookies(response, refresh_token, csrf_token)
    await audit(session, "login_success", ip, request.headers.get("user-agent"), user.id)
    return create_access_token(str(user.id))


async def refresh_session(raw_refresh: str | None, response: Response) -> str:
    if not raw_refresh:
        raise HTTPException(status_code=401, detail="missing_refresh_token")
    record = token_store.consume_refresh(raw_refresh)
    if not record:
        raise HTTPException(status_code=401, detail="invalid_refresh_token")
    refresh_token = create_refresh_token(record.user_id, record.family_id)
    token_store.store_refresh(refresh_token, record.user_id, record.family_id, settings.refresh_token_expire_days * 86400)
    csrf_token = token_store.new_session_csrf(record.user_id)
    _set_auth_cookies(response, refresh_token, csrf_token)
    return create_access_token(record.user_id)


async def forgot_password(session: AsyncSession, email: str, request: Request) -> None:
    user = await session.scalar(select(User).where(User.email == email.lower()))
    if user:
        raw = generate_raw_token()
        session.add(
            PasswordResetToken(
                token_hash=sha256_token(raw),
                user_id=user.id,
                expires_at=utcnow() + timedelta(minutes=settings.reset_token_expire_minutes),
                ip_address=request_ip(request),
            )
        )
        await audit(session, "password_reset_requested", request_ip(request), request.headers.get("user-agent"), user.id)
        await send_email(user.email, "Reset your SLOP password", f"{settings.frontend_url}/reset-password?token={raw}")


async def reset_password(session: AsyncSession, payload: ResetPasswordRequest, request: Request) -> None:
    if payload.is_honeypot:
        return
    token = await session.scalar(select(PasswordResetToken).where(PasswordResetToken.token_hash == sha256_token(payload.token)))
    if not token or token.used or token.expires_at <= utcnow():
        raise HTTPException(status_code=400, detail="invalid_or_expired_token")
    token.used = True
    user = await session.get(User, token.user_id)
    if not user:
        raise HTTPException(status_code=400, detail="invalid_token")
    user.hashed_password = hash_password(payload.new_password)
    token_store.revoke_user(str(user.id))
    await audit(session, "password_reset_completed", request_ip(request), request.headers.get("user-agent"), user.id)


async def enable_2fa(session: AsyncSession, user: User) -> dict:
    secret = user.totp_secret or pyotp.random_base32()
    user.totp_secret = secret
    otp_uri = pyotp.TOTP(secret).provisioning_uri(name=user.email, issuer_name="SLOP")
    img = qrcode.make(otp_uri)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
    for code in backup_codes:
        session.add(TotpBackupCode(user_id=user.id, code_hash=hash_password(code)))
    return {
        "qr_code_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "backup_codes": backup_codes,
    }
