import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from jose import JWTError, jwt  # type: ignore
from passlib.context import CryptContext

from app.core.config import settings
from app.core.rsa_keys import ensure_rsa_keys

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_token(subject: str, token_type: str, expires_delta: timedelta, extra: dict[str, Any] | None = None) -> str:
    private_key, _ = ensure_rsa_keys()
    now = utcnow()
    payload: dict[str, Any] = {
        "sub": subject,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": secrets.token_urlsafe(16),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, private_key, algorithm="RS256")


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    _, public_key = ensure_rsa_keys()
    try:
        payload = jwt.decode(token, public_key, algorithms=["RS256"])
    except JWTError as exc:
        raise ValueError("invalid_token") from exc
    if expected_type and payload.get("typ") != expected_type:
        raise ValueError("invalid_token_type")
    return dict(payload)


def create_access_token(user_id: str) -> str:
    return create_token(user_id, "access", timedelta(minutes=settings.access_token_expire_minutes))


def create_refresh_token(user_id: str, family_id: str | None = None) -> str:
    return create_token(
        user_id,
        "refresh",
        timedelta(days=settings.refresh_token_expire_days),
        {"family": family_id or secrets.token_urlsafe(12)},
    )


def generate_raw_token() -> str:
    return secrets.token_urlsafe(32)


def sha256_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def password_is_strong(password: str) -> bool:
    return (
        len(password) >= 8
        and any(char.isupper() for char in password)
        and any(char.isdigit() for char in password)
        and any(not char.isalnum() for char in password)
    )
