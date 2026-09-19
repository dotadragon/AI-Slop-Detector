from uuid import UUID

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_token
from app.db.session import get_session
from app.models.user import User


async def current_user(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing_access_token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = decode_token(token, "access")
        user_id = UUID(payload["sub"])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid_access_token") from exc
    user = await session.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="inactive_user")
    return user


async def optional_user(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User | None:
    if not authorization:
        return None
    return await current_user(authorization, session)
