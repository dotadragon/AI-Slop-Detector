from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import SecurityAuditLog


async def audit(
    session: AsyncSession,
    event_type: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
    user_id=None,
    details: dict | None = None,
) -> None:
    session.add(
        SecurityAuditLog(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details or {},
        )
    )
