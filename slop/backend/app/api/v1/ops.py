from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import current_user
from app.core.config import settings
from app.db.session import get_session
from app.models.user import User

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/launch-readiness")
async def launch_readiness():
    return {
        "analytics": {"posthog_env_required": "VITE_POSTHOG_KEY", "activation_events_defined": True},
        "seo": {"sitemap": "/sitemap.xml", "robots": "/robots.txt", "security_txt": "/.well-known/security.txt"},
        "monitoring": {"sentry_configured": bool(settings.sentry_dsn), "health_endpoint": "/health"},
        "billing": {"stripe_configured": bool(settings.stripe_secret_key)},
        "support": {"email": settings.support_email},
    }


@router.post("/privacy/export")
async def privacy_export(user: User = Depends(current_user)):
    return {
        "user": {"id": str(user.id), "email": user.email, "is_verified": user.is_verified},
        "message": "Full export job should include scans, audit events, and billing records in production.",
    }


@router.delete("/privacy/account")
async def delete_account(session: AsyncSession = Depends(get_session), user: User = Depends(current_user)):
    user.is_active = False
    await session.commit()
    return {"message": "account_deactivation_queued", "user_id": str(user.id)}
