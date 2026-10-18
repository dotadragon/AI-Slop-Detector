from pathlib import Path
import sys
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import current_user
from app.core.config import settings
from app.db.session import get_session
from app.ml.vram_manager import VRAM_MAX_CEILING_BYTES, get_vram_stats, gpu_inference_lock
from app.models.feedback import UserFeedback
from app.models.job import DetectionResult, Job
from app.models.user import User

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/telemetry")
async def hardware_telemetry():
    """
    Returns real-time GPU/VRAM hardware allocations, lock status, and safety ceiling.
    """
    vram = get_vram_stats()
    is_locked = (gpu_inference_lock._value == 0) if hasattr(gpu_inference_lock, "_value") else False
    return {
        "status": "online",
        "telemetry": {
            **vram,
            "vram_ceiling_mb": round(VRAM_MAX_CEILING_BYTES / (1024 * 1024), 2),
            "gpu_locked": is_locked,
        },
    }


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


class FeedbackIn(BaseModel):
    job_id: str
    verdict_type: str = Field(pattern="^(false_positive|false_negative)$")
    comment: str | None = Field(default=None, max_length=4000)


@router.post("/feedback")
async def submit_feedback(
    payload: FeedbackIn,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
):
    """Report a wrong verdict for a job — feeds the human review queue and
    becomes the long-term source of labeled training data."""
    import uuid as _uuid

    try:
        job_uuid = _uuid.UUID(payload.job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid job_id")

    job = await session.get(Job, job_uuid)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    result = (
        await session.execute(select(DetectionResult).where(DetectionResult.job_id == job_uuid))
    ).scalar_one_or_none()

    feedback = UserFeedback(
        job_id=job_uuid,
        user_id=user.id,
        verdict_type=payload.verdict_type,
        comment=payload.comment,
        context={
            "modality": job.job_type,
            "overall_score": result.overall_score if result else None,
            "job_status": job.status,
        },
    )
    session.add(feedback)
    await session.commit()
    return {"status": "received", "feedback_id": str(feedback.id)}


@router.get("/feedback")
async def list_feedback(
    status: str = "open",
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
    user: User = Depends(current_user),
):
    """Review queue: list feedback reports (most recent first)."""
    rows = (
        await session.execute(
            select(UserFeedback)
            .where(UserFeedback.status == status)
            .order_by(UserFeedback.created_at.desc())
            .limit(min(limit, 200))
        )
    ).scalars().all()
    return {
        "status": status,
        "count": len(rows),
        "items": [
            {
                "id": str(r.id),
                "job_id": str(r.job_id),
                "verdict_type": r.verdict_type,
                "comment": r.comment,
                "context": r.context,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.delete("/privacy/account")
async def delete_account(session: AsyncSession = Depends(get_session), user: User = Depends(current_user)):
    user.is_active = False
    await session.commit()
    return {"message": "account_deactivation_queued", "user_id": str(user.id)}


@router.get("/calibration-dashboard")
async def get_calibration_dashboard():
    """Serve the public calibration & accuracy dashboard HTML."""
    from fastapi.responses import HTMLResponse
    dashboard_path = None
    for p in Path(__file__).resolve().parents:
        cand = p / "dashboard" / "eval_dashboard.html"
        if not cand.exists():
            cand = p / "artifacts" / "eval_dashboard.html"
        if cand.exists():
            dashboard_path = cand
            break
        gen_script = p / "scripts" / "generate_dashboard.py"
        if not gen_script.exists():
            gen_script = p / "generate_dashboard.py"
        if gen_script.exists():
            import subprocess
            subprocess.run([sys.executable, str(gen_script)], check=False)
            if cand.exists():
                dashboard_path = cand
                break
    if dashboard_path and dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"), status_code=200)
    return HTMLResponse(content="<html><body><h2>Calibration dashboard not generated yet.</h2></body></html>", status_code=404)


@router.get("/calibration-report")
async def get_calibration_report():
    """Return heldout_report.json metrics as JSON."""
    import json
    for p in Path(__file__).resolve().parents:
        cand = p / "heldout_report.json"
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse heldout_report.json: {e}")
    raise HTTPException(status_code=404, detail="heldout_report.json not found")


