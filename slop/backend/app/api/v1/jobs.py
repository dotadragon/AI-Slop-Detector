import asyncio
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import current_user, optional_user
from app.db.session import get_session
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.services.auth_service import request_ip
from app.services.dossier_service import generate_dossier_html
from app.services.token_store import token_store

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}/status")
async def status(job_id: str, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    try:
        uuid_obj = UUID(job_id)
    except ValueError:
        return {"status": "missing", "progress": 0}
    job = await session.get(Job, uuid_obj)
    if not job:
        return {"status": "missing", "progress": 0}

    client_ip = request_ip(request)
    if user and job.user_id and job.user_id != user.id:
        return {"status": "forbidden", "progress": 0}
    elif not user and job.user_id is not None:
        return {"status": "unauthorized", "progress": 0}
    elif not user and job.ip_address and job.ip_address != client_ip:
        return {"status": "forbidden", "progress": 0}

    return {
        "status": job.status,
        "progress": token_store.progress.get(job_id, 100 if job.status == "completed" else 0),
        "created_at": job.created_at,
        "completed_at": job.completed_at,
    }


@router.get("/history")
async def history(request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    stmt = (
        select(Job, DetectionResult)
        .join(DetectionResult, DetectionResult.job_id == Job.id, isouter=True)
    )
    if user:
        stmt = stmt.where(Job.user_id == user.id)
    else:
        ip = request_ip(request)
        # Security fix: strictly scope anonymous history to caller's IP, preventing global cross-user leak
        stmt = stmt.where(and_(Job.ip_address == ip, Job.user_id.is_(None)))

    rows = await session.execute(stmt.order_by(Job.created_at.desc()).limit(50))
    return [
        {
            "job_id": str(job.id),
            "type": job.job_type,
            "score": result.overall_score if result else None,
            "status": job.status,
            "created_at": job.created_at,
        }
        for job, result in rows
    ]


@router.get("/{job_id}/dossier", response_class=HTMLResponse)
async def get_dossier(job_id: str, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    """
    Returns a sealed cryptographic evidence dossier in standalone, print-ready HTML format.
    """
    try:
        uuid_obj = UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid_job_id_format")

    job = await session.get(Job, uuid_obj)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")

    # Access control verification: authenticated owner or matching anonymous IP
    client_ip = request_ip(request)
    if user and job.user_id and job.user_id != user.id:
        raise HTTPException(status_code=403, detail="access_denied_to_job")
    elif not user and job.user_id is not None:
        raise HTTPException(status_code=401, detail="authentication_required")
    elif not user and job.ip_address and job.ip_address != client_ip:
        raise HTTPException(status_code=403, detail="access_denied_to_anonymous_job")

    # Fetch detection result
    res_stmt = select(DetectionResult).where(DetectionResult.job_id == uuid_obj)
    res_exec = await session.execute(res_stmt)
    result = res_exec.scalar_one_or_none()

    html = generate_dossier_html(job, result)
    return HTMLResponse(content=html, media_type="text/html")


@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    try:
        UUID(job_id)
    except ValueError:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        last_progress = -1
        while True:
            progress = token_store.progress.get(job_id, 0)
            if progress != last_progress or progress >= 100:
                await websocket.send_json({"job_id": job_id, "progress": progress})
                last_progress = progress
            if progress >= 100:
                break
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
