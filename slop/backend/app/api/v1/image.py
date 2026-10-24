import asyncio
from uuid import UUID

from fastapi import APIRouter, Depends, Request, UploadFile, WebSocket, WebSocketDisconnect, status
from starlette.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import optional_user
from app.core.config import settings
from app.core.security import utcnow
from app.db.session import get_session
from app.ml.image_detector import analyze_image, analyze_image_bytes
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.services.auth_service import request_ip
from app.utils.file_handler import save_upload

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/analyze")
async def analyze(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    path, mime, size = await save_upload(file, "image")
    result = await run_in_threadpool(analyze_image, str(path))
    job = Job(
        user_id=user.id if user else None,
        job_type="image",
        status="completed",
        file_path=str(path),
        file_name=file.filename,
        file_size_bytes=size,
        mime_type=mime,
        ip_address=request_ip(request),
        completed_at=utcnow(),
    )
    session.add(job)
    await session.flush()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], heatmap_path=str(path)))
    await session.commit()
    return {"job_id": str(job.id), "result": result, "status": "completed"}


@router.get("/result/{job_id}")
async def result(job_id: str, session: AsyncSession = Depends(get_session)):
    try:
        uuid_obj = UUID(job_id)
    except ValueError:
        return {"status": "missing", "job_id": job_id}
    job = await session.get(Job, uuid_obj)
    if not job:
        return {"status": "missing", "job_id": job_id}
    return {"status": job.status, "job_id": str(job.id)}


@router.get("/heatmap/{job_id}")
async def heatmap(job_id: str):
    return {"job_id": job_id, "message": "heatmap is returned inline as base64 in analyze responses"}


@router.websocket("/live")
async def live_image_detection(websocket: WebSocket):
    await websocket.accept()
    max_frame_bytes = settings.image_live_max_frame_kb * 1024
    frame_index = 0
    last_frame_time = 0.0
    min_frame_interval = 0.20  # Max 5 frames per second per connection to prevent thread exhaustion
    try:
        while True:
            frame = await websocket.receive_bytes()
            frame_index += 1
            now = asyncio.get_event_loop().time()
            if now - last_frame_time < min_frame_interval:
                await websocket.send_json({
                    "status": "rate_limited",
                    "frame_index": frame_index,
                    "detail": "frame_rate_exceeded_max_5fps",
                })
                continue

            last_frame_time = now

            if len(frame) > max_frame_bytes:
                await websocket.send_json({"error": "frame_too_large", "max_kb": settings.image_live_max_frame_kb})
                continue
            try:
                result = await run_in_threadpool(analyze_image_bytes, frame)
            except Exception:
                await websocket.send_json({"error": "invalid_image_frame", "frame_index": frame_index})
                continue
            await websocket.send_json({"status": "completed", "frame_index": frame_index, "result": result})
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
