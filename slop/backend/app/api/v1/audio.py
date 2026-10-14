import asyncio
import json
import threading
from uuid import UUID

from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.deps import optional_user
from app.core.config import settings
from app.core.security import utcnow
from app.db.session import get_session
from app.ml.audio_detector import analyze_audio
from app.ml.vram_manager import serialized_gpu_inference
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.services.auth_service import request_ip
from app.services.token_store import token_store
from app.utils.file_handler import save_upload

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/upload")
async def upload(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    if not settings.enable_audio_detector:
        return {"status": "disabled", "detail": "audio detector is feature-gated off until validated"}
    path, mime, size = await save_upload(file, "audio")
    user_id = user.id if user else None
    job = Job(user_id=user_id, job_type="audio", status="processing", file_path=str(path), file_name=file.filename, file_size_bytes=size, mime_type=mime, ip_address=request_ip(request))
    session.add(job)
    await session.flush()
    token_store.progress[str(job.id)] = 5

    async with serialized_gpu_inference("audio"):
        result = await run_in_threadpool(analyze_audio, str(path), str(job.id))

    token_store.progress[str(job.id)] = 100
    job.status = "completed"
    job.completed_at = utcnow()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], frame_data=[]))
    await session.commit()
    return {"job_id": str(job.id), "status": "completed", "result": result}


@router.post("/upload-stream")
async def upload_stream(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    """Upload audio with SSE progress (analysis phases reported per event)."""
    if not settings.enable_audio_detector:
        return StreamingResponse(
            (lambda: iter([f"data: {json.dumps({'type': 'disabled', 'detail': 'audio detector is feature-gated off'})}\n\n"]))(),
            media_type="text/event-stream",
        )
    path, mime, size = await save_upload(file, "audio")
    job = Job(
        user_id=user.id if user else None,
        job_type="audio",
        status="processing",
        file_path=str(path),
        file_name=file.filename,
        file_size_bytes=size,
        mime_type=mime,
        ip_address=request_ip(request),
    )
    session.add(job)
    await session.flush()
    job_id = str(job.id)
    token_store.progress[job_id] = 10

    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    done_sentinel = object()

    def _producer():
        try:
            token_store.progress[job_id] = 25
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "job_id": job_id, "progress_percent": 25})
            result = analyze_audio(str(path), job_id)
            token_store.progress[job_id] = 90
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", "job_id": job_id, "progress_percent": 90})
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "complete", "job_id": job_id, "result": result})
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "job_id": job_id, "error": str(exc)[:200]})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done_sentinel)

    async def sse_generator():
        async with serialized_gpu_inference("audio"):
            threading.Thread(target=_producer, daemon=True).start()
            final_result = None
            while True:
                item = await queue.get()
                if item is done_sentinel:
                    break
                yield f"data: {json.dumps(item)}\n\n"
                if isinstance(item, dict) and item.get("type") == "complete" and "result" in item:
                    final_result = item["result"]

            if final_result:
                try:
                    job.status = "completed"
                    job.completed_at = utcnow()
                    session.add(DetectionResult(job_id=job.id, overall_score=final_result["overall_score"], signals=final_result["signals"], frame_data=[]))
                    await session.commit()
                except Exception:
                    pass

            token_store.progress[job_id] = 100
            yield f"data: {json.dumps({'type': 'done', 'job_id': job_id})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/result/{job_id}")
async def result(job_id: str, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
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

    return {"status": job.status, "progress": token_store.progress.get(job_id, 100 if job.status == "completed" else 0)}
