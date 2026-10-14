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
from app.ml.document_forensics import analyze_document
from app.ml.vram_manager import serialized_gpu_inference
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.services.auth_service import request_ip
from app.services.token_store import token_store
from app.utils.file_handler import save_upload

router = APIRouter(prefix="/document", tags=["document"])


@router.post("/upload")
async def upload(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    if not settings.enable_document_forensics:
        return {"status": "disabled", "detail": "document forensics is feature-gated off until validated"}
    path, mime, size = await save_upload(file, "text")
    user_id = user.id if user else None
    job = Job(user_id=user_id, job_type="document", status="processing", file_path=str(path), file_name=file.filename, file_size_bytes=size, mime_type=mime, ip_address=request_ip(request))
    session.add(job)
    await session.flush()
    token_store.progress[str(job.id)] = 5

    async with serialized_gpu_inference("document"):
        result = await run_in_threadpool(analyze_document, str(path), str(job.id))

    token_store.progress[str(job.id)] = 100
    job.status = "completed"
    job.completed_at = utcnow()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], frame_data=[]))
    await session.commit()
    return {"job_id": str(job.id), "status": "completed", "result": result}


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
