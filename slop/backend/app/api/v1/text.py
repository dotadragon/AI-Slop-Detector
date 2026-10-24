from uuid import UUID
from fastapi import APIRouter, Depends, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.deps import optional_user
from app.core.security import utcnow
from app.db.session import get_session
from app.ml.text_detector import analyze_text
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.schemas.detection import AnalyzeResponse, TextAnalyzeRequest
from app.services.auth_service import request_ip
from app.utils.file_handler import extract_text_from_upload, chunk_text_sections

router = APIRouter(prefix="/text", tags=["text"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: TextAnalyzeRequest, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    result = await run_in_threadpool(analyze_text, payload.text)
    job = Job(user_id=user.id if user else None, job_type="text", status="completed", ip_address=request_ip(request), completed_at=utcnow())
    session.add(job)
    await session.flush()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], sentence_data=result["sentences"]))
    await session.commit()
    return {"job_id": str(job.id), "result": result, "status": "completed"}


@router.post("/upload")
async def upload_text(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    text = await extract_text_from_upload(file)
    sections = chunk_text_sections(text)
    result = await run_in_threadpool(analyze_text, text)

    # Multi-section document threat heat map for multi-page/large documents
    if len(sections) > 1:
        section_results = []
        for s in sections[:12]:
            s_res = await run_in_threadpool(analyze_text, s["text"])
            section_results.append({
                "index": s["index"],
                "word_count": s["word_count"],
                "score": s_res["overall_score"],
                "preview": s["text"][:120].strip() + ("..." if len(s["text"]) > 120 else ""),
            })
        result["section_scores"] = section_results
        result["signals"]["document_sections_analyzed"] = len(section_results)

    job = Job(user_id=user.id if user else None, job_type="text", status="completed", file_name=file.filename, ip_address=request_ip(request), completed_at=utcnow())
    session.add(job)
    await session.flush()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], sentence_data=result["sentences"]))
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
