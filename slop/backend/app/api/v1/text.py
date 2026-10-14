import re
from uuid import UUID
from fastapi import APIRouter, Depends, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.api.deps import optional_user
from app.core.security import utcnow
from app.db.session import get_session
from app.ml.text_detector import analyze_text
from app.ml.vram_manager import serialized_gpu_inference
from app.models.job import DetectionResult, Job
from app.models.user import User
from app.schemas.detection import AnalyzeResponse, SpecimenMetrics, TextAnalyzeRequest, TextCompareRequest, TextComparisonResponse
from app.services.auth_service import request_ip
from app.utils.file_handler import extract_text_from_upload, chunk_text_sections

router = APIRouter(prefix="/text", tags=["text"])


def _compute_ngram_overlap(text_a: str, text_b: str, n: int = 4) -> tuple[float, list[str]]:
    words_a = re.findall(r"\b\w+\b", text_a.lower())
    words_b = re.findall(r"\b\w+\b", text_b.lower())
    if len(words_a) < n or len(words_b) < n:
        return 0.0, []
    ngrams_a = set(" ".join(words_a[i : i + n]) for i in range(len(words_a) - n + 1))
    ngrams_b = set(" ".join(words_b[i : i + n]) for i in range(len(words_b) - n + 1))
    if not ngrams_a or not ngrams_b:
        return 0.0, []
    intersection = ngrams_a.intersection(ngrams_b)
    min_len = min(len(ngrams_a), len(ngrams_b))
    pct = round((len(intersection) / min_len) * 100.0, 2) if min_len > 0 else 0.0
    return pct, sorted(list(intersection))[:10]


@router.post("/compare", response_model=TextComparisonResponse)
async def compare_specimens(payload: TextCompareRequest):
    """
    Forensically compares two text specimens:
    Computes differential AI probability, burstiness delta, perplexity delta,
    and n-gram verbatim overlap fingerprinting.
    """
    async with serialized_gpu_inference("text"):
        res_a = await run_in_threadpool(analyze_text, payload.specimen_a)
        res_b = await run_in_threadpool(analyze_text, payload.specimen_b)

    sig_a = res_a.get("signals", {})
    sig_b = res_b.get("signals", {})

    metrics_a = SpecimenMetrics(
        overall_score=res_a["overall_score"],
        confidence_band=res_a.get("confidence_band", "UNCERTAIN"),
        burstiness=round(float(sig_a.get("burstiness", 0.0)), 3),
        perplexity=round(float(sig_a["perplexity"]), 2) if "perplexity" in sig_a and sig_a["perplexity"] is not None else None,
        vocabulary_richness=round(float(sig_a.get("vocabulary_richness", 0.0)), 3),
        char_count=len(payload.specimen_a),
        word_count=len(payload.specimen_a.split()),
    )

    metrics_b = SpecimenMetrics(
        overall_score=res_b["overall_score"],
        confidence_band=res_b.get("confidence_band", "UNCERTAIN"),
        burstiness=round(float(sig_b.get("burstiness", 0.0)), 3),
        perplexity=round(float(sig_b["perplexity"]), 2) if "perplexity" in sig_b and sig_b["perplexity"] is not None else None,
        vocabulary_richness=round(float(sig_b.get("vocabulary_richness", 0.0)), 3),
        char_count=len(payload.specimen_b),
        word_count=len(payload.specimen_b.split()),
    )

    delta_score = round(metrics_a.overall_score - metrics_b.overall_score, 3)
    delta_burstiness = round(metrics_a.burstiness - metrics_b.burstiness, 3)
    delta_perplexity = (
        round(metrics_a.perplexity - metrics_b.perplexity, 2)
        if metrics_a.perplexity is not None and metrics_b.perplexity is not None
        else None
    )

    overlap_pct, shared_phrases = _compute_ngram_overlap(payload.specimen_a, payload.specimen_b, n=4)

    if delta_score >= 0.30:
        verdict = f"Specimen A exhibits significantly higher synthetic AI probability (+{delta_score*100:.1f}%) than Specimen B."
    elif delta_score <= -0.30:
        verdict = f"Specimen B exhibits significantly higher synthetic AI probability (+{abs(delta_score)*100:.1f}%) than Specimen A."
    elif overlap_pct >= 25.0:
        verdict = f"High verbatim stylistic and phrase overlap ({overlap_pct}%) detected between both specimens."
    else:
        verdict = "Specimens demonstrate consistent stylistic variance with no extreme AI divergence."

    return TextComparisonResponse(
        specimen_a=metrics_a,
        specimen_b=metrics_b,
        delta_score=delta_score,
        delta_burstiness=delta_burstiness,
        delta_perplexity=delta_perplexity,
        ngram_overlap_pct=overlap_pct,
        shared_phrases=shared_phrases,
        verdict=verdict,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: TextAnalyzeRequest, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    async with serialized_gpu_inference("text"):
        result = await run_in_threadpool(analyze_text, payload.text)
    job = Job(user_id=user.id if user else None, job_type="text", status="completed", ip_address=request_ip(request), completed_at=utcnow())
    session.add(job)
    await session.flush()
    session.add(DetectionResult(job_id=job.id, overall_score=result["overall_score"], signals=result["signals"], sentence_data=result["sentences"]))
    await session.commit()
    # Banding passthrough lives inside `result` (zone_low/zone_high/confidence_band);
    # the AnalyzeResponse top-level contract is locked to 3 fields by test_api_contract.py.
    return {"job_id": str(job.id), "result": result, "status": "completed"}


@router.post("/upload")
async def upload_text(request: Request, file: UploadFile, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    text = await extract_text_from_upload(file)
    sections = chunk_text_sections(text)
    async with serialized_gpu_inference("text"):
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
    # Banding passthrough lives inside `result` (zone_low/zone_high/confidence_band);
    # the AnalyzeResponse top-level contract is locked to 3 fields by test_api_contract.py.
    return {"job_id": str(job.id), "result": result, "status": "completed"}


@router.get("/result/{job_id}")
async def result(job_id: str, request: Request, session: AsyncSession = Depends(get_session), user: User | None = Depends(optional_user)):
    try:
        uuid_obj = UUID(job_id)
    except ValueError:
        return {"status": "missing", "job_id": job_id}
    job = await session.get(Job, uuid_obj)
    if not job:
        return {"status": "missing", "job_id": job_id}

    client_ip = request_ip(request)
    if user and job.user_id and job.user_id != user.id:
        return {"status": "forbidden", "job_id": job_id}
    elif not user and job.user_id is not None:
        return {"status": "unauthorized", "job_id": job_id}
    elif not user and job.ip_address and job.ip_address != client_ip:
        return {"status": "forbidden", "job_id": job_id}

    return {"status": job.status, "job_id": str(job.id)}
