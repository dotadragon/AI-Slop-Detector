from fastapi import APIRouter

from app.api.v1 import auth, audio, billing, document, image, jobs, ops, text, video

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(text.router)
api_router.include_router(image.router)
api_router.include_router(video.router)
api_router.include_router(audio.router)
api_router.include_router(document.router)
api_router.include_router(jobs.router)
api_router.include_router(billing.router)
api_router.include_router(ops.router)


@api_router.get("/stats", tags=["ops"], summary="Alias for hardware telemetry")
async def get_stats():
    """Alias for /api/v1/ops/telemetry to stop 404 logging."""
    return await ops.hardware_telemetry()
