from fastapi import APIRouter

from app.api.v1 import auth, billing, image, jobs, ops, text, video

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(text.router)
api_router.include_router(image.router)
api_router.include_router(video.router)
api_router.include_router(jobs.router)
api_router.include_router(billing.router)
api_router.include_router(ops.router)
