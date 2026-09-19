from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.middleware import CsrfMiddleware, MethodGuardMiddleware, SecurityHeadersMiddleware, SlidingWindowDdosMiddleware
from app.core.rsa_keys import ensure_rsa_keys
from app.db.session import create_all

from app.utils.file_handler import cleanup_expired_uploads

logger = structlog.get_logger()


def configure_error_tracking() -> None:
    # ponytail: Rung 1 optional third-party telemetry, stdlib logging default
    if not settings.sentry_dsn:
        return
    try:
        import importlib
        sentry_sdk = importlib.import_module("sentry_sdk")
        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1, environment=settings.app_env)
    except ImportError:
        logger.warning("sentry_dsn_configured_but_sentry_sdk_not_installed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ponytail: Rung 1 direct SQLite schema creation on startup, zero alembic overhead
    ensure_rsa_keys()
    await create_all()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    purged = cleanup_expired_uploads()
    if purged > 0:
        logger.info("stale_uploads_purged_on_startup", count=purged)
    logger.info("startup_complete", app=settings.app_name, env=settings.app_env)
    yield


def create_app() -> FastAPI:
    configure_error_tracking()
    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(CsrfMiddleware)
    app.add_middleware(SlidingWindowDdosMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MethodGuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=["Authorization", "Content-Type", "X-CSRF-Token"],
    )
    app.include_router(api_router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "app": settings.app_name}

    return app


app = create_app()
