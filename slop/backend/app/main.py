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


def _verify_models_on_startup() -> None:
    # ponytail: Rule 1 & Stage B6 real forward pass verification on startup
    try:
        import io
        from PIL import Image
        from app.ml.text_detector import analyze_text
        from app.ml.image_detector import analyze_image_bytes

        text_res = analyze_text("SLOP startup verification pass.")
        if "overall_score" not in text_res:
            raise RuntimeError("Text detector failed to return overall_score")

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(buf, format="JPEG")
        img_res = analyze_image_bytes(buf.getvalue())
        if "overall_score" not in img_res:
            raise RuntimeError("Image detector failed to return overall_score")

        active_dev = settings.compute_device
        gpu_name = None
        vram_mb = 0.0
        if active_dev == "cuda":
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
            vram_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        logger.info(
            "models_verified_on_startup",
            text_status="ok",
            image_status="ok",
            device=active_dev,
            gpu=gpu_name,
            vram_mb=vram_mb,
        )
        print(f"models_verified_on_startup | device={active_dev} | gpu={gpu_name} | vram_mb={vram_mb}")
    except Exception as e:
        logger.critical("model_startup_verification_failed", error=str(e))
        raise RuntimeError(f"SLOP Model Startup Verification Failed: {e}") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ponytail: Rung 1 direct SQLite schema creation on startup, zero alembic overhead
    ensure_rsa_keys()
    await create_all()
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    purged = cleanup_expired_uploads()
    if purged > 0:
        logger.info("stale_uploads_purged_on_startup", count=purged)
    _verify_models_on_startup()
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

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus text exposition of inference latency, failures, and VRAM high-water mark."""
        from fastapi.responses import Response
        from app.ml.vram_manager import render_prometheus_metrics
        return Response(content=render_prometheus_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/calibration", include_in_schema=False)
    async def calibration_dashboard():
        """Serve the public calibration & accuracy dashboard HTML."""
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        import sys
        dashboard_path = None
        for p in Path(__file__).resolve().parents:
            cand = p / "dashboard" / "eval_dashboard.html"
            if cand.exists():
                dashboard_path = cand
                break
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

    return app


app = create_app()
