"""
SLOP Dynamic VRAM Manager — Production memory governor for RTX 4050 (6GB).
ponytail: lean, zero-overhead memory eviction & hardware profiling
"""

import asyncio
import gc
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import os
logger = logging.getLogger(__name__)

# ponytail: Rung 3 stdlib hardware serialization lock. Zero Redis/Celery queue bloat.
# Guarantees only one neural forward pass executes on the RTX 4050 at any moment.
gpu_inference_lock = asyncio.Semaphore(1)
_VRAM_CEILING_GB = float(os.environ.get("SLOP_VRAM_CEILING_GB", "5.6"))
VRAM_MAX_CEILING_BYTES = int(_VRAM_CEILING_GB * 1024 * 1024 * 1024)

# Phase 3 hardening: a stalled/half-open client must never hold the GPU lock
# indefinitely and deadlock every other modality. Acquisition above this timeout
# raises TimeoutUnavailableError instead of blocking forever.
GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS = 30.0

# Cross-modality VRAM auto-eviction tracker
_active_gpu_modality: str | None = None
_last_active_modality: str | None = None


class GPUInferenceTimeoutError(RuntimeError):
    """Raised when the serialized GPU inference lock cannot be acquired in time."""


# Phase 4 observability: lightweight stdlib metrics registry (Prometheus-style).
_metrics_lock = asyncio.Lock()
metrics: dict[str, float] = {
    "inference_latency_seconds_count": 0.0,
    "inference_latency_seconds_sum": 0.0,
    "inference_failures_total": 0.0,
    "gpu_lock_timeouts_total": 0.0,
}
metrics_by_modality: dict[str, dict[str, float]] = {}
vram_high_water_mark_mb: float = 0.0


def record_inference(modality: str, latency_seconds: float, failed: bool = False) -> None:
    """Record latency/failure metrics for a modality (called by serialized_gpu_inference)."""
    global vram_high_water_mark_mb
    bucket = metrics_by_modality.setdefault(modality, {"count": 0.0, "sum": 0.0, "failures": 0.0})
    bucket["count"] += 1
    bucket["sum"] += latency_seconds
    if failed:
        bucket["failures"] += 1
        metrics["inference_failures_total"] += 1
    metrics["inference_latency_seconds_count"] += 1
    metrics["inference_latency_seconds_sum"] += latency_seconds
    try:
        import torch
        if torch.cuda.is_available():
            vram_high_water_mark_mb = max(
                vram_high_water_mark_mb,
                round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
            )
    except Exception:
        pass


def render_prometheus_metrics() -> str:
    """Render the metrics registry in Prometheus text exposition format."""
    lines = [
        "# HELP slop_inference_latency_seconds Inference latency by modality",
        "# TYPE slop_inference_latency_seconds summary",
    ]
    total_count = metrics["inference_latency_seconds_count"]
    total_sum = metrics["inference_latency_seconds_sum"]
    lines.append(f'slop_inference_latency_seconds_sum {total_sum}')
    lines.append(f'slop_inference_latency_seconds_count {total_count}')
    for modality, bucket in metrics_by_modality.items():
        lines.append(f'slop_inference_latency_seconds_sum{{modality="{modality}"}} {bucket["sum"]}')
        lines.append(f'slop_inference_latency_seconds_count{{modality="{modality}"}} {bucket["count"]}')
    lines.append("# HELP slop_inference_failures_total Total inference failures by modality")
    lines.append("# TYPE slop_inference_failures_total counter")
    for modality, bucket in metrics_by_modality.items():
        lines.append(f'slop_inference_failures_total{{modality="{modality}"}} {bucket["failures"]}')
    lines.append("# HELP slop_vram_high_water_mark_mb Peak VRAM allocated (MB)")
    lines.append("# TYPE slop_vram_high_water_mark_mb gauge")
    lines.append(f"slop_vram_high_water_mark_mb {vram_high_water_mark_mb}")
    lines.append("# HELP slop_gpu_lock_timeouts_total GPU lock acquisition timeouts")
    lines.append("# TYPE slop_gpu_lock_timeouts_total counter")
    lines.append(f"slop_gpu_lock_timeouts_total {metrics['gpu_lock_timeouts_total']}")
    return "\n".join(lines) + "\n"


@asynccontextmanager
async def serialized_gpu_inference(modality: str = "general"):
    """
    Acquire hardware serialization lock for safe single-tenant GPU inference.
    Guarantees only one model forward pass occupies GPU VRAM at any time.
    Automatically checks headroom before, and evicts cache on exit.

    Phase 3: acquisition is bounded by GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS so a
    stalled/half-open client cannot deadlock all other modalities.
    """
    start = time.perf_counter()
    acquired = False
    try:
        try:
            acquired = await asyncio.wait_for(gpu_inference_lock.acquire(), timeout=GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            metrics["gpu_lock_timeouts_total"] += 1
            logger.error(
                "gpu_lock_acquire_timeout",
                extra={"modality": modality, "timeout_seconds": GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS},
            )
            raise GPUInferenceTimeoutError(
                f"GPU inference lock not acquired within {GPU_LOCK_ACQUIRE_TIMEOUT_SECONDS}s (modality={modality})"
            ) from None

        global _active_gpu_modality, _last_active_modality
        prev_modality = _last_active_modality
        needs_evict = False
        if prev_modality is not None and prev_modality != modality:
            needs_evict = True
        elif prev_modality is not None:
            try:
                import torch
                if torch.cuda.is_available() and torch.cuda.memory_allocated() > int(VRAM_MAX_CEILING_BYTES * 0.78):
                    needs_evict = True
            except Exception:
                pass

        if needs_evict:
            logger.info(
                "cross_modality_vram_auto_evict",
                extra={"prev_modality": prev_modality, "new_modality": modality},
            )
            release_all_model_singletons()
        _active_gpu_modality = modality
        _last_active_modality = modality

        check_vram_headroom()
        failed = False
        try:
            yield
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                evict_gpu_models()
            record_inference(modality, time.perf_counter() - start, failed=failed)
    finally:
        if acquired:
            gpu_inference_lock.release()


def get_active_device() -> str:
    """Detect available accelerator. Prefers CUDA if initialized."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        logger.warning("device_detection_failed", error=str(exc), exc_type=type(exc).__name__)
    return "cpu"


def get_vram_stats() -> dict[str, Any]:
    """Report live CUDA VRAM allocations in MB."""
    stats = {"device": get_active_device(), "cuda_available": False, "allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            stats["cuda_available"] = True
            stats["device_name"] = torch.cuda.get_device_name(0)
            stats["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            stats["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
            stats["peak_mb"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    except Exception:
        pass
    return stats


def evict_gpu_models(*models: Any) -> None:
    """
    Explicitly frees model weights and clears GPU context.
    ponytail: eliminates model thrashing across modalities on 6GB VRAM.
    """
    for m in models:
        if m is not None:
            del m
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# Every lazily-cached model singleton in the app, and the call that releases it.
# Kept as (module, attribute, action) so a new modality only has to be added here.
# action="cache_clear" -> functools.lru_cache holder; action="call" -> evict_*() helper.
_MODEL_SINGLETON_RELEASERS: tuple[tuple[str, str, str], ...] = (
    ("app.ml.text_detector", "_load_model", "cache_clear"),
    ("app.ml.text_detector", "_load_fine_tuned_model", "cache_clear"),
    ("app.ml.binoculars", "evict_binoculars_models", "call"),
    ("app.ml.perplexity", "evict_perplexity_model", "call"),
    ("app.ml.reconstruction_detector", "_unload_recon_model", "call"),
    ("app.ml.image_detector", "_load_image_model", "cache_clear"),
    ("app.ml.clip_probe", "evict_clip_models", "call"),
    ("app.ml.video_detector", "_load_video_deepfake_model", "cache_clear"),
    ("app.ml.audio_detector", "evict_audio_model", "call"),
)


def release_all_model_singletons() -> dict[str, bool]:
    """Drop every cross-modality model singleton, then reclaim the GPU cache.

    PRODUCTION CONTRACT
    -------------------
    A served request runs inside `serialized_gpu_inference`, which evicts that
    modality's weights as soon as the response is produced. Two modalities
    therefore never hold VRAM at the same time on the 6 GB RTX 4050; residency
    is deliberately scoped to one in-flight request.

    WHY THIS EXISTS OUTSIDE PRODUCTION
    ----------------------------------
    In-process test ordering does not honour that contract. A text test parks
    the RoBERTa / Qwen / Binoculars weights on the GPU, and a later video test
    then measures its own footprint *plus* those parked weights — inflating
    `torch.cuda.memory_allocated()` past the 5 GB ceiling even though the video
    path alone needs only ~1.6 GB. This function replays the governor's eviction
    discipline across all modalities so a VRAM measurement starts from the same
    clean baseline production guarantees, making it order-independent rather
    than a function of which test happened to run first.

    Returns a mapping of "module.attribute" -> released? so callers can log or
    assert on the outcome. Never raises: a modality that failed to import or
    whose holder is already gone is reported as False and the rest proceed.
    """
    import importlib

    global _active_gpu_modality, _last_active_modality
    _active_gpu_modality = None
    _last_active_modality = None

    released: dict[str, bool] = {}
    for module_name, attr_name, action in _MODEL_SINGLETON_RELEASERS:
        key = f"{module_name}.{attr_name}"
        try:
            module = importlib.import_module(module_name)
            target = getattr(module, attr_name)
            if action == "cache_clear":
                target.cache_clear()
            else:
                target()
            released[key] = True
        except Exception as exc:
            released[key] = False
            logger.warning(
                "model_singleton_release_failed",
                extra={"target": key, "error": str(exc), "exc_type": type(exc).__name__},
            )

    # Collect first (frees the now-unreferenced tensors), then hand the emptied
    # caching allocator blocks back to the driver.
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    return released


def check_vram_headroom() -> bool:
    """Returns True if VRAM usage is safely under 5.0 GB ceiling."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            if allocated >= VRAM_MAX_CEILING_BYTES:
                logger.warning(f"VRAM near ceiling ({allocated / 1024**3:.2f} GB). Triggering eviction.")
                torch.cuda.empty_cache()
                return False
    except Exception:
        pass
    return True
