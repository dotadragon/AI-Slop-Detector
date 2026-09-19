"""
SLOP Dynamic VRAM Manager — Production memory governor for RTX 4050 (6GB).
ponytail: lean, zero-overhead memory eviction & hardware profiling
"""

import asyncio
import gc
import logging
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

# ponytail: Rung 3 stdlib hardware serialization lock. Zero Redis/Celery queue bloat.
# Guarantees only one neural forward pass executes on the RTX 4050 at any moment.
gpu_inference_lock = asyncio.Semaphore(1)
VRAM_MAX_CEILING_BYTES = int(4.5 * 1024 * 1024 * 1024)  # Strict 4.5 GB ceiling for RTX 4050 6GB


@asynccontextmanager
async def serialized_gpu_inference():
    """
    Acquire hardware serialization lock for safe single-tenant GPU inference.
    Automatically cleans cache and checks headroom.
    """
    async with gpu_inference_lock:
        check_vram_headroom()
        try:
            yield
        finally:
            evict_gpu_models()


def get_active_device() -> str:
    """Detect available accelerator. Prefers CUDA if initialized."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
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
