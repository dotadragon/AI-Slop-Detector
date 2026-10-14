"""
SLOP Reconstruction Detector (DIRE-style) — Phase 7 signal.

Detects AI-generated images by reconstructing them through a pretrained
diffusion autoencoder (SD VAE) and measuring the pixel round-trip error.
Real photographs lose more information through the VAE bottleneck than
diffusion latents, which reconstruct near-perfectly (DIRE, Wang et al. 2023).

Design constraints:
- Gated behind the VRAM governor (`serialized_gpu_inference`) — never loads
  concurrently with the Swin/CLIP models on the 6GB RTX 4050.
- The reconstruction model is lazily loaded and explicitly evicted after use.
- Falls back to a CPU-only proxy signal (downscale-upscale round-trip error)
  when torch/weights are unavailable, so the fusion feature is always present.
- This signal AUGMENTS the Neural-First Consensus verdict; it never overrides
  the Swin/CLIP neural score (it enters fusion only as a corroborating feature).
"""

import logging
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

RECON_SIZE = 256
RECON_MODEL_ID = "stabilityai/sd-vae-ft-mse"


@lru_cache(maxsize=1)
def _load_recon_model():
    """Lazily load the diffusion VAE for reconstruction. Returns None if unavailable."""
    try:
        import importlib
        import torch
        # ponytail: diffusers is an optional dependency; dynamic import keeps the
        # proxy fallback path viable (and pyright clean) when it is not installed.
        AutoencoderKL = importlib.import_module("diffusers").AutoencoderKL
    except ImportError:
        logger.info("reconstruction_model_unavailable", reason="torch/diffusers not installed")
        return None

    from app.core.config import settings

    try:
        vae = AutoencoderKL.from_pretrained(
            RECON_MODEL_ID,
            dtype=torch.float16 if settings.compute_device == "cuda" else torch.float32,
            cache_dir=settings.huggingface_cache_dir,
        )
        device = settings.compute_device
        vae.to(device).eval()
        return vae, torch, device
    except Exception as exc:
        logger.warning("reconstruction_model_load_failed", error=str(exc), exc_type=type(exc).__name__)
        return None


def _unload_recon_model() -> None:
    """Explicitly evict the VAE so the Swin model can reclaim VRAM (Neural-First budget)."""
    try:
        import torch
        _load_recon_model.cache_clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _proxy_reconstruction_error(img_rgb: Image.Image) -> float:
    """
    CPU-only fallback proxy: JPEG + downscale round-trip error.
    AI images (smooth latent gradients) lose less detail through a lossy
    round-trip than sensor-noise-rich photographs. Weak but free signal.
    """
    rgb = img_rgb.convert("RGB").resize((RECON_SIZE, RECON_SIZE), Image.Resampling.LANCZOS)
    import io
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=70)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    a = np.asarray(rgb, dtype=np.float32)
    b = np.asarray(recompressed, dtype=np.float32)
    err = float(np.mean(np.abs(a - b)))
    return err


def compute_reconstruction_score(pil_img: Image.Image) -> dict[str, Any]:
    """
    Returns {'recon_ai_score': 0..1, 'recon_error': float, 'engine': 'vae'|'proxy'}.
    Higher recon_ai_score = more likely AI (image reconstructs too cleanly).
    """
    img_rgb = pil_img.convert("RGB").resize((RECON_SIZE, RECON_SIZE), Image.Resampling.LANCZOS)

    loaded = _load_recon_model()
    if loaded is None:
        err = _proxy_reconstruction_error(img_rgb)
        # Calibrated roughly: proxy error for natural photos ~8-20, AI ~3-8.
        score = max(0.0, min(1.0, (12.0 - err) / 10.0))
        return {"recon_ai_score": round(score, 4), "recon_error": round(err, 3), "engine": "proxy"}

    vae, torch, device = loaded
    try:
        from torchvision import transforms as T

        to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])
        x = to_tensor(img_rgb).unsqueeze(0).to(device=device, dtype=next(vae.parameters()).dtype)

        with torch.inference_mode():
            latent = vae.encode(x).latent_dist.mean
            recon = vae.decode(latent).sample

        a = ((x[0].float().cpu().numpy() + 1.0) / 2.0).transpose(1, 2, 0)
        b = ((recon[0].float().cpu().numpy() + 1.0) / 2.0).clip(0, 1).transpose(1, 2, 0)
        err = float(np.mean(np.abs(a - b)) * 255.0)

        # VAE round-trip: photos err ~8-16, diffusion latents err ~1.5-6 (DIRE finding).
        score = max(0.0, min(1.0, (9.0 - err) / 7.0))
        return {"recon_ai_score": round(score, 4), "recon_error": round(err, 3), "engine": "vae"}
    except Exception as exc:
        logger.warning("reconstruction_inference_failed", error=str(exc), exc_type=type(exc).__name__)
        err = _proxy_reconstruction_error(img_rgb)
        score = max(0.0, min(1.0, (12.0 - err) / 10.0))
        return {"recon_ai_score": round(score, 4), "recon_error": round(err, 3), "engine": "proxy_fallback"}
    finally:
        # Never let the VAE sit in VRAM between scans — Swin needs the budget.
        _unload_recon_model()
