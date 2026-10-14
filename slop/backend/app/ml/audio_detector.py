"""
SLOP Audio Detector — Wav2Vec2-based speech deepfake detection (Phase D11).

Model: fine-tuned Wav2Vec2 sequence classifier (deepfake vs real speech,
trained on ASVspoof-style data). Runs under the VRAM governor discipline:
  - model is lazily loaded once and shared process-wide
  - `evict_audio_model()` frees VRAM so other modalities can run
  - callers wrap inference in `serialized_gpu_inference("audio")`

Signal path:
  1. Load audio via torchaudio/soundfile (resampled to 16 kHz mono).
  2. Chunked Wav2Vec2 forward passes (10 s windows, 50% overlap).
  3. Frame-probability aggregation: max (attacker can splice real+fake),
     plus mean for calibration context, plus spectral-flatness-style
     heuristics that run CPU-only and work even without the model.

Feature-gated: `settings.enable_audio_detector` (default False) so it can
never silently influence scores before validation — same policy as the
reconstruction and frequency detectors.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import settings
from app.ml.fusion import fuse_audio_score
from app.ml.vram_manager import get_vram_stats

logger = logging.getLogger(__name__)

_base = Path(__file__).resolve()
# Binoculars-style fallback chain: layout differs between dev and installed layouts
MODEL_CANDIDATES = [
    _base.parents[4] / "Model" / "audio" / "wav2vec2_deepfake",
    _base.parents[3] / "Model" / "audio" / "wav2vec2_deepfake",
]
MODEL_DIR = next((p for p in MODEL_CANDIDATES if p.exists()), MODEL_CANDIDATES[-1])
SAMPLE_RATE = 16_000
WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0  # 50% overlap

_audio_model: dict[str, Any] = {}
_spectral_bundle: dict[str, Any] = {}


def _load_torchaudio() -> Any:
    return importlib.import_module("torchaudio")


def _load_soundfile() -> Any:
    return importlib.import_module("soundfile")


def _load_torch() -> Any:
    return importlib.import_module("torch")


def evict_audio_model() -> None:
    """Free VRAM/CPU cache held by the audio model (VRAM governor hook)."""
    _audio_model.clear()
    if _spectral_bundle:
        try:
            import gc

            gc.collect()
        except Exception:
            pass


def _get_model() -> dict[str, Any] | None:
    """Lazily load the Wav2Vec2 classifier; returns None when unavailable."""
    if _audio_model:
        return _audio_model
    try:
        torch = _load_torch()
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

        if not MODEL_DIR.exists():
            logger.warning("audio_model_missing", extra={"path": str(MODEL_DIR)})
            return None
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(MODEL_DIR))
        model = Wav2Vec2ForSequenceClassification.from_pretrained(str(MODEL_DIR))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        _audio_model.update(model=model, extractor=extractor, device=device, torch=torch)
        logger.info("audio_model_loaded", extra={"device": device, "path": str(MODEL_DIR)})
        return _audio_model
    except Exception as exc:
        logger.error("audio_model_load_failed", extra={"error": str(exc)[:200]})
        return None


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at SAMPLE_RATE; raises on failure."""
    torchaudio = _load_torchaudio()
    wav, sr = torchaudio.load(file_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.squeeze(0).numpy(), SAMPLE_RATE


def _chunks(wav: np.ndarray, sr: int) -> list[np.ndarray]:
    window = int(WINDOW_SECONDS * sr)
    hop = int(HOP_SECONDS * sr)
    if len(wav) <= window:
        return [wav]
    out = []
    for start in range(0, len(wav) - window + 1, hop):
        out.append(wav[start : start + window])
    return out


def _model_probabilities(wav: np.ndarray, sr: int) -> list[float]:
    """Per-chunk P(fake) from the Wav2Vec2 classifier; [] when model unavailable."""
    bundle = _get_model()
    if not bundle:
        return []
    model, extractor, device, torch = bundle["model"], bundle["extractor"], bundle["device"], bundle["torch"]
    probs: list[float] = []
    with torch.no_grad():
        for chunk in _chunks(wav, sr):
            inputs = extractor(chunk, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits[0]
            # id2label: 0=fake, 1=real → P(fake) = softmax(logits)[0]
            p = torch.softmax(logits.float(), dim=-1)
            probs.append(float(p[0].item()))
    return probs


def _spectral_features(wav: np.ndarray, sr: int) -> dict[str, float]:
    """CPU-only spectral heuristics: vocoder artifacts concentrate energy in
    narrow synthesis bands; natural speech has flatter broadband dynamics."""
    spec = np.abs(np.fft.rfft(wav[: sr * 30]))  # first 30 s max
    freqs = np.fft.rfftfreq(len(wav[: sr * 30]), 1 / sr)
    power = spec**2 + 1e-12
    # Spectral flatness (geometric mean / arithmetic mean): 0=tonal, 1=noise
    flatness = float(np.exp(np.log(power).mean()) / power.mean())
    # Energy ratio in vocoder band 4-8 kHz relative to total (synthesis artifact zone)
    band = (freqs >= 4000) & (freqs <= 8000)
    vocoder_ratio = float(power[band].sum() / power.sum())
    # Zero-crossing irregularity: jitter of inter-zero-crossing intervals
    signs = np.signbit(wav)
    crossings = np.where(signs[:-1] != signs[1:])[0]
    if len(crossings) > 2:
        intervals = np.diff(crossings).astype(float)
        jitter = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
    else:
        jitter = 0.0
    return {
        "spectral_flatness": round(flatness, 6),
        "vocoder_band_ratio": round(vocoder_ratio, 6),
        "zcr_jitter": round(jitter, 6),
    }


def analyze_audio(file_path: str, job_id: str) -> dict:
    """
    Full audio analysis. Score semantics match other modalities:
    overall_score ~ P(AI-generated/deepfake), 0..1.
    """
    stats_before = get_vram_stats()
    wav, sr = load_audio(file_path)
    duration = len(wav) / sr

    neural_probs = _model_probabilities(wav, sr)
    spectral = _spectral_features(wav, sr)

    if neural_probs:
        neural_fake = float(max(neural_probs))  # max: spliced attacks keep their fake chunk
        neural_mean = float(np.mean(neural_probs))
    else:
        # Model unavailable: fall back to spectral heuristic only, low confidence
        neural_fake, neural_mean = None, None

    # Heuristic score: vocoder-band excess and low jitter push toward fake
    h_score = 0.5 + 0.6 * (spectral["vocoder_band_ratio"] - 0.12) - 0.3 * (spectral["zcr_jitter"] - 0.5)
    heuristic_score = float(np.clip(h_score, 0.0, 1.0))

    if neural_fake is not None:
        fused = fuse_audio_score({"neural_fake": neural_fake, "heuristic_score": heuristic_score})
        overall, confidence = fused["overall_score"], fused["confidence"]
    else:
        fused = fuse_audio_score({"neural_fake": None, "heuristic_score": heuristic_score})
        overall, confidence = fused["overall_score"], fused["confidence"]

    if overall >= 0.5:
        verdict = "likely_ai" if overall >= 0.75 else "uncertain"
    else:
        verdict = "likely_human" if overall <= 0.25 else "uncertain"

    return {
        "job_id": job_id,
        "overall_score": round(float(overall), 4),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "signals": {
            "neural_fake_prob_max": None if neural_fake is None else round(neural_fake, 4),
            "neural_fake_prob_mean": None if neural_mean is None else round(neural_mean, 4),
            "neural_chunks": len(neural_probs),
            "heuristic_score": round(heuristic_score, 4),
            **spectral,
        },
        "duration_seconds": round(duration, 2),
        "vram": {
            "used_mb": stats_before.get("used_mb") if isinstance(stats_before, dict) else None,
        },
    }
