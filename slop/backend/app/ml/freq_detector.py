"""Frequency-domain image forgery signal (F3Net/FreqNet-inspired, Tier 5).

CPU-only spectral feature detector inspired by F3Net's frequency-aware
decomposition. Computes statistics over the 2-D DCT/FFT spectrum that
distinguish GAN/diffusion spectra (spectral peaks, up-sampling artifacts at
half-frequencies) from natural camera images. This is a *scaffold*: it is
gated behind `enable_freq_detector` (default OFF) and must be benchmarked on
real held-out data before its output is allowed to influence production
scores — same discipline as the reconstruction detector.

Outputs `freq_ai_score` in [0, 1]; higher = more AI-like spectrum.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_gray_array(img: Image.Image, size: int = 256) -> np.ndarray:
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.float32) / 255.0


def _radial_profile(magnitude: np.ndarray, bins: int = 32) -> np.ndarray:
    """Azimuthally averaged power spectrum (1-D radial profile)."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r = r.astype(int)
    nbins = min(bins, r.max() + 1)
    profile = np.zeros(nbins)
    for i in range(nbins):
        mask = r == i
        if mask.any():
            profile[i] = magnitude[mask].mean()
    return profile


def _spectral_peak_ratio(magnitude: np.ndarray) -> float:
    """Fraction of spectral energy in isolated peaks — GAN up-sampling leaves
    sharp harmonic spikes; natural photos have smooth spectra."""
    m = magnitude.copy()
    m[0, 0] = 0.0  # drop DC
    total = float(m.sum())
    if total <= 0:
        return 0.0
    # top 0.1% of bins vs total energy
    thresh = np.quantile(m, 0.999)
    return float(min(1.0, m[m >= thresh].sum() / total))


def _high_freq_energy_ratio(magnitude: np.ndarray) -> float:
    """Energy beyond 3/4 of Nyquist. Diffusion up-samplers often leave more or
    less high-frequency energy than camera optics; direction-free but useful."""
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / max(cy, cx, 1)
    total = float(magnitude.sum())
    if total <= 0:
        return 0.0
    return float(magnitude[r > 0.75].sum() / total)


def _profile_roughness(profile: np.ndarray) -> float:
    """Residual roughness of the log radial profile after removing its smooth
    trend — grid artifacts show up as periodic bumps."""
    logp = np.log1p(profile)
    # remove linear trend
    n = len(logp)
    t = np.arange(n)
    if n < 3:
        return 0.0
    slope, intercept = np.polyfit(t, logp, 1)
    resid = logp - (slope * t + intercept)
    return float(np.sqrt(np.mean(resid**2)))


def compute_freq_score(img: Image.Image) -> dict:
    """CPU-only frequency-domain AI-likeness score in [0, 1]."""
    arr = _to_gray_array(img)
    spectrum = np.fft.fftshift(np.fft.fft2(arr))
    magnitude = np.log1p(np.abs(spectrum))

    peak_ratio = _spectral_peak_ratio(np.abs(spectrum))
    hf_ratio = _high_freq_energy_ratio(np.abs(spectrum))
    roughness = _profile_roughness(_radial_profile(magnitude))

    # Heuristic fusion of raw spectral features into a 0-1 AI-likeness.
    # NOTE: uncalibrated scaffold — thresholds are placeholders pending the
    # benchmark on real AI/human image data. Do not promote without eval.
    raw = (
        0.40 * min(1.0, peak_ratio / 0.02)
        + 0.30 * min(1.0, roughness / 0.35)
        + 0.30 * min(1.0, hf_ratio / 0.05)
    )
    return {
        "freq_ai_score": round(float(min(1.0, max(0.0, raw))), 4),
        "freq_engine": "spectral_proxy",
        "spectral_features": {
            "peak_ratio": round(peak_ratio, 6),
            "high_freq_ratio": round(hf_ratio, 6),
            "profile_roughness": round(roughness, 6),
        },
    }
