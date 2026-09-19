"""
SLOP Neighboring Pixel Relationships (NPR) Extractor.
Reference: Chen et al. (2024) — Neighboring Pixel Relationships for Generative Image Forensics.

Real optical sensors produce natural neighboring pixel relationships dictated by optics,
Poisson-Gaussian photon noise, and Bayer Color Filter Array (CFA) demosaicing.
Generative AI diffusion models (SDXL, Midjourney, Flux) and GANs operate in continuous
latent spaces followed by VAE decoding, creating distinct artifacts in local pixel co-occurrence
and directional residual distributions.
"""

import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew


def compute_npr_features(image: Image.Image) -> dict[str, float]:
    """
    Extracts neighboring pixel relationship statistics across directional gradients
    and color channel couplings.
    """
    rgb = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    
    # 1. Directional neighboring differences
    # Horizontal difference: I(x+1, y) - I(x, y)
    dh = rgb[:, 1:, :] - rgb[:, :-1, :]
    # Vertical difference: I(x, y+1) - I(x, y)
    dv = rgb[1:, :, :] - rgb[:-1, :, :]
    # Diagonal difference: I(x+1, y+1) - I(x, y)
    dd = rgb[1:, 1:, :] - rgb[:-1, :-1, :]

    # 2. Residual moments across channels
    def get_moments(diff_arr):
        std_val = float(np.std(diff_arr))
        sk_val = float(np.mean(skew(diff_arr.reshape(-1, 3), axis=0)))
        kt_val = float(np.mean(kurtosis(diff_arr.reshape(-1, 3), axis=0)))
        return std_val, sk_val, kt_val

    std_h, sk_h, kt_h = get_moments(dh)
    std_v, sk_v, kt_v = get_moments(dv)
    std_d, sk_d, kt_d = get_moments(dd)

    # 3. Cross-channel neighbor correlation
    # In real cameras, demosaicing strongly couples G with R and B neighbors.
    # Generative VAE decoders exhibit decoupled or uniform channel correlations.
    r_diff = dh[:, :, 0].ravel()
    g_diff = dh[:, :, 1].ravel()
    b_diff = dh[:, :, 2].ravel()

    rg_corr = float(np.corrcoef(r_diff, g_diff)[0, 1]) if np.std(r_diff) > 1e-5 and np.std(g_diff) > 1e-5 else 0.5
    gb_corr = float(np.corrcoef(g_diff, b_diff)[0, 1]) if np.std(g_diff) > 1e-5 and np.std(b_diff) > 1e-5 else 0.5

    # 4. Anisotropy: ratio of horizontal/vertical vs diagonal residual energies
    # Real camera lenses and natural scenes exhibit smooth directional transitions.
    # VAE patch-based decoders exhibit anisotropic grid energy.
    hv_energy = (np.mean(dh**2) + np.mean(dv**2)) / 2.0
    diag_energy = np.mean(dd**2)
    anisotropy_ratio = float(hv_energy / max(diag_energy, 1e-5))

    # Mean kurtosis across all directions (leptokurtic distribution in real photos vs lighter tails in AI)
    mean_kurtosis = float((kt_h + kt_v + kt_d) / 3.0)

    # 5. Composite NPR AI Score
    # AI images generated via diffusion VAEs typically exhibit:
    # - Abnormally low kurtosis in residuals compared to natural scene edges (< 15.0)
    # - Higher anisotropy ratio (grid bias > 1.4 or < 0.6)
    # - Symmetrical or atypical inter-channel correlation
    ai_evidence = 0.0

    if mean_kurtosis < 12.0:
        ai_evidence += 0.40 * min(1.0, (12.0 - mean_kurtosis) / 8.0)
    elif mean_kurtosis > 28.0:
        # Extreme kurtosis from high-frequency synthetic texture injection
        ai_evidence += 0.25

    if abs(anisotropy_ratio - 1.0) > 0.35:
        ai_evidence += 0.30 * min(1.0, (abs(anisotropy_ratio - 1.0) - 0.35) / 0.4)

    # Cross-channel coupling divergence
    if abs(rg_corr - gb_corr) > 0.25:
        ai_evidence += 0.20

    npr_score = float(np.clip(ai_evidence + 0.10, 0.0, 1.0))

    return {
        "npr_score": round(npr_score, 3),
        "mean_kurtosis": round(mean_kurtosis, 2),
        "anisotropy_ratio": round(anisotropy_ratio, 3),
        "rg_correlation": round(rg_corr, 3),
        "gb_correlation": round(gb_corr, 3),
        "residual_std": round(float((std_h + std_v + std_d) / 3.0), 2),
    }
