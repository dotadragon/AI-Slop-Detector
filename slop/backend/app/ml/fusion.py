"""
SLOP Forensic Fusion Engine v1.0 — Single Score Authority
# ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING eval_accuracy.py AND HUMAN APPROVAL
Complies with Rule 2, Rule 3, and Rule 5 of ANTIGRAVITY_RULES.md.

This module is the EXCLUSIVE component authorized to compute final multi-modal
AI scores. No detector file (image_detector.py, text_detector.py, video_detector.py)
may contain hard clamps (e.g. overall = <constant>) or bypass logic outside of this engine.
"""

import math
import pickle
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FUSION_MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "Model" / "fusion"


def _clamp01(val: float) -> float:
    return max(0.0, min(1.0, float(val)))


def _sigmoid(x: float) -> float:
    if x < -30.0:
        return 0.0
    if x > 30.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _confidence_band(score: float) -> str:
    if score >= 0.85:
        return "DEFINITELY_AI"
    elif score >= 0.60:
        return "LIKELY_AI"
    elif score >= 0.40:
        return "UNCERTAIN"
    elif score >= 0.15:
        return "LIKELY_HUMAN"
    else:
        return "DEFINITELY_HUMAN"


def _confidence_range(score: float, disagreement: float = 0.0) -> list[float]:
    margin = 0.04 + min(0.10, max(0.0, float(disagreement)) * 0.20)
    return [round(max(0.0, score - margin), 3), round(min(1.0, score + margin), 3)]


def _find_fusion_model_path(modality: str) -> Path | None:
    base = Path(__file__).resolve()
    for p in [
        base.parents[4] / "Model" / "fusion" / f"{modality}.pkl",
        base.parents[3] / "Model" / "fusion" / f"{modality}.pkl",
        base.parents[2] / "Model" / "fusion" / f"{modality}.pkl",
    ]:
        if p.exists():
            return p
    return None


def _load_model_weights(modality: str) -> dict[str, Any] | None:
    pkl_path = _find_fusion_model_path(modality)
    if pkl_path and pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "weights" in data:
                    return data
        except Exception as e:
            logger.warning(f"Failed to load fusion model for {modality} from {pkl_path}: {e}")
    return None


# ============================================================================
# IMAGE FUSION ENGINE
# ============================================================================

def fuse_image_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines image forensic features into a calibrated final AI probability.
    Single authority for image scoring. No hard clamps allowed outside this function.
    """
    model_data = _load_model_weights("image")

    best_neural = features.get("best_neural")
    heuristic = float(features.get("heuristic_score", 0.0))
    cfa = float(features.get("cfa_score", 0.0))
    ela = float(features.get("ela_score", 0.0))
    fft = float(features.get("fft_score", 0.0))
    noise = float(features.get("noise_score", 0.0))
    dct = float(features.get("dct_score", 0.0))
    meta_risk = float(features.get("metadata_risk", 0.0))
    is_diagram = float(features.get("is_diagram", 0.0))
    is_real_cam = float(features.get("is_real_camera", 0.0))
    is_anime = float(features.get("is_anime_ai", 0.0))
    sig_detected = float(features.get("signature_detected", 0.0))
    sig_conf = float(features.get("signature_confidence", 0.0))
    c2pa_verified = float(features.get("c2pa_verified", 0.0))
    clip_probe = features.get("clip_probe_score")
    npr = features.get("npr_score")

    if model_data is not None:
        # Logistic regression inference
        w = model_data.get("weights", {})
        intercept = float(model_data.get("intercept", 0.0))
        logit = intercept
        contributions = {}
        feature_tuples = [
            ("best_neural", best_neural if best_neural is not None else 0.5),
            ("heuristic_score", heuristic),
            ("cfa_score", cfa),
            ("ela_score", ela),
            ("fft_score", fft),
            ("noise_score", noise),
            ("dct_score", dct),
            ("metadata_risk", meta_risk),
            ("is_diagram", is_diagram),
            ("is_real_camera", is_real_cam),
            ("is_anime_ai", is_anime),
            ("signature_detected", sig_detected),
            ("signature_confidence", sig_conf),
            ("c2pa_verified", c2pa_verified),
        ]
        if clip_probe is None and "clip_probe_score" in w:
            clip_probe = best_neural if best_neural is not None else heuristic
        if npr is None and "npr_score" in w:
            npr = best_neural if best_neural is not None else heuristic
        if clip_probe is not None and "clip_probe_score" in w:
            feature_tuples.append(("clip_probe_score", float(clip_probe)))
        if npr is not None and "npr_score" in w:
            feature_tuples.append(("npr_score", float(npr)))

        for feat_name, feat_val in feature_tuples:
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        contributions = {}
        if best_neural is not None:
            bn = float(best_neural)
            contributions["best_neural"] = bn

            disagreement = abs(heuristic - bn)
            # Consensus logic: Neural model holds primary authority (80%+)
            if bn >= 0.70:
                score = bn * 0.90 + heuristic * 0.10
            elif bn < 0.35:
                # Authentic photo or diagram suppression
                if is_diagram > 0.5 or is_real_cam > 0.5:
                    score = min(bn, 0.05)
                elif heuristic < 0.25:
                    score = 0.80 * bn + 0.20 * heuristic
                else:
                    score = 0.50 * bn + 0.50 * heuristic
            else:
                score = 0.70 * bn + 0.30 * heuristic
        else:
            score = heuristic

        # Digital signature corroboration
        if sig_detected > 0.5 and sig_conf > 0.60:
            score = max(score, sig_conf)

        # C2PA provenance authentication
        if c2pa_verified > 0.5:
            score *= 0.15

    final_score = round(_clamp01(score), 3)
    disagreement = abs(float(best_neural if best_neural is not None else 0.5) - heuristic)

    return {
        "overall_score": final_score,
        "confidence_band": _confidence_band(final_score),
        "confidence_range": _confidence_range(final_score, disagreement),
        "primary_signal": "neural_vision_transformer" if (best_neural is not None and best_neural >= 0.50) else "forensic_heuristics",
        "contributions": contributions,
    }


# ============================================================================
# TEXT FUSION ENGINE
# ============================================================================

def fuse_text_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines text forensic features into a calibrated final AI probability.
    Single authority for text scoring. No hard clamps allowed outside this function.
    """
    model_data = _load_model_weights("text")

    neural_prob = features.get("neural_prob")
    heuristic = float(features.get("heuristic_score", 0.0))
    llm_pattern = float(features.get("llm_pattern_score", 0.0))
    entropy = float(features.get("transitional_entropy", 0.5))
    burstiness = float(features.get("burstiness", 0.5))
    contraction = float(features.get("contraction_score", 0.5))
    zipf = float(features.get("zipf_deviation", 0.5))
    vocab = float(features.get("vocab_richness", 0.5))
    char_count = int(features.get("char_count", 500))
    bino_prob = features.get("binoculars_ai_prob")
    fast_prob = features.get("fast_detectgpt_ai_prob")

    if model_data is not None:
        w = model_data.get("weights", {})
        intercept = float(model_data.get("intercept", 0.0))
        logit = intercept
        contributions = {}
        feature_tuples = [
            ("neural_prob", neural_prob if neural_prob is not None else 0.5),
            ("heuristic_score", heuristic),
            ("llm_pattern_score", llm_pattern),
            ("transitional_entropy", entropy),
            ("burstiness", burstiness),
            ("contraction_score", contraction),
            ("zipf_deviation", zipf),
            ("vocab_richness", vocab),
        ]
        if bino_prob is None and "binoculars_ai_prob" in w:
            bino_prob = neural_prob if neural_prob is not None else heuristic
        if fast_prob is None and "fast_detectgpt_ai_prob" in w:
            fast_prob = neural_prob if neural_prob is not None else heuristic
        if bino_prob is not None and "binoculars_ai_prob" in w:
            feature_tuples.append(("binoculars_ai_prob", float(bino_prob)))
        if fast_prob is not None and "fast_detectgpt_ai_prob" in w:
            feature_tuples.append(("fast_detectgpt_ai_prob", float(fast_prob)))

        for feat_name, feat_val in feature_tuples:
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        contributions = {}
        if neural_prob is not None:
            np_val = float(neural_prob)
            contributions["neural_prob"] = np_val

            disagreement = abs(heuristic - np_val)

            # Case 1: Frontier LLM signature (heuristic/pattern high, RoBERTa low)
            if (heuristic > 0.55 and np_val < 0.35) or (llm_pattern >= 0.60 and np_val < 0.45):
                if llm_pattern >= 0.70:
                    score = max(llm_pattern, heuristic * 0.94)
                elif heuristic > 0.75:
                    score = heuristic * 0.90
                else:
                    score = 0.20 * np_val + 0.80 * heuristic
            # Case 2: Neural model detects AI with high confidence
            elif np_val > 0.65 and heuristic < 0.25:
                score = max(0.85, np_val * 0.95)
            # Case 3: Agreement
            elif disagreement < 0.20:
                score = 0.80 * np_val + 0.20 * heuristic
            # Case 4: Moderate disagreement
            else:
                if np_val >= 0.70:
                    score = 0.80 * np_val + 0.20 * heuristic
                elif np_val <= 0.25:
                    score = 0.75 * np_val + 0.25 * min(heuristic, 0.35)
                else:
                    score = 0.70 * np_val + 0.30 * heuristic

            if llm_pattern >= 0.80:
                score = max(score, llm_pattern * 0.95)
        else:
            score = heuristic

    final_score = round(_clamp01(score), 3)
    disagreement = abs(heuristic - float(neural_prob if neural_prob is not None else 0.5))

    return {
        "overall_score": final_score,
        "confidence_band": _confidence_band(final_score),
        "confidence_range": _confidence_range(final_score, disagreement),
        "primary_signal": "roberta_neural_classifier" if (neural_prob is not None and neural_prob >= 0.50) else "stylometric_forensics",
        "contributions": contributions,
    }


# ============================================================================
# VIDEO FUSION ENGINE
# ============================================================================

def fuse_video_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines video frame and temporal forensic features into a calibrated final AI probability.
    Single authority for video scoring. No hard clamps allowed outside this function.
    """
    model_data = _load_model_weights("video")

    peak_threat = float(features.get("peak_threat", 0.5))
    top_quartile = float(features.get("top_quartile_mean", 0.5))
    global_mean = float(features.get("global_mean", 0.5))
    is_screen_rec = float(features.get("is_screen_rec", 0.0))
    has_deepfake = float(features.get("has_deepfake_face", 0.0))
    max_face = float(features.get("max_face_score", 0.0))
    flicker = float(features.get("temporal_flicker", 0.0))
    jitter = float(features.get("temporal_jitter", 0.0))
    synthid = float(features.get("synthid_detected", 0.0))
    high_threat_count = int(features.get("high_threat_frame_count", 0))

    if model_data is not None:
        w = model_data.get("weights", {})
        intercept = float(model_data.get("intercept", 0.0))
        logit = intercept
        contributions = {}
        for feat_name, feat_val in [
            ("peak_threat", peak_threat),
            ("top_quartile_mean", top_quartile),
            ("global_mean", global_mean),
            ("is_screen_rec", is_screen_rec),
            ("has_deepfake_face", has_deepfake),
            ("max_face_score", max_face),
            ("temporal_flicker", flicker),
            ("temporal_jitter", jitter),
            ("synthid_detected", synthid),
        ]:
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        contributions = {}
        # 40% Peak + 40% Top-Quartile + 20% Global Mean
        score = 0.40 * peak_threat + 0.40 * top_quartile + 0.20 * global_mean

        if synthid > 0.5 and is_screen_rec <= 0.5:
            score = max(score, 0.85)
        elif peak_threat >= 0.80 and high_threat_count >= 2 and is_screen_rec <= 0.5:
            score = max(score, 0.80)
        elif is_screen_rec > 0.5:
            if has_deepfake <= 0.5:
                # Authentic human software UI screen recording
                score = min(score, 0.05)

    final_score = round(_clamp01(score), 3)

    return {
        "overall_score": final_score,
        "confidence_band": _confidence_band(final_score),
        "confidence_range": _confidence_range(final_score, abs(peak_threat - global_mean)),
        "primary_signal": "deepfake_face_vit" if has_deepfake > 0.5 else ("swin_generative_diffusion" if final_score >= 0.50 else "authentic_screen_capture"),
        "contributions": contributions,
    }

