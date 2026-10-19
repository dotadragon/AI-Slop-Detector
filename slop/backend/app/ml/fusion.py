# ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING eval_accuracy.py AND HUMAN APPROVAL
"""
SLOP Forensic Fusion Engine v1.0 — Single Score Authority
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

def resolve_fusion_dir() -> Path:
    """Resolve Model/fusion directory across ancestor paths."""
    base_dir = Path(__file__).resolve()
    for p in base_dir.parents:
        cand = p / "Model" / "fusion"
        if cand.exists():
            return cand
    return base_dir.parents[4] / "Model" / "fusion"


FUSION_MODEL_DIR = resolve_fusion_dir()

# Zero-shot signals the v2 GBDT head was trained with. Its feature vector is only
# meaningful in the context of all of them, so a call missing any one of these must
# not be scored by the GBDT (inventing values fabricates evidence, and passing None
# raises). Such calls are routed to the legacy logistic artifact instead.
_TEXT_ZERO_SHOT_SIGNALS = (
    "binoculars_ai_prob",
    "fast_detectgpt_ai_prob",
    "gltr_top10_frac",
    "gltr_tail_frac",
    "token_ll_std",
    "fast_z_norm",
    "lm_perplexity",
    "coherence_mean",
    "coherence_var",
)

# Production's pre-v2 logistic weights, kept on disk as the handling path for
# partial-signal calls. Loaded by name from the same Model/fusion directory.
_LEGACY_TEXT_MODEL = "text_pre_raid"


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


def _band_from_artifact(score: float, cal: dict[str, float]) -> str:
    thr = cal.get("binary_threshold")
    zlow = cal.get("zone_low")
    zhigh = cal.get("zone_high")
    zdef = cal.get("zone_definitely_human")
    # Definite AI edge: highest of decision edges we have
    candidates = [x for x in [thr, zhigh] if x is not None]
    ai_def = max(candidates) if candidates else thr

    # Order matters: most confident to least
    if zdef is not None and score <= zdef:
        return "DEFINITELY_HUMAN"
    if zlow is not None and score < zlow:
        return "LIKELY_HUMAN"
    if zhigh is not None and score < zhigh:
        return "UNCERTAIN"
    if thr is not None and score < thr:
        return "LIKELY_AI"
    if ai_def is not None and score >= ai_def:
        return "DEFINITELY_AI"
    return "UNCERTAIN"


def _confidence_range(score: float, disagreement: float = 0.0) -> list[float]:
    margin = 0.04 + min(0.10, max(0.0, float(disagreement)) * 0.20)
    return [round(max(0.0, score - margin), 3), round(min(1.0, score + margin), 3)]


def _find_fusion_model_path(modality: str) -> Path | None:
    p = resolve_fusion_dir() / f"{modality}.pkl"
    if p.exists():
        return p
    return None


# Per-(path, mtime_ns, size) cache of unpickled fusion artifacts. Fusion weights
# are read on every scoring call, which re-unpickled the same small pickles each
# time. Caching is keyed on file identity so an operator swapping Model/fusion
# artifacts on disk is picked up on the next call (no stale serving), while the
# steady-state cost drops to a dict lookup. Pickles are immutable-on-disk data
# (no shared mutable state), so the cached object is safe to reuse across calls.
_ARTIFACT_CACHE: dict[tuple[str, int, int], Any] = {}


def _load_pickle_artifact(path: Path) -> Any:
    """Unpickle path once per (mtime_ns, size); later calls reuse the object."""
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is None:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        _ARTIFACT_CACHE[key] = cached
    return cached


def _load_model_weights(modality: str) -> dict[str, Any] | None:
    pkl_path = _find_fusion_model_path(modality)
    if pkl_path and pkl_path.exists():
        try:
            data = _load_pickle_artifact(pkl_path)
            if isinstance(data, dict) and ("weights" in data or "model" in data):
                return data
        except Exception as e:
            logger.warning(f"Failed to load fusion model for {modality} from {pkl_path}: {e}")
    return None


def _load_legacy_text_logistic() -> dict[str, Any] | None:
    """Load the pre-v2 text logistic (weights/intercept form), if present.

    Used only as the fallback scorer for calls whose zero-shot signals are absent
    or None. Returns None when the artifact is missing or is not a logistic.
    """
    data = _load_model_weights(_LEGACY_TEXT_MODEL)
    if data is not None and "weights" in data and "intercept" in data:
        return data
    return None


def _load_calibrator(modality: str) -> dict[str, Any] | None:
    """Load the Platt calibrator artifact for a modality, if it exists."""
    p = resolve_fusion_dir() / f"{modality}_calibrator.pkl"
    if p.exists():
        try:
            return _load_pickle_artifact(p)
        except Exception as e:
            logger.warning(f"Failed to load calibrator for {modality} from {p}: {e}")
    return None


# ============================================================================
# IMAGE FUSION ENGINE
# ============================================================================

def _standardize(model_data: dict[str, Any], feature_tuples: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Apply stored train-set standardization if the model carries feature stats.

    New retrained models (retrain_text_fusion.py) store feature_mean/feature_std;
    legacy pickles do not, so inference is unchanged for them.
    """
    mu = model_data.get("feature_mean")
    sd = model_data.get("feature_std")
    if not isinstance(mu, list) or not isinstance(sd, list) or len(mu) != len(sd):
        return feature_tuples
    mu_map = dict(zip(model_data.get("features", []), (float(m) for m in mu)))
    sd_map = {n: (float(s) if float(s) >= 1e-9 else 1.0) for n, s in zip(model_data.get("features", []), sd)}
    out = []
    for name, val in feature_tuples:
        if name in mu_map:
            out.append((name, (float(val) - mu_map[name]) / sd_map.get(name, 1.0)))
        else:
            out.append((name, float(val)))
    return out


def fuse_image_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines image forensic features into a calibrated final AI probability.
    Single authority for image scoring. No hard clamps allowed outside this function.
    """
    model_data = _load_model_weights("image")
    signals_missing: list[str] = list(features.get("signals_missing", []))

    best_neural = features.get("best_neural")
    if best_neural is None and features.get("neural_ai_score") is not None:
        best_neural = features.get("neural_ai_score")
    if best_neural is None:
        signals_missing.append("best_neural")

    evidence_keys = ["heuristic_score", "cfa_score", "ela_score", "fft_score", "noise_score", "dct_score"]
    evidence_vals: dict[str, float] = {}
    for k in evidence_keys:
        v = features.get(k)
        if v is None:
            signals_missing.append(k)
            evidence_vals[k] = 0.0
        else:
            evidence_vals[k] = float(v)

    heuristic = evidence_vals["heuristic_score"]
    cfa = evidence_vals["cfa_score"]
    ela = evidence_vals["ela_score"]
    fft = evidence_vals["fft_score"]
    noise = evidence_vals["noise_score"]
    dct = evidence_vals["dct_score"]
    meta_risk = float(features.get("metadata_risk", 0.0))
    is_diagram = float(features.get("is_diagram", 0.0))
    is_real_cam = float(features.get("is_real_camera", 0.0))
    is_anime = float(features.get("is_anime_ai", 0.0))
    sig_detected = float(features.get("signature_detected", 0.0))
    sig_conf = float(features.get("signature_confidence", 0.0))
    c2pa_verified = float(features.get("c2pa_verified", 0.0))
    clip_probe = features.get("clip_probe_score")
    npr = features.get("npr_score")
    latent_grid_score = float(features.get("latent_grid_score", 0.0))

    bn_val = float(best_neural) if best_neural is not None else None
    effective_is_diagram = is_diagram
    if is_diagram > 0.5 and bn_val is not None:
        if bn_val >= 0.75:
            effective_is_diagram = 0.0
        elif bn_val >= 0.55:
            effective_is_diagram = 0.5
    elif bn_val is not None and bn_val >= 0.82 and latent_grid_score >= 0.50:
        effective_is_diagram = 0.0

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
            ("is_diagram", effective_is_diagram),
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

        for feat_name, feat_val in _standardize(model_data, feature_tuples):
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        contributions = {}
        if bn_val is not None:
            bn = bn_val
            contributions["best_neural"] = bn

            disagreement = abs(heuristic - bn)
            # Consensus logic: Neural model holds primary authority (80%+)
            if bn >= 0.70:
                score = bn * 0.90 + heuristic * 0.10
            elif bn >= 0.55:
                # Retain full neural detection score on AI images
                score = bn * 0.85 + heuristic * 0.15
            elif bn < 0.40:
                # Authentic photo or diagram suppression when neural score is already low (< 0.40)
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

        # C2PA provenance authentication — tiered trust multiplier (Phase 8).
        # fully_trusted 0.10 | self_signed 0.35 | unverified_claim 0.60 | broken_chain 0.80
        if c2pa_verified > 0.5:
            score *= float(features.get("c2pa_multiplier", 0.15))

    # Diagram constraints (Lead Directive 2 & Rule 6):
    # If neural_ai_score >= 0.55 (or best_neural >= 0.55), diagram_penalty MUST BE EXACTLY 0.0.
    # AI-generated diagrams, schematics, and flowcharts must retain their full neural detection score.
    # Only apply downward diagram adjustment when neural score is already low (< 0.40)
    # to prevent false positives on human technical drawings.
    diagram_penalty = 0.0
    if (is_diagram > 0.5 or effective_is_diagram > 0.5) and bn_val is not None:
        if bn_val >= 0.75:
            diagram_penalty = 0.0
            score = max(score, bn_val)
            if "is_diagram" in contributions:
                contributions["is_diagram"] = 0.0
        elif bn_val >= 0.55:
            target_moderated = round(min(max(bn_val - 0.24, 0.38), 0.45), 3)
            diagram_penalty = max(0.0, score - target_moderated)
            score = target_moderated
        elif bn_val < 0.40:
            diagram_penalty = max(0.0, score - min(score, 0.05))
            score = min(score, 0.05)
        else:
            diagram_penalty = min(0.15, max(0.0, score - (bn_val - 0.15)))
            score = max(score, bn_val - 0.15)

    final_score = round(_clamp01(score), 3)
    disagreement = abs(float(best_neural if best_neural is not None else 0.5) - heuristic)

    return {
        "overall_score": final_score,
        "confidence_band": _confidence_band(final_score),
        "confidence_range": _confidence_range(final_score, disagreement),
        "primary_signal": "neural_vision_transformer" if (best_neural is not None and best_neural >= 0.50) else "forensic_heuristics",
        "contributions": contributions,
        "signals_missing": signals_missing,
    }


# ============================================================================
# TEXT FUSION ENGINE
# ============================================================================

def _apply_code_forensics(
    score: float,
    features: dict[str, Any],
    contributions: dict[str, Any],
) -> tuple[float, str | None, str | None, list[float] | None]:
    """
    Code Forensics Track (GEMINI.md Directive 6):
    When is_code == 1.0:
      - If identifier_entropy < 0.40 and comment_ratio < 0.05:
        score = min(raw_score, 0.05)
        verdict = "Manual Coded (Human / Competitive)"
      - Else if comment_ratio > 0.40 or neural_ai_score > 0.85:
        score = max(raw_score, 0.88)
        verdict = "Vibe Coded (AI / Assistant Generated)"
      - Else:
        score = 0.50
        verdict = "Mixed / Unverified Code"
      - Set contributions['code_origin'] = verdict.
    """
    is_code = float(features.get("is_code", 0.0))
    if is_code != 1.0:
        return score, None, None, None

    raw_score = score
    ident_ent = float(features.get("identifier_entropy", 0.5))
    comm_rat = float(features.get("comment_ratio", 0.0))
    neural_prob = features.get("neural_prob")
    transformer_prob = features.get("transformer_prob")
    neural_ai_score = float(neural_prob if neural_prob is not None else (transformer_prob if transformer_prob is not None else 0.0))

    if ident_ent < 0.40 and comm_rat < 0.05:
        adj_score = min(raw_score, 0.05)
        verdict = "Manual Coded (Human / Competitive)"
        band = "DEFINITELY_HUMAN"
        rng = [0.0, 0.05]
    elif comm_rat > 0.40 or neural_ai_score > 0.85:
        adj_score = max(raw_score, 0.88)
        verdict = "Vibe Coded (AI / Assistant Generated)"
        band = "DEFINITELY_AI"
        rng = [0.88, 1.0]
    else:
        adj_score = 0.50
        verdict = "Mixed / Unverified Code"
        band = "UNCERTAIN"
        rng = [0.40, 0.60]

    contributions["code_origin"] = verdict
    return adj_score, verdict, band, rng


def fuse_text_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines text forensic features into a calibrated final AI probability.
    Single authority for text scoring. No hard clamps allowed outside this function.

    Supports three model formats loaded from Model/fusion/text.pkl:
      1. GBDT (v2+): dict with 'model' key -> LightGBM predict_proba -> Platt calibration
      2. Legacy logistic regression: dict with 'weights'/'intercept' keys -> sigmoid
      3. Neither on disk: heuristic consensus fallback

    A call carrying an incomplete zero-shot feature set is deliberately NOT scored by
    the GBDT: it is routed to the legacy logistic (see _TEXT_ZERO_SHOT_SIGNALS).
    """
    model_data = _load_model_weights("text")

    neural_prob = features.get("neural_prob")
    signals_missing = list(features.get("signals_missing", []))
    heuristic_raw = features.get("heuristic_score")
    if heuristic_raw is None:
        signals_missing.append("heuristic_score")
        heuristic = 0.0
    else:
        heuristic = float(heuristic_raw)
    llm_pattern = float(features.get("llm_pattern_score", 0.0))

    # Absent/None zero-shot signal -> this call is not a GBDT call. Route it to the
    # legacy logistic, which already skips missing features (and handles None).
    feature_names = model_data.get("features", []) if model_data else []
    required_zero_shot = [n for n in _TEXT_ZERO_SHOT_SIGNALS if not feature_names or n in feature_names]
    missing_zero_shot = [n for n in required_zero_shot if features.get(n) is None]
    if model_data is not None and "model" in model_data and missing_zero_shot:
        legacy = _load_legacy_text_logistic()
        if legacy is not None:
            logger.info(
                "text_fusion_gbdt_bypassed_incomplete_signals",
                extra={"missing": missing_zero_shot, "fallback": "legacy_logistic"},
            )
            model_data = legacy
        else:
            logger.error(
                "text_fusion_gbdt_bypassed_no_legacy_model",
                extra={"missing": missing_zero_shot, "fallback": "heuristic_only"},
            )
            model_data = None

    # ── GBDT + Platt calibration path (v2 promoted model) ──
    if model_data is not None and "model" in model_data:
        gbdt = model_data["model"]
        feature_names = model_data.get("features", [])
        calibrator_data = _load_calibrator("text")

        # Every zero-shot signal is guaranteed present and non-None here (the guard
        # above reroutes incomplete calls to the legacy path), so no value is invented
        # and no optimistic default can fabricate AI evidence.
        zero_shot = {name: float(features[name]) for name in _TEXT_ZERO_SHOT_SIGNALS if features.get(name) is not None}

        # Build feature vector in the exact order the GBDT was trained on
        feat_defaults = {
            "transformer_prob": float(features.get("transformer_prob", neural_prob if neural_prob is not None else 0.0)),
            "neural_prob": float(neural_prob if neural_prob is not None else 0.0),
            "heuristic_score": heuristic,
            "llm_pattern_score": llm_pattern,
            "transitional_entropy": float(features.get("transitional_entropy", 0.5)),
            "burstiness": float(features.get("burstiness", 0.5)),
            "contraction_score": float(features.get("contraction_score", 0.5)),
            "zipf_deviation": float(features.get("zipf_deviation", 0.5)),
            "vocab_richness": float(features.get("vocab_richness", 0.5)),
            "gltr_top10_frac": float(features.get("gltr_top10_frac", 0.5)),
            "gltr_tail_frac": float(features.get("gltr_tail_frac", 0.5)),
            "token_ll_std": float(features.get("token_ll_std", 0.0)),
            "fast_z_norm": float(features.get("fast_z_norm", 0.0)),
            **zero_shot,
        }

        import numpy as np
        row = np.array([[feat_defaults.get(fn, 0.0) for fn in feature_names]], dtype=np.float32)
        raw_prob = float(gbdt.predict_proba(row)[0, 1])

        contributions = {fn: round(feat_defaults.get(fn, 0.0), 4) for fn in feature_names}

        # Apply Platt calibration. The promoted path requires EVERY decision
        # number (binary threshold + all zone edges) to come from the artifact
        # on disk: there are no literals here, so no operating point can drift
        # away from the fitted model. If the artifact is absent or incomplete we
        # degrade explicitly rather than synthesising a threshold.
        platt = calibrator_data.get("calibrator") if calibrator_data else None
        required = ("binary_threshold", "zone_low", "zone_high", "zone_definitely_human")
        missing = [k for k in required if not calibrator_data or k not in calibrator_data]
        if platt is None or missing:
            cal_prob = raw_prob
            binary_threshold = zone_low = zone_high = zone_definitely_human = None
            conf_band = "UNCALIBRATED"
            verdict = None
            logger.error(
                "text_calibrator_incomplete: refusing to synthesise an operating "
                f"point (missing={missing}); returning raw GBDT probability"
            )
        else:
            assert calibrator_data is not None
            eps = 1e-6
            logit = float(np.log(max(raw_prob, eps) / max(1.0 - raw_prob, eps)))
            cal_prob = float(platt.predict_proba(np.array([[logit]]))[0, 1])
            zone_low = float(calibrator_data["zone_low"])
            zone_high = float(calibrator_data["zone_high"])
            zone_definitely_human = float(calibrator_data["zone_definitely_human"])
            binary_threshold = float(calibrator_data["binary_threshold"])

            effective_threshold = binary_threshold

            conf_band = _band_from_artifact(cal_prob, calibrator_data)
            if cal_prob >= effective_threshold:
                verdict = "AI"
            elif zone_low is not None and cal_prob >= zone_low:
                verdict = "UNCERTAIN"
            else:
                verdict = "HUMAN"

        if binary_threshold is not None and binary_threshold > 0.0:
            if cal_prob < binary_threshold:
                score = (cal_prob / binary_threshold) * 0.46
            else:
                score = 0.52 + 0.48 * ((cal_prob - binary_threshold) / max(1.0 - binary_threshold, 1e-5))
                score = max(score, 0.55)
        else:
            score = cal_prob

        # Frontier LLM consensus elevation:
        # Frontier reasoning and chat models (Claude 3.5, DeepSeek R1, GPT-4o, Llama 3.3)
        # exhibit elevated stylometric indicators (>= 0.30), presence of AI discourse markers (>= 1.0),
        # low burstiness (<= 0.12), and contraction avoidance (>= 0.60) that distinguish them from human prose.
        burst = float(features.get("burstiness", 0.5))
        stylometry = float(features.get("stylometry_score", 0.0))
        markers = float(features.get("marker_hits", 0.0))
        is_frontier_ai = (
            (stylometry >= 0.30 or markers >= 1.0)
            and (
                (burst <= 0.12 and float(features.get("contraction_score", 0.5)) >= 0.60)
                or llm_pattern >= 0.55
                or float(features.get("gltr_top10_frac", 0.0)) >= 0.55
            )
        )
        if is_frontier_ai:
            score = max(score, 0.65)
        elif llm_pattern >= 0.55:
            score = max(score, 0.75)
        code_score, code_verdict, code_band, code_rng = _apply_code_forensics(score, features, contributions)
        if code_verdict is not None:
            score = code_score
            conf_band = code_band

        disagreement = abs(heuristic - float(neural_prob if neural_prob is not None else 0.5))
        final_score = round(_clamp01(score), 3)
        final_range = code_rng if code_verdict is not None else _confidence_range(final_score, disagreement)

        ai_sent_frac = float(features.get("ai_sentence_fraction", 0.0))
        hu_sent_frac = float(features.get("human_sentence_fraction", 0.0))
        sent_std = float(features.get("sentence_prob_std", 0.0))
        sub_count = int(features.get("substantive_sentence_count", features.get("sentence_count", 0)))

        if sub_count >= 4 and ai_sent_frac >= 0.25 and hu_sent_frac >= 0.25 and sent_std >= 0.22:
            content_composition = "MIXED_AI_HUMAN"
        else:
            is_ai_verdict = (verdict == "AI") if verdict is not None else (final_score >= 0.50)
            content_composition = "PURE_AI" if is_ai_verdict else "PURE_HUMAN"

        return {
            "overall_score": final_score,
            "confidence_band": conf_band,
            "confidence_range": final_range,
            "primary_signal": "gbdt_platt_calibrated" if code_verdict is None else "code_forensics_ast_identifier",
            "contributions": contributions,
            "raw_gbdt_prob": round(raw_prob, 4),
            "calibrated_prob": float(cal_prob),
            "binary_threshold": None if binary_threshold is None else round(binary_threshold, 4),
            "zone_low": None if zone_low is None else round(zone_low, 4),
            "zone_high": None if zone_high is None else round(zone_high, 4),
            "zone_definitely_human": (
                None if zone_definitely_human is None else round(zone_definitely_human, 4)
            ),
            "verdict": verdict if (platt is not None and not missing) else None,
            "content_composition": content_composition,
            "model_path": "gbdt_platt_v2",
            "signals_missing": signals_missing,
        }

    # ── Legacy logistic regression path ──
    if model_data is not None and "weights" in model_data:
        w = model_data.get("weights", {})
        intercept = float(model_data.get("intercept", 0.0))
        logit = intercept
        contributions = {}
        entropy = float(features.get("transitional_entropy", 0.5))
        burstiness = float(features.get("burstiness", 0.5))
        contraction = float(features.get("contraction_score", 0.5))
        zipf = float(features.get("zipf_deviation", 0.5))
        vocab = float(features.get("vocab_richness", 0.5))
        bino_prob = features.get("binoculars_ai_prob")
        fast_prob = features.get("fast_detectgpt_ai_prob")

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

        for feat_name, feat_val in _standardize(model_data, feature_tuples):
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        # No model on disk: heuristic fallback
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

    code_score, code_verdict, code_band, code_rng = _apply_code_forensics(score, features, contributions)
    if code_verdict is not None:
        score = code_score
        conf_band = code_band
        conf_range = code_rng
    else:
        conf_band = _confidence_band(round(_clamp01(score), 3))
        conf_range = _confidence_range(round(_clamp01(score), 3), abs(heuristic - float(neural_prob if neural_prob is not None else 0.5)))

    final_score = round(_clamp01(score), 3)

    ai_sent_frac = float(features.get("ai_sentence_fraction", 0.0))
    hu_sent_frac = float(features.get("human_sentence_fraction", 0.0))
    sent_std = float(features.get("sentence_prob_std", 0.0))
    sub_count = int(features.get("substantive_sentence_count", features.get("sentence_count", 0)))

    if sub_count >= 4 and ai_sent_frac >= 0.25 and hu_sent_frac >= 0.25 and sent_std >= 0.22:
        content_composition = "MIXED_AI_HUMAN"
    else:
        content_composition = "PURE_AI" if final_score >= 0.50 else "PURE_HUMAN"

    return {
        "overall_score": final_score,
        "confidence_band": conf_band,
        "confidence_range": conf_range,
        "primary_signal": "code_forensics_ast_identifier" if code_verdict is not None else ("roberta_neural_classifier" if (neural_prob is not None and neural_prob >= 0.50) else "stylometric_forensics"),
        "contributions": contributions,
        "content_composition": content_composition,
        "model_path": "legacy_logistic_fallback",
        "signals_missing": missing_zero_shot,
    }


# ============================================================================
# VIDEO FUSION ENGINE
# ============================================================================

def fuse_audio_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines audio neural + heuristic signals into a final deepfake probability.
    Single authority for audio scoring. No hard clamps allowed outside this function.
    Neural-First doctrine: the Wav2Vec2 verdict dominates when available;
    the spectral heuristic is a corroborating (minor) term.
    """
    signals_missing: list[str] = list(features.get("signals_missing", []))
    neural_fake = features.get("neural_fake")
    if neural_fake is None:
        signals_missing.append("neural_fake")
    heuristic_raw = features.get("heuristic_score")
    if heuristic_raw is None:
        signals_missing.append("heuristic_score")
        heuristic = 0.5
    else:
        heuristic = float(heuristic_raw)
    contributions: dict[str, float] = {}

    if neural_fake is not None:
        nf = float(neural_fake)
        score = 0.75 * nf + 0.25 * heuristic
        confidence = 0.85
        contributions["neural_fake"] = round(0.75 * nf, 4)
        contributions["heuristic_score"] = round(0.25 * heuristic, 4)
    else:
        # Model unavailable: heuristic only, deliberately low confidence
        score = heuristic
        confidence = 0.35
        contributions["heuristic_score"] = round(heuristic, 4)

    if score >= 0.75:
        verdict = "likely_ai"
    elif score >= 0.5:
        verdict = "uncertain"
    elif score <= 0.25:
        verdict = "likely_human"
    else:
        verdict = "uncertain"

    return {
        "overall_score": round(float(score), 4),
        "confidence": round(confidence, 3),
        "verdict": verdict,
        "contributions": contributions,
        "signals_missing": signals_missing,
    }


def fuse_video_score(features: dict[str, Any]) -> dict[str, Any]:
    """
    Combines video frame and temporal forensic features into a calibrated final AI probability.
    Single authority for video scoring. No hard clamps allowed outside this function.
    """
    signals_missing: list[str] = list(features.get("signals_missing", []))
    core_keys = ["peak_threat", "top_quartile_mean", "global_mean"]
    for k in core_keys:
        if features.get(k) is None:
            signals_missing.append(k)

    if any(k in signals_missing for k in core_keys):
        return {
            "overall_score": 0.0,
            "confidence_band": "UNAVAILABLE",
            "confidence_range": [0.0, 0.0],
            "verdict": "unavailable",
            "primary_signal": "insufficient_video_frames",
            "contributions": {},
            "signals_missing": signals_missing,
        }

    peak_threat = float(features["peak_threat"])
    top_quartile = float(features["top_quartile_mean"])
    global_mean = float(features["global_mean"])
    is_screen_rec = float(features.get("is_screen_rec", 0.0))
    has_deepfake = float(features.get("has_deepfake_face", 0.0))
    max_face = float(features.get("max_face_score", 0.0))
    median_face = float(features.get("median_face_score", max_face))
    flicker = float(features.get("temporal_flicker", 0.0))
    jitter = float(features.get("temporal_jitter", 0.0))
    flow_residual = float(features.get("flow_residual_score", 0.0))
    color_instability = float(features.get("color_instability", 0.0))
    frame_score_std = float(features.get("frame_score_std", 0.0))
    synthid = float(features.get("synthid_detected", 0.0))
    high_threat_count = int(features.get("high_threat_frame_count", 0))
    frames_with_faces = int(features.get("frames_with_faces", 0))

    model_data = _load_model_weights("video")

    if model_data is not None and int(model_data.get("n_samples", 0)) >= 20:
        w = model_data.get("weights", {})
        intercept = float(model_data.get("intercept", 0.0))
        logit = intercept
        contributions = {}
        video_tuples = [
            ("peak_threat", peak_threat),
            ("top_quartile_mean", top_quartile),
            ("global_mean", global_mean),
            ("is_screen_rec", is_screen_rec),
            ("has_deepfake_face", has_deepfake),
            ("max_face_score", max_face),
            ("temporal_flicker", flicker),
            ("temporal_jitter", jitter),
            ("synthid_detected", synthid),
        ]
        for feat_name, feat_val in _standardize(model_data, video_tuples):
            coef = float(w.get(feat_name, 0.0))
            contrib = coef * float(feat_val)
            logit += contrib
            contributions[feat_name] = round(contrib, 4)

        score = _sigmoid(logit)
    else:
        contributions = {}
        # 40% Peak + 40% Top-Quartile + 20% Global Mean
        score = 0.40 * peak_threat + 0.40 * top_quartile + 0.20 * global_mean
        contributions["peak_threat"] = round(0.40 * peak_threat, 4)
        contributions["top_quartile_mean"] = round(0.40 * top_quartile, 4)
        contributions["global_mean"] = round(0.20 * global_mean, 4)

    # Post-logistic / post-heuristic video modality & temporal calibration
    if is_screen_rec > 0.5 and has_deepfake <= 0.5:
        # Authentic human software UI screen recording
        score = 0.04
        contributions["screen_recording_suppression"] = 0.04
    elif (has_deepfake > 0.5 or median_face >= 0.50 or max_face >= 0.70) and is_screen_rec <= 0.5:
        score = max(score, median_face, max_face, 0.82)
        contributions["facial_deepfake_consensus"] = round(float(score), 4)
    elif synthid > 0.5 and is_screen_rec <= 0.5:
        score = max(score, 0.85)
        contributions["synthid_corroboration"] = 0.85
    elif (flicker > 0.08 or jitter > 0.06) and score >= 0.40 and is_screen_rec <= 0.5 and (has_deepfake > 0.5 or frames_with_faces == 0):
        score = min(0.98, score + 0.06)
        contributions["temporal_jitter_corroboration"] = 0.06
    elif peak_threat >= 0.80 and high_threat_count >= 3 and is_screen_rec <= 0.5 and (frames_with_faces == 0 or has_deepfake > 0.5):
        score = max(score, 0.80)
        contributions["high_threat_persistence"] = 0.80

    # Natural real-camera footage correction: suppress false positives on authentic video
    if is_screen_rec <= 0.5 and has_deepfake <= 0.5 and median_face <= 0.45 and max_face <= 0.50 and synthid <= 0.5:
        if frames_with_faces > 0:
            # Video contains verified authentic human face(s) and zero synthetic signatures
            score = min(score, 0.20)
            contributions["natural_video_correction"] = round(float(score), 4)
        else:
            if color_instability <= 0.20 and jitter <= 0.10 and flow_residual <= 0.12:
                score = min(score * 0.45, 0.25)
                contributions["natural_video_correction"] = round(float(score), 4)

    final_score = round(_clamp01(score), 3)

    return {
        "overall_score": final_score,
        "confidence_band": _confidence_band(final_score),
        "confidence_range": _confidence_range(final_score, abs(peak_threat - global_mean)),
        "primary_signal": "deepfake_face_vit" if (has_deepfake > 0.5 or max_face >= 0.50) else ("swin_generative_diffusion" if final_score >= 0.50 else ("authentic_screen_capture" if is_screen_rec > 0.5 else "authentic_camera_video")),
        "contributions": contributions,
        "signals_missing": signals_missing,
    }


# ============================================================================
# DOCUMENT FUSION ENGINE
# ============================================================================

def fuse_document_score(
    figure_scores: list[float],
) -> tuple[float, float, str, float, float]:
    """
    Combines embedded figure forensic scores into a calibrated final AI probability.
    Single authority for document scoring (ANTIGRAVITY Rule 3).
    Returns (overall_score, confidence, verdict, max_score, ai_fraction).
    """
    if not figure_scores:
        return 0.5, 0.15, "uncertain", 0.0, 0.0

    max_score = max(figure_scores)
    ai_fraction = sum(1 for s in figure_scores if s >= 0.5) / len(figure_scores)
    overall = 0.6 * max_score + 0.4 * ai_fraction
    confidence = min(0.9, 0.35 + 0.1 * len(figure_scores))

    if overall >= 0.75:
        verdict = "likely_ai"
    elif overall <= 0.25:
        verdict = "likely_human"
    else:
        verdict = "uncertain"

    return float(overall), float(confidence), verdict, float(max_score), float(ai_fraction)


