"""Read-only None-probe for the text fusion zero-shot guard.

Confirms that calling fuse_text_score() with absent/None zero-shot signals reroutes
to the legacy logistic (or heuristic-only) instead of raising TypeError, while a
full-signal call still uses the promoted GBDT + Platt path.
"""
import os
import sys

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")
sys.path.insert(0, "slop/backend")

from app.ml.fusion import fuse_text_score  # noqa: E402

HUMAN_BASE = {
    "transformer_prob": 0.12,
    "neural_prob": 0.10,
    "heuristic_score": 0.18,
    "llm_pattern_score": 0.09,
    "transitional_entropy": 0.62,
    "burstiness": 0.55,
    "contraction_score": 0.40,
    "zipf_deviation": 0.33,
    "vocab_richness": 0.58,
}

FULL_ZERO_SHOT = {
    "binoculars_ai_prob": 0.08,
    "fast_detectgpt_ai_prob": 0.11,
    "lm_perplexity": 0.30,
    "coherence_mean": 0.55,
    "coherence_var": 0.12,
}

AI_ZERO_SHOT = {
    "binoculars_ai_prob": 0.93,
    "fast_detectgpt_ai_prob": 0.95,
    "lm_perplexity": 0.88,
    "coherence_mean": 0.80,
    "coherence_var": 0.05,
}

AI_BASE = {
    "transformer_prob": 0.97,
    "neural_prob": 0.95,
    "heuristic_score": 0.88,
    "llm_pattern_score": 0.82,
    "transitional_entropy": 0.20,
    "burstiness": 0.81,
    "contraction_score": 0.05,
    "zipf_deviation": 0.77,
    "vocab_richness": 0.44,
}

CASES = []
for missing in ("binoculars_ai_prob", "fast_detectgpt_ai_prob"):
    feats = dict(HUMAN_BASE)
    feats.update(FULL_ZERO_SHOT)
    feats[missing] = None
    CASES.append((f"{missing}=None", feats))
both = dict(HUMAN_BASE)
both.update(FULL_ZERO_SHOT)
both["binoculars_ai_prob"] = None
both["fast_detectgpt_ai_prob"] = None
CASES.append(("both=None", both))
lmcoh = dict(HUMAN_BASE)
lmcoh.update(FULL_ZERO_SHOT)
lmcoh["lm_perplexity"] = None
lmcoh["coherence_mean"] = None
lmcoh["coherence_var"] = None
CASES.append(("lm/coh=None", lmcoh))

full_human = dict(HUMAN_BASE)
full_human.update(FULL_ZERO_SHOT)
CASES.append(("FULL control (human-like)", full_human))
full_ai = dict(AI_BASE)
full_ai.update(AI_ZERO_SHOT)
CASES.append(("FULL control (AI-like)", full_ai))

print("=== None-probe (read-only) ===")
scores = []
for label, feats in CASES:
    try:
        res = fuse_text_score(feats)
        scores.append(res["overall_score"])
        print(
            f"  {label:28s} score={res['overall_score']:.4f} "
            f"band={res['confidence_band']:16s} signal={res['primary_signal']}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:28s} RAISED {type(exc).__name__}: {exc}")
        scores.append(None)

print(f"=== scores returned: {scores} ===")
print(f"=== valid (non-None) scores: {sum(s is not None for s in scores)}/{len(scores)} ===")
