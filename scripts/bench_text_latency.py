"""AFK Phase A — text latency baseline (read-only; no production code changes).

Times 5 representative text analyses through the REAL production path, with a
per-stage breakdown:

  _heuristic (pure CPU) / _model_prob (RoBERTa) / binoculars pair /
  _fine_tuned_model_prob / perplexity (Qwen 0.5B) / coherence /
  fusion total (GBDT predict + Platt apply)

Cold start (very first call, includes model loading) is reported separately from
warm calls. Run with the backend venv:

    slop/backend/.venv/Scripts/python.exe -X utf8 bench_text_latency.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")

VAL = ROOT / "raid_val" / "text"
SAMPLES = [
    "ai/gpt2/gpt2_0000.txt",
    "ai/llama_chat/llama_chat_0000.txt",
    "human/casual/casual_val_0000.txt",
    "human/forum/forum_val_0000.txt",
    "human/email/email_val_0000.txt",
]


def main() -> int:
    from app.ml import text_detector as td
    from app.ml.binoculars import compute_binoculars_and_curvature
    from app.ml.fusion import fuse_text_score
    from app.ml.perplexity import compute_coherence, compute_perplexity

    # ── Cold start: the very first analyze_text call in this process, before any
    # instrumentation touches a model, so the reported number includes every load.
    first_text = (VAL / SAMPLES[0]).read_text(encoding="utf-8", errors="ignore")
    t = time.perf_counter()
    first_res = td.analyze_text(first_text)
    cold_total = time.perf_counter() - t
    print(f"[COLD] first analyze_text() in process: {cold_total:.3f}s "
          f"(score={first_res.get('overall_score')})")

    rows: list[dict[str, Any]] = []
    for i, sid in enumerate(SAMPLES):
        text = (VAL / sid).read_text(encoding="utf-8", errors="ignore")

        stages: dict[str, float] = {}
        t = time.perf_counter()
        clean = td._normalize_and_sanitize_text(text)
        stages["normalize"] = time.perf_counter() - t

        t = time.perf_counter()
        h = td._heuristic(clean)
        stages["heuristic"] = time.perf_counter() - t

        t = time.perf_counter()
        mp = td._model_prob(clean)
        stages["model_prob_roberta"] = time.perf_counter() - t

        t = time.perf_counter()
        bino = compute_binoculars_and_curvature(clean)
        stages["binoculars_pair"] = time.perf_counter() - t

        t = time.perf_counter()
        tprob = td._fine_tuned_model_prob(clean)
        stages["fine_tuned_transformer"] = time.perf_counter() - t

        t = time.perf_counter()
        ppl = compute_perplexity(clean)
        stages["perplexity_qwen"] = time.perf_counter() - t

        t = time.perf_counter()
        coh = compute_coherence(clean)
        stages["coherence"] = time.perf_counter() - t

        import math as _math
        feats = {
            "transformer_prob": tprob if tprob is not None else (mp if mp is not None else 0.0),
            "neural_prob": mp,
            "heuristic_score": h["overall_score"],
            "llm_pattern_score": h["signals"].get("llm_pattern_score", 0.0),
            "transitional_entropy": h["signals"].get("transitional_entropy_variance", 0.5),
            "burstiness": h["signals"].get("burstiness", 0.5),
            "contraction_score": h["signals"].get("contraction_avoidance", 0.5),
            "zipf_deviation": h["signals"].get("zipf_deviation", 0.5),
            "vocab_richness": h["signals"].get("vocabulary_richness", 0.5),
            "char_count": len(clean),
            "binoculars_score": bino.get("binoculars_score"),
            "binoculars_ai_prob": bino.get("binoculars_ai_prob"),
            "fast_detectgpt_z": bino.get("fast_detectgpt_z"),
            "fast_detectgpt_ai_prob": bino.get("fast_detectgpt_ai_prob"),
            "lm_perplexity": ppl["perplexity"] if ppl["available"] and not _math.isnan(ppl["perplexity"]) else 0.0,
            "coherence_mean": coh["coherence_mean"] if coh["available"] and not _math.isnan(coh["coherence_mean"]) else 0.0,
            "coherence_var": coh["coherence_var"] if coh["available"] and not _math.isnan(coh["coherence_var"]) else 0.0,
        }

        t = time.perf_counter()
        fres = fuse_text_score(feats)
        stages["fusion_platt"] = time.perf_counter() - t

        t = time.perf_counter()
        e2e = td.analyze_text(text)
        stages["end_to_end_analyze_text"] = time.perf_counter() - t

        feature_sum = sum(v for k, v in stages.items()
                          if k not in ("fusion_platt", "end_to_end_analyze_text", "normalize"))
        slowest = max(((k, v) for k, v in stages.items()
                       if k not in ("fusion_platt", "end_to_end_analyze_text")), key=lambda kv: kv[1])
        rows.append({
            "sample": sid,
            "stages": {k: round(v, 4) for k, v in stages.items()},
            "feature_extraction_total": round(feature_sum, 4),
            "slowest_feature": f"{slowest[0]} ({slowest[1]:.3f}s)",
            "score": e2e.get("overall_score"),
            "band": e2e.get("confidence_band"),
        })
        print(f"[warm{i+1}] {sid:38s} e2e={stages['end_to_end_analyze_text']:.3f}s "
              f"features={feature_sum:.3f}s fusion={stages['fusion_platt']:.4f}s "
              f"slowest={slowest[0]}({slowest[1]:.2f}s) score={e2e.get('overall_score')}")

    # GBDT predict vs Platt apply, micro-timed on the promoted artifacts.
    import pickle
    import numpy as np
    from app.ml.fusion import _load_model_weights, _load_calibrator
    md = _load_model_weights("text")
    cal = _load_calibrator("text")
    assert md is not None and cal is not None
    names = md["features"]
    row = np.array([[0.5] * len(names)], dtype=np.float32)
    n = 200
    p: float = 0.5
    t = time.perf_counter()
    for _ in range(n):
        p = float(md["model"].predict_proba(row)[0, 1])
    gbdt_us = (time.perf_counter() - t) / n * 1e6
    platt = cal["calibrator"]
    logit = np.array([[float(np.log(p / (1 - p)))]])
    t = time.perf_counter()
    for _ in range(n):
        platt.predict_proba(logit)
    platt_us = (time.perf_counter() - t) / n * 1e6

    warm = [r["stages"]["end_to_end_analyze_text"] for r in rows]
    cold = cold_total
    summary = {
        "cold_start_total_s": round(cold, 4),
        "warm_total_s": [round(w, 4) for w in warm],
        "warm_mean_s": round(sum(warm) / len(warm), 4),
        "gbdt_predict_us": round(gbdt_us, 1),
        "platt_apply_us": round(platt_us, 1),
        "rows": rows,
    }
    print("\nSUMMARY")
    print(f"  cold start (very first call, incl. every model load): {cold:.3f}s")
    print(f"  warm calls: {[round(w,3) for w in warm]}  mean {sum(warm)/len(warm):.3f}s")
    print(f"  GBDT predict: {gbdt_us:.1f} us/call    Platt apply: {platt_us:.1f} us/call")
    (ROOT / "text_latency_baseline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote text_latency_baseline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
