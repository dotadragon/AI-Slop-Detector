"""STEP 5 — production-path verification of the promoted text model.

Drives the REAL production analyzer (`app.ml.text_detector.analyze_text`) over the
sealed val split (raid_val/text, n=180) and recomputes the promotion metrics from
what production actually emits — not from the training script.

The binary threshold is read from Model/fusion/text_calibrator.pkl (never
hardcoded). The fusion layer is wrapped (not replaced) so we can capture the
production `calibrated_prob` alongside the analyzer's own `overall_score`.

Writes a per-file cache so a timeout can be resumed without redoing GPU work.

Usage:
    python verify_text_production.py            # all 180, resumable
    python verify_text_production.py --limit 3  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")

VAL_DIR = ROOT / "raid_val" / "text"
CALIBRATOR_PKL = ROOT / "Model" / "fusion" / "text_calibrator.pkl"
CACHE = ROOT / "text_production_verdicts_cache.json"

EXPECTED = {
    "balanced_accuracy": 0.8708,
    "ai_recall": 0.7417,
    "human_recall": 1.0,
    "macro_f1": 0.8232,
    "tp": 89,
    "fn": 31,
    "tn": 60,
    "fp": 0,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N samples (smoke test)")
    args = ap.parse_args()

    with open(CALIBRATOR_PKL, "rb") as f:
        cal = pickle.load(f)
    threshold = float(cal["binary_threshold"])
    print("=" * 78)
    print("STEP 5 — PRODUCTION-PATH VERIFICATION (sealed raid_val/text)")
    print("=" * 78)
    print(f"calibrator      : {CALIBRATOR_PKL}")
    print(f"binary_threshold: {threshold:.10f}  (from disk, not hardcoded)")
    print(f"zone_low={cal['zone_low']:.6f} zone_high={cal['zone_high']:.6f} "
          f"zone_definitely_human={cal['zone_definitely_human']:.6f}")

    import app.ml.fusion as fusion

    captured: list[dict] = []
    original = fusion.fuse_text_score

    def wrapped(features):
        out = original(features)
        captured.append(out)
        return out

    # analyze_text imports fuse_text_score at call time, so patching the module
    # attribute is enough to observe (not alter) production fusion output.
    fusion.fuse_text_score = wrapped
    from app.ml.text_detector import analyze_text

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    items: list[tuple[str, int]] = []
    for label_dir, y in (("ai", 1), ("human", 0)):
        for gen_dir in sorted(p for p in (VAL_DIR / label_dir).iterdir() if p.is_dir()):
            for f in sorted(gen_dir.glob("*.txt")):
                items.append((f"{label_dir}/{gen_dir.name}/{f.name}", y))
    print(f"val samples: {len(items)}  (cached: {sum(1 for s,_ in items if s in cache)})")

    t0 = time.time()
    for i, (sid, y) in enumerate(items):
        if args.limit and i >= args.limit:
            break
        if sid in cache:
            continue
        text = (VAL_DIR / sid).read_text(encoding="utf-8", errors="ignore")
        captured.clear()
        res = analyze_text(text)
        fus = captured[-1] if captured else {}
        cache[sid] = {
            "truth": "AI" if y == 1 else "HUMAN",
            "overall_score": res.get("overall_score"),
            "confidence_band": res.get("confidence_band"),
            "calibrated_prob": fus.get("calibrated_prob"),
            "raw_gbdt_prob": fus.get("raw_gbdt_prob"),
            "fusion_threshold": fus.get("binary_threshold"),
            "fusion_zone_low": fus.get("zone_low"),
            "fusion_zone_high": fus.get("zone_high"),
            "fusion_zone_definitely_human": fus.get("zone_definitely_human"),
        }
        if (i + 1) % 10 == 0 or i == 0:
            CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  [{i+1}/{len(items)}] {sid} score={res.get('overall_score')} "
                  f"band={res.get('confidence_band')} ({time.time()-t0:.0f}s)", flush=True)
    CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    done = [(s, y) for s, y in items if s in cache and cache[s].get("calibrated_prob") is not None]
    print(f"\nresolved samples: {len(done)} / {len(items)}")

    # Rounding-risk check: how close does any sample sit to the threshold?
    gaps = []
    for s, _ in done:
        cp = cache[s]["calibrated_prob"]
        gaps.append(abs(float(cp) - threshold))
    print(f"min |calibrated_prob - threshold| = {min(gaps):.4f}  (rounding-safe if >> 1e-4)")

    tp = fn = tn = fp = 0
    disagree = []
    band_mismatch = []
    for s, y in done:
        rec = cache[s]
        pred = 1 if float(rec["calibrated_prob"]) >= threshold else 0
        pred_api = 1 if float(rec["overall_score"]) >= threshold else 0
        if pred != pred_api:
            disagree.append(s)
        tp += int(pred == 1 and y == 1)
        fn += int(pred == 0 and y == 1)
        tn += int(pred == 0 and y == 0)
        fp += int(pred == 1 and y == 0)
        # band must agree with the artifact edges
        b = rec["confidence_band"]
        expect_band = (
            "DEFINITELY_AI" if float(rec["calibrated_prob"]) >= max(threshold, cal["zone_high"])
            else "LIKELY_AI" if float(rec["calibrated_prob"]) >= min(threshold, cal["zone_high"])
            else "UNCERTAIN" if float(rec["calibrated_prob"]) >= cal["zone_low"]
            else "LIKELY_HUMAN" if float(rec["calibrated_prob"]) >= cal["zone_definitely_human"]
            else "DEFINITELY_HUMAN"
        )
        if b != expect_band:
            band_mismatch.append((s, b, expect_band))

    n_ai = sum(1 for _, y in done if y == 1)
    n_hu = len(done) - n_ai
    ai_recall = tp / n_ai if n_ai else 0.0
    human_recall = tn / n_hu if n_hu else 0.0
    prec_ai = tp / (tp + fp) if tp + fp else 0.0
    prec_h = tn / (tn + fn) if tn + fn else 0.0
    f1_ai = 2 * prec_ai * ai_recall / (prec_ai + ai_recall) if prec_ai + ai_recall else 0.0
    f1_h = 2 * prec_h * human_recall / (prec_h + human_recall) if prec_h + human_recall else 0.0
    got = {
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        "balanced_accuracy": 0.5 * (ai_recall + human_recall),
        "ai_recall": ai_recall,
        "human_recall": human_recall,
        "macro_f1": 0.5 * (f1_ai + f1_h),
    }

    print()
    print(f"tp={tp} fn={fn} tn={tn} fp={fp}")
    print(f"bacc        = {got['balanced_accuracy']*100:.2f}   (required 87.08)")
    print(f"ai_recall   = {got['ai_recall']*100:.2f} ({tp}/{n_ai})   (required 74.17 ({EXPECTED['tp']}/120))")
    print(f"human_recall= {got['human_recall']*100:.2f} ({tn}/{n_hu})   (required 100.00 (60/60))")
    print(f"macro_f1    = {got['macro_f1']:.4f}   (required 0.8232)")
    print(f"api-score-vs-fusion-prob disagreements: {len(disagree)} {disagree[:5]}")
    print(f"confidence-band mismatches: {len(band_mismatch)} {band_mismatch[:5]}")

    checks = {
        "balanced_accuracy": round(got["balanced_accuracy"], 4) == EXPECTED["balanced_accuracy"],
        "ai_recall": round(got["ai_recall"], 4) == EXPECTED["ai_recall"],
        "human_recall": round(got["human_recall"], 4) == EXPECTED["human_recall"],
        "macro_f1": round(got["macro_f1"], 4) == EXPECTED["macro_f1"],
        "tp": tp == EXPECTED["tp"], "fn": fn == EXPECTED["fn"],
        "tn": tn == EXPECTED["tn"], "fp": fp == EXPECTED["fp"],
        "all_samples": len(done) == len(items),
        "bands_consistent": not band_mismatch,
    }
    print()
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print()
    print("STEP 5 " + ("PASS — production path reproduces the gated val verdicts exactly"
                       if ok else "FAIL — production metrics differ from the gated val verdicts"))
    json.dump({"metrics": got, "checks": checks, "n": len(done),
               "threshold": threshold, "disagreements": disagree}, 
              open("text_production_verification.json", "w"), indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
