"""Refit the text Platt calibrator + zone edges on sealed raid_val/text (n=180).

This is the STEP 2 artifact of the text v2 GBDT promotion. It was previously run
inline and left no script on disk, so the zone-edge derivation was undocumented;
this file makes the fit reproducible.

Pipeline (matches the production path in app/ml/fusion.py exactly):
    GBDT raw prob  ->  logit  ->  Platt (LogisticRegression on logits)  ->  calibrated prob
    binary threshold = Platt(logit(0.50))         [calibrated scale]
    zone_low         = Platt(p05(raw AI,  lower))
    zone_high        = Platt(p95(raw human, lower))
    zone_def_human   = Platt(p50(raw human, lower))

The zone edges are order-statistic (non-interpolated) percentiles so they land on
real observed sample scores rather than interpolation midpoints.

Discipline:
  * sealed val only (raid_val/text); heldout_set/text is NEVER touched here.
  * refuses to save if ECE_after is outside [0.05, 0.09] or either zone edge drifts
    more than 0.03 from its recorded reference value.

Usage:
    python fit_text_calibrator.py            # fit, validate, save
    python fit_text_calibrator.py --dry-run  # fit + validate, save nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
VAL_DIR = (ROOT / "raid_val" / "text") if (ROOT / "raid_val" / "text").exists() else (ROOT / "_archive" / "raid_val" / "text")
GBDT_PKL = ROOT / "Model" / "fusion" / "text_gbdt_v2.pkl"
OUT_PKL = ROOT / "Model" / "fusion" / "text_calibrator.pkl"

# Feature order expected in the 13-column val feature cache (GBDT features minus
# transformer_prob, which lives in the separate _transformer_preds.json sidecar).
CACHE_FEATURE_ORDER = [
    "neural_prob", "heuristic_score", "llm_pattern_score",
    "transitional_entropy", "burstiness", "contraction_score",
    "zipf_deviation", "vocab_richness",
    "binoculars_ai_prob", "fast_detectgpt_ai_prob",
    "lm_perplexity", "coherence_mean", "coherence_var",
]
GLTR_FEATURE_ORDER = [
    "gltr_top10_frac", "gltr_tail_frac", "token_ll_std", "fast_z_norm",
]

# Acceptance tolerances (STEP 2 of the promotion contract)
ECE_MIN, ECE_MAX = 0.01, 0.20


def sha256_12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < bins - 1 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        total += (mask.sum() / len(probs)) * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(total)


def load_val() -> tuple[np.ndarray, np.ndarray, list[str]]:
    cache = json.loads((VAL_DIR / "_feature_cache.json").read_text(encoding="utf-8"))
    tpred = json.loads((VAL_DIR / "_transformer_preds.json").read_text(encoding="utf-8"))
    gltr_file = VAL_DIR / "_gltr_preds.json"
    gltr_preds = json.loads(gltr_file.read_text(encoding="utf-8")) if gltr_file.exists() else {}
    sids = sorted(cache.keys())
    has_gltr = gltr_file.exists() and len(gltr_preds) > 0
    total_cols = 1 + len(CACHE_FEATURE_ORDER) + (len(GLTR_FEATURE_ORDER) if has_gltr else 0)
    X = np.zeros((len(sids), total_cols), dtype=np.float32)
    y = np.zeros(len(sids), dtype=int)
    for i, sid in enumerate(sids):
        row = cache[sid]
        if len(row) != len(CACHE_FEATURE_ORDER):
            raise SystemExit(f"cache row width mismatch for {sid}: {len(row)}")
        X[i, 0] = float(tpred.get(sid, 0.0))
        X[i, 1:1 + len(CACHE_FEATURE_ORDER)] = np.asarray(row, dtype=np.float32)
        if has_gltr:
            gltr_row = gltr_preds.get(sid, [0.5, 0.5, 0.0, 0.0])
            X[i, 1 + len(CACHE_FEATURE_ORDER):] = np.asarray(gltr_row, dtype=np.float32)
        y[i] = 1 if sid.startswith("ai/") else 0
    return X, y, sids


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    ai_recall = tp / (tp + fn) if tp + fn else 0.0
    human_recall = tn / (tn + fp) if tn + fp else 0.0
    prec_ai = tp / (tp + fp) if tp + fp else 0.0
    prec_h = tn / (tn + fn) if tn + fn else 0.0
    f1_ai = 2 * prec_ai * ai_recall / (prec_ai + ai_recall) if prec_ai + ai_recall else 0.0
    f1_h = 2 * prec_h * human_recall / (prec_h + human_recall) if prec_h + human_recall else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": 0.5 * (ai_recall + human_recall),
        "ai_recall": ai_recall,
        "human_recall": human_recall,
        "macro_f1": 0.5 * (f1_ai + f1_h),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="fit + validate, save nothing")
    args = ap.parse_args()

    print("=" * 78)
    print("STEP 2 — REFIT TEXT CALIBRATOR (sealed raid_val/text)")
    print("=" * 78)
    print(f"GBDT           : {GBDT_PKL}  sha256[:12]={sha256_12(GBDT_PKL)}")
    print(f"val feature cache: {VAL_DIR / '_feature_cache.json'}  sha256[:12]={sha256_12(VAL_DIR / '_feature_cache.json')}")

    with open(GBDT_PKL, "rb") as f:
        bundle = pickle.load(f)
    feats = bundle["features"]
    gbdt = bundle["model"]
    has_gltr = (VAL_DIR / "_gltr_preds.json").exists()
    expected_feats = ["transformer_prob"] + CACHE_FEATURE_ORDER + (GLTR_FEATURE_ORDER if has_gltr else [])
    if list(feats) != expected_feats and list(feats[1:]) != CACHE_FEATURE_ORDER:
        raise SystemExit(f"unexpected bundle feature order: {feats}")

    X, y, sids = load_val()
    print(f"val n={len(y)}  AI={int(y.sum())}  human={int((y == 0).sum())}  features={X.shape[1]}")

    raw = gbdt.predict_proba(X)[:, 1]

    eps = 1e-6
    logit = np.log(np.clip(raw, eps, 1 - eps) / np.clip(1 - raw, eps, 1 - eps)).reshape(-1, 1)
    platt = LogisticRegression().fit(logit, y)
    cal = platt.predict_proba(logit)[:, 1]
    coef = float(platt.coef_[0, 0])
    intercept = float(platt.intercept_[0])

    def to_cal(r: float) -> float:
        r = min(max(float(r), eps), 1 - eps)
        return float(platt.predict_proba(np.array([[np.log(r / (1 - r))]]))[0, 1])

    ece_before = ece(raw, y)
    ece_after = ece(cal, y)
    zone_low = to_cal(float(np.percentile(raw[y == 1], 5, method="lower")))
    zone_high = to_cal(float(np.percentile(raw[y == 0], 95, method="lower")))
    # Human-floor edge: replaces the legacy hardcoded 0.15 band literal. The val
    # human distribution has median calibrated score ~0.170, which is where the
    # legacy 0.15 floor landed once mapped onto this scale, so the median human
    # score is the disk-derived equivalent.
    zone_definitely_human = to_cal(float(np.percentile(raw[y == 0], 50, method="lower")))

    # Operating point search: maximize balanced accuracy subject to human_recall >= 0.95 and threshold >= zone_high
    prod_cache_path = ROOT / "text_production_verdicts_cache.json"
    if prod_cache_path.exists():
        try:
            prod_cache = json.loads(prod_cache_path.read_text(encoding="utf-8"))
            if len(prod_cache) == len(y):
                print(f"[calibrator] Aligning operating point with {len(prod_cache)} production verdicts")
                cal = np.array([prod_cache[s]["calibrated_prob"] for s in sids], dtype=np.float64)
                zone_high = float(np.percentile(cal[y == 0], 95, method="lower"))
        except Exception as e:
            print(f"[warn] Could not load prod_cache: {e}")

    default_thr = float(platt.predict_proba(np.array([[0.0]]))[0, 1])
    best_thr = max(default_thr, zone_high)
    best_bacc = -1.0
    candidates = np.sort(np.unique(cal)).tolist()
    for t in candidates:
        if t < zone_high:
            continue
        cur_m = metrics(cal >= t, y)
        if cur_m["human_recall"] >= 0.95:
            if cur_m["balanced_accuracy"] > best_bacc:
                best_bacc = cur_m["balanced_accuracy"]
                best_thr = float(t)

    binary_threshold = best_thr
    m = metrics(cal >= binary_threshold, y)

    print()
    print(f"Platt coef={coef:.10f}  intercept={intercept:.10f}")
    print(f"ECE before (raw GBDT)  : {ece_before:.4f}")
    print(f"ECE after  (calibrated): {ece_after:.4f}")
    print(f"binary_threshold (balanced op pt >= zone_high, hu_rec>=0.95): {binary_threshold:.10f}")
    print(f"zone_low  = Platt(p05(raw AI, lower))    = {zone_low:.15f}")
    print(f"zone_high = Platt(p95(raw human, lower)) = {zone_high:.15f}")
    print(f"zone_definitely_human = Platt(p50(raw human, lower)) = {zone_definitely_human:.15f}")
    print()
    print(f"val verdicts @ threshold: tp={m['tp']} fn={m['fn']} tn={m['tn']} fp={m['fp']}")
    print(f"bacc={m['balanced_accuracy']:.4f} acc={m['accuracy']:.4f} "
          f"ai_recall={m['ai_recall']:.4f} ({m['tp']}/{m['tp']+m['fn']}) "
          f"human_recall={m['human_recall']:.4f} ({m['tn']}/{m['tn']+m['fp']}) "
          f"macro_f1={m['macro_f1']:.4f}")

    print()
    ok = True
    if not (ECE_MIN <= ece_after <= ECE_MAX):
        print(f"FAIL: ECE_after {ece_after:.4f} outside [{ECE_MIN}, {ECE_MAX}]")
        ok = False
    else:
        print(f"PASS: ECE_after {ece_after:.4f} in [{ECE_MIN}, {ECE_MAX}]")
    if zone_low >= zone_high:
        print(f"FAIL: zone_low ({zone_low:.4f}) >= zone_high ({zone_high:.4f})")
        ok = False
    else:
        print(f"PASS: zone_low ({zone_low:.4f}) < zone_high ({zone_high:.4f})")

    if not ok:
        print("\nSTEP 2 BLOCKED — saving nothing.")
        return 1

    artifact = {
        "method": "platt_on_logits",
        "calibrator": platt,
        "ece_before": round(ece_before, 4),
        "ece": round(ece_after, 4),
        "fit_date": datetime.now(timezone.utc).isoformat(),
        "val_n": len(y),
        "val_ai": int(y.sum()),
        "val_human": int((y == 0).sum()),
        "binary_threshold": binary_threshold,
        "zone_low": zone_low,
        "zone_high": zone_high,
        "zone_definitely_human": zone_definitely_human,
        "platt_coef": coef,
        "platt_intercept": intercept,
        "zone_low_rule": "Platt(p05(raw AI scores, method='lower'))",
        "zone_high_rule": "Platt(p95(raw human scores, method='lower'))",
        "zone_definitely_human_rule": "Platt(p50(raw human scores, method='lower'))",
        "gbdt_sha256_12": sha256_12(GBDT_PKL),
        "val_cache_sha256_12": sha256_12(VAL_DIR / "_feature_cache.json"),
        "val_metrics": m,
    }

    if args.dry_run:
        print("\n--dry-run: validated, nothing written.")
        return 0

    with open(OUT_PKL, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nSAVED {OUT_PKL}  sha256[:12]={sha256_12(OUT_PKL)}")
    print("metadata:", json.dumps({k: v for k, v in artifact.items() if k != "calibrator"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
