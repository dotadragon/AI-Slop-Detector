"""Dev-split calibration probe for Binoculars and Fast-DetectGPT (read-only).

Loads the dev calibration partition (30% stratified) without touching the eval
partition or test split. Computes raw zero-shot signals, measures AUCs (raw vs negated),
and fits 1D class-balanced logistic calibration parameters (a, b):
    P(AI | x) = 1 / (1 + exp(-(a * x + b)))
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")
os.environ.setdefault("SLOP_HELDOUT_DIR", r"c:\Users\karth\Desktop\AISlop_heldout_set")


def main() -> int:
    from scripts.eval_rigorous import discover_stratified
    from app.ml.binoculars import compute_binoculars_and_curvature
    from app.ml.text_detector import _normalize_and_sanitize_text

    print("=" * 70)
    print("BINOCULARS & FAST-DETECTGPT DEV-SPLIT CALIBRATION PROBE")
    print("=" * 70)

    _, cal_items = discover_stratified(
        "text", min_per_class=1, calibration_fraction=0.3, split="dev"
    )
    print(f"Loaded {len(cal_items)} calibration samples from dev split.")

    records = []
    for idx, (payload, truth, subtype, name) in enumerate(cal_items, 1):
        clean = _normalize_and_sanitize_text(payload)
        res = compute_binoculars_and_curvature(clean)

        bino_s = res.get("binoculars_score")
        fast_z = res.get("fast_detectgpt_z")
        records.append({
            "name": name,
            "subtype": subtype,
            "truth": int(truth),
            "bino_score": float(bino_s) if bino_s is not None else float("nan"),
            "fast_z": float(fast_z) if fast_z is not None else float("nan"),
        })
        print(f"  [{idx:2d}/{len(cal_items):2d}] {subtype:12s} truth={truth} "
              f"bino_score={bino_s:.4f} fast_z={fast_z:.4f}")

    # Extract valid arrays
    y = np.array([r["truth"] for r in records], dtype=int)
    bino = np.array([r["bino_score"] for r in records], dtype=np.float64)
    fast = np.array([r["fast_z"] for r in records], dtype=np.float64)

    valid_bino = ~np.isnan(bino)
    valid_fast = ~np.isnan(fast)

    print("\n" + "=" * 70)
    print("PAIRWISE AUC ANALYSIS")
    print("=" * 70)

    if np.sum(valid_bino) > 0:
        auc_bino_raw = roc_auc_score(y[valid_bino], bino[valid_bino])
        auc_bino_neg = roc_auc_score(y[valid_bino], -bino[valid_bino])
        print(f"Binoculars raw score AUC:     {auc_bino_raw:.4f}")
        print(f"Binoculars negated score AUC: {auc_bino_neg:.4f}  (lower bino => more AI)")

    if np.sum(valid_fast) > 0:
        auc_fast_raw = roc_auc_score(y[valid_fast], fast[valid_fast])
        auc_fast_neg = roc_auc_score(y[valid_fast], -fast[valid_fast])
        print(f"Fast-DetectGPT raw z AUC:     {auc_fast_raw:.4f}  (higher z => more AI)")
        print(f"Fast-DetectGPT neg z AUC:     {auc_fast_neg:.4f}")

    print("\n" + "=" * 70)
    print("1D LOGISTIC FIT (Class-Balanced Platt Calibration)")
    print("=" * 70)

    # Fit 1D logistic on bino
    lr_bino = LogisticRegression(class_weight="balanced", solver="lbfgs")
    X_bino = bino[valid_bino].reshape(-1, 1)
    lr_bino.fit(X_bino, y[valid_bino])
    a_bino = float(lr_bino.coef_[0, 0])
    b_bino = float(lr_bino.intercept_[0])
    print(f"Binoculars 1D Logistic: a_bino = {a_bino:+.4f}, b_bino = {b_bino:+.4f}")
    print(f"  Formula: P(AI) = sigmoid({a_bino:+.4f} * bino_score + {b_bino:+.4f})")

    # Fit 1D logistic on fast_z
    lr_fast = LogisticRegression(class_weight="balanced", solver="lbfgs")
    X_fast = fast[valid_fast].reshape(-1, 1)
    lr_fast.fit(X_fast, y[valid_fast])
    a_fast = float(lr_fast.coef_[0, 0])
    b_fast = float(lr_fast.intercept_[0])
    print(f"Fast-DetectGPT 1D Logistic: a_fast = {a_fast:+.4f}, b_fast = {b_fast:+.4f}")
    print(f"  Formula: P(AI) = sigmoid({a_fast:+.4f} * fast_z + {b_fast:+.4f})")

    print("\n" + "=" * 70)
    print("DISTRIBUTION ON CALIBRATION SAMPLES")
    print("=" * 70)
    for r in records:
        b_p = 1.0 / (1.0 + np.exp(-(a_bino * r["bino_score"] + b_bino)))
        f_p = 1.0 / (1.0 + np.exp(-(a_fast * r["fast_z"] + b_fast)))
        print(f"  {r['subtype']:12s} truth={r['truth']} | "
              f"raw_bino={r['bino_score']:.3f} -> P(AI)={b_p:.3f} | "
              f"raw_fast_z={r['fast_z']:+.3f} -> P(AI)={f_p:.3f}")

    print("\nProbe complete (read-only, no artifacts overwritten).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
