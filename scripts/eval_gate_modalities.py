"""Modality gate extension (Phase E16).

Complements `eval_gate.py` (text-only): re-evaluates the image and video
held-out splits with the current production detectors and blocks on a
balanced-accuracy regression beyond tolerance vs the recorded baselines in
`Model/fusion/promoted_baseline_modalities.json`.

The image/video "weights" are code paths (detectors + fixed fusion), so this
gate catches verdict regressions caused by detector or fusion code changes —
the image/video analogue of the text weight gate.

Usage:
  python eval_gate_modalities.py                 # gate check (exit 1 on regression)
  python eval_gate_modalities.py --update        # after intentional changes, refresh baseline
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(root / "slop" / "backend"))
sys.path.insert(0, str(root))

from scripts.eval_rigorous import discover_stratified, run_image, run_video, summarize  # noqa: E402

BASELINE = root / "Model" / "fusion" / "promoted_baseline_modalities.json"
TOLERANCE = 0.02


def modality_metrics(modality: str) -> dict:
    """Evaluate a modality's held-out split; mirrors eval_rigorous's core flow."""
    items, cal_items = discover_stratified(modality, min_per_class=3, calibration_fraction=0.2)
    if not items:
        return {"balanced_accuracy": None, "n": 0}
    if modality == "image":
        samples = run_image(items)
    else:
        samples = run_video(items)
    summary = summarize(samples, modality)
    tn, fp, fn, tp = summary["tn"], summary["fp"], summary["fn"], summary["tp"]
    denom = (tn + tp) or 1
    summary["balanced_accuracy"] = round((tn / denom + tp / denom) / 2, 4)
    return summary


def main() -> int:
    if "--update" in sys.argv:
        out = {}
        for mod in ("image", "video"):
            m = modality_metrics(mod)
            out[mod] = {"balanced_accuracy": m.get("balanced_accuracy"),
                        "accuracy_pct": m.get("accuracy_pct"), "n": m.get("total")}
            print(f"{mod}: bacc={m.get('balanced_accuracy')} n={m.get('total')}")
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(out, indent=1))
        print("modality baselines updated")
        return 0

    if not BASELINE.exists():
        print("NO MODALITY BASELINE — run `python eval_gate_modalities.py --update` after reviewing current numbers.")
        return 1

    base = json.loads(BASELINE.read_text())
    failed = False
    for mod in ("image", "video"):
        if mod not in base or base[mod].get("balanced_accuracy") is None:
            print(f"{mod}: no baseline recorded — skipping")
            continue
        cur = modality_metrics(mod)
        cur_bacc = cur.get("balanced_accuracy")
        base_bacc = base[mod]["balanced_accuracy"]
        print(f"{mod}: baseline bacc={base_bacc}  current bacc={cur_bacc}  n={cur.get('total')}")
        if cur_bacc is None or cur_bacc < base_bacc - TOLERANCE:
            print(f"EVAL GATE FAILED for {mod}: regression beyond tolerance. Merge blocked.")
            failed = True
    if failed:
        return 1
    print("MODALITY GATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
