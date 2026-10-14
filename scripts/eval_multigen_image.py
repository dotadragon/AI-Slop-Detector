#!/usr/bin/env python3
"""Multi-generator image evasion evaluation (fresh harness).

The original one-off script that produced `multigen_eval_results.json` is not in
the working tree, so this harness reproduces its methodology exactly:

    * iterate every AI family under heldout_set/image/ai/<family>/
    * run the real, zero-mock production image pipeline `_analyze_pil_image`
    * verdict = AI when the fused overall_score >= 0.50
    * report per-family n / detected_as_ai / detection_rate / mean_ai_score / scores

Read-only with respect to Model/ and the pristine 66-image gate set.

Usage:
    ENVIRONMENT=development SECRET_KEY=... python eval_multigen_image.py \
        [--output multigen_eval_results_today.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

from app.ml.image_detector import _analyze_pil_image  # noqa: E402

AI_ROOT = ROOT / "heldout_set" / "image" / "ai"
THRESHOLD = 0.50

# Canonical "multigen" set = the families that are NOT part of the pristine
# 66-image regression gate. sdxl_proxy IS in the gate (27 of the 66) so it is
# reported separately as an in-gate control and excluded from the multigen total.
MULTIGEN_FAMILIES = ["cascade_lcm", "dit", "kandinsky", "midjourney", "realvis_xl"]
IN_GATE_FAMILIES = ["sdxl_proxy"]


def evaluate() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for family_dir in sorted(p for p in AI_ROOT.iterdir() if p.is_dir()):
        files = sorted(
            p for p in family_dir.glob("*")
            if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
        )
        scores: list[float] = []
        for p in files:
            with Image.open(p) as img:
                rgb = img.convert("RGB")
            res = _analyze_pil_image(rgb, p.name, str(p))
            scores.append(round(float(res["overall_score"]), 4))
        detected = sum(1 for s in scores if s >= THRESHOLD)
        out[family_dir.name] = {
            "n": len(scores),
            "detected_as_ai": detected,
            "detection_rate": round(detected / len(scores), 4) if scores else 0.0,
            "mean_ai_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "scores": scores,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "multigen_eval_results_today.json")
    args = ap.parse_args()

    print(f"Multi-generator image eval @ threshold {THRESHOLD} (AI root: {AI_ROOT})")
    print("=" * 78)
    results = evaluate()

    print(f"{'family':<14}{'n':>4}{'detected':>10}{'rate':>9}{'mean_score':>12}")
    print("-" * 78)
    for fam, v in results.items():
        tag = "  (in-gate control)" if fam in IN_GATE_FAMILIES else ""
        print(f"{fam:<14}{v['n']:>4}{v['detected_as_ai']:>10}"
              f"{v['detection_rate']:>9.2%}{v['mean_ai_score']:>12.4f}{tag}")
    print("-" * 78)

    total_n = sum(results[f]["n"] for f in MULTIGEN_FAMILIES)
    total_det = sum(results[f]["detected_as_ai"] for f in MULTIGEN_FAMILIES)
    overall = total_det / total_n if total_n else 0.0
    print(f"{'MULTIGEN':<14}{total_n:>4}{total_det:>10}{overall:>9.2%}{'':>12}")
    print("=" * 78)
    print(f"OVERALL multigen recall @0.50 (5 non-gate families): {total_det}/{total_n} = {overall:.2%}")

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
