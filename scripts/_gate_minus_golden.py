#!/usr/bin/env python3
"""Gate-minus-golden measurement (read-only).

Scores the 52 fully-held-out gate images (excludes the 14 golden_set/image files
that the shipped image.pkl / clip_probe.pkl were fit on, n_samples=14) with the
real, zero-mock production image pipeline, and reports the accuracy delta against
the full-66 number (55/66 = 83.33%).

Nothing is written to Model/ or app code.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

from app.ml.image_detector import _analyze_pil_image  # noqa: E402

HELDOUT = [
    ("ai/sdxl_proxy", ROOT / "heldout_set" / "image" / "ai" / "sdxl_proxy", "AI"),
    ("human/picsum_unsplash", ROOT / "heldout_set" / "image" / "human" / "picsum_unsplash", "HUMAN"),
]
GOLDEN = [
    ("golden/ai", ROOT / "golden_set" / "image" / "ai", "AI"),
    ("golden/diagram", ROOT / "golden_set" / "image" / "diagram", "HUMAN"),
    ("golden/human", ROOT / "golden_set" / "image" / "human", "HUMAN"),
]


def score(datasets):
    rows = []
    for name, d, truth in datasets:
        for p in sorted(d.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            with Image.open(p) as img:
                rgb = img.convert("RGB")
            res = _analyze_pil_image(rgb, p.name, str(p))
            s = float(res["overall_score"])
            rows.append({"dataset": name, "file": p.name, "truth": truth,
                         "score": round(s, 3), "verdict": "AI" if s >= 0.50 else "HUMAN"})
    return rows


def summarize(rows, label):
    cor = sum(1 for r in rows if r["verdict"] == r["truth"])
    print(f"{label}: {cor}/{len(rows)} = {cor/len(rows):.2%}")
    for name in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == name]
        c = sum(1 for r in sub if r["verdict"] == r["truth"])
        fp = sum(1 for r in sub if r["truth"] == "HUMAN" and r["verdict"] == "AI")
        fn = sum(1 for r in sub if r["truth"] == "AI" and r["verdict"] == "HUMAN")
        print(f"    {name:24s} {c}/{len(sub)}   FP={fp} FN={fn}")
    return cor, len(rows)


print("Scoring the 52 non-golden gate images (fully held-out) live...")
h_rows = score(HELDOUT)
print("Scoring the 14 golden gate images live (for the delta breakdown)...")
g_rows = score(GOLDEN)

print()
print("=" * 72)
hc, hn = summarize(h_rows, "GATE-MINUS-GOLDEN (52 fully held-out)")
gc, gn = summarize(g_rows, "GOLDEN subset (14, in-sample for shipped artifacts)")
print("-" * 72)
ac, an = summarize(h_rows + g_rows, "FULL 66-GATE (live)")
print("=" * 72)
print(f"delta (52 vs 66) = {hc/hn*100:.2f}% - {ac/an*100:.2f}% = {(hc/hn - ac/an)*100:+.2f} pts")
print(f"delta (52 vs golden-14) = {hc/hn*100:.2f}% - {gc/gn*100:.2f}% = {(hc/hn - gc/gn)*100:+.2f} pts")

(ROOT / "gate_minus_golden_today.json").write_text(json.dumps(
    {"gate52": {"correct": hc, "n": hn, "acc": hc / hn},
     "golden14": {"correct": gc, "n": gn, "acc": gc / gn},
     "full66": {"correct": ac, "n": an, "acc": ac / an},
     "rows": h_rows + g_rows}, indent=2), encoding="utf-8")
print("Wrote gate_minus_golden_today.json")
