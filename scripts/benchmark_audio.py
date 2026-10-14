"""Benchmark the audio detector on heldout_set/audio (Phase D11 data discipline).

First real audio eval numbers: AUC + balanced accuracy on 50 Fake / 50 Real
10-second WAVs (Hemg/Deepfake-Audio-Dataset). Resumable via per-file cache.

Usage: python benchmark_audio.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "slop/backend")

CACHE = Path("audio_benchmark_cache.json")


def main() -> None:
    from app.ml.audio_detector import analyze_audio, evict_audio_model

    root = Path("heldout_set/audio")
    manifest = json.loads((root / "manifest.json").read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    rows = []  # (file, label_int) label 1 = AI
    for split in ("ai", "human"):
        for meta in manifest.get(split, {}).values():
            rows.append((str(root / meta["file"]), 1 if split == "ai" else 0))

    done = 0
    for fpath, label in rows:
        if fpath in cache:
            continue
        try:
            r = analyze_audio(fpath, "bench")
            cache[fpath] = {"score": r["overall_score"], "label": label}
        except Exception as exc:
            cache[fpath] = {"score": None, "label": label, "error": str(exc)[:120]}
        done += 1
        if done % 10 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  {sum(1 for v in cache.values() if v['score'] is not None)}/{len(rows)} scored", flush=True)
    CACHE.write_text(json.dumps(cache))
    evict_audio_model()

    valid = [(v["score"], v["label"]) for v in cache.values() if v["score"] is not None]
    n_err = len(cache) - len(valid)
    if len({lbl for _, lbl in valid}) < 2:
        print("Insufficient valid scores:", len(valid), "errors:", n_err)
        sys.exit(1)

    # AUC: exact pairwise (handles ties correctly; n=100 → 2500 pairs)
    pos = [s for s, l in valid if l == 1]
    neg = [s for s, l in valid if l == 0]
    wins = ties = 0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                ties += 1
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg))

    # Balanced accuracy at 0.5
    tp = sum(1 for s, l in valid if l == 1 and s >= 0.5)
    fn = len(pos) - tp
    tn = sum(1 for s, l in valid if l == 0 and s < 0.5)
    fp = len(neg) - tn
    tpr = tp / max(len(pos), 1)
    tnr = tn / max(len(neg), 1)
    bacc = (tpr + tnr) / 2

    result = {
        "n": len(valid), "errors": n_err,
        "ai_positive": len(pos), "human_negative": len(neg),
        "auc": round(auc, 4),
        "threshold": 0.5, "tpr": round(tpr, 4), "tnr": round(tnr, 4),
        "balanced_accuracy": round(bacc, 4),
    }
    Path("audio_benchmark_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
