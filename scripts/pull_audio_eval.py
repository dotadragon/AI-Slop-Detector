"""Pull the first audio eval set (Phase D11 data discipline).

Source: Hemg/Deepfake-Audio-Dataset — 100 ten-second WAVs, exactly 50 Fake
/ 50 Real. Small, balanced, directly downloadable via the HF datasets-server
cached assets. All 100 rows become the audio held-out eval set.

Usage: python pull_audio_eval.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

DATASET = "Hemg/Deepfake-Audio-Dataset"
OUT = Path("heldout_set/audio")
PAGE = 20


def fetch(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "slop-eval/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        print(f"  FAIL {dest.name}: {exc}", flush=True)
        return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    offset = 0
    total = fetched = 0
    while True:
        url = f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(DATASET, safe='/')}&config=default&split=train&offset={offset}&length={PAGE}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            i = r["row_idx"]
            label = r["row"]["label"]  # "Fake" | "Real"
            src = r["row"]["audio"][0]["src"]
            split = "ai" if label == "Fake" else "human"
            d = OUT / split
            d.mkdir(parents=True, exist_ok=True)
            dest = d / f"{DATASET.split('/')[-1]}_{i}.wav"
            total += 1
            if dest.exists() and str(i) in manifest.get(split, {}):
                continue
            if fetch(src, dest):
                manifest.setdefault(split, {})[str(i)] = {"file": dest.name, "dataset": DATASET, "label": label}
                fetched += 1
                time.sleep(0.2)
        offset += PAGE
        if len(rows) < PAGE:
            break

    manifest_path.write_text(json.dumps(manifest, indent=2))
    n_ai = len(manifest.get("ai", {}))
    n_hu = len(manifest.get("human", {}))
    print(f"Fetched {fetched}/{total} new. Set: ai={n_ai} human={n_hu}")
    sys.exit(0 if n_ai >= 40 and n_hu >= 40 else 1)


if __name__ == "__main__":
    import urllib.parse

    main()
