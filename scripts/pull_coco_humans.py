#!/usr/bin/env python3
"""Source real human photographs from COCO val2017 (content-matched negatives).

Why COCO val2017:
  * real camera/Flickr photography (no AI images, no stock-photo duplicates),
  * category annotations let us content-match the RealVis-XL set across
    portrait / street / nature themes,
  * neither COCO nor its splits are used anywhere in the SLOP gate or the
    34-image multigen AI evaluation set.

Outputs `heldout_set/image/human_v2/` (new directory; the gate stays untouched)
plus an overlap/duplicate audit against all 66 gate images and the 34 multigen AI.

Selection is deterministic (seed 42).
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "heldout_set" / "image" / "human_v2"
TMP_DIR = ROOT / "_coco_tmp"
ANN_ZIP = TMP_DIR / "annotations_trainval2017.zip"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"
SEED = 42
# RealVis-XL is portrait-dominant (CLIP zero-shot themes: portrait=7/10, nature=2,
# object_indoor=1), so the human pool is weighted to match.
TARGETS = {"portrait": 22, "street": 14, "nature": 14}  # 50 total

ANIMALS = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
           "zebra", "giraffe"}
VEHICLES = {"car", "bus", "truck", "motorcycle", "bicycle"}


def download_annotations() -> None:
    TMP_DIR.mkdir(exist_ok=True)
    if not ANN_ZIP.exists():
        print(f"Downloading {ANN_URL} ...")
        with requests.get(ANN_URL, stream=True, timeout=180) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            with open(ANN_ZIP, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done % (20 << 20) < (1 << 20):
                        print(f"  {done/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    print(f"  annotations zip: {ANN_ZIP.stat().st_size/1e6:.1f} MB")
    with zipfile.ZipFile(ANN_ZIP) as z:
        for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
            target = TMP_DIR / Path(name).name
            if not target.exists():
                target.write_bytes(z.read(name))
    for n in ("instances_val2017.json", "captions_val2017.json"):
        p = TMP_DIR / n
        print(f"  extracted {p.name}: {p.stat().st_size/1e6:.1f} MB")


def build_candidates() -> dict[str, list[tuple[int, str]]]:
    inst = json.loads((TMP_DIR / "instances_val2017.json").read_text(encoding="utf-8"))
    caps = json.loads((TMP_DIR / "captions_val2017.json").read_text(encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    meta = {im["id"]: im for im in inst["images"]}
    per_img = defaultdict(list)
    for a in inst["annotations"]:
        per_img[a["image_id"]].append((cat_name.get(a["category_id"], "?"), a.get("bbox", [0, 0, 0, 0])))
    cap_by_img = {}
    for c in caps["annotations"]:
        cap_by_img.setdefault(c["image_id"], c["caption"])

    portrait, street, nature = [], [], []
    for iid, anns in per_img.items():
        m = meta.get(iid)
        if not m:
            continue
        w, h = m.get("width", 0), m.get("height", 0)
        if min(w, h) < 480:
            continue
        names = [n for n, _ in anns]
        areas = {n: max((b[2] * b[3] for nn, b in anns if nn == n), default=0) for n in set(names)}
        n_person = names.count("person")
        cap = cap_by_img.get(iid, "")

        if n_person in (1, 2) and areas.get("person", 0) / (w * h) > 0.25:
            portrait.append((iid, cap))
        elif (set(names) & VEHICLES) and ({"traffic light", "stop sign"} & set(names)):
            street.append((iid, cap))
        elif (set(names) & ANIMALS) and n_person == 0:
            nature.append((iid, cap))
    return {"portrait": sorted(set(portrait)), "street": sorted(set(street)),
            "nature": sorted(set(nature))}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def existing_hashes() -> dict[str, str]:
    out = {}
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for root in (ROOT / "golden_set" / "image", ROOT / "heldout_set" / "image"):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and "human_v2" not in p.parts:
                out[sha256(p)] = str(p.relative_to(ROOT))
    return out


def main() -> int:
    download_annotations()
    cands = build_candidates()
    print("\nCOCO val2017 candidates after content filtering:",
          {k: len(v) for k, v in cands.items()})

    ref = existing_hashes()
    print(f"Reference hash set (all gate + multigen images): {len(ref)} files")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    picked = {}
    new_hashes = {}
    session = requests.Session()

    for theme, target in TARGETS.items():
        pool = list(cands[theme])
        rng.shuffle(pool)
        got = 0
        for iid, cap in pool:
            if got >= target:
                break
            dest = OUT_DIR / f"coco_{theme}_{iid}.jpg"
            if not dest.exists():
                try:
                    r = session.get(IMG_URL.format(iid), timeout=60)
                    r.raise_for_status()
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] {iid}: {e}")
                    continue
                if len(r.content) < 20_000:
                    continue
                dest.write_bytes(r.content)
            h = sha256(dest)
            if h in ref:
                print(f"  [DUPLICATE of {ref[h]}] removing {dest.name}")
                dest.unlink()
                continue
            if h in new_hashes:
                dest.unlink()
                continue
            new_hashes[h] = dest.name
            picked[dest.name] = {"theme": theme, "coco_id": iid, "caption": cap,
                                 "sha256": h, "bytes": dest.stat().st_size}
            got += 1
        print(f"  {theme:9s} sourced {got}/{target}")

    files = sorted(OUT_DIR.glob("*.jpg"))
    print(f"\nFINAL: {len(files)} images in {OUT_DIR.relative_to(ROOT)}")

    overlap = [h for h in new_hashes if h in ref]
    print(f"SHA-256 overlap vs 66 gate + 34 multigen AI ({len(ref)} files): {len(overlap)} (must be 0)")
    print(f"Internal duplicates: {len(files) - len(new_hashes)} (must be 0)")

    (ROOT / "human_v2_manifest.json").write_text(json.dumps(
        {"seed": SEED, "source": "COCO val2017", "n": len(files),
         "overlap_count": len(overlap), "reference_files": len(ref),
         "images": picked}, indent=2), encoding="utf-8")
    print("Wrote human_v2_manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
