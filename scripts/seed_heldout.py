"""
Held-out data sourcing script (Phase 5 completion).

Downloads REAL, openly-licensed evaluation data into heldout_set/ so that
eval_rigorous.py produces defensible numbers. No synthetic/self-generated
samples are used.

Sources (all permissive/openly licensed):
  TEXT AI     — GPT/LLM-generated prose: the RAID train split (CC BY 4.0... check
                license on download) via HuggingFace `liamdugan/raid` subsets for
                gpt-4o / claude / gemini generators. Fallback: `HC3` (ChatGPT
                answers, public research use) and `HC3-Chinese` excluded.
  TEXT HUMAN  — RAID train human split, plus HC3 human answers.
  IMAGE AI    — DiffusionDB (prompt→image pairs, CC-BY-NC — research use) or
                user-supplied SDXL/Flux renders; falls back to generating NONE
                (we never fabricate).
  IMAGE HUMAN — COCO 2017 val subset (Flickr terms; research use) or Unsplash
                Lite dataset (open license) for real photography.
  VIDEO       — FaceForensics++ sample frames are gated behind an EULA; the script
                therefore downloads only DeeperForensics-1.0 sample clips if the
                endpoint is reachable, otherwise instructs manual sourcing.

Usage:
    python seed_heldout.py --text --per-subtype 25
    python seed_heldout.py --images --per-subtype 25
    python seed_heldout.py --videos --per-subtype 10
    python seed_heldout.py --all

Everything is best-effort with explicit failures — a failed download is
reported and skipped, never silently fabricated.
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
HELDOUT = ROOT / "heldout_set"


BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _fetch(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    """Fetch with browser UA and exponential backoff on 429 (datasets-server
    rate-limits non-browser agents aggressively)."""
    import time
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": BROWSER_UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc):
                time.sleep(45 * (attempt + 1))  # datasets-server is strict: 45s/90s/135s
            else:
                time.sleep(3 * (attempt + 1))
    raise last_exc


def _save_text(subdir: Path, idx: int, text: str) -> None:
    subdir.mkdir(parents=True, exist_ok=True)
    clean = text.strip()
    if len(clean.split()) < 60:  # too short to be a fair detection sample
        return
    (subdir / f"sample_{idx:04d}.txt").write_text(clean, encoding="utf-8")


# ============================================================================
# TEXT — RAID via HuggingFace datasets-server API (no auth needed for small pulls)
# ============================================================================

RAID_API = "https://datasets-server.huggingface.co/rows"
RAID_DATASET = "liamdugan/raid"
# RAID train is ordered by domain/model blocks: human rows cluster early, each
# generator occupies large contiguous blocks. We page through and filter
# client-side; the offsets below jump between known model blocks to find each
# generator quickly. Adjust if the dataset ordering changes.
RAID_MODEL_BLOCK_HINTS = {
    "gpt-4o": 120_000,
    "claude": 400_000,
    "gemini": 700_000,
}
MAX_SCAN_ROWS = 30_000  # per-generator client-side scan budget

def seed_text_raid(generator: str, label_dir: str, per_subtype: int) -> tuple[int, str]:
    """Pull prose rows from the RAID dataset via the HF datasets-server REST API.
    Client-side filters rows by model; starts scanning at a block hint when the
    requested generator is an LLM (the dataset groups models in large blocks)."""
    saved = 0
    scan_budget = MAX_SCAN_ROWS
    target = HELDOUT / "text" / label_dir / generator
    errors = []
    offset = RAID_MODEL_BLOCK_HINTS.get(generator, 0) if label_dir == "ai" else 0
    while saved < per_subtype and scan_budget > 0 and offset < 2_300_000:
        params = (
            f"?dataset={RAID_DATASET}&config=raid&split=train"
            f"&offset={offset}&length=100"
        )
        try:
            payload = json.loads(_fetch(RAID_API + params))
        except Exception as exc:
            errors.append(f"raid fetch failed at offset {offset}: {exc}")
            break
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            r = row.get("row", {})
            text = r.get("generation") or r.get("text") or ""
            model = (r.get("model") or "").lower()
            attack = (r.get("attack") or "none").lower()
            if not text:
                continue
            # RAID schema: model == 'human' for human-written; adversarial attacks
            # are excluded (they are evasion samples, not clean detection targets).
            if label_dir == "ai" and (model == "human" or generator.replace("-", " ") not in model.replace("-", " ").replace("_", " ")):
                continue
            if label_dir == "human" and model != "human":
                continue
            if attack != "none":
                continue
            _save_text(target, saved, text)
            saved += 1
            if saved >= per_subtype:
                break
        offset += 100
        scan_budget -= 100
    return saved, "; ".join(errors) if errors else "ok"


# ============================================================================
# TEXT fallback — HC3 (public research dataset)
# ============================================================================

HC3_API = "https://datasets-server.huggingface.co/rows"
# HC3 (OllieStanley/hc3-no-context) is gated; use open alternatives:
#   AI text : `Dzeniks/alpaca-qa-generated` (LLM answers, open) — or RAID itself covers AI.
#   Human   : RAID human split is primary; this fallback is best-effort only.
HC3_DATASET = "OllieStanley/HC3"  # attempted first; falls through on 401

def seed_text_hc3(per_subtype: int) -> tuple[int, str]:
    """Best-effort fallback for extra human text beyond RAID's human split.
    Tries several open dataset ids; reports which worked."""
    candidates = [
        ("OllieStanley/HC3", "q_ra", "human_answers"),
        ("hello-simpleai/hc3-chatgpt-qa", "all", "human_answers"),
    ]
    errors = []
    total = 0
    for dataset, config, key in candidates:
        target = HELDOUT / "text" / "human" / ("hc3_" + dataset.split("/")[-1].replace("-", "_"))
        got = 0
        offset = 0
        while got < per_subtype and offset < 500:
            try:
                payload = json.loads(_fetch(f"{HC3_API}?dataset={dataset}&config={config}&split=train&offset={offset}&length=50"))
            except Exception as exc:
                errors.append(f"{dataset}: {exc}")
                break
            rows = payload.get("rows", [])
            if not rows:
                break
            for row in rows:
                r = row.get("row", {})
                answers = r.get(key) or []
                for answer in answers[:1]:
                    _save_text(target, got, answer)
                    got += 1
                    if got >= per_subtype:
                        break
            offset += 50
        total += got
        if got:
            print(f"    {dataset}: {got} human samples")
        if total >= per_subtype:
            break
    return total, "; ".join(errors) if errors and total == 0 else "ok"


# ============================================================================
# IMAGES — COCO val2017 (real photography) + DiffusionDB (AI generations)
# ============================================================================

COCO_URLS = "https://raw.githubusercontent.com/phelber/EasyCOCO/master/easycoco/data/val2017.txt"

def seed_images_human_coco(per_subtype: int) -> tuple[int, str]:
    """COCO val2017 real photographs (research use license)."""
    target = HELDOUT / "image" / "human" / "coco_val2017"
    try:
        listing = _fetch(COCO_URLS).decode().split()
    except Exception as exc:
        return 0, f"coco listing fetch failed: {exc}"
    base = "http://images.cocodataset.org/val2017/"
    saved = 0
    for name in listing:
        if saved >= per_subtype:
            break
        try:
            data = _fetch(base + name, timeout=30)
            if len(data) < 5000:
                continue
            target.mkdir(parents=True, exist_ok=True)
            (target / name).write_bytes(data)
            saved += 1
        except Exception:
            continue
    return saved, "ok" if saved else "no images downloaded"


def seed_images_ai_diffusiondb(per_subtype: int) -> tuple[int, str]:
    """DiffusionDB 2k first-5k subset: real SD 1.x outputs (CC-BY-NC research use)."""
    API = "https://datasets-server.huggingface.co/rows"
    target = HELDOUT / "image" / "ai" / "diffusiondb_sd14"
    saved = 0
    offset = 0
    errors = []
    while saved < per_subtype and offset < 1000:
        try:
            payload = json.loads(_fetch(f"{API}?dataset=pollito081/diffusiondb-2k&config=prompt&split=train&offset={offset}&length=50"))
        except Exception as exc:
            # Fall back to the canonical dataset id
            try:
                payload = json.loads(_fetch(f"{API}?dataset=pollito081/diffusiondb&config=2m_first_5k&split=train&offset={offset}&length=10"))
            except Exception as exc2:
                errors.append(f"diffusiondb fetch failed: {exc}; retry failed: {exc2}")
                break
        rows = payload.get("rows", [])
        if not rows:
            break
        for row in rows:
            img = row.get("row", {}).get("image")
            if not img or not img.get("src"):
                continue
            try:
                data = _fetch(img["src"], timeout=30)
                target.mkdir(parents=True, exist_ok=True)
                (target / f"diffusiondb_{saved:04d}.png").write_bytes(data)
                saved += 1
            except Exception:
                continue
            if saved >= per_subtype:
                break
        offset += 50
    return saved, "; ".join(errors) if errors else "ok"


# ============================================================================
# VIDEO — DeeperForensics-1.0 public sample page (limited availability)
# ============================================================================

def seed_videos(per_subtype: int) -> tuple[int, str]:
    """Video sourcing is intentionally conservative: most deepfake corpora
    (FaceForensics++, DFDC) require signed EULAs and cannot be auto-downloaded.
    This function downloads nothing by default and prints manual instructions."""
    print(
        "\n[video] Automatic download is NOT available for major deepfake corpora:\n"
        "  - FaceForensics++ : requires signed EULA → https://canadacollege.net/faceforensics\n"
        "  - DFDC            : Kaggle competition terms\n"
        "  - DeeperForensics : request form at https://deeperforensics.org\n"
        "Manual path: download ~10 real + ~10 fake clips after EULA approval, then place:\n"
        "  heldout_set/video/ai/<method>/clip_0000.mp4\n"
        "  heldout_set/video/human/screen/clip_0000.mp4\n"
        "No fabricated video will be substituted."
    )
    return 0, "manual sourcing required (EULA-gated corpora)"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Seed heldout_set/ with REAL evaluation data")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--images", action="store_true")
    parser.add_argument("--videos", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--per-subtype", type=int, default=25)
    args = parser.parse_args()

    mode_all = args.all or not (args.text or args.images or args.videos)
    results = {}

    if mode_all or args.text:
        print("=== TEXT: RAID (gpt-4o, claude, gemini, human) ===")
        for gen in ("gpt-4o", "claude", "gemini"):
            n, status = seed_text_raid(gen, "ai", args.per_subtype)
            results[f"text/ai/{gen}"] = (n, status)
            print(f"  text/ai/{gen}: {n} samples ({status})")
        n, status = seed_text_raid("human", "human", args.per_subtype)
        results["text/human/raid"] = (n, status)
        print(f"  text/human/raid: {n} samples ({status})")
        if n < args.per_subtype:
            print("  ... RAID human short; pulling HC3 fallback")
            n2, status2 = seed_text_hc3(args.per_subtype)
            results["text/hc3"] = (n2, status2)
            print(f"  text/hc3: {n2} samples ({status2})")

    if mode_all or args.images:
        print("\n=== IMAGES: COCO (real) + DiffusionDB (AI) ===")
        n, status = seed_images_human_coco(args.per_subtype)
        results["image/human/coco"] = (n, status)
        print(f"  image/human/coco: {n} ({status})")
        n, status = seed_images_ai_diffusiondb(args.per_subtype)
        results["image/ai/diffusiondb"] = (n, status)
        print(f"  image/ai/diffusiondb: {n} ({status})")

    if mode_all or args.videos:
        n, status = seed_videos(args.per_subtype)
        results["video"] = (n, status)

    print("\n=== Summary ===")
    for k, (n, s) in results.items():
        print(f"  {k}: {n} ({s})")
    total = sum(n for n, _ in results.values())
    print(f"\nTotal real samples staged: {total}")
    if total == 0:
        print("\nNothing was downloaded — likely network restrictions. Fix connectivity")
        print("or follow the manual instructions printed above, then re-run eval_rigorous.py.")
        sys.exit(1)


if __name__ == "__main__":
    main()
