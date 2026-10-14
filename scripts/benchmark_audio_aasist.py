"""Zero-shot benchmark of Spectra-AASIST3 on heldout_set/audio (A1/A2).

Compares against the incumbent Wav2Vec2 classifier head (AUC 0.493 on the
same set). The AASIST3 head outputs a logit where higher = more bona fide;
we invert to P(fake) = sigmoid(-logit) for comparability with the detector's
score semantics (higher = more AI).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "slop" / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "Model" / "audio" / "spectra_aasist3"))

CACHE = Path("audio_aasist_cache.json")
RESULT = Path("audio_aasist_result.json")
MODEL_DIR = Path("Model/audio/spectra_aasist3").resolve()
THRESHOLD = -1.0625009  # official classify() threshold from the model card


def main() -> None:
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import safetensors.torch

    from model import SpectraAASIST3  # type: ignore[import-not-found]

    print("Loading SpectraAASIST3...", flush=True)
    model = SpectraAASIST3()
    sd = safetensors.torch.load_file(str(MODEL_DIR / "model.safetensors"))
    model.load_state_dict(sd)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("device:", device, flush=True)

    root = Path("heldout_set/audio")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    for split in ("ai", "human"):
        for meta in manifest.get(split, {}).values():
            rows.append((str(root / meta["file"]), 1 if split == "ai" else 0))

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def load_wav(path: str) -> np.ndarray:
        import soundfile as sf

        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio

            wav = torchaudio.functional.resample(torch.from_numpy(wav[None, :]), sr, 16000)[0].numpy()
        return wav

    done = 0
    for fpath, label in rows:
        if fpath in cache:
            continue
        try:
            wav = load_wav(fpath)
            # chunk to the model's expected 64400 samples (~4s), max-pool logits
            chunk = 64400
            logits = []
            with torch.inference_mode():
                for s in range(0, len(wav), chunk):
                    seg = wav[s : s + chunk]
                    if len(seg) < 8000:
                        break
                    x = torch.from_numpy(seg)[None, :, None].to(device)
                    out = model(x)
                    logits.append(float(out[0, 1]))  # logit[1] = bona-fide score
            score_fake = 1.0 / (1.0 + np.exp(np.clip(min(logits), -30, 30)))  # invert: higher bona-fide -> lower P(fake)
            cache[fpath] = {"score": round(float(score_fake), 4), "label": label}
        except Exception as exc:
            cache[fpath] = {"score": None, "label": label, "error": str(exc)[:120]}
        done += 1
        if done % 10 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  {done}/{len(rows)}", flush=True)
    CACHE.write_text(json.dumps(cache))

    valid = [(v["score"], v["label"]) for v in cache.values() if v["score"] is not None]
    pos = [s for s, l in valid if l == 1]
    neg = [s for s, l in valid if l == 0]
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg)) if pos and neg else None
    tpr = sum(1 for s in pos if s >= 0.5) / max(len(pos), 1)
    tnr = sum(1 for s in neg if s < 0.5) / max(len(neg), 1)
    result = {
        "model": "lab260/Spectra-AASIST3 (zero-shot, official threshold for bacc; AUC threshold-free)",
        "n": len(valid),
        "errors": len(cache) - len(valid),
        "auc": round(auc, 4) if auc is not None else None,
        "balanced_accuracy_at_0.5": round((tpr + tnr) / 2, 4),
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "incumbent_wav2vec2_auc": 0.493,
    }
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
