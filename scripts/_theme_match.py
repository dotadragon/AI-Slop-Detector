#!/usr/bin/env python3
"""Content-match check: CLIP-theme distribution of RealVis-XL vs the new COCO humans.

Zero-shot CLIP text prototypes over the cached image embeddings (no training,
gate used read-only for comparison only).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

CLIP_DIR = ROOT / "Model" / "image" / "clip_vit_large"
THEMES = {
    "portrait": ["a close-up portrait of a person", "a photo of a person's face",
                 "a photograph of a human model posing"],
    "street": ["a street scene in a city", "an urban street with buildings and cars",
               "a road with traffic and vehicles"],
    "nature": ["a wildlife photograph of an animal", "a landscape photograph of nature",
               "a photo of plants and trees outdoors"],
    "object_indoor": ["a photograph of an object on a surface", "an indoor room scene",
                      "a still life photo of things"],
}


def main() -> int:
    import torch
    from safetensors import safe_open
    from transformers import AutoProcessor, CLIPTextModelWithProjection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
    tm = CLIPTextModelWithProjection.from_pretrained(
        str(CLIP_DIR), dtype=torch.float16, local_files_only=True).to(device).eval()

    protos = {}
    for name, prompts in THEMES.items():
        inp = proc(text=prompts, return_tensors="pt", padding=True)
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            e = tm(**inp).text_embeds.float().cpu().numpy()
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        v = e.mean(axis=0)
        protos[name] = v / np.linalg.norm(v)
    with safe_open(str(CLIP_DIR / "model.safetensors"), framework="pt") as f:
        logit_scale = float(f.get_tensor("logit_scale").exp().item())

    z = np.load(ROOT / "_clip_v2_embeddings.npz", allow_pickle=False)
    fams = [str(x) for x in z["family"]]
    emb = z["embeds"].astype(np.float64)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    names = list(protos)
    W = np.stack([protos[n] for n in names])
    sims = emb @ W.T

    def dist(mask, label):
        rows = sims[mask]
        idx = rows.argmax(axis=1)
        c = Counter(names[i] for i in idx)
        print(f"  {label:22s} " + "  ".join(f"{n}={c.get(n,0)}" for n in names))
        return c

    print("=" * 78)
    print("CONTENT MATCH (CLIP zero-shot theme distribution)")
    print("=" * 78)
    print("  (cached 100-image embedding set; RealVis-XL is the target to match)")
    dist(np.array([f == "realvis_xl" for f in fams]), "realvis_xl (target)")
    for f in ("midjourney", "kandinsky", "dit", "cascade_lcm"):
        dist(np.array([x == f for x in fams]), f)
    dist(np.array([f == "picsum_unsplash" for f in fams]), "picsum_unsplash (gate)")

    new = sorted((ROOT / "heldout_set" / "image" / "human_v2").glob("*.jpg"))
    from transformers import CLIPVisionModelWithProjection
    vm = CLIPVisionModelWithProjection.from_pretrained(
        str(CLIP_DIR), dtype=torch.float16, local_files_only=True).to(device).eval()
    ve = []
    for i in range(0, len(new), 16):
        chunk = []
        for p in new[i:i + 16]:
            from PIL import Image as _I
            with _I.open(p) as im:
                chunk.append(im.convert("RGB"))
        inp = proc(images=chunk, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            out = vm(**inp).image_embeds.float().cpu().numpy()
        ve.append(out)
    ve = np.concatenate(ve)
    ve = ve / np.linalg.norm(ve, axis=1, keepdims=True)
    sims2 = ve @ W.T
    c = Counter(names[i] for i in sims2.argmax(axis=1))
    print(f"  {'human_v2 (new, n=50)':22s} " + "  ".join(f"{n}={c.get(n,0)}" for n in names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
