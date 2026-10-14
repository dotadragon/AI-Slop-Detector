"""Read-only diagnosis of the STEP 5 failure (train/serve feature skew).

For the val samples whose verdict flipped between the training pipeline and the
production pipeline, capture the feature dict the production analyzer feeds to
`fuse_text_score` and diff it against the training-time cached row.

Writes nothing except stdout. Does not modify any model artifact.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")

VAL_DIR = ROOT / "raid_val" / "text"

FLIPPED = [
    "ai/llama_chat/llama_chat_0007.txt",
    "ai/mistral/mistral_0019.txt",
    "ai/mpt_chat/mpt_chat_0009.txt",
]
CONTROLS = [
    "ai/gpt2/gpt2_0000.txt",          # train_raw == prod_raw
    "ai/mistral/mistral_0004.txt",    # train_raw != prod_raw
]


def main() -> int:
    bundle = pickle.load(open(ROOT / "Model" / "fusion" / "text_gbdt_v2.pkl", "rb"))
    feats, gbdt = bundle["features"], bundle["model"]
    cal = pickle.load(open(ROOT / "Model" / "fusion" / "text_calibrator.pkl", "rb"))
    tr_cache = json.loads((VAL_DIR / "_feature_cache.json").read_text(encoding="utf-8"))
    tpred = json.loads((VAL_DIR / "_transformer_preds.json").read_text(encoding="utf-8"))

    import app.ml.fusion as fusion

    seen: list[dict] = []
    original = fusion.fuse_text_score

    def wrapped(features):
        seen.append(dict(features))
        return original(features)

    fusion.fuse_text_score = wrapped
    from app.ml.text_detector import analyze_text

    print("=" * 78)
    print("TRAIN/SERVE FEATURE SKEW DIAGNOSIS (read-only)")
    print("=" * 78)
    for sid in FLIPPED + CONTROLS:
        path = VAL_DIR / sid
        text = path.read_text(encoding="utf-8", errors="ignore")
        seen.clear()
        res = analyze_text(text)
        prod = seen[-1] if seen else {}
        row13 = tr_cache.get(sid)

        Xp = np.array([[float(prod.get(fn, 0.0)) for fn in feats]], dtype=np.float32)
        Xt = np.zeros((1, len(feats)), dtype=np.float32)
        Xt[0, 0] = float(tpred.get(sid, 0.0))
        if row13 is not None:
            Xt[0, 1:] = np.asarray(row13, dtype=np.float32)
        raw_p = float(gbdt.predict_proba(Xp)[0, 1])
        raw_t = float(gbdt.predict_proba(Xt)[0, 1])

        tag = "FLIPPED" if sid in FLIPPED else "control"
        print(f"\n[{tag}] {sid}")
        print(f"  production raw={raw_p:.4f}  training raw={raw_t:.4f}  "
              f"overall_score={res.get('overall_score')} band={res.get('confidence_band')}")
        if row13 is None:
            print("  NOT PRESENT in training feature cache")
            continue
        print(f"  {'feature':<24} {'training':>12} {'production':>12}  {'delta':>10}")
        for j, fn in enumerate(feats):
            tv, pv = float(Xt[0, j]), float(Xp[0, j])
            flag = "  <== DIFFERS" if abs(tv - pv) > 1e-6 else ""
            print(f"  {fn:<24} {tv:>12.6f} {pv:>12.6f}  {pv-tv:>10.6f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
