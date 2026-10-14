"""Micro-check: per-call cost of the fusion artifact loaders, before vs after.

.bak = old behaviour (open + pickle.load every call). Current = mtime-keyed cache.
Read-only with respect to behavior: same artifacts, same objects returned.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

N = 300

from app.ml import fusion as cur  # noqa: E402

spec = None
bak_src = (ROOT / "slop" / "backend" / "app" / "ml" / "fusion.py.bak").read_text(encoding="utf-8")
bak = importlib.util.module_from_spec(importlib.util.spec_from_loader("fusion_bak", None))
bak.__file__ = str(ROOT / "slop" / "backend" / "app" / "ml" / "fusion.py.bak")
exec(compile(bak_src, bak.__file__, "exec"), bak.__dict__)


def bench(loader_pair, label):
    load_w, load_c = loader_pair
    # warm once (populates cache in the current impl)
    load_w("text")
    load_c("text")
    t = time.perf_counter()
    for _ in range(N):
        load_w("text")
        load_c("text")
    per_call = (time.perf_counter() - t) / N * 1e3
    print(f"{label:12s} weights+calibrator per call: {per_call:.4f} ms")
    return per_call


before = bench((bak._load_model_weights, bak._load_calibrator), "BEFORE (.bak)")
after = bench((cur._load_model_weights, cur._load_calibrator), "AFTER (cached)")

# identity check: cached loader must return the SAME object across calls
a = cur._load_model_weights("text")
b = cur._load_model_weights("text")
print("cached object identity (a is b):", a is b)
c1 = cur._load_calibrator("text")
c2 = cur._load_calibrator("text")
print("cached calibrator identity (a is b):", c1 is c2)
print(f"combined saving: {before - after:.4f} ms/call ({100 * (before - after) / before:.0f}%)")
