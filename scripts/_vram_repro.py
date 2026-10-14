"""In-suite-order VRAM repro: text test first, then the video test.

Simulates the exact pytest ordering that trips the 5 GB ceiling:
  1. a text call (as test_security_audit.py does) parks text weights on the GPU
  2. the video test measures its own footprint + the parked weights

Run with SKIP_EVICT=1 to show the pre-fix number, or unset to show the fixed one.
"""
import os
import sys

os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("SECRET_KEY", "local-verification-key-not-for-production-0123456789")
sys.path.insert(0, "slop/backend")

from app.ml.vram_manager import get_vram_stats, release_all_model_singletons  # noqa: E402
from app.ml.text_detector import analyze_text  # noqa: E402
from app.ml.video_detector import analyze_video  # noqa: E402

SKIP_EVICT = os.environ.get("SKIP_EVICT") == "1"
TEST_VIDEO = r"C:\Users\karth\Downloads\test.mp4"

SAMPLE = (
    "The quarterly report outlines the main findings from our regional "
    "operations and highlights several areas for improvement next year."
)


def mb() -> float:
    return get_vram_stats()["allocated_mb"]


print(f"=== mode: {'PRE-FIX (no eviction)' if SKIP_EVICT else 'FIXED (module fixture eviction)'} ===")
print(f"allocated at start: {mb()} MB")

# Step 1 — text call, exactly like an earlier test in the suite.
res = analyze_text(SAMPLE)
print(f"text overall_score={res.get('overall_score')} keys={sorted(res.keys())}")
after_text = mb()
print(f"allocated after text call (text weights resident): {after_text} MB")

# Step 2 — the module-scoped fixture does this before test_video_production runs.
if not SKIP_EVICT:
    released = release_all_model_singletons()
    print(f"release_all_model_singletons() -> {released}")
    print(f"allocated after eviction: {mb()} MB")

# Step 3 — the video test itself.
if not os.path.exists(TEST_VIDEO):
    print("TEST VIDEO MISSING — cannot complete repro")
    sys.exit(2)
vres = analyze_video(TEST_VIDEO, "vram_repro")
video_alloc = mb()
print(f"video score={vres['overall_score']} signals_allocated={vres['signals'].get('vram_stats', {}).get('allocated_mb')}")
print(f"allocated after video call (measured): {video_alloc} MB")
print("CEILING: 5000 MB")
print(f"RESULT: {'PASS' if video_alloc < 5000 else 'FAIL'} ({video_alloc} MB)")
