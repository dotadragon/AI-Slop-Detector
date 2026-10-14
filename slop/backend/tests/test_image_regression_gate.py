"""
G7: Image Verdict Regression Gate Test.

Verifies that no silent verdict regressions occur across the full golden_set
and heldout_set (66 images) versus the committed production baseline in
Model/fusion/image_baseline_verdicts.json.

Prevents the class of bugs where adjusting a heuristic or fusion threshold
for one image silently flips dozens of others undetected.
"""

import json
import os
import sys
from pathlib import Path
import pytest

# Ensure root and scripts are on sys.path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_image_regression_gate import (  # noqa: E402
    DEFAULT_BASELINE_PATH,
    compare_against_baseline,
    run_evaluation,
)


def test_baseline_file_integrity():
    """Confirms the committed baseline exists, is valid JSON, and covers all 66 specimens."""
    assert DEFAULT_BASELINE_PATH.exists(), f"Baseline file missing at {DEFAULT_BASELINE_PATH}"
    with open(DEFAULT_BASELINE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "samples" in data, "Baseline file missing 'samples' dict"
    samples = data["samples"]
    assert len(samples) == 66, f"Expected exactly 66 baseline samples, found {len(samples)}"
    expected_correct = sum(1 for s in samples.values() if s["baseline_verdict"] == s["ground_truth"])
    assert data["baseline_correct"] == expected_correct
    assert data["baseline_accuracy_pct"] == round(100.0 * expected_correct / len(samples), 2)
    assert data["baseline_accuracy_pct"] >= 70.0

    for name, s in samples.items():
        assert "ground_truth" in s, f"Sample {name} missing ground_truth"
        assert s["ground_truth"] in ("AI", "HUMAN")
        assert "baseline_score" in s
        assert "baseline_verdict" in s
        assert s["baseline_verdict"] in ("AI", "HUMAN")


@pytest.mark.skipif(
    not os.environ.get("SLOP_HELDOUT_DIR"),
    reason="SLOP_HELDOUT_DIR environment variable is not set; external heldout image regression gate test skipped",
)
def test_image_verdict_regression_gate():
    """Runs full image comparison against committed baseline and asserts flips <= N."""
    max_flips = int(os.environ.get("MAX_ALLOWED_IMAGE_FLIPS", "0"))
    max_regressions = int(os.environ.get("MAX_ALLOWED_IMAGE_REGRESSIONS", "0"))

    live_results = run_evaluation()
    comparison = compare_against_baseline(live_results, DEFAULT_BASELINE_PATH)

    total_flips = comparison["total_flips"]
    regression_flips = comparison["regression_flips"]

    if total_flips > max_flips or regression_flips > max_regressions:
        error_lines = [
            f"Image regression gate failed: {total_flips} flip(s) detected (allowed: {max_flips}), "
            f"{regression_flips} regression(s) (allowed: {max_regressions}).",
            "Flipped specimens:",
        ]
        for f in comparison["flips"]:
            error_lines.append(
                f"  - [{f.get('flip_type', 'FLIP')}] {f['file']}: "
                f"Baseline={f['baseline_verdict']} ({f['baseline_score']}) -> "
                f"Live={f['live_verdict']} ({f['live_score']}, delta={f['delta']:+0.3f}), "
                f"Truth={f['ground_truth']}"
            )
        error_lines.append(
            "\nRemediation:\n"
            "If these flips are intentional improvements, update the committed baseline with:\n"
            "  python eval_image_regression_gate.py --update\n"
        )
        pytest.fail("\n".join(error_lines))
