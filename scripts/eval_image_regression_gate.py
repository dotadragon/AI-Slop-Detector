#!/usr/bin/env python3
"""Automated Image Verdict Regression Gate for CI and Pre-Merge Checks.

Evaluates the full golden_set + heldout_set (66 images) against the committed
baseline in Model/fusion/image_baseline_verdicts.json.

Fails the build (exit code 1) if more than N verdicts flip versus the baseline,
preventing silent accuracy regressions from fusion weight adjustments or gate
threshold modifications.

Usage:
    python eval_image_regression_gate.py                    # Fail on >0 flips (strict default)
    python eval_image_regression_gate.py --max-flips 2      # Allow up to 2 flips
    python eval_image_regression_gate.py --update           # Promote current outputs to baseline
    python eval_image_regression_gate.py --output-json report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

# Ensure backend modules can be imported
ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

from app.ml.image_detector import _analyze_pil_image  # noqa: E402

DEFAULT_BASELINE_PATH = ROOT / "Model" / "fusion" / "image_baseline_verdicts.json"
AUDIT_LOG_PATH = ROOT / "Model" / "fusion" / "retrain_audit_log.jsonl"

DATASETS = [
    ("golden/ai", ROOT / "golden_set" / "image" / "ai", "AI"),
    ("golden/diagram", ROOT / "golden_set" / "image" / "diagram", "HUMAN"),
    ("golden/human", ROOT / "golden_set" / "image" / "human", "HUMAN"),
    ("heldout/ai_sdxl", ROOT / "heldout_set" / "image" / "ai" / "sdxl_proxy", "AI"),
    ("heldout/human_picsum", ROOT / "heldout_set" / "image" / "human" / "picsum_unsplash", "HUMAN"),
]


def run_evaluation() -> dict[str, dict]:
    """Runs real zero-mock inference across all golden and heldout image samples."""
    results = {}
    for ds_name, ds_dir, truth in DATASETS:
        if not ds_dir.exists():
            print(f"[WARN] Dataset directory missing: {ds_dir}", file=sys.stderr)
            continue
        for p in sorted(ds_dir.glob("*")):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            with Image.open(p) as img:
                rgb = img.convert("RGB")
            res = _analyze_pil_image(rgb, p.name, str(p))
            score = float(res["overall_score"])
            verdict = "AI" if score >= 0.50 else "HUMAN"
            sig = res.get("signals", {})
            dig = res.get("digital_signature", {})
            results[p.name] = {
                "file": p.name,
                "dataset": ds_name,
                "path": str(p),
                "ground_truth": truth,
                "score": round(score, 3),
                "verdict": verdict,
                "neural_prob": round(float(sig.get("model_ai_probability") or 0.0), 4),
                "is_diagram": bool(sig.get("is_diagram")),
                "is_real_camera": bool(sig.get("is_real_camera")),
                "cfa_score": round(float(sig.get("cfa_score") or 0.0), 3),
                "signature_detected": bool(dig.get("detected")),
                "c2pa_verified": sig.get("provenance") == "verified",
                "heuristic_score": round(float(sig.get("heuristic_score") or 0.0), 3),
            }
    return results


def compare_against_baseline(
    live_results: dict[str, dict], baseline_path: Path
) -> dict:
    """Compares live outputs against committed baseline and categorizes all flips."""
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline file not found at {baseline_path}. Run with --update to generate it."
        )

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baseline_samples = baseline_data.get("samples", {})
    flips = []
    regressions = []
    improvements = []
    unflipped = []

    for name, live in sorted(live_results.items()):
        if name not in baseline_samples:
            flips.append({
                "file": name,
                "dataset": live["dataset"],
                "type": "NEW_SAMPLE",
                "ground_truth": live["ground_truth"],
                "baseline_verdict": "N/A",
                "baseline_score": None,
                "live_verdict": live["verdict"],
                "live_score": live["score"],
                "delta": None,
            })
            continue

        base = baseline_samples[name]
        b_verdict = base["baseline_verdict"]
        b_score = base["baseline_score"]
        l_verdict = live["verdict"]
        l_score = live["score"]
        truth = live["ground_truth"]
        delta = round(l_score - b_score, 3)

        sample_summary = {
            "file": name,
            "dataset": live["dataset"],
            "ground_truth": truth,
            "baseline_verdict": b_verdict,
            "baseline_score": b_score,
            "live_verdict": l_verdict,
            "live_score": l_score,
            "delta": delta,
            "neural_prob": live["neural_prob"],
            "is_diagram": live["is_diagram"],
            "is_real_camera": live["is_real_camera"],
            "cfa_score": live.get("cfa_score", 0.0),
            "signature_detected": live.get("signature_detected", False),
            "c2pa_verified": live.get("c2pa_verified", False),
            "heuristic_score": live.get("heuristic_score", 0.0),
        }

        if b_verdict != l_verdict:
            # Categorize flip
            if b_verdict == truth and l_verdict != truth:
                flip_type = "REGRESSION"
                direction = f"{b_verdict} (correct) -> {l_verdict} (wrong)"
                regressions.append(sample_summary)
            elif b_verdict != truth and l_verdict == truth:
                flip_type = "IMPROVEMENT"
                direction = f"{b_verdict} (wrong) -> {l_verdict} (correct)"
                improvements.append(sample_summary)
            else:
                flip_type = "NEUTRAL"
                direction = f"{b_verdict} -> {l_verdict}"

            sample_summary["flip_type"] = flip_type
            sample_summary["direction"] = direction
            flips.append(sample_summary)
        else:
            unflipped.append(sample_summary)

    total_samples = len(live_results)
    live_correct = sum(1 for s in live_results.values() if s["verdict"] == s["ground_truth"])
    base_correct = baseline_data.get("baseline_correct", 0)

    return {
        "baseline_path": str(baseline_path),
        "total_samples": total_samples,
        "baseline_accuracy": f"{base_correct}/{total_samples} ({base_correct/total_samples:.2%})",
        "live_accuracy": f"{live_correct}/{total_samples} ({live_correct/total_samples:.2%})",
        "total_flips": len(flips),
        "regression_flips": len(regressions),
        "improvement_flips": len(improvements),
        "flips": flips,
        "regressions": regressions,
        "improvements": improvements,
    }


def update_baseline(live_results: dict[str, dict], baseline_path: Path) -> None:
    """Updates the baseline JSON file with current live outputs."""
    total = len(live_results)
    correct = sum(1 for e in live_results.values() if e["verdict"] == e["ground_truth"])
    acc_pct = round(100.0 * correct / total, 2)

    samples = {}
    for name, r in live_results.items():
        samples[name] = {
            "file": name,
            "dataset": r["dataset"],
            "ground_truth": r["ground_truth"],
            "baseline_score": r["score"],
            "baseline_verdict": r["verdict"],
            "neural_score": r["neural_prob"],
            "is_diagram": r["is_diagram"],
            "is_real_camera": r["is_real_camera"],
        }

    doc = {
        "version": "1.0",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "description": "Committed baseline for image detection verdicts across golden_set and heldout_set (n=66).",
        "total_samples": total,
        "baseline_correct": correct,
        "baseline_accuracy_pct": acc_pct,
        "samples": samples,
    }

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    # Log to audit trail
    if AUDIT_LOG_PATH.parent.exists():
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "image_baseline_promoted",
                "total_samples": total,
                "accuracy_pct": acc_pct,
            }) + "\n")

    print(f"[OK] Promoted current outputs to baseline at {baseline_path} (accuracy: {correct}/{total}, {acc_pct}%).")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated CI regression gate for image forensics."
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help="Path to committed image baseline verdicts JSON.",
    )
    parser.add_argument(
        "--max-flips",
        type=int,
        default=int(os.environ.get("MAX_ALLOWED_IMAGE_FLIPS", "0")),
        help="Maximum total flips permitted before failing (default: 0 or MAX_ALLOWED_IMAGE_FLIPS).",
    )
    parser.add_argument(
        "--max-regressions",
        type=int,
        default=int(os.environ.get("MAX_ALLOWED_IMAGE_REGRESSIONS", "0")),
        help="Maximum regression flips permitted before failing (default: 0 or MAX_ALLOWED_IMAGE_REGRESSIONS).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Promote current live results as the new committed baseline.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Path to write JSON regression comparison report.",
    )
    args = parser.parse_args()

    import subprocess
    try:
        git_res = subprocess.run(
            ["git", "status", "--porcelain", "slop/backend/app/ml", "Model/fusion"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        mod_lines = [ln.strip() for ln in git_res.stdout.splitlines() if ln.strip()]
    except Exception:
        mod_lines = []

    print("Running zero-mock evaluation across golden_set and heldout_set (n=66)...")
    if mod_lines:
        print(f"[CI GATE ACTIVE] Detected {len(mod_lines)} modified image forensic component(s):")
        for ml in mod_lines[:8]:
            print(f"  * {ml}")
        if len(mod_lines) > 8:
            print(f"  * ... and {len(mod_lines) - 8} more")
    else:
        print("[CI GATE ACTIVE] Verifying image forensic pipeline against committed baseline.")
    live_results = run_evaluation()

    if args.update:
        update_baseline(live_results, args.baseline)
        return 0

    comparison = compare_against_baseline(live_results, args.baseline)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"Wrote comparison report to {args.output_json}")

    # Output formatted report
    print("\n" + "=" * 80)
    print("                    IMAGE VERDICT REGRESSION REPORT")
    print("=" * 80)
    print(f"Total Samples Evaluated: {comparison['total_samples']}")
    print(f"Baseline Accuracy:       {comparison['baseline_accuracy']}")
    print(f"Live Accuracy:           {comparison['live_accuracy']}")
    print(f"Total Flips Detected:    {comparison['total_flips']}  (Allowed: <= {args.max_flips})")
    print(f"  - Regressions:         {comparison['regression_flips']}  (Allowed: <= {args.max_regressions})")
    print(f"  - Improvements:        {comparison['improvement_flips']}")
    print("-" * 80)

    if comparison["flips"]:
        print("FLIP BREAKDOWN:")
        for flip in comparison["flips"]:
            tag = f"[{flip.get('flip_type', 'FLIP')}]"
            print(f"  {tag:<14} {flip['file']}")
            print(f"                 Ground Truth: {flip['ground_truth']} | Dataset: {flip['dataset']}")
            print(f"                 Baseline:     {flip['baseline_verdict']:<5} (score={flip['baseline_score']})")
            print(f"                 Live:         {flip['live_verdict']:<5} (score={flip['live_score']}, delta={flip['delta']:+0.3f})")
            sig_parts = [
                f"Neural: {flip.get('neural_prob')}",
                f"Heuristic: {flip.get('heuristic_score')}",
                f"Diagram: {flip.get('is_diagram')}",
                f"Camera: {flip.get('is_real_camera')}",
                f"CFA: {flip.get('cfa_score')}",
                f"Sig: {flip.get('signature_detected')}",
                f"C2PA: {flip.get('c2pa_verified')}",
            ]
            print(f"                 Signals:      {' | '.join(sig_parts)}")
            print()
    else:
        print("  [OK] Zero verdict flips detected. All 66 image classifications identical to baseline.")

    print("=" * 80)

    # Threshold checks
    failed = False
    fail_reasons = []

    if comparison["total_flips"] > args.max_flips:
        failed = True
        fail_reasons.append(
            f"Total flips ({comparison['total_flips']}) exceeded allowed maximum ({args.max_flips})."
        )

    if comparison["regression_flips"] > args.max_regressions:
        failed = True
        fail_reasons.append(
            f"Regression flips ({comparison['regression_flips']}) exceeded allowed maximum ({args.max_regressions})."
        )

    if failed:
        print("\n[FAIL] CI BUILD BLOCKED: Image regression gate failed:")
        for r in fail_reasons:
            print(f"   - {r}")
        print("\nRemediation:")
        print("   1. Review the flipped samples listed above.")
        print("   2. If the flip is an unintended regression, revert your change or tune thresholds.")
        print("   3. If the flip is an intentional, verified improvement, update the baseline with:")
        print("      python eval_image_regression_gate.py --update\n")
        return 1

    print("\n[PASS] IMAGE REGRESSION GATE PASSED.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
