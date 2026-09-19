"""
SLOP Evaluation & Accuracy Benchmark Harness
Enforces Rule 0 & Rule 5 of ANTIGRAVITY_RULES.md.
Executes live inference on real model weights across golden_set/.
Computes Accuracy, Precision, Recall, F1, and Confusion Matrix per modality.
Supports --quick flag for rapid pre/post edit verification.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import analyze_text
from app.ml.image_detector import analyze_image
from app.ml.video_detector import analyze_video

GOLDEN_SET_DIR = Path(__file__).resolve().parent / "golden_set"


def evaluate_text(quick: bool = False) -> dict:
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0, "samples": []}
    ai_dir = GOLDEN_SET_DIR / "text" / "ai"
    human_dir = GOLDEN_SET_DIR / "text" / "human"

    ai_files = sorted(ai_dir.glob("*.txt")) if ai_dir.exists() else []
    human_files = sorted(human_dir.glob("*.txt")) if human_dir.exists() else []

    if quick:
        ai_files = ai_files[:3]
        human_files = human_files[:3]

    # Evaluate AI samples (ground truth = 1)
    for f in ai_files:
        text = f.read_text(encoding="utf-8")
        res = analyze_text(text)
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["tp"] += 1
        else:
            results["fn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "AI", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    # Evaluate Human samples (ground truth = 0)
    for f in human_files:
        text = f.read_text(encoding="utf-8")
        res = analyze_text(text)
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["fp"] += 1
        else:
            results["tn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "HUMAN", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    return _compute_metrics(results)


def evaluate_image(quick: bool = False) -> dict:
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0, "samples": []}
    ai_dir = GOLDEN_SET_DIR / "image" / "ai"
    human_dir = GOLDEN_SET_DIR / "image" / "human"
    diag_dir = GOLDEN_SET_DIR / "image" / "diagram"

    ai_files = sorted([f for f in ai_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]) if ai_dir.exists() else []
    human_files = sorted([f for f in human_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]) if human_dir.exists() else []
    diag_files = sorted([f for f in diag_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg")]) if diag_dir.exists() else []

    if quick:
        ai_files = ai_files[:2]
        human_files = human_files[:1]
        diag_files = diag_files[:1]

    # Evaluate AI samples
    for f in ai_files:
        res = analyze_image(str(f))
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["tp"] += 1
        else:
            results["fn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "AI", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    # Evaluate Human photo samples
    for f in human_files:
        res = analyze_image(str(f))
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["fp"] += 1
        else:
            results["tn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "HUMAN_PHOTO", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    # Evaluate Human diagram samples
    for f in diag_files:
        res = analyze_image(str(f))
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["fp"] += 1
        else:
            results["tn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "HUMAN_DIAGRAM", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    return _compute_metrics(results)


def evaluate_video(quick: bool = False) -> dict:
    results = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0, "samples": []}
    ai_dir = GOLDEN_SET_DIR / "video" / "ai"
    human_dir = GOLDEN_SET_DIR / "video" / "human_screen"

    ai_files = sorted(ai_dir.glob("*.mp4")) if ai_dir.exists() else []
    human_files = sorted(human_dir.glob("*.mp4")) if human_dir.exists() else []

    for f in ai_files:
        res = analyze_video(str(f), f"eval_{f.stem}")
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["tp"] += 1
        else:
            results["fn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "AI_VIDEO", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    for f in human_files:
        res = analyze_video(str(f), f"eval_{f.stem}")
        score = res["overall_score"]
        pred_ai = score >= 0.50
        if pred_ai:
            results["fp"] += 1
        else:
            results["tn"] += 1
        results["total"] += 1
        results["samples"].append({"file": f.name, "truth": "HUMAN_SCREEN", "score": score, "pred": "AI" if pred_ai else "HUMAN"})

    return _compute_metrics(results)


def _compute_metrics(r: dict) -> dict:
    tp, fp, tn, fn = r["tp"], r["fp"], r["tn"], r["fn"]
    total = r["total"]
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
        "samples": r["samples"],
    }


def main():
    parser = argparse.ArgumentParser(description="SLOP Accuracy Evaluation")
    parser.add_argument("--quick", action="store_true", help="Run fast verification subset")
    parser.add_argument("--save", type=str, default="accuracy_report.json", help="Save JSON report")
    args = parser.parse_args()

    print("==================================================")
    print(f"  SLOP ACCURACY EVALUATION (mode: {'QUICK' if args.quick else 'FULL'})")
    print("==================================================")

    print("\n[1/3] Evaluating Text Forensics Pipeline...")
    text_res = evaluate_text(quick=args.quick)
    print(f"  Accuracy: {text_res['accuracy']}% | Precision: {text_res['precision']}% | Recall: {text_res['recall']}% | (TP={text_res['tp']}, FP={text_res['fp']}, TN={text_res['tn']}, FN={text_res['fn']})")

    print("\n[2/3] Evaluating Image Forensics Pipeline...")
    img_res = evaluate_image(quick=args.quick)
    print(f"  Accuracy: {img_res['accuracy']}% | Precision: {img_res['precision']}% | Recall: {img_res['recall']}% | (TP={img_res['tp']}, FP={img_res['fp']}, TN={img_res['tn']}, FN={img_res['fn']})")

    print("\n[3/3] Evaluating Video Forensics Pipeline...")
    vid_res = evaluate_video(quick=args.quick)
    print(f"  Accuracy: {vid_res['accuracy']}% | Precision: {vid_res['precision']}% | Recall: {vid_res['recall']}% | (TP={vid_res['tp']}, FP={vid_res['fp']}, TN={vid_res['tn']}, FN={vid_res['fn']})")

    overall_acc = round((text_res["accuracy"] + img_res["accuracy"] + vid_res["accuracy"]) / 3, 2)
    print("\n--------------------------------------------------")
    print(f"  OVERALL MULTI-MODAL ACCURACY: {overall_acc}%")
    print("--------------------------------------------------")

    report = {
        "overall_accuracy": overall_acc,
        "text": text_res,
        "image": img_res,
        "video": vid_res,
    }

    report_path = Path(__file__).resolve().parent / args.save
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}\n")

if __name__ == "__main__":
    main()
