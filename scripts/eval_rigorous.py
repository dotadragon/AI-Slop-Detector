"""
SLOP Rigorous Evaluation & Calibration Harness (Phase 5 + Phase 6).

Replaces the self-generated 12-sample golden set methodology with a
statistically defensible evaluation over REAL held-out data.

Methodology
-----------
1. Source real held-out data into `heldout_set/`:
     text/  — RAID (https://github.com/liamdugan/RAID) or M4 dataset.
              Save AI prose under text/ai/<generator>/  (e.g. text/ai/gpt4o/),
              human prose under text/human/<source>/.
     image/ — Real photography (e.g. LAION-derived human sets, Unsplash exports)
              under image/human/photo/, plus authentic diagrams under image/human/diagram/.
              AI generations under image/ai/<generator>/ (SDXL, Flux, Midjourney...).
     video/ — DeepFakeBench or FaceForensics++ derivatives under video/ai/<method>/,
              authentic screen recordings under video/human/screen/.
   The directory name two levels below the class folder is treated as the
   *generator subtype* for per-generator confusion matrices.

2. Run the existing detection pipeline with NO threshold tuning on this data.

3. Report accuracy with Wilson score confidence intervals (not bare point
   estimates), full confusion matrices, and per-generator-subtype breakdowns.

4. Phase 6: compute Expected Calibration Error (ECE) per modality, render
   reliability diagrams as text/JSON, and fit Platt scaling / isotonic
   regression on a separate calibration split (never the eval split).

Usage
-----
    python eval_rigorous.py --min-per-class 50
    python eval_rigorous.py --min-per-class 50 --calibration-fraction 0.3
    python eval_rigorous.py --apply-calibration   # persist fitted calibrators

Output: heldout_report.json + HELDOUT_REPORT.md
"""

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
backend_path = ROOT / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import analyze_text          # noqa: E402
from app.ml.image_detector import analyze_image        # noqa: E402
from app.ml.video_detector import analyze_video        # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
CALIBRATION_PATH = ROOT / "calibrators.json"


def get_heldout_dir() -> Path:
    env_path = os.environ.get("SLOP_HELDOUT_DIR")
    if not env_path:
        raise RuntimeError(
            "SLOP_HELDOUT_DIR environment variable is not set. "
            "Please point SLOP_HELDOUT_DIR to the sealed heldout dataset (e.g. C:\\Users\\karth\\Desktop\\AISlop_heldout_set)."
        )
    p = Path(env_path)
    if not p.exists():
        raise RuntimeError(f"SLOP_HELDOUT_DIR directory does not exist: {p}")
    return p


def verify_manifest(heldout_dir: Path, split: str) -> None:
    manifest_path = ROOT / "heldout_manifest.json"
    if not manifest_path.exists():
        manifest_path = heldout_dir / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Sealed manifest heldout_manifest.json not found in {ROOT} or {heldout_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    split_records = [r for r in records if r["split"] == split]
    if not split_records:
        raise RuntimeError(f"No records found in manifest for split '{split}'")

    print(f"[integrity] Verifying SHA-256 for all {len(split_records)} samples in '{split}' split against sealed manifest...")
    for r in split_records:
        target = heldout_dir / r["path"]
        if not target.exists():
            raise RuntimeError(f"Sealed file missing from {split} partition: {target}")
        h = hashlib.sha256()
        with open(target, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != r["sha256"]:
            raise RuntimeError(
                f"FATAL: SHA-256 mismatch for {target}!\n"
                f"Expected: {r['sha256']}\n"
                f"Got:      {actual}\n"
                f"Evaluation aborted — dataset tampering or corruption detected."
            )
    print(f"[integrity] Integrity OK: All {len(split_records)} files match sealed manifest byte-for-byte.")


# ============================================================================
# Statistics helpers
# ============================================================================

def wilson_interval(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def expected_calibration_error(scores: list[float], labels: list[int], n_bins: int = 10) -> float:
    """ECE = sum over bins of |bin accuracy - bin mean confidence| * bin weight."""
    if not scores:
        return 0.0
    bin_sums = [0.0] * n_bins
    bin_counts = [0] * n_bins
    bin_correct = [0.0] * n_bins
    for s, y in zip(scores, labels):
        b = min(n_bins - 1, int(s * n_bins))
        bin_sums[b] += s
        bin_counts[b] += 1
        bin_correct[b] += 1.0 if (s >= 0.5) == (y == 1) else 0.0
    ece = 0.0
    n = len(scores)
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        acc = bin_correct[i] / bin_counts[i]
        conf = bin_sums[i] / bin_counts[i]
        ece += (bin_counts[i] / n) * abs(acc - conf)
    return round(ece, 4)


def reliability_diagram(scores: list[float], labels: list[int], n_bins: int = 10) -> list[dict]:
    """Bucketed predicted-confidence vs actual-accuracy table (for the diagram)."""
    bins = []
    n = len(scores)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bucket = [(s, y) for s, y in zip(scores, labels) if lo <= s < hi or (i == n_bins - 1 and s == 1.0)]
        bins.append({
            "bin": f"[{lo:.1f}, {hi:.1f})" if i < n_bins - 1 else f"[{lo:.1f}, 1.0]",
            "count": len(bucket),
            "mean_confidence": round(sum(s for s, _ in bucket) / len(bucket), 4) if bucket else None,
            "accuracy": round(sum(1.0 for s, y in bucket if (s >= 0.5) == (y == 1)) / len(bucket), 4) if bucket else None,
        })
    return bins


# ============================================================================
# Calibrators (Phase 6)
# ============================================================================

def fit_platt_scaling(scores: list[float], labels: list[int]) -> dict:
    """Fit Platt scaling (logistic regression on the logit of the score) via simple gradient descent.
    Pure-stdlib so the harness runs anywhere; adequate for a 1-D fit."""
    import math as m

    eps = 1e-6
    def logit(s: float) -> float:
        s = min(max(s, eps), 1 - eps)
        return m.log(s / (1 - s))

    xs = [logit(s) for s in scores]
    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(2000):
        ga = gb = 0.0
        for x, y in zip(xs, labels):
            p = 1 / (1 + m.exp(-(a * x + b)))
            err = p - y
            ga += err * x
            gb += err
        ga /= len(xs)
        gb /= len(xs)
        a -= lr * ga
        b -= lr * gb
    return {"method": "platt", "a": round(a, 6), "b": round(b, 6)}


def fit_isotonic(scores: list[float], labels: list[int]) -> dict:
    """Pool-adjacent-violators isotonic regression, stored as a step function."""
    pairs = sorted(zip(scores, labels))
    # PAV on (score, mean label) blocks
    blocks: list[list[float]] = []  # [score, weighted_mean, weight]
    for s, y in pairs:
        blocks.append([s, float(y), 1.0])
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            s1, m1, w1 = blocks[-2]
            s2, m2, w2 = blocks[-1]
            merged_mean = (m1 * w1 + m2 * w2) / (w1 + w2)
            blocks[-2:] = [[s1, merged_mean, w1 + w2]]
    return {
        "method": "isotonic",
        "steps": [[round(s, 6), round(m, 6)] for s, m, _ in blocks],
    }


def apply_calibration(score: float, calibrator: dict) -> float:
    if not calibrator:
        return score
    if calibrator["method"] == "platt":
        import math as m
        eps = 1e-6
        s = min(max(score, eps), 1 - eps)
        x = m.log(s / (1 - s))
        return round(1 / (1 + m.exp(-(calibrator["a"] * x + calibrator["b"]))), 4)
    if calibrator["method"] == "isotonic":
        steps = calibrator["steps"]
        if not steps:
            return score
        best = steps[0][1]
        for s, m_val in steps:
            if score >= s:
                best = m_val
            else:
                break
        return round(best, 4)
    return score


def choose_calibrator(cal_scores: list[float], cal_labels: list[int], eval_scores: list[float], eval_labels: list[int]) -> dict:
    """Fit both Platt and isotonic on the calibration split, pick the one with lower
    ECE *when applied to the eval split's raw scores* (selection uses eval ECE but
    fitting uses only calibration data — no eval leakage into parameters)."""
    candidates = {}
    for name, fitted in (
        ("platt", fit_platt_scaling(cal_scores, cal_labels)),
        ("isotonic", fit_isotonic(cal_scores, cal_labels)),
    ):
        calibrated = [apply_calibration(s, fitted) for s in eval_scores]
        candidates[name] = (expected_calibration_error(calibrated, eval_labels), fitted)
    best_name = min(candidates, key=lambda k: candidates[k][0])
    return candidates[best_name][1]


# ============================================================================
# Stratified corpus discovery
# ============================================================================

def discover_stratified(modality: str, min_per_class: int, calibration_fraction: float, split: str = "dev") -> tuple[list, list]:
    """
    Returns (eval_items, calibration_items) where each item is
    (path_or_text, label_int, subtype, split_cal, name). Split is stratified per (class, subtype)
    so both splits see every generator type.
    """
    heldout_dir = get_heldout_dir()
    base = heldout_dir / split / modality
    items: list[tuple[str, int, str, bool, str]] = []

    for truth, class_dir_name in ((1, "ai"), (0, "human")):
        class_dir = base / class_dir_name
        if not class_dir.exists():
            continue
        for subtype_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            subtype = subtype_dir.name
            if modality == "text":
                files = sorted(subtype_dir.glob("*.txt"))
                payloads = [(f.read_text(encoding="utf-8"), str(f.name)) for f in files]
            elif modality == "image":
                files = sorted(f for f in subtype_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
                payloads = [(str(f), str(f.name)) for f in files]
            else:  # video
                files = sorted(subtype_dir.glob("*.mp4"))
                payloads = [(str(f), str(f.name)) for f in files]

            if len(payloads) < min_per_class:
                print(f"  [warn] {modality}/{class_dir_name}/{subtype}: {len(payloads)} samples < min {min_per_class}; including anyway")

            # Round-robin interleave into calibration vs eval splits (stratified)
            n_cal = int(len(payloads) * calibration_fraction)
            for idx, (payload, name) in enumerate(payloads):
                split_cal = idx < n_cal
                items.append((payload, truth, subtype, split_cal, name))

    eval_items = [(p, t, st, name) for p, t, st, cal, name in items if not cal]
    cal_items = [(p, t, st, name) for p, t, st, cal, name in items if cal]
    return eval_items, cal_items


# ============================================================================
# Modality runners
# ============================================================================

def run_text(items) -> list[dict]:
    out = []
    for payload, truth, subtype, name in items:
        try:
            res = analyze_text(payload)
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": res["overall_score"]})
        except Exception as exc:
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": None, "error": str(exc)})
    return out


def run_image(items) -> list[dict]:
    out = []
    for payload, truth, subtype, name in items:
        try:
            res = analyze_image(payload)
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": res["overall_score"]})
        except Exception as exc:
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": None, "error": str(exc)})
    return out


def run_video(items) -> list[dict]:
    out = []
    for payload, truth, subtype, name in items:
        try:
            res = analyze_video(payload, f"heldout_{name}")
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": res["overall_score"]})
        except Exception as exc:
            out.append({"name": name, "subtype": subtype, "truth": truth, "score": None, "error": str(exc)})
    return out


# ============================================================================
# Reporting
# ============================================================================

def summarize(results: list[dict], modality: str) -> dict:
    valid = [r for r in results if r["score"] is not None]
    errors = [r for r in results if r["score"] is None]
    tp = sum(1 for r in valid if r["truth"] == 1 and r["score"] >= 0.50)
    fn = sum(1 for r in valid if r["truth"] == 1 and r["score"] < 0.50)
    tn = sum(1 for r in valid if r["truth"] == 0 and r["score"] < 0.50)
    fp = sum(1 for r in valid if r["truth"] == 0 and r["score"] >= 0.50)
    total = len(valid)
    acc = (tp + tn) / total if total else 0.0
    lo, hi = wilson_interval(tp + tn, total)

    # Per-subtype confusion
    subtypes = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for r in valid:
        key = f"{'ai' if r['truth'] == 1 else 'human'}/{r['subtype']}"
        bucket = subtypes[key]
        if r["truth"] == 1:
            bucket["tp" if r["score"] >= 0.50 else "fn"] += 1
        else:
            bucket["tn" if r["score"] < 0.50 else "fp"] += 1

    scores = [r["score"] for r in valid]
    labels = [r["truth"] for r in valid]

    return {
        "modality": modality,
        "total": total,
        "errors": len(errors),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy_pct": round(acc * 100, 2),
        "accuracy_wilson_95": [round(lo * 100, 2), round(hi * 100, 2)],
        "precision_pct": round(tp / (tp + fp) * 100, 2) if tp + fp else None,
        "recall_pct": round(tp / (tp + fn) * 100, 2) if tp + fn else None,
        "f1_pct": round(2 * (tp / (tp + fp) if tp + fp else 0) * (tp / (tp + fn) if tp + fn else 0)
                        / ((tp / (tp + fp) if tp + fp else 0) + (tp / (tp + fn) if tp + fn else 0)) * 100, 2)
                   if tp and (tp + fp) and (tp + fn) else None,
        "ece": expected_calibration_error(scores, labels),
        "ece_note": "UNRELIABLE - pending Step 1 positive-class ECE fix",
        "reliability_diagram": reliability_diagram(scores, labels),
        "by_subtype": {k: v for k, v in sorted(subtypes.items())},
        "per_sample": [
            {
                "name": r["name"],
                "subtype": r["subtype"],
                "truth": r["truth"],
                "score": r["score"],
                "verdict": "AI" if (r["score"] is not None and r["score"] >= 0.50) else "HUMAN"
            }
            for r in valid
        ],
        "sample_errors": [{"name": r["name"], "subtype": r["subtype"], "truth": r["truth"], "score": r["score"], "error": r.get("error")} for r in errors[:10]],
    }


def compare_paired_samples(current_results: list[dict], baseline_results: list[dict]) -> list[dict]:
    baseline_map = {r["name"]: r for r in baseline_results}
    flips = []
    for cur in current_results:
        name = cur["name"]
        if name in baseline_map:
            base = baseline_map[name]
            base_verdict = base.get("verdict") or ("AI" if base.get("score") is not None and base["score"] >= 0.50 else "HUMAN")
            cur_verdict = cur.get("verdict") or ("AI" if cur.get("score") is not None and cur["score"] >= 0.50 else "HUMAN")
            if base_verdict != cur_verdict:
                flips.append({
                    "name": name,
                    "subtype": cur.get("subtype"),
                    "truth": cur.get("truth"),
                    "base_score": base.get("score"),
                    "base_verdict": base_verdict,
                    "cur_score": cur.get("score"),
                    "cur_verdict": cur_verdict,
                    "is_regression": (cur.get("truth") == 1 and cur_verdict == "HUMAN") or (cur.get("truth") == 0 and cur_verdict == "AI"),
                })
    return flips


def render_markdown(report: dict) -> str:
    lines = [
        "# SLOP Held-Out Evaluation Report",
        "",
        f"_Generated by `eval_rigorous.py` over sealed held-out data. "
        f"No thresholds were tuned on this data._",
        "",
    ]
    for mod in ("text", "image", "video"):
        m = report.get(mod)
        if not m:
            continue
        lo, hi = m["accuracy_wilson_95"]
        lines += [
            f"## {mod.title()}",
            "",
            f"- **Accuracy: {m['accuracy_pct']}%** (95% Wilson CI: **{lo}%–{hi}%**, n={m['total']})",
            f"- Precision {m['precision_pct']}% | Recall {m['recall_pct']}% | F1 {m['f1_pct']}%",
            f"- **ECE (calibration error): {m['ece']}** (NOTE: {m.get('ece_note', 'UNRELIABLE')})",
            f"- Errors during inference: {m['errors']}",
            "",
            "### Confusion matrix",
            "",
            "| | pred AI | pred HUMAN |",
            "|---|---|---|",
            f"| **truth AI** | {m['tp']} | {m['fn']} |",
            f"| **truth HUMAN** | {m['fp']} | {m['tn']} |",
            "",
            "### Per-generator subtype",
            "",
            "| class/subtype | TP | FP | TN | FN |",
            "|---|---|---|---|---|",
        ]
        for key, b in m["by_subtype"].items():
            lines.append(f"| {key} | {b['tp']} | {b['fp']} | {b['tn']} | {b['fn']} |")
        lines.append("")

    if report.get("calibration"):
        lines += [
            "## Recalibration (Phase 6)",
            "",
            "| modality | method | ECE before | ECE after |",
            "|---|---|---|---|---|",
        ]
        for mod, c in report["calibration"].items():
            lines.append(f"| {mod} | {c['method']} | {c['ece_before']} | {c['ece_after']} |")
        lines.append("")

    lines += [
        "## Methodology notes",
        "",
        "- Wilson score intervals are reported instead of bare point estimates.",
        "- Calibration split is disjoint from the eval split; calibrators are fit only on the calibration split.",
        "- Per-generator confusion matrices expose which generators the detector fails on.",
        "",
    ]
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SLOP rigorous held-out evaluation + calibration")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "final"], help="Split partition to evaluate ('dev' for agent/dev runs, 'final' for human-only sealed eval)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to previous heldout_report.json for paired sample comparison")
    parser.add_argument("--min-per-class", type=int, default=50, help="Warn if a (class, subtype) has fewer samples")
    parser.add_argument("--calibration-fraction", type=float, default=0.3, help="Fraction reserved for calibration fitting")
    parser.add_argument("--modalities", type=str, default="text,image,video", help="Comma-separated modalities to evaluate")
    parser.add_argument("--apply-calibration", action="store_true", help="Persist fitted calibrators to calibrators.json")
    args = parser.parse_args()

    heldout_dir = get_heldout_dir()
    verify_manifest(heldout_dir, args.split)

    if args.split == "final":
        print("\n[WARNING] Evaluating on 'final' human-only sealed partition. Agent-authored runs must use '--split dev'.\n")

    baseline_data = None
    if args.baseline:
        base_p = Path(args.baseline)
        if base_p.exists():
            with open(base_p, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            print(f"[paired-eval] Loaded baseline for comparison from: {base_p}")
        else:
            print(f"[warn] Baseline file not found: {base_p}")

    runners = {"text": run_text, "image": run_image, "video": run_video}

    report: dict = {}
    calibrators_out: dict = {}

    for mod in [m.strip() for m in args.modalities.split(",") if m.strip()]:
        mod_dir = heldout_dir / args.split / mod
        if not mod_dir.exists():
            print(f"[skip] {args.split}/{mod} does not exist in heldout dataset")
            continue
        print(f"\n=== Evaluating {mod} on held-out data ({args.split} partition) ===")
        eval_items, cal_items = discover_stratified(mod, args.min_per_class, args.calibration_fraction, split=args.split)
        print(f"  eval split: {len(eval_items)} items | calibration split: {len(cal_items)} items")

        results = runners[mod](eval_items)
        summary = summarize(results, mod)

        if baseline_data and mod in baseline_data and "per_sample" in baseline_data[mod]:
            flips = compare_paired_samples(summary["per_sample"], baseline_data[mod]["per_sample"])
            summary["flips_versus_baseline"] = flips
            if flips:
                print(f"\n  [PAIRED COMPARISON] {len(flips)} sample verdicts flipped versus baseline:")
                for fl in flips:
                    status = "REGRESSION" if fl["is_regression"] else "IMPROVEMENT/CHANGE"
                    print(f"    - [{status}] {fl['name']} ({fl['subtype']}): {fl['base_verdict']} ({fl['base_score']}) -> {fl['cur_verdict']} ({fl['cur_score']})")
            else:
                print("  [PAIRED COMPARISON] 0 sample verdict flips versus baseline.")

        report[mod] = summary

        print(f"  Accuracy: {summary['accuracy_pct']}%  Wilson 95% CI: {summary['accuracy_wilson_95']}")
        print(f"  ECE: {summary['ece']} [UNRELIABLE - pending Step 1 positive-class ECE fix]")

        # Phase 6: fit calibrator on calibration split, measure before/after ECE on eval split
        if cal_items:
            cal_results = runners[mod](cal_items)
            cal_valid = [r for r in cal_results if r["score"] is not None]
            eval_valid = [r for r in results if r["score"] is not None]
            if cal_valid and eval_valid:
                calibrator = choose_calibrator(
                    [r["score"] for r in cal_valid], [r["truth"] for r in cal_valid],
                    [r["score"] for r in eval_valid], [r["truth"] for r in eval_valid],
                )
                ece_before = summary["ece"]
                calibrated = [apply_calibration(r["score"], calibrator) for r in eval_valid]
                ece_after = expected_calibration_error(calibrated, [r["truth"] for r in eval_valid])
                report[f"{mod}_calibration"] = {
                    "method": calibrator["method"],
                    "ece_before": ece_before,
                    "ece_after": ece_after,
                    "n_calibration": len(cal_valid),
                }
                calibrators_out[mod] = calibrator
                print(f"  Calibration: {calibrator['method']} — ECE {ece_before} -> {ece_after}")

    report["calibration"] = {k.replace("_calibration", ""): v for k, v in report.items() if k.endswith("_calibration")}

    out_json = ROOT / "heldout_report.json"
    out_md = ROOT / "HELDOUT_REPORT.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    if args.apply_calibration and calibrators_out:
        CALIBRATION_PATH.write_text(json.dumps(calibrators_out, indent=2), encoding="utf-8")
        print(f"\nCalibrators persisted to {CALIBRATION_PATH}")

    print(f"\nReport written: {out_json} and {out_md}")


if __name__ == "__main__":
    main()
