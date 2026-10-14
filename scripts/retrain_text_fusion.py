"""
Retrain the text fusion model — PROPERLY.

Fixes over the degenerate first attempt:
  1. Three-way split: TRAIN (raid_train/) fit; VAL (raid_val/) promotion decisions;
     TEST (heldout_set/) touched ONCE at the end, never used for any decision.
  2. Promotion gate: balanced accuracy + macro-F1 + HARD FLOOR on minority-class
     (human) recall >= 0.80. A constant "always AI" classifier cannot pass.
  3. Cross-validated regularization-strength search instead of a fixed default.
  4. Coefficient sanity check: fails the run if |intercept| dominates all feature
     weights (the degenerate constant-output signature).
  5. Out-of-distribution human sanity check (casual text, short text) before promotion.
  6. Audit log: every promoted version records train composition, val/test metrics,
     and per-sample prediction diffs vs the previous model.
"""

import json
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
backend_path = root / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import _normalize_and_sanitize_text, _heuristic, _model_prob  # noqa: E402
from app.ml.binoculars import compute_binoculars_and_curvature  # noqa: E402
from app.ml.perplexity import compute_perplexity, compute_coherence, evict_perplexity_model  # noqa: E402

TRAIN_DIR = root / "raid_train" / "text"
VAL_DIR = root / "raid_val" / "text"
TEST_DIR = root / "heldout_set" / "text"
FUSION_PKL = root / "Model" / "fusion" / "text.pkl"
AUDIT_LOG = root / "Model" / "fusion" / "retrain_audit_log.jsonl"

FEATURES = [
    "neural_prob", "heuristic_score", "llm_pattern_score",
    "transitional_entropy", "burstiness", "contraction_score",
    "zipf_deviation", "vocab_richness",
    "binoculars_ai_prob", "fast_detectgpt_ai_prob",
    "lm_perplexity", "coherence_mean", "coherence_var",
]

# ---- Promotion gate thresholds (Tier 0 fix) ----
MIN_BALANCED_ACC = 0.75
MIN_MACRO_F1 = 0.70
MIN_HUMAN_RECALL = 0.80  # hard floor: a degenerate always-AI model cannot pass
MAX_INTERCEPT_DOMINANCE = 3.0  # |intercept| must not exceed 3x the largest |weight|


def extract_row(text: str) -> list[float]:
    clean = _normalize_and_sanitize_text(text)
    h = _heuristic(clean)
    mp = _model_prob(clean)
    b = compute_binoculars_and_curvature(clean)
    sig = h["signals"]
    ppl = compute_perplexity(clean)
    coh = compute_coherence(clean)
    import math
    return [
        mp if mp is not None else 0.0,
        h["overall_score"],
        sig.get("llm_pattern_score", 0.0),
        sig.get("transitional_entropy_variance", 0.5),
        sig.get("burstiness", 0.5),
        sig.get("contraction_avoidance", 0.5),
        sig.get("zipf_deviation", 0.5),
        sig.get("vocabulary_richness", 0.5),
        float(b.get("binoculars_ai_prob", 0.70)),
        float(b.get("fast_detectgpt_ai_prob", 0.70)),
        ppl["perplexity"] if ppl["available"] and not math.isnan(ppl["perplexity"]) else 0.0,
        coh["coherence_mean"] if coh["available"] and not math.isnan(coh["coherence_mean"]) else 0.0,
        coh["coherence_var"] if coh["available"] and not math.isnan(coh["coherence_var"]) else 0.0,
    ]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic_l2(X, y, l2_reg, epochs=6000, lr=0.05):
    n, d = X.shape
    # class weights: balance AI vs human regardless of sampling imbalance
    n1 = max(float(np.sum(y == 1)), 1.0)
    n0 = max(float(np.sum(y == 0)), 1.0)
    sw = np.where(y == 1, 0.5 * (n0 + n1) / n1, 0.5 * (n0 + n1) / n0)
    sw_n = sw / sw.mean()  # keep gradient scale comparable
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = (p - y) * sw_n
        w -= lr * ((X.T @ err) / n + l2_reg * w / n)
        b -= lr * np.mean(err)
    return w, b


def kfold_indices(n: int, k: int = 5, seed: int = 42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def cv_select_l2(X, y, candidates=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0), k: int = 5):
    """5-fold CV over regularization strength, maximizing balanced accuracy."""
    folds = kfold_indices(len(y), k)
    results = {}
    for l2 in candidates:
        fold_bacc = []
        for f in folds:
            test_idx = f
            train_idx = np.setdiff1d(np.arange(len(y)), f)
            w, b = fit_logistic_l2(X[train_idx], y[train_idx], l2_reg=l2, epochs=3000)
            p = sigmoid(X[test_idx] @ w + b) >= 0.5
            yt = y[test_idx]
            tp = float(np.sum((p == 1) & (yt == 1)))
            tn = float(np.sum((p == 0) & (yt == 0)))
            fp = float(np.sum((p == 1) & (yt == 0)))
            fn = float(np.sum((p == 0) & (yt == 1)))
            rec1 = tp / (tp + fn) if tp + fn else 0.0
            rec0 = tn / (tn + fp) if tn + fp else 0.0
            fold_bacc.append(0.5 * (rec1 + rec0))
        results[l2] = float(np.mean(fold_bacc))
        print(f"  l2={l2:5.1f} -> CV balanced acc = {results[l2]:.4f}", flush=True)
    best_l2 = max(results, key=results.get)
    return best_l2, results


def load_split(base: Path, use_cache: bool = True) -> tuple[list, list, list]:
    """Returns (rows, labels, ids) for a split directory.

    Feature extraction is slow (serialized GPU inference); results are cached
    per-file in <base>/_feature_cache.json so interrupted runs resume.
    """
    rows, labels, ids = [], [], []
    cache_path = base / "_feature_cache.json"
    cache = {}
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    dirty = False
    for label_dir, y in (("ai", 1), ("human", 0)):
        d = base / label_dir
        if not d.exists():
            continue
        for gen_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            for f in sorted(gen_dir.glob("*.txt")):
                sid = f"{label_dir}/{gen_dir.name}/{f.name}"
                ids.append(sid)
                labels.append(y)
                if use_cache and sid in cache and len(cache[sid]) == len(FEATURES):
                    rows.append(cache[sid])
                    continue
                rows.append(extract_row(f.read_text(encoding="utf-8")))
                cache[sid] = rows[-1]
                dirty = True
                if dirty and sum(1 for v in cache.values() if len(v) == len(FEATURES)) % 10 == 0:
                    cache_path.write_text(json.dumps(cache))
                    print(f"      cached {sum(1 for v in cache.values() if len(v) == len(FEATURES))} 13-feat rows ({sid})", flush=True)
    if dirty:
        cache_path.write_text(json.dumps(cache))
    return rows, labels, ids


def score_with(weights: dict, row: list[float]) -> float:
    x = np.array(row, dtype=np.float64)
    mu = np.array(weights.get("feature_mean", [0.0] * len(FEATURES)), dtype=np.float64)
    sd = np.array(weights.get("feature_std", [1.0] * len(FEATURES)), dtype=np.float64)
    xz = (x - mu) / np.where(sd < 1e-9, 1.0, sd)
    logit = float(weights.get("intercept", 0.0))
    for name, val in zip(FEATURES, xz):
        logit += float(weights.get("weights", {}).get(name, 0.0)) * float(val)
    return sigmoid(logit)


def evaluate(weights: dict, base: Path) -> dict:
    rows, labels, ids = load_split(base)
    tp = fp = tn = fn = 0
    per_gen = {}
    per_sample = {}
    for row, y, sid in zip(rows, labels, ids):
        s = score_with(weights, row)
        pred = 1 if s >= 0.5 else 0
        key = sid.split("/")[1]
        g = per_gen.setdefault(f"{sid.split('/')[0]}/{key}", {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        bucket = "tp" if (pred == 1 and y == 1) else "fn" if (pred == 0 and y == 1) else "fp" if pred == 1 else "tn"
        g[bucket] += 1
        if y == 1:
            tp += pred == 1; fn += pred == 0
        else:
            tn += pred == 0; fp += pred == 1
        per_sample[sid] = round(float(s), 4)
    rec1 = tp / (tp + fn) if tp + fn else 0.0
    rec0 = tn / (tn + fp) if tn + fp else 0.0
    prec1 = tp / (tp + fp) if tp + fp else 0.0
    prec0 = tn / (tn + fn) if tn + fn else 0.0
    f1_ai = 2 * prec1 * rec1 / (prec1 + rec1) if prec1 + rec1 else 0.0
    f1_human = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
    return {
        "n": tp + fp + tn + fn,
        "balanced_accuracy": round(0.5 * (rec1 + rec0), 4),
        "accuracy": round((tp + tn) / (tp + fp + tn + fn), 4) if (tp + fp + tn + fn) else 0.0,
        "macro_f1": round(0.5 * (f1_ai + f1_human), 4),
        "ai_recall": round(rec1, 4),
        "human_recall": round(rec0, 4),
        "ai_precision": round(prec1, 4),
        "human_precision": round(prec0, 4),
        "per_gen": per_gen,
        "per_sample": per_sample,
    }


def coefficient_sanity(model: dict) -> tuple[bool, str]:
    """Fail if |intercept| dominates all feature weights (constant-classifier signature)."""
    intercept = abs(float(model["intercept"]))
    max_w = max((abs(v) for v in model["weights"].values()), default=0.0)
    if max_w == 0:
        return False, "all feature weights are zero"
    ratio = intercept / max_w
    ok = ratio <= MAX_INTERCEPT_DOMINANCE
    return ok, f"|intercept|={intercept:.3f} vs max|weight|={max_w:.3f} (ratio {ratio:.2f}, limit {MAX_INTERCEPT_DOMINANCE})"


OOD_HUMAN_SAMPLES = [
    # casual chat, non-native English, short texts — must NOT be flagged as AI
    "lol yeah i saw that movie last night it was pretty good tbh, the ending was kinda weird tho",
    "omw to the store, need anything? text me back",
    "I am liking very much this city, the peoples here is friendly and food is delicious.",
    "hey sorry for late reply!! was busy with work stuff, will look at it tomorrow morning",
    "idk man, sounds kinda sus to me but ok whatever you say lol",
    "The meeting is at 3pm tomorrow. Please bring the report. Thanks!",
    "u there? need quick help with the printer again sry",
    "just got back from vacation, will catch up on emails this week",
]


def ood_human_sanity(model: dict) -> tuple[bool, float]:
    """Fraction of OOD human texts that must stay below 0.5."""
    flagged = 0
    for text in OOD_HUMAN_SAMPLES:
        s = score_with(model, extract_row(text))
        if s >= 0.5:
            flagged += 1
    ok = flagged <= len(OOD_HUMAN_SAMPLES) * 0.25  # allow <=25% OOD FPR
    return ok, flagged / len(OOD_HUMAN_SAMPLES)


def main():
    print("=== Tier 1 retrain: balanced data, val/test discipline, real promotion gate ===\n")

    # ---- 1. Load TRAIN and VAL ----
    # Force re-extraction: train set changed since the last cache (552 -> 516 samples)
    print("[1] Extracting TRAIN features (cached per file)...")
    X_train_rows, y_train, ids_train = load_split(TRAIN_DIR)
    X_train = np.array(X_train_rows, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    n_ai, n_hu = int(y_train.sum()), int((1 - y_train).sum())
    print(f"    train: n={len(y_train)} (AI={n_ai}, human={n_hu}, ratio={n_ai / max(n_hu, 1):.2f}:1)")

    print("[2] Extracting VAL features...")
    X_val_rows, y_val, ids_val = load_split(VAL_DIR)
    evict_perplexity_model()  # free VRAM before evaluation passes
    X_val = np.array(X_val_rows, dtype=np.float64)
    y_val = np.array(y_val, dtype=np.float64)
    print(f"    val: n={len(y_val)} (AI={int(y_val.sum())}, human={int((1 - y_val).sum())})")

    if len(y_val) == 0:
        print("FATAL: raid_val/ is empty — refusing to make promotion decisions without a val set.")
        sys.exit(2)

    # ---- 2. CV regularization selection on TRAIN only ----
    print("[3] CV over L2 strength (train only)...")
    best_l2, cv_results = cv_select_l2(X_train, y_train)
    print(f"    selected l2={best_l2}")

    w, b = fit_logistic_l2(X_train, y_train, l2_reg=best_l2, epochs=8000)
    # Standardize using TRAIN stats only; stored so inference reproduces it.
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xz_train = (X_train - mu) / sd
    # Refit in standardized space (CV selected l2 applies to scaled data)
    best_l2z, cv_results_z = cv_select_l2(Xz_train, y_train)
    print(f"    selected l2 (scaled space)={best_l2z}")
    w, b = fit_logistic_l2(Xz_train, y_train, l2_reg=best_l2z, epochs=8000)
    new_model = {
        "weights": {name: float(wi) for name, wi in zip(FEATURES, w)},
        "intercept": float(b),
        "feature_mean": [float(m) for m in mu],
        "feature_std": [float(s) for s in sd],
        "features": FEATURES,
        "trained_on": {
            "source": "RAID train split (multi-offset)",
            "n": len(y_train), "ai": n_ai, "human": n_hu,
            "generators": sorted({i.split("/")[1] for i in ids_train if i.startswith("ai")}),
        },
        "cv_selected_l2": best_l2z,
        "cv_results": {str(k): v for k, v in cv_results_z.items()},
        "created": datetime.now(timezone.utc).isoformat(),
    }

    # ---- 3. Coefficient sanity check ----
    ok, msg = coefficient_sanity(new_model)
    print(f"[4] Coefficient sanity: {msg} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("FATAL: degenerate model (intercept dominates). Not promoting anything.")
        (root / "raid_text_model_candidate.pkl").write_bytes(pickle.dumps(new_model))
        sys.exit(3)

    # ---- 4. Evaluate OLD vs NEW on VAL (promotion decisions use VAL only) ----
    old_model = pickle.loads(FUSION_PKL.read_bytes()) if FUSION_PKL.exists() else {"weights": {}, "intercept": 0.0}
    old_val = evaluate(old_model, VAL_DIR)
    new_val = evaluate(new_model, VAL_DIR)
    print(f"\n[5] VAL:  old bacc={old_val['balanced_accuracy']:.3f} macroF1={old_val['macro_f1']:.3f} humanRec={old_val['human_recall']:.3f}")
    print(f"     VAL:  new bacc={new_val['balanced_accuracy']:.3f} macroF1={new_val['macro_f1']:.3f} humanRec={new_val['human_recall']:.3f}")

    # ---- 5. OOD human sanity ----
    ok, ood_fpr = ood_human_sanity(new_model)
    print(f"[6] OOD human sanity: OOD FPR={ood_fpr:.2f} -> {'PASS' if ok else 'FAIL'}")

    # ---- 6. Promotion gate on VAL metrics ----
    gate = (
        new_val["balanced_accuracy"] >= MIN_BALANCED_ACC
        and new_val["macro_f1"] >= MIN_MACRO_F1
        and new_val["human_recall"] >= MIN_HUMAN_RECALL
        and new_val["balanced_accuracy"] > old_val["balanced_accuracy"]
        and ok
    )
    print(f"[7] Promotion gate (bacc>={MIN_BALANCED_ACC}, macroF1>={MIN_MACRO_F1}, humanRec>={MIN_HUMAN_RECALL}, beats-old, OOD-ok): {'PASS' if gate else 'REJECT'}")

    audit_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "train_composition": new_model["trained_on"],
        "cv_selected_l2": best_l2z,
        "coefficients": {"intercept": new_model["intercept"], **new_model["weights"]},
        "val_metrics_old": {k: v for k, v in old_val.items() if k != "per_sample"},
        "val_metrics_new": {k: v for k, v in new_val.items() if k != "per_sample"},
        "ood_human_fpr": ood_fpr,
        "gate_passed": gate,
        "promoted": False,
    }

    if not gate:
        (root / "raid_text_model_candidate.pkl").write_bytes(pickle.dumps(new_model))
        audit_entry["promoted"] = False
        with open(AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")
        print("\nNOT PROMOTED — candidate saved to raid_text_model_candidate.pkl; audit logged.")
        return

    # ---- 7. Promote, then touch TEST exactly once ----
    if FUSION_PKL.exists():
        shutil.copy2(FUSION_PKL, root / "Model" / "fusion" / "text_prev_promoted.pkl")
    FUSION_PKL.write_bytes(pickle.dumps(new_model))
    print(f"\n[8] PROMOTED to {FUSION_PKL}")

    print("[9] TEST split (first and only touch, reported honestly)...")
    test_res = evaluate(new_model, TEST_DIR)
    old_test = evaluate(old_model, TEST_DIR)
    print(f"    TEST old: bacc={old_test['balanced_accuracy']:.3f} acc={old_test['accuracy']:.3f} macroF1={old_test['macro_f1']:.3f} aiRec={old_test['ai_recall']:.3f} humanRec={old_test['human_recall']:.3f}")
    print(f"    TEST new: bacc={test_res['balanced_accuracy']:.3f} acc={test_res['accuracy']:.3f} macroF1={test_res['macro_f1']:.3f} aiRec={test_res['ai_recall']:.3f} humanRec={test_res['human_recall']:.3f}")

    audit_entry["promoted"] = True
    audit_entry["test_metrics_new"] = {k: v for k, v in test_res.items() if k != "per_sample"}
    audit_entry["test_metrics_old"] = {k: v for k, v in old_test.items() if k != "per_sample"}
    audit_entry["test_per_sample_new"] = test_res["per_sample"]
    audit_entry["test_per_sample_old"] = old_test["per_sample"]
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_entry) + "\n")
    print(f"\nAudit entry appended to {AUDIT_LOG}")
    print("Done.")


if __name__ == "__main__":
    main()
