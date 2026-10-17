"""
Train a LightGBM gradient-boosted decision tree ensemble combining
the fine-tuned transformer predictions with the 13 hand-engineered features.
Enforces sealed data discipline (fit on raid_train, select on raid_val, test sealed).
"""

import argparse
import json
import logging
import math
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

root = Path(__file__).resolve().parent.parent
backend_path = root / "slop" / "backend"
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "scripts"))
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import _normalize_and_sanitize_text, _heuristic  # noqa: E402
from scripts.retrain_text_fusion import FEATURES, OOD_HUMAN_SAMPLES, extract_row  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_GBDT_FEATURES = ["transformer_prob"] + FEATURES + [
    "gltr_top10_frac",
    "gltr_tail_frac",
    "token_ll_std",
    "fast_z_norm",
]


def load_features_and_transformer(split_dir: Path):
    cache_path = split_dir / "_feature_cache.json"
    trans_path = split_dir / "_transformer_preds.json"
    gltr_path = split_dir / "_gltr_preds.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache missing: {cache_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transformer predictions missing: {trans_path}")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    trans_preds = json.loads(trans_path.read_text(encoding="utf-8"))
    gltr_preds = json.loads(gltr_path.read_text(encoding="utf-8")) if gltr_path.exists() else {}

    X, y, ids = [], [], []

    # Iterate over available transformer prediction IDs
    for sid, t_prob in trans_preds.items():
        # Label from prefix
        label = 1 if sid.startswith("ai/") else 0
        gltr_row = gltr_preds.get(sid, [0.5, 0.5, 0.0, 0.0])
        if sid in cache:
            row = [t_prob] + cache[sid] + gltr_row
        else:
            # Fallback if row not in feature cache (e.g. newly added samples)
            f_path = split_dir / sid
            if f_path.exists():
                logger.info(f"Extracting 13-feature row for {sid}...")
                feat_row = extract_row(f_path.read_text(encoding="utf-8"))
                cache[sid] = feat_row
                row = [t_prob] + feat_row + gltr_row
            else:
                continue

        X.append(row)
        y.append(label)
        ids.append(sid)

    cache_path.write_text(json.dumps(cache))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), ids


def evaluate_gbdt(model, X, y, ids):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    tp = float(np.sum((preds == 1) & (y == 1)))
    tn = float(np.sum((preds == 0) & (y == 0)))
    fp = float(np.sum((preds == 1) & (y == 0)))
    fn = float(np.sum((preds == 0) & (y == 1)))

    rec1 = tp / (tp + fn) if (tp + fn) else 0.0
    rec0 = tn / (tn + fp) if (tn + fp) else 0.0
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    prec0 = tn / (tn + fn) if (tn + fn) else 0.0

    bacc = 0.5 * (rec1 + rec0)
    acc = (tp + tn) / len(y) if len(y) else 0.0
    f1_ai = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) else 0.0
    f1_hu = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) else 0.0
    macro_f1 = 0.5 * (f1_ai + f1_hu)

    per_gen = {}
    for sid, p, true_y in zip(ids, preds, y):
        parts = sid.split("/")
        gen_key = f"{parts[0]}/{parts[1]}"
        g = per_gen.setdefault(gen_key, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        if true_y == 1:
            if p == 1:
                g["tp"] += 1
            else:
                g["fn"] += 1
        else:
            if p == 0:
                g["tn"] += 1
            else:
                g["fp"] += 1

    return {
        "balanced_accuracy": round(float(bacc), 4),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "ai_recall": round(float(rec1), 4),
        "human_recall": round(float(rec0), 4),
        "ai_precision": round(float(prec1), 4),
        "human_precision": round(float(prec0), 4),
        "per_gen": per_gen,
        "probs": dict(zip(ids, [round(float(p), 4) for p in probs])),
    }


def main():
    train_dir = (root / "raid_train" / "text") if (root / "raid_train" / "text").exists() else (root / "_archive" / "raid_train" / "text")
    val_dir = (root / "raid_val" / "text") if (root / "raid_val" / "text").exists() else (root / "_archive" / "raid_val" / "text")
    model_output = root / "Model" / "fusion" / "text_gbdt_v2.pkl"

    logger.info("Loading train and val feature matrices...")
    X_train, y_train, train_ids = load_features_and_transformer(train_dir)
    X_val, y_val, val_ids = load_features_and_transformer(val_dir)

    logger.info(f"Train shape: {X_train.shape}, labels: AI={sum(y_train)}, Human={len(y_train) - sum(y_train)}")
    logger.info(f"Val shape: {X_val.shape}, labels: AI={sum(y_val)}, Human={len(y_val) - sum(y_val)}")

    # Monotonic constraints (18 features total):
    # 0: transformer_prob (+1)
    # 1: neural_prob (+1)
    # 2: heuristic_score (0)
    # 3: llm_pattern_score (+1)
    # 9: binoculars_ai_prob (+1)
    # 10: fast_detectgpt_ai_prob (+1)
    # 11: lm_perplexity (-1)
    # 12: coherence_mean (+1)
    # 14: gltr_top10_frac (+1)
    # 15: gltr_tail_frac (-1)
    # 16: token_ll_std (-1)
    # 17: fast_z_norm (+1)
    monotone = [0] * len(ALL_GBDT_FEATURES)
    monotone[0] = 1   # transformer_prob
    monotone[1] = 1   # neural_prob
    monotone[3] = 1   # llm_pattern_score
    monotone[9] = 1   # binoculars_ai_prob
    monotone[10] = 1  # fast_detectgpt_ai_prob
    monotone[11] = -1 # lm_perplexity (higher perplexity must never increase AI probability)
    monotone[12] = 1  # coherence_mean
    monotone[14] = 1  # gltr_top10_frac
    monotone[15] = -1 # gltr_tail_frac
    monotone[16] = -1 # token_ll_std
    monotone[17] = 1  # fast_z_norm

    # Sample weighting: 1.5x on hard generators (ai/mistral, ai/gpt2) and key human subsets (human/email, human/human)
    n_ai = sum(y_train)
    n_hu = len(y_train) - n_ai
    w_ai = len(y_train) / (2.0 * n_ai)
    w_hu = len(y_train) / (2.0 * n_hu)
    weights = np.zeros(len(y_train), dtype=np.float32)
    for i, sid in enumerate(train_ids):
        base = w_ai if y_train[i] == 1 else w_hu
        if "mistral" in sid:
            base *= 1.5
        elif "gpt2" in sid:
            base *= 1.5
        elif "email" in sid:
            base *= 1.5
        elif sid.startswith("human/human"):
            base *= 1.5
        weights[i] = base

    from sklearn.model_selection import StratifiedKFold

    candidates = [
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 6, "min_child_samples": 35, "reg_lambda": 20.0},
        {"n_estimators": 80, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 5, "min_child_samples": 30, "reg_lambda": 15.0},
        {"n_estimators": 120, "learning_rate": 0.04, "max_depth": 3, "num_leaves": 6, "min_child_samples": 35, "reg_lambda": 25.0},
        {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 4, "num_leaves": 7, "min_child_samples": 30, "reg_lambda": 20.0},
        {"n_estimators": 90, "learning_rate": 0.05, "max_depth": 3, "num_leaves": 6, "min_child_samples": 25, "reg_lambda": 20.0},
    ]

    print("\n--- 5-Fold Stratified Cross-Validation on Training Split ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_candidate: dict[str, float | int] = dict(candidates[0])
    best_cv_bacc = -1.0

    for idx, cand in enumerate(candidates, 1):
        cv_scores = []
        for train_fold, val_fold in skf.split(X_train, y_train):
            clf_cv = lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                subsample=0.8,
                colsample_bytree=1.0,
                monotone_constraints=monotone,
                random_state=42,
                verbosity=-1,
                **cand,
            )
            clf_cv.fit(X_train[train_fold], y_train[train_fold], sample_weight=weights[train_fold])
            preds_cv = (clf_cv.predict_proba(X_train[val_fold])[:, 1] >= 0.5).astype(int)
            bacc_fold = balanced_accuracy_score(y_train[val_fold], preds_cv)
            cv_scores.append(bacc_fold)
        mean_cv = float(np.mean(cv_scores))
        print(f"  Candidate {idx}: {cand} -> 5-Fold CV BAcc = {mean_cv:.4f}")
        if mean_cv > best_cv_bacc:
            best_cv_bacc = mean_cv
            best_candidate = cand

    print(f"\nSelected Best Candidate by 5-Fold CV: {best_candidate} (CV BAcc: {best_cv_bacc:.4f})")

    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        subsample=0.8,
        colsample_bytree=1.0,
        monotone_constraints=monotone,
        random_state=42,
        verbosity=-1,
        **best_candidate,
    )

    logger.info("Fitting best LightGBM classifier with monotonic constraints and hard-generator weighting...")
    clf.fit(X_train, y_train, sample_weight=weights)

    train_metrics = evaluate_gbdt(clf, X_train, y_train, train_ids)
    val_metrics = evaluate_gbdt(clf, X_val, y_val, val_ids)

    logger.info(f"Train BAcc: {train_metrics['balanced_accuracy']:.4f} | AI Rec: {train_metrics['ai_recall']:.4f} | Hu Rec: {train_metrics['human_recall']:.4f}")
    logger.info(f"Val BAcc: {val_metrics['balanced_accuracy']:.4f} | AI Rec: {val_metrics['ai_recall']:.4f} | Hu Rec: {val_metrics['human_recall']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")

    print("\nPer-Generator Validation Recall & Key Subgroups:")
    for gen, stats in sorted(val_metrics["per_gen"].items()):
        if gen.startswith("ai/"):
            n_tot = stats["tp"] + stats["fn"]
            rec = stats["tp"] / n_tot if n_tot else 0.0
            print(f"  {gen:20s}: {stats['tp']:2d}/{n_tot:2d} ({rec:6.2%})")
        else:
            n_tot = stats["tn"] + stats["fp"]
            spec = stats["tn"] / n_tot if n_tot else 0.0
            print(f"  {gen:20s}: {stats['tn']:2d}/{n_tot:2d} ({spec:6.2%} specificity)")

    # Feature importances
    imp = dict(zip(ALL_GBDT_FEATURES, [int(x) for x in clf.feature_importances_.tolist()]))
    print("\nFeature Importances (split count):")
    for feat, count in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:25s}: {count}")

    # Check promotion gates vs Step 3 baseline (87.50% BAcc, 0.8426 F1, hu_spec >= 0.9333)
    step3_bacc = 0.8750
    step3_f1 = 0.8426
    val_bacc = val_metrics["balanced_accuracy"]
    val_f1 = val_metrics["macro_f1"]
    val_hu_spec = val_metrics["human_recall"]

    improved = (val_bacc > step3_bacc or (val_bacc >= step3_bacc and val_f1 > step3_f1)) and (val_hu_spec >= 0.9333)
    logger.info(f"Promotion gate checks: BAcc({val_bacc:.4f}) > {step3_bacc} or (BAcc>={step3_bacc} and F1({val_f1:.4f}) > {step3_f1}), HuSpec({val_hu_spec:.4f}) >= 0.9333 -> Promoted: {improved}")

    # Save model package to text_gbdt_v2.pkl
    model_pkg = {
        "model": clf,
        "features": ALL_GBDT_FEATURES,
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probs"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probs"},
        "feature_importances": imp,
        "gate_passed": improved,
        "best_candidate": best_candidate,
        "cv_bacc": best_cv_bacc,
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as f:
        pickle.dump(model_pkg, f)
    logger.info(f"Saved GBDT model package to {model_output}")

    if improved:
        promoted_output = root / "Model" / "fusion" / "text.pkl"
        with open(promoted_output, "wb") as f:
            pickle.dump(model_pkg, f)
        logger.info(f"PROMOTED 18-feature GBDT model package to {promoted_output}")
    else:
        logger.info(f"18-feature GBDT did not strictly beat Step 3 baseline ({step3_bacc:.4f} BAcc, {step3_f1:.4f} F1). Keeping Step 3 text.pkl untouched.")

    # Write summary json
    summary_path = root / "Model" / "fusion" / "text_gbdt_summary.json"
    summary_path.write_text(json.dumps({
        "train_metrics": model_pkg["train_metrics"],
        "val_metrics": model_pkg["val_metrics"],
        "feature_importances": imp,
        "gate_passed": model_pkg["gate_passed"],
    }, indent=2))
    logger.info(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
