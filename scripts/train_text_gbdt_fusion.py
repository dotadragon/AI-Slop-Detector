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
from retrain_text_fusion import FEATURES, OOD_HUMAN_SAMPLES, extract_row  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ALL_GBDT_FEATURES = ["transformer_prob"] + FEATURES


def load_features_and_transformer(split_dir: Path):
    cache_path = split_dir / "_feature_cache.json"
    trans_path = split_dir / "_transformer_preds.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache missing: {cache_path}")
    if not trans_path.exists():
        raise FileNotFoundError(f"Transformer predictions missing: {trans_path}")

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    trans_preds = json.loads(trans_path.read_text(encoding="utf-8"))

    X, y, ids = [], [], []

    # Iterate over available transformer prediction IDs
    for sid, t_prob in trans_preds.items():
        # Label from prefix
        label = 1 if sid.startswith("ai/") else 0
        if sid in cache:
            row = [t_prob] + cache[sid]
        else:
            # Fallback if row not in feature cache (e.g. newly added samples)
            f_path = split_dir / sid
            if f_path.exists():
                logger.info(f"Extracting 13-feature row for {sid}...")
                feat_row = extract_row(f_path.read_text(encoding="utf-8"))
                cache[sid] = feat_row
                row = [t_prob] + feat_row
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
    train_dir = root / "raid_train" / "text"
    val_dir = root / "raid_val" / "text"
    model_output = root / "Model" / "fusion" / "text_gbdt.pkl"

    logger.info("Loading train and val feature matrices...")
    X_train, y_train, train_ids = load_features_and_transformer(train_dir)
    X_val, y_val, val_ids = load_features_and_transformer(val_dir)

    logger.info(f"Train shape: {X_train.shape}, labels: AI={sum(y_train)}, Human={len(y_train) - sum(y_train)}")
    logger.info(f"Val shape: {X_val.shape}, labels: AI={sum(y_val)}, Human={len(y_val) - sum(y_val)}")

    # Monotonic constraints:
    # 0: transformer_prob (+1)
    # 1: neural_prob (+1)
    # 2: heuristic_score (0)
    # 3: llm_pattern_score (+1)
    # 9: binoculars_ai_prob (+1)
    # 10: fast_detectgpt_ai_prob (+1)
    monotone = [0] * len(ALL_GBDT_FEATURES)
    monotone[0] = 1   # transformer_prob
    monotone[1] = 1   # neural_prob
    monotone[3] = 1   # llm_pattern_score
    monotone[9] = 1   # binoculars_ai_prob
    monotone[10] = 1  # fast_detectgpt_ai_prob

    clf = lgb.LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=7,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        monotone_constraints=monotone,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )

    logger.info("Fitting LightGBM classifier...")
    clf.fit(X_train, y_train)

    train_metrics = evaluate_gbdt(clf, X_train, y_train, train_ids)
    val_metrics = evaluate_gbdt(clf, X_val, y_val, val_ids)

    logger.info(f"Train BAcc: {train_metrics['balanced_accuracy']:.4f} | AI Rec: {train_metrics['ai_recall']:.4f} | Hu Rec: {train_metrics['human_recall']:.4f}")
    logger.info(f"Val BAcc: {val_metrics['balanced_accuracy']:.4f} | AI Rec: {val_metrics['ai_recall']:.4f} | Hu Rec: {val_metrics['human_recall']:.4f} | Macro F1: {val_metrics['macro_f1']:.4f}")

    print("\nPer-Generator Validation Recall:")
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
    imp = dict(zip(ALL_GBDT_FEATURES, [int(x) for x in clf.feature_importances_]))
    print("\nFeature Importances (split count):")
    for feat, count in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:25s}: {count}")

    # Check promotion gates
    bacc_pass = val_metrics["balanced_accuracy"] >= 0.75
    hu_rec_pass = val_metrics["human_recall"] >= 0.80
    f1_pass = val_metrics["macro_f1"] >= 0.70

    logger.info(f"Promotion gate checks: BAcc>=0.75: {bacc_pass}, HuRec>=0.80: {hu_rec_pass}, F1>=0.70: {f1_pass}")

    # Save model package
    model_pkg = {
        "model": clf,
        "features": ALL_GBDT_FEATURES,
        "train_metrics": {k: v for k, v in train_metrics.items() if k != "probs"},
        "val_metrics": {k: v for k, v in val_metrics.items() if k != "probs"},
        "feature_importances": imp,
        "gate_passed": bool(bacc_pass and hu_rec_pass and f1_pass),
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    with open(model_output, "wb") as f:
        pickle.dump(model_pkg, f)
    logger.info(f"Saved GBDT model package to {model_output}")

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
