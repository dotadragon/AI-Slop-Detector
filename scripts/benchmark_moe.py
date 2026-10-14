"""Per-generator mixture-of-experts text scorer (Phase D12) — BENCHMARK ONLY.

Architecture:
  1. Router: softmax classifier over standardized features predicting
     {human, gpt2, llama_chat, mistral, mistral_chat, mpt, mpt_chat}.
  2. Experts: one binary logistic expert per AI family (trained family-vs-human).
  3. Inference: if router says human with confidence >= ROUTER_MARGIN -> expert
     for the predicted family; final score = that expert's P(AI).

This is a BENCHMARK script: nothing is promoted. It reports val balanced
accuracy vs the production single-fusion baseline so the enable decision is
data-driven. The gate is the same as retrain_text_fusion.py.

Usage: python benchmark_moe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retrain_text_fusion import (  # noqa: E402
    FEATURES,
    TRAIN_DIR,
    VAL_DIR,
    fit_logistic_l2,
    load_split,
    sigmoid,
)

FAMILIES = ["gpt2", "llama_chat", "mistral", "mistral_chat", "mpt", "mpt_chat"]
ROUTER_MARGIN = 0.55  # router must be this confident to route to human
OUT = Path("moe_benchmark_result.json")


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    return mu, np.where(sd < 1e-9, 1.0, sd)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_router(Xs: np.ndarray, classes: list[str], epochs=4000, lr=0.05, l2=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression (full-batch gradient descent)."""
    n, d = Xs.shape
    k = len(classes)
    W = np.zeros((d, k))
    b = np.zeros(k)
    # class weights: balance each family to equal total mass
    counts = np.array([max(1, int(np.sum(classes[i] == "human" if False else (np.array(CLASS_IDX) == i).sum()))) for i in range(k)], dtype=float)
    w_map = (n / counts) / (n / counts).mean()
    Y = np.zeros((n, k))
    for i, ci in enumerate(CLASS_IDX):
        Y[i, ci] = 1.0
    sw = w_map[CLASS_IDX]
    sw = sw / sw.mean()
    for _ in range(epochs):
        P = softmax(Xs @ W + b)
        G = (P - Y) * sw[:, None]
        W -= lr * ((Xs.T @ G) / n + l2 * W / n)
        b -= lr * (G.sum(axis=0) / n)
    return W, b


def main() -> None:
    global CLASS_IDX

    print("Loading train split (cached)...", flush=True)
    rows, labels, ids = load_split(TRAIN_DIR)
    X = np.array(rows, dtype=np.float64)
    y = np.array(labels)

    # family label per row: "human" or generator dir name (from id "ai/<gen>/<file>")
    fam = np.array(["human" if l == 0 else i.split("/")[1] for i, l in zip(ids, labels)])
    classes = ["human"] + FAMILIES
    CLASS_IDX = [classes.index(f) for f in fam]

    mu, sd = standardize_fit(X)
    Xs = (X - mu) / sd

    print("Fitting router + experts...", flush=True)
    W, b = fit_router(Xs, classes)

    experts: dict[str, tuple[np.ndarray, float]] = {}
    for f in FAMILIES:
        mask = (fam == f) | (fam == "human")
        Xe, ye = Xs[mask], y[mask]
        best_l2, _ = _cv_l2(Xe, ye)
        w, b0 = fit_logistic_l2(Xe, ye, l2_reg=best_l2)
        experts[f] = (w, b0)
        print(f"  expert[{f}] l2={best_l2} n={int(mask.sum())}", flush=True)

    print("Loading val split (cached)...", flush=True)
    vrows, vlabels, vids = load_split(VAL_DIR)
    Xv = (np.array(vrows, dtype=np.float64) - mu) / sd
    yv = np.array(vlabels)
    vfam = np.array(["human" if l == 0 else i.split("/")[1] for i, l in zip(vids, vlabels)])

    # MoE inference
    P = softmax(Xv @ W + b)
    pred_family = [classes[i] for i in P.argmax(axis=1)]
    router_conf = P.max(axis=1)

    preds = np.zeros(len(yv))
    routed = {"human_direct": 0, "expert": 0, "human_fallback": 0}
    for i in range(len(yv)):
        f = pred_family[i]
        conf = router_conf[i]
        if f == "human" and conf >= ROUTER_MARGIN:
            preds[i] = 0.0
            routed["human_direct"] += 1
        elif f in experts:
            w, b0 = experts[f]
            preds[i] = float(sigmoid(Xv[i] @ w + b0))
            routed["expert"] += 1
        else:
            preds[i] = 0.0
            routed["human_fallback"] += 1

    def metrics(p: np.ndarray, yt: np.ndarray) -> dict:
        ph = (p < 0.5).astype(int)  # human-correct
        yh = (yt == 0).astype(int)
        acc = float((ph == yh).mean())
        rec_ai = float(((p >= 0.5) & (yt == 1)).sum() / max(1, (yt == 1).sum()))
        rec_hu = float(((p < 0.5) & (yt == 0)).sum() / max(1, (yt == 0).sum()))
        return {"bacc": round(0.5 * (rec_ai + rec_hu), 4), "acc": round(acc, 4),
                "ai_recall": round(rec_ai, 4), "human_recall": round(rec_hu, 4)}

    moe_m = metrics(preds, yv)

    # per-family AI recall
    per_family = {}
    for f in FAMILIES:
        m = np.array([x == f for x in vfam])
        if m.sum():
            per_family[f] = {"n": int(m.sum()), "recall": round(float((preds[m] >= 0.5).mean()), 4)}

    baseline = json.loads(Path("Model/fusion/promoted_baseline.json").read_text())
    result = {
        "n_val": len(yv),
        "router_classes": classes,
        "routing_counts": routed,
        "moe_val": moe_m,
        "baseline_val": baseline,
        "gate": {"min_bacc": 0.75, "min_human_recall": 0.80},
        "passes_gate": bool(moe_m["bacc"] >= 0.75 and moe_m["human_recall"] >= 0.80 and moe_m["bacc"] > baseline.get("val_balanced_accuracy", 0)),
        "per_family_ai_recall": per_family,
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def _cv_l2(Xe: np.ndarray, ye: np.ndarray) -> tuple[float, dict]:
    from retrain_text_fusion import cv_select_l2

    return cv_select_l2(Xe, ye)


if __name__ == "__main__":
    main()
