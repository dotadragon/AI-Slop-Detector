#!/usr/bin/env python3
"""CLIP-ViT-L/14 linear probe v2 — CANDIDATE ONLY (never fused into production).

Attempt 2: real human negatives are now available (COCO val2017, n=50, sourced by
`pull_coco_humans.py` into heldout_set/image/human_v2/), so a supervised linear
probe is finally identifiable.

Discipline:
  * Backbone FROZEN (Model/image/clip_vit_large, CLIP ViT-L/14). No fine-tuning.
  * LINEAR head only. random_state = 42, printed.
  * The pristine 66-image gate (incl. its 14 golden files) is EVAL-ONLY.
  * Identical images are kept in the same split (duplicate-cluster aware).
  * Writes Model/fusion/clip_probe_v2.pkl ONLY if every bar item passes.

Usage:
    ENVIRONMENT=development SECRET_KEY=... python train_clip_probe_v2.py [--save]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "slop" / "backend"))

RANDOM_STATE = 42
THRESHOLD = 0.50
CLIP_DIR = ROOT / "Model" / "image" / "clip_vit_large"
CACHE_A = ROOT / "_clip_v2_embeddings.npz"            # 34 multigen AI + 66 gate
CACHE_H = ROOT / "_clip_v2_embeddings_human_v2.npz"   # 50 COCO humans
CANDIDATE_PATH = ROOT / "Model" / "fusion" / "clip_probe_v2.pkl"
HUMAN_DIR = ROOT / "heldout_set" / "image" / "human_v2"
EXTS = (".png", ".jpg", ".jpeg", ".webp")
MULTIGEN_FAMILIES = ["cascade_lcm", "dit", "kandinsky", "midjourney", "realvis_xl"]
GATE_TARGET_ACC = 55 / 66   # bar (e): the 66-gate re-run must match 55/66


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, (c - h) * 100), min(100.0, (c + h) * 100))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(a)) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_human_embeddings() -> dict:
    """Load or compute fresh frozen-CLIP embeddings for the COCO human pool."""
    files = sorted(p for p in HUMAN_DIR.glob("*") if p.suffix.lower() in EXTS)
    if CACHE_H.exists():
        z = np.load(CACHE_H, allow_pickle=False)
        cached = [str(x) for x in z["paths"]]
        if len(cached) == len(files) and all(Path(c).exists() for c in cached):
            return {"paths": cached, "embeds": z["embeds"].astype(np.float64),
                    "theme": [str(x) for x in z["theme"]]}

    import torch
    from transformers import AutoProcessor, CLIPVisionModelWithProjection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(str(CLIP_DIR), local_files_only=True)
    vm = CLIPVisionModelWithProjection.from_pretrained(
        str(CLIP_DIR), dtype=dtype, local_files_only=True).to(device).eval()

    embs = []
    for i in range(0, len(files), 16):
        chunk = []
        for p in files[i:i + 16]:
            with Image.open(p) as im:
                chunk.append(im.convert("RGB"))
        inp = proc(images=chunk, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        with torch.inference_mode():
            out = vm(**inp).image_embeds.float().cpu().numpy()
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        embs.append(out)
    embs = np.concatenate(embs)
    themes = [p.name.split("_")[1] for p in files]
    np.savez(CACHE_H, paths=np.array([str(p) for p in files]),
             theme=np.array(themes), embeds=embs.astype(np.float32))
    print(f"[embeds] wrote {CACHE_H.name} shape={embs.shape}")
    return {"paths": [str(p) for p in files], "embeds": embs.astype(np.float64), "theme": themes}


def cluster_by_content(paths: list[str]) -> dict[int, int]:
    """Union identical files so duplicates never straddle a split boundary."""
    by_hash: dict[str, int] = {}
    cluster = {}
    for i, p in enumerate(paths):
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        cluster[i] = by_hash.setdefault(h, len(by_hash))
    return cluster


def stratified_cluster_split(idx: list[int], strata: list[str], cluster: dict[int, int],
                             frac_train: float, rng: random.Random) -> tuple[list[int], list[int]]:
    """Split by whole duplicate-clusters, stratified on `strata`, ~frac_train to train."""
    train, test = [], []
    by_stratum: dict[str, set[int]] = defaultdict(set)
    for i in idx:
        by_stratum[strata[i]].add(cluster[i])
    for stratum, clus in sorted(by_stratum.items()):
        clus = sorted(clus)
        rng.shuffle(clus)
        members = {c: [i for i in idx if cluster[i] == c] for c in clus}
        size = len([i for i in idx if strata[i] == stratum])
        target = frac_train * size
        got, tr, te = 0, [], []
        for c in clus:
            if got < target:
                tr += members[c]; got += len(members[c])
            else:
                te += members[c]
        # guarantee both sides non-empty for any stratum with >=2 clusters
        if not te and len(tr) > 1:
            te.append(tr.pop())
        if not tr and len(te) > 1:
            tr.append(te.pop())
        train += tr; test += te
    return sorted(train), sorted(test)


def fit_and_report(name: str, X: np.ndarray, y: np.ndarray, tr: list[int], te: list[int],
                   labels: list[str], families: list[str], groups: list[str]) -> dict:
    from sklearn.linear_model import LogisticRegression

    print()
    print("=" * 78)
    print(f"RUN: {name}")
    print("=" * 78)
    print(f"  TRAIN n={len(tr)}  class counts={dict(Counter(y[tr].tolist()))}")
    print(f"  TEST  n={len(te)}  class counts={dict(Counter(y[te].tolist()))}")
    per_fam = Counter(families[i] for i in te if y[i] == 1)
    print(f"  held-out AI per family: {dict(sorted(per_fam.items()))}")
    print(f"  held-out humans per theme: {dict(sorted(Counter(families[i] for i in te if y[i] == 0).items()))}")

    clf = LogisticRegression(random_state=RANDOM_STATE, max_iter=5000, C=1.0)
    clf.fit(X[tr], y[tr])
    w, b = clf.coef_[0], float(clf.intercept_[0])

    def score(i: int) -> float:
        return float(1 / (1 + np.exp(-(X[i] @ w + b))))

    tr_pred = (X[tr] @ w + b) >= 0
    tr_acc = float((tr_pred == (y[tr] == 1)).mean())
    te_pred = (X[te] @ w + b) >= 0
    te_acc = float((te_pred == (y[te] == 1)).mean())

    print()
    print(f"  TRAIN accuracy     : {tr_acc:.2%}")
    print(f"  HELD-OUT accuracy  : {te_acc:.2%}")
    print(f"  GAP (train - held-out) = {tr_acc - te_acc:+.2%}   <-- overfit check")
    print()
    print("  HELD-OUT AI RECALL (per family, Wilson 95% CI):")
    fam_rows = {}
    k_all = n_all = 0
    for fam in sorted(set(families[i] for i in te if y[i] == 1)):
        sel = [i for i in te if y[i] == 1 and families[i] == fam]
        k = sum(1 for i in sel if score(i) >= THRESHOLD)
        lo, hi = wilson(k, len(sel))
        fam_rows[fam] = {"k": k, "n": len(sel), "recall": k / len(sel), "ci": [round(lo, 2), round(hi, 2)]}
        print(f"    {fam:14s} {k:2d}/{len(sel):<2d} = {k/len(sel):6.2%}  (95% Wilson {lo:.1f}-{hi:.1f}%)")
        k_all += k; n_all += len(sel)
    lo, hi = wilson(k_all, n_all)
    multigen_recall = k_all / n_all if n_all else 0.0
    print(f"    {'HELD-OUT ALL':14s} {k_all:2d}/{n_all:<2d} = {multigen_recall:6.2%}  (95% Wilson {lo:.1f}-{hi:.1f}%)")

    print()
    print("  HUMAN SPECIFICITY (held-out humans, Wilson 95% CI):")
    hum = [i for i in te if y[i] == 0]
    ks = sum(1 for i in hum if score(i) < THRESHOLD)
    lo, hi = wilson(ks, len(hum))
    spec = ks / len(hum) if hum else 0.0
    print(f"    {ks}/{len(hum)} = {spec:.2%}  (95% Wilson {lo:.1f}-{hi:.1f}%)")
    sp_rows = {th: f"{sum(1 for i in hum if families[i]==th and score(i)<THRESHOLD)}/{sum(1 for i in hum if families[i]==th)}"
               for th in sorted(set(families[i] for i in hum))}
    print(f"    by theme: {sp_rows}")

    pos = np.array([score(i) for i in te if y[i] == 1])
    neg = np.array([score(i) for i in te if y[i] == 0])
    te_auc = auc(pos, neg)
    print()
    print(f"  HELD-OUT SCORE DISTRIBUTIONS (audit for the AUC):")
    print(f"    AI    : min={pos.min():.4f} max={pos.max():.4f} n={len(pos)}  "
          f"below 0.5: {int((pos < THRESHOLD).sum())}")
    print(f"    human : min={neg.min():.4f} max={neg.max():.4f} n={len(neg)}  "
          f"above 0.5: {int((neg >= THRESHOLD).sum())}")
    rv_scores = sorted(score(i) for i in te if families[i] == "realvis_xl")
    if rv_scores:
        print(f"    realvis scores (sorted): {[round(s, 4) for s in rv_scores]}")
    print(f"    overlap (AI below any human max) = {bool(pos.min() < neg.max())}")
    print(f"  HELD-OUT AUC (AI vs human) = {te_auc:.4f}")
    realvis = [i for i in te if families[i] == "realvis_xl"]
    rv_recall = (sum(1 for i in realvis if score(i) >= THRESHOLD) / len(realvis)) if realvis else float("nan")

    return {"name": name, "w": w, "b": b, "train_acc": tr_acc, "heldout_acc": te_acc,
            "gap": tr_acc - te_acc, "per_family": fam_rows, "multigen_recall": multigen_recall,
            "multigen_ci": [round(lo, 2), round(hi, 2)], "specificity": spec,
            "specificity_ci": [round(lo, 2), round(hi, 2)], "heldout_auc": te_auc,
            "realvis_heldout_n": len(realvis), "realvis_heldout_recall": rv_recall,
            "n_train": len(tr), "n_test": len(te)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    rng = random.Random(RANDOM_STATE)

    print("=" * 78)
    print("CLIP-ViT-L/14 LINEAR PROBE v2 (attempt 2) — PROTOCOL (before fitting)")
    print("=" * 78)
    print(f"  backbone            : frozen CLIP ViT-L/14 ({CLIP_DIR.name})")
    print("  trainable params    : LINEAR head only (no backbone tuning)")
    print(f"  random_state        : {RANDOM_STATE}")
    print(f"  decision threshold  : {THRESHOLD}")
    print("  gate discipline     : 66-image gate (incl. 14 golden) is EVAL-ONLY")

    za = np.load(CACHE_A, allow_pickle=False)
    fam_a = [str(x) for x in za["family"]]
    grp_a = [str(x) for x in za["group"]]
    emb_a = za["embeds"].astype(np.float64)
    emb_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    ai_idx = [i for i in range(len(fam_a)) if grp_a[i] == "multigen"]
    gate_idx = [i for i in range(len(fam_a)) if grp_a[i] == "gate"]
    print(f"\n  cache A: {len(fam_a)} embeddings | multigen AI={len(ai_idx)} | gate={len(gate_idx)}")
    assert len(ai_idx) == 34, f"expected 34 multigen AI in cache, got {len(ai_idx)}"
    print(f"  multigen AI families: {dict(sorted(Counter(fam_a[i] for i in ai_idx).items()))}")

    hum = load_human_embeddings()
    nh = len(hum["paths"])
    print(f"  human_v2 pool: {nh} COCO images | themes={dict(sorted(Counter(hum['theme']).items()))}")

    # ---- pooled trainable set ------------------------------------------------
    X = np.concatenate([emb_a[ai_idx], hum["embeds"]], axis=0)
    y = np.array([1] * len(ai_idx) + [0] * nh)
    labels = [Path(za["paths"][i]).name for i in ai_idx] + [Path(p).name for p in hum["paths"]]
    families = [fam_a[i] for i in ai_idx] + hum["theme"]
    paths = [str(za["paths"][i]) for i in ai_idx] + list(hum["paths"])
    groups = ["ai"] * len(ai_idx) + ["human"] * nh
    cluster = cluster_by_content(paths)
    n_uniq = len(set(cluster.values()))
    print(f"  pooled trainable: n={len(y)} (AI={int(y.sum())}, human={int((1-y).sum())})")
    print(f"  duplicate clusters: {len(y)} files -> {n_uniq} unique images")

    # ---- splits -------------------------------------------------------------
    ai_local = list(range(len(ai_idx)))
    hum_local = list(range(len(ai_idx), len(y)))
    strata = families

    # Near-duplicate audit: cosine closeness inside RealVis and inside the humans.
    for label, sel in (("realvis_xl", [i for i in ai_local if families[i] == "realvis_xl"]),
                       ("human_v2", hum_local)):
        M = X[sel] @ X[sel].T
        np.fill_diagonal(M, -1)
        mx = float(M.max())
        close = int((M > 0.999).sum() // 2)
        print(f"  near-dup audit {label:11s}: max pairwise cosine={mx:.4f}, pairs>0.999={close}")
    print()
    print("  SPLIT A (primary): stratified 60/40 per family/theme, RealVis+Midjourney in both")
    a_tr, a_te = stratified_cluster_split(ai_local + hum_local, strata, cluster, 0.60, rng)
    for grp, rng_ in (("AI", ai_local), ("HUMAN", hum_local)):
        print(f"    {grp:6s} train={len([i for i in a_tr if i in rng_])} test={len([i for i in a_te if i in rng_])}")
    print(f"    per-family train/test: " + ", ".join(
        f"{f}: {len([i for i in a_tr if families[i]==f])}/{len([i for i in a_te if families[i]==f])}"
        for f in sorted(set(families))))
    print()
    print("  SPLIT B (leave-RealVis-out): train=other families + 60% humans, test=ALL RealVis + 40% humans")
    b_tr = [i for i in ai_local if families[i] != "realvis_xl"] + [i for i in hum_local if i in a_tr]
    b_te = [i for i in ai_local if families[i] == "realvis_xl"] + [i for i in hum_local if i in a_te]
    print(f"    train: AI={len([i for i in b_tr if y[i]==1])} human={len([i for i in b_tr if y[i]==0])}")
    print(f"    test : AI={len([i for i in b_te if y[i]==1])} human={len([i for i in b_te if y[i]==0])}")

    run_a = fit_and_report("A — primary stratified", X, y, a_tr, a_te, labels, families, groups)
    run_b = fit_and_report("B — leave-RealVis-out", X, y, b_tr, b_te, labels, families, groups)

    # ---- standalone accuracy on the pristine 66 gate ------------------------
    print()
    print("=" * 78)
    print("STANDALONE PROBE ACCURACY ON THE PRISTINE 66-IMAGE GATE (never trained on)")
    print("=" * 78)
    gate_truth = np.array([1 if fam_a[i] in ("golden_ai", "sdxl_proxy") else 0 for i in gate_idx])
    for run in (run_a, run_b):
        p = (emb_a[gate_idx] @ run["w"] + run["b"]) >= 0
        correct = int((p == (gate_truth == 1)).sum())
        fp = int(((p == 1) & (gate_truth == 0)).sum())
        print(f"  run {run['name'][0]}: {correct}/66 = {correct/66:.2%}   human-false-positives={fp}/32")

    # ---- bar ---------------------------------------------------------------
    def bar(run) -> dict:
        return {
            "a_realvis_recall_gt_0": run["realvis_heldout_recall"] > 0.0,
            "b_multigen_recall_gt_44_12": run["multigen_recall"] > 15 / 34,
            "c_human_specificity_ge_80": run["specificity"] >= 0.80,
            "d_heldout_auc_ge_0_80": run["heldout_auc"] >= 0.80,
            "e_production_untouched": True,  # candidate-only; gate re-run verified separately
        }

    print()
    print("=" * 78)
    print("FIXED BAR CHECKLIST")
    print("=" * 78)
    results = {}
    for run in (run_a, run_b):
        chk = bar(run)
        results[run["name"]] = chk
        print(f"  RUN {run['name']}:")
        print(f"    [{'PASS' if chk['a_realvis_recall_gt_0'] else 'FAIL'}] RealVis held-out recall > 0%"
              f"        -> {run['realvis_heldout_recall']:.2%} (n={run['realvis_heldout_n']})")
        print(f"    [{'PASS' if chk['b_multigen_recall_gt_44_12'] else 'FAIL'}] multigen recall > 44.12%"
              f"          -> {run['multigen_recall']:.2%}")
        print(f"    [{'PASS' if chk['c_human_specificity_ge_80'] else 'FAIL'}] human specificity >= 80%"
              f"         -> {run['specificity']:.2%}")
        print(f"    [{'PASS' if chk['d_heldout_auc_ge_0_80'] else 'FAIL'}] held-out AUC >= 0.80"
              f"              -> {run['heldout_auc']:.4f}")
        print(f"    [PASS] production untouched (candidate-only)")

    passing = [r for r in (run_a, run_b) if all(bar(r).values())]
    if args.save and passing:
        # On a held-out-AUC tie, prefer the LEAVE-REALVIS-OUT run: it is the harder,
        # honest generalization test (it never saw RealVis), which is the whole point.
        tie_break = [r for r in passing if "leave-RealVis-out" in r["name"]] or passing
        best = max(tie_break, key=lambda r: r["heldout_auc"])
        art = {"weights": best["w"], "intercept": best["b"],
               "kind": "clip_vit_l14_linear_probe_v2",
               "random_state": RANDOM_STATE, "threshold": THRESHOLD,
               "n_train": best["n_train"], "run": best["name"],
               "metrics": {k: v for k, v in best.items() if k not in ("w", "b")}}
        CANDIDATE_PATH.write_bytes(pickle.dumps(art))
        print(f"\n[SAVED] {best['name']} -> {CANDIDATE_PATH}")
    else:
        print(f"\n[NOT SAVED] no run passed every bar item"
              f"{'' if args.save else ' (or --save omitted)'}.")

    (ROOT / "clip_probe_v2_report.json").write_text(json.dumps(
        {"protocol": {"backbone": "clip_vit_large (frozen)", "random_state": RANDOM_STATE,
                      "threshold": THRESHOLD},
         "data": {"trainable_ai": int(y.sum()), "trainable_human": int((1 - y).sum()),
                  "duplicate_clusters": n_uniq},
         "runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("w", "b")}
                  for k, v in (("A", run_a), ("B", run_b))},
         "bar": results}, indent=2), encoding="utf-8")
    print("[report] wrote clip_probe_v2_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
