"""
SLOP Adversarial Evasion Robustness Evaluation.

Measures how much the text detector's verdict degrades under realistic evasion
attacks, using REAL held-out AI samples from heldout_set/text/ai/.

Attacks evaluated
-----------------
1. homoglyph     — ASCII letters swapped for visually-identical Cyrillic/Greek
                   codepoints. NOTE: NFKC does NOT fold cross-script homoglyphs
                   (Cyrillic 'а' U+0430 has no compatibility decomposition to
                   Latin 'a'), so this genuinely bypasses the sanitizer unless
                   the pipeline has dedicated cross-script detection.
2. zero_width    — zero-width spaces/joiners inserted between every word.
                   Defense: _normalize_and_sanitize_text strips these. Expected
                   to have ~zero effect — this test VERIFIES the defense works.
3. whitespace    — real spaces/newlines doubled and injected between words
                   (not stripped by the sanitizer; stresses tokenization).
4. paraphrase    — light deterministic paraphrase (synonym swaps + contraction
                   expansion). Simulates the cheapest real-world evasion;
                   a full LLM paraphrase attack would be strictly stronger.
5. fullwidth     — fullwidth Unicode forms (NFKC-foldable). Expected to be
                   neutralized — verifies NFKC normalization works.

Metrics per attack (AI-class samples):
  - evasion_rate : fraction of samples detected as AI when clean but HUMAN when attacked
  - mean_score_delta : mean(attack_score - clean_score)  (negative = evasion worked)
And for human samples under whitespace attack: FPR change.

Usage:  python eval_evasion.py [--per-generator 2] [--with-human]
"""

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

backend_path = Path(__file__).resolve().parent.parent / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import analyze_text  # noqa: E402

import os

ROOT = Path(__file__).resolve().parent
_env_heldout = os.environ.get("SLOP_HELDOUT_DIR")
if _env_heldout and (Path(_env_heldout) / "dev" / "text").exists():
    HELDOUT = Path(_env_heldout) / "dev" / "text"
elif (ROOT / "heldout_set" / "text").exists():
    HELDOUT = ROOT / "heldout_set" / "text"
elif (ROOT / "_archive" / "raid_val" / "text").exists():
    HELDOUT = ROOT / "_archive" / "raid_val" / "text"
else:
    HELDOUT = ROOT / "raid_val" / "text"

# ---------------------------------------------------------------------------
# Attack implementations
# ---------------------------------------------------------------------------

# Visually identical cross-script pairs (Cyrillic/Greek -> Latin lookalikes).
# These have NO NFKC compatibility decomposition — they survive sanitization.
HOMOGLYPHS = {
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "y": "у",
    "x": "х", "i": "і", "s": "ѕ", "j": "ј", "A": "А", "B": "В",
    "E": "Е", "K": "К", "M": "М", "H": "Н", "O": "О", "P": "Р",
    "C": "С", "T": "Т", "X": "Х",
}

SYNONYMS = {
    "important": "significant", "however": "nonetheless", "additionally": "moreover",
    "therefore": "consequently", "utilize": "employ", "demonstrate": "show",
    "numerous": "many", "facilitate": "help", "commence": "begin",
    "terminate": "end", "subsequently": "later", "furthermore": "also",
    "comprehensive": "thorough", "sufficient": "enough", "attempt": "try",
    "assist": "aid", "purchase": "buy", "require": "need", "obtain": "get",
    "individuals": "people", "approximately": "about", "particularly": "especially",
}

ZERO_WIDTH = "\u200b"


def attack_homoglyph(text: str, rate: float = 0.25) -> str:
    """Swap ~rate of swappable ASCII letters for cross-script lookalikes."""
    rng = random.Random(42)
    out = []
    for ch in text:
        if ch in HOMOGLYPHS and rng.random() < rate:
            out.append(HOMOGLYPHS[ch])
        else:
            out.append(ch)
    return "".join(out)


def attack_zero_width(text: str) -> str:
    """Insert a zero-width space after every real space."""
    return text.replace(" ", " " + ZERO_WIDTH)


def attack_whitespace(text: str) -> str:
    """Double real whitespace between words and after sentence ends (survives sanitization)."""
    import re
    text = re.sub(r"(?<=\S) (?=\S)", "  ", text)  # double intra-sentence spaces
    text = re.sub(r"\. ", ".\n", text)            # newline after sentence periods
    return text


def attack_paraphrase(text: str, swap_rate: float = 0.5) -> str:
    """Light deterministic paraphrase: synonym swaps + expand contractions.
    Cheaper than an LLM rewrite; a real paraphrase attack is stronger than this."""
    rng = random.Random(42)
    import re
    # Expand contractions first (defeats contraction-based stylometrics)
    expansions = {
        "don't": "do not", "can't": "cannot", "won't": "will not",
        "it's": "it is", "they're": "they are", "we're": "we are",
        "isn't": "is not", "doesn't": "does not", "didn't": "did not",
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
        "I'm": "I am", "you're": "you are", "that's": "that is",
    }
    for c, e in expansions.items():
        text = text.replace(c, e).replace(c.lower(), e)

    words = text.split()
    out = []
    for w in words:
        key = w.strip(".,;:!?\"'()").lower()
        if key in SYNONYMS and rng.random() < swap_rate:
            # preserve capitalization
            rep = SYNONYMS[key]
            if w[0].isupper():
                rep = rep.capitalize()
            out.append(w.replace(w.strip(".,;:!?\"'()"), rep))
        else:
            out.append(w)
    return " ".join(out)


_FW_MAP = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ０１２３４５６７８９",
)

def attack_fullwidth(text: str) -> str:
    """Fullwidth forms — NFKC-foldable, so the sanitizer should fold these back
    to the exact original ASCII and the score must be unchanged (defense check)."""
    return text.translate(_FW_MAP)


ATTACKS = {
    "homoglyph": attack_homoglyph,
    "zero_width": attack_zero_width,
    "whitespace": attack_whitespace,
    "paraphrase": attack_paraphrase,
    "fullwidth": attack_fullwidth,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_samples(per_generator: int) -> list[tuple[str, str, str]]:
    """(label, generator, text) — balanced subset across RAID generators."""
    samples = []
    ai_dir = HELDOUT / "ai"
    if ai_dir.exists():
        for gen_dir in sorted(ai_dir.iterdir()):
            files = sorted(gen_dir.glob("*.txt"))[:per_generator]
            for f in files:
                samples.append(("ai", gen_dir.name, f.read_text(encoding="utf-8")))
    human_dir = HELDOUT / "human"
    if human_dir.exists():
        for gen_dir in sorted(human_dir.iterdir()):
            files = sorted(gen_dir.glob("*.txt"))[:per_generator]
            for f in files:
                samples.append(("human", gen_dir.name, f.read_text(encoding="utf-8")))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-generator", type=int, default=2)
    args = parser.parse_args()

    samples = load_samples(args.per_generator)
    print(f"Loaded {len(samples)} held-out samples (per-generator cap: {args.per_generator})\n")

    report = {"per_sample": [], "summary": {}}
    deltas = {name: [] for name in ATTACKS}
    evasions = {name: 0 for name in ATTACKS}
    ai_total = 0
    human_fpr_changes = []

    for label, gen, text in samples[:60]:
        try:
            clean = analyze_text(text)["overall_score"]
        except Exception as exc:
            print(f"[skip] {label}/{gen}: inference error {exc}")
            continue

        row = {"label": label, "generator": gen, "clean_score": clean}
        attack_scores = {}

        if label == "ai":
            ai_total += 1
        for name, fn in ATTACKS.items():
            try:
                s = analyze_text(fn(text))["overall_score"]
            except Exception:
                s = clean  # attack produced unparseable text; score unchanged
            attack_scores[name] = s
            if label == "ai":
                deltas[name].append(s - clean)
                if clean >= 0.50 and s < 0.50:
                    evasions[name] += 1
            elif name == "whitespace" and clean < 0.50 <= s:
                human_fpr_changes.append(gen)

        if label == "ai":
            for name, s in attack_scores.items():
                print(f"  ai/{gen:12s} clean={clean:.3f}  {name}={s:.3f}  (d={s - clean:+.3f})")
        else:
            for name, s in attack_scores.items():
                print(f"  human/{gen:9s} clean={clean:.3f}  {name}={s:.3f}  (d={s - clean:+.3f})")

        row.update(attack_scores)
        report["per_sample"].append(row)

    # Summary
    print("\n=== EVASION ROBUSTNESS SUMMARY (AI-class samples) ===")
    for name in ATTACKS:
        n = len(deltas[name])
        mean_delta = sum(deltas[name]) / n if n else 0.0
        report["summary"][name] = {
            "n_ai_samples": n,
            "mean_score_delta": round(mean_delta, 4),
            "evasion_rate": round(evasions[name] / ai_total, 3) if ai_total else None,
            "evasions": evasions[name],
            "ai_total": ai_total,
        }
        print(f"  {name:12s} mean dScore={mean_delta:+.4f}   evasion rate={evasions[name]}/{ai_total} = {evasions[name] / max(ai_total,1):.1%}")

    if human_fpr_changes:
        print(f"\n  whitespace attack flipped {len(human_fpr_changes)} human sample(s) to AI (FPR increase)")

    out = ROOT / "evasion_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written: {out}")


if __name__ == "__main__":
    main()
