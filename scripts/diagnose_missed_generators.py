"""Diagnose per-signal contributions for missed generators (llama-chat, mpt-chat)."""
import sys
from pathlib import Path

backend_path = Path(__file__).resolve().parent / "slop" / "backend"
sys.path.insert(0, str(backend_path))

from app.ml.text_detector import analyze_text, _heuristic, _model_prob, _normalize_and_sanitize_text

HELDOUT = Path(__file__).resolve().parent / "heldout_set" / "text"


def sample(gen: str, idx: int = 0) -> str:
    files = sorted((HELDOUT / "ai" / gen).glob("*.txt"))
    return files[idx].read_text(encoding="utf-8")


print(f"{'gen':12s} {'final':>6s} {'neural':>7s} {'heur':>6s} {'llm_pat':>7s} {'burst':>6s} {'contr':>6s} {'vocab':>6s} {'zipf':>6s} {'bino':>7s}")
print("-" * 85)

for gen in ["llama-chat", "mpt-chat", "mistral", "mpt", "gpt2"]:
    for idx in range(2):
        text = sample(gen, idx)
        res = analyze_text(text)
        clean = _normalize_and_sanitize_text(text)
        h = _heuristic(clean)
        mp = _model_prob(clean)
        sig = h["signals"]
        try:
            from app.ml.binoculars import compute_binoculars_and_curvature
            bino = compute_binoculars_and_curvature(clean).get("binoculars_ai_prob")
        except Exception:
            bino = None
        contrib = res.get("contributions", {})
        top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        print(f"{gen:12s} {res['overall_score']:6.3f} "
              f"{(mp if mp is not None else float('nan')):7.3f} "
              f"{h['overall_score']:6.3f} "
              f"{sig.get('llm_pattern_score', 0):7.3f} "
              f"{sig.get('burstiness', 0):6.3f} "
              f"{sig.get('contraction_avoidance', 0):6.3f} "
              f"{sig.get('vocabulary_richness', 0):6.3f} "
              f"{sig.get('zipf_deviation', 0):6.3f} "
              f"{(bino if bino is not None else float('nan')):7.3f}")
        print(f"{'':12s} top fusion contributions: {top}  mode={res.get('primary_signal','')}")

# Human control
for idx in range(2):
    text = (HELDOUT / "human" / "human" / f"sample_{idx:04d}.txt").read_text(encoding="utf-8")
    res = analyze_text(text)
    clean = _normalize_and_sanitize_text(text)
    h = _heuristic(clean)
    mp = _model_prob(clean)
    print(f"{'HUMAN':12s} {res['overall_score']:6.3f} "
          f"{(mp if mp is not None else float('nan')):7.3f} {h['overall_score']:6.3f} "
          f"{h['signals'].get('llm_pattern_score', 0):7.3f}")
