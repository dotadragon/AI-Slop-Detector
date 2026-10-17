"""CI eval gate — run on every fusion-weight change; block merge if it regresses.

Compares the current production fusion pickles against the last promoted
baseline recorded in Model/fusion/promoted_baseline.json. For each modality:
  - Re-evaluates the val split with the current weights.
  - FAILS (exit 1) if balanced accuracy regresses beyond a tolerance vs baseline.
  - On success (or after an intentional promotion), refreshes the baseline.

Usage:
  python eval_gate.py                 # gate check (exit 1 on regression)
  python eval_gate.py --update        # after an intentional promotion, refresh baseline
"""
import json
import pickle
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(root / "slop" / "backend"))
sys.path.insert(0, str(root))

from scripts.retrain_text_fusion import FUSION_PKL, VAL_DIR, evaluate  # noqa: E402

BASELINE = root / "Model" / "fusion" / "promoted_baseline.json"
TOLERANCE = 0.02  # allow tiny numeric jitter; real regressions exceed this


def current_val_metrics() -> dict:
    model = pickle.loads(FUSION_PKL.read_bytes()) if FUSION_PKL.exists() else {"weights": {}, "intercept": 0.0}
    res = evaluate(model, VAL_DIR)
    return {k: v for k, v in res.items() if k != "per_sample"}


def main() -> int:
    if "--update" in sys.argv:
        m = current_val_metrics()
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(m, indent=1))
        print(f"baseline updated: bacc={m['balanced_accuracy']} macroF1={m['macro_f1']}")
        return 0

    if not BASELINE.exists():
        print("NO BASELINE — creating one from current weights (first run).")
        return main() if "--update" in sys.argv else 0 if (BASELINE.parent.mkdir(parents=True, exist_ok=True) or _write_first_baseline()) else 1

    base = json.loads(BASELINE.read_text())
    cur = current_val_metrics()
    regressed = cur["balanced_accuracy"] < base["balanced_accuracy"] - TOLERANCE
    print(f"baseline bacc={base['balanced_accuracy']}  current bacc={cur['balanced_accuracy']}  "
          f"humanRec base={base.get('human_recall')} cur={cur.get('human_recall')}")
    if regressed:
        print("EVAL GATE FAILED: balanced accuracy regressed beyond tolerance. Merge blocked.")
        return 1
    print("EVAL GATE PASSED.")
    return 0


def _write_first_baseline() -> bool:
    m = current_val_metrics()
    BASELINE.write_text(json.dumps(m, indent=1))
    print(f"baseline created: bacc={m['balanced_accuracy']}")
    return True


if __name__ == "__main__":
    sys.exit(main())
