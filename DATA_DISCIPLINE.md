# Data Discipline Policy

## Sacred Splits

### heldout_set/ — NEVER touch for training
- `heldout_set/` is the held-out evaluation corpus. It must NEVER be used for training, calibration, or threshold tuning.
- Any script that reads heldout_set/ must be an evaluation script only.
- Violating this rule invalidates the benchmark.

### golden_set/ — Training data
- `golden_set/` is training / regression data.
- `eval_accuracy.py` is a REGRESSION SMOKE TEST on training data, NOT an accuracy benchmark.
- Passing eval_accuracy.py is necessary but not sufficient for a promotion.

## Canonical Benchmark
- `eval_rigorous.py` is the canonical held-out benchmark.
- Any claim about model accuracy MUST cite eval_rigorous.py results on heldout_set/.
- Results must be recorded in HELDOUT_REPORT.md.

## Promotion Requirements
- Minimum 100 samples per class per modality before a modality can be called "validated".
- Held-out performance must be documented in HELDOUT_REPORT.md before promotion.
- Threshold changes require re-running eval_rigorous.py and updating HELDOUT_REPORT.md.

## Current Honest Status (as of last eval_rigorous.py run)

| Modality | Held-out N | Accuracy / AUC | Status |
|----------|-----------|----------------|--------|
| Text     | 66        | 33.33% (AI recall 21.8% at threshold 0.9149; ~60% at 0.55) | ⚠️ Under-threshold — hotfix applied |
| Image    | 52        | 75.68% (ECE 0.39) | ⚠️ Experimental |
| Video    | 8         | 50% (0/4 human specificity) | ⚠️ Insufficient data |
| Audio    | 100       | AUC 0.493 (worse than chance) | 🚫 DISABLED |
| Document | 24 containers | AUC 0.868 | ⚠️ Small sample |

See HELDOUT_REPORT.md for raw numbers and methodology.
