# ANTIGRAVITY_RULES.md — Mandatory Agent Protocol for SLOP
Applies to: Gemini 3.8 High (or any model) operating inside Antigravity on this repo.

## RULE 0 — Session Bookend (MANDATORY, NO EXCEPTIONS)
1. BEFORE any edit: run `git add -A && git commit -m "checkpoint: pre-agent-session"`.
   If working tree is already clean, skip. Never start editing on a dirty, uncommitted tree.
2. BEFORE any edit to `app/ml/*` or `app/core/*`: run `python eval_accuracy.py --quick`
   and record the printed accuracy numbers in your response.
3. AFTER any edit to `app/ml/*` or `app/core/*`: run `python eval_accuracy.py --quick` again.
4. If accuracy on ANY modality drops by more than 2 percentage points compared to the
   BEFORE run, you MUST:
   a. Revert the change (`git checkout -- <file>` or `git reset --hard HEAD`)
   b. Report to the user: "Change reverted — caused accuracy regression from X% to Y%"
   c. NEVER "fix forward" blindly on a regression without the user's explicit go-ahead.
5. AFTER any edit anywhere in `slop/backend/`: run `pytest -v slop/backend/tests` and report
   the exact pass/fail count. Do not declare a task "done" if any test fails.

## RULE 1 — Zero-Mock (unchanged, reaffirmed)
- Never mock inference outputs, hardcode scores, or silently catch-and-suppress exceptions
  in `app/ml/`. Any `except Exception:` block in a detector file MUST either re-raise,
  log with `logger.exception(...)`, or explicitly justify with a `# ponytail:` comment
  why suppression is safe here. Bare `except: pass` / `except: return None` is FORBIDDEN.

## RULE 2 — Protected Code (DO NOT SIMPLIFY)
The following are marked with `# ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING
eval_accuracy.py AND HUMAN APPROVAL` comments. If you (the agent) are tempted to
"clean up," "simplify," or "remove unnecessary-looking" logic here — STOP. Ask the user first.
  - The Neural-First Consensus gate in `image_detector.py` (`best_neural >= 0.70` bypass)
  - Any fusion/consensus function shared across modalities
  - `vram_manager.py` semaphore acquire/release logic
  - Label index mapping (`id2label`) for any of the 3 models
  - The feature-vector fusion model in `app/ml/fusion.py` (see Rule 5)

## RULE 3 — No New Hard Clamps
- Never write `overall = <constant>` or `final_score = <constant>` based on a heuristic
  condition (e.g. `if is_diagram: overall = 0.03`). ALL new signals MUST be added as a
  named entry in the `features` dict passed to the fusion model (Rule 5), never as a
  direct override of the final score.
- Self-check before ending your turn: grep your own diff for the regex
  `(overall|final_score|score)\s*=\s*0\.\d+` outside of `fusion.py`. If found, undo it.

## RULE 4 — Environment Discipline
- Never run `pip install` without pinning exact versions matching `requirements.txt`.
- Never introduce a second Python environment or leave `pytest` resolving to a different
  interpreter than the server. If you detect two interpreters, STOP and report to the user
  — do not "fix" it yourself by installing packages into both silently.
- Before finishing any session touching ML code, run:
  `python verify_env.py` — must print "ENV OK". If it fails, halt and report.

## RULE 5 — Fusion Model Is the Only Score Authority
- `app/ml/fusion.py` contains the ONLY code allowed to combine features into a final
  score. It is a trained logistic/LightGBM model loaded from `Model/fusion/*.pkl`.
- New forensic signals (heuristics, new model voters) are added as new dict keys fed
  INTO this function. They must NEVER be added as bypass conditions before or after it.
- Retraining `fusion.py`'s weights requires re-running `train_fusion.py` on
  `golden_set/` and re-running `eval_accuracy.py` to confirm improvement, with results
  reported before the change is considered accepted.

## RULE 6 — API Contract Stability
- Response field names in `schemas/` are versioned. Renaming, removing, or changing the
  meaning of ANY existing response field (e.g. `ai_probability`) requires updating
  `test_api_contract.py` in the SAME commit, and confirming the frontend
  (`ResultPanel.tsx`, `SignalGrid.tsx`) is updated in the SAME commit. Never change one
  side without the other in the same turn.

## RULE 7 — Reporting Format
At the end of every session, the agent MUST output:
  - Files changed (list)
  - `eval_accuracy.py` before/after numbers per modality
  - `pytest` pass/fail count before/after
  - Explicit confirmation: "No hard clamps added. No protected code modified
    without approval. Env verified OK."
If any of these cannot be confirmed truthfully, the agent must say so explicitly
rather than omitting it.
