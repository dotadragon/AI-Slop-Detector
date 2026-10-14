# ANTIGRAVITY_RULES.md — Mandatory Agent Protocol for SLOP
Applies to: Gemini 3.8 High (or any model) operating inside Antigravity on this repo.

## RULE 0 — Session Bookend (MANDATORY, NO EXCEPTIONS)
1. BEFORE any edit: ensure working tree is clean and branched per task. Never edit on a dirty, uncommitted tree.
2. BEFORE any edit to `app/ml/*` or `app/core/*`: run `python scripts/eval_rigorous.py --split dev` on the sealed split (via `SLOP_HELDOUT_DIR`) and record accuracy and Wilson CI in the response. State explicitly: "ECE is UNRELIABLE (pending Step 1 positive-class ECE fix)".
3. AFTER any edit to `app/ml/*` or `app/core/*`: run `python scripts/eval_rigorous.py --split dev --baseline <pre_report.json>` and compare per-sample (paired). List every flipped sample; do not rely on CI overlap.
4. If accuracy or paired sample performance on ANY modality regresses, you MUST:
   a. Revert ONLY the files modified during this session via `git restore <files>`. NEVER execute `git reset --hard`.
   b. Report to the user: "Change reverted — caused sample flip regressions: <list of flips>".
   c. NEVER "fix forward" blindly on a regression without the user's explicit go-ahead.
5. AFTER any edit anywhere in `slop/backend/`: run `pytest -v slop/backend/tests` and report the exact pass, fail, and skip counts. Do not declare a task "done" if any test fails.

## RULE 1 — Zero-Mock (unchanged, reaffirmed)
- Never mock inference outputs, hardcode scores, or silently catch-and-suppress exceptions in `app/ml/`.
- Any `except Exception:` block in a detector file MUST either re-raise, log with `logger.exception(...)`, or explicitly justify with a `# ponytail:` comment why suppression is safe here. Bare `except: pass` / `except: return None` is FORBIDDEN.

## RULE 2 — Protected Code
The following are marked with `# ponytail: PROTECTED - DO NOT MODIFY WITHOUT RE-RUNNING eval_rigorous.py ON THE SEALED SPLIT AND HUMAN APPROVAL` comments. If tempted to "clean up," "simplify," or "remove" logic here — STOP. Ask the user first:
  - The Neural-First Consensus gate in `image_detector.py`
  - Any fusion/consensus function shared across modalities
  - `vram_manager.py` semaphore acquire/release logic
  - Label index mapping (`id2label`) for any of the models
  - The feature-vector fusion model in `app/ml/fusion.py`

## RULE 2A — Test Data & Evaluation Sealing (FAIL-CLOSED)
- The evaluation dataset lives OUTSIDE the repository workspace (referenced via `SLOP_HELDOUT_DIR`) and is marked read-only at the OS level. It is partitioned into `dev/` (agent-accessible) and `final/` (human-only).
- The agent is STRICTLY FORBIDDEN from modifying:
  - `.github/**`
  - `scripts/verify_agent_integrity.py`
  - `scripts/lint_rules.py`
  - `slop/ANTIGRAVITY_RULES.md`
  - `GEMINI.md`
  - `heldout_manifest.json` / `heldout_manifest.sha256`
  - `validation_status.json` / `reliability.json`
  - External held-out dataset (`c:\Users\karth\Desktop\AISlop_heldout_set/**`)
  - `scripts/eval_rigorous.py`
  - `calibrators.json`
  - `Model/fusion/*.pkl`
  - Any threshold artifacts (`thresholds.json`, etc.)
- The agent may run `scripts/eval_rigorous.py --split dev` and report numbers verbatim. It may NEVER edit the evaluation logic, tune against the test split, or alter ground truth.
- Pre-commit hooks and CI checks fail-closed if any PR or commit touches these protected assets.
- Prohibited agent actions: Never run `gh` write commands (PR merge, label change), git merge, or `git commit --no-verify`.

## RULE 3 — No Hard Clamps & No Imputed Evidence Features
- Never write `overall = <constant>` or `final_score = <constant>` based on a heuristic condition.
- Banned: Imputing *evidence features* (`score`, `perplexity`, `ai_prob`, etc.). Never write `.get("score", 0.5)`.
- A missing evidence feature or failed extraction must result in a degraded or unavailable score, with the omitted feature explicitly recorded in `signals_missing`. Never invent evidence.
- Permitted: Documented boolean presence flags (e.g. `is_diagram`, `has_face`) may default to `False` if absent.
- Backlog note: All existing violations of imputed evidence features across the codebase are formally logged as the backlog to be resolved in Step 3.

## RULE 4 — Environment Discipline
- Never run `pip install` without pinning exact versions matching `requirements.txt`.
- Never introduce a second Python environment or leave `pytest` resolving to a different interpreter than the server.
- Before finishing any session touching ML code, run `python scripts/verify_env.py` — must print "ENV OK".

## RULE 5 — Fusion Model Is the Only Score Authority
- `app/ml/fusion.py` contains the ONLY code allowed to combine features into a final score.
- New forensic signals are added as new dict keys fed INTO this function. They must NEVER be added as bypass conditions before or after it.

## RULE 6 — API Contract Stability
- Response field names in `schemas/` are versioned. Renaming, removing, or changing the meaning of existing response fields requires updating `test_api_contract.py` and the frontend in the SAME commit.

## RULE 7 — Automated CI Rule Linting
- Manual self-attestation is replaced by automated CI linting via `python scripts/lint_rules.py`.
- Any pull request or commit that introduces hard clamps or imputed evidence features will fail CI.

## RULE 8 — Evidence Requirement & Zero-Speculation
- Every factual claim about model performance, thresholds, or code behavior MUST cite `file:line` or paste verbatim raw CLI output. Speculative assertions are prohibited.
- Every pipeline change MUST include an automated test and a before/after eval table.

## RULE 9 — Session Scoping & Plan Enforcement
- Exactly ONE task per session.
- Plan first, await user review and approval before touching code.
- The approved task plan must explicitly enumerate all target files. Modifying any file not listed in the approved plan is strictly prohibited without explicit user approval.
- Stop for user review immediately upon task completion.
