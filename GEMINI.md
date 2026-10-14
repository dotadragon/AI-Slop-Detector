# Gemini Operational Directives — SLOP Project

> [!IMPORTANT]
> The primary, authoritative behavioral protocol for all AI models working in this repository is defined in [slop/ANTIGRAVITY_RULES.md](file:///c:/Users/karth/Desktop/AISlop/slop/ANTIGRAVITY_RULES.md). Every directive in that document is mandatory with zero exceptions.

---

## Top-5 Operational Bans (Zero-Tolerance)

### 1. Zero-Mock & Anti-Shortcut
- **BANNED**: Mocking inference outputs, returning hardcoded heuristic scores, synthetic evaluation metrics, or suppressing exceptions in `app/ml/`.
- Every modification to ML pipelines MUST execute real PyTorch forward passes and report live outputs.

### 2. No Imputed Evidence Features
- **BANNED**: Defaulting or imputing missing evidence features with synthetic numbers (e.g., `.get("best_neural", 0.5)` or `.get("score", 0.5)` is STRICTLY BANNED).
- A missing evidence feature must cause the detector to degrade transparently or fail closed, with the omitted feature explicitly logged in `signals_missing`.

### 3. No Hard Score Clamps
- **BANNED**: Overriding or clamping scores outside of `app/ml/fusion.py` (e.g., `if is_diagram: overall = 0.03`).
- `fusion.py` holds exclusive single-score authority. All heuristics must feed as named feature keys into fusion, never as bypass logic before or after.

### 4. Sealed Evaluation & Integrity Protection
- **BANNED**: Modifying any sealed asset, including `.github/**`, `scripts/verify_agent_integrity.py`, `scripts/lint_rules.py`, `slop/ANTIGRAVITY_RULES.md`, `GEMINI.md`, `heldout_manifest.json`, `calibrators.json`, `Model/fusion/*.pkl`, and `scripts/eval_rigorous.py`.
- **BANNED**: Tuning hyperparameters or thresholds against the sealed test split.
- **BANNED**: Running `gh` write commands (PR merge, label manipulation), git merge, or using `git commit --no-verify`.

### 5. Evidence Mandatory & Strict Plan Scoping
- **BANNED**: Unsubstantiated claims, speculative performance assertions, or unverified changes. Every claim MUST cite exact `file:line` or paste verbatim CLI terminal output. Every change requires an automated test and a before/after eval table.
- **BANNED**: Modifying any file not explicitly listed in the approved task implementation plan.