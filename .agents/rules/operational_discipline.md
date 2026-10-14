# Operational Discipline & Invariants

This document codifies architectural invariants and operational rules learned from local development, evaluation integrity, and multi-origin verification.

---

## 1. Localhost Hostname Discipline & Defense-in-Depth
- **Problem**: Multi-origin local setups (e.g. Vite on `:3000`, FastAPI on `:8000`) encounter cross-origin isolation bugs, cookie mismatches, and CSRF token drops if one origin uses `localhost` and another uses `127.0.0.1`.
- **Invariants**:
  1. `vite.config.ts` must explicitly bind `server.host = '127.0.0.1'`, set `server.strictPort = true`, and configure HMR to use `127.0.0.1`.
  2. Frontend entry point (`src/main.tsx`) must enforce immediate pre-mount client redirection from `localhost` to `127.0.0.1`:
     ```ts
     if (location.hostname === 'localhost') {
       const url = new URL(location.href);
       url.hostname = '127.0.0.1';
       location.replace(url.toString());
     }
     ```

---

## 2. Pre-Commit Hook Scope Discipline
- **Problem**: Calling repository-wide status checks (`git status --porcelain`) in `.git/hooks/pre-commit` causes commits of completely valid staged code to fail when unrelated untracked scratch files or in-progress work exist in the working directory.
- **Invariants**:
  - Pre-commit integrity guards must inspect strictly staged changes:
    ```bash
    git diff --cached --name-only
    ```
  - Commits are rejected only if a file *currently staged for commit* matches sealed path patterns.
  - CI PR verification checks `git diff origin/main...HEAD`.

---

## 3. Evaluation Harness Composite Keying
- **Problem**: When computing paired per-sample metrics against baseline reports, keying sample dictionaries by bare file basename (`sample_0012.txt`) causes collisions across different generator partitions or classes (e.g. `gpt2/sample_0012.txt` vs `human/sample_0012.txt`), producing false verdict flips.
- **Invariants**:
  - Sample comparison dictionaries must always be keyed by a unique composite tuple or composite identifier:
    ```python
    key = f"{r.get('modality', '')}:{r.get('subtype', '')}:{r.get('name', '')}"
    ```
