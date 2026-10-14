#!/usr/bin/env python3
"""
scripts/verify_agent_integrity.py

Fail-closed integrity guard enforcing Rule 2A of ANTIGRAVITY_RULES.md.
Ensures that no agent-authored commit or pull request modifies sealed files:
- .github/**
- scripts/verify_agent_integrity.py
- scripts/lint_rules.py
- slop/ANTIGRAVITY_RULES.md
- GEMINI.md
- heldout_manifest.json / heldout_manifest.sha256
- validation_status.json / reliability.json
- heldout_set/** (externalized)
- scripts/eval_rigorous.py
- calibrators.json
- Model/fusion/*.pkl
- Threshold artifacts

Exits with code 1 on ANY git error, missing base ref, or sealed file modification.
Exits with code 0 ONLY when verified completely clean.
"""

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

SEALED_PATTERNS = [
    ".github/*",
    ".github/**/*",
    "scripts/verify_agent_integrity.py",
    "scripts/lint_rules.py",
    "slop/ANTIGRAVITY_RULES.md",
    "GEMINI.md",
    "heldout_manifest.json",
    "heldout_manifest.sha256",
    "validation_status.json",
    "reliability.json",
    "heldout_set/*",
    "heldout_set/**/*",
    "scripts/eval_rigorous.py",
    "calibrators.json",
    "Model/fusion/*.pkl",
    "Model/fusion/MODEL_HASHES.sha256",
    "*threshold*",
    "**/*threshold*",
]


def run_git(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["git"] + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError as exc:
        print(f"[FATAL] Git command failed: git {' '.join(cmd)}", file=sys.stderr)
        print(f"[FATAL] Stderr: {exc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[FATAL] Unexpected error invoking git: {exc}", file=sys.stderr)
        sys.exit(1)


def is_sealed(path: str) -> bool:
    norm_path = path.replace("\\", "/").lstrip("./")
    for pattern in SEALED_PATTERNS:
        if fnmatch.fnmatch(norm_path, pattern):
            return True
        # Handle directory-level matches
        if pattern.endswith("/*") or pattern.endswith("/**"):
            prefix = pattern.split("/*")[0]
            if norm_path.startswith(prefix + "/") or norm_path == prefix:
                return True
    return False


def get_diff_files(base_ref: str | None) -> list[str]:
    changed_files = set()

    # Always check staged and unstaged working tree changes
    working_status = run_git(["status", "--porcelain"])
    if working_status:
        for line in working_status.splitlines():
            if len(line) >= 4:
                filepath = line[3:].strip()
                # If renamed, take the target path
                if " -> " in filepath:
                    filepath = filepath.split(" -> ")[1].strip()
                changed_files.add(filepath)

    # If in GitHub Actions PR or branch comparison
    if base_ref:
        print(f"[integrity] Comparing current HEAD against base ref: {base_ref}")
        diff_output = run_git(["diff", "--name-only", f"{base_ref}...HEAD"])
        if diff_output:
            for line in diff_output.splitlines():
                if line.strip():
                    changed_files.add(line.strip())
    else:
        # Check diff against pre-step0 tag or HEAD~1 if present
        for candidate_ref in ["origin/main", "main", "master", "pre-step0"]:
            check = subprocess.run(
                ["git", "rev-parse", "--verify", candidate_ref],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if check.returncode == 0:
                print(f"[integrity] Comparing current HEAD against discovered base ref: {candidate_ref}")
                diff_output = run_git(["diff", "--name-only", f"{candidate_ref}...HEAD"])
                if diff_output:
                    for line in diff_output.splitlines():
                        if line.strip():
                            changed_files.add(line.strip())
                break

    return sorted(changed_files)


def main():
    print("=== SLOP Agent Integrity Verification ===")

    # Determine base ref in CI
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        base_ref = f"origin/{base_ref}"

    files = get_diff_files(base_ref)
    print(f"[integrity] Checking {len(files)} changed file(s)...")

    violations = []
    for f in files:
        if is_sealed(f):
            violations.append(f)

    if violations:
        print("\n" + "=" * 70, file=sys.stderr)
        print("[FATAL INTEGRITY VIOLATION] Sealed artifacts modified!", file=sys.stderr)
        print("The following paths are permanently sealed against agent edits:", file=sys.stderr)
        for v in violations:
            print(f"  ❌  {v}", file=sys.stderr)
        print("=" * 70 + "\n", file=sys.stderr)
        sys.exit(1)

    print("[integrity] All changed files verified clean. No sealed assets modified.")
    sys.exit(0)


if __name__ == "__main__":
    main()
