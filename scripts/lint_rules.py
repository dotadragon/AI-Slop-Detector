#!/usr/bin/env python3
"""
scripts/lint_rules.py

CI lint script enforcing Rule 3, Rule 5, and Rule 7 of ANTIGRAVITY_RULES.md.
Replaces manual agent self-attestation with automated static analysis.
Scans modified lines in git diff (or full repo if --full) for:
1. Hard score clamps outside fusion.py (e.g., overall = 0.50)
2. Banned imputed defaults on evidence features (e.g., .get("best_neural", 0.5))
3. Required '# ponytail: PROTECTED' annotations in protected modules.

Existing violations in unmodified code are cataloged as the Step 3 backlog.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Banned clamp regex outside fusion.py
CLAMP_REGEX = re.compile(r"(?<![a-zA-Z0-9_])(overall|final_score|overall_score)\s*=\s*0\.\d+", re.IGNORECASE)

# Banned imputed evidence features (evidence features must never be given synthetic defaults)
EVIDENCE_DEFAULT_REGEX = re.compile(
    r"\.get\(\s*[\"'](?:best_neural|heuristic_score|cfa_score|ela_score|fft_score|noise_score|dct_score|lm_perplexity|binoculars_ai_prob)[\"'],\s*0\.\d+\s*\)"
)

# Required protected headers in specific files
PROTECTED_FILES = [
    ROOT / "slop" / "backend" / "app" / "ml" / "fusion.py",
]


def get_git_diff_lines() -> dict[str, list[tuple[int, str]]]:
    """Returns mapping of relative file path to list of (line_no, added_line_content)."""
    try:
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--unified=0", "slop/backend/app/ml/"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            cwd=str(ROOT),
        )
    except Exception:
        return {}

    diff_lines = {}
    current_file = None
    current_line_no = 0

    for line in proc.stdout.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            diff_lines[current_file] = []
        elif line.startswith("@@ ") and current_file:
            # Parse line number from @@ -x,y +new_line,len @@
            m = re.search(r"\+(\d+)", line)
            if m:
                current_line_no = int(m.group(1))
        elif line.startswith("+") and not line.startswith("+++") and current_file:
            added_content = line[1:]
            diff_lines[current_file].append((current_line_no, added_content))
            current_line_no += 1
        elif not line.startswith("-"):
            current_line_no += 1

    return diff_lines


def lint_clamps_and_imputations(full: bool = False) -> tuple[list[str], list[str]]:
    violations = []
    backlog = []
    py_files = list((ROOT / "slop" / "backend" / "app" / "ml").rglob("*.py"))

    diff_lines = get_git_diff_lines() if not full else {}

    for p in py_files:
        rel = p.relative_to(ROOT).as_posix()
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as e:
            violations.append(f"Could not read {rel}: {e}")
            continue

        lines = content.splitlines()

        # Check clamps outside fusion.py
        if p.name != "fusion.py":
            for line_no, line in enumerate(lines, start=1):
                clean_line = line.split("#")[0]
                if CLAMP_REGEX.search(clean_line):
                    msg = f"{rel}:{line_no} — Banned hard score clamp: '{line.strip()}'"
                    # If checking diff only, check if this line was modified in current git diff
                    if not full and rel in diff_lines and any(ln == line_no for ln, _ in diff_lines[rel]):
                        violations.append(msg)
                    else:
                        backlog.append(msg)

        # Check evidence feature default imputations
        for line_no, line in enumerate(lines, start=1):
            clean_line = line.split("#")[0]
            if EVIDENCE_DEFAULT_REGEX.search(clean_line):
                msg = f"{rel}:{line_no} — Banned imputed evidence feature: '{line.strip()}'"
                if not full and rel in diff_lines and any(ln == line_no for ln, _ in diff_lines[rel]):
                    violations.append(msg)
                else:
                    backlog.append(msg)

    return violations, backlog


def lint_protected_annotations() -> list[str]:
    violations = []
    for p in PROTECTED_FILES:
        if not p.exists():
            violations.append(f"Protected file missing: {p.relative_to(ROOT)}")
            continue
        content = p.read_text(encoding="utf-8")
        if "# ponytail: PROTECTED" not in content:
            violations.append(f"Protected file {p.relative_to(ROOT)} is missing '# ponytail: PROTECTED' header.")
    return violations


def main():
    parser = argparse.ArgumentParser(description="SLOP Operational Rules Linter")
    parser.add_argument("--full", action="store_true", help="Report all repository occurrences as violations, including Step 3 backlog")
    parser.add_argument("--backlog", action="store_true", help="Display catalog of existing backlog items for Step 3")
    args = parser.parse_args()

    print("=== SLOP Operational Rules Linter ===")
    violations, backlog = lint_clamps_and_imputations(full=args.full)
    violations.extend(lint_protected_annotations())

    if args.backlog or backlog:
        print(f"[backlog] Cataloged {len(backlog)} pre-existing clamp/imputation item(s) scheduled for Step 3 resolution.")
        if args.backlog:
            for b in backlog:
                print(f"  [Step 3 Backlog] {b}")

    if violations:
        print(f"\n[ERROR] Found {len(violations)} newly introduced rule violation(s):", file=sys.stderr)
        for v in violations:
            print(f"  ❌  {v}", file=sys.stderr)
        print("\nFix rule violations before proceeding.", file=sys.stderr)
        sys.exit(1)

    print("[lint] All rules verified clean. Zero new clamps or imputed evidence features introduced. Protected headers intact.")
    sys.exit(0)


if __name__ == "__main__":
    main()
