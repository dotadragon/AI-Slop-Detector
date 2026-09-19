"""
G4: Static test preventing heuristic clamps outside fusion.py (ANTIGRAVITY_RULES Rule 3).
"""

import re
from pathlib import Path

def test_no_hard_clamps_outside_fusion():
    ml_dir = Path(__file__).resolve().parents[1] / "app" / "ml"
    pattern = re.compile(r"(overall|final_score)\s*=\s*0\.\d+")

    violations = []
    for py_file in ml_dir.glob("*_detector.py"):
        content = py_file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines(), start=1):
            if pattern.search(line) and not line.strip().startswith("#"):
                violations.append(f"{py_file.name}:{i} -> {line.strip()}")

    assert not violations, f"Hard clamps found outside fusion.py:\n" + "\n".join(violations)
