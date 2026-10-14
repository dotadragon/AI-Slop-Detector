"""READ-ONLY audit: Model/ vs models/ file-level diff.

Reports, per side: file count + total bytes. Then:
  - files unique to Model/ (by relative path)
  - files unique to models/ (by relative path)
  - same relative path, different sha256 (with sizes)
  - content-identical pairs (same sha256, different paths) — semantic overlap

Nothing is modified, merged, moved, or renamed. __pycache__ dirs are excluded
from both sides (byte-code cache, not model content).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SIDES = {"Model": ROOT / "Model", "models": ROOT / "models"}
SKIP_DIRS = {"__pycache__", ".git", ".venv", "node_modules"}


def walk(base: Path) -> dict[str, tuple[str, int]]:
    out: dict[str, tuple[str, int]] = {}
    if not base.exists():
        return out
    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        rel = str(p.relative_to(base)).replace("\\", "/")
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        out[rel] = (h, p.stat().st_size)
    return out


def main() -> None:
    trees = {name: walk(base) for name, base in SIDES.items()}
    for name, files in trees.items():
        total = sum(sz for _, sz in files.values())
        print(f"[{name}] files={len(files)}  total_bytes={total}  ({total / (1024 * 1024):.1f} MB)")

    m_keys = set(trees["Model"])
    s_keys = set(trees["models"])
    only_m = sorted(m_keys - s_keys)
    only_s = sorted(s_keys - m_keys)
    both = m_keys & s_keys
    diff_hash = [k for k in both if trees["Model"][k][0] != trees["models"][k][0]]

    def fmt(name: str, files: dict, keys: list[str], limit: int = 400) -> str:
        lines = []
        for k in keys[:limit]:
            h, sz = files[k]
            lines.append(f"  {k}  [{sz} B]  sha256:{h[:12]}")
        if len(keys) > limit:
            lines.append(f"  ... (+{len(keys) - limit} more)")
        return "\n".join(lines)

    print(f"\n=== unique to Model/ ({len(only_m)}) ===")
    print(fmt("Model", trees["Model"], only_m))
    print(f"\n=== unique to models/ ({len(only_s)}) ===")
    print(fmt("models", trees["models"], only_s))
    print(f"\n=== same relpath, different content ({len(diff_hash)}) ===")
    print(fmt("both", trees["Model"], diff_hash) if diff_hash else "  (none)")

    # Content-identity intersection: same bytes anywhere on both sides.
    m_by_hash: dict[str, list[str]] = {}
    for rel, (h, _sz) in trees["Model"].items():
        m_by_hash.setdefault(h, []).append(rel)
    identical = []
    for rel, (h, sz) in trees["models"].items():
        if h in m_by_hash:
            identical.append((sz, rel, m_by_hash[h][0]))
    identical.sort(reverse=True)
    print(f"\n=== content-identical across sides ({len(identical)}) ===")
    for sz, s_rel, m_rel in identical[:80]:
        print(f"  {sz:>12,} B  models/{s_rel}  ==  Model/{m_rel}")
    if len(identical) > 80:
        print(f"  ... (+{len(identical) - 80} more)")
    dup_bytes = sum(sz for sz, _, _ in identical)
    print(f"  total duplicated bytes: {dup_sz:,} ({dup_bytes / (1024 * 1024):.1f} MB)" if (dup_sz := dup_bytes) else "")


if __name__ == "__main__":
    main()
