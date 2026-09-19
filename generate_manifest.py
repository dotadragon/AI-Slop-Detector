"""
Generate Model Manifest with SHA256 hashes and file sizes for Model/ directory.
SLOP Production Trust Infrastructure (Stage B4).
"""

import hashlib
import json
from pathlib import Path

def generate_manifest():
    model_dir = Path(r"c:\Users\karth\Desktop\AISlop\Model")
    manifest = {
        "version": "1.0",
        "description": "SLOP verified local model weights manifest",
        "models": {}
    }

    for file_path in sorted(model_dir.rglob("*")):
        if file_path.is_file() and not file_path.name.endswith(".json") and not file_path.name.startswith("."):
            rel_path = file_path.relative_to(model_dir).as_posix()
            sha256 = hashlib.sha256()
            size = file_path.stat().st_size
            with open(file_path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    sha256.update(chunk)
            manifest["models"][rel_path] = {
                "sha256": sha256.hexdigest(),
                "size_bytes": size,
                "size_mb": round(size / (1024 * 1024), 2)
            }
            print(f"Hashed {rel_path}: {manifest['models'][rel_path]['size_mb']} MB")

    out_path = model_dir / "model_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {out_path} with {len(manifest['models'])} model artifacts.")

if __name__ == "__main__":
    generate_manifest()
