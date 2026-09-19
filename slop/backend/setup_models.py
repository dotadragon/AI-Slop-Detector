"""
SLOP Model Setup — Downloads and verifies segregated models:
  - Text:  Hello-SimpleAI/chatgpt-detector-roberta (stored in Model/text & models/text)
  - Image: umm-maybe/AI-image-detector (stored in Model/image & models/image)
  - Video: dima806/deepfake_vs_real_image_detection (stored in Model/video & models/video)
"""

import os
import shutil
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "Model"
BACKEND_MODEL_DIR = Path(__file__).resolve().parent / "models"


def _download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp_path, dest_path)


def download_text_model():
    """Download Hello-SimpleAI/chatgpt-detector-roberta (~498MB)."""
    text_dir = ROOT_MODEL_DIR / "text"
    backend_text = BACKEND_MODEL_DIR / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    backend_text.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/Hello-SimpleAI/chatgpt-detector-roberta/resolve/main/"
    files = ["config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "pytorch_model.bin"]

    print("  Checking text model (Hello-SimpleAI/chatgpt-detector-roberta)...")
    for fname in files:
        target = text_dir / fname
        if not target.exists() or target.stat().st_size == 0:
            print(f"    Downloading {fname}...")
            _download_file(base + fname, target)
        b_target = backend_text / fname
        if not b_target.exists():
            shutil.copy2(target, b_target)

    # Verify load
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    AutoModelForSequenceClassification.from_pretrained(str(text_dir), local_files_only=True)
    AutoTokenizer.from_pretrained(str(text_dir), local_files_only=True)
    print("  [OK] Text model ready in Model/text")
    return True


def download_image_model():
    """Download umm-maybe/AI-image-detector (~347MB)."""
    img_dir = ROOT_MODEL_DIR / "image"
    backend_img = BACKEND_MODEL_DIR / "image"
    img_dir.mkdir(parents=True, exist_ok=True)
    backend_img.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/umm-maybe/AI-image-detector/resolve/main/"
    files = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]

    print("  Checking image model (umm-maybe/AI-image-detector)...")
    for fname in files:
        target = img_dir / fname
        if not target.exists() or target.stat().st_size == 0:
            print(f"    Downloading {fname}...")
            _download_file(base + fname, target)
        b_target = backend_img / fname
        if not b_target.exists():
            shutil.copy2(target, b_target)

    # Verify load
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    AutoModelForImageClassification.from_pretrained(str(img_dir), local_files_only=True)
    AutoImageProcessor.from_pretrained(str(img_dir), local_files_only=True)
    print("  [OK] Image model ready in Model/image")
    return True


def download_video_model():
    """Download dima806/deepfake_vs_real_image_detection (~343MB)."""
    vid_dir = ROOT_MODEL_DIR / "video"
    backend_vid = BACKEND_MODEL_DIR / "video"
    vid_dir.mkdir(parents=True, exist_ok=True)
    backend_vid.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/dima806/deepfake_vs_real_image_detection/resolve/main/"
    files = ["config.json", "preprocessor_config.json", "model.safetensors"]

    print("  Checking video deepfake model (dima806/deepfake_vs_real_image_detection)...")
    for fname in files:
        target = vid_dir / fname
        if not target.exists() or target.stat().st_size == 0:
            print(f"    Downloading {fname}...")
            _download_file(base + fname, target)
        b_target = backend_vid / fname
        if not b_target.exists():
            shutil.copy2(target, b_target)

    # Verify load
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    AutoModelForImageClassification.from_pretrained(str(vid_dir), local_files_only=True)
    AutoImageProcessor.from_pretrained(str(vid_dir), local_files_only=True)
    print("  [OK] Video model ready in Model/video")
    return True


def main():
    print("=" * 60)
    print("  SLOP Segregated Model Setup & Verification")
    print("=" * 60)
    print(f"\nRoot Model Directory:    {ROOT_MODEL_DIR.resolve()}")
    print(f"Backend Model Directory: {BACKEND_MODEL_DIR.resolve()}\n")

    results = {}
    print("[1/3] Text detection model")
    results["text"] = download_text_model()

    print("\n[2/3] Image detection model")
    results["image"] = download_image_model()

    print("\n[3/3] Video deepfake model")
    results["video"] = download_video_model()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for name, ok in results.items():
        status = "[OK] Ready in Model/" + name if ok else "[FAIL]"
        print(f"  {name:>8}: {status}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
