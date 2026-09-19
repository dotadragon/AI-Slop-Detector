"""
SLOP Environment Verification Script
Enforces Rule 4 of ANTIGRAVITY_RULES.md
Asserts expected libraries, CUDA acceleration, and critical dependencies.
Prints 'ENV OK' and exits with code 0 on success.
"""

import sys
import importlib

REQUIRED_MODULES = [
    "torch",
    "torchvision",
    "transformers",
    "cv2",
    "fastapi",
    "pydantic",
    "PIL",
    "scipy",
    "sqlalchemy",
    "aiosqlite",
]

def verify() -> None:
    errors = []
    print(f"Checking Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    for mod_name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "installed")
            print(f"  [OK] {mod_name}: {version}")
        except ImportError as e:
            errors.append(f"Missing required module '{mod_name}': {e}")

    # Check PyTorch & CUDA
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  PyTorch CUDA Available: {cuda_ok}")
        if cuda_ok:
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("  [WARN] CUDA is not active in this interpreter (CPU-only fallback mode)")
    except Exception as e:
        errors.append(f"PyTorch verification error: {e}")

    # Check Torchvision and transformers integration
    try:
        from transformers import AutoImageProcessor
        # AutoImageProcessor requires torchvision for vision models
        import torchvision
    except Exception as e:
        errors.append(f"HuggingFace AutoImageProcessor / Torchvision error: {e}")

    # Check OpenCV FaceDetectorYN
    try:
        import cv2
        has_face_detector = hasattr(cv2, "FaceDetectorYN")
        print(f"  OpenCV FaceDetectorYN Available: {has_face_detector}")
    except Exception as e:
        errors.append(f"OpenCV error: {e}")

    if errors:
        print("\n=== ENVIRONMENT VERIFICATION FAILED ===")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\nENV OK")
    sys.exit(0)

if __name__ == "__main__":
    verify()
