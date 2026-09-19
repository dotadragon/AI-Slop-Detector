"""
G3: Environment and dependency verification test.
"""

import importlib
import pytest

def test_pinned_dependencies_importable():
    modules = [
        "fastapi",
        "pydantic",
        "PIL",
        "scipy",
        "sqlalchemy",
        "aiosqlite",
    ]
    for mod in modules:
        m = importlib.import_module(mod)
        assert hasattr(m, "__version__") or hasattr(m, "__file__"), f"Failed importing {mod}"

def test_cv2_installed():
    cv2 = importlib.import_module("cv2")
    assert hasattr(cv2, "__version__")
    assert cv2.__version__.startswith("4.") or cv2.__version__.startswith("5.")
