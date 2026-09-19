import sys
from pathlib import Path
import asyncio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import create_all


def pytest_sessionstart(session):
    asyncio.run(create_all())
