import os
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _ROOT / ".env"
_loaded = False


def load_project_env():
    """Load .env from the repo root (cwd-independent). Safe to call repeatedly."""
    global _loaded
    if _loaded:
        return
    load_dotenv(_ENV_FILE, override=False)
    _loaded = True


def env(name, default=None):
    load_project_env()
    return os.getenv(name, default)
