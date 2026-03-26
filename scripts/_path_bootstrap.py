"""Shared import bootstrap for direct ``python scripts/...`` execution."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_repo_imports() -> Path:
    """Prepend repository-local import roots and return the repo root."""
    for path in (PROJECT_ROOT, SCRIPTS_DIR):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return PROJECT_ROOT
