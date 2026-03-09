"""Helpers for repo-local path resolution and trusted pickle loading.

Pickle is a trusted-only serialization format because loading a crafted pickle
can execute arbitrary code. These helpers do not make pickle safe; they narrow
the default trust boundary to repo-local artifacts and require explicit opt-in
before loading pickles from outside the repository.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    """Return a path rooted at the repository."""
    return REPO_ROOT.joinpath(*parts)


def require_repo_local_path(
    path: Path | str,
    *,
    description: str,
    allow_external: bool = False,
) -> Path:
    """Resolve a path and ensure it stays within the repository by default."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if allow_external:
        return resolved

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{description} must be stored under the repository root ({REPO_ROOT}) "
            "unless you explicitly allow external pickle inputs."
        ) from exc
    return resolved


def load_trusted_pickle(
    path: Path | str,
    *,
    description: str,
    allow_external: bool = False,
) -> Any:
    """Load a trusted pickle from a repo-local path by default."""
    resolved = require_repo_local_path(
        path,
        description=description,
        allow_external=allow_external,
    )
    with resolved.open("rb") as fh:
        return pickle.load(fh)
