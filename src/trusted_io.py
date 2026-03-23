"""Helpers for repo-local path resolution and trusted artifact IO.

Binary Python serialization can execute arbitrary code on load.  These helpers
do not make it safe; they narrow the default trust boundary to repo-local
artifacts and require explicit opt-in before loading from outside the
repository.

See ``docs/implementation/artifact_persistence_policy.md`` for the full
artifact classification (trusted vs. portable) and migration status.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    """Return a path rooted at the repository."""
    return REPO_ROOT.joinpath(*parts)


def require_repo_local_path(
    path: Path | str,
    *,
    description: str,
    must_exist: bool = True,
    allow_external: bool = False,
) -> Path:
    """Resolve *path* and optionally enforce the repo-local sandbox.

    Args:
        path: Filesystem path (string or ``Path``).
        description: Human-readable label used in error messages.
        must_exist: When ``True`` (default), raise ``FileNotFoundError`` if
            the resolved path does not exist on disk.
        allow_external: When ``True``, skip the repo-boundary check.

    Returns:
        The resolved, absolute ``Path``.

    Raises:
        FileNotFoundError: If *must_exist* is ``True`` and the path is missing.
        ValueError: If the path falls outside ``REPO_ROOT`` and
            *allow_external* is ``False``.
    """
    resolved = Path(path).expanduser().resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    if allow_external:
        return resolved

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"{description} must be stored under the repository root ({REPO_ROOT}) "
            "unless you explicitly allow external paths."
        ) from exc
    return resolved


def load_trusted_pickle(
    path: Path | str,
    *,
    description: str,
    allow_external: bool = False,
) -> Any:
    """Load a trusted artifact from a repo-local path (default sandbox).

    Args:
        path: Path to the ``.pkl`` file.
        description: Human-readable label for error messages.
        allow_external: Bypass the repo-boundary check.

    Returns:
        The deserialized Python object.
    """
    resolved = require_repo_local_path(
        path,
        description=description,
        allow_external=allow_external,
    )
    logger.debug("Loading trusted artifact: %s", resolved)
    with resolved.open("rb") as fh:
        return pickle.load(fh)


def save_trusted_pickle(
    obj: Any,
    path: Path | str,
    *,
    description: str,
    allow_external: bool = False,
) -> Path:
    """Serialize *obj* to a binary artifact inside the repo-local sandbox.

    Parent directories are created automatically.

    Args:
        obj: The Python object to serialize.
        path: Destination ``.pkl`` path.
        description: Human-readable label for error messages.
        allow_external: Bypass the repo-boundary check.

    Returns:
        The resolved, absolute ``Path`` where the artifact was written.
    """
    resolved = require_repo_local_path(
        path,
        description=description,
        must_exist=False,
        allow_external=allow_external,
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("wb") as fh:
        pickle.dump(obj, fh)
    logger.info("Saved trusted artifact: %s", resolved)
    return resolved
