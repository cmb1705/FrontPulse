"""
Artifact freshness and input integrity checks.

Ensures that pipeline artifacts are not built on stale or missing upstream
inputs.  Provides a manifest-save utility so that generated feature matrices
record which inputs produced them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash digest of a file for provenance tracking.

    Args:
        file_path: Path to file to hash.
        algorithm: Hash algorithm name (default sha256).

    Returns:
        Hex digest string.
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class StaleInputError(RuntimeError):
    """Raised when a required upstream artifact is missing or stale."""


def require_inputs(
    files: dict[str, Path],
    *,
    context: str = "",
) -> None:
    """Fail hard if any required input file does not exist.

    Args:
        files: Mapping of logical name to file path.
        context: Optional context string for the error message.

    Raises:
        StaleInputError: If any file is missing.
    """
    missing = {name: path for name, path in files.items() if not path.exists()}
    if missing:
        lines = [f"  {name}: {path}" for name, path in missing.items()]
        ctx = f" ({context})" if context else ""
        raise StaleInputError(
            f"Required input files missing{ctx}:\n" + "\n".join(lines)
        )


def check_freshness(
    output: Path,
    inputs: dict[str, Path],
    *,
    context: str = "",
) -> None:
    """Fail if the output file is older than any of its inputs.

    Only checks files that exist on both sides.  If the output does not
    exist, the check passes (it will be created).

    Args:
        output: Path to the derived artifact.
        inputs: Mapping of logical name to upstream file path.
        context: Optional context for the error message.

    Raises:
        StaleInputError: If the output is stale relative to any input.
    """
    if not output.exists():
        return
    output_mtime = output.stat().st_mtime
    stale: list[tuple[str, Path]] = []
    for name, path in inputs.items():
        if path.exists() and path.stat().st_mtime > output_mtime:
            stale.append((name, path))
    if stale:
        lines = [f"  {name}: {path}" for name, path in stale]
        ctx = f" ({context})" if context else ""
        raise StaleInputError(
            f"Output {output} is stale{ctx} -- these inputs are newer:\n"
            + "\n".join(lines)
        )


def build_input_manifest(
    files: dict[str, Path | None],
) -> dict[str, Any]:
    """Build a provenance manifest for a set of input files.

    Args:
        files: Mapping of logical name to file path (None = absent).

    Returns:
        Manifest dict with per-file path, sha256, size, and mtime.
    """
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    entries: dict[str, dict[str, str | None]] = {}
    for name, file_path in files.items():
        if file_path is None or not Path(file_path).exists():
            entries[name] = {"path": str(file_path) if file_path else None}
            continue
        fp = Path(file_path)
        entries[name] = {
            "path": str(fp),
            "sha256": compute_file_hash(fp),
            "size_bytes": str(fp.stat().st_size),
            "mtime": datetime.fromtimestamp(
                fp.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    manifest["inputs"] = entries
    return manifest


def save_input_manifest(
    files: dict[str, Path | None],
    output_dir: Path,
    filename: str = "input_manifest.json",
) -> Path:
    """Save an input manifest alongside generated artifacts.

    Args:
        files: Mapping of logical name to input file path.
        output_dir: Directory where the manifest will be saved.
        filename: Manifest filename (default ``input_manifest.json``).

    Returns:
        Path to the saved manifest file.
    """
    manifest = build_input_manifest(files)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    LOG.info("Saved input manifest to %s", out_path)
    return out_path


def validate_manifest_exists(
    artifact_dir: Path,
    filename: str = "input_manifest.json",
) -> bool:
    """Check whether an input manifest exists and is valid JSON.

    Args:
        artifact_dir: Directory to check.
        filename: Expected manifest filename.

    Returns:
        True if the manifest exists and contains an ``inputs`` key.
    """
    manifest_path = artifact_dir / filename
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data, dict) and "inputs" in data
    except (json.JSONDecodeError, OSError):
        return False
