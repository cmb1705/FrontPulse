"""
Run provenance tracking for reproducible experiment artifacts.

Persists the exact run configuration, input file provenance, and code identity
alongside experiment outputs so that any result can be reconstructed from
saved metadata alone.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _get_git_sha(repo_root: Path | None = None) -> str | None:
    """Return the current HEAD git SHA, or None if unavailable."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        if repo_root is not None:
            kwargs["cwd"] = str(repo_root)
        result = subprocess.run(cmd, **kwargs)  # noqa: S603
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_git_dirty(repo_root: Path | None = None) -> bool | None:
    """Return True if the working tree has uncommitted changes."""
    try:
        cmd = ["git", "status", "--porcelain"]
        kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "timeout": 5,
        }
        if repo_root is not None:
            kwargs["cwd"] = str(repo_root)
        result = subprocess.run(cmd, **kwargs)  # noqa: S603
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def collect_run_provenance(
    args: Any,
    input_files: dict[str, Path | str | None],
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect full run provenance metadata from CLI args and input files.

    Args:
        args: Parsed argparse.Namespace with all CLI flags.
        input_files: Mapping of logical name to file path for each input.
            None values are recorded as missing.
        output_dir: Where experiment artifacts are written.
        repo_root: Repository root for git SHA lookup.
        extra: Additional key-value pairs to include in provenance.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    provenance: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "python_version": sys.version,
    }

    # Code identity
    git_sha = _get_git_sha(repo_root)
    if git_sha is not None:
        provenance["git_sha"] = git_sha
        dirty = _get_git_dirty(repo_root)
        if dirty is not None:
            provenance["git_dirty"] = dirty

    # CLI arguments (full namespace for exact reproducibility)
    args_dict: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    provenance["cli_args"] = args_dict

    # Input file provenance (path + hash for each input)
    inputs: dict[str, dict[str, str | None]] = {}
    for name, file_path in input_files.items():
        if file_path is None:
            inputs[name] = {"path": None, "sha256": None}
            continue
        fp = Path(file_path)
        entry: dict[str, str | None] = {"path": str(fp)}
        if fp.exists():
            entry["sha256"] = compute_file_hash(fp)
            entry["size_bytes"] = str(fp.stat().st_size)
        else:
            entry["sha256"] = None
            entry["size_bytes"] = None
        inputs[name] = entry
    provenance["input_files"] = inputs

    # Output directory
    provenance["output_dir"] = str(output_dir)

    # Extra metadata (caller-supplied)
    if extra:
        provenance["extra"] = extra

    return provenance


def save_run_provenance(
    provenance: dict[str, Any],
    output_dir: Path,
    filename: str = "run_provenance.json",
) -> Path:
    """Save provenance metadata to a JSON file in the experiment directory.

    Args:
        provenance: Provenance dict from :func:`collect_run_provenance`.
        output_dir: Experiment output directory.
        filename: Output filename (default ``run_provenance.json``).

    Returns:
        Path to the saved provenance file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, default=str)
    return out_path


def validate_provenance_exists(
    experiment_dir: Path,
    filename: str = "run_provenance.json",
) -> bool:
    """Check whether provenance metadata exists in an experiment directory.

    Args:
        experiment_dir: Path to the experiment output directory.
        filename: Expected provenance filename.

    Returns:
        True if provenance file exists and is valid JSON.
    """
    prov_path = experiment_dir / filename
    if not prov_path.exists():
        return False
    try:
        with open(prov_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data, dict) and "cli_args" in data
    except (json.JSONDecodeError, OSError):
        return False
