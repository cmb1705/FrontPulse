"""Model versioning registry for MSD pipeline.

Tracks trained model versions with metadata, metrics, and lineage
information.  Each version is stored as a directory containing the
model artifact, feature list, evaluation metrics, and a version
manifest that links the training run to its inputs and configuration.

The registry itself is a flat directory of version folders plus a
``registry.json`` index that summarises every version for quick lookup
without opening individual manifests.

Typical layout::

    data/psc/out/models/msd/
        registry.json                       # version index
        v_20260323_001/
            model.pkl                       # trained pipeline
            feature_names.json              # ordered feature list
            manifest.json                   # full version metadata
        v_20260401_001/
            ...
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.trusted_io import load_trusted_pickle, save_trusted_pickle

logger = logging.getLogger(__name__)

_VERSION_DIR_RE = re.compile(r"^v_(\d{8})_(\d{3})$")


@dataclass
class ModelVersion:
    """Metadata for a single trained model version."""

    version_id: str
    created_at: str
    model_type: str
    train_quarters: str | None = None
    predict_quarters: str | None = None
    n_train_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    parent_version: str | None = None
    retrain_mode: str = "full"
    notes: str = ""

    def summary_metrics(self) -> dict[str, float]:
        """Return key metrics for registry index comparison."""
        keys = [
            "cv_pr_auc_mean", "cv_roc_auc_mean", "cv_recall_mean",
            "cv_precision_mean", "cv_f1_mean",
            "pr_auc_test", "roc_auc_test", "recall_test",
            "precision_test", "f1_test",
        ]
        return {k: self.metrics[k] for k in keys if k in self.metrics}


def _next_version_id(registry_dir: Path, date_str: str | None = None) -> str:
    """Generate the next version ID for today (or *date_str*).

    Format: ``v_YYYYMMDD_NNN`` where NNN is a zero-padded sequence
    number within the day.
    """
    if date_str is None:
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    existing_seq = 0
    if registry_dir.exists():
        for child in registry_dir.iterdir():
            match = _VERSION_DIR_RE.match(child.name)
            if match and match.group(1) == date_str:
                existing_seq = max(existing_seq, int(match.group(2)))
    return f"v_{date_str}_{existing_seq + 1:03d}"


def save_versioned_model(
    pipeline: Any,
    version: ModelVersion,
    registry_dir: Path,
    *,
    allow_external: bool = False,
) -> Path:
    """Save a trained pipeline as a versioned model artifact.

    Creates the version directory, writes the model binary, feature
    names, and manifest, then updates the registry index.

    Args:
        pipeline: Trained sklearn/imblearn pipeline object.
        version: Populated ``ModelVersion`` with metrics and config.
        registry_dir: Root of the model registry
            (e.g. ``data/psc/out/models/msd``).
        allow_external: Bypass the repo-boundary check for the model
            artifact.  Useful in tests.

    Returns:
        Path to the created version directory.
    """
    version_dir = registry_dir / version.version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    # Model artifact (binary serialization required for sklearn pipelines)
    model_path = version_dir / "model.pkl"
    save_trusted_pickle(
        pipeline, model_path,
        description="MSD versioned model",
        allow_external=allow_external,
    )

    # Feature names
    features_path = version_dir / "feature_names.json"
    features_path.write_text(json.dumps(version.feature_names, indent=2))

    # Full manifest
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(version), indent=2, default=str),
    )

    # Update registry index
    _update_registry_index(registry_dir, version)

    logger.info("Saved model version %s to %s", version.version_id, version_dir)
    return version_dir


def load_model_version(
    registry_dir: Path,
    version_id: str | None = None,
    *,
    allow_external: bool = False,
) -> tuple[Any, ModelVersion]:
    """Load a model pipeline and its version metadata.

    Args:
        registry_dir: Root of the model registry.
        version_id: Specific version to load.  If ``None``, loads the
            latest version.
        allow_external: Bypass the repo-boundary check for the model
            artifact.  Useful in tests.

    Returns:
        Tuple of (pipeline, ModelVersion).

    Raises:
        FileNotFoundError: If the version directory or model file does
            not exist.
    """
    if version_id is None:
        version_id = get_latest_version_id(registry_dir)
        if version_id is None:
            raise FileNotFoundError(
                f"No model versions found in {registry_dir}"
            )

    version_dir = registry_dir / version_id
    if not version_dir.exists():
        raise FileNotFoundError(f"Version directory not found: {version_dir}")

    model_path = version_dir / "model.pkl"
    pipeline = load_trusted_pickle(
        model_path,
        description="MSD versioned model",
        allow_external=allow_external,
    )

    manifest_path = version_dir / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text())
    version = ModelVersion(**manifest_data)

    return pipeline, version


def list_versions(registry_dir: Path) -> list[ModelVersion]:
    """List all model versions in chronological order.

    Args:
        registry_dir: Root of the model registry.

    Returns:
        List of ``ModelVersion`` objects sorted by version_id (oldest first).
    """
    index_path = registry_dir / "registry.json"
    if not index_path.exists():
        return []

    index = json.loads(index_path.read_text())
    versions = []
    for entry in index.get("versions", []):
        manifest_path = registry_dir / entry["version_id"] / "manifest.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            versions.append(ModelVersion(**data))
    return versions


def get_latest_version_id(registry_dir: Path) -> str | None:
    """Return the most recent version ID, or ``None`` if no versions exist."""
    index_path = registry_dir / "registry.json"
    if not index_path.exists():
        return None
    index = json.loads(index_path.read_text())
    entries = index.get("versions", [])
    if not entries:
        return None
    return entries[-1]["version_id"]


def compare_versions(
    current: ModelVersion,
    previous: ModelVersion,
) -> dict[str, Any]:
    """Compare metrics between two model versions.

    Args:
        current: The newer model version.
        previous: The older model version to compare against.

    Returns:
        Dictionary with metric deltas and a summary assessment.
    """
    cur_metrics = current.summary_metrics()
    prev_metrics = previous.summary_metrics()

    deltas: dict[str, float] = {}
    for key in sorted(set(cur_metrics) | set(prev_metrics)):
        cur_val = cur_metrics.get(key)
        prev_val = prev_metrics.get(key)
        if cur_val is not None and prev_val is not None:
            deltas[key] = round(cur_val - prev_val, 6)

    # Determine if the new model is an improvement
    # Primary metric: PR-AUC (best for imbalanced detection)
    primary_key = None
    for candidate in ["cv_pr_auc_mean", "pr_auc_test"]:
        if candidate in deltas:
            primary_key = candidate
            break

    improved = None
    if primary_key is not None:
        improved = deltas[primary_key] > 0

    return {
        "current_version": current.version_id,
        "previous_version": previous.version_id,
        "deltas": deltas,
        "primary_metric": primary_key,
        "improved": improved,
    }


def _update_registry_index(
    registry_dir: Path,
    version: ModelVersion,
) -> None:
    """Append or update a version entry in the registry index."""
    index_path = registry_dir / "registry.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"versions": []}

    entry = {
        "version_id": version.version_id,
        "created_at": version.created_at,
        "model_type": version.model_type,
        "retrain_mode": version.retrain_mode,
        "parent_version": version.parent_version,
        "n_train_samples": version.n_train_samples,
        "n_features": version.n_features,
        "summary_metrics": version.summary_metrics(),
    }

    # Replace if version already exists, else append
    existing_ids = [e["version_id"] for e in index["versions"]]
    if version.version_id in existing_ids:
        idx = existing_ids.index(version.version_id)
        index["versions"][idx] = entry
    else:
        index["versions"].append(entry)

    index_path.write_text(json.dumps(index, indent=2))
