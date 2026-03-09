from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import copy

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_QUARTER_RE = re.compile(r"(\d{4}Q[1-4])")


def extract_quarter(text: str) -> str:
    match = _QUARTER_RE.search(text)
    if not match:
        raise ValueError(f"Quarter token not found in '{text}'")
    return match.group(1)


def quarter_sort_key(quarter: str) -> int:
    return pd.Period(quarter, freq="Q").ordinal


def list_quarter_files(
    directory: Path,
    pattern: str,
) -> List[Tuple[str, Path]]:
    pairs: dict[str, Path] = {}
    for path in directory.glob(pattern):
        try:
            quarter = extract_quarter(path.stem)
        except ValueError:
            continue
        pairs.setdefault(quarter, path)
    return [(q, pairs[q]) for q in sorted(pairs.keys(), key=quarter_sort_key)]


def quarter_end(quarter: str) -> pd.Timestamp:
    period = pd.Period(quarter, freq="Q")
    return period.asfreq("Q", "end").to_timestamp()


def quarter_start(quarter: str) -> pd.Timestamp:
    period = pd.Period(quarter, freq="Q")
    return period.asfreq("Q", "start").to_timestamp()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def iter_quarter_slices(
    slice_dir: Path,
    pattern: str = "by_quarter__*.parquet",
) -> Iterator[Tuple[str, Path]]:
    for quarter, path in list_quarter_files(slice_dir, pattern):
        yield quarter, path


# =====================================================================
# Standardized Metric Output System (Task 1.1)
# =====================================================================

# Standard schema definitions for metric outputs
GLOBAL_METRIC_SCHEMA = pa.schema([
    ("quarter", pa.string()),
    ("value", pa.float64()),
])

FRONT_METRIC_SCHEMA = pa.schema([
    ("front_id", pa.string()),
    ("quarter", pa.string()),
    ("value", pa.float64()),
])

LINEAGE_METRIC_SCHEMA = pa.schema([
    ("lineage_id", pa.int64()),
    ("quarter", pa.string()),
    ("value", pa.float64()),
])


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of a file for provenance tracking.

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_metric_output_paths(
    metric_name: str,
    out_dir: Path,
    level: str,
) -> Dict[str, Path]:
    """
    Get standardized output paths for a metric at a given aggregation level.

    Args:
        metric_name: Name of metric (e.g., "author_influx")
        out_dir: Base output directory (e.g., data/out/metrics)
        level: Aggregation level ("global", "front", or "lineage")

    Returns:
        Dictionary with keys: "parquet", "metadata"
    """
    level_dir = out_dir / level
    ensure_dir(level_dir)

    return {
        "parquet": level_dir / f"{metric_name}.parquet",
        "metadata": level_dir / f"{metric_name}_metadata.json",
    }


def validate_metric_schema(
    df: pd.DataFrame,
    level: str,
    metric_name: str,
) -> None:
    """
    Validate that a dataframe matches the expected schema for a metric level.

    Args:
        df: DataFrame to validate
        level: Expected level ("global", "front", or "lineage")
        metric_name: Metric name (for error messages)

    Raises:
        ValueError: If schema doesn't match expectations
    """
    if level == "global":
        required = ["quarter", "value"]
    elif level == "front":
        required = ["front_id", "quarter", "value"]
    elif level == "lineage":
        required = ["lineage_id", "quarter", "value"]
    else:
        raise ValueError(f"Unknown level: {level}")

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Metric {metric_name} at {level} level missing required columns: {missing}"
        )

    # Check for nulls in key columns
    for col in required[:-1]:  # All except 'value' (which can be null for missing data)
        if df[col].isna().any():
            raise ValueError(
            f"Metric {metric_name} at {level} level has null values in key column: {col}"
        )


def write_metric_parquet(
    df: pd.DataFrame,
    output_path: Path,
    level: str,
    metric_name: str,
) -> None:
    """
    Write a metric dataframe to parquet with schema validation.

    Args:
        df: DataFrame to write
        output_path: Path to output parquet file
        level: Aggregation level ("global", "front", or "lineage")
        metric_name: Metric name (for validation)
    """
    validate_metric_schema(df, level, metric_name)

    # Select schema based on level
    if level == "global":
        schema = GLOBAL_METRIC_SCHEMA
    elif level == "front":
        schema = FRONT_METRIC_SCHEMA
    elif level == "lineage":
        schema = LINEAGE_METRIC_SCHEMA
    else:
        raise ValueError(f"Unknown level: {level}")

    # Convert to pyarrow table with explicit schema
    # Only use core schema columns, additional columns will be added dynamically
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Write parquet
    pq.write_table(table, output_path, compression="snappy")


def write_metric_metadata(
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Write metric metadata to JSON sidecar file.

    Args:
        metadata: Metadata dictionary
        output_path: Path to output JSON file
    """
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)


def create_metric_metadata(
    metric_name: str,
    description: str,
    formula: str,
    units: str,
    parameters: Dict[str, Any],
    input_files: List[Path],
    level: str,
    column_descriptions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Create standardized metadata dictionary for a metric.

    Args:
        metric_name: Name of the metric
        description: Human-readable description
        formula: Mathematical formula or algorithm description
        units: Units of measurement
        parameters: Computation parameters (e.g., window sizes, thresholds)
        input_files: List of input data files used
        level: Aggregation level ("global", "front", or "lineage")
        column_descriptions: Optional dict mapping column names to descriptions

    Returns:
        Metadata dictionary
    """
    return {
        "metric_name": metric_name,
        "level": level,
        "description": description,
        "formula": formula,
        "units": units,
        "parameters": parameters,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_files": [
            {
                "path": str(p),
                "hash_sha256": compute_file_hash(p) if p.exists() else None,
            }
            for p in input_files
        ],
        "column_descriptions": column_descriptions or {},
    }


def write_placeholder_metric(
    metric_name: str,
    out_dir: Path,
    level: str,
    base_metadata: Dict[str, Any],
    placeholder_reason: str,
) -> None:
    """
    Emit an empty placeholder parquet + metadata for levels lacking implementation.

    Args:
        metric_name: Name of the metric (e.g., "author_influx")
        out_dir: Base metrics output directory
        level: "front" or "lineage"
        base_metadata: Metadata dictionary from the global output
        placeholder_reason: Explanation stored in metadata parameters
    """
    if level not in {"front", "lineage"}:
        raise ValueError(f"Placeholder level must be 'front' or 'lineage', got {level}")

    if level == "front":
        placeholder_df = pd.DataFrame(columns=["front_id", "quarter", "value"])
    else:
        placeholder_df = pd.DataFrame(columns=["lineage_id", "quarter", "value"])

    paths = get_metric_output_paths(metric_name, out_dir, level)
    write_metric_parquet(placeholder_df, paths["parquet"], level, metric_name)

    metadata = copy.deepcopy(base_metadata)
    metadata["level"] = level
    metadata["generated_at"] = datetime.now(timezone.utc).isoformat()
    metadata["description"] = (
        f"{base_metadata.get('description', '')} (placeholder output)"
    )
    base_params = metadata.get("parameters", {})
    metadata["parameters"] = {
        **base_params,
        "placeholder_reason": placeholder_reason,
    }
    metadata["input_files"] = []

    write_metric_metadata(metadata, paths["metadata"])
    manifest_path = out_dir / "manifest.json"
    update_manifest(manifest_path, metric_name, level, metadata, paths)


# =====================================================================
# Manifest Management System (Task 1.2)
# =====================================================================

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """
    Load existing manifest file or create empty structure.

    Args:
        manifest_path: Path to manifest.json file

    Returns:
        Manifest dictionary
    """
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            return json.load(f)
    else:
        return {
            "manifest_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "metrics": {},
        }


def update_manifest(
    manifest_path: Path,
    metric_name: str,
    level: str,
    metadata: Dict[str, Any],
    output_files: Dict[str, Path],
) -> None:
    """
    Update central manifest file with new metric information.

    Args:
        manifest_path: Path to manifest.json file
        metric_name: Name of the metric
        level: Aggregation level ("global", "front", or "lineage")
        metadata: Metric metadata dictionary
        output_files: Dict with keys "parquet", "metadata" pointing to output files
    """
    manifest = load_manifest(manifest_path)
    manifest["last_updated"] = datetime.now(timezone.utc).isoformat()

    # Create metric key
    metric_key = f"{metric_name}_{level}"

    # Calculate output file hashes
    parquet_hash = None
    if output_files["parquet"].exists():
        parquet_hash = compute_file_hash(output_files["parquet"])

    metadata_hash = None
    if output_files["metadata"].exists():
        metadata_hash = compute_file_hash(output_files["metadata"])

    # Update manifest entry
    manifest["metrics"][metric_key] = {
        "metric_name": metric_name,
        "level": level,
        "description": metadata.get("description", ""),
        "formula": metadata.get("formula", ""),
        "units": metadata.get("units", ""),
        "parameters": metadata.get("parameters", {}),
        "last_generated": metadata.get("generated_at", ""),
        "outputs": {
            "parquet": {
                "path": str(output_files["parquet"].relative_to(manifest_path.parent)),
                "hash_sha256": parquet_hash,
            },
            "metadata": {
                "path": str(output_files["metadata"].relative_to(manifest_path.parent)),
                "hash_sha256": metadata_hash,
            },
        },
    }

    # Write manifest atomically (write to temp, then rename)
    temp_path = manifest_path.with_suffix(".json.tmp")
    with open(temp_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Atomic rename
    temp_path.replace(manifest_path)


def verify_manifest_entry(
    manifest_path: Path,
    metric_name: str,
    level: str,
) -> Dict[str, Any]:
    """
    Verify a manifest entry and check file integrity.

    Args:
        manifest_path: Path to manifest.json file
        metric_name: Name of the metric
        level: Aggregation level

    Returns:
        Dictionary with verification results

    Raises:
        KeyError: If metric not found in manifest
    """
    manifest = load_manifest(manifest_path)
    metric_key = f"{metric_name}_{level}"

    if metric_key not in manifest["metrics"]:
        raise KeyError(f"Metric {metric_key} not found in manifest")

    entry = manifest["metrics"][metric_key]
    results = {
        "metric_key": metric_key,
        "exists": True,
        "parquet_valid": False,
        "metadata_valid": False,
        "errors": [],
    }

    # Verify parquet file
    parquet_path = manifest_path.parent / entry["outputs"]["parquet"]["path"]
    expected_hash = entry["outputs"]["parquet"]["hash_sha256"]

    if not parquet_path.exists():
        results["errors"].append(f"Parquet file not found: {parquet_path}")
    elif expected_hash:
        actual_hash = compute_file_hash(parquet_path)
        if actual_hash == expected_hash:
            results["parquet_valid"] = True
        else:
            results["errors"].append(
                f"Parquet hash mismatch: expected {expected_hash[:8]}..., got {actual_hash[:8]}..."
            )

    # Verify metadata file
    metadata_path = manifest_path.parent / entry["outputs"]["metadata"]["path"]
    expected_hash = entry["outputs"]["metadata"]["hash_sha256"]

    if not metadata_path.exists():
        results["errors"].append(f"Metadata file not found: {metadata_path}")
    elif expected_hash:
        actual_hash = compute_file_hash(metadata_path)
        if actual_hash == expected_hash:
            results["metadata_valid"] = True
        else:
            results["errors"].append(
                f"Metadata hash mismatch: expected {expected_hash[:8]}..., got {actual_hash[:8]}..."
            )

    return results

