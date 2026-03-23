"""Data ingestion orchestration supporting multiple source types."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .openalex import fetch_openalex, results_to_df


def apply_source_overrides(
    source: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a datasource config merged with runtime overrides."""
    merged = dict(source)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if key == "filters" and isinstance(value, dict):
            filters = dict(merged.get("filters") or {})
            filters.update(value)
            merged["filters"] = filters
        else:
            merged[key] = value
    return merged


def _read_one(src: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[List[Dict[str, Any]]]]:
    """
    Read data from a single source configuration.

    Args:
        src: Source configuration dictionary with:
            - kind: "openalex", "csv", or "parquet"
            - For OpenAlex: entity, mailto or api_key, filters, select, etc.
            - For CSV: path, dtypes, date_cols
            - For Parquet: path

    Returns:
        Tuple of (DataFrame, raw_records). raw_records is only populated
        for OpenAlex sources, None otherwise.

    Raises:
        ValueError: If source kind is unsupported
    """
    kind = src.get("kind", "csv").lower()
    if kind == "openalex":
        mailto = src.get("mailto")
        api_key = src.get("api_key")
        if not mailto and not api_key:
            raise ValueError(
                "OpenAlex source requires either an API key (OPENALEX_API_KEY in .env) "
                "or a contact email (--mailto / config/settings.yaml)."
            )
        results = fetch_openalex(
            entity=src.get("entity", "works"),
            mailto=mailto,
            api_key=api_key,
            filters=src.get("filters"),
            search=src.get("search"),
            select=src.get("select"),
            sort=src.get("sort"),
            per_page=int(src.get("per_page", 200)),
            max_records=src.get("max_records"),
        )
        df = results_to_df(src.get("entity", "works"), results)
        return df, results
    elif kind == "csv":
        df = pd.read_csv(src["path"], dtype=src.get("dtypes"), parse_dates=src.get("date_cols", []))
        return df, None
    elif kind == "parquet":
        df = pd.read_parquet(src["path"])
        return df, None
    else:
        raise ValueError(f"Unsupported kind: {kind}")

def ingest(
    datasources_yaml: str | Path,
    source_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Optional[List[Dict[str, Any]]]]:
    """
    Load data from datasources configuration file.

    Reads the 'primary' source from the YAML configuration and returns
    a normalized DataFrame with standardized column names.

    Args:
        datasources_yaml: Path to YAML configuration file defining sources
        source_overrides: Optional runtime overrides applied to the primary
            source without rewriting the tracked YAML file.

    Returns:
        Tuple of (DataFrame, raw_records). raw_records contains the original
        JSON response from OpenAlex if applicable, None otherwise.

    Example:
        >>> df, raw = ingest("config/datasources.yaml")
        >>> df.columns  # Normalized: lowercase, underscores
    """
    cfg = yaml.safe_load(Path(datasources_yaml).read_text())
    primary = apply_source_overrides(cfg["sources"]["primary"], source_overrides)
    df, raw = _read_one(primary)
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
                  .str.lower()
    )
    return df, raw
