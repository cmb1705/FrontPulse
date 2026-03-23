"""Schema validation and type coercion for DataFrames."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


def enforce_schema(df: pd.DataFrame, schema_yaml: str | Path) -> pd.DataFrame:
    """
    Apply schema validation and type coercion to a DataFrame.

    Args:
        df: Input DataFrame to validate and transform
        schema_yaml: Path to YAML schema file defining:
            - required: dict of required column names
            - coerce_types: dict mapping column names to target dtypes
            - constraints.non_null: list of columns that must not contain nulls
            - indexes: list of columns to set as index
            - sort_by: list of columns to sort by

    Returns:
        Validated and transformed DataFrame

    Raises:
        ValueError: If non-null constraints are violated

    Example:
        >>> df = enforce_schema(df, "config/schema.yaml")
    """
    sch: Dict[str, Any] = yaml.safe_load(Path(schema_yaml).read_text())
    req: Dict[str, Any] = sch.get("required", {})

    # add missing required columns
    for col in req:
        if col not in df.columns:
            df[col] = pd.NA

    # coerce types
    coerce: Dict[str, Any] = sch.get("coerce_types", {})
    for c, t in coerce.items():
        if c in df.columns:
            if str(t).startswith("datetime"):
                df[c] = pd.to_datetime(df[c], errors="coerce")
            elif str(t).lower() in ("boolean", "bool"):
                df[c] = df[c].astype("boolean")
            else:
                try:
                    df[c] = df[c].astype(t)
                except Exception:
                    pass

    # non-null checks
    constraints: Dict[str, Any] = sch.get("constraints", {}) or {}
    nn: List[str] = constraints.get("non_null", [])
    for c in nn:
        if c in df.columns and df[c].isna().any():
            raise ValueError(f"Nulls found in non-null column: {c}")

    # index and sort
    indexes: List[str] = sch.get("indexes", [])
    for c in indexes:
        if c in df.columns:
            df = df.set_index(c, drop=False)

    sort_by: List[str] = sch.get("sort_by", [])
    if sort_by:
        df = df.sort_values(sort_by)

    return df
