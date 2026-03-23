"""Temporal and categorical slicing of DataFrames based on YAML configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import yaml


def apply_slices(
    df: pd.DataFrame,
    slices_yaml: str | Path,
    *,
    cutoff: Optional[pd.Timestamp] = None
) -> Dict[str, pd.DataFrame]:
    """
    Apply slicing rules from YAML configuration to create data partitions.

    Args:
        df: Input DataFrame to slice
        slices_yaml: Path to YAML file defining slice specifications.
            Each slice can have:
            - expr: pandas query expression (e.g., "pub_year >= 2020")
            - groupby: column(s) to group by (e.g., "pub_qtr")
        cutoff: Optional timestamp for query expressions (available as "cutoff" variable).
            Defaults to 2 years before today.

    Returns:
        Dictionary mapping slice names to DataFrames. For grouped slices,
        names are suffixed with group keys (e.g., "by_quarter__2020Q1").

    Example:
        >>> slices = apply_slices(df, "config/slices.yaml", cutoff=pd.Timestamp("2022-12-31"))
        >>> slices["by_quarter__2020Q1"]  # DataFrame for 2020 Q1
    """
    cfg: Dict[str, Any] = yaml.safe_load(Path(slices_yaml).read_text())
    out: Dict[str, pd.DataFrame] = {}
    env: Dict[str, Any] = {}

    if cutoff is None:
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=365 * 2)
    env["cutoff"] = cutoff

    for name, spec in (cfg.get("slices") or {}).items():
        expr: Optional[str] = spec.get("expr")
        groupby: Optional[Union[str, list]] = spec.get("groupby")

        if expr and groupby:
            sub: pd.DataFrame = df.query(expr, local_dict=env)
            for keys, grp in sub.groupby(groupby, dropna=False):
                k: str = "_".join(str(x) for x in (keys if isinstance(keys, tuple) else (keys,)))
                out[f"{name}__{k}"] = grp
        elif expr:
            out[name] = df.query(expr, local_dict=env)
        elif groupby:
            for keys, grp in df.groupby(groupby, dropna=False):
                k: str = "_".join(str(x) for x in (keys if isinstance(keys, tuple) else (keys,)))
                out[f"{name}__{k}"] = grp
        else:
            out[name] = df.copy()

    return out
