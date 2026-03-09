#!/usr/bin/env python3
"""
Shared helpers for adding seasonal context to lineage time series.

The comprehensive visualization + evaluation scripts both need:
    - Rolling 4-quarter aggregates (mean + sum)
    - Cumulative counts (linear + log)
    - Year-over-year deltas/percentages
    - Quarter ordering helpers for consistent plotting
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def quarter_to_timestamp(series: pd.Series) -> pd.Series:
    """Convert YYYYQX strings to pandas timestamps at quarter start."""
    return pd.PeriodIndex(series.astype(str), freq="Q").to_timestamp()


def quarter_to_numeric(series: pd.Series) -> pd.Series:
    """Represent YYYYQX strings as year.fraction for continuous plotting."""
    periods = pd.PeriodIndex(series.astype(str), freq="Q")
    return periods.year + (periods.quarter - 1) / 4


def add_seasonal_context(
    timeseries: pd.DataFrame,
    *,
    id_col: str = "lineage_id",
    quarter_col: str = "quarter",
    value_col: str = "new_works",
    prefix: str | None = None,
) -> pd.DataFrame:
    """
    Enrich a lineage-level time series with rolling + YoY metrics.

    Returns a copy sorted by (id_col, quarter) with the following columns:
        - <prefix>rolling_4q_mean / <prefix>rolling_4q_sum
        - <prefix>cumulative / <prefix>log_cumulative
        - <prefix>yoy_delta / <prefix>yoy_pct
        - quarter_timestamp / quarter_numeric (for plotting)
    """
    if prefix is None:
        prefix = f"{value_col}_"

    if timeseries.empty:
        result = timeseries.copy()
        result["quarter_timestamp"] = pd.NaT
        result["quarter_numeric"] = np.nan
        return result

    ts = timeseries.copy()
    ts[quarter_col] = ts[quarter_col].astype(str)
    ts["quarter_timestamp"] = quarter_to_timestamp(ts[quarter_col])
    ts["quarter_numeric"] = quarter_to_numeric(ts[quarter_col])
    ts = ts.sort_values([id_col, "quarter_timestamp"])

    group = ts.groupby(id_col, group_keys=False)
    value = group[value_col]

    ts[f"{prefix}rolling_4q_mean"] = value.transform(lambda s: s.rolling(4, min_periods=1).mean())
    ts[f"{prefix}rolling_4q_sum"] = value.transform(lambda s: s.rolling(4, min_periods=1).sum())
    ts[f"{prefix}cumulative"] = value.cumsum()
    ts[f"{prefix}log_cumulative"] = np.log1p(ts[f"{prefix}cumulative"])
    ts[f"{prefix}yoy_delta"] = value.transform(lambda s: s - s.shift(4))

    def _pct_change(arr: pd.Series) -> pd.Series:
        prev = arr.shift(4)
        pct = np.where(prev > 0, (arr / prev - 1.0) * 100.0, np.nan)
        return pct

    ts[f"{prefix}yoy_pct"] = value.transform(_pct_change)

    return ts


def attach_temporal_context(
    frame: pd.DataFrame,
    enriched_ts: pd.DataFrame,
    *,
    id_col: str = "lineage_id",
    quarter_col: str = "quarter",
    prefix: str = "new_works_",
    value_col: str = "new_works",
    suffix: str = "",
) -> pd.DataFrame:
    """
    Merge the enriched time-series fields onto an arbitrary dataframe.
    """
    metric_cols = [c for c in enriched_ts.columns if c.startswith(prefix)]
    value_cols = [value_col] if value_col in enriched_ts.columns else []
    context_cols = [id_col, quarter_col, *value_cols, "quarter_timestamp", "quarter_numeric", *metric_cols]
    context = enriched_ts[context_cols].copy()
    merged = frame.merge(context, on=[id_col, quarter_col], how="left", suffixes=("", suffix))
    return merged
