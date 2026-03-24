"""Stable-lineage filter for reproducible lineage subset selection.

Provides functions to filter lineages by minimum lifespan (number of quarters
with activity), producing a cohort of stable lineages suitable for onset
detection and downstream analysis.

Thresholds recommended by the stability baseline report (FP-ax4.1):
- 8 quarters: onset detection experiments (matches onset detector w+c=6 minimum)
- 4 quarters: exploratory stability analysis

Usage:
    from src.stable_lineage_filter import filter_stable_lineages

    stable_ids = filter_stable_lineages(timeseries_df, min_quarters=8)
    filtered_df = timeseries_df[timeseries_df["lineage_id"].isin(stable_ids)]
"""

from __future__ import annotations

from typing import Dict, Optional, Set

import pandas as pd


def compute_lineage_lifespans(timeseries_df: pd.DataFrame) -> Dict[int, int]:
    """Compute lifespan (number of quarters) for each lineage.

    Args:
        timeseries_df: DataFrame with at least ``lineage_id`` and ``quarter``
            columns.  One row per (lineage_id, quarter).

    Returns:
        Dict mapping lineage_id to number of quarters present.
    """
    lifespans = timeseries_df.groupby("lineage_id")["quarter"].nunique()
    return dict(zip(lifespans.index.astype(int), lifespans.values.astype(int)))


def filter_stable_lineages(
    timeseries_df: pd.DataFrame,
    min_quarters: int = 8,
    max_quarters: Optional[int] = None,
) -> Set[int]:
    """Return the set of lineage IDs that meet the minimum lifespan threshold.

    Args:
        timeseries_df: DataFrame with ``lineage_id`` and ``quarter`` columns.
        min_quarters: Minimum number of quarters a lineage must be active.
            Default 8 (recommended for onset detection).
        max_quarters: Optional upper bound on lifespan.  Useful for excluding
            lineages that span the entire dataset (possible artifacts).

    Returns:
        Set of lineage_id values that pass the filter.
    """
    lifespans = compute_lineage_lifespans(timeseries_df)
    result = {
        lid for lid, span in lifespans.items()
        if span >= min_quarters and (max_quarters is None or span <= max_quarters)
    }
    return result


def summarize_filter(
    timeseries_df: pd.DataFrame,
    min_quarters: int = 8,
    max_quarters: Optional[int] = None,
) -> Dict[str, object]:
    """Summarize the effect of a stable-lineage filter.

    Returns a dict with counts and percentages for reporting.

    Args:
        timeseries_df: DataFrame with ``lineage_id`` and ``quarter`` columns.
        min_quarters: Minimum number of quarters.
        max_quarters: Optional upper bound.

    Returns:
        Dict with keys: total_lineages, stable_lineages, stable_pct,
        total_records, stable_records, stable_records_pct, min_quarters,
        max_quarters.
    """
    lifespans = compute_lineage_lifespans(timeseries_df)
    total = len(lifespans)
    stable_ids = {
        lid for lid, span in lifespans.items()
        if span >= min_quarters and (max_quarters is None or span <= max_quarters)
    }
    stable_count = len(stable_ids)

    total_records = len(timeseries_df)
    stable_records = int(
        timeseries_df["lineage_id"].isin(stable_ids).sum()
    )

    return {
        "total_lineages": total,
        "stable_lineages": stable_count,
        "stable_pct": round(100.0 * stable_count / max(total, 1), 1),
        "total_records": total_records,
        "stable_records": stable_records,
        "stable_records_pct": round(100.0 * stable_records / max(total_records, 1), 1),
        "min_quarters": min_quarters,
        "max_quarters": max_quarters,
    }
