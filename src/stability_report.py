"""Reusable community stability diagnostics.

Computes lineage lifespan distributions, partition instability (VI),
paper identity alignment (PIA), and activity metrics from the lineage
timeseries CSV.  Designed for iterative use during clustering experiments.

Usage:
    from src.stability_report import compute_stability_report

    report = compute_stability_report(timeseries_df)
    # report is a dict with lifespan, vi, pia, and activity sections
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.stable_lineage_filter import compute_lineage_lifespans


# -- Lifespan diagnostics -----------------------------------------------------


def compute_lifespan_stats(
    timeseries_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute lifespan distribution statistics.

    Args:
        timeseries_df: Lineage timeseries with ``lineage_id`` and ``quarter``.

    Returns:
        Dict with summary stats, percentiles, and lifespan buckets.
    """
    lifespans = compute_lineage_lifespans(timeseries_df)
    if not lifespans:
        return {"total_lineages": 0}

    values = np.array(list(lifespans.values()), dtype=float)
    total = len(values)

    # Summary stats
    stats: Dict[str, Any] = {
        "total_lineages": total,
        "mean_lifespan": round(float(np.mean(values)), 1),
        "median_lifespan": int(np.median(values)),
        "std_lifespan": round(float(np.std(values)), 1),
        "min_lifespan": int(np.min(values)),
        "max_lifespan": int(np.max(values)),
    }

    # Percentiles
    percentile_points = [5, 10, 25, 50, 75, 90, 95, 99]
    stats["percentiles"] = {
        f"p{p}": int(np.percentile(values, p))
        for p in percentile_points
    }

    # Lifespan buckets
    buckets = [
        ("1_quarter", 1, 1),
        ("2_quarters", 2, 2),
        ("3_4_quarters", 3, 4),
        ("5_8_quarters", 5, 8),
        ("9_16_quarters", 9, 16),
        ("17_32_quarters", 17, 32),
        ("33_64_quarters", 33, 64),
        ("65_plus_quarters", 65, None),
    ]
    bucket_stats: List[Dict[str, Any]] = []
    for label, lo, hi in buckets:
        if hi is None:
            count = int(np.sum(values >= lo))
        else:
            count = int(np.sum((values >= lo) & (values <= hi)))
        bucket_stats.append({
            "label": label,
            "count": count,
            "pct": round(100.0 * count / total, 1),
        })
    stats["buckets"] = bucket_stats

    return stats


# -- VI diagnostics -----------------------------------------------------------


def compute_vi_stats(
    timeseries_df: pd.DataFrame,
    vi_column: str = "VI_vs_prev_quarter",
) -> Dict[str, Any]:
    """Compute Variation of Information statistics across quarters.

    Args:
        timeseries_df: Lineage timeseries with ``quarter`` and ``vi_column``.
        vi_column: Column name for VI values.

    Returns:
        Dict with summary stats and temporal trend.
    """
    if vi_column not in timeseries_df.columns:
        return {"available": False, "reason": f"Column {vi_column} not found"}

    # VI is a partition-level metric -- take the mean per quarter
    quarter_vi = (
        timeseries_df.groupby("quarter")[vi_column]
        .mean()
        .dropna()
        .sort_index()
    )
    if quarter_vi.empty:
        return {"available": False, "reason": "No VI data"}

    values = quarter_vi.values.astype(float)
    quarters = list(quarter_vi.index)

    stats: Dict[str, Any] = {
        "available": True,
        "n_quarters": len(values),
        "mean_vi": round(float(np.mean(values)), 3),
        "median_vi": round(float(np.median(values)), 3),
        "std_vi": round(float(np.std(values)), 3),
        "min_vi": round(float(np.min(values)), 3),
        "max_vi": round(float(np.max(values)), 3),
        "min_quarter": str(quarters[int(np.argmin(values))]),
        "max_quarter": str(quarters[int(np.argmax(values))]),
    }

    # Temporal trend: split into early/middle/late thirds
    n = len(quarters)
    if n >= 6:
        third = n // 3
        stats["temporal_trend"] = {
            "early_mean": round(float(np.mean(values[:third])), 3),
            "middle_mean": round(float(np.mean(values[third:2 * third])), 3),
            "late_mean": round(float(np.mean(values[2 * third:])), 3),
        }

    return stats


# -- PIA diagnostics ----------------------------------------------------------


def compute_pia_stats(
    timeseries_df: pd.DataFrame,
    pia_column: str = "pia_rate",
) -> Dict[str, Any]:
    """Compute Paper Identity Alignment statistics.

    Args:
        timeseries_df: Lineage timeseries with ``pia_column``.
        pia_column: Column name for PIA rate.

    Returns:
        Dict with summary stats.
    """
    if pia_column not in timeseries_df.columns:
        return {"available": False, "reason": f"Column {pia_column} not found"}

    pia_values = timeseries_df[pia_column].dropna()
    if pia_values.empty:
        return {"available": False, "reason": "No PIA data"}

    values = pia_values.values.astype(float)

    return {
        "available": True,
        "n_records": len(values),
        "mean_pia": round(float(np.mean(values)), 4),
        "median_pia": round(float(np.median(values)), 4),
        "std_pia": round(float(np.std(values)), 4),
        "mean_per_quarter": round(
            float(timeseries_df.groupby("quarter")[pia_column].mean().mean()),
            4,
        ),
    }


# -- Activity diagnostics -----------------------------------------------------


def compute_activity_stats(
    timeseries_df: pd.DataFrame,
    activity_column: str = "new_works",
) -> Dict[str, Any]:
    """Compute activity (new works) statistics per lineage-quarter.

    Args:
        timeseries_df: Lineage timeseries with ``activity_column``.
        activity_column: Column name for new works count.

    Returns:
        Dict with summary stats.
    """
    if activity_column not in timeseries_df.columns:
        return {"available": False, "reason": f"Column {activity_column} not found"}

    values = timeseries_df[activity_column].dropna().values.astype(float)
    if len(values) == 0:
        return {"available": False, "reason": "No activity data"}

    return {
        "available": True,
        "n_records": len(values),
        "mean_new_works": round(float(np.mean(values)), 1),
        "median_new_works": round(float(np.median(values)), 1),
        "zero_count": int(np.sum(values == 0)),
        "zero_pct": round(100.0 * float(np.sum(values == 0)) / len(values), 1),
    }


# -- Combined report ----------------------------------------------------------


def compute_stability_report(
    timeseries_df: pd.DataFrame,
    vi_column: str = "VI_vs_prev_quarter",
    pia_column: str = "pia_rate",
    activity_column: str = "new_works",
) -> Dict[str, Any]:
    """Compute a full stability report from a lineage timeseries.

    Args:
        timeseries_df: Lineage timeseries DataFrame.
        vi_column: Column name for VI values.
        pia_column: Column name for PIA rate.
        activity_column: Column name for new works count.

    Returns:
        Dict with sections: lifespan, vi, pia, activity.
    """
    return {
        "lifespan": compute_lifespan_stats(timeseries_df),
        "vi": compute_vi_stats(timeseries_df, vi_column),
        "pia": compute_pia_stats(timeseries_df, pia_column),
        "activity": compute_activity_stats(timeseries_df, activity_column),
    }


def format_report_text(report: Dict[str, Any]) -> str:
    """Format a stability report as human-readable text.

    Args:
        report: Output from compute_stability_report.

    Returns:
        Multi-line text string.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("COMMUNITY STABILITY REPORT")
    lines.append("=" * 60)

    # Lifespan section
    ls = report.get("lifespan", {})
    lines.append("")
    lines.append("LINEAGE LIFESPAN")
    lines.append("-" * 40)
    if ls.get("total_lineages", 0) == 0:
        lines.append("  No lineage data available.")
    else:
        lines.append(f"  Total lineages:  {ls['total_lineages']:,}")
        lines.append(f"  Mean lifespan:   {ls['mean_lifespan']} quarters")
        lines.append(f"  Median lifespan: {ls['median_lifespan']} quarter(s)")
        lines.append(f"  Std deviation:   {ls['std_lifespan']}")
        lines.append(f"  Range:           {ls['min_lifespan']} -- {ls['max_lifespan']}")
        if "percentiles" in ls:
            lines.append("  Percentiles:")
            for k, v in ls["percentiles"].items():
                lines.append(f"    {k.upper()}: {v}")
        if "buckets" in ls:
            lines.append("  Lifespan buckets:")
            for b in ls["buckets"]:
                lines.append(f"    {b['label']:<25} {b['count']:>5}  ({b['pct']}%%)")

    # VI section
    vi = report.get("vi", {})
    lines.append("")
    lines.append("PARTITION INSTABILITY (VI)")
    lines.append("-" * 40)
    if not vi.get("available"):
        lines.append(f"  Not available: {vi.get('reason', 'unknown')}")
    else:
        lines.append(f"  Quarters with VI: {vi['n_quarters']}")
        lines.append(f"  Mean VI:          {vi['mean_vi']} bits")
        lines.append(f"  Median VI:        {vi['median_vi']} bits")
        lines.append(f"  Range:            {vi['min_vi']} -- {vi['max_vi']} bits")
        lines.append(f"  Min quarter:      {vi['min_quarter']}")
        lines.append(f"  Max quarter:      {vi['max_quarter']}")
        if "temporal_trend" in vi:
            t = vi["temporal_trend"]
            lines.append(f"  Temporal trend:   early={t['early_mean']} mid={t['middle_mean']} late={t['late_mean']}")

    # PIA section
    pia = report.get("pia", {})
    lines.append("")
    lines.append("PAPER IDENTITY ALIGNMENT (PIA)")
    lines.append("-" * 40)
    if not pia.get("available"):
        lines.append(f"  Not available: {pia.get('reason', 'unknown')}")
    else:
        lines.append(f"  Records with PIA: {pia['n_records']:,}")
        lines.append(f"  Mean PIA rate:    {pia['mean_pia']}")
        lines.append(f"  Median PIA rate:  {pia['median_pia']}")
        lines.append(f"  Per-quarter mean: {pia['mean_per_quarter']}")

    # Activity section
    act = report.get("activity", {})
    lines.append("")
    lines.append("ACTIVITY")
    lines.append("-" * 40)
    if not act.get("available"):
        lines.append(f"  Not available: {act.get('reason', 'unknown')}")
    else:
        lines.append(f"  Records:          {act['n_records']:,}")
        lines.append(f"  Mean new works:   {act['mean_new_works']}")
        lines.append(f"  Median new works: {act['median_new_works']}")
        lines.append(f"  Zero-activity:    {act['zero_count']:,} ({act['zero_pct']}%%)")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
