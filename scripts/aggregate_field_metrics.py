#!/usr/bin/env python3
"""
Aggregate field-level metrics for the PSC corpus.

Outputs per-quarter totals (new works, cumulative) along with rolling,
YoY, acceleration, percentile, and seasonal baseline statistics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from utils.quarter_utils import normalize_quarter, quarter_to_int  # noqa: E402

from src.domain_registry import (  # noqa: E402
    add_domain_args,
    apply_domain_path_defaults,
    resolve_script_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate field-wide metrics.")
    parser.add_argument(
        "--timeseries",
        default=None,
        help="Lineage timeseries CSV (must contain quarter + new_works).",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="CSV destination for field metrics.",
    )
    parser.add_argument(
        "--out-parquet",
        default=None,
        help="Parquet destination for field metrics.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON file recording metadata about the run.",
    )
    parser.add_argument(
        "--max-quarters",
        type=int,
        default=None,
        help="Optional limit on number of quarters (useful for smoke tests).",
    )
    add_domain_args(parser)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "quarter" not in df or "new_works" not in df:
        raise ValueError("timeseries must contain columns quarter and new_works")
    df = df.copy()
    df["quarter"] = df["quarter"].astype(str).apply(normalize_quarter)
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)
    df["quarter_of_year"] = df["quarter"].str[-1].astype(int)
    return df


def compute_field_metrics(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("quarter")
    agg = grouped["new_works"].agg(
        total_new_works="sum",
        median_new_works="median",
        mean_new_works="mean",
        lineage_count="count",
    )
    percentiles = grouped["new_works"].quantile([0.5, 0.75, 0.9]).unstack()
    percentiles.columns = ["new_works_p50", "new_works_p75", "new_works_p90"]
    agg = agg.join(percentiles)
    agg["quarter_int"] = agg.index.to_series().apply(quarter_to_int)
    agg = agg.sort_values("quarter_int").reset_index()

    totals = agg["total_new_works"]
    agg["cumulative_new_works"] = totals.cumsum()
    agg["total_new_works_roll_mean_2"] = totals.rolling(window=2, min_periods=1).mean()
    agg["total_new_works_roll_mean_4"] = totals.rolling(window=4, min_periods=1).mean()
    agg["total_new_works_roll_std_4"] = totals.rolling(window=4, min_periods=2).std()
    agg["total_new_works_diff"] = totals.diff().fillna(0.0)
    agg["total_new_works_yoy_delta"] = totals - totals.shift(4)
    agg["total_new_works_yoy_ratio"] = (totals / totals.shift(4)).replace([pd.NA, pd.NaT, float("inf")], pd.NA)

    cum = agg["cumulative_new_works"]
    agg["cumulative_growth_rate"] = cum.pct_change().replace([pd.NA, float("inf")], pd.NA)
    agg["cumulative_acceleration"] = agg["cumulative_growth_rate"].diff()

    agg["quarter_of_year"] = agg["quarter"].str[-1].astype(int)
    seasonal = (
        agg.groupby("quarter_of_year")["total_new_works"]
        .transform("median")
        .rename("seasonal_baseline_new_works")
    )
    agg["seasonal_baseline_new_works"] = seasonal
    agg["seasonal_adjusted_new_works"] = agg["total_new_works"] - agg["seasonal_baseline_new_works"]

    agg["total_new_works_yoy_ratio"] = agg["total_new_works_yoy_ratio"].fillna(0.0)
    agg["seasonal_baseline_new_works"] = agg["seasonal_baseline_new_works"].fillna(0.0)
    agg["seasonal_adjusted_new_works"] = agg["seasonal_adjusted_new_works"].fillna(0.0)
    return agg


def trim_quarters(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return df
    return df.nsmallest(limit, "quarter_int")


def save_outputs(metrics: pd.DataFrame, out_csv: Path, out_parquet: Path) -> None:
    ensure_parent(out_csv)
    ensure_parent(out_parquet)
    metrics.to_csv(out_csv, index=False)
    metrics.to_parquet(out_parquet, index=False)


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()

    paths = resolve_script_paths(args, REPO_ROOT)
    apply_domain_path_defaults(args, paths, {
        "timeseries": ("lineage_tracking", "lineage_timeseries.csv", "data/out/02_lineage_tracking/lineage_timeseries.csv"),
        "out_csv": ("front_aggregation", "field_metrics.csv", "data/out/04_front_aggregation/field_metrics.csv"),
        "out_parquet": ("front_aggregation", "field_metrics.parquet", "data/out/04_front_aggregation/field_metrics.parquet"),
        "metadata": ("front_aggregation", "field_metrics_metadata.json", "data/out/04_front_aggregation/field_metrics_metadata.json"),
    })

    ts_path = Path(args.timeseries)
    out_csv = Path(args.out_csv)
    out_parquet = Path(args.out_parquet)
    metadata_path = Path(args.metadata)

    df = load_timeseries(ts_path)
    metrics = compute_field_metrics(df)
    metrics = trim_quarters(metrics, args.max_quarters)

    save_outputs(metrics, out_csv, out_parquet)

    metadata = {
        "timeseries": str(ts_path),
        "quarters": len(metrics),
        "min_quarter": metrics["quarter"].min(),
        "max_quarter": metrics["quarter"].max(),
        "columns": list(metrics.columns),
    }
    write_metadata(metadata_path, metadata)


if __name__ == "__main__":
    main()
