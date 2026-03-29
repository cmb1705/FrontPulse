#!/usr/bin/env python3
"""
Aggregate front-level Tripwire alerts to field-level signals.

This script reads the Tripwire output (front x quarter alerts),
produces field-level alert summaries, and writes helper files for
higher-level evaluation.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ALERT_COL_CANDIDATES = ["field_alert", "alert", "alerted"]
FRONT_COL_CANDIDATES = ["front", "front_id", "community", "community_id", "lineage_id"]
DATE_COL_CANDIDATES = ["quarter", "period", "date", "time", "year_q"]


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower:
            return lower[name.lower()]
    for col in df.columns:
        cl = col.lower()
        if any(name.lower() in cl for name in candidates):
            return col
    return None


def load_front_alerts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Front alert file not found: {path}")

    df = pd.read_csv(path)

    alert_col = _find_column(df, ALERT_COL_CANDIDATES)
    front_col = _find_column(df, FRONT_COL_CANDIDATES)
    date_col = _find_column(df, DATE_COL_CANDIDATES)

    if not alert_col or not front_col or not date_col:
        raise ValueError(
            "Could not detect required columns in alerts file. "
            f"alert_col={alert_col}, front_col={front_col}, date_col={date_col}"
        )

    # Copy relevant columns and standardise names
    df = df.rename(columns={
        alert_col: "alert",
        front_col: "front",
        date_col: "quarter",
    })

    if df["alert"].dtype != bool:
        df["alert"] = df["alert"].astype(bool)

    print(f"Loaded {len(df)} rows from {path}")

    alerts = df[df["alert"].astype(bool)].copy()
    print(f"  -> {len(alerts)} rows flagged as alerts")
    return alerts


def aggregate_to_field_simple(alerts: pd.DataFrame) -> pd.DataFrame:
    agg = (
        alerts.groupby("quarter")
        .agg(
            n_fronts_alerted=("front", "nunique"),
            fronts=("front", lambda x: ", ".join(sorted(set(x)))),
            max_rr=("rr_obs_over_mu", "max"),
            total_excess=("excess_obs_minus_mu", "sum"),
            min_q_value=("q_value", "min"),
            total_count=("count", "sum"),
        )
        .reset_index()
    )

    agg["field_alert"] = True
    return agg


def aggregate_to_field_weighted(alerts: pd.DataFrame, *, min_fronts: int = 1, min_rr: float = 1.5) -> pd.DataFrame:
    agg = aggregate_to_field_simple(alerts)
    agg["field_alert"] = (agg["n_fronts_alerted"] >= min_fronts) | (agg["max_rr"] >= min_rr)
    return agg


def create_field_timeseries(front_timeseries_path: Path) -> pd.DataFrame:
    if not front_timeseries_path.exists():
        raise FileNotFoundError(f"Front timeseries file not found: {front_timeseries_path}")

    df = pd.read_csv(front_timeseries_path)
    if "quarter" not in df.columns:
        raise ValueError(f"Expected a 'quarter' column in {front_timeseries_path}")

    field_ts = df.set_index("quarter").sum(axis=1).reset_index()
    field_ts.columns = ["period", "new_works"]
    field_ts["lineage_id"] = "field_aggregate"
    field_ts["quarter"] = field_ts["period"]
    field_ts["n_communities"] = 1
    field_ts["VI_vs_prev_quarter"] = 0.0
    field_ts["pia_eligible"] = True

    return field_ts[
        [
            "lineage_id",
            "period",
            "quarter",
            "new_works",
            "n_communities",
            "VI_vs_prev_quarter",
            "pia_eligible",
        ]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Tripwire front alerts to field level")
    parser.add_argument(
        "--alerts",
        type=Path,
        default=Path("data/out/05_tripwire_detection/alerts_tripwire.csv"),
        help="Path to Tripwire front-level alerts CSV",
    )
    parser.add_argument(
        "--front-timeseries",
        type=Path,
        default=Path("data/out/04_front_aggregation/front_timeseries_delta.csv"),
        help="Path to front-level delta timeseries (wide format)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/out/05_tripwire_detection"),
        help="Directory for aggregated outputs",
    )
    parser.add_argument("--min-fronts", type=int, default=2, help="Minimum fronts to trigger weighted field alert")
    parser.add_argument(
        "--min-rr", type=float, default=2.5, help="Minimum risk ratio to trigger weighted field alert"
    )
    return parser.parse_args()


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    args = parse_args()

    alerts = load_front_alerts(args.alerts)

    print(f"\nQuarters with alerts: {alerts['quarter'].nunique()}")
    print(f"Fronts with at least one alert: {alerts['front'].nunique()}")

    field_simple = aggregate_to_field_simple(alerts)
    print("\n=== Simple Aggregation (any front -> field alert) ===")
    print(f"Field-level alert quarters: {len(field_simple)}")

    if not field_simple.empty:
        preview_cols = ["quarter", "n_fronts_alerted", "fronts", "max_rr", "min_q_value"]
        print(field_simple[preview_cols].head(10))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    simple_out = args.out_dir / "alerts_field_level_aggregated.csv"
    field_simple.to_csv(simple_out, index=False)
    print(f"Saved field-level aggregated alerts to {simple_out}")

    field_ts = create_field_timeseries(args.front_timeseries)
    ts_out = args.out_dir / "field_timeseries_aggregated.csv"
    field_ts.to_csv(ts_out, index=False)
    print(f"Saved field-level time series to {ts_out}")

    field_weighted = aggregate_to_field_weighted(
        alerts, min_fronts=args.min_fronts, min_rr=args.min_rr
    )
    print("\n=== Weighted Aggregation (custom thresholds) ===")
    print(f"Field-level alert quarters: {field_weighted['field_alert'].sum()}")

    filtered = field_simple[~field_simple["quarter"].isin(field_weighted[field_weighted["field_alert"]]["quarter"])]
    if not filtered.empty:
        print(f"\nFiltered out {len(filtered)} quarters with weak signals:")
        print(filtered[["quarter", "n_fronts_alerted", "fronts", "max_rr"]])

    return field_simple, field_ts


if __name__ == "__main__":
    main()
