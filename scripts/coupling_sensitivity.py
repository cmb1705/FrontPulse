#!/usr/bin/env python3
"""
Quick sensitivity probe for bibliographic coupling weights.

Uses the cached coupling edge matrix (shared reference counts, normalized scores,
and publication-year offsets) to estimate how many pairs survive under alternate
thresholds, and how the coupling weights shift when β or the temporal decay λ
are tweaked. Results are aggregated per parameter combination and written to
stdout (and optionally to CSV).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def parse_float_list(value: str, *, default: list[float]) -> list[float]:
    if value is None:
        return default
    parts = [s.strip() for s in value.split(",") if s.strip()]
    return [float(p) for p in parts] if parts else default


def load_coupling_cache(cache_dir: Path) -> pd.DataFrame:
    cache_file = cache_dir / "coupling_edges.parquet"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Coupling cache not found at {cache_file}. "
            "Run the cumulative graph build with --enable-coupling first."
        )
    df = pd.read_parquet(cache_file)
    if df.empty:
        raise ValueError(f"Coupling cache at {cache_file} is empty.")
    df = df.copy()
    df["shared_refs"] = df["shared_refs"].astype(int)
    df["ref_count_a"] = df["ref_count_a"].astype(int)
    df["ref_count_b"] = df["ref_count_b"].astype(int)
    df["coupling_score"] = df["coupling_score"].astype(float)
    df["year_a"] = df["year_a"].astype(float)
    df["year_b"] = df["year_b"].astype(float)
    df["year_diff"] = np.abs(df["year_a"] - df["year_b"])
    df["year_diff"] = df["year_diff"].fillna(0.0)
    return df


def evaluate_combination(
    df: pd.DataFrame,
    beta: float,
    decay: float,
    min_shared: int,
    min_score: float,
) -> dict:
    mask = (df["shared_refs"] >= min_shared) & (df["coupling_score"] >= min_score)
    sub = df.loc[mask]
    if sub.empty:
        return {
            "beta": beta,
            "lambda": decay,
            "min_shared": min_shared,
            "min_score": min_score,
            "pairs_retained": 0,
            "weight_sum": 0.0,
            "weight_mean": 0.0,
            "weight_std": 0.0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "weight_p95": 0.0,
            "median_year_diff": 0.0,
        }

    weights = beta * sub["coupling_score"].values * np.exp(-decay * sub["year_diff"].values)
    weight_sum = float(np.sum(weights))
    weight_mean = float(np.mean(weights))
    weight_std = float(np.std(weights))
    weight_min = float(np.min(weights))
    weight_max = float(np.max(weights))
    weight_p95 = float(np.percentile(weights, 95))
    median_year_diff = float(np.median(sub["year_diff"].values))

    return {
        "beta": beta,
        "lambda": decay,
        "min_shared": min_shared,
        "min_score": min_score,
        "pairs_retained": int(len(sub)),
        "weight_sum": weight_sum,
        "weight_mean": weight_mean,
        "weight_std": weight_std,
        "weight_min": weight_min,
        "weight_max": weight_max,
        "weight_p95": weight_p95,
        "median_year_diff": median_year_diff,
    }


def run_sweep(
    df: pd.DataFrame,
    beta_values: Iterable[float],
    decay_values: Iterable[float],
    min_shared: int,
    min_score: float,
    parallel_workers: int,
) -> pd.DataFrame:
    combos: list[tuple[float, float]] = [(b, l) for b in beta_values for l in decay_values]
    if parallel_workers and parallel_workers > 1 and len(combos) > 1:
        from concurrent.futures import ThreadPoolExecutor

        results: list[dict] = []
        try:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                for res in executor.map(
                    lambda combo: evaluate_combination(df, combo[0], combo[1], min_shared, min_score),
                    combos,
                ):
                    results.append(res)
        except Exception as exc:
            print(f"[Sensitivity] Parallel evaluation failed ({exc!r}); falling back to serial execution.")
            results = [
                evaluate_combination(df, beta, decay, min_shared, min_score) for beta, decay in combos
            ]
    else:
        results = [
            evaluate_combination(df, beta, decay, min_shared, min_score) for beta, decay in combos
        ]

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/out/cache_coupling"),
        help="Directory containing coupling_edges.parquet (default: data/out/cache_coupling)",
    )
    parser.add_argument(
        "--beta",
        type=str,
        default="0.3",
        help="Comma-separated list of β values to test (default: 0.3)",
    )
    parser.add_argument(
        "--decay",
        type=str,
        default="0.15",
        help="Comma-separated list of λ decay values to test (default: 0.15)",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=5,
        help="Shared reference threshold for inclusion (default: 5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Coupling score threshold for inclusion (default: 0.05)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=13,
        help="Number of worker threads for parameter sweep (default: 13)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional CSV output path for the sensitivity table",
    )
    args = parser.parse_args()

    beta_values = parse_float_list(args.beta, default=[0.3])
    decay_values = parse_float_list(args.decay, default=[0.15])

    df = load_coupling_cache(args.cache_dir)

    results_df = run_sweep(
        df=df,
        beta_values=beta_values,
        decay_values=decay_values,
        min_shared=args.min_shared,
        min_score=args.min_score,
        parallel_workers=args.parallel,
    ).sort_values(["beta", "lambda"]).reset_index(drop=True)

    pd.set_option("display.max_columns", None)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.out, index=False)
        print(f"[Sensitivity] Wrote results to {args.out}")


if __name__ == "__main__":
    main()
