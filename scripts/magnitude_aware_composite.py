#!/usr/bin/env python
"""
Compute magnitude-aware composite scores for MSD breakthrough predictions.

The composite score blends the calibrated inflection probability with
growth magnitude features to prioritize alerts that are both likely and large.
Optionally performs random-search weight tuning to maximize precision@K.

Output:
  * Updated predictions CSV with `impact_score` and `impact_tier` columns
  * JSON summary of precision@K comparisons vs. probability-only ranking

Example:
python scripts/magnitude_aware_composite.py \
  --predictions data/out/experiments/msd_inflection/leakage_free/breakthrough_predictions.csv \
  --features data/out/02_lineage_tracking/lineage_multisignal_features.csv \
  --out-csv data/out/experiments/msd_inflection/leakage_free/breakthrough_predictions.csv \
  --summary-json data/out/experiments/msd_inflection/leakage_free/magnitude_composite_summary.json \
  --optimize-weights --n-iters 400 --random-seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_KEYS = ["prob_z", "growth_z", "accel_z", "cumulative_z", "milestone_z"]
DEFAULT_WEIGHTS = {
    "prob_z": 0.50,
    "growth_z": 0.20,
    "accel_z": 0.15,
    "cumulative_z": 0.10,
    "milestone_z": 0.05,
}


def _zscore(series: pd.Series) -> pd.Series:
    """Compute z-score with safe handling of constant/empty input."""
    values = series.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


def prepare_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add z-scored feature columns required for the impact computation."""
    df = df.copy()
    df["new_works_roll_mean_4q"] = df["new_works_roll_mean_4q"].fillna(0)
    df["growth_acceleration"] = df["growth_acceleration"].fillna(0)
    df["cumulative_works"] = df["cumulative_works"].fillna(0)
    df["quarters_since_last_milestone"] = df["quarters_since_last_milestone"].fillna(24)

    df["prob_z"] = _zscore(df["inflection_probability"])
    df["growth_mag"] = np.log1p(df["new_works_roll_mean_4q"].clip(lower=0))
    df["growth_z"] = _zscore(df["growth_mag"])

    df["accel_z"] = _zscore(df["growth_acceleration"])
    df["cumulative_log"] = np.log1p(df["cumulative_works"].clip(lower=0))
    df["cumulative_z"] = _zscore(df["cumulative_log"])

    milestone_closeness = np.exp(
        -df["quarters_since_last_milestone"].clip(lower=0) / 4.0
    )
    df["milestone_z"] = _zscore(milestone_closeness)
    return df


def compute_composite(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    assign_tiers: bool = True,
) -> pd.DataFrame:
    """Compute magnitude-aware composite score and, optionally, impact tiers."""
    weights = weights.copy() if weights else DEFAULT_WEIGHTS.copy()
    df = df.copy()
    df["impact_score"] = sum(df[key] * weights[key] for key in FEATURE_KEYS)

    metadata = {"weights": weights, "high_cut": None, "medium_cut": None}
    if assign_tiers:
        alerts = df[df["is_inflection_pred"] == 1]
        tier_source = alerts if len(alerts) >= 10 else df
        metadata["high_cut"] = float(tier_source["impact_score"].quantile(0.90))
        metadata["medium_cut"] = float(tier_source["impact_score"].quantile(0.60))
        df["impact_tier"] = np.select(
            [
                df["impact_score"] >= metadata["high_cut"],
                df["impact_score"] >= metadata["medium_cut"],
            ],
            ["tier_1_high", "tier_2_medium"],
            default="tier_3_watch",
        )
    else:
        df["impact_tier"] = "tier_3_watch"
    return df, metadata


def precision_at_k(df: pd.DataFrame, k: int, score_col: str) -> float:
    """Compute precision@k for predicted alerts sorted by score_col."""
    alerts = df[df["is_inflection_pred"] == 1].sort_values(score_col, ascending=False)
    if k <= 0 or alerts.empty:
        return float("nan")
    subset = alerts.head(k)
    if subset.empty:
        return float("nan")
    return float(subset["is_inflection_true"].mean())


def summarize_metrics(df: pd.DataFrame, ks: list[int]) -> dict:
    """Build summary metrics comparing composite vs probability rankings."""
    summary = {
        "n_rows": int(len(df)),
        "n_alerts": int(df["is_inflection_pred"].sum()),
    }
    for k in ks:
        summary[f"precision_at_{k}_impact"] = precision_at_k(df, k, "impact_score")
        summary[f"precision_at_{k}_probability"] = precision_at_k(
            df, k, "inflection_probability"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute magnitude-aware composite scores for MSD alerts."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to breakthrough_predictions.csv",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Path to lineage_multisignal_features.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Destination CSV with impact_score and impact_tier columns.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Where to write JSON summary metrics.",
    )
    parser.add_argument(
        "--optimize-weights",
        action="store_true",
        help="Run random-search weight tuning before computing scores.",
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=300,
        help="Number of random-search iterations when optimizing weights.",
    )
    parser.add_argument(
        "--min-prob-weight",
        type=float,
        default=0.35,
        help="Minimum weight allocated to the probability term during optimization.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for weight optimization."
    )
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[25, 50, 100],
        help="Top-K values for precision@K evaluation.",
    )
    return parser.parse_args()


def sample_weight_dict(
    rng: np.random.Generator, min_prob_weight: float
) -> dict[str, float]:
    """Sample a valid weight dictionary satisfying the probability lower bound."""
    while True:
        vec = rng.random(len(FEATURE_KEYS))
        vec = vec / vec.sum()
        weights = dict(zip(FEATURE_KEYS, vec))
        if weights["prob_z"] >= min_prob_weight:
            return weights


def evaluate_weights(
    df: pd.DataFrame,
    weights: dict[str, float],
    ks: list[int],
) -> float:
    """Return mean precision@K for the given weight dictionary."""
    df_temp = df.copy()
    df_temp["impact_score"] = sum(df_temp[key] * weights[key] for key in FEATURE_KEYS)
    scores = [
        precision_at_k(df_temp, k, "impact_score")
        for k in ks
        if not np.isnan(k)
    ]
    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return float("-inf")
    return float(np.mean(scores))


def optimize_weights(
    df: pd.DataFrame,
    ks: list[int],
    n_iters: int,
    min_prob_weight: float,
    seed: int,
) -> dict[str, float]:
    """Random-search optimization for weight selection."""
    rng = np.random.default_rng(seed)
    best_weights = DEFAULT_WEIGHTS.copy()
    best_score = evaluate_weights(df, best_weights, ks)

    for _ in range(n_iters):
        candidate = sample_weight_dict(rng, min_prob_weight)
        score = evaluate_weights(df, candidate, ks)
        if score > best_score:
            best_score = score
            best_weights = candidate
    return best_weights, best_score


def main() -> None:
    args = parse_args()
    preds = pd.read_csv(args.predictions)
    feature_cols = [
        "new_works_roll_mean_4q",
        "growth_acceleration",
        "cumulative_works",
        "quarters_since_last_milestone",
    ]
    preds = preds.drop(columns=[c for c in feature_cols if c in preds.columns])

    feats = pd.read_csv(
        args.features,
        usecols=["lineage_id", "quarter", *feature_cols],
    )

    df = preds.merge(
        feats,
        on=["lineage_id", "quarter"],
        how="left",
        validate="one_to_one",
    )
    df = prepare_feature_columns(df)

    best_weights = DEFAULT_WEIGHTS.copy()
    best_objective = None
    if args.optimize_weights:
        best_weights, best_objective = optimize_weights(
            df,
            ks=args.topk,
            n_iters=args.n_iters,
            min_prob_weight=args.min_prob_weight,
            seed=args.random_seed,
        )

    df_with_scores, metadata = compute_composite(df, weights=best_weights)
    summary = summarize_metrics(df_with_scores, args.topk)
    summary.update(metadata)
    if best_objective is not None:
        summary["optimized_precision_mean"] = best_objective

    drop_cols = [
        "prob_z",
        "growth_mag",
        "growth_z",
        "accel_z",
        "cumulative_log",
        "cumulative_z",
        "milestone_z",
    ]
    for col in drop_cols:
        if col in df_with_scores.columns:
            df_with_scores = df_with_scores.drop(columns=col)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_with_scores.to_csv(args.out_csv, index=False)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
