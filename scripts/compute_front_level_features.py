#!/usr/bin/env python3
"""
Compute front-level features by aggregating lineage-level features.

Consumes the lineage multisignal features CSV and a lineage-to-front mapping,
then produces the canonical front-level onset series CSV defined in the
front-level series contract (docs/implementation/frontpulse_program/
front_level_series_contract.md).

Features are aggregated per (front_id, quarter) using appropriate reduction
operations: sums for counts, means for rates/scores, and counts for lineage
tallies.  Growth and acceleration columns are recomputed on the front-level
aggregates to avoid averaging growth rates across lineages with different
baselines.

Usage:
    python scripts/compute_front_level_features.py \
        --lineage-features data/out/02_lineage_tracking/lineage_multisignal_features.csv \
        --mapping data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv \
        --onset-labels data/out/02_lineage_tracking/onset_labels.csv \
        --out data/out/04_front_aggregation/front_onset_series.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _path_bootstrap import ensure_repo_imports

REPO_ROOT = ensure_repo_imports()

from utils.quarter_utils import quarter_key  # type: ignore  # noqa: E402

from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

LOG = logging.getLogger("front_level_features")


# ---------------------------------------------------------------------------
# Aggregation semantics: which lineage features survive front-level grouping
# ---------------------------------------------------------------------------

# Features aggregated by sum (additive quantities)
SUM_FEATURES = [
    "new_works",
    "novel_terms",
    "cross_domain_refs",
    "within_lineage_refs",
    "n_new_papers",
]

# Features aggregated by mean (rates, scores, intensities)
MEAN_FEATURES = [
    "novelty_rate",
    "novelty_momentum",
    "cross_domain_share",
    "citation_balance",
    "semantic_velocity",
    "velocity_acceleration",
    "dormancy_length",
    "awakening_intensity",
]

# Context features aggregated by mean (trailing rolling z-scores and deltas)
CONTEXT_MEAN_FEATURES = [
    "author_influx_z",
    "author_influx_roll_2q",
    "author_influx_roll_4q",
    "citation_velocity_z",
    "citation_velocity_roll_2q",
    "citation_velocity_roll_4q",
    "reference_vitality_z",
    "reference_vitality_roll_2q",
    "reference_vitality_roll_4q",
    "topic_diversity_z",
    "topic_diversity_roll_2q",
    "topic_diversity_roll_4q",
    "cross_cluster_bridging_z",
    "cross_cluster_bridging_roll_2q",
    "cross_cluster_bridging_roll_4q",
]

# Convergence features aggregated by mean
CONVERGENCE_MEAN_FEATURES = [
    "conv_max_semantic_sim",
    "conv_mean_top5_sim",
    "conv_semantic_velocity",
    "conv_author_migration_rate",
    "conv_citation_bridge_rate",
    "conv_terminology_overlap",
    "conv_composite_score",
    "conv_composite_score_roll_2q",
    "conv_composite_score_roll_4q",
    "conv_max_semantic_sim_roll_2q",
    "conv_max_semantic_sim_roll_4q",
]


def configure_logging(verbose: bool) -> None:
    """Set up logging with appropriate level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_mapping(
    mapping_path: Path,
    min_similarity: float = 0.65,
) -> dict[int, str]:
    """Load lineage-to-front mapping and return lineage_id -> front_id dict.

    Supports two mapping formats:
    - Tight milestone mapping (has 'similarity' and 'mapped_fronts' columns):
      filter by similarity >= min_similarity, use 'mapped_fronts' as front_id.
    - Standard mapping (has 'primary_front' and 'confidence' columns):
      filter by confidence in {high, medium}, use 'primary_front' as front_id.
    """
    df = pd.read_csv(mapping_path)

    if "mapped_fronts" in df.columns and "similarity" in df.columns:
        # Tight milestone mapping format
        df = df[df["similarity"] >= min_similarity].copy()
        # Each lineage may appear multiple times (multiple events); take
        # the highest-similarity mapping per lineage.
        df = df.sort_values("similarity", ascending=False).drop_duplicates(
            subset=["lineage_id"], keep="first",
        )
        return dict(zip(df["lineage_id"].astype(int), df["mapped_fronts"].astype(str)))

    if "primary_front" in df.columns:
        # Standard mapping format
        if "confidence" in df.columns:
            df = df[df["confidence"].isin(["high", "medium"])].copy()
        return dict(zip(df["lineage_id"].astype(int), df["primary_front"].astype(str)))

    msg = (
        f"Unrecognized mapping format in {mapping_path}. "
        "Expected 'mapped_fronts'+'similarity' or 'primary_front'+'confidence'."
    )
    raise ValueError(msg)


def load_onset_labels(onset_path: Path) -> dict[int, str]:
    """Load onset labels and return lineage_id -> onset_quarter dict.

    Only includes lineages where onset_detected == 1.
    """
    df = pd.read_csv(onset_path)
    detected = df[df["onset_detected"] == 1]
    return dict(zip(
        detected["lineage_id"].astype(int),
        detected["onset_quarter"].astype(str),
    ))


def derive_front_onset(
    _front_id: str,
    mapped_lineages: set[int],
    lineage_onsets: dict[int, str],
) -> str | None:
    """Derive a front's onset quarter as the earliest onset among its lineages."""
    onset_quarters = []
    for lin_id in mapped_lineages:
        oq = lineage_onsets.get(lin_id)
        if oq is not None:
            onset_quarters.append(oq)
    if not onset_quarters:
        return None
    return min(onset_quarters, key=quarter_key)


def compute_front_growth_columns(front_df: pd.DataFrame) -> pd.DataFrame:
    """Compute growth rate, acceleration, rolling stats, and YoY delta.

    Operates on a single front's records sorted by quarter.
    """
    nw = front_df["new_works"].values.astype(float)
    n = len(nw)

    # Growth rate: (nw[t] - nw[t-1]) / max(nw[t-1], 1)
    growth_rate = np.full(n, np.nan)
    for i in range(1, n):
        growth_rate[i] = (nw[i] - nw[i - 1]) / max(nw[i - 1], 1.0)
    front_df["growth_rate"] = growth_rate

    # Growth acceleration: growth_rate[t] - growth_rate[t-1]
    growth_accel = np.full(n, np.nan)
    for i in range(2, n):
        if not (np.isnan(growth_rate[i]) or np.isnan(growth_rate[i - 1])):
            growth_accel[i] = growth_rate[i] - growth_rate[i - 1]
    front_df["growth_acceleration"] = growth_accel

    # Trailing 4-quarter rolling mean and std
    roll_mean = np.full(n, np.nan)
    roll_std = np.full(n, np.nan)
    for i in range(n):
        window_start = max(0, i - 3)
        window = nw[window_start : i + 1]
        roll_mean[i] = float(np.mean(window))
        roll_std[i] = float(np.std(window, ddof=0)) if len(window) > 1 else 0.0
    front_df["new_works_roll_mean_4q"] = roll_mean
    front_df["new_works_roll_std_4q"] = roll_std

    # Year-over-year delta
    yoy = np.full(n, np.nan)
    for i in range(4, n):
        yoy[i] = nw[i] - nw[i - 4]
    front_df["new_works_yoy_delta"] = yoy

    return front_df


def assign_lifecycle_stage(
    quarters_since_onset: int | None,
    onset_detected: bool,
    _new_works_roll_mean_4q: float,
    _peak_roll_mean: float,
    consecutive_decline: int,
    growth_window: int = 12,
) -> str:
    """Assign lifecycle stage for a single front-quarter record."""
    if not onset_detected:
        return "no_onset"
    if quarters_since_onset is None or quarters_since_onset < 0:
        return "pre_onset"
    if quarters_since_onset <= growth_window:
        return "growth"
    if consecutive_decline >= 4:
        return "decline"
    return "mature"


def build_front_series(
    lineage_features: pd.DataFrame,
    lineage_to_front: dict[int, str],
    lineage_onsets: dict[int, str],
    growth_window: int = 12,
) -> pd.DataFrame:
    """Aggregate lineage features to front-level series.

    Returns a DataFrame conforming to the front-level series contract.
    """
    # Filter to mapped lineages only
    mapped_ids = set(lineage_to_front.keys())
    df = lineage_features[lineage_features["lineage_id"].isin(mapped_ids)].copy()
    df["front_id"] = df["lineage_id"].map(lineage_to_front)

    if df.empty:
        LOG.warning("No lineage features matched the mapping. Output will be empty.")
        return pd.DataFrame()

    all_quarters = sorted(df["quarter"].unique(), key=quarter_key)
    all_fronts = sorted(df["front_id"].unique())
    LOG.info(
        "Aggregating %d lineage records -> %d fronts x %d quarters",
        len(df), len(all_fronts), len(all_quarters),
    )

    # Identify which optional feature columns exist in the data
    available_sum = [c for c in SUM_FEATURES if c in df.columns]
    available_mean = [c for c in MEAN_FEATURES if c in df.columns]
    available_context = [c for c in CONTEXT_MEAN_FEATURES if c in df.columns]
    available_conv = [c for c in CONVERGENCE_MEAN_FEATURES if c in df.columns]

    # Build aggregation spec
    agg_spec: dict[str, tuple[str, str]] = {}
    for col in available_sum:
        agg_spec[col] = (col, "sum")
    for col in available_mean + available_context + available_conv:
        agg_spec[col] = (col, "mean")
    # Count of active lineages
    agg_spec["n_lineages"] = ("lineage_id", "nunique")

    grouped = df.groupby(["front_id", "quarter"]).agg(**agg_spec).reset_index()

    # Build complete (front_id, quarter) grid to fill gaps with zeros
    grid = pd.MultiIndex.from_product(
        [all_fronts, all_quarters], names=["front_id", "quarter"],
    )
    grid_df = pd.DataFrame(index=grid).reset_index()
    result = grid_df.merge(grouped, on=["front_id", "quarter"], how="left")

    # Fill missing numeric columns with 0
    numeric_cols = available_sum + available_mean + available_context + available_conv
    for col in numeric_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0.0)
    result["n_lineages"] = result["n_lineages"].fillna(0).astype(int)

    # Ensure new_works is present and integer
    if "new_works" not in result.columns:
        result["new_works"] = 0
    result["new_works"] = result["new_works"].fillna(0).astype(int)

    # Sort for stable output
    result = result.sort_values(
        ["front_id", "quarter"],
        key=lambda col: col.map(quarter_key) if col.name == "quarter" else col,
    ).reset_index(drop=True)

    # Cumulative works per front
    result["cumulative_works"] = result.groupby("front_id")["new_works"].cumsum()

    # Cumulative distinct lineages per front
    # Track which lineages have ever appeared per front
    front_lineage_sets: dict[str, set[int]] = {f: set() for f in all_fronts}
    lineage_cum_counts: list[int] = []
    for _, row in result.iterrows():
        fid = row["front_id"]
        q = row["quarter"]
        # Find lineages active in this front-quarter
        mask = (df["front_id"] == fid) & (df["quarter"] == q)
        active_lineages = set(df.loc[mask, "lineage_id"].unique())
        front_lineage_sets[fid] |= active_lineages
        lineage_cum_counts.append(len(front_lineage_sets[fid]))
    result["n_lineages_cumulative"] = lineage_cum_counts

    # Compute growth columns per front
    front_dfs = []
    for fid in all_fronts:
        fdf = result[result["front_id"] == fid].copy()
        fdf = compute_front_growth_columns(fdf)
        front_dfs.append(fdf)
    result = pd.concat(front_dfs, ignore_index=True)

    # Derive front-level onset annotations
    front_onsets: dict[str, str | None] = {}
    front_mapped_lineages: dict[str, set[int]] = {f: set() for f in all_fronts}
    for lin_id, fid in lineage_to_front.items():
        if fid in front_mapped_lineages:
            front_mapped_lineages[fid].add(lin_id)

    for fid in all_fronts:
        front_onsets[fid] = derive_front_onset(
            fid, front_mapped_lineages[fid], lineage_onsets,
        )

    onset_detected_col = []
    onset_quarter_col = []
    quarters_since_onset_col = []
    is_onset_quarter_col = []
    lifecycle_stage_col = []

    # Track rolling mean peaks and decline counters for lifecycle
    front_peak_roll: dict[str, float] = dict.fromkeys(all_fronts, 0.0)
    front_decline_count: dict[str, int] = dict.fromkeys(all_fronts, 0)
    prev_roll_mean: dict[str, float] = dict.fromkeys(all_fronts, 0.0)

    for _, row in result.iterrows():
        fid = row["front_id"]
        q = row["quarter"]
        oq = front_onsets.get(fid)
        has_onset = oq is not None

        onset_detected_col.append(1 if has_onset else 0)
        onset_quarter_col.append(oq if has_onset else None)

        if has_onset and oq is not None:
            q_idx = all_quarters.index(q)
            oq_idx = all_quarters.index(oq) if oq in all_quarters else None
            if oq_idx is not None:
                qso = q_idx - oq_idx
                quarters_since_onset_col.append(qso if qso >= 0 else None)
                is_onset_quarter_col.append(1 if qso == 0 else 0)
            else:
                quarters_since_onset_col.append(None)
                is_onset_quarter_col.append(0)
        else:
            quarters_since_onset_col.append(None)
            is_onset_quarter_col.append(0)

        # Lifecycle tracking
        rm = row.get("new_works_roll_mean_4q", 0.0)
        if not np.isnan(rm):
            if rm > front_peak_roll[fid]:
                front_peak_roll[fid] = rm
                front_decline_count[fid] = 0
            elif rm < prev_roll_mean.get(fid, 0.0):
                front_decline_count[fid] += 1
            else:
                front_decline_count[fid] = 0
            prev_roll_mean[fid] = rm

        qso_val = quarters_since_onset_col[-1]
        lifecycle_stage_col.append(assign_lifecycle_stage(
            quarters_since_onset=qso_val,
            onset_detected=has_onset,
            new_works_roll_mean_4q=rm if not np.isnan(rm) else 0.0,
            peak_roll_mean=front_peak_roll[fid],
            consecutive_decline=front_decline_count[fid],
            growth_window=growth_window,
        ))

    result["onset_detected"] = onset_detected_col
    result["onset_quarter"] = onset_quarter_col
    result["quarters_since_onset"] = quarters_since_onset_col
    result["is_onset_quarter"] = is_onset_quarter_col
    result["lifecycle_stage"] = lifecycle_stage_col

    # Count onset lineages per front
    n_onset_lineages = []
    for _, row in result.iterrows():
        fid = row["front_id"]
        count = sum(
            1 for lin_id in front_mapped_lineages.get(fid, set())
            if lin_id in lineage_onsets
        )
        n_onset_lineages.append(count)
    result["n_onset_lineages"] = n_onset_lineages

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute front-level features from lineage-level aggregates.",
    )
    parser.add_argument(
        "--lineage-features",
        default=None,
        help="Path to lineage multisignal features CSV.",
    )
    parser.add_argument(
        "--mapping",
        default=None,
        help="Path to lineage-to-front mapping CSV.",
    )
    parser.add_argument(
        "--onset-labels",
        default=None,
        help="Path to onset labels CSV. If provided, onset annotations are added.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.65,
        help="Minimum similarity threshold for tight mappings (default: 0.65).",
    )
    parser.add_argument(
        "--growth-window",
        type=int,
        default=12,
        help="Quarters after onset classified as 'growth' stage (default: 12).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for front-level series CSV.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    add_domain_args(parser)
    return parser.parse_args()


def main() -> None:
    """Entry point for front-level feature computation."""
    args = parse_args()
    configure_logging(args.verbose)

    paths = resolve_script_paths(args, REPO_ROOT)
    if args.lineage_features is None:
        args.lineage_features = str(paths.lineage_tracking / "lineage_multisignal_features.csv") if paths else "data/out/02_lineage_tracking/lineage_multisignal_features.csv"
    if args.mapping is None:
        args.mapping = str(paths.experiments / "stage0_tight_mapping/milestone_lineage_mapping_tight.csv") if paths else "data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv"
    if args.out is None:
        args.out = str(paths.front_aggregation / "front_onset_series.csv") if paths else "data/out/04_front_aggregation/front_onset_series.csv"

    LOG.info("Loading lineage features from %s", args.lineage_features)
    lineage_features = pd.read_csv(args.lineage_features)
    lineage_features["quarter"] = lineage_features["quarter"].astype(str)
    LOG.info(
        "Loaded %d records, %d lineages, %d quarters",
        len(lineage_features),
        lineage_features["lineage_id"].nunique(),
        lineage_features["quarter"].nunique(),
    )

    LOG.info("Loading mapping from %s", args.mapping)
    lineage_to_front = load_mapping(Path(args.mapping), args.min_similarity)
    LOG.info("Mapped %d lineages to fronts", len(lineage_to_front))

    # Report front distribution
    front_counts: dict[str, int] = {}
    for fid in lineage_to_front.values():
        front_counts[fid] = front_counts.get(fid, 0) + 1
    for fid in sorted(front_counts):
        LOG.info("  %-30s %d lineages", fid, front_counts[fid])

    lineage_onsets: dict[int, str] = {}
    if args.onset_labels:
        LOG.info("Loading onset labels from %s", args.onset_labels)
        lineage_onsets = load_onset_labels(Path(args.onset_labels))
        LOG.info("Loaded %d onset labels", len(lineage_onsets))

    result = build_front_series(
        lineage_features,
        lineage_to_front,
        lineage_onsets,
        growth_window=args.growth_window,
    )

    if result.empty:
        LOG.error("No output produced. Check mapping and lineage features.")
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    LOG.info("Wrote %d records to %s", len(result), out_path)

    # Summary statistics
    n_fronts = result["front_id"].nunique()
    n_quarters = result["quarter"].nunique()
    n_onset = result["onset_detected"].sum() // n_quarters if n_quarters > 0 else 0
    LOG.info(
        "Summary: %d fronts, %d quarters, %d fronts with onset detected",
        n_fronts, n_quarters, n_onset,
    )


if __name__ == "__main__":
    main()
