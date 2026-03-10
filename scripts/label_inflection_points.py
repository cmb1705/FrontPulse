#!/usr/bin/env python3
"""
Label lineage-quarter inflection points using logistic fits and derivative heuristics.

Outputs a CSV with one row per detected inflection (lineage_id, quarter, metadata).
Downstream consumers can left-join this file to multi-signal features to mark positives.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import curve_fit  # type: ignore
except Exception:  # pragma: no cover - SciPy optional
    curve_fit = None

from persistence_utils import ensure_persistence_column
from utils.quarter_utils import quarter_to_int, int_to_quarter

# Onset detector (Phase 1 prospective-safe labeling)
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_REPO))
from src.onset_detector import OnsetResult, detect_onset

LOG = logging.getLogger("inflection_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label lineage inflection points.")
    parser.add_argument(
        "--timeseries",
        default="data/out/02_lineage_tracking/lineage_timeseries.csv",
        help="Lineage time-series CSV with columns lineage_id, quarter, new_works (default: %(default)s)",
    )
    parser.add_argument(
        "--milestones",
        default="data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv",
        help="Milestone-lineage mapping CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="data/out/02_lineage_tracking/inflection_labels.csv",
        help="Path to write inflection labels CSV (default: %(default)s)",
    )
    parser.add_argument("--min-points", type=int, default=12, help="Minimum quarters required to attempt logistic fit.")
    parser.add_argument("--logistic-r2", type=float, default=0.85, help="Minimum R^2 for logistic fit acceptance.")
    parser.add_argument("--deriv-window", type=int, default=3, help="Rolling window (quarters) for derivative smoothing.")
    parser.add_argument("--deriv-k", type=float, default=1.5, help="Std-dev multiplier for derivative threshold.")
    parser.add_argument(
        "--min-cumulative-works",
        type=int,
        default=20,
        help="Minimum cumulative works required before considering a lineage for detection.",
    )
    parser.add_argument(
        "--min-recent-works",
        type=int,
        default=8,
        help="Minimum total new works in the last 4 quarters required for detection.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Optional directory to save sample plots. If omitted, plots are skipped.",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=40,
        help="Number of detected lineages to plot when --plot-dir is provided (default: %(default)s)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling plots (default: %(default)s)",
    )
    parser.add_argument(
        "--qa-borderline-csv",
        default=None,
        help="Optional path to write borderline detections (near thresholds).",
    )
    parser.add_argument(
        "--qa-borderline-delta",
        type=float,
        default=0.05,
        help="Margin within threshold considered borderline (default: %(default)s).",
    )
    parser.add_argument(
        "--qa-borderline-plot-dir",
        default=None,
        help="Optional directory to save plots for borderline detections.",
    )
    parser.add_argument(
        "--qa-borderline-plot-limit",
        type=int,
        default=40,
        help="Maximum number of borderline plots to generate (default: %(default)s).",
    )
    parser.add_argument(
        "--min-future-quarters",
        type=int,
        default=2,
        help="Minimum quarters after detected index required to keep detection.",
    )
    parser.add_argument(
        "--deriv-persistence",
        type=int,
        default=2,
        help="Number of quarters after a derivative detection that must stay above baseline.",
    )
    parser.add_argument(
        "--min-acceleration-growth",
        type=float,
        default=0.5,
        help="Minimum fractional cumulative growth required within the next window (e.g., 0.5 = +50%%).",
    )
    parser.add_argument(
        "--acceleration-window",
        type=int,
        default=4,
        help="Number of quarters after a detection to measure cumulative growth.",
    )
    parser.add_argument(
        "--field-metrics",
        default="data/out/04_front_aggregation/field_metrics.parquet",
        help="Field metrics file (parquet or CSV) used for context comparisons.",
    )
    parser.add_argument(
        "--disable-field-metrics",
        action="store_true",
        help="Skip loading field metrics (no field context columns or guards).",
    )
    parser.add_argument(
        "--min-field-growth-ratio",
        type=float,
        default=0.0,
        help="Minimum lineage vs field growth ratio required to keep a detection (0 disables).",
    )
    parser.add_argument(
        "--max-lineages",
        type=int,
        default=None,
        help="Optional limit on number of lineages to process (smoke tests).",
    )

    # Detection mode (Phase 1 onset support)
    parser.add_argument(
        "--mode",
        choices=["retrospective", "onset", "comparison"],
        default="retrospective",
        help="Detection mode: 'retrospective' (logistic/derivative), "
             "'onset' (prospective-safe onset detection), or "
             "'comparison' (both side-by-side). Default: retrospective.",
    )
    # Onset-specific parameters (see onset_label_specification.md)
    parser.add_argument(
        "--onset-smoothing-window",
        type=int,
        default=3,
        help="Trailing rolling-mean window for onset detection (default: 3).",
    )
    parser.add_argument(
        "--onset-growth-threshold",
        type=float,
        default=0.10,
        help="Minimum QoQ growth rate for onset trigger (default: 0.10).",
    )
    parser.add_argument(
        "--onset-confirmation-quarters",
        type=int,
        default=3,
        help="Consecutive quarters of positive growth required (default: 3).",
    )
    parser.add_argument(
        "--onset-min-count",
        type=int,
        default=3,
        help="Minimum smoothed count to avoid noise triggers (default: 3).",
    )

    return parser.parse_args()


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_field_metrics(path: Path) -> pd.DataFrame:
    if not path or not path.exists():
        LOG.warning("Field metrics file %s not found; continuing without field context.", path)
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df.copy()
    df["quarter"] = df["quarter"].astype(str)
    rename_map = {col: (col if col == "quarter" else f"field_{col}") for col in df.columns}
    df.rename(columns=rename_map, inplace=True)
    return df


def build_field_metrics_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}
    lookup: Dict[str, Dict[str, float]] = {}
    records = df.to_dict(orient="records")
    for row in records:
        quarter = str(row["quarter"])
        lookup[quarter] = {k: v for k, v in row.items() if k != "quarter"}
    return lookup


def logistic_func(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0)))


@dataclass
class LogisticResult:
    idx: int
    quarter_int: int
    score: float
    L: float
    k: float
    x0: float


@dataclass
class DerivativeResult:
    idx: int
    quarter_int: int
    score: float
    threshold: float


def fit_logistic(cum_values: np.ndarray, min_points: int, r2_threshold: float) -> Optional[LogisticResult]:
    if curve_fit is None or len(cum_values) < min_points:
        return None

    x = np.arange(len(cum_values), dtype=float)
    y = cum_values.astype(float)
    if np.any(np.isnan(y)) or np.allclose(y[-1], 0):
        return None

    L0 = y.max()
    if L0 <= 0:
        return None
    k0 = 1.0
    x0 = len(y) / 2.0
    bounds = ([0.5 * L0, 0.01, 0], [2 * L0, 5.0, len(y)])

    try:
        params, _ = curve_fit(logistic_func, x, y, p0=(L0, k0, x0), bounds=bounds, maxfev=20000)
        pred = logistic_func(x, *params)
    except Exception:
        return None

    residuals = y - pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    inflection = params[2]
    idx = int(round(inflection))
    if r2 < r2_threshold or idx <= 1 or idx >= len(y) - 2:
        return None
    return LogisticResult(idx=idx, quarter_int=None, score=float(r2), L=float(params[0]), k=float(params[1]), x0=float(params[2]))


def derivative_detection(
    cum_values: np.ndarray,
    window: int,
    threshold_k: float,
) -> Optional[DerivativeResult]:
    if len(cum_values) < 6:
        return None
    series = pd.Series(cum_values)
    first = series.diff().fillna(0.0)
    if window > 1:
        first = first.rolling(window=window, min_periods=1, center=True).mean()
    baseline = first.mean()
    std = first.std()
    if pd.isna(std) or std == 0:
        return None
    threshold = baseline + threshold_k * std
    second = first.diff().fillna(0.0)
    mask = (first >= threshold) & (second > 0)
    idx_candidates = np.where(mask)[0]
    if len(idx_candidates) == 0:
        return None
    idx = int(idx_candidates[0])
    score = float(first.iloc[idx])
    return DerivativeResult(idx=idx, quarter_int=None, score=score, threshold=float(threshold))


def build_milestone_lookup(milestone_path: Path) -> Dict[int, List[Tuple[int, str]]]:
    if not milestone_path.exists():
        LOG.warning("Milestone file %s not found; lag metrics will be null.", milestone_path)
        return {}
    df = pd.read_csv(milestone_path)
    lookup: Dict[int, List[Tuple[int, str]]] = {}
    for _, row in df.iterrows():
        try:
            lineage_id = int(row["lineage_id"])
            quarter_int = quarter_to_int(str(row["event_quarter"]))
        except Exception:
            continue
        lookup.setdefault(lineage_id, []).append((quarter_int, str(row["event_id"])))
    return lookup


def find_nearest_milestone(
    lineage_id: int,
    inflection_q: int,
    lookup: Dict[int, List[Tuple[int, str]]],
) -> Tuple[Optional[str], Optional[int], str]:
    candidates = lookup.get(lineage_id)
    if not candidates:
        return None, None, "no_milestone"
    nearest = min(candidates, key=lambda pair: abs(pair[0] - inflection_q))
    event_quarter, event_id = nearest
    lag = inflection_q - event_quarter
    bucket = bucket_lag(lag)
    return event_id, lag, bucket


def bucket_lag(lag: Optional[int]) -> str:
    if lag is None:
        return "no_milestone"
    abs_lag = abs(lag)
    if abs_lag < 4:
        return "<4q"
    if abs_lag < 8:
        return "4-8q"
    if abs_lag < 12:
        return "8-12q"
    return ">=12q"


def detect_inflection_for_lineage(
    lineage_id: int,
    lineage_df: pd.DataFrame,
    milestone_lookup: Dict[int, List[Tuple[int, str]]],
    field_metrics: Optional[Dict[str, Dict[str, float]]],
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    lineage_df = lineage_df.sort_values("quarter_order")
    total_cum = lineage_df["new_works"].sum()
    recent_cum = lineage_df["new_works"].tail(4).sum()
    if total_cum < args.min_cumulative_works or recent_cum < args.min_recent_works:
        return None
    cum_works = lineage_df["new_works"].cumsum().to_numpy(dtype=float)
    if len(cum_works) == 0 or np.allclose(cum_works[-1], 0):
        return None

    logistic_result = fit_logistic(cum_works, args.min_points, args.logistic_r2)
    result_type = None
    score = None
    idx = None
    threshold = None
    margin = None
    tail_gap = args.min_future_quarters if hasattr(args, "min_future_quarters") else 2
    future_window = args.acceleration_window
    min_growth = args.min_acceleration_growth

    if logistic_result:
        idx = logistic_result.idx
        score = logistic_result.score
        result_type = "logistic"
        threshold = args.logistic_r2
        margin = score - threshold if threshold is not None else None
    else:
        derivative_result = derivative_detection(cum_works, args.deriv_window, args.deriv_k)
        if derivative_result:
            idx = derivative_result.idx
            score = derivative_result.score
            result_type = "derivative"
            threshold = derivative_result.threshold
            margin = score - threshold if threshold is not None else None
            # Require follow-through: at least args.deriv_persistence positives after idx
            persistence = lineage_df["new_works"].iloc[idx + 1 : idx + 1 + args.deriv_persistence].sum()
            if persistence < args.min_recent_works / 2:
                return None
    if idx is None or idx >= len(lineage_df) - tail_gap:
        return None

    # Minimum acceleration requirement: cumulative must grow by >= min_growth fraction within next window
    current_cum = cum_works[idx]
    future_end = min(len(cum_works) - 1, idx + future_window)
    future_cum = cum_works[future_end]
    if current_cum <= 0:
        return None
    growth_fraction = (future_cum - current_cum) / current_cum
    if growth_fraction < min_growth:
        return None

    quarter_int = int(lineage_df.iloc[idx]["quarter_int"])
    quarter_label = str(lineage_df.iloc[idx]["quarter"])
    field_context = field_metrics.get(quarter_label) if field_metrics else None
    lineage_new_works = float(lineage_df.iloc[idx]["new_works"])
    field_growth_ratio = None
    field_cumulative_ratio = None
    field_growth_fraction_delta = None
    field_acceleration_delta = None

    if field_context:
        field_total = field_context.get("field_total_new_works")
        if field_total and field_total > 0:
            field_growth_ratio = lineage_new_works / field_total
        field_cumulative = field_context.get("field_cumulative_new_works")
        if field_cumulative and field_cumulative > 0:
            field_cumulative_ratio = current_cum / field_cumulative
        field_growth_rate = field_context.get("field_cumulative_growth_rate")
        if field_growth_rate is not None and not pd.isna(field_growth_rate):
            field_growth_fraction_delta = growth_fraction - float(field_growth_rate)
        field_acceleration = field_context.get("field_cumulative_acceleration")
        if field_acceleration is not None and not pd.isna(field_acceleration):
            field_acceleration_delta = growth_fraction - float(field_acceleration)

    if args.min_field_growth_ratio > 0:
        if field_growth_ratio is None or field_growth_ratio < args.min_field_growth_ratio:
            return None

    event_id, lag, bucket = find_nearest_milestone(lineage_id, quarter_int, milestone_lookup)
    return {
        "lineage_id": lineage_id,
        "quarter": quarter_label,
        "is_inflection_onset": 1,
        "inflection_score": score,
        "inflection_type": result_type,
        "nearest_milestone_id": event_id,
        "lag_since_milestone": lag,
        "lag_bucket": bucket,
        "inflection_threshold": threshold,
        "threshold_margin": margin,
        "field_growth_ratio": field_growth_ratio,
        "field_cumulative_ratio": field_cumulative_ratio,
        "field_growth_fraction_delta": field_growth_fraction_delta,
        "field_acceleration_delta": field_acceleration_delta,
        **(field_context or {}),
    }


# ---------------------------------------------------------------------------
# Onset detection mode (Phase 1)
# ---------------------------------------------------------------------------


def detect_onset_for_lineage(
    lineage_id: int,
    lineage_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Run onset detection on a single lineage and return a result dict.

    Always returns a dict (one row per lineage), even when onset is not
    detected, matching the output schema in onset_label_specification.md.
    """
    sorted_df = lineage_df.sort_values("quarter_int")
    quarters = sorted_df["quarter"].tolist()
    counts = sorted_df["new_works"].astype(int).tolist()

    result: OnsetResult = detect_onset(
        quarters,
        counts,
        smoothing_window=args.onset_smoothing_window,
        growth_threshold=args.onset_growth_threshold,
        confirmation_quarters=args.onset_confirmation_quarters,
        min_count=args.onset_min_count,
    )

    return {
        "lineage_id": lineage_id,
        "onset_quarter": result.quarter or "",
        "onset_detected": int(result.detected),
        "onset_reason": result.reason,
        "onset_growth_rate": result.growth_rate,
        "onset_smoothed_count": result.smoothed_count,
        "onset_confirmation_length": result.confirmation_length,
        "early_onset": result.early_onset,
        "smoothing_window": args.onset_smoothing_window,
        "growth_threshold": args.onset_growth_threshold,
        "confirmation_window": args.onset_confirmation_quarters,
    }


def prepare_timeseries(ts_path: Path) -> pd.DataFrame:
    df = pd.read_csv(ts_path)
    required_cols = {"lineage_id", "quarter", "new_works"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Timeseries file missing columns: {missing}")
    df = df.copy()
    df["lineage_id"] = df["lineage_id"].astype(int)
    df["quarter"] = df["quarter"].astype(str)
    df["quarter_int"] = df["quarter"].apply(quarter_to_int)
    df["quarter_order"] = df.groupby("lineage_id")["quarter_int"].rank(method="first").astype(int) - 1
    df["new_works"] = df["new_works"].fillna(0).astype(float)
    return df


def save_metadata(out_path: Path, payload: Dict[str, object]) -> None:
    meta_path = out_path.with_suffix(".json")
    ensure_directory(meta_path)
    meta_path.write_text(json.dumps(payload, indent=2))


def generate_plots(
    detections: pd.DataFrame,
    ts_df: pd.DataFrame,
    samples: int,
    plot_dir: Path,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - matplotlib optional
        LOG.warning("matplotlib unavailable (%s); skipping plots.", exc)
        return

    lineages = detections["lineage_id"].unique().tolist()
    if not lineages:
        return
    random.Random(seed).shuffle(lineages)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for lineage_id in lineages[:samples]:
        lineage_df = ts_df[ts_df["lineage_id"] == lineage_id].sort_values("quarter_order")
        cumulative = lineage_df["new_works"].cumsum()
        plt.figure(figsize=(8, 4))
        plt.plot(lineage_df["quarter"], cumulative, label="Cumulative works")
        det_row = detections[detections["lineage_id"] == lineage_id].iloc[0]
        plt.axvline(det_row["quarter"], color="red", linestyle="--", label="Detected inflection")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Lineage {lineage_id}")
        plt.tight_layout()
        plot_path = plot_dir / f"lineage_{lineage_id}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()


def plot_borderline_detection(
    lineage_df: pd.DataFrame,
    detection_row: pd.Series,
    out_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        LOG.warning("matplotlib unavailable (%s); skipping borderline plot for lineage %s.", exc, detection_row["lineage_id"])
        return

    lineage_df = lineage_df.sort_values("quarter_order")
    cumulative = lineage_df["new_works"].cumsum()
    quarter_labels = lineage_df["quarter"]

    plt.figure(figsize=(9, 4))
    plt.plot(quarter_labels, cumulative, label="Cumulative works", color="steelblue")
    plt.scatter(quarter_labels, lineage_df["new_works"], label="New works", color="gray", alpha=0.5, s=15)

    det_quarter = detection_row["quarter"]
    plt.axvline(det_quarter, color="red", linestyle="--", label=f"Detected ({det_quarter})")

    annotation = (
        f"Type: {detection_row['inflection_type']}\n"
        f"Score: {detection_row['inflection_score']:.3f}\n"
        f"Threshold: {detection_row['inflection_threshold']:.3f}\n"
        f"Margin: {detection_row['threshold_margin']:.3f}\n"
        f"Lag bucket: {detection_row['lag_bucket']}"
    )
    text_box = plt.text(
        0.02,
        0.98,
        annotation,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", boxstyle="round,pad=0.4"),
    )

    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cumulative works")
    plt.title(f"Lineage {detection_row['lineage_id']} borderline detection")
    legend = plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ts_path = Path(args.timeseries)
    milestone_path = Path(args.milestones)
    out_path = Path(args.out)

    LOG.info("Loading timeseries from %s", ts_path)
    ts_df = prepare_timeseries(ts_path)
    if args.max_lineages:
        keep_ids = sorted(ts_df["lineage_id"].unique())[: args.max_lineages]
        ts_df = ts_df[ts_df["lineage_id"].isin(keep_ids)]
        LOG.info("Limiting processing to %d lineages for smoke run.", len(keep_ids))

    mode = getattr(args, "mode", "retrospective")
    LOG.info("Detection mode: %s", mode)
    total_lineages = ts_df["lineage_id"].nunique()
    LOG.info("Processing %d lineages...", total_lineages)

    # ── Onset-only mode ────────────────────────────────────────────────
    if mode == "onset":
        onset_rows: List[Dict[str, object]] = []
        for lineage_id, group in ts_df.groupby("lineage_id"):
            onset_rows.append(detect_onset_for_lineage(int(lineage_id), group, args))

        onset_df = pd.DataFrame(onset_rows)
        ensure_directory(out_path)
        onset_df.to_csv(out_path, index=False)

        n_detected = int(onset_df["onset_detected"].sum())
        LOG.info(
            "Wrote %d onset labels (%d detected, %.1f%%) to %s",
            len(onset_df), n_detected,
            100 * n_detected / max(len(onset_df), 1), out_path,
        )

        metadata = {
            "timeseries": str(ts_path),
            "output": str(out_path),
            "mode": "onset",
            "total_lineages": total_lineages,
            "onset_detected": n_detected,
            "onset_pct": round(100 * n_detected / max(total_lineages, 1), 2),
            "parameters": {
                "smoothing_window": args.onset_smoothing_window,
                "growth_threshold": args.onset_growth_threshold,
                "confirmation_quarters": args.onset_confirmation_quarters,
                "min_count": args.onset_min_count,
            },
        }
        save_metadata(out_path, metadata)
        return

    # ── Retrospective or comparison mode ───────────────────────────────
    milestone_lookup = build_milestone_lookup(milestone_path)
    field_metrics_lookup: Dict[str, Dict[str, float]] = {}
    if not args.disable_field_metrics:
        field_metrics_df = load_field_metrics(Path(args.field_metrics))
        field_metrics_lookup = build_field_metrics_lookup(field_metrics_df)
        LOG.info("Loaded field metrics for %d quarters.", len(field_metrics_lookup))
    else:
        LOG.info("Field metrics disabled; skipping corpus-level guard.")

    detections: List[Dict[str, object]] = []
    onset_rows_cmp: List[Dict[str, object]] = []

    for lineage_id, group in ts_df.groupby("lineage_id"):
        lid = int(lineage_id)
        result = detect_inflection_for_lineage(
            lid, group, milestone_lookup, field_metrics_lookup, args,
        )
        if result:
            detections.append(result)
        if mode == "comparison":
            onset_rows_cmp.append(detect_onset_for_lineage(lid, group, args))

    if not detections:
        LOG.warning("No inflections detected.")
        if mode != "comparison":
            return

    detections_df = pd.DataFrame(detections) if detections else pd.DataFrame()
    ensure_directory(out_path)

    if mode == "comparison" and onset_rows_cmp:
        # Merge retrospective and onset labels side-by-side
        onset_df = pd.DataFrame(onset_rows_cmp)
        if not detections_df.empty:
            combined = detections_df.merge(onset_df, on="lineage_id", how="outer")
        else:
            combined = onset_df
        combined.to_csv(out_path, index=False)
        n_retro = len(detections_df)
        n_onset = int(onset_df["onset_detected"].sum())
        LOG.info(
            "Comparison mode: %d retrospective + %d onset labels -> %s",
            n_retro, n_onset, out_path,
        )
    else:
        detections_df.to_csv(out_path, index=False)
    LOG.info("Wrote %d inflections to %s", len(detections_df), out_path)

    metadata = {
        "timeseries": str(ts_path),
        "milestones": str(milestone_path),
        "output": str(out_path),
        "mode": mode,
        "total_lineages": total_lineages,
        "detections": len(detections_df),
        "logistic_success": int((detections_df["inflection_type"] == "logistic").sum()) if "inflection_type" in detections_df.columns else 0,
        "derivative_success": int((detections_df["inflection_type"] == "derivative").sum()) if "inflection_type" in detections_df.columns else 0,
        "parameters": {
            "min_points": args.min_points,
            "logistic_r2": args.logistic_r2,
            "deriv_window": args.deriv_window,
            "deriv_k": args.deriv_k,
            "min_field_growth_ratio": args.min_field_growth_ratio,
            "field_metrics": None if args.disable_field_metrics else str(args.field_metrics),
        },
    }
    if args.qa_borderline_csv or args.qa_borderline_plot_dir:
        borderline = detections_df[
            detections_df["threshold_margin"].notna()
            & (detections_df["threshold_margin"] <= args.qa_borderline_delta)
        ].copy()
        borderline.sort_values(["inflection_type", "threshold_margin"], inplace=True)

    if args.qa_borderline_csv:
        qa_path = Path(args.qa_borderline_csv)
        ensure_directory(qa_path)
        borderline.to_csv(qa_path, index=False)
        metadata["qa_borderline_csv"] = str(qa_path)
        metadata["qa_borderline_count"] = int(len(borderline))
        LOG.info(
            "Wrote %d borderline detections (margin <= %.3f) to %s",
            len(borderline),
            args.qa_borderline_delta,
            qa_path,
        )

    if args.qa_borderline_plot_dir and not borderline.empty:
        plot_dir = Path(args.qa_borderline_plot_dir)
        limited = borderline.head(args.qa_borderline_plot_limit)
        for _, row in limited.iterrows():
            lineage_id = int(row["lineage_id"])
            lineage_df = ts_df[ts_df["lineage_id"] == lineage_id]
            out_file = plot_dir / f"lineage_{lineage_id}_{row['quarter'].replace('/', '-')}.png"
            plot_borderline_detection(lineage_df, row, out_file)
        metadata["qa_borderline_plot_dir"] = str(plot_dir)
        metadata["qa_borderline_plots"] = int(len(limited))
        LOG.info(
            "Saved %d borderline plots (limit %d) to %s",
            len(limited),
            args.qa_borderline_plot_limit,
            plot_dir,
        )

    save_metadata(out_path, metadata)

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        LOG.info("Saving sample plots to %s", plot_dir)
        generate_plots(detections_df, ts_df, args.plot_samples, plot_dir, args.random_seed)


if __name__ == "__main__":
    main()
