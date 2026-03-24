#!/usr/bin/env python3
"""
Compute lineage-level multi-signal features to support breakthrough detection.

Signals implemented:
  • NPMI-based novelty: count/rate of new technical terms per lineage-quarter.
  • Dormancy vs. awakening: consecutive zero-output quarters and rebound intensity.
  • Cross-domain citation share: fraction of references pointing outside the lineage.
  • Citation disruption (Wu et al. style CD index) using future citing works.

Outputs a CSV keyed by (lineage_id, quarter).
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # Optional dependency used for logistic fits
    from scipy.optimize import curve_fit  # type: ignore
except Exception:  # pragma: no cover - SciPy optional
    curve_fit = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.trusted_io import load_trusted_pickle, save_trusted_pickle  # type: ignore

from scripts.compute_lineage_ctfidf import (  # type: ignore
    TECHNICAL_BIGRAMS,
    FORMULA_PATTERNS,
    STOPWORDS,
)
from src.lineage_text_store import LineageTextStore  # type: ignore
from src.raw_store import RawStore  # type: ignore
from utils.quarter_utils import quarter_key, quarter_to_int, int_to_quarter


LOG = logging.getLogger("lineage_multisignal")
_MP_REFERENCES_BY_WORK: Dict[str, List[str]] = {}
_MP_WORK_LINEAGE: Dict[str, int] = {}
_MP_CITED_BY_MAP: Dict[str, set] = {}
_MP_PUB_YEAR_BY_WORK: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# Utility helpers


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_field_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOG.warning("Field metrics file %s not found; skipping field-relative features.", path)
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df.copy()
    df["quarter"] = df["quarter"].astype(str)
    return df


def merge_field_metrics(features_df: pd.DataFrame, field_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge field-level aggregates and derive relative lineage features.

    Parameters
    ----------
    features_df : pd.DataFrame
        Lineage-quarter feature table.
    field_metrics_df : pd.DataFrame
        Field-level aggregates with a `quarter` column.
    """
    if features_df.empty or field_metrics_df.empty:
        return features_df

    fm = field_metrics_df.rename(columns=lambda c: f"field_{c}" if c != "quarter" else c)
    merged = features_df.merge(fm, on="quarter", how="left")

    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denom = denominator.replace(0, np.nan)
        ratio = numerator / denom
        return ratio.replace([np.inf, -np.inf], 0).fillna(0)

    if {"new_works", "field_total_new_works"}.issubset(merged.columns):
        merged["relative_new_works"] = _safe_ratio(merged["new_works"], merged["field_total_new_works"])
    else:
        merged["relative_new_works"] = 0.0

    if {"cumulative_works", "field_cumulative_new_works"}.issubset(merged.columns):
        merged["relative_cumulative_works"] = _safe_ratio(
            merged["cumulative_works"],
            merged["field_cumulative_new_works"],
        )
    else:
        merged["relative_cumulative_works"] = 0.0

    merged["growth_vs_field"] = merged["growth_rate_diff"] - merged.get("field_total_new_works_diff", 0)
    merged["acceleration_vs_field"] = merged["growth_acceleration"] - merged.get("field_cumulative_acceleration", 0)

    if {"new_works", "field_new_works_p75"}.issubset(merged.columns):
        merged["new_works_over_p75"] = _safe_ratio(merged["new_works"], merged["field_new_works_p75"])
    else:
        merged["new_works_over_p75"] = 0.0

    return merged


def logistic_func(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    return L / (1.0 + np.exp(-k * (x - x0)))


def fit_logistic_curve(values: np.ndarray, min_points: int = 8, r2_threshold: float = 0.7) -> Optional[Dict[str, float]]:
    if curve_fit is None or len(values) < min_points:
        return None
    if np.allclose(values[-1], 0):
        return None
    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    L0 = float(y.max())
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
    if r2 < r2_threshold:
        return None
    return {
        "logistic_carrying_capacity": float(params[0]),
        "logistic_growth_rate": float(params[1]),
        "logistic_midpoint_idx": float(params[2]),
        "logistic_fit_r2": float(r2),
    }


def default_logistic_params() -> Dict[str, Any]:
    return {
        "logistic_carrying_capacity": 0.0,
        "logistic_growth_rate": 0.0,
        "logistic_midpoint_idx": float("nan"),
        "logistic_midpoint_quarter": None,
        "logistic_fit_r2": 0.0,
    }


def compute_growth_features(timeseries_df: pd.DataFrame) -> Dict[Tuple[int, str], Dict[str, float]]:
    features: Dict[Tuple[int, str], Dict[str, float]] = {}
    grouped = timeseries_df.groupby("lineage_id")
    logistic_available = curve_fit is not None

    for lineage_id, group in grouped:
        group_sorted = group.sort_values("quarter", key=lambda col: col.map(quarter_key)).reset_index(drop=True)
        new_works = group_sorted["new_works"].fillna(0.0).astype(float)
        cumulative = new_works.cumsum()

        growth_rate = new_works.diff().fillna(0.0)
        growth_acceleration = growth_rate.diff().fillna(0.0)

        # Year-over-year comparisons (shift by 4 quarters)
        new_works_yoy = new_works.shift(4)
        growth_rate_yoy = growth_rate.shift(4)
        new_works_yoy_delta = (new_works - new_works_yoy).fillna(0.0)
        new_works_yoy_ratio = (new_works / new_works_yoy).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        growth_rate_yoy_delta = (growth_rate - growth_rate_yoy).fillna(0.0)

        roll_mean_2 = new_works.rolling(window=2, min_periods=1).mean()
        roll_mean_4 = new_works.rolling(window=4, min_periods=1).mean()
        roll_std_4 = new_works.rolling(window=4, min_periods=2).std().fillna(0.0)

        logistic_params_per_idx: List[Dict[str, Any]] = [
            default_logistic_params() for _ in range(len(group_sorted))
        ]
        if logistic_available:
            for idx in range(len(group_sorted)):
                fit = fit_logistic_curve(cumulative.iloc[: idx + 1].to_numpy())
                if not fit:
                    continue
                params = default_logistic_params()
                params.update(fit)
                mid_idx = int(round(fit["logistic_midpoint_idx"]))
                mid_idx = min(max(mid_idx, 0), idx)
                params["logistic_midpoint_quarter"] = group_sorted.iloc[mid_idx]["quarter"]
                logistic_params_per_idx[idx] = params

        for idx, row in group_sorted.iterrows():
            key = (int(lineage_id), str(row["quarter"]))
            features[key] = {
                "cumulative_works": float(cumulative.loc[idx]),
                "growth_rate_diff": float(growth_rate.loc[idx]),
                "growth_acceleration": float(growth_acceleration.loc[idx]),
                "new_works_roll_mean_2q": float(roll_mean_2.loc[idx]),
                "new_works_roll_mean_4q": float(roll_mean_4.loc[idx]),
                "new_works_roll_std_4q": float(roll_std_4.loc[idx]),
                "new_works_yoy_delta": float(new_works_yoy_delta.loc[idx]),
                "new_works_yoy_ratio": float(new_works_yoy_ratio.loc[idx]),
                "growth_rate_yoy_delta": float(growth_rate_yoy_delta.loc[idx]),
            }
            features[key].update(logistic_params_per_idx[idx])

    if not logistic_available:
        LOG.warning("SciPy not available; logistic features set to defaults.")
    return features


def load_milestone_lookup(milestone_path: Path) -> Dict[int, List[Tuple[int, str]]]:
    if not milestone_path.exists():
        LOG.warning("Milestone file %s not found; milestone proximity features disabled.", milestone_path)
        return {}
    df = pd.read_csv(milestone_path)
    lookup: Dict[int, List[Tuple[int, str]]] = {}
    for _, row in df.iterrows():
        try:
            lineage_id = int(row["lineage_id"])
            quarter = str(row["event_quarter"])
        except Exception:
            continue
        lookup.setdefault(lineage_id, []).append((quarter_to_int(quarter), str(row.get("event_id", ""))))
    for lineage_id, entries in lookup.items():
        lookup[lineage_id] = sorted(entries, key=lambda pair: pair[0])
    return lookup


def compute_milestone_proximity(
    timeseries_df: pd.DataFrame,
    milestone_path: Path,
) -> Dict[Tuple[int, str], Dict[str, float]]:
    lookup = load_milestone_lookup(milestone_path)
    if not lookup:
        return {}

    features: Dict[Tuple[int, str], Dict[str, float]] = {}
    for lineage_id, group in timeseries_df.groupby("lineage_id"):
        group_sorted = group.sort_values("quarter", key=lambda col: col.map(quarter_key)).reset_index(drop=True)
        milestones = lookup.get(int(lineage_id), [])
        if not milestones:
            continue
        milestone_idx = 0
        prev_quarter = None
        for _, row in group_sorted.iterrows():
            quarter_label = str(row["quarter"])
            quarter_int = quarter_to_int(quarter_label)
            while milestone_idx < len(milestones) and milestones[milestone_idx][0] < quarter_int:
                prev_quarter = milestones[milestone_idx]
                milestone_idx += 1
            since = float(quarter_int - prev_quarter[0]) if prev_quarter else float("nan")
            features[(int(lineage_id), quarter_label)] = {
                "quarters_since_last_milestone": since,
                "last_milestone_id": prev_quarter[1] if prev_quarter else None,
            }
    return features


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using the same normalization as Phase 3 (c-TF-IDF).
    """
    import re

    text = text.lower()
    for bigram in TECHNICAL_BIGRAMS:
        lower = bigram.lower()
        text = text.replace(lower, lower.replace(" ", "_"))
    for pattern, normalized in FORMULA_PATTERNS.items():
        text = re.sub(pattern, normalized, text)

    tokens = re.findall(r"\b[\w\-]+\b", text)
    return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]


def load_partition(partition_path: Path) -> Dict[int, List[str]]:
    """
    Load partition JSON and invert {work_id -> community} to {community -> [work_ids]}.
    """
    data = json.loads(partition_path.read_text())
    inverted: Dict[int, List[str]] = defaultdict(list)
    for work_id, comm_id in data.get("labels", {}).items():
        inverted[int(comm_id)].append(work_id)
    return inverted


def iter_raw_records(raw_dir: Path) -> Iterable[Tuple[str, Dict]]:
    """
    Iterate through all works in the raw ingest.
    """
    base_paths = sorted(raw_dir.glob("openalex_raw_*_part*.jsonl"))
    for jsonl_path in base_paths:
        base = jsonl_path.with_suffix("")
        store = RawStore.from_basepath(base)
        for work_id in store._index:
            yield work_id, store.get_json(work_id)
        store.close()


# ---------------------------------------------------------------------------
# Global metric loading and context features


def load_global_metrics(metrics_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Load global metric parquet files and return dict keyed by quarter.

    Returns:
        global_metrics[quarter] = {
            'author_influx': value,
            'citation_velocity': value,
            'reference_vitality': value,
            'topic_diversity': value,
            'cross_cluster_bridging': value,
        }
    """
    metric_files = {
        'author_influx': 'author_influx.parquet',
        'citation_velocity': 'citation_velocity.parquet',
        'reference_vitality': 'reference_vitality.parquet',
        'topic_diversity': 'topic_diversity.parquet',
        'cross_cluster_bridging': 'cross_cluster_bridging.parquet',
    }

    global_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)

    for metric_name, filename in metric_files.items():
        filepath = metrics_dir / 'global' / filename
        if not filepath.exists():
            LOG.warning("Metric file not found: %s", filepath)
            continue

        df = pd.read_parquet(filepath)
        for _, row in df.iterrows():
            quarter = str(row['quarter'])
            value = float(row.get('value', 0.0))
            # Handle NaN values
            if pd.isna(value):
                value = 0.0
            global_metrics[quarter][metric_name] = value

    LOG.info("Loaded global metrics for %d quarters", len(global_metrics))
    return global_metrics


def compute_context_features(
    timeseries_df: pd.DataFrame,
    global_metrics: Dict[str, Dict[str, float]],
) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Compute context features using global metrics.

    For each metric, compute:
      - Quarter-over-quarter delta
      - Rolling averages (1, 2, 4 quarters)
      - Z-score per metric (across all quarters)
      - Max/min deviations in recent 4 quarters

    Note: We don't compute field-normalized ratios since we don't have
    lineage-level metric values yet (only lineage-level features).
    """
    context_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    field_metrics_df = pd.DataFrame()

    if not global_metrics:
        LOG.warning("No global metrics loaded; context features will be empty")
        return context_features

    # Prepare metric time series for z-score computation
    metric_names = ['author_influx', 'citation_velocity', 'reference_vitality',
                    'topic_diversity', 'cross_cluster_bridging']

    # Build quarter-sorted metric series for each metric
    quarters_sorted = sorted(global_metrics.keys(), key=quarter_key)
    metric_series: Dict[str, List[float]] = {name: [] for name in metric_names}

    for quarter in quarters_sorted:
        for metric_name in metric_names:
            value = global_metrics[quarter].get(metric_name, 0.0)
            metric_series[metric_name].append(value)

    # Compute mean and std for z-scores
    metric_stats: Dict[str, Tuple[float, float]] = {}
    for metric_name, values in metric_series.items():
        arr = np.array(values)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        metric_stats[metric_name] = (mean_val, std_val if std_val > 0 else 1.0)

    # For each lineage-quarter, compute context features
    for row in timeseries_df.itertuples():
        key = (row.lineage_id, row.quarter)
        quarter = row.quarter

        if quarter not in global_metrics:
            # No global metrics for this quarter; skip or use zeros
            continue

        features: Dict[str, float] = {}

        # Get current quarter index
        try:
            q_idx = quarters_sorted.index(quarter)
        except ValueError:
            continue

        for metric_name in metric_names:
            current_value = global_metrics[quarter].get(metric_name, 0.0)

            # Z-score
            mean_val, std_val = metric_stats[metric_name]
            z_score = (current_value - mean_val) / std_val
            features[f"{metric_name}_z"] = z_score

            # Quarter-over-quarter delta
            if q_idx > 0:
                prev_quarter = quarters_sorted[q_idx - 1]
                prev_value = global_metrics[prev_quarter].get(metric_name, 0.0)
                delta = current_value - prev_value
            else:
                delta = 0.0
            features[f"{metric_name}_qoq_delta"] = delta

            # Rolling averages (1, 2, 4 quarters)
            # 1-quarter is just current value
            features[f"{metric_name}_roll_1q"] = current_value

            # 2-quarter average
            if q_idx >= 1:
                values_2q = [
                    global_metrics[quarters_sorted[i]].get(metric_name, 0.0)
                    for i in range(max(0, q_idx - 1), q_idx + 1)
                ]
                features[f"{metric_name}_roll_2q"] = np.mean(values_2q)
            else:
                features[f"{metric_name}_roll_2q"] = current_value

            # 4-quarter average
            if q_idx >= 3:
                values_4q = [
                    global_metrics[quarters_sorted[i]].get(metric_name, 0.0)
                    for i in range(max(0, q_idx - 3), q_idx + 1)
                ]
                features[f"{metric_name}_roll_4q"] = np.mean(values_4q)
            else:
                # Use available quarters
                values_4q = [
                    global_metrics[quarters_sorted[i]].get(metric_name, 0.0)
                    for i in range(0, q_idx + 1)
                ]
                features[f"{metric_name}_roll_4q"] = np.mean(values_4q) if values_4q else current_value

            # Max/min deviations in recent 4 quarters
            if q_idx >= 3:
                values_4q = [
                    global_metrics[quarters_sorted[i]].get(metric_name, 0.0)
                    for i in range(max(0, q_idx - 3), q_idx + 1)
                ]
                max_dev = max(values_4q) - current_value
                min_dev = current_value - min(values_4q)
            else:
                values_4q = [
                    global_metrics[quarters_sorted[i]].get(metric_name, 0.0)
                    for i in range(0, q_idx + 1)
                ]
                max_dev = max(values_4q) - current_value if values_4q else 0.0
                min_dev = current_value - min(values_4q) if values_4q else 0.0

            features[f"{metric_name}_max_dev_4q"] = max_dev
            features[f"{metric_name}_min_dev_4q"] = min_dev

        context_features[key] = features

    LOG.info("Computed context features for %d lineage-quarter pairs", len(context_features))
    return context_features


# ---------------------------------------------------------------------------
# Feature computation


def build_lineage_quarter_papers(
    lineage_registry: Dict[int, Dict[str, Dict]],
    partitions_dir: Path,
    quarters_sorted: List[str],
    max_lineages: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, str], List[str]], Dict[str, int]]:
    """
    Determine new papers for each lineage and quarter.

    Returns:
        lineage_quarter_papers[(lineage_id, quarter)] = [work_ids]
        work_lineage[work_id] = lineage_id
    """
    lineage_ids = sorted(lineage_registry.keys())
    if max_lineages:
        lineage_ids = lineage_ids[:max_lineages]

    lineage_quarter_papers: Dict[Tuple[int, str], List[str]] = defaultdict(list)
    work_lineage: Dict[str, int] = {}
    seen_papers: Dict[int, set] = defaultdict(set)

    for quarter in quarters_sorted:
        partition_path = partitions_dir / f"part_{quarter}.json"
        if not partition_path.exists():
            continue
        inverted = load_partition(partition_path)

        for lineage_id in lineage_ids:
            quarter_map = lineage_registry.get(lineage_id, {})
            if quarter not in quarter_map:
                continue
            community_map = quarter_map[quarter]
            for comm_id in community_map.keys():
                comm_int = int(comm_id)
                papers = inverted.get(comm_int, [])
                if not papers:
                    continue
                new_papers = [
                    pid for pid in papers if pid not in seen_papers[lineage_id]
                ]
                if not new_papers:
                    continue
                lineage_quarter_papers[(lineage_id, quarter)].extend(new_papers)
                for pid in new_papers:
                    seen_papers[lineage_id].add(pid)
                    work_lineage[pid] = lineage_id
    return lineage_quarter_papers, work_lineage


def load_references(
    raw_dir: Path,
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Parse raw records to extract references and publication year.
    Optionally cache results to speed up future runs.
    """
    if cache_path and cache_path.exists() and not force_refresh:
        LOG.info("Loading reference cache from %s", cache_path)
        cached = load_trusted_pickle(
            cache_path, description="reference cache",
        )
        return cached["references"], cached["pub_years"]

    LOG.info("Parsing raw OpenAlex records for references (this may take a while)...")
    references_by_work: Dict[str, List[str]] = {}
    pub_year_by_work: Dict[str, int] = {}

    start = time.perf_counter()
    for work_id, record in iter_raw_records(raw_dir):
        pub_year_by_work[work_id] = int(record.get("publication_year") or 0)
        refs = [
            ref.rsplit("/", 1)[-1]
            for ref in record.get("referenced_works", [])
            if isinstance(ref, str)
        ]
        references_by_work[work_id] = refs

    LOG.info(
        "Loaded references for %s works in %.1f s",
        f"{len(references_by_work):,}",
        time.perf_counter() - start,
    )

    if cache_path:
        save_trusted_pickle(
            {"references": references_by_work, "pub_years": pub_year_by_work},
            cache_path,
            description="reference cache",
        )

    return references_by_work, pub_year_by_work


def build_cited_by_map(references_by_work: Dict[str, List[str]]) -> Dict[str, set]:
    """
    Build inverse mapping: work_id -> set of works that cite it.
    """
    cited_by: Dict[str, set] = defaultdict(set)
    for citing_work, refs in references_by_work.items():
        for ref in refs:
            cited_by[ref].add(citing_work)
    return cited_by


def compute_dormancy_features(timeseries_df: pd.DataFrame) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Compute dormancy length and awakening intensity per lineage-quarter.
    """
    features: Dict[Tuple[int, str], Dict[str, float]] = {}
    for lineage_id, group in timeseries_df.groupby("lineage_id"):
        group = group.sort_values("quarter", key=lambda col: col.map(quarter_key))
        zero_streak = 0
        for row in group.itertuples():
            key = (row.lineage_id, row.quarter)
            awakening = float(row.new_works * max(zero_streak, 1)) if row.new_works > 0 else 0.0
            features[key] = {
                "dormancy_length": float(zero_streak),
                "awakening_intensity": awakening,
            }
            if row.new_works > 0:
                zero_streak = 0
            else:
                zero_streak += 1
    return features


def _compute_novelty_serial(
    extractor,
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    lineage_ids: Iterable[int],
) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Compute novelty metrics based on new technical terms per quarter.
    """
    novelty: Dict[Tuple[int, str], Dict[str, float]] = {}
    seen_terms: Dict[int, set] = defaultdict(set)
    text_cache: Dict[str, str] = {}

    for lineage_id in lineage_ids:
        quarters = sorted(
            {q for (lin, q) in lineage_quarter_papers.keys() if lin == lineage_id},
            key=quarter_key,
        )
        if not quarters:
            continue

        for quarter in quarters:
            key = (lineage_id, quarter)
            paper_ids = lineage_quarter_papers.get(key, [])
            if not paper_ids:
                novelty[key] = {"novel_terms": 0.0, "novelty_rate": 0.0}
                continue

            batch_ids = [pid for pid in paper_ids if pid not in text_cache]
            if batch_ids:
                texts = extractor.get_texts_batch(batch_ids)
                text_cache.update(texts)

            quarter_terms: set = set()
            for pid in paper_ids:
                quarter_terms.update(tokenize(text_cache.get(pid, "")))

            new_terms = quarter_terms - seen_terms[lineage_id]
            seen_terms[lineage_id].update(quarter_terms)

            total_terms = len(quarter_terms) or 1
            novelty[key] = {
                "novel_terms": float(len(new_terms)),
                "novelty_rate": float(len(new_terms) / total_terms),
            }
    return novelty


def compute_novelty(
    extractor,
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    lineage_ids: Iterable[int],
    n_workers: Optional[int] = None,
) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Dispatch novelty computation to serial or parallel implementation.
    """
    workers = 1 if n_workers is None else int(n_workers)
    if workers == 0:
        workers = max(1, mp.cpu_count() - 1)
    if workers <= 1:
        return _compute_novelty_serial(extractor, lineage_quarter_papers, lineage_ids)
    try:
        return compute_novelty_parallel(extractor, lineage_quarter_papers, lineage_ids, workers)
    except Exception:
        LOG.exception("Parallel novelty computation failed; falling back to serial execution.")
        return _compute_novelty_serial(extractor, lineage_quarter_papers, lineage_ids)


def compute_novelty_for_lineage(
    lineage_id: int,
    quarters: List[str],
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    text_data: Dict[str, str],
) -> List[Tuple[Tuple[int, str], Dict[str, float]]]:
    results: List[Tuple[Tuple[int, str], Dict[str, float]]] = []
    seen_terms: set = set()

    for quarter in quarters:
        key = (lineage_id, quarter)
        paper_ids = lineage_quarter_papers.get(key, [])

        if not paper_ids:
            results.append((key, {"novel_terms": 0.0, "novelty_rate": 0.0}))
            continue

        quarter_terms: set = set()
        for pid in paper_ids:
            text = text_data.get(pid, "")
            if text:
                quarter_terms.update(tokenize(text))

        new_terms = quarter_terms - seen_terms
        seen_terms.update(quarter_terms)

        total_terms = len(quarter_terms) or 1
        results.append((key, {
            "novel_terms": float(len(new_terms)),
            "novelty_rate": float(len(new_terms) / total_terms),
        }))

    return results


def compute_novelty_parallel(
    extractor,
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    lineage_ids: Iterable[int],
    n_workers: int,
) -> Dict[Tuple[int, str], Dict[str, float]]:
    workers = max(1, n_workers)
    if workers == 1:
        return _compute_novelty_serial(extractor, lineage_quarter_papers, lineage_ids)

    LOG.info("Using %d worker processes for novelty computation", workers)

    lineage_ids_list = list(lineage_ids)
    all_paper_ids: set = set()
    for papers in lineage_quarter_papers.values():
        all_paper_ids.update(papers)

    LOG.info("Loading texts for %d unique papers...", len(all_paper_ids))
    start_load = time.perf_counter()
    text_cache: Dict[str, str] = {}
    batch_size = 5000
    paper_ids_list = list(all_paper_ids)
    for i in range(0, len(paper_ids_list), batch_size):
        batch = paper_ids_list[i:i + batch_size]
        texts = extractor.get_texts_batch(batch)
        text_cache.update(texts)
        if (i // batch_size) % 20 == 0:
            LOG.debug("  Loaded %d / %d texts", len(text_cache), len(all_paper_ids))
    LOG.info("Text loading complete in %.1fs", time.perf_counter() - start_load)

    lineage_quarters: Dict[int, List[str]] = {}
    for lineage_id in lineage_ids_list:
        quarters = sorted(
            {q for (lin, q) in lineage_quarter_papers.keys() if lin == lineage_id},
            key=quarter_key,
        )
        if quarters:
            lineage_quarters[lineage_id] = quarters

    start_compute = time.perf_counter()
    LOG.info("Processing %d lineages in parallel...", len(lineage_quarters))

    args_list = [
        (lineage_id, lineage_quarters[lineage_id], lineage_quarter_papers, text_cache)
        for lineage_id in sorted(lineage_quarters.keys())
    ]

    novelty: Dict[Tuple[int, str], Dict[str, float]] = {}
    with mp.Pool(processes=workers) as pool:
        for lineage_results in pool.starmap(compute_novelty_for_lineage, args_list):
            for key, metrics in lineage_results:
                novelty[key] = metrics

    LOG.info(
        "Novelty computation complete in %.1fs (%d lineage-quarter pairs)",
        time.perf_counter() - start_compute,
        len(novelty),
    )
    return novelty


def compute_cross_domain_share(
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    work_lineage: Dict[str, int],
    references_by_work: Dict[str, List[str]],
) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Compute fraction of references from lineage-quarter papers that target other lineages.
    """
    results: Dict[Tuple[int, str], Dict[str, float]] = {}
    for key, papers in lineage_quarter_papers.items():
        cross_refs = same_refs = 0
        lineage_id, _ = key
        for pid in papers:
            refs = references_by_work.get(pid, [])
            for ref in refs:
                ref_lineage = work_lineage.get(ref)
                if ref_lineage is None:
                    continue
                if ref_lineage == lineage_id:
                    same_refs += 1
                else:
                    cross_refs += 1
        total = cross_refs + same_refs
        share = float(cross_refs / total) if total else 0.0
        results[key] = {
            "cross_domain_share": share,
            "cross_domain_refs": float(cross_refs),
            "within_lineage_refs": float(same_refs),
        }
    return results


def compute_cd_index(
    lineage_quarter_papers: Dict[Tuple[int, str], List[str]],
    references_by_work: Dict[str, List[str]],
    cited_by_map: Dict[str, set],
    pub_year_by_work: Dict[str, int],
) -> Dict[Tuple[int, str], Dict[str, float]]:
    """
    Compute Wu et al. style disruption index for each lineage-quarter.
    """
    results: Dict[Tuple[int, str], Dict[str, float]] = {}

    for key, papers in lineage_quarter_papers.items():
        cd_values: List[float] = []
        for pid in papers:
            pub_year = pub_year_by_work.get(pid, 0)
            refs = references_by_work.get(pid, [])
            refs_set = set(refs)

            future_citers = [
                citer for citer in cited_by_map.get(pid, set())
                if pub_year_by_work.get(citer, 0) > pub_year
            ]

            n_u = n_v = 0
            for citer in future_citers:
                citer_refs = references_by_work.get(citer, [])
                if set(citer_refs) & refs_set:
                    n_v += 1
                else:
                    n_u += 1

            n_w = 0
            candidate_ref_citers: set = set()
            for ref in refs:
                for citer in cited_by_map.get(ref, set()):
                    if citer == pid:
                        continue
                    if pub_year_by_work.get(citer, 0) <= pub_year:
                        continue
                    candidate_ref_citers.add(citer)
            for citer in candidate_ref_citers:
                citer_refs = references_by_work.get(citer, [])
                if pid not in citer_refs:
                    n_w += 1

            denom = n_u + n_v + n_w
            cd = (n_u - n_v) / denom if denom else 0.0
            cd_values.append(cd)

        if cd_values:
            results[key] = {
                "cd_index": float(np.mean(cd_values)),
                "cd_min": float(np.min(cd_values)),
                "cd_max": float(np.max(cd_values)),
                "n_papers_cd": float(len(cd_values)),
            }
        else:
            results[key] = {
                "cd_index": 0.0,
                "cd_min": 0.0,
                "cd_max": 0.0,
                "n_papers_cd": 0.0,
            }
    return results


# ---------------------------------------------------------------------------
# Lifecycle stage features (onset + maturation labels)
# ---------------------------------------------------------------------------


def build_lifecycle_features(
    onset_labels_path: Optional[str],
    maturation_labels_path: Optional[str],
    all_quarters_sorted: List[str],
) -> Dict[Tuple[int, str], Dict[str, object]]:
    """Compute lifecycle stage features from onset and maturation labels.

    For each (lineage_id, quarter) key, produces:
    - ``lifecycle_stage``: categorical (pre_onset, growth, post_maturation, never_grew)
    - ``lifecycle_pre_onset``, ``lifecycle_growth``, ``lifecycle_post_maturation``,
      ``lifecycle_never_grew``: one-hot encoding of lifecycle_stage
    - ``is_matured``: binary flag (1 if lineage has a maturation label)
    - ``quarters_since_maturation``: count of quarters after maturation (0 before)

    All features are leakage-safe (derived from trailing-only onset/maturation
    detection).

    Args:
        onset_labels_path: Path to onset_labels.csv, or None.
        maturation_labels_path: Path to maturation_labels.csv, or None.
        all_quarters_sorted: All quarter labels in chronological order.

    Returns:
        (lineage_id, quarter) -> feature dict.
    """
    # Build quarter index for distance computation
    quarter_index = {q: i for i, q in enumerate(all_quarters_sorted)}

    # Load onset labels: lineage_id -> onset_quarter
    onset_lookup: Dict[int, str] = {}
    if onset_labels_path:
        path = Path(onset_labels_path)
        if path.exists():
            odf = pd.read_csv(path)
            for _, row in odf.iterrows():
                if int(row.get("onset_detected", 0)) == 1:
                    lid = int(row["lineage_id"])
                    oq = str(row["onset_quarter"])
                    if oq:
                        onset_lookup[lid] = oq
            LOG.info("Loaded %d onset labels from %s", len(onset_lookup), path)
        else:
            LOG.warning("Onset labels file %s not found; skipping.", path)

    # Load maturation labels: lineage_id -> maturation_quarter
    mat_lookup: Dict[int, str] = {}
    if maturation_labels_path:
        path = Path(maturation_labels_path)
        if path.exists():
            mdf = pd.read_csv(path)
            for _, row in mdf.iterrows():
                if int(row.get("maturation_detected", 0)) == 1:
                    lid = int(row["lineage_id"])
                    mq = str(row["maturation_quarter"])
                    if mq:
                        mat_lookup[lid] = mq
            LOG.info("Loaded %d maturation labels from %s", len(mat_lookup), path)
        else:
            LOG.warning("Maturation labels file %s not found; skipping.", path)

    if not onset_lookup and not mat_lookup:
        return {}

    all_lineages = set(onset_lookup.keys()) | set(mat_lookup.keys())
    result: Dict[Tuple[int, str], Dict[str, object]] = {}

    for lid in all_lineages:
        onset_q = onset_lookup.get(lid)
        mat_q = mat_lookup.get(lid)
        onset_idx = quarter_index.get(onset_q, -1) if onset_q else -1
        mat_idx = quarter_index.get(mat_q, -1) if mat_q else -1
        has_onset = onset_idx >= 0
        has_mat = mat_idx >= 0

        for q in all_quarters_sorted:
            qi = quarter_index[q]

            # Determine lifecycle stage
            if not has_onset and not has_mat:
                stage = "never_grew"
            elif has_mat and qi >= mat_idx:
                stage = "post_maturation"
            elif has_onset and qi >= onset_idx:
                stage = "growth"
            else:
                stage = "pre_onset"

            # Quarters since maturation (0 before, count after)
            if has_mat and qi >= mat_idx:
                q_since = qi - mat_idx
            else:
                q_since = 0

            result[(lid, q)] = {
                "lifecycle_stage": stage,
                "lifecycle_pre_onset": 1 if stage == "pre_onset" else 0,
                "lifecycle_growth": 1 if stage == "growth" else 0,
                "lifecycle_post_maturation": 1 if stage == "post_maturation" else 0,
                "lifecycle_never_grew": 1 if stage == "never_grew" else 0,
                "is_matured": 1 if has_mat else 0,
                "quarters_since_maturation": q_since,
            }

    return result


# ---------------------------------------------------------------------------
# Pipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute lineage-level multi-signal features.")
    ap.add_argument("--registry", default="data/out/02_lineage_tracking/lineage_registry.json")
    ap.add_argument("--timeseries", default="data/out/02_lineage_tracking/lineage_timeseries.csv")
    ap.add_argument("--raw-dir", default="data/current_ingest/raw")
    ap.add_argument("--partitions-dir", default="data/out/cache_cum/partitions_cum")
    ap.add_argument("--reference-cache", default="data/out/cache_lineage/reference_data.pkl")
    ap.add_argument("--out", default="data/out/02_lineage_tracking/lineage_multisignal_features.csv")
    ap.add_argument("--max-lineages", type=int, default=None, help="Limit number of lineages (smoke tests).")
    ap.add_argument("--metrics-dir", default="data/out/metrics", help="Directory containing global metrics parquet files.")
    ap.add_argument("--enable-context-features", action="store_true", help="Enable context features from global metrics.")
    ap.add_argument("--field-metrics", default="data/out/04_front_aggregation/field_metrics.parquet", help="Field metrics file (parquet or CSV).")
    ap.add_argument("--disable-field-metrics", action="store_true", help="Disable field-level feature integration.")
    ap.add_argument(
        "--n-workers",
        type=int,
        default=12,
        help="Worker processes for novelty computation (set <=1 for serial execution).",
    )
    ap.add_argument(
        "--milestones",
        default="data/out/experiments/stage0_tight_mapping/milestone_lineage_mapping_tight.csv",
        help="Milestone-lineage mapping used for proximity metrics (default: %(default)s).",
    )
    ap.add_argument(
        "--enable-milestone-proximity",
        action="store_true",
        help="Compute quarters-since/quarters-until milestone features per lineage.",
    )
    ap.add_argument(
        "--onset-labels",
        default=None,
        help="Onset labels CSV (from label_inflection_points.py --mode onset). "
             "Used for lifecycle stage features.",
    )
    ap.add_argument(
        "--maturation-labels",
        default=None,
        help="Maturation labels CSV (from label_inflection_points.py --mode maturation). "
             "Used for lifecycle stage and maturation proximity features.",
    )
    ap.add_argument(
        "--convergence-features",
        default=None,
        help="Convergence features CSV (from compute_convergence_features.py). "
             "Merged on (lineage_id, quarter) to add conv_* columns.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--force-cache-refresh", action="store_true", help="Recompute reference cache even if present.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    registry_path = Path(args.registry)
    timeseries_path = Path(args.timeseries)
    raw_dir = Path(args.raw_dir)
    partitions_dir = Path(args.partitions_dir)
    cache_path = Path(args.reference_cache) if args.reference_cache else None
    out_path = Path(args.out)

    start_total = time.perf_counter()

    LOG.info("Loading LineageTextStore (registry + abstract extractor)...")
    store = LineageTextStore(
        registry_path=registry_path,
        raw_dir=raw_dir,
        partitions_dir=partitions_dir,
        graphs_dir=None,
        verbose=False,
    )

    lineage_registry = {
        int(lin_id): quarters for lin_id, quarters in store.registry_by_lineage.items()
    }
    lineage_ids_full = sorted(lineage_registry.keys())
    LOG.info("Total lineages available: %d", len(lineage_ids_full))
    if args.max_lineages:
        LOG.info("Limiting to first %d lineages for this run (smoke test)", args.max_lineages)

    LOG.info("Loading timeseries data...")
    timeseries_df = pd.read_csv(timeseries_path)
    timeseries_df["quarter"] = timeseries_df["quarter"].astype(str)
    if args.max_lineages:
        keep_ids = set(lineage_ids_full[: args.max_lineages])
        timeseries_df = timeseries_df[timeseries_df["lineage_id"].isin(keep_ids)]
    quarters_sorted = sorted(timeseries_df["quarter"].unique(), key=quarter_key)

    LOG.info("Computing growth derivatives and logistic summaries...")
    growth_features = compute_growth_features(timeseries_df)

    milestone_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    if args.enable_milestone_proximity:
        LOG.info("Computing milestone proximity metrics from %s", args.milestones)
        milestone_features = compute_milestone_proximity(timeseries_df, Path(args.milestones))

    # Load global metrics and compute context features if enabled
    context_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    if args.enable_context_features:
        LOG.info("Loading global metrics from %s...", args.metrics_dir)
        metrics_dir = Path(args.metrics_dir)
        global_metrics = load_global_metrics(metrics_dir)
        if global_metrics:
            LOG.info("Computing context features...")
            context_features = compute_context_features(timeseries_df, global_metrics)
        else:
            LOG.warning("No global metrics loaded; context features will be empty")

    field_metrics_df = pd.DataFrame()
    if not args.disable_field_metrics:
        LOG.info("Loading field metrics from %s...", args.field_metrics)
        field_metrics_df = load_field_metrics(Path(args.field_metrics))

    LOG.info("Building lineage-quarter paper lists...")
    lineage_quarter_papers, work_lineage = build_lineage_quarter_papers(
        lineage_registry,
        partitions_dir,
        quarters_sorted,
        max_lineages=args.max_lineages,
    )
    LOG.info(
        "Collected %s lineage-quarter entries, %s unique works",
        f"{len(lineage_quarter_papers):,}",
        f"{len(work_lineage):,}",
    )

    references_by_work, pub_year_by_work = load_references(
        raw_dir,
        cache_path=cache_path,
        force_refresh=args.force_cache_refresh,
    )
    cited_by_map = build_cited_by_map(references_by_work)

    LOG.info("Computing dormancy features...")
    dormancy_features = compute_dormancy_features(timeseries_df)

    LOG.info("Computing novelty features...")
    lineage_ids_for_novelty = sorted({lin for lin, _ in lineage_quarter_papers.keys()})
    novelty_features = compute_novelty(
        store.extractor,
        lineage_quarter_papers,
        lineage_ids_for_novelty,
        n_workers=args.n_workers,
    )

    LOG.info("Precomputing per-work citation statistics...")
    work_cross_stats, work_cd_stats = compute_work_metrics(
        work_lineage,
        references_by_work,
        cited_by_map,
        pub_year_by_work,
        n_workers=args.n_workers or 1,
    )

    LOG.info("Aggregating cross-domain share per lineage-quarter...")
    cross_domain_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    cd_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    for key, papers in lineage_quarter_papers.items():
        lineage_id, _ = key

        cross_refs = same_refs = 0
        cd_vals: List[float] = []
        for pid in papers:
            cross, same = work_cross_stats.get(pid, (0, 0))
            cross_refs += cross
            same_refs += same
            cd_val, _ = work_cd_stats.get(pid, (0.0, 0.0))
            cd_vals.append(cd_val)

        total = cross_refs + same_refs
        cross_domain_features[key] = {
            "cross_domain_share": float(cross_refs / total) if total else 0.0,
            "cross_domain_refs": float(cross_refs),
            "within_lineage_refs": float(same_refs),
        }

        if cd_vals:
            cd_features[key] = {
                "cd_index": float(np.mean(cd_vals)),
                "cd_min": float(np.min(cd_vals)),
                "cd_max": float(np.max(cd_vals)),
                "n_papers_cd": float(len(cd_vals)),
            }
        else:
            cd_features[key] = {
                "cd_index": 0.0,
                "cd_min": 0.0,
                "cd_max": 0.0,
                "n_papers_cd": 0.0,
            }

    # Lifecycle stage features (from onset + maturation labels)
    lifecycle_features: Dict[Tuple[int, str], Dict[str, object]] = {}
    if args.onset_labels or args.maturation_labels:
        all_quarters_sorted = sorted(
            timeseries_df["quarter"].unique(), key=quarter_to_int,
        )
        lifecycle_features = build_lifecycle_features(
            args.onset_labels, args.maturation_labels, all_quarters_sorted,
        )
        if lifecycle_features:
            LOG.info("Computed lifecycle features for %d entries", len(lifecycle_features))

    # Convergence features (from compute_convergence_features.py)
    convergence_features: Dict[Tuple[int, str], Dict[str, float]] = {}
    if args.convergence_features:
        conv_path = Path(args.convergence_features)
        if conv_path.exists():
            LOG.info("Loading convergence features from %s", conv_path)
            conv_df = pd.read_csv(conv_path)
            conv_cols = [c for c in conv_df.columns if c.startswith("conv_")]
            for _row in conv_df.itertuples(index=False):
                _key = (int(_row.lineage_id), str(_row.quarter))
                convergence_features[_key] = {
                    c: float(getattr(_row, c, 0.0)) for c in conv_cols
                }
            LOG.info(
                "Loaded convergence features: %d entries, %d columns",
                len(convergence_features), len(conv_cols),
            )
        else:
            LOG.warning("Convergence features file not found: %s", conv_path)

    LOG.info("Assembling feature table...")
    records = []
    for row in timeseries_df.itertuples():
        key = (row.lineage_id, row.quarter)
        record = {
            "lineage_id": int(row.lineage_id),
            "quarter": row.quarter,
            "new_works": float(getattr(row, "new_works", 0)),
        }
        q_label = str(row.quarter)
        try:
            q_num = int(q_label[-1])
        except Exception:
            q_num = (quarter_to_int(q_label) % 4) + 1
        for q in range(1, 5):
            record[f"is_quarter_{q}"] = 1 if q == q_num else 0
        q_label = str(row.quarter)
        q_num = int(q_label[-1]) if isinstance(q_label, str) and len(q_label) >= 2 else (quarter_to_int(q_label) % 4 + 1)
        for q in range(1, 5):
            record[f"is_quarter_{q}"] = 1 if q == q_num else 0
        record.update(novelty_features.get(key, {"novel_terms": 0.0, "novelty_rate": 0.0}))
        record.update(cross_domain_features.get(key, {
            "cross_domain_share": 0.0,
            "cross_domain_refs": 0.0,
            "within_lineage_refs": 0.0,
        }))
        record.update(cd_features.get(key, {
            "cd_index": 0.0,
            "cd_min": 0.0,
            "cd_max": 0.0,
            "n_papers_cd": 0.0,
        }))
        record.update(dormancy_features.get(key, {
            "dormancy_length": 0.0,
            "awakening_intensity": 0.0,
        }))
        record["n_new_papers"] = float(len(lineage_quarter_papers.get(key, [])))
        record.update(growth_features.get(key, {
            "cumulative_works": 0.0,
            "growth_rate_diff": 0.0,
            "growth_acceleration": 0.0,
            "new_works_roll_mean_2q": 0.0,
            "new_works_roll_mean_4q": 0.0,
            "new_works_roll_std_4q": 0.0,
            "logistic_carrying_capacity": 0.0,
            "logistic_growth_rate": 0.0,
            "logistic_midpoint_idx": np.nan,
            "logistic_midpoint_quarter": None,
            "logistic_fit_r2": 0.0,
        }))
        if milestone_features:
            record.update(milestone_features.get(key, {
                "quarters_since_last_milestone": float("nan"),
                "last_milestone_id": None,
            }))
        # Add context features if enabled
        if context_features:
            record.update(context_features.get(key, {}))
        # Add lifecycle stage features if labels provided
        if lifecycle_features:
            record.update(lifecycle_features.get(key, {
                "lifecycle_stage": "unknown",
                "lifecycle_pre_onset": 0,
                "lifecycle_growth": 0,
                "lifecycle_post_maturation": 0,
                "lifecycle_never_grew": 0,
                "is_matured": 0,
                "quarters_since_maturation": 0,
            }))
        # Add convergence features if provided
        if convergence_features:
            record.update(convergence_features.get(key, {}))
        records.append(record)

    features_df = pd.DataFrame.from_records(records)
    if features_df.empty:
        LOG.warning("Feature table is empty! Check filters and inputs.")
    else:
        features_df.sort_values(["lineage_id", "quarter"], key=lambda col: col.map(quarter_key), inplace=True)

    features_df = merge_field_metrics(features_df, field_metrics_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(out_path, index=False)
    LOG.info("Feature table written to %s", out_path)

    LOG.info("Feature summary (head):\n%s", features_df.head().to_string(index=False))
    LOG.info("Feature summary (describe):\n%s", features_df.describe().to_string())

    LOG.info("Total runtime: %.1f s", time.perf_counter() - start_total)


def _init_work_metrics_pool(
    references_by_work: Dict[str, List[str]],
    work_lineage: Dict[str, int],
    cited_by_map: Dict[str, set],
    pub_year_by_work: Dict[str, int],
) -> None:
    global _MP_REFERENCES_BY_WORK, _MP_WORK_LINEAGE, _MP_CITED_BY_MAP, _MP_PUB_YEAR_BY_WORK
    _MP_REFERENCES_BY_WORK = references_by_work
    _MP_WORK_LINEAGE = work_lineage
    _MP_CITED_BY_MAP = cited_by_map
    _MP_PUB_YEAR_BY_WORK = pub_year_by_work


def _compute_work_metrics_single(pid: str) -> Optional[Tuple[str, Tuple[int, int], Tuple[float, float]]]:
    lineage_id = _MP_WORK_LINEAGE.get(pid)
    if lineage_id is None:
        return None

    refs = _MP_REFERENCES_BY_WORK.get(pid, [])
    cross = same = 0
    for ref in refs:
        ref_lineage = _MP_WORK_LINEAGE.get(ref)
        if ref_lineage is None:
            continue
        if ref_lineage == lineage_id:
            same += 1
        else:
            cross += 1

    pub_year = _MP_PUB_YEAR_BY_WORK.get(pid, 0)
    refs_set = set(refs)
    future_citers = [
        citer for citer in _MP_CITED_BY_MAP.get(pid, set())
        if _MP_PUB_YEAR_BY_WORK.get(citer, 0) > pub_year
    ]
    n_u = n_v = 0
    for citer in future_citers:
        citer_refs_set = set(_MP_REFERENCES_BY_WORK.get(citer, []))
        if citer_refs_set & refs_set:
            n_v += 1
        else:
            n_u += 1

    ref_future_citers: set = set()
    for ref in refs:
        ref_future_citers.update({
            citer for citer in _MP_CITED_BY_MAP.get(ref, set())
            if citer != pid and _MP_PUB_YEAR_BY_WORK.get(citer, 0) > pub_year
        })

    n_w = 0
    for citer in ref_future_citers:
        citer_refs_set = set(_MP_REFERENCES_BY_WORK.get(citer, []))
        if pid not in citer_refs_set:
            n_w += 1

    denom = n_u + n_v + n_w
    cd_val = (n_u - n_v) / denom if denom else 0.0
    return pid, (cross, same), (cd_val, float(denom))


def compute_work_metrics(
    work_lineage: Dict[str, int],
    references_by_work: Dict[str, List[str]],
    cited_by_map: Dict[str, set],
    pub_year_by_work: Dict[str, int],
    n_workers: int,
) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, Tuple[float, float]]]:
    if n_workers <= 1:
        work_cross_stats: Dict[str, Tuple[int, int]] = {}
        for pid, lineage_id in work_lineage.items():
            cross = same = 0
            for ref in references_by_work.get(pid, []):
                ref_lineage = work_lineage.get(ref)
                if ref_lineage is None:
                    continue
                if ref_lineage == lineage_id:
                    same += 1
                else:
                    cross += 1
            work_cross_stats[pid] = (cross, same)

        work_cd_stats: Dict[str, Tuple[float, float]] = {}
        for pid, refs in references_by_work.items():
            if pid not in work_lineage:
                continue
            pub_year = pub_year_by_work.get(pid, 0)
            refs_set = set(refs)
            future_citers = [
                citer for citer in cited_by_map.get(pid, set())
                if pub_year_by_work.get(citer, 0) > pub_year
            ]
            n_u = n_v = 0
            for citer in future_citers:
                citer_refs_set = set(references_by_work.get(citer, []))
                if citer_refs_set & refs_set:
                    n_v += 1
                else:
                    n_u += 1

            ref_future_citers: set = set()
            for ref in refs:
                ref_future_citers.update({
                    citer for citer in cited_by_map.get(ref, set())
                    if citer != pid and pub_year_by_work.get(citer, 0) > pub_year
                })

            n_w = 0
            for citer in ref_future_citers:
                citer_refs_set = set(references_by_work.get(citer, []))
                if pid not in citer_refs_set:
                    n_w += 1

            denom = n_u + n_v + n_w
            cd_val = (n_u - n_v) / denom if denom else 0.0
            work_cd_stats[pid] = (cd_val, float(denom))

        return work_cross_stats, work_cd_stats

    LOG.info("Parallelizing work-level citation metrics with %d workers...", n_workers)
    work_cross_stats: Dict[str, Tuple[int, int]] = {}
    work_cd_stats: Dict[str, Tuple[float, float]] = {}
    with mp.Pool(
        processes=n_workers,
        initializer=_init_work_metrics_pool,
        initargs=(references_by_work, work_lineage, cited_by_map, pub_year_by_work),
    ) as pool:
        for result in pool.imap_unordered(_compute_work_metrics_single, references_by_work.keys(), chunksize=500):
            if not result:
                continue
            pid, cross_same, cd_stats = result
            work_cross_stats[pid] = cross_same
            work_cd_stats[pid] = cd_stats
    return work_cross_stats, work_cd_stats


if __name__ == "__main__":
    mp.freeze_support()
    main()
