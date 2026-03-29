#!/usr/bin/env python3
from __future__ import annotations

"""
Tripwire backtest + validation helper.

Usage:
  python scripts/evaluate_tripwire.py `
      --counts data/out/04_front_aggregation/front_timeseries_long.csv `
      --metrics data/out/04_front_aggregation/front_metrics.csv `
      --outdir data/out/06_validation/current `
      --start 2012-Q1 --end 2025-Q4

Outputs (under --outdir):
  - tripwire_alerts.csv
  - validation_results.csv
  - validation_report.md
  - validation_dashboard.png (comprehensive visualization)
"""

import argparse  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from _path_bootstrap import ensure_repo_imports  # noqa: E402

repo_root = ensure_repo_imports()

import pandas as pd  # noqa: E402


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


def load_counts(path: Path) -> pd.DataFrame:
    """
    Load FRONT-LEVEL aggregated counts (after lineage->front aggregation).

    Expected columns:
    - quarter: Time period (e.g., 2010Q1)
    - front: Research front name (e.g., "inverted_architecture")
    - count: Aggregated publication count for that front in that quarter

    NOTE: This is POST-aggregation data. Lineage/community_id belongs to pre-aggregation.
    """
    df = pd.read_csv(path)
    quarter_col = _find_column(df, ["quarter", "period", "date"])
    front_col = _find_column(df, ["front"])
    count_col = _find_column(df, ["count", "pub_count", "observed"])

    if not quarter_col or not front_col or not count_col:
        raise ValueError(
            "Counts table is missing required columns for FRONT-LEVEL data. "
            f"Detected quarter_col={quarter_col}, front_col={front_col}, count_col={count_col}. "
            f"This script expects front-level aggregated data (after aggregate_lineages_to_fronts.py), "
            f"not lineage-level data. Column 'front' is research front name, NOT lineage_id/community_id."
        )

    df = df.rename(
        columns={
            quarter_col: "quarter",
            front_col: "community_id",  # Rename to community_id for PSCTripwireValidator compatibility
            count_col: "pub_count",
        }
    )
    if "boundary_citation_share" not in df:
        df["boundary_citation_share"] = 0.0

    df["quarter"] = df["quarter"].astype(str).str.replace(r"(?<=\d)Q", "-Q", regex=True)
    return df[["community_id", "quarter", "pub_count", "boundary_citation_share"]]


def _resolve_metrics_path(path: Path) -> Path | None:
    """
    Resolve metrics path, handling legacy front_* filenames renamed to lineage_*.
    """
    if path.exists():
        return path

    alternates: list[Path] = []

    # Support post-rename filenames (front_* -> lineage_*)
    if "front_metrics" in path.name:
        alternates.append(path.with_name(path.name.replace("front_", "lineage_")))

    # Handle root-level rename (front_metrics_cumulative.csv -> lineage_metrics.csv)
    if path.name == "front_metrics_cumulative.csv":
        alternates.append(path.with_name("lineage_metrics.csv"))

    for alt in alternates:
        if alt.exists():
            print(f"[INFO] Metrics file {path} not found; using renamed file {alt}.")
            return alt

    return None


def _placeholder_metadata(fallback_counts: pd.DataFrame) -> pd.DataFrame:
    """Build minimal metadata table from counts if external metrics are unavailable."""
    df = fallback_counts[["community_id", "quarter"]].copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df["vi_score"] = 0.0
    df["persistent_core_overlap"] = 0.0
    return df


def load_metrics(path: Path, fallback_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Load FRONT-LEVEL quality metrics (optional, can be generated from counts).

    Expected columns:
    - quarter: Time period
    - front: Research front name
    - vi_score: Variation of information (optional)
    - persistent_core_overlap: Core stability metric (optional)

    NOTE: This is POST-aggregation data at the front level, not lineage level.
    """
    resolved_path = _resolve_metrics_path(path)

    if resolved_path is None:
        print(f"[WARN] Metrics file {path} not found; building placeholder metadata from publication counts.")
        return _placeholder_metadata(fallback_counts)

    try:
        if resolved_path.suffix.lower() == ".parquet":
            df = pd.read_parquet(resolved_path)
        else:
            df = pd.read_csv(resolved_path)
    except Exception as exc:
        print(f"[WARN] Failed to load metrics file {resolved_path}: {exc}. Using placeholder metadata.")
        return _placeholder_metadata(fallback_counts)

    quarter_col = _find_column(df, ["quarter", "period", "date"])
    front_col = _find_column(df, ["front"])

    if not quarter_col or not front_col:
        print(
            "[WARN] Metrics table is missing required columns for front-level data "
            f"(quarter={quarter_col}, front={front_col}). Using placeholder metadata."
        )
        return _placeholder_metadata(fallback_counts)

    df = df.rename(
        columns={
            quarter_col: "quarter",
            front_col: "community_id",
        }
    )
    df["quarter"] = df["quarter"].astype(str).str.replace(r"(?<=\d)Q", "-Q", regex=True)
    if "vi_score" not in df.columns:
        df["vi_score"] = 0.0
    if "persistent_core_overlap" not in df.columns:
        df["persistent_core_overlap"] = 0.0
    return df


def load_validation_modules(module_dir: Path):
    """Load local-only PSC validation helpers from the private study directory."""
    if not module_dir.exists():
        raise FileNotFoundError(
            f"Validation helper directory not found at {module_dir}. "
            "PSC-specific validation helpers are stored locally under _local/psc/references."
        )

    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

    validator_module = importlib.import_module("psc_tripwire_validator")
    viz_module = importlib.import_module("psc_validation_viz")
    return validator_module.PSCTripwireValidator, viz_module


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate tripwire alerts against expert events.")
    ap.add_argument("--counts", default="data/out/04_front_aggregation/front_timeseries_delta_long.csv",
                    help="Front-level delta counts (new works per quarter) - REQUIRED for spike detection")
    ap.add_argument("--metrics", default="data/out/04_front_aggregation/front_metrics.csv")
    ap.add_argument("--milestones", required=True, help="Milestone CSV file for validation.")
    ap.add_argument(
        "--validator-dir",
        default=str(repo_root / "_local" / "psc" / "references"),
        help="Directory containing local-only PSC validation helper modules (default: %(default)s).",
    )
    ap.add_argument("--outdir", default="data/out/06_validation")
    ap.add_argument("--start", default=None, help="Start quarter for backtest (e.g., 2009-Q1)")
    ap.add_argument("--end", default=None, help="End quarter for backtest (e.g., 2025-Q4)")
    ap.add_argument("--history", type=int, default=8, help="History window for NB model (quarters)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pub_counts = load_counts(Path(args.counts))
    community_meta = load_metrics(Path(args.metrics), pub_counts)

    start_q = args.start or pub_counts["quarter"].min()
    end_q = args.end or pub_counts["quarter"].max()
    PSCTripwireValidator, viz = load_validation_modules(Path(args.validator_dir))

    print(f"[Milestone] Loading milestones from {args.milestones}")
    validator = PSCTripwireValidator(
        publication_counts=pub_counts,
        community_metadata=community_meta,
        milestone_csv=args.milestones
    )

    print(f"[Tripwire] Running rolling backtest {start_q} -> {end_q} (history={args.history}q)")
    alerts = validator.run_rolling_backtest(start_quarter=start_q, end_quarter=end_q, history_window=args.history)
    alerts.to_csv(outdir / "tripwire_alerts.csv", index=False)
    print(f"[Tripwire] Alerts written to {outdir / 'tripwire_alerts.csv'}")

    print("[Tripwire] Validating against expert event catalog…")
    results = validator.validate_against_events()
    results.to_csv(outdir / "validation_results.csv", index=False)
    print(f"[Tripwire] Validation results written to {outdir / 'validation_results.csv'}")

    report_path = outdir / "validation_report.md"
    validator.generate_validation_report(report_path)
    print(f"[Tripwire] Validation report written to {report_path}")

    # Comprehensive validation dashboard
    try:
        viz.plot_comprehensive_validation_dashboard(
            results.copy(),
            alerts.copy(),
            output_path=outdir / "validation_dashboard.png"
        )
        print(f"[Tripwire] Comprehensive validation dashboard written to {outdir / 'validation_dashboard.png'}")
    except Exception as exc:
        print(f"[Tripwire] WARNING: Failed to generate validation dashboard: {exc}")


if __name__ == "__main__":
    main()
