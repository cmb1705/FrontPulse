#!/usr/bin/env python3
"""Run BOCPD changepoint detection on front-level time series.

Reads a front-level series CSV conforming to the front-level series contract
(front_id, quarter, new_works as minimum columns), runs Bayesian Online
Changepoint Detection per front, and outputs a CSV with per-quarter
changepoint probabilities and binary alerts.

Usage:
    python scripts/run_bocpd_detector.py
    python scripts/run_bocpd_detector.py --input data/out/04_front_aggregation/front_onset_series.csv
    python scripts/run_bocpd_detector.py --hazard-rate 0.04 --threshold 0.3 --detection-window 4

Output columns: front_id, quarter, bocpd_changepoint_prob, bocpd_alert,
bocpd_map_run_length.  When --merge is set, the BOCPD columns are joined onto
the input series for a single enriched output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.bocpd import BOCPDConfig, run_bocpd_on_fronts  # noqa: E402
from src.domain_registry import add_domain_args, resolve_script_paths  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run BOCPD changepoint detection on front-level series.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    parser.add_argument(
        "--input",
        default=None,
        help="Path to front-level series CSV",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for BOCPD results CSV",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge BOCPD columns onto input series instead of standalone output",
    )

    # BOCPD parameters
    parser.add_argument(
        "--alpha0",
        type=float,
        default=1.0,
        help="Gamma shape prior (default: %(default)s)",
    )
    parser.add_argument(
        "--beta0",
        type=float,
        default=0.1,
        help="Gamma rate prior (default: %(default)s)",
    )
    parser.add_argument(
        "--hazard-rate",
        type=float,
        default=1 / 50,
        help="Constant hazard rate, P(changepoint) per quarter (default: %(default)s)",
    )
    parser.add_argument(
        "--max-run-length",
        type=int,
        default=40,
        help="Maximum run length for truncation (default: %(default)s)",
    )
    parser.add_argument(
        "--detection-window",
        type=int,
        default=3,
        help="Quarters to aggregate for changepoint prob (default: %(default)s)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Changepoint probability threshold for alerts (default: %(default)s)",
    )

    # Output control
    parser.add_argument(
        "--config-out",
        default=None,
        help="Path to write BOCPD config as JSON (for reproducibility)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    add_domain_args(parser)
    return parser.parse_args()


def load_front_series(path: str) -> pd.DataFrame:
    """Load and validate front-level series CSV.

    Args:
        path: Path to the front-level series CSV.

    Returns:
        DataFrame with at least (front_id, quarter, new_works) columns,
        sorted by (front_id, quarter).

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    df = pd.read_csv(filepath)
    required = {"front_id", "quarter", "new_works"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_values(["front_id", "quarter"]).reset_index(drop=True)
    logger.info(
        "Loaded %d records for %d fronts from %s",
        len(df),
        df["front_id"].nunique(),
        filepath,
    )
    return df


def main() -> None:
    """Run BOCPD detector on front-level series."""
    args = parse_args()

    paths = resolve_script_paths(args, _REPO)
    if args.input is None:
        args.input = str(paths.front_aggregation / "front_onset_series.csv") if paths else "data/out/04_front_aggregation/front_onset_series.csv"
    if args.output is None:
        args.output = str(paths.front_aggregation / "bocpd_results.csv") if paths else "data/out/04_front_aggregation/bocpd_results.csv"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Build config from CLI args
    config = BOCPDConfig(
        alpha0=args.alpha0,
        beta0=args.beta0,
        hazard_rate=args.hazard_rate,
        max_run_length=args.max_run_length,
        detection_window=args.detection_window,
        threshold=args.threshold,
    )
    logger.info("BOCPD config: %s", config)

    # Load input
    series_df = load_front_series(args.input)

    # Run BOCPD
    bocpd_df = run_bocpd_on_fronts(series_df, config)

    # Optionally merge onto input
    if args.merge:
        out_df = series_df.merge(bocpd_df, on=["front_id", "quarter"], how="left")
    else:
        out_df = bocpd_df

    # Ensure output directory exists
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_path, index=False)

    # Summary statistics
    n_fronts = bocpd_df["front_id"].nunique()
    n_alerts = int(bocpd_df["bocpd_alert"].sum())
    fronts_with_alerts = bocpd_df.loc[
        bocpd_df["bocpd_alert"] == 1, "front_id"
    ].nunique()

    logger.info(
        "Wrote %d records (%d fronts) to %s",
        len(out_df),
        n_fronts,
        out_path,
    )
    logger.info(
        "Alerts: %d total across %d fronts (threshold=%.3f)",
        n_alerts,
        fronts_with_alerts,
        config.threshold,
    )

    # Save config for reproducibility
    if args.config_out:
        config_path = Path(args.config_out)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_dict = {
            "alpha0": config.alpha0,
            "beta0": config.beta0,
            "hazard_rate": config.hazard_rate,
            "max_run_length": config.max_run_length,
            "detection_window": config.detection_window,
            "threshold": config.threshold,
        }
        config_path.write_text(json.dumps(config_dict, indent=2))
        logger.info("Config written to %s", config_path)


if __name__ == "__main__":
    main()
