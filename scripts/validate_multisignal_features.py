#!/usr/bin/env python3
"""
Validate lineage multisignal features output.

Checks:
- Schema validation (expected columns present)
- Coverage statistics (% of lineage-quarters with metrics)
- Data quality (outliers, NaN patterns, value ranges)
- Context feature consistency

Outputs a JSON validation report with warnings/errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import numpy as np


# Expected core feature columns (from original implementation)
CORE_FEATURES = [
    "lineage_id",
    "quarter",
    "new_works",
    "novel_terms",
    "novelty_rate",
    "cross_domain_share",
    "cross_domain_refs",
    "within_lineage_refs",
    "cd_index",
    "cd_min",
    "cd_max",
    "n_papers_cd",
    "dormancy_length",
    "awakening_intensity",
    "n_new_papers",
]

# Expected context feature columns (from Task 2.1)
CONTEXT_FEATURES = []
for metric in ["author_influx", "citation_velocity", "reference_vitality", "topic_diversity", "cross_cluster_bridging"]:
    CONTEXT_FEATURES.extend([
        f"{metric}_z",
        f"{metric}_qoq_delta",
        f"{metric}_roll_1q",
        f"{metric}_roll_2q",
        f"{metric}_roll_4q",
        f"{metric}_max_dev_4q",
        f"{metric}_min_dev_4q",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multisignal features output")
    parser.add_argument(
        "--features-csv",
        default="data/out/02_lineage_tracking/lineage_multisignal_features.csv",
        type=Path,
        help="Path to features CSV file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write validation report JSON (optional)",
    )
    parser.add_argument(
        "--check-context-features",
        action="store_true",
        help="Enable validation of context features (from Task 2.1)",
    )
    parser.add_argument(
        "--min-coverage-threshold",
        type=float,
        default=0.5,
        help="Minimum acceptable coverage (0.0-1.0) for context features",
    )
    return parser.parse_args()


def validate_schema(df: pd.DataFrame, check_context: bool) -> Dict[str, Any]:
    """Validate that expected columns are present."""
    result = {
        "status": "pass",
        "errors": [],
        "warnings": [],
    }

    # Check core features
    missing_core = set(CORE_FEATURES) - set(df.columns)
    if missing_core:
        result["status"] = "fail"
        result["errors"].append(f"Missing core features: {sorted(missing_core)}")

    # Check context features if enabled
    if check_context:
        missing_context = set(CONTEXT_FEATURES) - set(df.columns)
        if missing_context:
            result["warnings"].append(
                f"Missing context features: {len(missing_context)} of {len(CONTEXT_FEATURES)}"
            )
            if len(missing_context) == len(CONTEXT_FEATURES):
                result["errors"].append(
                    "All context features missing - was --enable-context-features used?"
                )
                result["status"] = "fail"

    result["columns_found"] = len(df.columns)
    result["expected_core"] = len(CORE_FEATURES)
    result["expected_context"] = len(CONTEXT_FEATURES) if check_context else 0

    return result


def check_coverage(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    """Calculate coverage statistics for specified features."""
    result = {
        "total_rows": len(df),
        "total_lineages": df["lineage_id"].nunique(),
        "total_quarters": df["quarter"].nunique(),
        "coverage_by_feature": {},
        "overall_coverage": 0.0,
    }

    for feat in features:
        if feat not in df.columns:
            result["coverage_by_feature"][feat] = {
                "coverage": 0.0,
                "non_null": 0,
                "null": len(df),
                "status": "missing",
            }
            continue

        non_null = df[feat].notna().sum()
        null_count = df[feat].isna().sum()
        coverage = non_null / len(df) if len(df) > 0 else 0.0

        result["coverage_by_feature"][feat] = {
            "coverage": round(coverage, 4),
            "non_null": int(non_null),
            "null": int(null_count),
            "status": "ok" if coverage >= 0.5 else "low",
        }

    # Calculate overall coverage
    if features:
        coverages = [v["coverage"] for v in result["coverage_by_feature"].values() if v["status"] != "missing"]
        result["overall_coverage"] = round(np.mean(coverages), 4) if coverages else 0.0

    return result


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check for data quality issues."""
    result = {
        "status": "pass",
        "warnings": [],
        "stats": {},
    }

    # Check for duplicate lineage-quarter pairs
    duplicates = df.duplicated(subset=["lineage_id", "quarter"], keep=False).sum()
    if duplicates > 0:
        result["warnings"].append(f"Found {duplicates} duplicate lineage-quarter pairs")
        result["status"] = "warning"

    # Check numeric features for extreme outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == "lineage_id":
            continue

        values = df[col].dropna()
        if len(values) == 0:
            continue

        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        outlier_rate = outliers / len(values) if len(values) > 0 else 0.0

        if outlier_rate > 0.05:  # More than 5% outliers
            result["warnings"].append(
                f"{col}: {outlier_rate:.1%} outliers (>{int(outliers)} values)"
            )

        result["stats"][col] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "outliers": int(outliers),
            "outlier_rate": round(outlier_rate, 4),
        }

    return result


def check_context_feature_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    """Check consistency of context features."""
    result = {
        "status": "pass",
        "warnings": [],
        "z_score_checks": {},
        "rolling_checks": {},
    }

    # Check z-scores are in reasonable range (-5 to +5)
    z_score_cols = [col for col in df.columns if col.endswith("_z")]
    for col in z_score_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue

        extreme = ((values < -5) | (values > 5)).sum()
        extreme_rate = extreme / len(values) if len(values) > 0 else 0.0

        result["z_score_checks"][col] = {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "extreme_count": int(extreme),
            "extreme_rate": round(extreme_rate, 4),
        }

        if extreme_rate > 0.01:  # More than 1% extreme z-scores
            result["warnings"].append(
                f"{col}: {extreme_rate:.1%} values outside [-5, +5] range"
            )

    # Check rolling averages are consistent (roll_1q <= roll_2q <= roll_4q in general)
    for metric in ["author_influx", "citation_velocity", "reference_vitality", "topic_diversity", "cross_cluster_bridging"]:
        roll_1q = f"{metric}_roll_1q"
        roll_2q = f"{metric}_roll_2q"
        roll_4q = f"{metric}_roll_4q"

        if all(col in df.columns for col in [roll_1q, roll_2q, roll_4q]):
            # Check that rolling averages smooth out volatility
            vol_1q = df[roll_1q].std()
            vol_2q = df[roll_2q].std()
            vol_4q = df[roll_4q].std()

            result["rolling_checks"][metric] = {
                "vol_1q": round(float(vol_1q), 4),
                "vol_2q": round(float(vol_2q), 4),
                "vol_4q": round(float(vol_4q), 4),
                "smoothing_ok": vol_1q >= vol_2q >= vol_4q,
            }

            if not (vol_1q >= vol_2q >= vol_4q):
                result["warnings"].append(
                    f"{metric}: Rolling averages don't show expected smoothing pattern"
                )

    if result["warnings"]:
        result["status"] = "warning"

    return result


def main() -> None:
    args = parse_args()

    # Check file exists
    if not args.features_csv.exists():
        print(f"ERROR: Features file not found: {args.features_csv}")
        sys.exit(1)

    print(f"Loading features from: {args.features_csv}")
    df = pd.read_csv(args.features_csv)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Lineages: {df['lineage_id'].nunique()}, Quarters: {df['quarter'].nunique()}")

    # Run validation checks
    validation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(args.features_csv),
        "summary": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "total_lineages": int(df["lineage_id"].nunique()),
            "total_quarters": int(df["quarter"].nunique()),
        },
        "checks": {},
    }

    print("\n" + "=" * 70)
    print("Schema Validation")
    print("=" * 70)
    schema_result = validate_schema(df, args.check_context_features)
    validation_report["checks"]["schema"] = schema_result

    if schema_result["status"] == "pass":
        print("[OK] Schema validation passed")
    else:
        print("[FAIL] Schema validation failed")

    for error in schema_result.get("errors", []):
        print(f"  ERROR: {error}")
    for warning in schema_result.get("warnings", []):
        print(f"  WARNING: {warning}")

    print("\n" + "=" * 70)
    print("Coverage Analysis - Core Features")
    print("=" * 70)
    core_coverage = check_coverage(df, CORE_FEATURES[2:])  # Skip lineage_id, quarter
    validation_report["checks"]["core_coverage"] = core_coverage

    print(f"Overall coverage: {core_coverage['overall_coverage']:.1%}")

    # Show features with low coverage
    low_coverage_core = [
        feat for feat, stats in core_coverage["coverage_by_feature"].items()
        if stats["coverage"] < 0.9 and stats["status"] != "missing"
    ]
    if low_coverage_core:
        print(f"\nFeatures with <90% coverage:")
        for feat in low_coverage_core:
            stats = core_coverage["coverage_by_feature"][feat]
            print(f"  {feat}: {stats['coverage']:.1%}")

    if args.check_context_features:
        print("\n" + "=" * 70)
        print("Coverage Analysis - Context Features")
        print("=" * 70)
        context_coverage = check_coverage(df, CONTEXT_FEATURES)
        validation_report["checks"]["context_coverage"] = context_coverage

        print(f"Overall coverage: {context_coverage['overall_coverage']:.1%}")

        if context_coverage["overall_coverage"] < args.min_coverage_threshold:
            print(f"WARNING: Context feature coverage ({context_coverage['overall_coverage']:.1%}) "
                  f"below threshold ({args.min_coverage_threshold:.1%})")

        # Show context features by metric
        for metric in ["author_influx", "citation_velocity", "reference_vitality", "topic_diversity", "cross_cluster_bridging"]:
            metric_features = [f for f in CONTEXT_FEATURES if f.startswith(metric)]
            coverages = [
                context_coverage["coverage_by_feature"][f]["coverage"]
                for f in metric_features
                if f in context_coverage["coverage_by_feature"]
            ]
            if coverages:
                avg_cov = np.mean(coverages)
                print(f"  {metric}: {avg_cov:.1%}")

    print("\n" + "=" * 70)
    print("Data Quality Checks")
    print("=" * 70)
    quality_result = check_data_quality(df)
    validation_report["checks"]["data_quality"] = quality_result

    if quality_result["status"] == "pass":
        print("[OK] Data quality checks passed")
    elif quality_result["status"] == "warning":
        print("[WARNING] Data quality issues detected")

    for warning in quality_result.get("warnings", []):
        print(f"  WARNING: {warning}")

    if args.check_context_features:
        print("\n" + "=" * 70)
        print("Context Feature Consistency")
        print("=" * 70)
        consistency_result = check_context_feature_consistency(df)
        validation_report["checks"]["context_consistency"] = consistency_result

        if consistency_result["status"] == "pass":
            print("[OK] Context feature consistency checks passed")
        else:
            print("[WARNING] Context feature consistency issues")

        for warning in consistency_result.get("warnings", []):
            print(f"  WARNING: {warning}")

    # Final summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)

    all_checks_passed = all(
        check.get("status") in ["pass", "ok", None]
        for check in validation_report["checks"].values()
    )

    if all_checks_passed:
        print("[OK] All validation checks passed")
        validation_report["overall_status"] = "pass"
    else:
        print("[WARNING] Some validation checks failed or have warnings")
        validation_report["overall_status"] = "warning"

    # Write JSON report if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(validation_report, f, indent=2)
        print(f"\nValidation report written to: {args.output_json}")

    # Exit with non-zero if validation failed
    if validation_report["overall_status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
