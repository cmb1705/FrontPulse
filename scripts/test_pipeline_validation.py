#!/usr/bin/env python3
"""
Pipeline Validation Suite (Task 5.2)

Regression tests covering:
1. Metric schema validation
2. Feature integration edge cases (missing data)
3. MSD performance expectations (allowable drift)

Usage:
    python scripts/test_pipeline_validation.py
    python scripts/test_pipeline_validation.py --metrics-dir data/out/metrics
    python scripts/test_pipeline_validation.py --check-performance --baseline-recall 0.85
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from _path_bootstrap import ensure_repo_imports

repo_root = ensure_repo_imports()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from metrics.common import (  # noqa: E402
    FRONT_METRIC_SCHEMA,
    GLOBAL_METRIC_SCHEMA,
    LINEAGE_METRIC_SCHEMA,
    load_manifest,
)

# ============================================================================
# Test 1: Metric Schema Validation
# ============================================================================

def validate_metric_schema(parquet_path: Path, expected_schema, metric_name: str, level: str) -> tuple[bool, list[str]]:  # noqa: ARG001
    """
    Validate that a parquet file conforms to expected schema.

    Returns:
        (is_valid, errors)
    """
    errors = []

    if not parquet_path.exists():
        errors.append(f"File not found: {parquet_path}")
        return False, errors

    try:
        # Read schema from parquet file
        table = pq.read_table(parquet_path)
        actual_schema = table.schema

        # Check each expected field
        for expected_field in expected_schema:
            field_name = expected_field.name
            expected_type = expected_field.type

            if field_name not in actual_schema.names:
                errors.append(f"Missing required field: {field_name}")
                continue

            actual_field = actual_schema.field(field_name)
            actual_type = actual_field.type

            # Allow compatible types (e.g., int32 vs int64)
            if not _types_compatible(expected_type, actual_type):
                errors.append(f"Field '{field_name}' type mismatch: expected {expected_type}, got {actual_type}")

        # Check for unexpected extra columns (warning only)
        extra_fields = set(actual_schema.names) - {f.name for f in expected_schema}
        if extra_fields:
            # Extra columns are OK for metrics (they may have auxiliary data)
            pass

        is_valid = len(errors) == 0
        return is_valid, errors

    except Exception as e:
        errors.append(f"Error reading parquet file: {e}")
        return False, errors


def _types_compatible(expected, actual) -> bool:
    """Check if two PyArrow types are compatible."""
    import pyarrow as pa

    # Exact match
    if expected == actual:
        return True

    # Integer compatibility
    if pa.types.is_integer(expected) and pa.types.is_integer(actual):
        return True

    # Float compatibility
    if pa.types.is_floating(expected) and pa.types.is_floating(actual):
        return True

    # String compatibility
    return bool(pa.types.is_string(expected) and pa.types.is_string(actual))


def test_metric_schemas(metrics_dir: Path) -> dict[str, Any]:
    """
    Test all metric outputs conform to standardized schemas.

    Returns:
        results dict with pass/fail status and errors
    """
    print("\n" + "=" * 70)
    print("TEST 1: METRIC SCHEMA VALIDATION")
    print("=" * 70)

    results = {
        "test_name": "metric_schema_validation",
        "passed": True,
        "total_metrics": 0,
        "passed_metrics": 0,
        "failed_metrics": 0,
        "errors": []
    }

    metric_names = [
        "author_influx",
        "citation_velocity",
        "reference_vitality",
        "topic_diversity",
        "cross_cluster_bridging"
    ]

    schema_map = {
        "global": GLOBAL_METRIC_SCHEMA,
        "front": FRONT_METRIC_SCHEMA,
        "lineage": LINEAGE_METRIC_SCHEMA,
    }

    for metric_name in metric_names:
        for level, expected_schema in schema_map.items():
            results["total_metrics"] += 1

            parquet_path = metrics_dir / level / f"{metric_name}.parquet"
            print(f"\n[TEST] {metric_name} ({level})")

            is_valid, errors = validate_metric_schema(
                parquet_path, expected_schema, metric_name, level
            )

            if is_valid:
                print("   [PASS] Schema valid")
                results["passed_metrics"] += 1
            else:
                print("   [FAIL] Schema validation failed")
                for error in errors:
                    print(f"      - {error}")
                results["passed"] = False
                results["failed_metrics"] += 1
                results["errors"].extend(errors)

    print("\n" + "-" * 70)
    print(f"Schema Validation Summary: {results['passed_metrics']}/{results['total_metrics']} passed")

    return results


# ============================================================================
# Test 2: Feature Integration Edge Cases
# ============================================================================

def test_missing_data_handling() -> dict[str, Any]:
    """
    Test that feature computation gracefully handles missing data.

    Edge cases:
    - Missing global metrics for some quarters
    - Sparse lineage data
    - Missing context features
    """
    print("\n" + "=" * 70)
    print("TEST 2: FEATURE INTEGRATION EDGE CASES")
    print("=" * 70)

    results = {
        "test_name": "feature_integration_edge_cases",
        "passed": True,
        "total_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "errors": []
    }

    # Import feature computation modules
    from multi_signal_detector import engineer_features, select_features

    # Test Case 1: Missing columns (should not crash)
    print("\n[TEST CASE 1] Missing optional columns")
    results["total_cases"] += 1

    try:
        df_minimal = pd.DataFrame({
            'lineage_id': [1, 2, 3],
            'quarter': ['2020Q1', '2020Q2', '2020Q3'],
            'new_works': [10, 15, 20],
            'novel_terms': [5, 8, 10],
            'novelty_rate': [0.5, 0.53, 0.5],
            'cross_domain_share': [0.1, 0.2, 0.15],
            'cross_domain_refs': [5, 10, 8],
            'within_lineage_refs': [20, 30, 25],
            'cd_index': [0.0, 0.1, 0.05],
            'cd_min': [-0.5, -0.3, -0.4],
            'cd_max': [0.5, 0.6, 0.55],
            'n_papers_cd': [10, 15, 12],
            'dormancy_length': [0, 1, 0],
            'awakening_intensity': [0.0, 0.5, 0.0],
            'n_new_papers': [5, 8, 10],
            'semantic_velocity': [0.1, 0.2, 0.15],
            'total_works': [100, 120, 140],
            'growth_rate': [0.05, 0.10, 0.08],
        })

        df_engineered = engineer_features(df_minimal)
        X, features = select_features(df_engineered)

        assert len(features) >= 20, "Should have at least 20 core features"
        print(f"   [PASS] Handled minimal feature set: {len(features)} features")
        results["passed_cases"] += 1

    except Exception as e:
        print(f"   [FAIL] {e}")
        results["passed"] = False
        results["failed_cases"] += 1
        results["errors"].append(f"Case 1: {e}")

    # Test Case 2: Mixed presence of context features (some lineages have, some don't)
    print("\n[TEST CASE 2] Mixed presence of context features")
    results["total_cases"] += 1

    try:
        df_mixed = df_minimal.copy()

        # Add context features for only some rows
        for metric in ['author_influx', 'citation_velocity']:
            df_mixed[f'{metric}_z'] = [0.5, np.nan, -0.3]
            df_mixed[f'{metric}_qoq_delta'] = [0.1, np.nan, -0.05]
            df_mixed[f'{metric}_roll_1q'] = [0.8, np.nan, 0.75]
            df_mixed[f'{metric}_roll_2q'] = [0.79, np.nan, 0.76]
            df_mixed[f'{metric}_roll_4q'] = [0.80, np.nan, 0.77]
            df_mixed[f'{metric}_max_dev_4q'] = [1.5, np.nan, 1.2]
            df_mixed[f'{metric}_min_dev_4q'] = [-1.0, np.nan, -0.8]

        df_engineered = engineer_features(df_mixed)
        X, features = select_features(df_engineered)

        # Should handle NaN gracefully
        assert X.isna().sum().sum() == 0, "NaNs should be filled"
        print(f"   [PASS] Handled mixed context features: {len(features)} features")
        results["passed_cases"] += 1

    except Exception as e:
        print(f"   [FAIL] {e}")
        results["passed"] = False
        results["failed_cases"] += 1
        results["errors"].append(f"Case 2: {e}")

    # Test Case 3: Extreme values
    print("\n[TEST CASE 3] Extreme values (inf, very large numbers)")
    results["total_cases"] += 1

    try:
        df_extreme = df_minimal.copy()
        df_extreme.loc[0, 'growth_rate'] = np.inf  # Extreme growth
        df_extreme.loc[1, 'semantic_velocity'] = -np.inf  # Extreme velocity
        df_extreme.loc[2, 'novelty_rate'] = 1e10  # Very large number

        df_engineered = engineer_features(df_extreme)
        X, features = select_features(df_engineered)

        # Should replace inf with 0 or clip
        assert not np.isinf(X.values).any(), "Inf values should be handled"
        print(f"   [PASS] Handled extreme values: {len(features)} features")
        results["passed_cases"] += 1

    except Exception as e:
        print(f"   [FAIL] {e}")
        results["passed"] = False
        results["failed_cases"] += 1
        results["errors"].append(f"Case 3: {e}")

    # Test Case 4: Empty DataFrame
    print("\n[TEST CASE 4] Empty DataFrame")
    results["total_cases"] += 1

    try:
        df_empty = pd.DataFrame(columns=df_minimal.columns)

        # Should not crash
        df_engineered = engineer_features(df_empty)
        X, features = select_features(df_engineered)

        assert len(X) == 0, "Should return empty result"
        print("   [PASS] Handled empty DataFrame")
        results["passed_cases"] += 1

    except Exception as e:
        print(f"   [FAIL] {e}")
        results["passed"] = False
        results["failed_cases"] += 1
        results["errors"].append(f"Case 4: {e}")

    print("\n" + "-" * 70)
    print(f"Edge Case Summary: {results['passed_cases']}/{results['total_cases']} passed")

    return results


# ============================================================================
# Test 3: MSD Performance Expectations
# ============================================================================

def test_msd_performance_drift(
    baseline_recall: float = 0.85,
    baseline_precision: float = 0.20,
    tolerance: float = 0.05
) -> dict[str, Any]:
    """
    Test that MSD performance hasn't drifted beyond acceptable thresholds.

    This is a regression test to catch unexpected performance degradation.

    Args:
        baseline_recall: Expected recall (from Phase 2 analysis)
        baseline_precision: Expected precision
        tolerance: Allowable drift (e.g., 0.05 = 5%)
    """
    print("\n" + "=" * 70)
    print("TEST 3: MSD PERFORMANCE REGRESSION CHECK")
    print("=" * 70)
    print(f"Baseline: Recall={baseline_recall:.3f}, Precision={baseline_precision:.3f}")
    print(f"Tolerance: ±{tolerance:.1%}")

    results = {
        "test_name": "msd_performance_drift",
        "passed": True,
        "baseline_recall": baseline_recall,
        "baseline_precision": baseline_precision,
        "tolerance": tolerance,
        "errors": []
    }

    # Check if tuning results exist
    tuning_dir = repo_root / "data" / "out" / "experiments" / "msd_tuning_smoke_test"
    best_config_path = tuning_dir / "best_config.json"

    if not best_config_path.exists():
        print(f"\n[SKIP] No tuning results found at {best_config_path}")
        print("   Run msd_meta_tune.py first to generate baseline results")
        results["passed"] = None  # Not failed, just skipped
        return results

    # Load tuning results
    with open(best_config_path) as f:
        best_config = json.load(f)

    actual_recall = best_config.get("mean_recall", 0.0)
    actual_precision = best_config.get("mean_precision", 0.0)

    print("\n[TEST] Current Performance")
    print(f"   Recall:    {actual_recall:.3f}")
    print(f"   Precision: {actual_precision:.3f}")

    # Check recall drift
    recall_drift = abs(actual_recall - baseline_recall)
    if recall_drift > tolerance:
        error_msg = f"Recall drift {recall_drift:.3f} exceeds tolerance {tolerance:.3f}"
        print(f"   [FAIL] {error_msg}")
        results["passed"] = False
        results["errors"].append(error_msg)
    else:
        print(f"   [PASS] Recall drift {recall_drift:.3f} within tolerance")

    # Check precision drift
    precision_drift = abs(actual_precision - baseline_precision)
    if precision_drift > tolerance:
        error_msg = f"Precision drift {precision_drift:.3f} exceeds tolerance {tolerance:.3f}"
        print(f"   [FAIL] {error_msg}")
        results["passed"] = False
        results["errors"].append(error_msg)
    else:
        print(f"   [PASS] Precision drift {precision_drift:.3f} within tolerance")

    results["actual_recall"] = actual_recall
    results["actual_precision"] = actual_precision
    results["recall_drift"] = recall_drift
    results["precision_drift"] = precision_drift

    print("\n" + "-" * 70)
    if results["passed"]:
        print("Performance Regression: [PASS] No significant drift detected")
    else:
        print("Performance Regression: [FAIL] Drift exceeds tolerance")

    return results


# ============================================================================
# Test 4: Manifest Integrity
# ============================================================================

def test_manifest_integrity(metrics_dir: Path) -> dict[str, Any]:
    """
    Test that manifest is well-formed and matches actual outputs.
    """
    print("\n" + "=" * 70)
    print("TEST 4: MANIFEST INTEGRITY")
    print("=" * 70)

    results = {
        "test_name": "manifest_integrity",
        "passed": True,
        "errors": []
    }

    manifest_path = metrics_dir / "manifest.json"

    if not manifest_path.exists():
        error_msg = f"Manifest not found: {manifest_path}"
        print(f"   [FAIL] {error_msg}")
        results["passed"] = False
        results["errors"].append(error_msg)
        return results

    try:
        manifest = load_manifest(manifest_path)

        # Check required top-level fields
        print("\n[TEST] Manifest structure")
        # Accept both "version" and "manifest_version"
        has_version = "version" in manifest or "manifest_version" in manifest
        if not has_version:
            error_msg = "Missing required field: version or manifest_version"
            print(f"   [FAIL] {error_msg}")
            results["passed"] = False
            results["errors"].append(error_msg)
        else:
            version_field = "version" if "version" in manifest else "manifest_version"
            print(f"   [OK] Field '{version_field}' present")

        required_fields = ["last_updated", "metrics"]
        for field in required_fields:
            if field not in manifest:
                error_msg = f"Missing required field: {field}"
                print(f"   [FAIL] {error_msg}")
                results["passed"] = False
                results["errors"].append(error_msg)
            else:
                print(f"   [OK] Field '{field}' present")

        # Check metrics section
        # Manifest keys metrics as "{name}_{level}", e.g., "author_influx_global"
        print("\n[TEST] Metric entries")
        metrics = manifest.get("metrics", {})
        expected_metrics = ["author_influx", "citation_velocity", "reference_vitality",
                           "topic_diversity", "cross_cluster_bridging"]

        for metric_name in expected_metrics:
            # Check if at least one level exists for this metric
            metric_keys = [k for k in metrics if k.startswith(metric_name)]
            if not metric_keys:
                error_msg = f"Missing metric entry: {metric_name}"
                print(f"   [FAIL] {error_msg}")
                results["passed"] = False
                results["errors"].append(error_msg)
            else:
                print(f"   [OK] Metric '{metric_name}' registered ({len(metric_keys)} level(s))")

                # Check first entry has required fields
                metric_entry = metrics[metric_keys[0]]
                required_metric_fields = ["description", "formula", "metric_name", "level"]
                for field in required_metric_fields:
                    if field not in metric_entry:
                        error_msg = f"Metric '{metric_keys[0]}' missing field: {field}"
                        print(f"      [WARN] {error_msg}")
                        # Warning only, not failure

        print("\n" + "-" * 70)
        if results["passed"]:
            print("Manifest Integrity: [PASS]")
        else:
            print("Manifest Integrity: [FAIL]")

    except Exception as e:
        error_msg = f"Error loading manifest: {e}"
        print(f"   [FAIL] {error_msg}")
        results["passed"] = False
        results["errors"].append(error_msg)

    return results


# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline validation suite (Task 5.2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--metrics-dir',
        type=Path,
        default=Path('data/out/metrics'),
        help='Path to metrics output directory (default: data/out/metrics)'
    )

    parser.add_argument(
        '--check-performance',
        action='store_true',
        help='Enable performance regression testing (requires tuning results)'
    )

    parser.add_argument(
        '--baseline-recall',
        type=float,
        default=0.85,
        help='Expected baseline recall (default: 0.85 from Phase 2)'
    )

    parser.add_argument(
        '--baseline-precision',
        type=float,
        default=0.20,
        help='Expected baseline precision (default: 0.20)'
    )

    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='Allowable performance drift (default: 0.05)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PIPELINE VALIDATION SUITE (Task 5.2)")
    print("=" * 70)
    print(f"Metrics directory: {args.metrics_dir}")
    print(f"Performance check: {'enabled' if args.check_performance else 'disabled'}")

    # Run all tests
    all_results = []

    # Test 1: Metric Schema Validation
    try:
        result1 = test_metric_schemas(args.metrics_dir)
        all_results.append(result1)
    except Exception as e:
        print(f"\n[ERROR] Test 1 crashed: {e}")
        all_results.append({
            "test_name": "metric_schema_validation",
            "passed": False,
            "errors": [str(e)]
        })

    # Test 2: Feature Integration Edge Cases
    try:
        result2 = test_missing_data_handling()
        all_results.append(result2)
    except Exception as e:
        print(f"\n[ERROR] Test 2 crashed: {e}")
        all_results.append({
            "test_name": "feature_integration_edge_cases",
            "passed": False,
            "errors": [str(e)]
        })

    # Test 3: MSD Performance Regression (optional)
    if args.check_performance:
        try:
            result3 = test_msd_performance_drift(
                args.baseline_recall,
                args.baseline_precision,
                args.tolerance
            )
            all_results.append(result3)
        except Exception as e:
            print(f"\n[ERROR] Test 3 crashed: {e}")
            all_results.append({
                "test_name": "msd_performance_drift",
                "passed": False,
                "errors": [str(e)]
            })

    # Test 4: Manifest Integrity
    try:
        result4 = test_manifest_integrity(args.metrics_dir)
        all_results.append(result4)
    except Exception as e:
        print(f"\n[ERROR] Test 4 crashed: {e}")
        all_results.append({
            "test_name": "manifest_integrity",
            "passed": False,
            "errors": [str(e)]
        })

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUITE SUMMARY")
    print("=" * 70)

    passed_tests = sum(1 for r in all_results if r.get("passed") is True)
    failed_tests = sum(1 for r in all_results if r.get("passed") is False)
    skipped_tests = sum(1 for r in all_results if r.get("passed") is None)
    len(all_results)

    for result in all_results:
        test_name = result["test_name"]
        status = result.get("passed")
        if status is True:
            print(f"[PASS] {test_name}")
        elif status is False:
            print(f"[FAIL] {test_name}")
            for error in result.get("errors", []):
                print(f"   - {error}")
        else:
            print(f"[SKIP] {test_name}")

    print("\n" + "-" * 70)
    print(f"Results: {passed_tests} passed, {failed_tests} failed, {skipped_tests} skipped")

    # Exit code
    if failed_tests > 0:
        print("\n[FAIL] Validation suite failed")
        sys.exit(1)
    else:
        print("\n[PASS] Validation suite passed")
        sys.exit(0)


if __name__ == '__main__':
    main()
