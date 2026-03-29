#!/usr/bin/env python3
"""
Smoke test for configuration loader (Task 5.3).

Tests configuration loading and accessor functions.
"""

from __future__ import annotations

import sys

from _path_bootstrap import ensure_repo_imports

repo_root = ensure_repo_imports()

from config_loader import (  # noqa: E402
    get_config_summary,
    get_enabled_metrics,
    get_feature_config,
    get_metric_config,
    get_model_config,
    get_pipeline_config,
    get_tuning_config,
    load_config,
    validate_config,
)


def test_config_loading():
    """Test basic configuration loading."""
    print("=" * 70)
    print("TEST: Configuration Loading")
    print("=" * 70)

    config = load_config()
    assert config is not None, "Config should not be None"
    assert 'version' in config, "Config should have version field"
    print(f"[PASS] Loaded config version {config['version']}")


def test_metric_config():
    """Test metric configuration access."""
    print("\n" + "=" * 70)
    print("TEST: Metric Configuration")
    print("=" * 70)

    enabled_metrics = get_enabled_metrics()
    print(f"[INFO] Found {len(enabled_metrics)} enabled metrics: {enabled_metrics}")
    assert len(enabled_metrics) > 0, "Should have at least one enabled metric"
    print("[PASS] Enabled metrics loaded")

    # Test individual metric config
    for metric in enabled_metrics[:2]:  # Test first 2
        metric_config = get_metric_config(metric)
        assert 'enabled' in metric_config, f"Metric {metric} should have 'enabled' field"
        print(f"[PASS] Metric '{metric}' config loaded")


def test_feature_config():
    """Test feature configuration access."""
    print("\n" + "=" * 70)
    print("TEST: Feature Configuration")
    print("=" * 70)

    feature_config = get_feature_config()
    assert 'context_features' in feature_config, "Should have context_features section"

    context_config = feature_config['context_features']
    is_enabled = context_config.get('enabled', False)
    print(f"[INFO] Context features enabled: {is_enabled}")

    if is_enabled:
        include_config = context_config.get('include', {})
        print(f"[INFO] Z-scores: {include_config.get('z_scores', False)}")
        print(f"[INFO] QoQ deltas: {include_config.get('qoq_deltas', False)}")

    print("[PASS] Feature config loaded")


def test_model_config():
    """Test model configuration access."""
    print("\n" + "=" * 70)
    print("TEST: Model Configuration")
    print("=" * 70)

    model_config = get_model_config()
    assert 'training' in model_config, "Should have training section"
    assert 'calibration' in model_config, "Should have calibration section"

    training_config = model_config['training']
    default_model = training_config.get('default_model', 'unknown')
    use_smote = training_config.get('use_smote', False)

    print(f"[INFO] Default model: {default_model}")
    print(f"[INFO] SMOTE: {use_smote}")
    print("[PASS] Model config loaded")


def test_tuning_config():
    """Test tuning configuration access."""
    print("\n" + "=" * 70)
    print("TEST: Tuning Configuration")
    print("=" * 70)

    tuning_config = get_tuning_config()
    is_enabled = tuning_config.get('enabled', False)
    n_trials = tuning_config.get('n_trials', 0)

    print(f"[INFO] Tuning enabled: {is_enabled}")
    print(f"[INFO] Number of trials: {n_trials}")

    if is_enabled:
        search_space = tuning_config.get('search_space', {})
        model_types = search_space.get('model_types', [])
        print(f"[INFO] Model types to search: {model_types}")

    print("[PASS] Tuning config loaded")


def test_pipeline_config():
    """Test pipeline configuration access."""
    print("\n" + "=" * 70)
    print("TEST: Pipeline Configuration")
    print("=" * 70)

    pipeline_config = get_pipeline_config()
    assert 'metric_refresh' in pipeline_config, "Should have metric_refresh section"
    assert 'paths' in pipeline_config, "Should have paths section"

    paths = pipeline_config['paths']
    metrics_dir = paths.get('metrics_dir', 'unknown')
    print(f"[INFO] Metrics directory: {metrics_dir}")

    cache_config = pipeline_config.get('cache', {})
    abstract_cache = cache_config.get('enable_abstract_cache', False)
    print(f"[INFO] Abstract cache: {abstract_cache}")

    print("[PASS] Pipeline config loaded")


def test_validation():
    """Test configuration validation."""
    print("\n" + "=" * 70)
    print("TEST: Configuration Validation")
    print("=" * 70)

    issues = validate_config()

    if issues:
        print(f"[WARNING] Found {len(issues)} validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("[PASS] Configuration is valid (no issues found)")


def test_summary():
    """Test configuration summary generation."""
    print("\n" + "=" * 70)
    print("TEST: Configuration Summary")
    print("=" * 70)

    summary = get_config_summary()
    assert len(summary) > 0, "Summary should not be empty"
    assert "CONFIGURATION SUMMARY" in summary, "Summary should contain header"

    print("[PASS] Configuration summary generated")
    print("\n" + summary)


def main():
    """Run all tests."""
    try:
        test_config_loading()
        test_metric_config()
        test_feature_config()
        test_model_config()
        test_tuning_config()
        test_pipeline_config()
        test_validation()
        test_summary()

        print("\n" + "=" * 70)
        print("[SUCCESS] All configuration tests passed!")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
