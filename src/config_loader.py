"""
Configuration loader for Multi-Signal Context Integration (Task 5.3).

Provides centralized configuration management for:
- Metric parameters
- Feature engineering settings
- Model training hyperparameters
- Meta-learning search spaces
- Pipeline integration settings

Usage:
    from src.config_loader import load_config, get_metric_config, get_model_config

    # Load full configuration
    config = load_config()

    # Access specific sections
    metric_params = get_metric_config('citation_velocity')
    model_config = get_model_config()
    tuning_config = get_tuning_config()
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "multisignal_config.yaml"


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default location.

    Returns:
        Configuration dictionary

    Raises:
        ConfigurationError: If config file not found or invalid
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")


def get_metric_config(metric_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific metric.

    Args:
        metric_name: Name of metric (e.g., 'author_influx')
        config: Configuration dict (loaded if not provided)

    Returns:
        Metric configuration dict

    Raises:
        ConfigurationError: If metric not found in config
    """
    if config is None:
        config = load_config()

    metrics_config = config.get('metrics', {})

    if metric_name not in metrics_config:
        raise ConfigurationError(f"Metric '{metric_name}' not found in configuration")

    return metrics_config[metric_name]


def get_enabled_metrics(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Get list of enabled metrics.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        List of enabled metric names
    """
    if config is None:
        config = load_config()

    metrics_config = config.get('metrics', {})
    enabled = metrics_config.get('enabled', [])

    # Filter by individual metric enabled flags
    result = []
    for metric_name in enabled:
        metric_config = metrics_config.get(metric_name, {})
        if metric_config.get('enabled', True):  # Default to enabled if not specified
            result.append(metric_name)

    return result


def get_feature_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get feature engineering configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Feature configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('features', {})


def get_context_features_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get context features configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Context features configuration dict
    """
    feature_config = get_feature_config(config)
    return feature_config.get('context_features', {})


def get_model_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get model training configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Model configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('model', {})


def get_tuning_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get meta-learning / hyperparameter tuning configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Tuning configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('tuning', {})


def get_pipeline_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get pipeline integration configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Pipeline configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('pipeline', {})


def get_validation_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get validation and monitoring configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Validation configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('validation', {})


def get_logging_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get logging and audit trail configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Logging configuration dict
    """
    if config is None:
        config = load_config()

    return config.get('logging', {})


def validate_config(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Validate configuration and return list of warnings/errors.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        List of validation issues (empty if valid)
    """
    if config is None:
        config = load_config()

    issues = []

    # Check required top-level sections
    required_sections = ['metrics', 'features', 'model', 'tuning', 'pipeline']
    for section in required_sections:
        if section not in config:
            issues.append(f"Missing required section: {section}")

    # Validate metrics section
    metrics_config = config.get('metrics', {})
    if 'enabled' not in metrics_config:
        issues.append("Missing 'metrics.enabled' list")

    # Validate feature normalization strategy
    feature_config = config.get('features', {})
    norm_config = feature_config.get('normalization', {})
    valid_strategies = ['standard', 'robust', 'minmax', 'none']
    strategy = norm_config.get('strategy', 'standard')
    if strategy not in valid_strategies:
        issues.append(f"Invalid normalization strategy: {strategy}. Must be one of {valid_strategies}")

    # Validate model configuration
    model_config = config.get('model', {})
    training_config = model_config.get('training', {})
    if 'default_model' not in training_config:
        issues.append("Missing 'model.training.default_model'")

    # Validate tuning search space
    tuning_config = config.get('tuning', {})
    if tuning_config.get('enabled', False):
        search_space = tuning_config.get('search_space', {})
        if 'model_types' not in search_space:
            issues.append("Tuning enabled but 'tuning.search_space.model_types' not defined")

    return issues


def get_config_summary(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate human-readable summary of configuration.

    Args:
        config: Configuration dict (loaded if not provided)

    Returns:
        Multi-line string summary
    """
    if config is None:
        config = load_config()

    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-SIGNAL CONTEXT INTEGRATION - CONFIGURATION SUMMARY")
    lines.append("=" * 70)

    # Version
    version = config.get('version', 'unknown')
    last_updated = config.get('last_updated', 'unknown')
    lines.append(f"Version: {version}")
    lines.append(f"Last Updated: {last_updated}")

    # Metrics
    lines.append("\n[METRICS]")
    enabled_metrics = get_enabled_metrics(config)
    lines.append(f"  Enabled: {len(enabled_metrics)} metrics")
    for metric in enabled_metrics:
        lines.append(f"    - {metric}")

    # Features
    lines.append("\n[FEATURES]")
    feature_config = get_feature_config(config)
    context_config = feature_config.get('context_features', {})
    context_enabled = context_config.get('enabled', False)
    lines.append(f"  Context features: {'enabled' if context_enabled else 'disabled'}")
    if context_enabled:
        include_config = context_config.get('include', {})
        lines.append(f"    Z-scores: {include_config.get('z_scores', False)}")
        lines.append(f"    QoQ deltas: {include_config.get('qoq_deltas', False)}")
        lines.append(f"    Rolling averages: {include_config.get('rolling_averages', False)}")
        lines.append(f"    Burst detection: {include_config.get('burst_detection', False)}")

    norm_config = feature_config.get('normalization', {})
    lines.append(f"  Normalization strategy: {norm_config.get('strategy', 'standard')}")

    # Model
    lines.append("\n[MODEL]")
    model_config = get_model_config(config)
    training_config = model_config.get('training', {})
    lines.append(f"  Default model: {training_config.get('default_model', 'unknown')}")
    lines.append(f"  SMOTE: {training_config.get('use_smote', False)}")
    calibration_config = model_config.get('calibration', {})
    lines.append(f"  Calibration: {calibration_config.get('method', 'none')}")

    # Tuning
    lines.append("\n[TUNING]")
    tuning_config = get_tuning_config(config)
    tuning_enabled = tuning_config.get('enabled', False)
    lines.append(f"  Enabled: {tuning_enabled}")
    if tuning_enabled:
        lines.append(f"  Trials: {tuning_config.get('n_trials', 'unknown')}")
        search_space = tuning_config.get('search_space', {})
        model_types = search_space.get('model_types', [])
        lines.append(f"  Model families: {', '.join(model_types)}")

    # Pipeline
    lines.append("\n[PIPELINE]")
    pipeline_config = get_pipeline_config(config)
    metric_refresh = pipeline_config.get('metric_refresh', {})
    lines.append(f"  Metric refresh: {metric_refresh.get('enabled_by_default', False)}")
    cache_config = pipeline_config.get('cache', {})
    lines.append(f"  Abstract cache: {cache_config.get('enable_abstract_cache', True)}")

    # Validation
    lines.append("\n[VALIDATION]")
    validation_config = get_validation_config(config)
    schema_val = validation_config.get('schema_validation', {})
    lines.append(f"  Schema validation: {schema_val.get('enabled', True)}")
    perf_test = validation_config.get('performance_testing', {})
    lines.append(f"  Performance testing: {perf_test.get('enabled', False)}")

    lines.append("=" * 70)

    return "\n".join(lines)


# Convenience function for CLI tools
def print_config_summary(config_path: Optional[Path] = None):
    """
    Load and print configuration summary.

    Args:
        config_path: Path to configuration file (optional)
    """
    try:
        config = load_config(config_path)
        issues = validate_config(config)

        if issues:
            print("[WARNING] Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            print()

        print(get_config_summary(config))

    except ConfigurationError as e:
        print(f"[ERROR] {e}")
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(print_config_summary())
