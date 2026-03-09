"""Configuration loader for pipeline defaults."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

# Path to defaults configuration
DEFAULTS_PATH = Path("config/defaults.yaml")

# Cached defaults (loaded once)
_DEFAULTS_CACHE: Dict[str, Any] | None = None


def load_defaults() -> Dict[str, Any]:
    """
    Load default configuration values from config/defaults.yaml.

    Returns:
        Dictionary containing all default configuration values

    Raises:
        FileNotFoundError: If defaults.yaml doesn't exist
        yaml.YAMLError: If defaults.yaml is malformed
    """
    global _DEFAULTS_CACHE

    if _DEFAULTS_CACHE is not None:
        return _DEFAULTS_CACHE

    if not DEFAULTS_PATH.exists():
        raise FileNotFoundError(
            f"Defaults configuration not found: {DEFAULTS_PATH}\n"
            "This file should contain all default parameter values."
        )

    try:
        _DEFAULTS_CACHE = yaml.safe_load(DEFAULTS_PATH.read_text())
        return _DEFAULTS_CACHE
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse {DEFAULTS_PATH}: {e}")


def get_default(section: str, key: str, fallback: Any = None) -> Any:
    """
    Get a specific default value.

    Args:
        section: Configuration section (e.g., 'coupling', 'community')
        key: Parameter key within section
        fallback: Value to return if key not found

    Returns:
        Configuration value or fallback

    Example:
        >>> get_default('coupling', 'alpha')
        1.0
        >>> get_default('coupling', 'beta')
        0.3
    """
    defaults = load_defaults()
    return defaults.get(section, {}).get(key, fallback)


def get_coupling_defaults() -> Dict[str, Any]:
    """Get all coupling parameter defaults."""
    return load_defaults().get("coupling", {})


def get_community_defaults() -> Dict[str, Any]:
    """Get all community detection parameter defaults."""
    return load_defaults().get("community", {})


def get_ingest_defaults() -> Dict[str, Any]:
    """Get all ingest parameter defaults."""
    return load_defaults().get("ingest", {})


def get_graph_defaults() -> Dict[str, Any]:
    """Get all graph building parameter defaults."""
    return load_defaults().get("graphs", {})


def get_memory_defaults() -> Dict[str, Any]:
    """Get all memory management parameter defaults."""
    return load_defaults().get("memory", {})
