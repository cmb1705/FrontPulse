"""Settings management for 2YP pipeline configuration."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional

import yaml

# New explicit settings path (preferred)
YAML_SETTINGS_PATH: pathlib.Path = pathlib.Path("config/settings.yaml")

# Legacy hidden settings path (backward compatibility)
LEGACY_JSON_PATH: pathlib.Path = pathlib.Path(".2yp_settings.json")
SETTINGS_PATH = LEGACY_JSON_PATH  # Backward compatibility alias

DEFAULTS: Dict[str, Any] = {
    "topics_id": "<PSC_TOPIC_ID>",
    "from_date": "2000-01-01",
    "to_date": "2025-08-30",
    "max_records": None,          # None = uncapped
    "per_page": 200,
    "graph_mode": "cumulative",   # none|annual|delta|both|cumulative
    "mailto": None,
}


def _flatten_yaml_settings(yaml_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested YAML structure to flat dictionary for backward compatibility.

    Args:
        yaml_data: Nested settings from YAML file

    Returns:
        Flat settings dictionary compatible with existing code
    """
    flat = {}
    if "openalex" in yaml_data:
        oa = yaml_data["openalex"]
        flat["mailto"] = oa.get("mailto")
        flat["topics_id"] = oa.get("topics_id", DEFAULTS["topics_id"])
        flat["per_page"] = oa.get("per_page", DEFAULTS["per_page"])
        flat["max_records"] = oa.get("max_records")
    if "dates" in yaml_data:
        dates = yaml_data["dates"]
        flat["from_date"] = dates.get("from_date", DEFAULTS["from_date"])
        flat["to_date"] = dates.get("to_date", DEFAULTS["to_date"])
    if "graphs" in yaml_data:
        graphs = yaml_data["graphs"]
        flat["graph_mode"] = graphs.get("mode", DEFAULTS["graph_mode"])
    return flat


def load_settings() -> Dict[str, Any]:
    """
    Load settings from config/settings.yaml or .2yp_settings.json (legacy).

    Priority:
    1. config/settings.yaml (new explicit format)
    2. .2yp_settings.json (legacy hidden file)
    3. Defaults

    Returns:
        Dictionary containing settings with defaults applied for missing keys
    """
    # Try new YAML format first (preferred)
    if YAML_SETTINGS_PATH.exists():
        try:
            yaml_data = yaml.safe_load(YAML_SETTINGS_PATH.read_text())
            if yaml_data:
                flat = _flatten_yaml_settings(yaml_data)
                return {**DEFAULTS, **flat}
        except Exception:
            pass  # Fall through to legacy path

    # Fall back to legacy JSON format
    if LEGACY_JSON_PATH.exists():
        try:
            return {**DEFAULTS, **json.loads(LEGACY_JSON_PATH.read_text())}
        except Exception:
            pass

    return DEFAULTS.copy()


def save_settings(cfg: Dict[str, Any]) -> None:
    """
    Save settings to .2yp_settings.json.

    Args:
        cfg: Settings dictionary to persist
    """
    SETTINGS_PATH.write_text(json.dumps(cfg, indent=2))


def redact_mailto(mailto: Optional[str]) -> Optional[str]:
    """Mask an email address before writing it to logs or metadata."""
    if not mailto:
        return None
    value = str(mailto).strip()
    if "@" not in value:
        return value
    local_part, domain = value.split("@", 1)
    if not local_part:
        return f"***@{domain}"
    return f"{local_part[0]}***@{domain}"


def summary(cfg: Dict[str, Any]) -> str:
    """
    Generate a one-line summary of settings for logging.

    Args:
        cfg: Settings dictionary

    Returns:
        Human-readable summary string
    """
    return (
        f"topics.id={cfg['topics_id']}, "
        f"from={cfg['from_date']}, to={cfg['to_date']}, "
        f"max_records={cfg['max_records']}, per_page={cfg['per_page']}, "
        f"graph_mode={cfg['graph_mode']}, mailto={redact_mailto(cfg.get('mailto'))}"
    )
