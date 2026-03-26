"""Settings management for 2YP pipeline configuration."""
from __future__ import annotations

import contextlib
import json
import pathlib
from typing import Any

import yaml

# New explicit settings path (preferred)
YAML_SETTINGS_PATH: pathlib.Path = pathlib.Path("config/settings.yaml")

# Legacy hidden settings path (backward compatibility)
LEGACY_JSON_PATH: pathlib.Path = pathlib.Path(".2yp_settings.json")
SETTINGS_PATH = LEGACY_JSON_PATH  # Backward compatibility alias

DEFAULTS: dict[str, Any] = {
    "topics_id": "<PSC_TOPIC_ID>",
    "from_date": "2000-01-01",
    "to_date": "2025-08-30",
    "max_records": None,          # None = uncapped
    "per_page": 200,
    "graph_mode": "cumulative",   # none|annual|delta|both|cumulative
    "mailto": None,
    "last_ingested_date": None,   # Watermark for incremental ingestion
}


def _flatten_yaml_settings(yaml_data: dict[str, Any]) -> dict[str, Any]:
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
        if "last_ingested_date" in dates:
            flat["last_ingested_date"] = dates["last_ingested_date"]
    if "graphs" in yaml_data:
        graphs = yaml_data["graphs"]
        flat["graph_mode"] = graphs.get("mode", DEFAULTS["graph_mode"])
    return flat


def load_settings(
    domain_settings_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Load settings with optional per-domain state path.

    Priority:
    1. Per-domain settings file (if provided and exists)
    2. config/settings.yaml (shared defaults)
    3. .2yp_settings.json (legacy hidden file)
    4. Built-in defaults

    When ``domain_settings_path`` is provided, it supplies the mutable
    runtime state (e.g., ``last_ingested_date`` watermark) while shared
    defaults still load from config/settings.yaml for base configuration.

    Args:
        domain_settings_path: Optional path to a per-domain JSON settings
            file (e.g., ``data/psc/settings.json``).

    Returns:
        Dictionary containing settings with defaults applied for missing keys.
    """
    result = DEFAULTS.copy()

    # Layer 1: shared YAML defaults
    if YAML_SETTINGS_PATH.exists():
        try:
            yaml_data = yaml.safe_load(YAML_SETTINGS_PATH.read_text())
            if yaml_data:
                result.update(_flatten_yaml_settings(yaml_data))
        except Exception:
            pass

    # Layer 2: legacy JSON (only if no YAML and no domain path)
    if (
        not YAML_SETTINGS_PATH.exists()
        and domain_settings_path is None
        and LEGACY_JSON_PATH.exists()
    ):
        with contextlib.suppress(Exception):
            result.update(json.loads(LEGACY_JSON_PATH.read_text()))

    # Layer 3: per-domain state overrides (highest priority for mutable state)
    if domain_settings_path is not None and domain_settings_path.exists():
        try:
            domain_data = json.loads(domain_settings_path.read_text())
            if domain_data:
                result.update(domain_data)
        except Exception:
            pass

    return result


def save_settings(
    cfg: dict[str, Any],
    domain_settings_path: pathlib.Path | None = None,
) -> None:
    """Save settings, optionally to a per-domain path.

    When ``domain_settings_path`` is provided, mutable runtime state
    (watermarks, last-used parameters) is written there instead of
    the global ``.2yp_settings.json``.

    Args:
        cfg: Settings dictionary to persist.
        domain_settings_path: Optional per-domain JSON path. When
            provided, settings are saved here; otherwise the global
            legacy path is used.
    """
    target = domain_settings_path if domain_settings_path else SETTINGS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(cfg, indent=2))


def redact_mailto(mailto: str | None) -> str | None:
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


def summary(cfg: dict[str, Any]) -> str:
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
