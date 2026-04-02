from __future__ import annotations

import json
from pathlib import Path

from src.settings import save_settings


def test_save_settings_omits_api_keys(tmp_path: Path) -> None:
    """Persisted settings must not contain OpenAlex API keys."""
    target = tmp_path / "settings.json"
    cfg = {
        "topics_id": "T10247",
        "from_date": "2003-01-01",
        "to_date": "",
        "per_page": 200,
        "graph_mode": "cumulative",
        "last_ingested_date": "2026-03-30",
        "api_key": "top-level-secret",
        "source_overrides": {
            "api_key": "nested-secret",
            "filters": {
                "topics.id": "T10247",
                "from_publication_date": "2003-01-01",
                "to_publication_date": "",
            },
        },
    }

    save_settings(cfg, domain_settings_path=target)

    saved = json.loads(target.read_text())
    assert "api_key" not in saved
    assert "api_key" not in saved["source_overrides"]
    assert saved["last_ingested_date"] == "2026-03-30"
    assert saved["source_overrides"]["filters"]["topics.id"] == "T10247"
