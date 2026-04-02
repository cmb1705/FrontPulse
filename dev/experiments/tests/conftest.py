"""Pytest fixtures for 2YP test suite."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def sample_works_df() -> pd.DataFrame:
    """Provide a sample DataFrame with OpenAlex works data."""
    return pd.DataFrame({
        "id": ["W001", "W002", "W003", "W004"],
        "title": [
            "Machine Learning Applications",
            "Deep Learning Survey",
            "Neural Networks",
            "AI Ethics"
        ],
        "publication_date": [
            "2020-03-15",
            "2021-07-20",
            "2022-11-05",
            "2023-02-14"
        ],
        "cited_by_count": [10, 25, 5, 8],
        "authorships": [
            [{"author": {"id": "A1"}}],
            [{"author": {"id": "A2"}}, {"author": {"id": "A3"}}],
            [{"author": {"id": "A1"}}],
            [{"author": {"id": "A4"}}]
        ],
        "topics": [
            [{"id": "T1", "display_name": "Machine Learning"}],
            [{"id": "T1", "display_name": "Machine Learning"}],
            [{"id": "T2", "display_name": "Neural Networks"}],
            [{"id": "T3", "display_name": "Ethics"}]
        ]
    })


@pytest.fixture
def sample_citation_graph_data() -> dict[str, Any]:
    """Provide sample citation graph data for testing."""
    return {
        "nodes": ["W001", "W002", "W003", "W004"],
        "edges": [
            ("W002", "W001"),  # W002 cites W001
            ("W003", "W001"),  # W003 cites W001
            ("W004", "W002"),  # W004 cites W002
        ],
        "node_attrs": {
            "W001": {"pub_date": "2020-03-15", "cited_by_count": 10},
            "W002": {"pub_date": "2021-07-20", "cited_by_count": 25},
            "W003": {"pub_date": "2022-11-05", "cited_by_count": 5},
            "W004": {"pub_date": "2023-02-14", "cited_by_count": 8},
        }
    }


@pytest.fixture
def sample_schema_yaml(temp_dir: Path) -> str:
    """Provide a sample schema YAML file for testing."""
    schema_path = temp_dir / "schema.yaml"
    schema_content = """
columns:
  id:
    type: str
    nullable: false
  title:
    type: str
    nullable: false
  publication_date:
    type: datetime64[ns]
    nullable: true
  cited_by_count:
    type: int64
    nullable: true
    """
    schema_path.write_text(schema_content)
    return str(schema_path)


@pytest.fixture
def sample_slices_yaml(temp_dir: Path) -> str:
    """Provide a sample slices YAML file for testing."""
    slices_path = temp_dir / "slices.yaml"
    slices_content = """
slices:
  recent:
    filter:
      - pub_year >= 2021

  high_impact:
    filter:
      - cited_by_count > 10

  ml_topic:
    filter:
      - topics.id == "T1"
    """
    slices_path.write_text(slices_content)
    return str(slices_path)


@pytest.fixture
def sample_settings_json(temp_dir: Path) -> str:
    """Provide a sample settings JSON file for testing."""
    settings_path = temp_dir / ".2yp_settings.json"
    settings_content = {
        "mailto": "test@example.com",
        "filters": {"topics.id": "T10247"},
        "slice_mode": "annual",
        "graph_mode": "annual"
    }
    import json
    with open(settings_path, "w") as f:
        json.dump(settings_content, f, indent=2)
    return str(settings_path)
