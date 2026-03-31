"""Tests for run.validate_topic_alignment -- post-ingest topic validation.

Verifies that the function correctly compares DataFrame primary_topic_id
values against the configured topic filter in the datasource YAML.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from run import validate_topic_alignment

pytestmark = pytest.mark.unit


# -- Helpers -----------------------------------------------------------------

def _make_config(tmp_path: Path, topic_id: str | None = "T10247") -> Path:
    """Write a minimal datasource YAML and return its path.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        topic_id: Topic ID to embed in the config, or None to omit
            the ``topics.id`` key entirely.

    Returns:
        Path to the generated YAML file.
    """
    cfg: dict = {"sources": {"primary": {"filters": {}}}}
    if topic_id is not None:
        cfg["sources"]["primary"]["filters"]["topics.id"] = topic_id
    path = tmp_path / "datasources.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


def _make_df(
    matching: int,
    non_matching: int,
    topic_id: str = "T10247",
) -> pd.DataFrame:
    """Build a DataFrame with a mix of matching and non-matching topic rows.

    Args:
        matching: Number of rows whose ``primary_topic_id`` matches *topic_id*.
        non_matching: Number of rows with a different topic URL.
        topic_id: The topic ID (without the OpenAlex URL prefix).

    Returns:
        DataFrame with a ``primary_topic_id`` column.
    """
    url = f"https://openalex.org/{topic_id}"
    rows = [url] * matching + ["https://openalex.org/T99999"] * non_matching
    return pd.DataFrame({"primary_topic_id": rows})


# -- Tests -------------------------------------------------------------------

class TestValidateTopicAlignment:
    """Unit tests for validate_topic_alignment."""

    def test_missing_column_returns_true(self, tmp_path: Path) -> None:
        """DataFrame without primary_topic_id should skip validation and return True."""
        config = _make_config(tmp_path)
        df = pd.DataFrame({"title": ["some paper"]})
        logger = logging.getLogger("test")

        result = validate_topic_alignment(df, config, logger)

        assert result is True

    def test_matching_data_above_threshold(self, tmp_path: Path) -> None:
        """DataFrame where >30% of rows match the configured topic returns True."""
        config = _make_config(tmp_path, topic_id="T10247")
        # 4 matching out of 10 = 40%, above the 30% default threshold.
        df = _make_df(matching=4, non_matching=6)
        logger = logging.getLogger("test")

        result = validate_topic_alignment(df, config, logger)

        assert result is True

    def test_data_below_threshold_returns_false(self, tmp_path: Path) -> None:
        """DataFrame where <30% match returns False."""
        config = _make_config(tmp_path, topic_id="T10247")
        # 2 matching out of 10 = 20%, below the 30% default threshold.
        df = _make_df(matching=2, non_matching=8)
        logger = logging.getLogger("test")

        result = validate_topic_alignment(df, config, logger)

        assert result is False

    def test_missing_topic_filter_returns_true(self, tmp_path: Path) -> None:
        """Config YAML without topics.id should skip validation and return True."""
        config = _make_config(tmp_path, topic_id=None)
        df = _make_df(matching=0, non_matching=10)
        logger = logging.getLogger("test")

        result = validate_topic_alignment(df, config, logger)

        assert result is True

    def test_custom_threshold(self, tmp_path: Path) -> None:
        """Verify that min_match_fraction parameter overrides the default threshold."""
        config = _make_config(tmp_path, topic_id="T10247")
        # 5 matching out of 10 = 50%.
        df = _make_df(matching=5, non_matching=5)
        logger = logging.getLogger("test")

        # With a high threshold (60%), 50% match should fail.
        assert validate_topic_alignment(df, config, logger, min_match_fraction=0.6) is False

        # With a low threshold (40%), 50% match should pass.
        assert validate_topic_alignment(df, config, logger, min_match_fraction=0.4) is True
