"""Tests for OpenAlex API key authentication support."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# fetch_openalex: API key support
# ---------------------------------------------------------------------------


def _mock_empty_response() -> MagicMock:
    """Return a mock HTTP response with zero results."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "meta": {"count": 0, "next_cursor": None},
        "results": [],
    }
    return mock


def test_fetch_openalex_accepts_api_key():
    """fetch_openalex must accept api_key parameter without mailto."""
    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = _mock_empty_response()
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        fetch_openalex("works", api_key="test_key_123", max_records=1)

        call_args = mock_sess.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["api_key"] == "test_key_123"
        assert "mailto" not in params


def test_fetch_openalex_mailto_backward_compat():
    """fetch_openalex must still work with mailto only (no api_key)."""
    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = _mock_empty_response()
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        fetch_openalex("works", mailto="test@example.com", max_records=1)

        call_args = mock_sess.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["mailto"] == "test@example.com"


def test_fetch_openalex_neither_raises():
    """fetch_openalex must raise ValueError if neither api_key nor mailto."""
    from src.openalex import fetch_openalex

    with pytest.raises(ValueError, match="api_key.*mailto"):
        fetch_openalex("works")


def test_user_agent_frontpulse_branding_api_key():
    """User-Agent header must say FrontPulse (not 2YP) when using api_key."""
    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = _mock_empty_response()
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        fetch_openalex("works", api_key="test_key", max_records=1)

        call_args = mock_sess.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "FrontPulse" in headers["User-Agent"]
        assert "2YP" not in headers["User-Agent"]


def test_user_agent_includes_mailto_when_provided():
    """User-Agent header must include mailto when provided."""
    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = _mock_empty_response()
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        fetch_openalex("works", mailto="user@example.com", max_records=1)

        call_args = mock_sess.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "FrontPulse/1.0" in headers["User-Agent"]
        assert "user@example.com" in headers["User-Agent"]


# ---------------------------------------------------------------------------
# _read_one: API key passthrough
# ---------------------------------------------------------------------------


def test_read_one_passes_api_key():
    """_read_one must pass api_key through to fetch_openalex."""
    import pandas as pd

    with patch("src.ingest.fetch_openalex") as mock_fetch:
        mock_fetch.return_value = []
        with patch("src.ingest.results_to_df") as mock_df:
            mock_df.return_value = pd.DataFrame()

            from src.ingest import _read_one

            _read_one({
                "kind": "openalex",
                "entity": "works",
                "api_key": "test_key_456",
                "per_page": 200,
            })

            mock_fetch.assert_called_once()
            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs.get("api_key") == "test_key_456"


def test_read_one_accepts_either_auth():
    """_read_one must not raise if api_key is provided without mailto."""
    import pandas as pd

    with patch("src.ingest.fetch_openalex") as mock_fetch:
        mock_fetch.return_value = []
        with patch("src.ingest.results_to_df") as mock_df:
            mock_df.return_value = pd.DataFrame()

            from src.ingest import _read_one

            _read_one({
                "kind": "openalex",
                "entity": "works",
                "api_key": "key_only",
                "per_page": 200,
            })


def test_read_one_raises_without_any_auth():
    """_read_one must raise ValueError if neither mailto nor api_key."""
    from src.ingest import _read_one

    with pytest.raises(ValueError, match="API key.*email|email.*API key"):
        _read_one({"kind": "openalex", "entity": "works", "per_page": 200})


def test_read_one_mailto_only_backward_compat():
    """Full path: ingest with mailto only (no api_key) must work."""
    import pandas as pd

    with patch("src.ingest.fetch_openalex") as mock_fetch:
        mock_fetch.return_value = []
        with patch("src.ingest.results_to_df") as mock_df:
            mock_df.return_value = pd.DataFrame()

            from src.ingest import _read_one

            _read_one({
                "kind": "openalex",
                "entity": "works",
                "mailto": "test@example.com",
                "per_page": 200,
            })

            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs.get("mailto") == "test@example.com"
            assert call_kwargs.get("api_key") is None
