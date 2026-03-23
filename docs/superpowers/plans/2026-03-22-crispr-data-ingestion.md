# CRISPR Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CRISPR gene editing as a second research domain with API key authentication support.

**Architecture:** Extend the existing OpenAlex ingestion pipeline to support API key auth alongside mailto, create a CRISPR-specific datasource config, and wire the key from a `.env` file through `run.py` -> `ingest.py` -> `openalex.py`.

**Tech Stack:** Python 3.9+, python-dotenv, OpenAlex API, pytest

**Spec:** `docs/superpowers/specs/2026-03-21-crispr-data-ingestion-design.md`

**Beads Epic:** FP-bii (tasks FP-bii.1 through FP-bii.7)

---

## File Map

| File | Action | Responsibility |
| --- | --- | --- |
| `config/datasources_crispr.yaml` | Create | CRISPR datasource config (topic T10878) |
| `config/front_aliases_crispr.yaml` | Create | Placeholder CRISPR research front aliases |
| `src/openalex.py` | Modify (lines 24-35, 65, 79-82) | Add `api_key` param, update User-Agent |
| `src/ingest.py` | Modify (lines 50-67) | Wire `api_key` through `_read_one()` |
| `run.py` | Modify (lines 1-15, 157-168, 657-669) | Load dotenv, update `build_source_overrides`, relax mailto validation |
| `requirements.txt` | Modify | Add `python-dotenv>=1.0` |
| `.env.template` | Create | Template with `OPENALEX_API_KEY=` |
| `tests/test_smoke.py` | Modify | Add CRISPR config and API key tests |

---

## Task 1: Add python-dotenv and .env.template (FP-bii.6)

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore` (add `!.env.template` negation so the file can be committed)
- Create: `.env.template`
- Test: `tests/test_smoke.py`

- [ ] **Step 0: Update .gitignore to allow .env.template**

Line 76 of `.gitignore` contains `.env.*` which would block committing `.env.template`.
Add a negation rule immediately after:

```
.env.*
!.env.template
```

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smoke.py` -- a test that `.env.template` exists and contains the expected key name:

```python
def test_env_template_exists() -> None:
    """Env template must exist with API key placeholder."""
    path = PROJECT_ROOT / ".env.template"
    assert path.exists(), ".env.template missing from project root"
    text = path.read_text()
    assert "OPENALEX_API_KEY" in text, "OPENALEX_API_KEY placeholder missing"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_env_template_exists -v`
Expected: FAIL (file does not exist)

- [ ] **Step 3: Create `.env.template`**

```
# OpenAlex API key -- get yours at https://openalex.org/settings/api
OPENALEX_API_KEY=
```

- [ ] **Step 4: Add python-dotenv to requirements.txt**

Add after the `pydantic` line (line 8):

```
python-dotenv>=1.0
```

- [ ] **Step 5: Install the new dependency**

Run: `.venv/Scripts/pip install python-dotenv>=1.0`

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_env_template_exists -v`
Expected: PASS

- [ ] **Step 7: Run full smoke suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 8: Lint**

Run: `.venv/Scripts/python -m ruff check src/ tests/ --fix && .venv/Scripts/python -m ruff check src/ tests/`

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt .env.template tests/test_smoke.py
git commit -m "feat(FP-bii.6): add python-dotenv dep and .env.template"
```

---

## Task 2: Create CRISPR datasource config (FP-bii.1)

**Files:**
- Create: `config/datasources_crispr.yaml`
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_smoke.py` -- add `config/datasources_crispr.yaml` to `_CONFIG_FILES` and add a CRISPR topic ID test:

```python
def test_crispr_datasource_topic_id() -> None:
    """CRISPR config requires OpenAlex topic T10878."""
    path = PROJECT_ROOT / "config" / "datasources_crispr.yaml"
    if not path.exists():
        pytest.skip("datasources_crispr.yaml not present")
    raw = path.read_text()
    assert "T10878" in raw, "CRISPR topic ID T10878 missing"
```

Also add `"config/datasources_crispr.yaml"` to the `_CONFIG_FILES` list so the YAML-parse test covers it.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_crispr_datasource_topic_id -v`
Expected: SKIP (file not present -- acceptable) or FAIL

- [ ] **Step 3: Create the config file**

Create `config/datasources_crispr.yaml`:

```yaml
sources:
  primary:
    kind: openalex
    entity: works
    # Keep personal contact info in .env or pass --mailto.
    mailto: null
    max_records: null
    per_page: 200
    filters:
      topics.id: T10878
      from_publication_date: '2000-01-01'
      to_publication_date: ''
    select: null
    sort: null
merges: []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_crispr_datasource_topic_id -v`
Expected: PASS

- [ ] **Step 5: Run full smoke suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add config/datasources_crispr.yaml tests/test_smoke.py
git commit -m "feat(FP-bii.1): add CRISPR datasource config for topic T10878"
```

---

## Task 3: Create CRISPR front aliases placeholder (FP-bii.2)

**Files:**
- Create: `config/front_aliases_crispr.yaml`
- Modify: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add `"config/front_aliases_crispr.yaml"` to `_CONFIG_FILES` (if not already there) and add:

```python
def test_crispr_front_aliases_structure() -> None:
    """CRISPR front aliases must define canonical names for known fronts."""
    path = PROJECT_ROOT / "config" / "front_aliases_crispr.yaml"
    if not path.exists():
        pytest.skip("front_aliases_crispr.yaml not present")
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), "front_aliases_crispr.yaml must be a dict"
    assert "fronts" in data, "Must have a 'fronts' key"
    assert len(data["fronts"]) >= 5, "Should define at least 5 known CRISPR fronts"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_crispr_front_aliases_structure -v`
Expected: SKIP or FAIL

- [ ] **Step 3: Create the config file**

Create `config/front_aliases_crispr.yaml`:

```yaml
# Research Front Definitions for CRISPR Gene Editing
# Placeholder -- will be refined after inspecting Leiden clustering output.
# Not consumed by the pipeline until front-mapping is implemented (Phase 2+).

fronts:
  cas9_methodology:
    canonical: "Cas9 Editing Methodology"
    aliases: ["CRISPR-Cas9", "RNA-guided endonuclease"]
  eukaryotic_applications:
    canonical: "Eukaryotic/Mammalian Applications"
    aliases: []
  in_vivo_editing:
    canonical: "In Vivo Therapeutic Editing"
    aliases: []
  base_editing:
    canonical: "Base Editing"
    aliases: ["cytosine base editor", "adenine base editor", "CBE", "ABE"]
  prime_editing:
    canonical: "Prime Editing"
    aliases: []
  diagnostics:
    canonical: "CRISPR Diagnostics"
    aliases: ["SHERLOCK", "DETECTR", "Cas13"]
  clinical_therapeutics:
    canonical: "Clinical/FDA-Approved Therapies"
    aliases: ["Casgevy", "exa-cel", "sickle cell"]
```

- [ ] **Step 4: Run tests and verify**

Run: `.venv/Scripts/python -m pytest tests/test_smoke.py::test_crispr_front_aliases_structure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/front_aliases_crispr.yaml tests/test_smoke.py
git commit -m "feat(FP-bii.2): add CRISPR front aliases placeholder"
```

---

## Task 4: Add API key auth to src/openalex.py (FP-bii.3)

This is the critical-path task. All downstream code changes depend on this.

**Files:**
- Modify: `src/openalex.py:24-35,65,79-82`
- Create: `tests/test_openalex_auth.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_openalex_auth.py`:

```python
"""Tests for OpenAlex API key authentication support."""
from __future__ import annotations

import pytest


def test_fetch_openalex_accepts_api_key(monkeypatch):
    """fetch_openalex must accept api_key parameter without mailto."""
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "meta": {"count": 0, "next_cursor": None},
        "results": [],
    }

    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = mock_response
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        results = fetch_openalex(
            "works",
            api_key="test_key_123",
            max_records=1,
        )

        # Verify api_key was passed in params
        call_args = mock_sess.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["api_key"] == "test_key_123"
        assert "mailto" not in params or params.get("mailto") is None


def test_fetch_openalex_mailto_backward_compat(monkeypatch):
    """fetch_openalex must still work with mailto only (no api_key)."""
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "meta": {"count": 0, "next_cursor": None},
        "results": [],
    }

    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = mock_response
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        results = fetch_openalex(
            "works",
            mailto="test@example.com",
            max_records=1,
        )

        call_args = mock_sess.get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params")
        assert params["mailto"] == "test@example.com"


def test_fetch_openalex_neither_raises():
    """fetch_openalex must raise ValueError if neither api_key nor mailto."""
    from src.openalex import fetch_openalex

    with pytest.raises(ValueError, match="api_key.*mailto"):
        fetch_openalex("works")


def test_user_agent_frontpulse_branding(monkeypatch):
    """User-Agent header must say FrontPulse, not 2YP."""
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "meta": {"count": 0, "next_cursor": None},
        "results": [],
    }

    with patch("src.openalex.requests.Session") as mock_sess_cls:
        mock_sess = MagicMock()
        mock_sess.get.return_value = mock_response
        mock_sess_cls.return_value = mock_sess

        from src.openalex import fetch_openalex

        fetch_openalex("works", api_key="test_key", max_records=1)

        call_args = mock_sess.get.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "FrontPulse" in headers["User-Agent"]
        assert "2YP" not in headers["User-Agent"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py -v`
Expected: FAIL (TypeError -- unexpected keyword argument 'api_key')

- [ ] **Step 3: Implement the changes**

In `src/openalex.py`, modify `fetch_openalex`:

**Signature (lines 24-35):** Change `mailto: str` to `mailto: str | None = None` and add `api_key: str | None = None`:

```python
def fetch_openalex(
    entity: str,
    *,
    mailto: str | None = None,
    api_key: str | None = None,
    filters: Optional[Dict[str, Any]] = None,
    search: Optional[str] = None,
    select: Optional[List[str]] = None,
    sort: Optional[str] = None,
    per_page: int = 200,
    max_records: Optional[int] = None,
    sleep_s: float = 0.12,
) -> List[Dict[str, Any]]:
```

**Validation (after docstring, before params dict):** Add at line ~65 before the params dict:

```python
    if not api_key and not mailto:
        raise ValueError(
            "Either api_key or mailto is required. Set OPENALEX_API_KEY in your "
            ".env file or pass --mailto."
        )
```

**Params dict (line 65):** Build conditionally:

```python
    params: Dict[str, Any] = {"per-page": per_page, "cursor": "*"}
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
```

**User-Agent header (lines 79-82):** Update branding:

```python
    ua = "FrontPulse/1.0"
    if mailto:
        ua += f" (+mailto:{mailto})"
    headers = {
        "User-Agent": ua,
        "Accept": "application/json",
    }
```

**Docstring:** Update `mailto` description to note it is optional when `api_key` is provided. Add `api_key` parameter doc.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py -v`
Expected: All 4 PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 6: Lint**

Run: `.venv/Scripts/python -m ruff check src/openalex.py tests/test_openalex_auth.py --fix && .venv/Scripts/python -m ruff check src/openalex.py tests/test_openalex_auth.py`

- [ ] **Step 7: Commit**

```bash
git add src/openalex.py tests/test_openalex_auth.py
git commit -m "feat(FP-bii.3): add API key auth and rebrand User-Agent to FrontPulse"
```

---

## Task 5: Wire API key through _read_one() in src/ingest.py (FP-bii.4)

**Depends on:** Task 4 (FP-bii.3)

**Files:**
- Modify: `src/ingest.py:50-67`
- Modify: `tests/test_openalex_auth.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_openalex_auth.py`:

```python
def test_read_one_passes_api_key():
    """_read_one must pass api_key through to fetch_openalex."""
    from unittest.mock import patch, MagicMock
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
    from unittest.mock import patch
    import pandas as pd

    with patch("src.ingest.fetch_openalex") as mock_fetch:
        mock_fetch.return_value = []
        with patch("src.ingest.results_to_df") as mock_df:
            mock_df.return_value = pd.DataFrame()

            from src.ingest import _read_one

            # Should NOT raise -- api_key alone is sufficient
            _read_one({
                "kind": "openalex",
                "entity": "works",
                "api_key": "key_only",
                "per_page": 200,
            })


def test_read_one_raises_without_any_auth():
    """_read_one must raise ValueError if neither mailto nor api_key."""
    from src.ingest import _read_one

    with pytest.raises(ValueError, match="api_key.*mailto|mailto.*api_key"):
        _read_one({"kind": "openalex", "entity": "works", "per_page": 200})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py::test_read_one_passes_api_key -v`
Expected: FAIL

- [ ] **Step 3: Implement the changes**

In `src/ingest.py`, modify `_read_one()` (lines 50-67):

```python
    kind = src.get("kind", "csv").lower()
    if kind == "openalex":
        mailto = src.get("mailto")
        api_key = src.get("api_key")
        if not mailto and not api_key:
            raise ValueError(
                "OpenAlex source requires either an API key (OPENALEX_API_KEY in .env) "
                "or a contact email (--mailto / config/settings.yaml)."
            )
        results = fetch_openalex(
            entity=src.get("entity", "works"),
            mailto=mailto,
            api_key=api_key,
            filters=src.get("filters"),
            search=src.get("search"),
            select=src.get("select"),
            sort=src.get("sort"),
            per_page=int(src.get("per_page", 200)),
            max_records=src.get("max_records"),
        )
```

Update the docstring for `_read_one()` to mention `api_key` as an alternative to `mailto`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py -v`
Expected: All 7 PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python -m ruff check src/ingest.py tests/test_openalex_auth.py --fix
git add src/ingest.py tests/test_openalex_auth.py
git commit -m "feat(FP-bii.4): wire api_key through _read_one to fetch_openalex"
```

---

## Task 6: Update run.py mailto/api_key validation (FP-bii.5)

**Depends on:** Tasks 4, 5, and 1 (FP-bii.3, FP-bii.4, FP-bii.6)

**Files:**
- Modify: `run.py:1-15,157-168,657-669`
- Modify: `tests/test_openalex_auth.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_openalex_auth.py`:

```python
def test_build_source_overrides_includes_api_key():
    """build_source_overrides must include api_key when present."""
    # Inline import to avoid run.py's heavy imports at module level
    import importlib
    import sys

    # We need to test run.py's build_source_overrides function
    spec = importlib.util.spec_from_file_location(
        "run_module",
        str(Path(__file__).resolve().parents[1] / "run.py"),
    )
    run_mod = importlib.util.module_from_spec(spec)

    # Mock heavy imports that run.py needs
    from unittest.mock import MagicMock
    for mod_name in [
        "src.graph_build", "src.settings", "src.raw_store",
        "src.community_detect", "src.metrics", "src.feature_engineering",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    try:
        spec.loader.exec_module(run_mod)
    except Exception:
        pytest.skip("Could not load run.py in isolation")

    cfg = {
        "per_page": 200,
        "max_records": None,
        "mailto": "test@example.com",
        "api_key": "test_key_789",
        "topics_id": "T10878",
        "from_date": "2000-01-01",
        "to_date": "",
    }
    overrides = run_mod.build_source_overrides(cfg)
    assert overrides.get("api_key") == "test_key_789"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py::test_build_source_overrides_includes_api_key -v`
Expected: FAIL (api_key not in overrides)

- [ ] **Step 3: Implement the changes**

**3a. Add `import os` to run.py** (line 2, with other stdlib imports):

`run.py` does not currently import `os`. Add it to the imports at line 2:

```python
import argparse, pathlib, json, sys, os
```

**3b. Add dotenv import to run.py** (near top, after line 11):

```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables
```

**3c. Update `build_source_overrides`** (lines 157-168) to include `api_key`:

```python
def build_source_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build runtime datasource overrides without rewriting tracked YAML."""
    overrides: Dict[str, Any] = {
        "per_page": int(cfg["per_page"]),
        "max_records": None if cfg["max_records"] in (None, "", "None") else int(cfg["max_records"]),
        "mailto": cfg.get("mailto"),
        "filters": {
            "topics.id": cfg["topics_id"],
            "from_publication_date": cfg["from_date"],
            "to_publication_date": cfg["to_date"],
        },
    }
    api_key = cfg.get("api_key")
    if api_key:
        overrides["api_key"] = api_key
    return overrides
```

**3d. Update mailto validation block** (lines 657-669):

Replace lines 657-669 with the following. Note: the existing `build_source_overrides`
call at line 669 (`settings["source_overrides"] = build_source_overrides(settings)`)
is NOT replaced -- it remains in place after this block, so `settings["api_key"]`
(set below) will be available when `build_source_overrides` reads it:

```python
    # Validate authentication (mailto or API key)
    mailto_effective = args.mailto.strip() if args.mailto else (settings.get("mailto") or "")
    api_key = os.environ.get("OPENALEX_API_KEY", "").strip()

    if not mailto_effective and not api_key:
        logger.error("Authentication required for OpenAlex API.")
        logger.error("Provide one of:")
        logger.error("  - OPENALEX_API_KEY in .env file or environment")
        logger.error("  - --mailto YOUR_EMAIL")
        logger.error("  - Interactive setup: python run.py --configure")
        return None

    if mailto_effective:
        settings["mailto"] = mailto_effective
        if args.mailto:
            save_settings(settings)

    if api_key:
        settings["api_key"] = api_key
```

Steps 3a-3d cover all `run.py` changes. No further modifications needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_openalex_auth.py::test_build_source_overrides_includes_api_key -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 6: Lint and commit**

```bash
.venv/Scripts/python -m ruff check run.py --fix
git add run.py tests/test_openalex_auth.py
git commit -m "feat(FP-bii.5): update run.py to accept API key auth, load dotenv"
```

---

## Task 7: End-to-end verification (FP-bii.7)

**Depends on:** All previous tasks

**Files:**
- Modify: `tests/test_openalex_auth.py`

- [ ] **Step 1: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass (including all new tests from tasks 1-6)

- [ ] **Step 2: Verify API key is loaded from .env**

Run a quick Python check:

```bash
.venv/Scripts/python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.environ.get('OPENALEX_API_KEY', '')
print(f'API key loaded: {bool(key)} (length: {len(key)})')
"
```

Expected: `API key loaded: True (length: 22)`

- [ ] **Step 3: Verify backward compatibility (mailto only)**

Write and run integration test (add to `tests/test_openalex_auth.py`):

```python
def test_integration_mailto_only_path():
    """Full path: ingest with mailto only (no api_key) must work."""
    from unittest.mock import patch, MagicMock
    import pandas as pd
    from src.ingest import _read_one

    with patch("src.ingest.fetch_openalex") as mock_fetch:
        mock_fetch.return_value = []
        with patch("src.ingest.results_to_df") as mock_df:
            mock_df.return_value = pd.DataFrame()

            _read_one({
                "kind": "openalex",
                "entity": "works",
                "mailto": "test@example.com",
                "per_page": 200,
            })

            call_kwargs = mock_fetch.call_args.kwargs
            assert call_kwargs.get("mailto") == "test@example.com"
            assert call_kwargs.get("api_key") is None
```

- [ ] **Step 4: Verify CRISPR config structure matches PSC config**

```python
def test_crispr_config_structural_parity():
    """CRISPR datasource config must have same keys as PSC config."""
    psc = yaml.safe_load((PROJECT_ROOT / "config/datasources.yaml").read_text())
    crispr = yaml.safe_load((PROJECT_ROOT / "config/datasources_crispr.yaml").read_text())

    psc_keys = set(psc.keys())
    crispr_keys = set(crispr.keys())
    assert psc_keys == crispr_keys, f"Key mismatch: {psc_keys.symmetric_difference(crispr_keys)}"

    psc_primary_keys = set(psc["sources"]["primary"].keys())
    crispr_primary_keys = set(crispr["sources"]["primary"].keys())
    assert psc_primary_keys == crispr_primary_keys
```

- [ ] **Step 5: Run ALL tests including new verification tests**

Run: `.venv/Scripts/python -m pytest tests/ -x -q --tb=short -k "not test_core_group"`
Expected: All pass

- [ ] **Step 6: Lint everything**

Run: `.venv/Scripts/python -m ruff check src/ tests/ run.py --fix && .venv/Scripts/python -m ruff check src/ tests/ run.py`

- [ ] **Step 7: Final commit**

```bash
git add tests/test_openalex_auth.py
git commit -m "test(FP-bii.7): add end-to-end verification tests for CRISPR ingestion"
```

---

## Execution Order Summary

```
Task 1 (FP-bii.6) ──┐
Task 2 (FP-bii.1) ──┤
Task 3 (FP-bii.2) ──┼── independent, can run in parallel
Task 4 (FP-bii.3) ──┘
                     │
Task 5 (FP-bii.4) ──┤── depends on Task 4
                     │
Task 6 (FP-bii.5) ──┤── depends on Tasks 1, 4, 5
                     │
Task 7 (FP-bii.7) ──┘── depends on all above
```

## Post-Implementation

After all tasks pass:
1. Close epic FP-bii in beads
2. Push to remote
3. Update project memory with CRISPR domain configuration patterns
