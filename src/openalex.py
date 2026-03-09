"""OpenAlex API client for fetching and parsing bibliometric data."""
from __future__ import annotations
import time
from typing import Dict, Any, Optional, List
import requests
import pandas as pd
from tqdm import tqdm

BASE: str = "https://api.openalex.org"


def _kv_to_filter(filters: Dict[str, Any]) -> str:
    """Convert filter dictionary to OpenAlex filter string format."""
    parts = []
    for k, v in (filters or {}).items():
        if v is None or v == "":
            continue
        if isinstance(v, (list, tuple, set)):
            parts.append(f"{k}:{'|'.join(str(x) for x in v)}")
        else:
            parts.append(f"{k}:{v}")
    return ",".join(parts)

def fetch_openalex(
    entity: str,
    *,
    mailto: str,
    filters: Optional[Dict[str, Any]] = None,
    search: Optional[str] = None,
    select: Optional[List[str]] = None,
    sort: Optional[str] = None,
    per_page: int = 200,
    max_records: Optional[int] = None,
    sleep_s: float = 0.12,
) -> List[Dict[str, Any]]:
    """
    Fetch entities from OpenAlex API using cursor pagination.

    Args:
        entity: OpenAlex entity type (e.g., "works", "authors", "institutions")
        mailto: Contact email for polite pool access
        filters: Dictionary of filter parameters (e.g., {"topics.id": "T10247"})
        search: Search query string
        select: List of fields to return (reduces response size)
        sort: Sort parameter (e.g., "publication_date:desc")
        per_page: Results per page (max 200)
        max_records: Maximum total records to fetch (None for unlimited)
        sleep_s: Sleep duration between requests (default 0.12s for polite pool)

    Returns:
        List of entity dictionaries from OpenAlex

    Raises:
        RuntimeError: If API returns 403 Forbidden
        requests.HTTPError: For other HTTP errors

    Example:
        >>> results = fetch_openalex(
        ...     entity="works",
        ...     mailto="researcher@university.edu",
        ...     filters={"topics.id": "T10247"},
        ...     max_records=1000
        ... )
    """
    params = {"per-page": per_page, "cursor": "*", "mailto": mailto}
    if search:
        params["search"] = search
    filt = _kv_to_filter(filters or {})
    if filt:
        params["filter"] = filt
    if select:
        params["select"] = ",".join(select)
    if sort:
        params["sort"] = sort

    url = f"{BASE}/{entity}"
    out: List[Dict[str, Any]] = []
    sess = requests.Session()
    headers = {
        "User-Agent": f"2YP-RF-Ingest/0.1 (+mailto:{mailto})",
        "Accept": "application/json",
    }

    total = None
    pbar = None
    max_retries = 5
    base_retry_delay = 2.0

    while True:
        retry_count = 0
        while retry_count < max_retries:
            try:
                r = sess.get(url, params=params, headers=headers, timeout=60)
                if r.status_code == 403:
                    raise RuntimeError(
                        f"403 Forbidden from OpenAlex. URL={r.url}\nResponse text: {r.text[:500]}"
                    )
                if r.status_code == 500:
                    # Retry 500 errors with exponential backoff
                    retry_count += 1
                    if retry_count >= max_retries:
                        r.raise_for_status()  # Give up after max retries
                    retry_delay = base_retry_delay * (2 ** (retry_count - 1))
                    if pbar:
                        pbar.write(f"OpenAlex 500 error, retrying in {retry_delay:.1f}s (attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                r.raise_for_status()
                break  # Success, exit retry loop
            except requests.exceptions.RequestException as e:
                # For non-HTTP errors, retry with backoff
                retry_count += 1
                if retry_count >= max_retries:
                    raise
                retry_delay = base_retry_delay * (2 ** (retry_count - 1))
                if pbar:
                    pbar.write(f"Request error: {e}, retrying in {retry_delay:.1f}s (attempt {retry_count}/{max_retries})")
                time.sleep(retry_delay)

        data = r.json()

        if total is None:
            # meta.count may be None for some queries
            total = (data.get("meta") or {}).get("count")
            pbar = tqdm(total=total or 0, unit="work", desc="OpenAlex", disable=(total is None))

        results = data.get("results") or []
        out.extend(results)
        if pbar:
            pbar.update(len(results))

        if max_records is not None and len(out) >= max_records:
            out = out[:max_records]
            if pbar:
                pbar.close()
            break

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor or not results:
            if pbar:
                pbar.close()
            break
        params["cursor"] = next_cursor
        time.sleep(sleep_s)
    return out

def flatten_work(w: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten nested OpenAlex 'works' entity into a flat dictionary.

    Extracts key fields including:
    - Work identifiers (work_id, DOI)
    - Publication metadata (title, date, type, language)
    - Citation metrics
    - Open access status
    - Primary venue
    - Author information (first author, all author IDs)
    - Institution affiliations and international collaboration flag
    - Primary topic classification
    - Referenced works list

    Args:
        w: Raw OpenAlex work dictionary

    Returns:
        Flattened dictionary with normalized field names
    """
    row: Dict[str, Any] = {}
    row["work_id"] = (w.get("id") or "").split("/")[-1] or None
    row["work_id_url"] = w.get("id")
    row["doi"] = w.get("doi")
    row["title"] = w.get("display_name")
    row["publication_year"] = w.get("publication_year")
    row["publication_date"] = w.get("publication_date") or (
        f"{int(w['publication_year'])}-01-01" if w.get("publication_year") else None
    )
    row["type"] = w.get("type")
    row["language"] = w.get("language")
    row["cited_by_count"] = w.get("cited_by_count")

    oa = w.get("open_access") or {}
    row["is_oa"] = oa.get("is_oa")
    row["oa_status"] = oa.get("oa_status")
    row["oa_url"] = oa.get("oa_url")
    row["has_fulltext"] = w.get("has_fulltext")

    pl = w.get("primary_location") or {}
    src = pl.get("source") or {}
    row["primary_venue_id"] = src.get("id")
    row["primary_venue_name"] = src.get("display_name")
    row["primary_venue_type"] = src.get("type")

    authorships = w.get("authorships") or []
    row["author_count"] = len(authorships)
    author_ids: list[str] = []
    if authorships:
        a0 = authorships[0] or {}
        a0_author = a0.get("author") or {}
        row["first_author_id"] = a0_author.get("id")
        row["first_author_name"] = a0_author.get("display_name")
        row["first_author_orcid"] = a0_author.get("orcid")
    for a in authorships:
        author = a.get("author") or {}
        aid = author.get("id")
        if aid:
            author_ids.append(aid)
    inst_ids, inst_ccs = set(), set()
    for a in authorships:
        for inst in (a.get("institutions") or []):
            iid = inst.get("id"); cc = inst.get("country_code")
            if iid: inst_ids.add(iid)
            if cc: inst_ccs.add(cc)
    row["author_ids"] = ",".join(author_ids) if author_ids else None
    row["institution_ids"] = ",".join(sorted(inst_ids)) if inst_ids else None
    row["institution_country_codes"] = ",".join(sorted(inst_ccs)) if inst_ccs else None
    row["is_international_collab"] = (len(inst_ccs) > 1) if inst_ccs else None

    best_t = None
    for t in (w.get("topics") or []):
        if best_t is None or (t.get("score") or 0) > (best_t.get("score") or 0):
            best_t = t
    if best_t:
        row["primary_topic_id"] = best_t.get("id")
        row["primary_topic_name"] = best_t.get("display_name")
        dom = (best_t.get("domain") or {})
        fld = (best_t.get("field") or {})
        row["primary_topic_domain"] = dom.get("display_name")
        row["primary_topic_field"] = fld.get("display_name")

    refs = w.get("referenced_works") or []
    norm_refs = []
    for r in refs:
        norm_refs.append(r.split("/")[-1] if isinstance(r, str) else str(r))
    row["referenced_works"] = norm_refs
    row["ref_count"] = len(norm_refs)
    return row

def results_to_df(entity: str, results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert OpenAlex API results to a pandas DataFrame.

    For 'works' entities, applies custom flattening via flatten_work().
    For other entities, uses pandas json_normalize().

    Args:
        entity: Entity type (e.g., "works", "authors")
        results: List of entity dictionaries from OpenAlex API

    Returns:
        DataFrame with normalized column names (lowercase, underscores)

    Example:
        >>> df = results_to_df("works", fetch_openalex(...))
    """
    if entity != "works":
        return pd.json_normalize(results, max_level=2)
    rows = [flatten_work(w) for w in results]
    df = pd.DataFrame(rows)
    df.columns = (
        df.columns.str.strip()
                  .str.replace(" ", "_")
                  .str.replace("-", "_")
                  .str.lower()
    )
    return df
