from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def normalize_quarter(label: str) -> str:
    """Normalize strings like '2020Q1', '2020-Q1', or '2020 Q1' to '2020Q1'."""
    if not isinstance(label, str):
        return "0000Q1"
    clean = label.strip().upper().replace(" ", "").replace("-", "")
    if "Q" in clean:
        year, q = clean.split("Q", 1)
    else:
        year, q = clean[:-1], clean[-1]
    year = year.zfill(4)
    q = ''.join(ch for ch in q if ch.isdigit())[:1] or "1"
    return f"{year}Q{q}"


def quarter_key(q: str) -> Tuple[int, int]:
    normalized = normalize_quarter(q)
    return int(normalized[:4]), int(normalized[-1])


def quarter_to_int(label: str) -> int:
    normalized = normalize_quarter(label)
    year = int(normalized[:4])
    quarter = int(normalized[-1])
    return year * 4 + (quarter - 1)


def int_to_quarter(value: int) -> str:
    year = value // 4
    quarter = value % 4 + 1
    return f"{year}Q{quarter}"


def describe_quarter_range(start: Optional[str], end: Optional[str]) -> str:
    if start and end:
        return f"{normalize_quarter(start)} to {normalize_quarter(end)}"
    if start:
        return f"{normalize_quarter(start)} onwards"
    if end:
        return f"through {normalize_quarter(end)}"
    return "full history"


def _range_slug(start: Optional[str], end: Optional[str]) -> str:
    def normalize(value: Optional[str], fallback: str) -> str:
        if not value:
            return fallback
        return normalize_quarter(value).lower()

    return f"{normalize(start, 'min')}_{normalize(end, 'max')}"


def snapshot_dataset(df: pd.DataFrame, directory: Path, prefix: str, start: Optional[str], end: Optional[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    slug = _range_slug(start, end)
    path = directory / f"{prefix}_{slug}.parquet"
    safe_df = df.copy()
    object_cols = safe_df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        safe_df[object_cols] = safe_df[object_cols].astype(str)
    safe_df.to_parquet(path, index=False)
    print(f"   Saved {prefix} snapshot ({len(df):,} rows) -> {path}")


def filter_by_quarter(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    label: str = "dataset",
) -> pd.DataFrame:
    if not start and not end:
        print(f"   {label}: {len(df):,} rows (full history)")
        return df.copy()

    quarter_idx = df["quarter"].astype(str).apply(quarter_to_int)
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= quarter_idx >= quarter_to_int(start)
    if end:
        mask &= quarter_idx <= quarter_to_int(end)

    filtered = df[mask].copy()
    min_q = filtered["quarter"].min() if not filtered.empty else "N/A"
    max_q = filtered["quarter"].max() if not filtered.empty else "N/A"
    print(
        f"   {label}: {len(filtered):,} rows "
        f"({describe_quarter_range(start, end)}, {min_q} to {max_q})"
    )
    return filtered
