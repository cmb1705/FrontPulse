"""DataFrame transformation utilities for adding derived time columns."""
from __future__ import annotations
import pandas as pd


def add_time_vars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived time columns (pub_year, pub_qtr) to a DataFrame.

    Args:
        df: DataFrame with 'publication_date' column

    Returns:
        Copy of DataFrame with added columns:
            - pub_year: integer year
            - pub_qtr: string quarter label (e.g., "2020Q3")

    Example:
        >>> df = pd.DataFrame({"publication_date": ["2020-07-15"]})
        >>> df = add_time_vars(df)
        >>> df["pub_qtr"].iloc[0]
        '2020Q3'
    """
    df = df.copy()
    df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    df["pub_year"] = df["publication_date"].dt.year
    q = df["publication_date"].dt.quarter
    y = df["publication_date"].dt.year
    df["pub_qtr"] = (y.astype("Int64").astype(str) + "Q" + q.astype("Int64").astype(str)).astype("string")
    return df
