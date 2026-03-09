#!/usr/bin/env python3
"""Helper utilities for enforcing multi-quarter persistence on detections."""

from __future__ import annotations

import pandas as pd


def ensure_persistence_column(
    frame: pd.DataFrame,
    *,
    threshold: float,
    window: int = 2,
    prob_col: str = "inflection_probability",
    id_col: str = "lineage_id",
    quarter_col: str = "quarter",
    column_name: str = "is_inflection_pred_persistent",
    force: bool = False,
) -> pd.DataFrame:
    """
    Ensure `frame` contains a persistence column based on consecutive quarters.

    A row is marked persistent when the current probability ≥ threshold AND
    the rolling-`window` mean of the probability also ≥ threshold.
    """
    if not force and column_name in frame.columns and not frame[column_name].isna().all():
        return frame

    if window <= 1:
        frame[column_name] = (frame[prob_col] >= threshold).astype(int)
        return frame

    working = frame.copy()
    working["_quarter_timestamp"] = pd.PeriodIndex(working[quarter_col].astype(str), freq="Q").to_timestamp()
    working = working.sort_values([id_col, "_quarter_timestamp"])
    rolling_mean = working.groupby(id_col, group_keys=False)[prob_col].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    persistent_mask = (working[prob_col] >= threshold) & (rolling_mean >= threshold)
    working[column_name] = persistent_mask.astype(int)
    working = working.sort_index()
    frame[column_name] = working[column_name]
    return frame
