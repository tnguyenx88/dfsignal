"""Normalize daily or monthly observations to week-ending Saturday."""

import pandas as pd


def to_week_saturday(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    return (dates + pd.to_timedelta((5 - dates.dt.dayofweek) % 7, unit="D")).dt.date


def aggregate_weekly(frame: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["week"] = pd.to_datetime(to_week_saturday(normalized[date_column]))
    return normalized.groupby("week", as_index=False)[value_column].mean()
