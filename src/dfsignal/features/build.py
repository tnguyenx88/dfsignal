"""Leakage-safe weekly feature assembly."""

import pandas as pd


def build_features(demand: pd.DataFrame, external: pd.DataFrame | None = None, target_column: str = "pos_qty") -> pd.DataFrame:
    """Join canonical demand and already-normalized external weekly signals."""

    frame = demand.sort_values("week").copy()
    frame["week"] = pd.to_datetime(frame["week"])
    if target_column not in frame.columns:
        raise ValueError(f"Configured target column not present: {target_column}")
    frame["target"] = pd.to_numeric(frame[target_column], errors="raise")
    frame["week_of_year"] = frame["week"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["week"].dt.month
    frame["quarter"] = frame["week"].dt.quarter
    frame["christmas"] = frame["week"].dt.month.eq(12).astype(int)
    frame["black_friday"] = ((frame["week"].dt.month == 11) & (frame["week"].dt.day >= 22)).astype(int)
    frame["cyber_monday"] = frame["black_friday"]
    frame["back_to_school"] = ((frame["week"].dt.month == 8) & (frame["week"].dt.day >= 7)).astype(int)
    for lag in (1, 2, 4, 8, 13, 26, 52):
        frame[f"lag_{lag}"] = frame["target"].shift(lag)
    for window in (4, 8, 13, 26, 52):
        frame[f"rolling_mean_{window}"] = frame["target"].shift(1).rolling(window).mean()
    for window in (4, 13):
        frame[f"rolling_std_{window}"] = frame["target"].shift(1).rolling(window).std()
    frame["wow"] = frame["target"].pct_change().shift(1)
    frame["yoy"] = frame["target"].pct_change(52).shift(1)
    if external is not None and not external.empty:
        normalized = external.copy()
        normalized["week"] = pd.to_datetime(normalized["week"])
        frame = frame.merge(normalized, on="week", how="left", validate="one_to_one")
    # DuckDB-backed Parquet reads may use Arrow extension dtypes; estimators
    # require ordinary NumPy numeric arrays at the model boundary.
    for column in frame.select_dtypes(include=["number"]).columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame
