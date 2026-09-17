"""Manual product launch event adapter."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"event_date", "event_type", "vendor", "product_family", "source", "confidence"}
EVENT_STATUSES = {"REVIEWED", "AUTO_DERIVED", "REVIEW", "EXAMPLE", "DISABLED"}


def _is_example_event(frame: pd.DataFrame) -> pd.Series:
    """Identify explicit placeholders without deleting manual-event support."""
    text = frame.astype("string").fillna("").agg(" ".join, axis=1).str.casefold()
    explicit = (
        frame.get("is_example_event", pd.Series(False, index=frame.index))
        .astype("string")
        .str.casefold()
        .isin({"true", "1", "yes", "y"})
    )
    return explicit | text.str.contains(r"\b(?:example|placeholder|generic)\b", regex=True)


def read_launch_events(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Launch events missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    if frame["event_date"].isna().any():
        raise ValueError("Launch events contain invalid event_date values")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="raise").clip(0, 1)
    example = _is_example_event(frame)
    frame["is_example_event"] = example.astype("boolean")
    if "event_status" not in frame.columns:
        frame["event_status"] = "REVIEWED"
    frame["event_status"] = frame["event_status"].astype("string").str.upper().fillna("REVIEW")
    frame.loc[example, "event_status"] = "EXAMPLE"
    frame.loc[~frame["event_status"].isin(EVENT_STATUSES), "event_status"] = "REVIEW"
    return frame

def usable_launch_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Return reviewed/derived manual events, excluding examples and disabled rows."""
    if frame.empty:
        return frame.copy()
    status = frame.get("event_status", pd.Series("REVIEWED", index=frame.index)).astype("string").str.upper()
    return frame.loc[
        frame.get("is_example_event", pd.Series(False, index=frame.index)).astype("boolean").fillna(False).eq(False)
        & status.isin({"REVIEWED", "AUTO_DERIVED"})
    ].copy()
