"""FRED CSV adapter with bounded retries."""

from io import StringIO
import time

import httpx
import pandas as pd


def fetch_fred_series(
    series_id: str,
    base_url: str,
    start_date: str,
    end_date: str | None = None,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch one configured FRED series and return normalized observations."""

    params = {"id": series_id, "cosd": start_date}
    if end_date:
        params["coed"] = end_date
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = httpx.get(base_url, params=params, timeout=30.0)
            response.raise_for_status()
            frame = pd.read_csv(StringIO(response.text))
            frame.columns = ["observation_date", "value"]
            frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            return frame.dropna(subset=["value"])
        except (httpx.HTTPError, ValueError, pd.errors.ParserError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.25 * (2**attempt))
    raise RuntimeError(f"FRED series fetch failed: {series_id}") from last_error
