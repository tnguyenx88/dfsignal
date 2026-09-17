"""Deterministic synthetic internal demand for the PoC."""

import numpy as np
import pandas as pd


def generate_synthetic_demand(start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Generate clearly labelled weekly sample demand without company data."""

    dates = pd.date_range(start, end, freq="W-SAT")
    rng = np.random.default_rng(seed)
    index = np.arange(len(dates))
    seasonal = 18 * np.sin(2 * np.pi * dates.isocalendar().week.to_numpy() / 52)
    trend = index * 0.12
    demand = np.maximum(10, 120 + trend + seasonal + rng.normal(0, 5, len(dates)))
    return pd.DataFrame(
        {
            "week": dates.date,
            "business_family": "Memory / PC Components",
            "demand_qty": demand.round(3),
            "demand_revenue": (demand * 85).round(2),
            "region": "Global",
            "channel": "Synthetic",
            "source": "SYNTHETIC",
        }
    )
