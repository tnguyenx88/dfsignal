"""Time-series backtesting metrics and horizon-bucket evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .intervals import build_conformal_intervals, interval_metrics


HORIZON_BUCKETS: tuple[tuple[int, int, str], ...] = (
    (1, 4, "1-4"),
    (5, 8, "5-8"),
    (9, 13, "9-13"),
    (14, 26, "14-26"),
)


def _aligned_arrays(actual: Sequence[float], predicted: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(actual, dtype=float).reshape(-1)
    forecast = np.asarray(predicted, dtype=float).reshape(-1)
    if observed.size != forecast.size:
        raise ValueError("actual and predicted must have equal lengths")
    return observed, forecast


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    observed, forecast = _aligned_arrays(actual, predicted)
    return float(np.abs(observed - forecast).sum() / max(np.abs(observed).sum(), 1e-9))


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    observed, forecast = _aligned_arrays(actual, predicted)
    return float((forecast - observed).sum() / max(np.abs(observed).sum(), 1e-9))


def mase(actual: np.ndarray, predicted: np.ndarray, training: np.ndarray, season_length: int = 1) -> float:
    observed, forecast = _aligned_arrays(actual, predicted)
    train = np.asarray(training, dtype=float).reshape(-1)
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    if len(train) <= season_length:
        return float("nan")
    differences = np.abs(train[season_length:] - train[:-season_length])
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return float("nan")
    scale = float(differences.mean())
    return float(np.abs(observed - forecast).mean() / max(scale, 1e-9))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error over aligned forecast pairs."""

    observed, forecast = _aligned_arrays(actual, predicted)
    return float(np.sqrt(np.mean((observed - forecast) ** 2)))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compare forecast and actual direction relative to their first value."""

    observed, forecast = _aligned_arrays(actual, predicted)
    if len(observed) < 2:
        return float("nan")
    actual_direction = np.sign(np.diff(observed))
    predicted_direction = np.sign(np.diff(forecast))
    return float(np.mean(actual_direction == predicted_direction))


def horizon_bucket(horizon: int) -> str:
    """Map a positive forecast horizon to the configured evaluation bucket."""

    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    for lower, upper, label in HORIZON_BUCKETS:
        if lower <= horizon <= upper:
            return label
    return "27+"


def add_horizon_buckets(frame: pd.DataFrame, horizon_column: str = "horizon") -> pd.DataFrame:
    """Return evaluation rows with explicit ``1-4`` through ``14-26`` bins."""

    if horizon_column not in frame.columns:
        raise ValueError(f"evaluation frame is missing {horizon_column!r}")
    result = frame.copy()
    result["horizon_bucket"] = result[horizon_column].map(horizon_bucket)
    return result


def evaluate_forecast(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    training: Sequence[float] | None = None,
    season_length: int = 1,
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
    horizon: int | None = None,
) -> dict[str, float | str]:
    """Evaluate point forecasts and, when supplied, prediction intervals."""

    observed, forecast = _aligned_arrays(actual, predicted)
    metrics: dict[str, float | str] = {
        "wape": wape(observed, forecast),
        "bias": bias(observed, forecast),
        "rmse": rmse(observed, forecast),
        "directional_accuracy": directional_accuracy(observed, forecast),
    }
    if training is not None:
        metrics["mase"] = mase(observed, forecast, np.asarray(training, dtype=float), season_length)
    if horizon is not None:
        metrics["horizon_bucket"] = horizon_bucket(horizon)
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be supplied together")
    if lower is not None and upper is not None:
        interval = interval_metrics(observed, lower, upper)
        metrics["interval_coverage"] = interval["coverage"]
        metrics["interval_width"] = interval["width"]
        metrics["interval_n"] = interval["n"]
    return metrics


def _seasonal_calibration_residuals(training: np.ndarray, season_length: int) -> np.ndarray:
    if len(training) <= season_length:
        return np.empty(0, dtype=float)
    residuals = training[season_length:] - training[:-season_length]
    return residuals[np.isfinite(residuals)]


def rolling_backtest(
    frame: pd.DataFrame,
    horizon: int = 4,
    minimum_train: int = 52,
    *,
    season_length: int = 52,
    week_column: str = "week",
    target: str = "target",
    interval_coverage: float | None = None,
) -> pd.DataFrame:
    """Run expanding-window Seasonal Naive evaluation.

    The original point-metric columns remain available.  Each row now also
    carries a horizon bucket; callers can opt into residual-calibrated
    interval bounds and metrics with ``interval_coverage``.
    """

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if minimum_train < 1:
        raise ValueError("minimum_train must be positive")
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    missing = [column for column in (week_column, target) if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if interval_coverage is not None and not 0.0 < float(interval_coverage) < 1.0:
        raise ValueError("interval_coverage must be strictly between 0 and 1")

    ordered = frame.dropna(subset=[target]).sort_values(week_column).reset_index(drop=True)
    values = ordered[target].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for end in range(minimum_train, len(values) - horizon + 1, horizon):
        train, actual = values[:end], values[end : end + horizon]
        predicted = np.resize(
            train[-season_length:] if len(train) >= season_length else train[-1:],
            horizon,
        ).astype(float, copy=False)
        metrics = evaluate_forecast(
            actual,
            predicted,
            training=train,
            season_length=1,
            horizon=horizon,
        )
        row: dict[str, Any] = {
            "cutoff": ordered.iloc[end - 1][week_column],
            "horizon": horizon,
            "horizon_bucket": metrics.pop("horizon_bucket"),
            "model": "seasonal_naive",
            **metrics,
        }
        if interval_coverage is not None:
            residuals = _seasonal_calibration_residuals(train, season_length)
            if residuals.size:
                lower, upper = build_conformal_intervals(
                    predicted,
                    residuals,
                    float(interval_coverage),
                )
                interval = interval_metrics(actual, lower, upper)
                row.update(
                    {
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "interval_coverage": interval["coverage"],
                        "interval_width": interval["width"],
                        "interval_n": interval["n"],
                    }
                )
            else:
                row.update(
                    {
                        "lower_bound": np.full(horizon, np.nan),
                        "upper_bound": np.full(horizon, np.nan),
                        "interval_coverage": float("nan"),
                        "interval_width": float("nan"),
                        "interval_n": 0.0,
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_backtest_with_intervals(
    frame: pd.DataFrame,
    horizon: int = 4,
    minimum_train: int = 52,
    *,
    coverage: float = 0.9,
    season_length: int = 52,
    week_column: str = "week",
    target: str = "target",
) -> pd.DataFrame:
    """Explicit orchestration entry point for residual-calibrated evaluation."""

    return rolling_backtest(
        frame,
        horizon=horizon,
        minimum_train=minimum_train,
        season_length=season_length,
        week_column=week_column,
        target=target,
        interval_coverage=coverage,
    )


def summarize_horizon_buckets(
    evaluation: pd.DataFrame,
    *,
    horizon_column: str = "horizon",
) -> pd.DataFrame:
    """Aggregate numeric metrics by the fixed horizon buckets."""

    bucketed = add_horizon_buckets(evaluation, horizon_column)
    numeric = bucketed.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [column for column in numeric if column != horizon_column]
    if not numeric:
        return bucketed[["horizon_bucket"]].drop_duplicates().reset_index(drop=True)
    return bucketed.groupby("horizon_bucket", as_index=False, observed=True)[numeric].mean()
