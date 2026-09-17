"""Residual-based prediction intervals and interval evaluation metrics.

Intervals in this module are calibrated from observed forecast residuals.  No
fixed percentage band is used: the radius is a finite-sample conformal-style
quantile of absolute residuals supplied by the caller.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


ArrayLike = Sequence[float] | np.ndarray


def _as_vector(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _finite_residuals(residuals: ArrayLike) -> np.ndarray:
    values = _as_vector(residuals, "residuals")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("residuals must contain at least one finite value")
    return values


def conformal_radius(residuals: ArrayLike, coverage: float = 0.9) -> float:
    """Return a finite-sample conformal-style absolute-residual radius.

    The ``higher`` quantile is deliberately conservative for small calibration
    sets.  With ``n`` residuals, the selected rank is
    ``ceil((n + 1) * coverage)`` capped at ``n``.
    """

    if not isinstance(coverage, (float, int, np.floating, np.integer)):
        raise TypeError("coverage must be a number between 0 and 1")
    coverage = float(coverage)
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between 0 and 1")
    absolute = np.abs(_finite_residuals(residuals))
    rank = min(absolute.size, max(1, math.ceil((absolute.size + 1) * coverage)))
    quantile = (rank - 1) / absolute.size
    return float(np.quantile(absolute, quantile, method="higher"))


def build_conformal_intervals(
    point_forecast: ArrayLike,
    residuals: ArrayLike,
    coverage: float = 0.9,
    *,
    nonnegative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric residual/conformal-style intervals around forecasts.

    ``residuals`` should be generated from an earlier, history-only
    calibration window.  The optional ``nonnegative`` constraint is explicit;
    it clips only the lower bound for quantities that cannot be negative.
    """

    point = _as_vector(point_forecast, "point_forecast")
    if not np.isfinite(point).all():
        raise ValueError("point_forecast must contain only finite values")
    radius = conformal_radius(residuals, coverage)
    lower = point - radius
    upper = point + radius
    if nonnegative:
        lower = np.maximum(lower, 0.0)
    return lower, upper


def residual_conformal_interval(
    point_forecast: ArrayLike,
    residuals: ArrayLike,
    coverage: float = 0.9,
    *,
    nonnegative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit alias for :func:`build_conformal_intervals`."""

    return build_conformal_intervals(
        point_forecast,
        residuals,
        coverage,
        nonnegative=nonnegative,
    )


def interval_coverage(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> float:
    """Measure the fraction of finite actuals contained in an interval."""

    observed = _as_vector(actual, "actual")
    lower_values = _as_vector(lower, "lower")
    upper_values = _as_vector(upper, "upper")
    if not (len(observed) == len(lower_values) == len(upper_values)):
        raise ValueError("actual, lower, and upper must have equal lengths")
    finite = np.isfinite(observed) & np.isfinite(lower_values) & np.isfinite(upper_values)
    if not finite.any():
        return float("nan")
    if np.any(lower_values[finite] > upper_values[finite]):
        raise ValueError("lower bounds must not exceed upper bounds")
    covered = (observed[finite] >= lower_values[finite]) & (observed[finite] <= upper_values[finite])
    return float(np.mean(covered))


def interval_width(lower: ArrayLike, upper: ArrayLike) -> float:
    """Measure mean finite interval width."""

    lower_values = _as_vector(lower, "lower")
    upper_values = _as_vector(upper, "upper")
    if len(lower_values) != len(upper_values):
        raise ValueError("lower and upper must have equal lengths")
    finite = np.isfinite(lower_values) & np.isfinite(upper_values)
    if not finite.any():
        return float("nan")
    if np.any(lower_values[finite] > upper_values[finite]):
        raise ValueError("lower bounds must not exceed upper bounds")
    return float(np.mean(upper_values[finite] - lower_values[finite]))


def interval_metrics(actual: ArrayLike, lower: ArrayLike, upper: ArrayLike) -> dict[str, float]:
    """Return coverage and mean width for one forecast interval set."""

    observed = _as_vector(actual, "actual")
    lower_values = _as_vector(lower, "lower")
    upper_values = _as_vector(upper, "upper")
    if not (len(observed) == len(lower_values) == len(upper_values)):
        raise ValueError("actual, lower, and upper must have equal lengths")
    finite = np.isfinite(observed) & np.isfinite(lower_values) & np.isfinite(upper_values)
    return {
        "coverage": interval_coverage(observed, lower_values, upper_values),
        "width": interval_width(lower_values, upper_values),
        "n": float(finite.sum()),
    }


# Names that read naturally in orchestration code.
conformal_interval = build_conformal_intervals
mean_interval_width = interval_width
