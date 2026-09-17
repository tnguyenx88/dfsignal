"""Forecast models and explicit future-forecast helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .metadata import ModelRunMetadata


EXCLUDED_FEATURES = {
    "week",
    "business_family",
    "target",
    "pos_qty",
    "demand_qty",
    "demand_revenue",
    "region",
    "channel",
    "source",
    "source_id",
}
DEFAULT_FUTURE_HORIZONS = (4, 8, 13, 26)


@dataclass(frozen=True, slots=True)
class ETSForecastResult:
    """ETS output plus fit diagnostics needed for safe orchestration."""

    forecast: np.ndarray
    converged: bool
    warning: str | None
    configuration: Mapping[str, Any]
    training_length: int
    used_fallback: bool
    model_warning: bool
    metadata: ModelRunMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "forecast", np.asarray(self.forecast, dtype=float))
        object.__setattr__(self, "configuration", dict(self.configuration))

    @property
    def fallback_used(self) -> bool:
        """Compatibility alias for callers that phrase the state as a noun."""

        return self.used_fallback


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)):
        raise TypeError("horizon must be an integer")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    return int(horizon)


def seasonal_naive(history: pd.Series, horizon: int, season_length: int = 52) -> np.ndarray:
    """Forecast by repeating the latest seasonal cycle or latest value."""

    horizon = _validate_horizon(horizon)
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = pd.Series(history).dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError("history must contain at least one non-null observation")
    if horizon == 0:
        return np.empty(0, dtype=float)
    if len(values) < season_length:
        return np.repeat(values[-1], horizon)
    return np.resize(values[-season_length:], horizon).astype(float, copy=False)


def _warning_text(captured: list[warnings.WarningMessage]) -> str | None:
    messages: list[str] = []
    for item in captured:
        text = str(item.message).strip()
        if not text:
            continue
        category = getattr(item.category, "__name__", "")
        rendered = f"{category}: {text}" if category else text
        if rendered not in messages:
            messages.append(rendered)
    return "; ".join(messages) or None


def _convergence_status(result: Any) -> bool:
    """Read optimizer status without depending on a statsmodels version."""

    retvals = getattr(result, "mle_retvals", None)
    if retvals is None:
        return True
    getter = getattr(retvals, "get", None)
    if getter is None:
        return True
    converged = getter("converged", None)
    if converged is not None:
        return bool(converged)
    success = getter("success", None)
    if success is not None:
        return bool(success)
    message = str(getter("message", "")).lower()
    if "converg" in message:
        return True
    return True


def _history_values(history: pd.Series) -> np.ndarray:
    values = pd.Series(history).dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("history contains non-finite observations")
    return values


def _ets_configuration(values: np.ndarray, season_length: int) -> dict[str, Any]:
    seasonal_periods = season_length if len(values) >= season_length * 2 else None
    return {
        "trend": "add",
        "seasonal": "add" if seasonal_periods is not None else None,
        "seasonal_periods": seasonal_periods,
        "season_length": season_length,
        "damped_trend": False,
        "initialization_method": "estimated",
    }


def _temporal_bounds(history: pd.Series) -> tuple[Any, Any]:
    indexed = pd.Series(history).dropna()
    if indexed.empty or not isinstance(indexed.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        return None, None
    index = indexed.index
    if isinstance(index, pd.PeriodIndex):
        index = index.to_timestamp()
    return index[0], index[-1]


def _result_metadata(
    history: pd.Series,
    *,
    model_name: str,
    target: str,
    horizon: int,
    parameters: Mapping[str, Any],
    converged: bool,
    warning: str | None,
    forecast_origin: Any = None,
) -> ModelRunMetadata:
    start, end = _temporal_bounds(history)
    return ModelRunMetadata(
        model_name=model_name,
        target=target,
        training_start=start,
        training_end=end,
        forecast_origin=forecast_origin if forecast_origin is not None else end,
        horizon=horizon,
        parameters=parameters,
        converged=converged,
        warning=warning,
    )


def ets_forecast_with_metadata(
    history: pd.Series,
    horizon: int,
    season_length: int = 52,
    *,
    target: str | None = None,
    forecast_origin: Any = None,
) -> ETSForecastResult:
    """Fit ETS while returning local warning and fallback diagnostics.

    Warnings are captured only inside the fit/forecast call and returned to the
    caller.  The process-wide warning filters are never changed.  A failed or
    non-convergent fit uses Seasonal Naive as an explicit fallback; a
    convergent fit that emitted a warning keeps its forecast but marks
    ``model_warning=True``.
    """

    horizon = _validate_horizon(horizon)
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    values = _history_values(history)
    configuration = _ets_configuration(values, season_length)
    warning_text: str | None = None
    converged = True
    used_fallback = False

    if len(values) < 2:
        warning_text = "ETS requires at least two non-null observations; seasonal_naive fallback used."
        converged = False
        used_fallback = True
        prediction = seasonal_naive(pd.Series(values), horizon, season_length)
    else:
        fitted: Any | None = None
        caught: list[warnings.WarningMessage] = []
        try:
            # This context is deliberately local: callers still receive the
            # diagnostics, while unrelated code retains its warning policy.
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                fitted = ExponentialSmoothing(
                    values,
                    trend=configuration["trend"],
                    seasonal=configuration["seasonal"],
                    seasonal_periods=configuration["seasonal_periods"],
                    damped_trend=configuration["damped_trend"],
                    initialization_method=configuration["initialization_method"],
                ).fit()
                prediction = np.asarray(fitted.forecast(horizon), dtype=float)
                caught = captured
            warning_text = _warning_text(caught)
            converged = _convergence_status(fitted)
            if not converged:
                warning_text = (
                    f"{warning_text}; ETS optimizer did not converge"
                    if warning_text
                    else "ETS optimizer did not converge"
                )
                used_fallback = True
                prediction = seasonal_naive(pd.Series(values), horizon, season_length)
        except Exception as exc:
            # Fallback is intentional recovery from a model-fit failure; the
            # exception text is retained as a visible model warning.
            warning_text = f"ETS fit failed ({type(exc).__name__}): {exc}; seasonal_naive fallback used."
            converged = False
            used_fallback = True
            prediction = seasonal_naive(pd.Series(values), horizon, season_length)

    model_warning = bool(warning_text) or used_fallback or not converged
    parameters = dict(configuration)
    metadata = _result_metadata(
        history,
        model_name="ets",
        target=target or getattr(history, "name", None) or "target",
        horizon=horizon,
        parameters=parameters,
        converged=converged,
        warning=warning_text,
        forecast_origin=forecast_origin,
    )
    return ETSForecastResult(
        forecast=prediction,
        converged=converged,
        warning=warning_text,
        configuration=configuration,
        training_length=len(values),
        used_fallback=used_fallback,
        model_warning=model_warning,
        metadata=metadata,
    )


def ets_forecast_detailed(
    history: pd.Series,
    horizon: int,
    season_length: int = 52,
    *,
    target: str | None = None,
    forecast_origin: Any = None,
) -> ETSForecastResult:
    """Alias with an explicit name for orchestration integrations."""

    return ets_forecast_with_metadata(
        history,
        horizon,
        season_length,
        target=target,
        forecast_origin=forecast_origin,
    )


def ets_forecast(history: pd.Series, horizon: int, season_length: int = 52) -> np.ndarray:
    """Backward-compatible ETS API returning only point forecasts."""

    return ets_forecast_with_metadata(history, horizon, season_length).forecast.copy()


def lightgbm_forecast(train: pd.DataFrame, future: pd.DataFrame, target: str = "target") -> np.ndarray:
    feature_columns = [c for c in train.columns if c not in EXCLUDED_FEATURES]
    if not feature_columns:
        raise ValueError("LightGBM requires at least one non-target feature column")
    clean_train = train.dropna(subset=feature_columns + [target])
    if clean_train.empty:
        observed = train[target].dropna()
        if observed.empty:
            raise ValueError("train must contain at least one non-null target")
        return np.repeat(observed.iloc[-1], len(future)).astype(float)
    model = LGBMRegressor(n_estimators=200, learning_rate=0.04, max_depth=4, random_state=7, verbosity=-1)
    model.fit(clean_train[feature_columns], clean_train[target])
    missing = [column for column in feature_columns if column not in future.columns]
    if missing:
        raise ValueError(f"future is missing LightGBM features: {missing}")
    return np.asarray(
        model.predict(future[feature_columns].fillna(clean_train[feature_columns].median())),
        dtype=float,
    )


def _observed_frame(
    history: pd.DataFrame | pd.Series,
    *,
    week_column: str,
    target: str,
) -> pd.DataFrame:
    if isinstance(history, pd.Series):
        if not isinstance(history.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise ValueError("Series history must have a DatetimeIndex or PeriodIndex for future forecasts")
        index = history.index.to_timestamp() if isinstance(history.index, pd.PeriodIndex) else history.index
        frame = pd.DataFrame({week_column: pd.to_datetime(index), target: history.to_numpy()})
    else:
        missing = [column for column in (week_column, target) if column not in history.columns]
        if missing:
            raise ValueError(f"history is missing required columns: {missing}")
        frame = history.copy()
        frame[week_column] = pd.to_datetime(frame[week_column], errors="raise")

    frame = frame.dropna(subset=[target]).sort_values(week_column).reset_index(drop=True)
    if frame.empty:
        raise ValueError("history must contain at least one non-null target")
    if frame[week_column].duplicated().any():
        raise ValueError("history must contain at most one observed row per week")
    return frame


def _future_weeks(latest_week: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        latest_week + pd.to_timedelta(np.arange(1, horizon + 1), unit="W"),
        name="week",
    )


def future_forecast(
    history: pd.DataFrame | pd.Series,
    horizon: int = 4,
    *,
    model_name: str = "ets",
    week_column: str = "week",
    target: str = "target",
    season_length: int = 52,
    future_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast weeks strictly after the latest observed target week.

    The helper generates weekly labels rather than reusing rows from a
    holdout.  LightGBM requires caller-supplied future feature values because
    unknown future covariates must not be fabricated.
    """

    horizon = _validate_horizon(horizon)
    if horizon == 0:
        return pd.DataFrame(
            columns=[
                week_column,
                "model_name",
                "target",
                "horizon",
                "lead_week",
                "point_forecast",
                "run_id",
                "converged",
                "warning",
                "model_warning",
                "used_fallback",
            ]
        )
    observed = _observed_frame(history, week_column=week_column, target=target)
    latest_week = pd.Timestamp(observed[week_column].iloc[-1])
    weeks = _future_weeks(latest_week, horizon)
    model_key = model_name.lower().strip()
    model_warning: str | None = None
    used_fallback = False
    converged = True
    observed_series = observed.set_index(week_column)[target]
    if model_key == "seasonal_naive":
        prediction = seasonal_naive(observed_series, horizon, season_length)
        metadata = _result_metadata(
            observed_series,
            model_name=model_key,
            target=target,
            horizon=horizon,
            parameters={"season_length": season_length},
            converged=True,
            warning=None,
            forecast_origin=latest_week,
        )
    elif model_key == "ets":
        ets_result = ets_forecast_with_metadata(
            observed_series,
            horizon,
            season_length,
            target=target,
            forecast_origin=latest_week,
        )
        prediction = ets_result.forecast
        metadata = ets_result.metadata
        if metadata is None:
            raise RuntimeError("ETS result did not include run metadata")
        converged = ets_result.converged
        model_warning = ets_result.warning
        used_fallback = ets_result.used_fallback
    elif model_key == "lightgbm":
        if future_features is None:
            raise ValueError("future_features are required for true future LightGBM forecasts")
        future = future_features.copy()
        if week_column not in future.columns:
            raise ValueError(f"future_features is missing required column: {week_column}")
        future[week_column] = pd.to_datetime(future[week_column], errors="raise")
        future = future.sort_values(week_column).reset_index(drop=True)
        if not future[week_column].equals(pd.Series(weeks, name=week_column)):
            raise ValueError("future_features weeks must exactly match weeks after the latest observation")
        prediction = lightgbm_forecast(observed, future, target=target)
        metadata = _result_metadata(
            observed_series,
            model_name=model_key,
            target=target,
            horizon=horizon,
            parameters={"feature_columns": [c for c in observed.columns if c not in EXCLUDED_FEATURES]},
            converged=True,
            warning=None,
            forecast_origin=latest_week,
        )
    else:
        raise ValueError(f"unsupported model_name: {model_name}")

    return pd.DataFrame(
        {
            week_column: weeks,
            "model_name": model_key,
            "target": target,
            "horizon": horizon,
            "lead_week": np.arange(1, horizon + 1, dtype=int),
            "point_forecast": np.asarray(prediction, dtype=float),
            "run_id": metadata.run_id,
            "converged": converged,
            "warning": model_warning,
            "model_warning": bool(model_warning) or used_fallback or not converged,
            "used_fallback": used_fallback,
        }
    )


def future_forecasts(
    history: pd.DataFrame | pd.Series,
    horizons: tuple[int, ...] = DEFAULT_FUTURE_HORIZONS,
    *,
    model_name: str = "ets",
    week_column: str = "week",
    target: str = "target",
    season_length: int = 52,
    future_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return true future rows for each requested 4/8/13/26-week horizon."""

    requested = tuple(_validate_horizon(item) for item in horizons)
    if any(item == 0 for item in requested):
        raise ValueError("future forecast horizons must be positive")
    if not requested:
        return pd.DataFrame()

    frames = [
        future_forecast(
            history,
            horizon=item,
            model_name=model_name,
            week_column=week_column,
            target=target,
            season_length=season_length,
            future_features=future_features,
        )
        for item in requested
    ]
    return pd.concat(frames, ignore_index=True)


def forecast_future(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Explicit orchestration alias for :func:`future_forecast`."""

    return future_forecast(*args, **kwargs)


def forecast_future_horizons(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Explicit orchestration alias for :func:`future_forecasts`."""

    return future_forecasts(*args, **kwargs)
