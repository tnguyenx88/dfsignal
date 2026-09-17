"""Metadata contracts for reproducible forecasting runs.

The metadata object is intentionally independent of a persistence backend.  An
orchestrator can serialize :meth:`ModelRunMetadata.to_dict` alongside its
forecast table without coupling model code to Parquet, DuckDB, or any other
storage system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


TemporalValue = date | datetime | str | None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalise_temporal(value: TemporalValue) -> TemporalValue:
    """Return a stable, timezone-aware value where the input is a datetime."""

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return value


def _serialise(value: Any) -> Any:
    """Convert common temporal values to JSON-compatible representations."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ModelRunMetadata:
    """Description of one model fit and forecast request.

    ``training_start`` and ``training_end`` are retained as separate fields so
    the object is easy to flatten into a table.  ``training_bounds`` is also
    exposed and accepted for callers that naturally hold a two-value bound.
    Bounds and origins are labels supplied by the caller (normally weekly
    dates); no filesystem or backend assumptions are made here.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    model_name: str = ""
    target: str = "target"
    training_start: TemporalValue = None
    training_end: TemporalValue = None
    forecast_origin: TemporalValue = None
    horizon: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    converged: bool | None = None
    warning: str | None = None
    created_at: datetime = field(default_factory=_utc_now)
    training_bounds: tuple[TemporalValue, TemporalValue] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.horizon < 0:
            raise ValueError("horizon must be non-negative")

        start = _normalise_temporal(self.training_start)
        end = _normalise_temporal(self.training_end)
        supplied_bounds = self.training_bounds
        if supplied_bounds is not None:
            if len(supplied_bounds) != 2:
                raise ValueError("training_bounds must contain exactly two values")
            bound_start, bound_end = (_normalise_temporal(item) for item in supplied_bounds)
            if start is None:
                start = bound_start
            if end is None:
                end = bound_end
            supplied_bounds = (bound_start, bound_end)
        else:
            supplied_bounds = (start, end)

        object.__setattr__(self, "training_start", start)
        object.__setattr__(self, "training_end", end)
        object.__setattr__(self, "training_bounds", supplied_bounds)
        object.__setattr__(self, "forecast_origin", _normalise_temporal(self.forecast_origin))
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "created_at", _normalise_temporal(self.created_at) or _utc_now())

    @classmethod
    def from_training_bounds(
        cls,
        *,
        model_name: str,
        target: str,
        training_bounds: tuple[TemporalValue, TemporalValue] | None,
        forecast_origin: TemporalValue,
        horizon: int,
        parameters: Mapping[str, Any] | None = None,
        converged: bool | None = None,
        warning: str | None = None,
        run_id: str | None = None,
        created_at: datetime | None = None,
    ) -> "ModelRunMetadata":
        """Construct metadata when bounds are already available as a pair."""

        return cls(
            run_id=run_id or uuid4().hex,
            model_name=model_name,
            target=target,
            training_bounds=training_bounds,
            forecast_origin=forecast_origin,
            horizon=horizon,
            parameters=parameters or {},
            converged=converged,
            warning=warning,
            created_at=created_at or _utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-compatible representation for sidecars/tables."""

        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "target": self.target,
            "training_start": _serialise(self.training_start),
            "training_end": _serialise(self.training_end),
            "training_bounds": _serialise(self.training_bounds),
            "forecast_origin": _serialise(self.forecast_origin),
            "horizon": self.horizon,
            "parameters": _serialise(self.parameters),
            "converged": self.converged,
            "warning": self.warning,
            "created_at": _serialise(self.created_at),
        }
