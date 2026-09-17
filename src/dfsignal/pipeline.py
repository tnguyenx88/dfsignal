"""Independent local pipeline stages and their run-all composition.

Every stage accepts a :class:`~dfsignal.orchestration.PipelineContext` when
called by the orchestrator.  For convenience, the same functions can be
called directly with no context (or with ``root_dir=...``); they then create a
local context and read/write only through the storage backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from .config import load_yaml
from .evaluation.backtest import (
    bias,
    evaluate_forecast,
    horizon_bucket,
    mase,
    rmse,
    wape,
)
from .evaluation.intervals import build_conformal_intervals, interval_metrics
from .explainability.shap_values import explain_lightgbm
from .features.build import build_features
from .forecasting.metadata import ModelRunMetadata
from .forecasting.models import (
    EXCLUDED_FEATURES,
    ets_forecast_with_metadata,
    lightgbm_forecast,
    seasonal_naive,
)
from .ingestion.events import read_launch_events, usable_launch_events
from .ingestion.fred import fetch_fred_series
from .ingestion.gpu import derive_gpu_launch_events, derive_gpu_weekly_signals, download_gpu_dataset, normalize_gpu_specs, read_gpu_specs
from .ingestion.internal import SyntheticDemandSource
from .ingestion.steam import read_steam_hardware
from .storage import LocalStorageBackend, StorageBackend
from .transformation.steam import (
    apply_steam_quality_flags,
    derive_steam_signals,
    diagnose_steam_launch_adoption,
    join_launch_adoption,
    validate_steam_source,
)
from .transformation.weekly import aggregate_weekly, to_week_saturday
from .validation.gpu import generate_gpu_validation_report, persist_gpu_validation_report
from .orchestration import (
    PipelineContext,
    PipelineRunError,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    make_context,
    run_pipeline,
    run_stage,
)


FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_START_DATE = "2016-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_HORIZONS = (4, 8, 13, 26)
SOURCE_IDS = {
    "fred": "FRED",
    "gpu": "KAGGLE_GPU_SPECS",
    "steam": "STEAM_HARDWARE",
    "events": "MANUAL_LAUNCH_EVENTS",
    "demand": "SYNTHETIC",
}


def _context(
    value: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    config_dir: str | Path = "config",
) -> PipelineContext:
    if isinstance(value, PipelineContext):
        return value
    if value is not None and not isinstance(value, (str, Path)):
        storage = value
    elif value is not None:
        root_dir = value
    return make_context(
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )


def _yaml(context: PipelineContext, filename: str) -> dict[str, Any]:
    path = context.config_dir / filename
    if not path.exists():
        # Preserve the original project convention when a caller supplies a
        # relative config directory that does not contain this file.
        path = Path(filename)
    if not path.exists():
        return {}
    return load_yaml(path)


def _model_config(context: PipelineContext) -> dict[str, Any]:
    return _yaml(context, "models.yaml")


def _source_config(context: PipelineContext) -> dict[str, Any]:
    return _yaml(context, "sources.yaml")


def _record_source(
    context: PipelineContext,
    source_id: str,
    *,
    status: str,
    row_count: int | None = None,
    frame: pd.DataFrame | None = None,
    error: str | None = None,
    table_name: str | None = None,
    layer: str | None = None,
) -> None:
    latest = _date_bound(frame, maximum=True) if frame is not None else None
    state = {
        "source_id": source_id,
        "status": status,
        "row_count": row_count if row_count is not None else (len(frame) if frame is not None else None),
        "latest_data_week": latest,
        "error": error,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    context.source_states[source_id] = state
    recorder = getattr(context.storage, "record_source_refresh", None)
    if callable(recorder):
        recorder(
            source_id,
            status=status,
            row_count=state["row_count"],
            min_date=_date_bound(frame) if frame is not None else None,
            max_date=latest,
            latest_data_week=latest,
            table_name=table_name,
            layer=layer,
            error=error,
        )
    if frame is not None and latest is not None:
        setter = getattr(context.storage, "set_watermark", None)
        if callable(setter):
            setter(source_id, "latest_data_week", latest)


def _date_bound(frame: pd.DataFrame | None, maximum: bool = False) -> str | None:
    if frame is None or frame.empty:
        return None
    for column in ("week", "date", "observation_date", "event_date", "release_date"):
        if column not in frame.columns:
            continue
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        if not values.empty:
            return str((values.max() if maximum else values.min()).date())
    return None


def _read_table(context: PipelineContext, table_name: str, layer: str) -> pd.DataFrame:
    if not context.storage.table_exists(table_name, layer):
        raise FileNotFoundError(f"Required table is not available: {layer}/{table_name}")
    return context.storage.read_table(table_name, layer)


def _write(context: PipelineContext, table_name: str, frame: pd.DataFrame, layer: str, source_id: str, **kwargs: Any) -> Path:
    return context.storage.write_table(table_name, frame, layer, source_id, **kwargs)


def _load_ingest_tables(context: PipelineContext) -> dict[str, pd.DataFrame]:
    """Load Bronze inputs from the backend, falling back to this run's artifact.

    Reading persisted tables first keeps independently callable stages honest:
    a stage can run in a fresh process and still sees the exact input written by
    its dependency, rather than depending on an in-memory hand-off.
    """

    artifact = context.artifacts.get("ingest")
    artifact_tables = dict(artifact) if isinstance(artifact, Mapping) else {}
    tables: dict[str, pd.DataFrame] = {}
    locations = {
        "demand": ("internal_demand", "bronze"),
        "fred": ("fred_weekly_raw", "bronze"),
        "gpu_raw": ("gpu_specs_raw", "bronze"),
        "steam": ("steam_hardware", "bronze"),
        "events": ("launch_events", "bronze"),
    }
    for key, (table, layer) in locations.items():
        if context.storage.table_exists(table, layer):
            tables[key] = context.storage.read_table(table, layer)
        elif key in artifact_tables and isinstance(artifact_tables[key], pd.DataFrame):
            tables[key] = artifact_tables[key]
    tables.setdefault("fred", pd.DataFrame(columns=["week"]))
    tables.setdefault("gpu_raw", pd.DataFrame())
    tables.setdefault("steam", pd.DataFrame())
    tables.setdefault("events", pd.DataFrame())
    if "demand" not in tables:
        raise FileNotFoundError("Ingested internal demand is not available")
    return tables


def _load_transformed(context: PipelineContext) -> dict[str, pd.DataFrame]:
    """Load Silver inputs from storage before consulting in-memory artifacts."""

    artifact = context.artifacts.get("transform")
    artifact_tables = dict(artifact) if isinstance(artifact, Mapping) else {}
    locations = {
        "demand": ("internal_demand_weekly", "silver"),
        "macro": ("macro_weekly", "silver"),
        "gpu_products": ("gpu_product", "silver"),
        "gpu_signals": ("gpu_event_weekly", "silver"),
        "events": ("event_weekly", "silver"),
        "steam": ("steam_weekly", "silver"),
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, (table, layer) in locations.items():
        if context.storage.table_exists(table, layer):
            tables[key] = context.storage.read_table(table, layer)
        elif key in artifact_tables and isinstance(artifact_tables[key], pd.DataFrame):
            tables[key] = artifact_tables[key]
    tables.setdefault("macro", pd.DataFrame(columns=["week"]))
    tables.setdefault("gpu_products", pd.DataFrame())
    tables.setdefault("gpu_signals", pd.DataFrame(columns=["week"]))
    tables.setdefault("events", pd.DataFrame(columns=["week"]))
    tables.setdefault("steam", pd.DataFrame(columns=["week"]))
    if "demand" not in tables:
        raise FileNotFoundError("Transformed internal demand is not available")
    tables["external"] = _merge_external(
        [tables["macro"], tables["gpu_signals"], tables["events"], tables["steam"]]
    )
    return tables


def _load_features(context: PipelineContext) -> pd.DataFrame:
    """Read the Gold feature table from storage before using an artifact."""

    if context.storage.table_exists("demand_features", "gold"):
        return context.storage.read_table("demand_features", "gold")
    artifact = context.artifacts.get("features")
    if isinstance(artifact, pd.DataFrame):
        return artifact
    return _read_table(context, "demand_features", "gold")


def _load_train(context: PipelineContext) -> dict[str, Any]:
    artifact = context.artifacts.get("train")
    if isinstance(artifact, Mapping):
        return dict(artifact)
    features = _load_features(context)
    target = _target_column(context, features)
    valid = features.dropna(subset=["week", target]).copy()
    if valid.empty:
        raise ValueError("No observed feature rows are available for training")
    horizons = _horizons(context)
    holdout_size = min(max(horizons), max(1, len(valid) // 5))
    if len(valid) <= holdout_size:
        raise ValueError("Not enough complete feature rows for a training split")
    return {
        "features": features,
        "valid": valid,
        "train": valid.iloc[:-holdout_size].copy(),
        "holdout": valid.iloc[-holdout_size:].copy(),
        "models": {},
    }


def _horizons(context: PipelineContext) -> tuple[int, ...]:
    configured = _model_config(context).get("forecast_horizons_weeks", DEFAULT_HORIZONS)
    values = sorted({int(value) for value in configured if int(value) > 0})
    return tuple(values or DEFAULT_HORIZONS)


def _target_column(context: PipelineContext, frame: pd.DataFrame) -> str:
    target = _model_config(context).get("target", {})
    if not isinstance(target, Mapping):
        raise ValueError("Model target configuration must be a mapping")
    configured = str(target.get("column", "")).strip()
    if not configured:
        raise ValueError("Model target column must be configured")
    allowed = {
        str(value).strip()
        for value in target.get("allowed_columns", ())
        if str(value).strip()
    }
    if allowed and configured not in allowed:
        raise ValueError(f"Configured target is not allowed: {configured}")
    if configured in frame.columns:
        return configured
    fallback = target.get("fallback_column")
    if fallback and str(fallback) in frame.columns:
        return str(fallback)
    raise ValueError(f"Configured target column is not present: {configured}")
def _feature_config(context: PipelineContext) -> dict[str, Any]:
    return _yaml(context, "features.yaml")


def _provenance_frame(
    frame: pd.DataFrame,
    *,
    source_id: str,
    source_table: str,
    layer: str,
) -> pd.DataFrame:
    """Attach row-level provenance without changing source-native values."""

    result = frame.copy()
    if "source_id" not in result.columns:
        result["source_id"] = source_id
    else:
        result["source_id"] = result["source_id"].astype("string").fillna(source_id)
    result["provenance_source_id"] = result["source_id"].astype("string")
    result["provenance_table"] = source_table
    result["provenance_layer"] = layer
    if "week" in result.columns:
        result["source_as_of"] = pd.to_datetime(result["week"], errors="coerce")
    return result


def _numeric_columns(frame: pd.DataFrame, *, exclude: Sequence[str] = ()) -> list[str]:
    excluded = set(exclude) | {"week", "source_id", "source", "provenance_source_id", "provenance_table", "provenance_layer", "source_as_of"}
    return [
        str(column)
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _external_column_groups(transformed: Mapping[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Return numeric external feature columns by their Silver source table."""

    groups: dict[str, list[str]] = {}
    for group, key in (("macro", "macro"), ("gpu_signal", "gpu_signals"), ("event", "events"), ("steam", "steam")):
        frame = transformed.get(key, pd.DataFrame())
        groups[group] = _numeric_columns(frame)
    return groups


def _feature_set_columns(
    context: PipelineContext,
    transformed: Mapping[str, pd.DataFrame],
    feature_frame: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Resolve the four comparable, nested feature sets from YAML."""

    config = _feature_config(context)
    lag_weeks = config.get("lag_weeks", [1, 2, 4, 8, 13, 26, 52])
    rolling_means = config.get("rolling_mean_weeks", [4, 8, 13, 26, 52])
    rolling_stds = config.get("rolling_std_weeks", [4, 13])
    history = [f"lag_{int(value)}" for value in lag_weeks]
    history.extend(f"rolling_mean_{int(value)}" for value in rolling_means)
    history.extend(f"rolling_std_{int(value)}" for value in rolling_stds)
    history.extend(["wow", "yoy"])
    calendar = [str(value) for value in config.get(
        "known_calendar_features",
        ["week_of_year", "month", "quarter", "christmas", "black_friday", "cyber_monday", "back_to_school"],
    )]
    groups = _external_column_groups(transformed)
    macro = groups["macro"]
    gpu_signal = groups["gpu_signal"] + groups["event"]
    steam_signal = groups["steam"]
    available = set(feature_frame.columns) if feature_frame is not None else None

    def present(columns: Sequence[str]) -> list[str]:
        unique = list(dict.fromkeys(str(column) for column in columns))
        return [column for column in unique if available is None or column in available]

    resolved = {
        "history_only": present(history),
        "calendar": present(history + calendar),
        "macro": present(history + calendar + macro),
        "gpu_signal": present(history + calendar + macro + gpu_signal),
        "steam_signal": present(history + calendar + macro + steam_signal),
        "all_external": present(history + calendar + macro + gpu_signal + steam_signal),
    }
    configured = config.get("feature_sets")
    if isinstance(configured, Mapping):
        for name, definition in configured.items():
            if name not in resolved or not isinstance(definition, Mapping):
                continue
            includes = definition.get("include")
            if not isinstance(includes, Sequence) or isinstance(includes, (str, bytes)):
                continue
            columns: list[str] = []
            for include in includes:
                columns.extend({
                    "history": history,
                    "history_only": history,
                    "calendar": calendar,
                    "macro": macro,
                    "gpu_signal": gpu_signal,
                    "events": groups["event"],
                    "steam": groups["steam"],
                }.get(str(include), [str(include)]))
            resolved[str(name)] = present(columns)
    return resolved


def _model_frame(frame: pd.DataFrame, columns: Sequence[str], target: str) -> pd.DataFrame:
    """Select numeric model inputs while retaining the public week/target names."""

    selected = ["week", target] + [column for column in columns if column in frame.columns and column not in {"week", target}]
    result = frame.loc[:, list(dict.fromkeys(selected))].copy()
    result["week"] = pd.to_datetime(result["week"], errors="raise")
    result[target] = pd.to_numeric(result[target], errors="coerce")
    return result


def _history_series(frame: pd.DataFrame, target: str) -> pd.Series:
    if frame.empty:
        raise ValueError("Cannot fit a model without observed history")
    ordered = frame.dropna(subset=[target]).sort_values("week")
    if ordered.empty:
        raise ValueError("Cannot fit a model without observed target values")
    return pd.Series(
        pd.to_numeric(ordered[target], errors="raise").to_numpy(dtype=float),
        index=pd.DatetimeIndex(pd.to_datetime(ordered["week"], errors="raise")),
        name=target,
    )


def _load_macro(start_date: str, end_date: str, config_path: str | Path = "config/fred.yaml") -> pd.DataFrame:
    """Fetch configured FRED series and normalize them to weekly observations."""

    config = load_yaml(config_path)
    weekly: pd.DataFrame | None = None
    for series in config.get("fred_series", []):
        if not series.get("enabled", False):
            continue
        observations = fetch_fred_series(series["series_id"], FRED_BASE_URL, start_date, end_date)
        normalized = aggregate_weekly(observations, "observation_date", "value").rename(
            columns={"value": series["feature_name"]}
        )
        weekly = normalized if weekly is None else weekly.merge(normalized, on="week", how="outer", validate="one_to_one")
    return weekly if weekly is not None else pd.DataFrame(columns=["week"])


def ingest(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    config_dir: str | Path = "config",
) -> dict[str, pd.DataFrame]:
    """Ingest synthetic demand and optional public source extracts."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=True if fetch_external is None else fetch_external,
        start_date=start_date or DEFAULT_START_DATE,
        end_date=end_date or DEFAULT_END_DATE,
        config_dir=config_dir,
    )
    demand = SyntheticDemandSource(ctx.start_date, ctx.end_date).read()
    _write(ctx, "internal_demand", demand, "bronze", SOURCE_IDS["demand"], mode="replace")
    _record_source(ctx, SOURCE_IDS["demand"], status="success", frame=demand, table_name="internal_demand", layer="bronze")

    macro = pd.DataFrame(columns=["week"])
    gpu_raw = pd.DataFrame(columns=["product_name", "release_date"])
    if ctx.fetch_external:
        fred_config = ctx.config_dir / "fred.yaml"
        if not fred_config.exists():
            fred_config = Path("config/fred.yaml")
        macro = _load_macro(ctx.start_date, ctx.end_date, fred_config)
        _write(ctx, "fred_weekly_raw", macro, "bronze", SOURCE_IDS["fred"], mode="replace")
        _record_source(ctx, SOURCE_IDS["fred"], status="success", frame=macro, table_name="fred_weekly_raw", layer="bronze")

        resolver = getattr(ctx.storage, "resolve_local_path", None)
        if not callable(resolver):
            raise RuntimeError("Storage backend cannot resolve the local GPU source archive")
        gpu_archive = download_gpu_dataset(resolver("gpu_specs.zip", "external"))
        gpu_raw = read_gpu_specs(gpu_archive)
        _write(ctx, "gpu_specs_raw", gpu_raw, "bronze", SOURCE_IDS["gpu"], mode="replace")
        _record_source(ctx, SOURCE_IDS["gpu"], status="success", frame=gpu_raw, table_name="gpu_specs_raw", layer="bronze")
    else:
        _write(ctx, "fred_weekly_raw", macro, "bronze", SOURCE_IDS["fred"], mode="replace")
        _write(ctx, "gpu_specs_raw", gpu_raw, "bronze", SOURCE_IDS["gpu"], mode="replace")
        _record_source(ctx, SOURCE_IDS["fred"], status="skipped")
        _record_source(ctx, SOURCE_IDS["gpu"], status="skipped")

    source_config = _source_config(ctx).get("sources", {})
    steam_config = source_config.get("steam_hardware", {}) if isinstance(source_config, Mapping) else {}
    steam_enabled = bool(steam_config.get("enabled", False))
    resolver = getattr(ctx.storage, "resolve_local_path", None)
    configured_path = steam_config.get("path") if isinstance(steam_config, Mapping) else None
    if configured_path:
        candidate = Path(str(configured_path))
        steam_path = candidate if candidate.is_absolute() else (
            resolver(candidate, "external") if callable(resolver) else candidate
        )
    elif callable(resolver):
        steam_path = resolver("steam_hardware", "external")
    else:
        steam_path = None

    steam = read_steam_hardware(enabled=False)
    try:
        if not steam_enabled:
            status_value = "disabled" if not bool(steam_config.get("enabled", False)) else "skipped"
            _write(ctx, "steam_hardware", steam, "bronze", SOURCE_IDS["steam"], mode="replace")
            _record_source(ctx, SOURCE_IDS["steam"], status=status_value, table_name="steam_hardware", layer="bronze")
        else:
            steam = read_steam_hardware(
                steam_path,
                source_id=str(steam_config.get("source_id", SOURCE_IDS["steam"])),
                source_name=str(steam_config.get("source_name", "Steam Hardware and Software Survey extract")),
                source_url=steam_config.get("source_url"),
                mapping_path=ctx.config_dir / "steam_hardware.yaml",
            )
            anomaly_path = ctx.config_dir / "steam_anomalies.yaml"
            if not anomaly_path.exists():
                anomaly_path = Path("config/steam_anomalies.yaml")
            steam = apply_steam_quality_flags(steam, anomaly_path=anomaly_path)
            _write(ctx, "steam_hardware", steam, "bronze", SOURCE_IDS["steam"], mode="replace")
            _record_source(ctx, SOURCE_IDS["steam"], status="success", frame=steam, table_name="steam_hardware", layer="bronze")
            metadata_columns = [
                "source_id", "source_name", "source_url", "original_filename", "file_hash",
                "load_timestamp", "row_count", "min_date", "max_date", "schema_version", "status",
            ]
            metadata = steam[metadata_columns].drop_duplicates().reset_index(drop=True)
            _write(ctx, "steam_source_metadata", metadata, "cache", "DFSIGNAL_SOURCE_METADATA", mode="replace")
    except FileNotFoundError as exc:
        _write(ctx, "steam_hardware", steam, "bronze", SOURCE_IDS["steam"], mode="replace")
        _record_source(ctx, SOURCE_IDS["steam"], status="missing_optional", error=str(exc), table_name="steam_hardware", layer="bronze")
    except (ValueError, TypeError) as exc:
        _write(ctx, "steam_hardware", steam, "bronze", SOURCE_IDS["steam"], mode="replace")
        _record_source(ctx, SOURCE_IDS["steam"], status="invalid", error=str(exc), table_name="steam_hardware", layer="bronze")

    events_path = ctx.config_dir / "launch_events.csv"
    if not events_path.exists():
        events_path = Path("config/launch_events.csv")
    if not events_path.exists():
        raise FileNotFoundError(f"Launch events configuration not found: {events_path}")
    events = read_launch_events(events_path)
    events["source_id"] = SOURCE_IDS["events"]
    _write(ctx, "launch_events", events, "bronze", SOURCE_IDS["events"], mode="replace")
    _record_source(ctx, SOURCE_IDS["events"], status="success", frame=events, table_name="launch_events", layer="bronze")
    result = {"demand": demand, "fred": macro, "gpu_raw": gpu_raw, "steam": steam, "events": events}
    ctx.artifacts["ingest"] = result
    return result


def validate(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Validate Bronze contracts and persist a reviewable GPU report."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    tables = _load_ingest_tables(ctx)
    checks: list[dict[str, str]] = []

    demand = tables["demand"]
    target = _target_column(ctx, demand)
    required = {"week", target}
    missing = sorted(required - set(demand.columns))
    if missing:
        raise ValueError(f"Internal demand is missing columns: {missing}")
    checks.append({"check": "internal_demand_schema", "status": "passed", "detail": f"target={target}"})
    parsed_weeks = pd.to_datetime(demand["week"], errors="coerce")
    if parsed_weeks.isna().any():
        raise ValueError("Internal demand contains invalid week values")
    if demand.empty:
        raise ValueError("Internal demand is empty")
    checks.append(
        {"check": "internal_demand_dates", "status": "passed", "detail": _date_bound(demand, maximum=True) or ""}
    )
    if pd.to_numeric(demand[target], errors="coerce").isna().any():
        raise ValueError(f"Internal demand target contains non-numeric values: {target}")
    checks.append({"check": "internal_demand_target", "status": "passed", "detail": target})

    duplicate_keys = [
        column
        for column in ("week", "business_family", "region", "channel", "source_id")
        if column in demand.columns
    ]
    if duplicate_keys and demand.duplicated(subset=duplicate_keys).any():
        raise ValueError(f"Internal demand contains duplicate keys: {duplicate_keys}")
    checks.append({"check": "internal_demand_keys", "status": "passed", "detail": ",".join(duplicate_keys)})

    events = tables.get("events", pd.DataFrame())
    if events.empty:
        raise ValueError("Launch events are empty")
    checks.append({"check": "launch_events", "status": "passed", "detail": str(len(events))})

    for key, source_id in (
        ("fred", SOURCE_IDS["fred"]),
        ("gpu_raw", SOURCE_IDS["gpu"]),
        ("steam", SOURCE_IDS["steam"]),
    ):
        frame = tables.get(key, pd.DataFrame())
        state = ctx.source_states.get(source_id, {})
        if frame.empty and state.get("status") in {"unavailable", "skipped", "disabled", "missing_optional", "invalid"}:
            checks.append(
                {"check": f"{key}_optional", "status": "reported", "detail": state.get("status", "absent")}
            )
        elif frame.empty and key != "steam":
            checks.append({"check": f"{key}_optional", "status": "reported", "detail": "absent"})
        else:
            checks.append({"check": f"{key}_schema", "status": "passed", "detail": str(len(frame))})

    gpu_raw = tables.get("gpu_raw", pd.DataFrame())
    gpu_report_dir = ctx.output_dir / "validation"
    if not gpu_raw.empty:
        gpu_products = normalize_gpu_specs(gpu_raw, config_dir=ctx.config_dir)
        gpu_signals = derive_gpu_weekly_signals(gpu_products)
        gpu_report = generate_gpu_validation_report(
            products=gpu_products,
            raw=gpu_raw,
            weekly_signals=gpu_signals,
            output_dir=gpu_report_dir,
        )
        checks.extend(
            [
                {
                    "check": "gpu_coverage",
                    "status": str(gpu_report["coverage"].get("status", "REVIEW")).lower(),
                    "detail": json.dumps(gpu_report["coverage"], default=str, sort_keys=True),
                },
                {
                    "check": "gpu_sanity",
                    "status": str(gpu_report["diagnostics"].get("status", "REVIEW")).lower(),
                    "detail": json.dumps(gpu_report["diagnostics"], default=str, sort_keys=True),
                },
                {
                    "check": "gpu_review_sample",
                    "status": "reported",
                    "detail": str(len(gpu_report.get("sample", pd.DataFrame()))),
                },
            ]
        )
    else:
        empty_checks = pd.DataFrame(
            [
                {
                    "check": "gpu_source_available",
                    "status": "REVIEW",
                    "observed": 0,
                    "rule": "GPU source is optional for local runs but should be reviewed before signal use",
                }
            ]
        )
        gpu_report = {
            "status": "REVIEW",
            "summary": {
                "status": "REVIEW",
                "product_rows": 0,
                "sample_rows": 0,
                "coverage_status": "REVIEW",
                "diagnostics_status": "REVIEW",
                "sample_status": "REVIEW",
            },
            "coverage": {
                "status": "REVIEW",
                "total_rows": 0,
                "valid_date_rows": 0,
                "invalid_or_missing_date_rows": 0,
                "coverage_basis": "no GPU rows were available",
            },
            "diagnostics": {"status": "REVIEW", "reason": "no GPU rows were available"},
            "checks": empty_checks,
            "weekly_diagnostics": pd.DataFrame(
                columns=["week", "gpu_launch_count", "spike_threshold", "spike_flag", "diagnostic_status"]
            ),
            "generation_conflict_rows": pd.DataFrame(
                columns=["source_row_number", "gpu_id", "product_name"]
            ),
            "sample": pd.DataFrame(columns=["sample_rank", "sample_stratum"]),
        }
        checks.extend(
            [
                {"check": "gpu_coverage", "status": "reported", "detail": "source absent"},
                {"check": "gpu_sanity", "status": "reported", "detail": "source absent"},
                {"check": "gpu_review_sample", "status": "reported", "detail": "0"},
            ]
        )
    steam_report = validate_steam_source(
        tables.get("steam", pd.DataFrame()),
        anomaly_path=ctx.config_dir / "steam_anomalies.yaml",
        mapping_path=ctx.config_dir / "steam_hardware.yaml",
    )
    steam_checks = steam_report.get("checks", pd.DataFrame())
    if isinstance(steam_checks, pd.DataFrame) and not steam_checks.empty:
        steam_checks = steam_checks.copy()
        steam_checks["detail"] = steam_checks["detail"].astype(str)
        checks.extend(
            {
                "check": f"steam_{row.check}",
                "status": str(row.status).lower(),
                "detail": str(row.detail),
            }
            for row in steam_checks.itertuples(index=False)
        )
        _write(ctx, "steam_validation", steam_checks, "cache", "DFSIGNAL_STEAM_VALIDATION", mode="replace")
    else:
        checks.append({"check": "steam_source", "status": "reported", "detail": steam_report.get("status", "REVIEW")})


    report = pd.DataFrame(checks, columns=["check", "status", "detail"])
    _write(ctx, "validation_report", report, "cache", "DFSIGNAL_VALIDATION", mode="replace")
    ctx.artifacts["gpu_validation"] = gpu_report
    ctx.artifacts["validate"] = report
    return report


def _weekly_demand(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["week"] = pd.to_datetime(normalized["week"])
    normalized[target] = pd.to_numeric(normalized[target], errors="raise")
    dimensions = [column for column in ("week", "business_family", "region", "channel", "source_id") if column in normalized.columns]
    if normalized.duplicated(subset=dimensions).any():
        aggregation = {target: "sum"}
        for column in normalized.columns:
            if column not in dimensions and column != target:
                aggregation[column] = "first"
        normalized = normalized.groupby(dimensions, as_index=False).agg(aggregation)
    return normalized.sort_values("week").reset_index(drop=True)


def _merge_external(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge numeric Silver signals while retaining source provenance."""

    merged: pd.DataFrame | None = None
    source_ids: list[str] = []
    metadata_columns = {
        "source_id",
        "source",
        "provenance_source_id",
        "provenance_table",
        "provenance_layer",
        "source_as_of",
    }
    for frame in frames:
        if frame is None or frame.empty or "week" not in frame.columns:
            continue
        current = frame.copy()
        for column in ("source_id", "source", "provenance_source_id"):
            if column in current.columns:
                source_ids.extend(current[column].dropna().astype(str).unique().tolist())
        current["week"] = pd.to_datetime(current["week"], errors="raise")
        value_columns = [
            column
            for column in current.columns
            if column != "week"
            and column not in metadata_columns
            and pd.api.types.is_numeric_dtype(current[column])
        ]
        current = current[["week"] + value_columns]
        if not value_columns:
            continue
        current = current.sort_values("week").drop_duplicates(subset=["week"], keep="last")
        merged = current if merged is None else merged.merge(
            current,
            on="week",
            how="outer",
            validate="one_to_one",
        )
    if merged is None:
        return pd.DataFrame(columns=["week", "external_source_ids"])
    merged = merged.sort_values("week").reset_index(drop=True)
    value_columns = [column for column in merged.columns if column != "week"]
    if value_columns:
        merged[value_columns] = merged[value_columns].ffill().fillna(0)
    merged["external_source_ids"] = "|".join(sorted(set(source_ids))) if source_ids else ""
    return merged


def _event_weekly(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate usable manual events; examples remain in Bronze only."""
    events = usable_launch_events(events)
    if events.empty:
        return pd.DataFrame(columns=["week", "launch_event_count", "gpu_launch_score", "cpu_launch_score"])
    frame = events.copy()
    frame["week"] = pd.to_datetime(to_week_saturday(frame["event_date"]))
    event_type = frame["event_type"].astype(str).str.upper()
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").fillna(0.0)
    frame["gpu_launch_score"] = (event_type.str.contains("GPU") * confidence).astype(float)
    frame["cpu_launch_score"] = (event_type.str.contains("CPU") * confidence).astype(float)
    result = frame.groupby("week", as_index=False).agg(
        launch_event_count=("event_type", "size"),
        gpu_launch_score=("gpu_launch_score", "sum"),
        cpu_launch_score=("cpu_launch_score", "sum"),
    )
    result["launch_plus_4_weeks"] = result["launch_event_count"].rolling(4, min_periods=1).sum()
    result["launch_plus_13_weeks"] = result["launch_event_count"].rolling(13, min_periods=1).sum()
    return result


def _steam_weekly(
    steam: pd.DataFrame,
    launch_events: pd.DataFrame | None = None,
    *,
    mapping_path: str | Path | None = None,
    publication_lag_months: int = 0,
) -> pd.DataFrame:
    """Normalize optional Steam source and derive observed weekly signals."""
    if steam.empty:
        return pd.DataFrame(columns=["week", "source_id"])
    result = derive_steam_signals(
        steam,
        mapping_path=mapping_path,
        publication_lag_months=publication_lag_months,
    )
    return join_launch_adoption(result, launch_events if launch_events is not None else pd.DataFrame())


def transform(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> dict[str, pd.DataFrame]:
    """Normalize Bronze inputs into persisted, provenance-preserving Silver tables."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    tables = _load_ingest_tables(ctx)
    demand = tables["demand"]
    target = _target_column(ctx, demand)
    weekly_demand = _provenance_frame(
        _weekly_demand(demand, target),
        source_id=SOURCE_IDS["demand"],
        source_table="internal_demand_weekly",
        layer="silver",
    )
    _write(ctx, "internal_demand_weekly", weekly_demand, "silver", SOURCE_IDS["demand"], mode="replace")

    macro = _provenance_frame(
        tables.get("fred", pd.DataFrame(columns=["week"])),
        source_id=SOURCE_IDS["fred"],
        source_table="macro_weekly",
        layer="silver",
    )
    gpu_products = pd.DataFrame()
    gpu_signals = pd.DataFrame(columns=["week"])
    gpu_events = pd.DataFrame()
    gpu_raw = tables.get("gpu_raw", pd.DataFrame())
    if not gpu_raw.empty:
        gpu_products = normalize_gpu_specs(gpu_raw, config_dir=ctx.config_dir)
        gpu_products = _provenance_frame(
            gpu_products,
            source_id=SOURCE_IDS["gpu"],
            source_table="gpu_product",
            layer="silver",
        )
        gpu_signals = derive_gpu_weekly_signals(gpu_products)
        gpu_events = derive_gpu_launch_events(
            gpu_products,
            config_path=ctx.config_dir / "gpu_signal.yaml",
        )
    if gpu_products.empty:
        gpu_products = _provenance_frame(
            pd.DataFrame(columns=["gpu_id", "product_name", "release_date"]),
            source_id=SOURCE_IDS["gpu"],
            source_table="gpu_product",
            layer="silver",
        )
    gpu_events = _provenance_frame(
        gpu_events,
        source_id=SOURCE_IDS["gpu"],
        source_table="gpu_events",
        layer="silver",
    )
    gpu_signals = _provenance_frame(
        gpu_signals,
        source_id=SOURCE_IDS["gpu"],
        source_table="gpu_event_weekly",
        layer="silver",
    )
    _write(ctx, "gpu_product", gpu_products, "silver", SOURCE_IDS["gpu"], mode="replace")
    _write(ctx, "gpu_events", gpu_events, "silver", SOURCE_IDS["gpu"], mode="replace")
    _write(ctx, "gpu_event_weekly", gpu_signals, "silver", SOURCE_IDS["gpu"], mode="replace")

    event_weekly = _event_weekly(tables.get("events", pd.DataFrame()))
    event_weekly = _provenance_frame(
        event_weekly,
        source_id=SOURCE_IDS["events"],
        source_table="event_weekly",
        layer="silver",
    )
    _write(ctx, "event_weekly", event_weekly, "silver", SOURCE_IDS["events"], mode="replace")

    steam_config = _yaml(ctx, "steam_hardware.yaml")
    steam_weekly = _steam_weekly(
        tables.get("steam", pd.DataFrame()),
        tables.get("events", pd.DataFrame()),
        mapping_path=ctx.config_dir / "steam_hardware.yaml",
        publication_lag_months=int(steam_config.get("publication_lag_months", 0)),
    )
    steam_weekly = _provenance_frame(
        steam_weekly,
        source_id=SOURCE_IDS["steam"],
        source_table="steam_weekly",
        layer="silver",
    )
    _write(ctx, "steam_weekly", steam_weekly, "silver", "DFSIGNAL_STEAM_SIGNALS", mode="replace")
    steam_diagnostics = diagnose_steam_launch_adoption(
        steam_weekly,
        tables.get("events", pd.DataFrame()),
    )
    _write(ctx, "steam_launch_diagnostics", steam_diagnostics, "cache", "DFSIGNAL_STEAM_DIAGNOSTICS", mode="replace")
    external = _merge_external([macro, gpu_signals, event_weekly, steam_weekly])
    result = {
        "demand": weekly_demand,
        "macro": macro,
        "gpu_events": gpu_events,
        "gpu_products": gpu_products,
        "gpu_signals": gpu_signals,
        "events": event_weekly,
        "steam": steam_weekly,
        "steam_diagnostics": steam_diagnostics,
        "external": external,
    }
    ctx.artifacts["transform"] = result
    return result

def features(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Build Gold features from Silver backend tables and retain provenance."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    transformed = _load_transformed(ctx)
    demand = transformed["demand"].copy()
    target = _target_column(ctx, demand)
    feature_frame = build_features(demand, transformed.get("external"), target_column=target)
    external = transformed.get("external")
    if isinstance(external, pd.DataFrame) and not external.empty:
        signal_columns = [
            column
            for column in _numeric_columns(external)
            if column in feature_frame.columns
        ]
        if signal_columns:
            # Sparse event/source calendars use an explicit zero for absence.
            feature_frame[signal_columns] = feature_frame[signal_columns].fillna(0)
    source_ids = [
        str(frame["source_id"].dropna().astype(str).iloc[0])
        for frame in transformed.values()
        if isinstance(frame, pd.DataFrame)
        and not frame.empty
        and "source_id" in frame.columns
        and not frame["source_id"].dropna().empty
    ]
    feature_frame["feature_source_ids"] = "|".join(sorted(set(source_ids)))
    feature_frame["feature_provenance_table"] = "demand_features"
    feature_frame["feature_provenance_layer"] = "gold"
    feature_frame["feature_source_as_of"] = pd.to_datetime(feature_frame["week"], errors="coerce")
    _write(ctx, "demand_features", feature_frame, "gold", "DFSIGNAL_FEATURE_PIPELINE", mode="replace")
    ctx.artifacts["features"] = feature_frame
    return feature_frame
def _external_strategy(context: PipelineContext, group: str) -> str:
    config = _feature_config(context)
    strategies = config.get("external_feature_strategies", {})
    if not isinstance(strategies, Mapping):
        return "observed_only"
    value = strategies.get(group, strategies.get("default", "observed_only"))
    return str(value).strip().lower() or "observed_only"


def _calendar_row(week: pd.Timestamp) -> dict[str, float | int]:
    iso_week = int(week.isocalendar().week)
    return {
        "week_of_year": iso_week,
        "month": int(week.month),
        "quarter": int(week.quarter),
        "christmas": int(week.month == 12),
        "black_friday": int(week.month == 11 and week.day >= 22),
        "cyber_monday": int(week.month == 11 and week.day >= 22),
        "back_to_school": int(week.month == 8 and week.day >= 7),
    }


def _future_external_values(
    context: PipelineContext,
    feature_frame: pd.DataFrame,
    week: pd.Timestamp,
    columns: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    origin: pd.Timestamp,
) -> dict[str, float]:
    """Resolve future external values without consulting rows after origin."""

    ordered = feature_frame.copy()
    ordered["week"] = pd.to_datetime(ordered["week"], errors="raise")
    observed = ordered.loc[ordered["week"].le(origin)].sort_values("week")
    result: dict[str, float] = {}
    for group, group_columns in groups.items():
        strategy = _external_strategy(context, group)
        if strategy not in {"observed_only", "last_observed", "carry_forward", "zero"}:
            raise ValueError(f"Unsupported future external-feature strategy for {group}: {strategy}")
        for column in group_columns:
            if column not in columns:
                continue
            if strategy == "observed_only":
                value = float("nan")
            elif strategy == "zero":
                value = 0.0
            else:
                value = 0.0
                if column in observed.columns:
                    values = pd.to_numeric(observed[column], errors="coerce").dropna()
                    if not values.empty:
                        value = float(values.iloc[-1])
            result[column] = value
    return result


def _target_feature_row(
    values: Sequence[float],
    columns: Sequence[str],
    *,
    lag_weeks: Sequence[int],
    rolling_means: Sequence[int],
    rolling_stds: Sequence[int],
) -> dict[str, float]:
    """Compute target-derived features from observed values and prior predictions."""

    index = len(values)
    row: dict[str, float] = {}
    for lag in lag_weeks:
        name = f"lag_{int(lag)}"
        if name in columns:
            row[name] = float(values[index - int(lag)]) if index >= int(lag) else float("nan")
    for window in rolling_means:
        name = f"rolling_mean_{int(window)}"
        if name in columns:
            start = max(0, index - int(window))
            prior = values[start:index]
            row[name] = float(np.mean(prior)) if len(prior) >= int(window) else float("nan")
    for window in rolling_stds:
        name = f"rolling_std_{int(window)}"
        if name in columns:
            start = max(0, index - int(window))
            prior = values[start:index]
            row[name] = float(np.std(prior, ddof=1)) if len(prior) >= int(window) else float("nan")
    if "wow" in columns:
        row["wow"] = (
            float(values[-1] / values[-2] - 1.0)
            if len(values) >= 2 and values[-2] != 0
            else float("nan")
        )
    if "yoy" in columns:
        row["yoy"] = (
            float(values[-1] / values[-53] - 1.0)
            if len(values) >= 53 and values[-53] != 0
            else float("nan")
        )
    return row


def _future_feature_row(
    context: PipelineContext,
    feature_frame: pd.DataFrame,
    values: Sequence[float],
    week: pd.Timestamp,
    columns: Sequence[str],
    feature_set: str,
    origin: pd.Timestamp,
    transformed: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    config = _feature_config(context)
    row: dict[str, Any] = {"week": week}
    calendar = _calendar_row(week)
    for column in columns:
        if column in calendar:
            row[column] = calendar[column]
    row.update(
        _target_feature_row(
            values,
            columns,
            lag_weeks=config.get("lag_weeks", [1, 2, 4, 8, 13, 26, 52]),
            rolling_means=config.get("rolling_mean_weeks", [4, 8, 13, 26, 52]),
            rolling_stds=config.get("rolling_std_weeks", [4, 13]),
        )
    )
    groups = _external_column_groups(transformed)
    row.update(_future_external_values(context, feature_frame, week, columns, groups, origin))
    for column in columns:
        row.setdefault(column, float("nan"))
    return row


def _recursive_lightgbm_forecast(
    context: PipelineContext,
    history: pd.DataFrame,
    feature_frame: pd.DataFrame,
    weeks: Sequence[pd.Timestamp],
    *,
    target: str,
    columns: Sequence[str],
    feature_set: str,
    transformed: Mapping[str, pd.DataFrame],
    origin: pd.Timestamp,
) -> np.ndarray:
    """Predict one future row at a time so target lags never read holdout labels."""

    model_train = _model_frame(history, columns, target).dropna(subset=[target])
    if model_train.empty:
        raise ValueError("LightGBM requires observed training rows")
    values = model_train.sort_values("week")[target].astype(float).tolist()
    predictions: list[float] = []
    for week in weeks:
        row = _future_feature_row(
            context,
            feature_frame,
            values,
            pd.Timestamp(week),
            columns,
            feature_set,
            origin,
            transformed,
        )
        future = pd.DataFrame([row])
        prediction = lightgbm_forecast(model_train, future, target=target)
        value = float(prediction[0])
        predictions.append(value)
        values.append(value)
    return np.asarray(predictions, dtype=float)


def _rolling_origins(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    target: str,
    horizon: int,
    minimum_train: int,
    cutoff_count: int,
) -> tuple[pd.DataFrame, list[int]]:
    required = ["week", target] + list(columns)
    available = frame.dropna(subset=list(dict.fromkeys(required))).copy()
    available["week"] = pd.to_datetime(available["week"], errors="raise")
    available = available.sort_values("week").reset_index(drop=True)
    if len(available) <= minimum_train + horizon - 1:
        return available, []
    positions = list(range(minimum_train, len(available) - horizon + 1, max(1, horizon)))
    if not positions:
        positions = [len(available) - horizon]
    return available, positions[-max(1, cutoff_count):]


def _model_prediction(
    context: PipelineContext,
    model_name: str,
    train_frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    future_weeks: Sequence[pd.Timestamp],
    *,
    target: str,
    columns: Sequence[str],
    feature_set: str,
    transformed: Mapping[str, pd.DataFrame],
    origin: pd.Timestamp,
) -> tuple[np.ndarray, ModelRunMetadata | None]:
    horizon = len(future_weeks)
    model_key = model_name.lower()
    history = train_frame.sort_values("week")
    series = _history_series(history, target)
    if model_key == "seasonal_naive":
        prediction = seasonal_naive(series, horizon)
        metadata = ModelRunMetadata.from_training_bounds(
            model_name=model_key,
            target=target,
            training_bounds=(series.index.min(), series.index.max()),
            forecast_origin=origin,
            horizon=horizon,
            parameters={"season_length": 52, "feature_set": feature_set},
            converged=True,
        )
        return prediction, metadata
    if model_key == "ets":
        result = ets_forecast_with_metadata(
            series,
            horizon,
            target=target,
            forecast_origin=origin,
        )
        return result.forecast, result.metadata
    if model_key == "lightgbm":
        prediction = _recursive_lightgbm_forecast(
            context,
            history,
            feature_frame,
            future_weeks,
            target=target,
            columns=columns,
            feature_set=feature_set,
            transformed=transformed,
            origin=origin,
        )
        metadata = ModelRunMetadata.from_training_bounds(
            model_name=model_key,
            target=target,
            training_bounds=(series.index.min(), series.index.max()),
            forecast_origin=origin,
            horizon=horizon,
            parameters={"feature_columns": list(columns), "feature_set": feature_set},
            converged=True,
        )
        return prediction, metadata
    raise ValueError(f"Unsupported model: {model_name}")


def _model_names(context: PipelineContext) -> tuple[str, ...]:
    configured = _model_config(context).get("models", {})
    if not isinstance(configured, Mapping) or not configured:
        configured = {"seasonal_naive": True, "ets": True, "lightgbm": True}
    return tuple(
        name for name in ("seasonal_naive", "ets", "lightgbm")
        if bool(configured.get(name, False))
    )


def _metadata_row(metadata: ModelRunMetadata | None, **extra: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if metadata is not None:
        payload = metadata.to_dict()
        payload["parameters"] = json.dumps(payload.get("parameters", {}), sort_keys=True, default=str)
        payload["training_bounds"] = json.dumps(payload.get("training_bounds"), default=str)
        values.update(payload)
    values.update(extra)
    return values


def backtest(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Compare configured feature sets with expanding-window evaluations."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    feature_frame = _load_features(ctx)
    transformed = _load_transformed(ctx)
    target = _target_column(ctx, feature_frame)
    model_config = _model_config(ctx)
    backtest_config = model_config.get("backtest", {}) if isinstance(model_config, Mapping) else {}
    minimum_train = int(backtest_config.get("minimum_train_weeks", 52))
    cutoff_count = int(backtest_config.get("cutoffs", 5))
    horizons = _horizons(ctx)
    feature_sets = _feature_set_columns(ctx, transformed, feature_frame)
    feature_set_order = tuple(
        name for name in ("history_only", "calendar", "macro", "gpu_signal", "all_external")
        if name in feature_sets
    )
    rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []

    for feature_set in feature_set_order:
        columns = feature_sets.get(feature_set, [])
        for horizon in horizons:
            ordered, positions = _rolling_origins(
                feature_frame,
                columns,
                target=target,
                horizon=horizon,
                minimum_train=minimum_train,
                cutoff_count=cutoff_count,
            )
            for end in positions:
                train_frame = ordered.iloc[:end].copy()
                test_frame = ordered.iloc[end : end + horizon].copy()
                if len(test_frame) != horizon:
                    continue
                origin = pd.Timestamp(train_frame["week"].iloc[-1])
                future_weeks = [pd.Timestamp(value) for value in test_frame["week"]]
                baseline = seasonal_naive(_history_series(train_frame, target), horizon)
                baseline_metrics = evaluate_forecast(
                    test_frame[target].to_numpy(dtype=float),
                    baseline,
                    training=train_frame[target].to_numpy(dtype=float),
                    season_length=1,
                    horizon=horizon,
                )
                for model_name in _model_names(ctx):
                    predicted, metadata = _model_prediction(
                        ctx,
                        model_name,
                        train_frame,
                        feature_frame,
                        future_weeks,
                        target=target,
                        columns=columns,
                        feature_set=feature_set,
                        transformed=transformed,
                        origin=origin,
                    )
                    metrics = evaluate_forecast(
                        test_frame[target].to_numpy(dtype=float),
                        predicted,
                        training=train_frame[target].to_numpy(dtype=float),
                        season_length=1,
                        horizon=horizon,
                    )
                    metrics["mae"] = float(
                        np.mean(
                            np.abs(
                                test_frame[target].to_numpy(dtype=float)
                                - np.asarray(predicted, dtype=float)
                            )
                        )
                    )
                    metrics["bias"] = bias(
                        test_frame[target].to_numpy(dtype=float),
                        np.asarray(predicted, dtype=float),
                    )
                    model_row = {
                        "cutoff": origin,
                        "horizon": horizon,
                        "horizon_weeks": horizon,
                        "horizon_bucket": metrics.pop("horizon_bucket"),
                        "feature_set": feature_set,
                        "model": model_name,
                        "model_name": model_name,
                        **metrics,
                        "baseline_model": "seasonal_naive",
                        "baseline_wape": float(baseline_metrics["wape"]),
                        "fva": float(baseline_metrics["wape"] - metrics["wape"]),
                        "fva_pct": float(
                            (baseline_metrics["wape"] - metrics["wape"])
                            / max(float(baseline_metrics["wape"]), 1e-9)
                        ),
                        "status": "evaluated",
                        "converged": None if metadata is None else metadata.converged,
                        "warning": None if metadata is None else metadata.warning,
                    }
                    rows.append(model_row)
                    for week, actual, value in zip(
                        future_weeks,
                        test_frame[target].to_numpy(dtype=float),
                        predicted,
                    ):
                        residual_rows.append(
                            {
                                "cutoff": origin,
                                "week": week,
                                "horizon": horizon,
                                "horizon_weeks": horizon,
                                "horizon_bucket": horizon_bucket(horizon),
                                "feature_set": feature_set,
                                "model": model_name,
                                "model_name": model_name,
                                "actual": float(actual),
                                "predicted": float(value),
                                "residual": float(actual - value),
                            }
                        )

    performance_columns = [
        "cutoff",
        "horizon",
        "horizon_weeks",
        "horizon_bucket",
        "feature_set",
        "model",
        "model_name",
        "wape",
        "mae",
        "mase",
        "rmse",
        "bias",
        "baseline_model",
        "baseline_wape",
        "fva",
        "fva_pct",
        "status",
        "converged",
        "warning",
    ]
    result = pd.DataFrame(rows)
    for column in performance_columns:
        if column not in result.columns:
            result[column] = None
    result = result[performance_columns]
    residuals = pd.DataFrame(
        residual_rows,
        columns=[
            "cutoff",
            "week",
            "horizon",
            "horizon_weeks",
            "horizon_bucket",
            "feature_set",
            "model",
            "model_name",
            "actual",
            "predicted",
            "residual",
        ],
    )
    if result.empty:
        fva_summary = pd.DataFrame(
            columns=["feature_set", "model", "horizon_bucket", "wape", "fva", "fva_pct", "evaluations"]
        )
    else:
        fva_summary = (
            result.groupby(["feature_set", "model", "horizon_bucket"], as_index=False, observed=True)
            .agg(
                wape=("wape", "mean"),
                fva=("fva", "mean"),
                fva_pct=("fva_pct", "mean"),
                evaluations=("wape", "size"),
            )
        )
    _write(
        ctx,
        "model_performance",
        result,
        "gold",
        "DFSIGNAL_EVALUATION",
        mode="replace",
        key_columns=("cutoff", "horizon", "feature_set", "model"),
    )
    _write(
        ctx,
        "backtest_residuals",
        residuals,
        "gold",
        "DFSIGNAL_EVALUATION",
        mode="replace",
        key_columns=("cutoff", "week", "horizon", "feature_set", "model"),
    )
    _write(
        ctx,
        "feature_value_add",
        fva_summary,
        "gold",
        "DFSIGNAL_EVALUATION",
        mode="replace",
        key_columns=("feature_set", "model", "horizon_bucket"),
    )
    ctx.artifacts["backtest_residuals"] = residuals
    ctx.artifacts["fva"] = fva_summary
    ctx.artifacts["backtest"] = result
    return result


def train(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Prepare deterministic splits and persist model-run diagnostics."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    feature_frame = _load_features(ctx)
    target = _target_column(ctx, feature_frame)
    valid = feature_frame.dropna(subset=["week", target]).sort_values("week").reset_index(drop=True)
    if valid.empty:
        raise ValueError("No observed feature rows are available for training")
    horizons = _horizons(ctx)
    holdout_size = min(max(horizons), max(1, len(valid) // 5))
    if len(valid) <= holdout_size:
        raise ValueError("Not enough complete feature rows for a training split")
    train_frame = valid.iloc[:-holdout_size].copy()
    holdout = valid.iloc[-holdout_size:].copy()

    transformed = _load_transformed(ctx)
    feature_sets = _feature_set_columns(ctx, transformed, feature_frame)
    forecast_feature_set = str(
        _model_config(ctx).get("forecast_feature_set", "gpu_signal")
    )
    model_config = _model_config(ctx)
    configured_models = model_config.get("models", {}) if isinstance(model_config, Mapping) else {}
    if not isinstance(configured_models, Mapping) or not configured_models:
        configured_models = {"seasonal_naive": True, "ets": True, "lightgbm": True}
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata_rows: list[dict[str, Any]] = []
    for model_name in ("seasonal_naive", "ets", "lightgbm"):
        enabled = bool(configured_models.get(model_name, False))
        if not enabled:
            metadata_rows.append(
                {
                    "model_name": model_name,
                    "model": model_name,
                    "enabled": False,
                    "status": "disabled",
                    "training_rows": 0,
                    "holdout_rows": 0,
                    "interval_available": False,
                    "trained_at": None,
                    "converged": None,
                    "warning": None,
                }
            )
            continue
        series = _history_series(train_frame, target)
        metadata: ModelRunMetadata
        if model_name == "ets":
            ets_result = ets_forecast_with_metadata(
                series,
                max(horizons),
                target=target,
                forecast_origin=pd.Timestamp(train_frame["week"].iloc[-1]),
            )
            if ets_result.metadata is None:
                raise RuntimeError("ETS result did not include run metadata")
            metadata = ets_result.metadata
            status = "ready" if ets_result.converged else "warning"
            extra = {
                "model_warning": ets_result.model_warning,
                "used_fallback": ets_result.used_fallback,
            }
        else:
            metadata = ModelRunMetadata.from_training_bounds(
                model_name=model_name,
                target=target,
                training_bounds=(series.index.min(), series.index.max()),
                forecast_origin=pd.Timestamp(train_frame["week"].iloc[-1]),
                horizon=max(horizons),
                parameters={
                    "feature_set": forecast_feature_set,
                    "feature_columns": feature_sets.get(forecast_feature_set, []),
                },
                converged=True,
            )
            status = "ready"
            extra = {"model_warning": False, "used_fallback": False}
        metadata_rows.append(
            _metadata_row(
                metadata,
                model=model_name,
                model_name=model_name,
                enabled=True,
                status=status,
                training_rows=len(train_frame),
                holdout_rows=len(holdout),
                interval_available=False,
                trained_at=trained_at,
                **extra,
            )
        )
    model_status = pd.DataFrame(metadata_rows)
    _write(
        ctx,
        "model_training",
        model_status,
        "gold",
        "DFSIGNAL_TRAIN",
        mode="replace",
        key_columns=("model_name",),
    )
    result = {
        "features": feature_frame,
        "valid": valid,
        "train": train_frame,
        "holdout": holdout,
        "models": model_status,
    }
    ctx.artifacts["train"] = result
    return result


def _future_frame(
    train_frame: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    horizon: int = 4,
) -> pd.DataFrame:
    """Return unlabeled weeks after the latest observed training week.

    ``holdout`` remains accepted for compatibility with older callers but is
    intentionally ignored; a holdout row is never a future forecast feature.
    """

    del holdout
    if train_frame.empty:
        raise ValueError("Cannot create forecast horizon without training rows")
    if "week" not in train_frame.columns:
        raise ValueError("train_frame is missing required column: week")
    latest = pd.to_datetime(train_frame["week"], errors="raise").max()
    return pd.DataFrame({"week": [latest + pd.Timedelta(weeks=i) for i in range(1, int(horizon) + 1)]})

def _load_backtest_residuals(context: PipelineContext) -> pd.DataFrame:
    artifact = context.artifacts.get("backtest_residuals")
    if isinstance(artifact, pd.DataFrame):
        return artifact
    if context.storage.table_exists("backtest_residuals", "gold"):
        return context.storage.read_table("backtest_residuals", "gold")
    return pd.DataFrame()


def _calibration_residuals(
    residuals: pd.DataFrame,
    *,
    model_name: str,
    horizon: int,
    feature_set: str,
) -> np.ndarray:
    if residuals.empty or "residual" not in residuals.columns:
        return np.empty(0, dtype=float)
    selected = residuals.copy()
    if "model" in selected.columns:
        selected = selected[selected["model"].astype(str).eq(model_name)]
    elif "model_name" in selected.columns:
        selected = selected[selected["model_name"].astype(str).eq(model_name)]
    if "horizon" in selected.columns:
        horizon_rows = selected[selected["horizon"].astype(int).eq(horizon)]
        if not horizon_rows.empty:
            selected = horizon_rows
    if "feature_set" in selected.columns:
        exact = selected[selected["feature_set"].astype(str).eq(feature_set)]
        if not exact.empty:
            selected = exact
    values = pd.to_numeric(selected["residual"], errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def forecast(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Generate true future forecasts and residual/conformal intervals."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    trained = _load_train(ctx)
    feature_frame = trained["features"].copy()
    target = _target_column(ctx, feature_frame)
    observed = feature_frame.dropna(subset=["week", target]).copy()
    observed["week"] = pd.to_datetime(observed["week"], errors="raise")
    observed = observed.sort_values("week").reset_index(drop=True)
    if observed.empty:
        raise ValueError("Cannot forecast without observed demand")
    origin = pd.Timestamp(observed["week"].iloc[-1])
    transformed = _load_transformed(ctx)
    feature_sets = _feature_set_columns(ctx, transformed, feature_frame)
    configured_feature_set = str(
        _model_config(ctx).get("forecast_feature_set", "gpu_signal")
    )
    feature_set = configured_feature_set if configured_feature_set in feature_sets else "calendar"
    model_status = trained.get("models")
    enabled_models = set(_model_names(ctx))
    if isinstance(model_status, pd.DataFrame) and "enabled" in model_status.columns:
        enabled_models = set(
            model_status.loc[model_status["enabled"].astype(bool), "model_name"].astype(str)
        )
    if not enabled_models:
        raise ValueError("No forecasting models are enabled")
    interval_config = _model_config(ctx).get("intervals", {})
    coverage = float(interval_config.get("coverage", 0.9)) if isinstance(interval_config, Mapping) else 0.9
    if not 0.0 < coverage < 1.0:
        raise ValueError("interval coverage must be strictly between 0 and 1")
    residuals = _load_backtest_residuals(ctx)
    forecast_run_id = str(ctx.run_id or uuid4().hex)
    steam_frame = transformed.get("steam", pd.DataFrame())
    steam_last_observed = (
        pd.to_datetime(steam_frame["week"], errors="coerce").max()
        if isinstance(steam_frame, pd.DataFrame) and "week" in steam_frame.columns and not steam_frame.empty
        else None
    )
    steam_policy = str(
        _yaml(ctx, "steam_hardware.yaml").get("future_strategy", "OBSERVED_ONLY")
    ).upper()
    future_policy = json.dumps(
        {
            "steam_future_strategy": steam_policy,
            "steam_last_observed_week": steam_last_observed,
            "future_steam_values_fabricated": False,
        },
        default=str,
        sort_keys=True,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    for horizon in _horizons(ctx):
        future_weeks = [
            origin + pd.Timedelta(weeks=offset)
            for offset in range(1, horizon + 1)
        ]
        for model_name in sorted(enabled_models):
            prediction, metadata = _model_prediction(
                ctx,
                model_name,
                observed,
                feature_frame,
                future_weeks,
                target=target,
                columns=feature_sets[feature_set],
                feature_set=feature_set,
                transformed=transformed,
                origin=origin,
            )
            calibration = _calibration_residuals(
                residuals,
                model_name=model_name,
                horizon=horizon,
                feature_set=feature_set,
            )
            if calibration.size == 0:
                # A model-specific backtest row may be absent in a deliberately
                # reduced configuration; use another horizon for that same model
                # before declaring intervals unavailable.
                calibration = _calibration_residuals(
                    residuals,
                    model_name=model_name,
                    horizon=horizon,
                    feature_set="",
                )
            if calibration.size:
                lower, upper = build_conformal_intervals(
                    prediction,
                    calibration,
                    coverage,
                    nonnegative=True,
                )
                interval_available = True
                interval_width = float(np.mean(upper - lower))
            else:
                lower = np.full(len(prediction), np.nan, dtype=float)
                upper = np.full(len(prediction), np.nan, dtype=float)
                interval_available = False
                interval_width = float("nan")
            metadata_rows.append(
                _metadata_row(
                    metadata,
                    forecast_run_id=forecast_run_id,
                    feature_set=feature_set,
                    interval_available=interval_available,
                    interval_calibration_coverage=coverage if interval_available else None,
                    interval_width=interval_width if interval_available else None,
                    future_external_policy=future_policy,
                    steam_future_strategy=steam_policy,
                    steam_last_observed_week=steam_last_observed,
                )
            )
            for lead, (week, value, lower_value, upper_value) in enumerate(
                zip(future_weeks, prediction, lower, upper),
                start=1,
            ):
                rows.append(
                    {
                        "forecast_run_id": forecast_run_id,
                        "origin": origin,
                        "horizon_week": pd.Timestamp(week),
                        "model": model_name,
                        "target": target,
                        "value": float(value),
                        "lower_bound": float(lower_value) if np.isfinite(lower_value) else np.nan,
                        "upper_bound": float(upper_value) if np.isfinite(upper_value) else np.nan,
                        "horizon": horizon,
                        "lead_week": lead,
                        "feature_set": feature_set,
                        "forecast_origin": origin,
                        "week": pd.Timestamp(week),
                        "horizon_weeks": horizon,
                        "model_name": model_name,
                        "point_forecast": float(value),
                        "interval_available": interval_available,
                        "interval_calibration_coverage": coverage if interval_available else np.nan,
                        "interval_width": interval_width if interval_available else np.nan,
                        "converged": None if metadata is None else metadata.converged,
                        "warning": None if metadata is None else metadata.warning,
                        "model_run_id": None if metadata is None else metadata.run_id,
                        "created_at": created_at,
                        "source_id": "DFSIGNAL_MODEL",
                    }
                )
    forecast_columns = [
        "forecast_run_id",
        "origin",
        "horizon_week",
        "model",
        "target",
        "value",
        "lower_bound",
        "upper_bound",
        "horizon",
        "lead_week",
        "feature_set",
        "forecast_origin",
        "week",
        "horizon_weeks",
        "model_name",
        "point_forecast",
        "interval_available",
        "interval_calibration_coverage",
        "interval_width",
        "converged",
        "warning",
        "model_run_id",
        "created_at",
        "source_id",
    ]
    result = pd.DataFrame(rows, columns=forecast_columns)
    _write(
        ctx,
        "forecast",
        result,
        "gold",
        "DFSIGNAL_MODEL",
        mode="replace",
        key_columns=("forecast_run_id", "horizon_week", "horizon", "model"),
    )
    metadata_frame = pd.DataFrame(metadata_rows)
    if not metadata_frame.empty:
        _write(
            ctx,
            "model_run_metadata",
            metadata_frame,
            "gold",
            "DFSIGNAL_MODEL",
            mode="replace",
            key_columns=("run_id", "forecast_run_id", "horizon"),
        )
    ctx.artifacts["forecast_model_metadata"] = metadata_frame
    ctx.artifacts["forecast"] = result
    return result


def explain(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Persist LightGBM model-attributed SHAP drivers."""

    ctx = _context(
        context,
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    trained = _load_train(ctx)
    valid = trained["valid"]
    transformed = _load_transformed(ctx)
    target = _target_column(ctx, valid)
    feature_sets = _feature_set_columns(ctx, transformed, valid)
    feature_set = str(_model_config(ctx).get("forecast_feature_set", "gpu_signal"))
    columns = feature_sets.get(feature_set, feature_sets.get("calendar", []))
    model_input = _model_frame(valid, columns, target)
    drivers = explain_lightgbm(model_input, target=target)
    drivers = drivers.copy()
    drivers.insert(0, "model_name", "lightgbm")
    drivers.insert(1, "feature_set", feature_set)
    drivers["interval_available"] = False
    drivers["created_at"] = datetime.now(timezone.utc).isoformat()
    drivers["source_id"] = "DFSIGNAL_SHAP"
    _write(
        ctx,
        "forecast_drivers",
        drivers,
        "gold",
        "DFSIGNAL_SHAP",
        mode="replace",
        key_columns=("model_name", "feature_name"),
    )
    ctx.artifacts["explain"] = drivers
    return drivers


def status(
    context: PipelineContext | StorageBackend | str | Path | None = None,
    *,
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    storage: StorageBackend | None = None,
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Return pipeline, source, model, forecast, and data recency health."""

    ctx = _context(context, root_dir=root_dir, output_dir=output_dir, storage=storage, fetch_external=fetch_external, config_dir=config_dir)
    refresh = getattr(ctx.storage, "read_source_refresh", None)
    refresh_frame = refresh() if callable(refresh) else pd.DataFrame()
    source_tables = {
        SOURCE_IDS["fred"]: ("fred_weekly_raw", "bronze"),
        SOURCE_IDS["gpu"]: ("gpu_specs_raw", "bronze"),
        SOURCE_IDS["steam"]: ("steam_hardware", "bronze"),
    }
    sources: dict[str, dict[str, Any]] = {}
    for source_id, (table_name, layer) in source_tables.items():
        row = refresh_frame[refresh_frame["source_id"].astype(str).eq(source_id)].iloc[-1].to_dict() if not refresh_frame.empty and "source_id" in refresh_frame.columns and not refresh_frame[refresh_frame["source_id"].astype(str).eq(source_id)].empty else None
        if row is None and ctx.storage.table_exists(table_name, layer):
            metadata = ctx.storage.read_metadata(table_name, layer)
            row = {"status": "available", **metadata}
        if row is None:
            row = {"status": "not_run", "source_id": source_id}
        sources[source_id] = {
            "status": _json_value(row.get("status", "not_run")),
            "updated_at": _json_value(row.get("updated_at") or row.get("refreshed_at") or row.get("load_timestamp")),
            "row_count": _json_value(row.get("row_count")),
            "latest_data_week": _json_value(row.get("latest_data_week") or row.get("max_date")),
            "error": _json_value(row.get("error")),
        }

    runs_reader = getattr(ctx.storage, "read_pipeline_runs", None)
    runs_frame = runs_reader() if callable(runs_reader) else pd.DataFrame()
    runs: list[dict[str, Any]] = []
    if not runs_frame.empty:
        for _, row in runs_frame.sort_values("started_at", na_position="first").iterrows():
            record = {str(key): _json_value(value) for key, value in row.to_dict().items()}
            if isinstance(record.get("stages"), str):
                try:
                    record["stages"] = json.loads(record["stages"])
                except json.JSONDecodeError:
                    pass
            runs.append(record)
    latest_run = runs[-1] if runs else None
    if latest_run is None:
        health = "not_run"
    elif latest_run.get("status") == "success":
        health = "healthy"
    else:
        health = "failed"
    if sources[SOURCE_IDS["steam"]]["status"] == "unavailable" and health == "healthy":
        health = "degraded"

    model_rows = pd.DataFrame()
    if ctx.storage.table_exists("model_training", "gold"):
        model_rows = ctx.storage.read_table("model_training", "gold")
    forecast_frame = (
        ctx.storage.read_table("forecast", "gold") if ctx.storage.table_exists("forecast", "gold") else pd.DataFrame()
    )
    models: dict[str, dict[str, Any]] = {}
    for model_name in ("seasonal_naive", "ets", "lightgbm"):
        selected = model_rows[model_rows["model_name"].astype(str).eq(model_name)] if not model_rows.empty and "model_name" in model_rows.columns else pd.DataFrame()
        forecast_selected = forecast_frame[forecast_frame["model_name"].astype(str).eq(model_name)] if not forecast_frame.empty and "model_name" in forecast_frame.columns else pd.DataFrame()
        if selected.empty and forecast_selected.empty:
            models[model_name] = {"status": "not_run", "interval_available": False}
        else:
            row = selected.iloc[-1].to_dict() if not selected.empty else {}
            models[model_name] = {
                "status": _json_value(row.get("status", "available")),
                "enabled": _json_value(row.get("enabled", True)),
                "training_rows": _json_value(row.get("training_rows")),
                "last_trained_at": _json_value(row.get("trained_at")),
                "forecast_rows": int(len(forecast_selected)),
                "interval_available": bool(
                    not forecast_selected.empty
                    and {"lower_bound", "upper_bound"}.issubset(forecast_selected.columns)
                    and forecast_selected[["lower_bound", "upper_bound"]].notna().all().all()
                ),
            }

    horizons = sorted(forecast_frame["horizon_weeks"].dropna().astype(int).unique().tolist()) if not forecast_frame.empty and "horizon_weeks" in forecast_frame.columns else []
    intervals_available = bool(
        not forecast_frame.empty
        and {"lower_bound", "upper_bound"}.issubset(forecast_frame.columns)
        and forecast_frame[["lower_bound", "upper_bound"]].notna().all().all()
    )
    latest_data: dict[str, str | None] = {}
    for table in ctx.storage.list_tables() if callable(getattr(ctx.storage, "list_tables", None)) else []:
        latest_data[str(table["table_name"])] = _json_value(table.get("latest_data_week") or table.get("max_date"))

    report = {
        "pipeline": {
            "health": health,
            "run_count": len(runs),
            "latest_run": latest_run,
            "runs": runs,
        },
        "sources": sources,
        "models": models,
        "forecast": {
            "horizons_weeks": horizons,
            "intervals_available": intervals_available,
            "row_count": int(len(forecast_frame)),
            "latest_forecast_week": _date_bound(forecast_frame, maximum=True),
        },
        "latest_data_weeks": latest_data,
    }
    ctx.artifacts["status"] = report
    return report


def _json_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _export_outputs(context: PipelineContext, artifacts: Mapping[str, Any]) -> None:
    exporter = getattr(context.storage, "export_table", None)
    if not callable(exporter):
        return
    for table_name, artifact_key in (
        ("feature_dataset", "features"),
        ("forecast", "forecast"),
        ("backtest", "backtest"),
        ("shap_drivers", "explain"),
    ):
        frame = artifacts.get(artifact_key)
        if isinstance(frame, pd.DataFrame):
            exporter(table_name, frame, context.output_dir)


def run_all(
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    fetch_external: bool = True,
    *,
    storage: StorageBackend | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    config_dir: str | Path = "config",
) -> dict[str, Any]:
    """Run every stage through the same independently callable stage functions."""

    owns_storage = storage is None
    ctx = make_context(
        root_dir=root_dir,
        output_dir=output_dir,
        storage=storage,
        start_date=start_date,
        end_date=end_date,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )
    try:
        run = run_pipeline(ctx, STAGE_ORDER, raise_on_failure=True)
        _export_outputs(ctx, ctx.artifacts)
        health = status(ctx)
        return {
            "features": ctx.artifacts["features"],
            "forecast": ctx.artifacts["forecast"],
            "backtest": ctx.artifacts["backtest"],
            "drivers": ctx.artifacts["explain"],
            "run": run.as_dict(),
            "status": health,
        }
    except PipelineRunError:
        raise
    finally:
        if owns_storage:
            ctx.storage.close()


def run_demo() -> dict[str, Any]:
    """Compatibility wrapper for the legacy local demo command."""

    return run_all()


__all__ = [
    "PipelineContext",
    "STAGE_DEPENDENCIES",
    "STAGE_ORDER",
    "backtest",
    "explain",
    "features",
    "forecast",
    "ingest",
    "run_all",
    "run_demo",
    "run_stage",
    "status",
    "train",
    "transform",
    "validate",
]
