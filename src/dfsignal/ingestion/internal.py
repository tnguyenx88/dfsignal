"""Internal-demand landing, mapping, provenance, and validation boundaries.

This module deliberately does not run an internal-data experiment. It only
prepares a safe adapter for a future approved de-identified extract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import load_yaml


CANONICAL_COLUMNS = (
    "week",
    "business_segment",
    "product_group",
    "product_family",
    "region",
    "channel",
    "pos_qty",
    "pos_revenue",
    "shipment_qty",
    "shipment_revenue",
    "booking_qty",
    "booking_revenue",
    "asp",
    "inventory_qty",
)
CANONICAL_DIMENSIONS = (
    "business_segment",
    "product_group",
    "product_family",
    "region",
    "channel",
)
CANONICAL_MEASURES = (
    "pos_qty",
    "pos_revenue",
    "shipment_qty",
    "shipment_revenue",
    "booking_qty",
    "booking_revenue",
    "asp",
    "inventory_qty",
)
DEMAND_TARGETS = ("pos_qty", "shipment_qty", "booking_qty")
SENSITIVE_COLUMN_TOKENS = (
    "customer",
    "email",
    "address",
    "employee",
    "order_number",
    "order_no",
    "serial",
    "transaction",
)
SUPPORTED_SUFFIXES = frozenset({".csv", ".parquet"})


@dataclass(frozen=True)
class InternalValidationResult:
    """Safe, row-count-only validation result for a future internal file."""

    status: str
    source_status: str
    checks: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_status": self.source_status,
            "checks": list(self.checks),
            "provenance": dict(self.provenance) if self.provenance else None,
        }


class InternalDemandSource:
    """Protocol-like base for future local internal-demand adapters."""

    def read(self) -> pd.DataFrame:
        raise NotImplementedError


class SyntheticDemandSource:
    """Generate realistic-shaped, clearly labelled synthetic weekly demand."""

    def __init__(self, start: str = "2016-01-01", end: str = "2025-12-31", seed: int = 7) -> None:
        self.start, self.end, self.seed = start, end, seed

    def read(self) -> pd.DataFrame:
        dates = pd.date_range(self.start, self.end, freq="W-SAT")
        rng = np.random.default_rng(self.seed)
        index = np.arange(len(dates))
        week = dates.isocalendar().week.to_numpy()
        annual = 16 * np.sin(2 * np.pi * week / 52)
        holiday = np.where(dates.month == 12, 22, 0)
        hardware_cycle = 10 * np.sin(2 * np.pi * index / 104)
        demand = np.maximum(10, 120 + index * 0.10 + annual + holiday + hardware_cycle + rng.normal(0, 4, len(dates)))
        return pd.DataFrame(
            {
                "week": dates,
                "business_family": "Memory / PC Components",
                "region": "Global",
                "channel": "Synthetic",
                "pos_qty": demand.round(3),
                "source_id": "SYNTHETIC",
            }
        )


class LocalCSVInternalDemandSource(InternalDemandSource):
    """Read and map one approved CSV or Parquet internal extract."""

    def __init__(self, path: str | Path, config_path: str | Path = "config/internal_data.yaml") -> None:
        self.path = Path(path)
        self.config_path = Path(config_path)

    def read(self) -> pd.DataFrame:
        config = load_internal_config(self.config_path)
        raw = read_internal_file(self.path)
        return map_internal_columns(raw, config)


LocalParquetInternalDemandSource = LocalCSVInternalDemandSource
LocalInternalDemandSource = LocalCSVInternalDemandSource


def load_internal_config(path: str | Path = "config/internal_data.yaml") -> dict[str, Any]:
    """Load and minimally validate the internal source configuration."""

    config = load_yaml(path)
    source = config.get("source", {})
    if not isinstance(source, Mapping):
        raise ValueError("internal_data.yaml source must be a mapping")
    if str(source.get("source_id", "")).strip() != "internal_demand":
        raise ValueError("internal_data.yaml source.source_id must be internal_demand")
    if bool(source.get("enabled", False)):
        raise ValueError("internal_demand must remain disabled during preparation")
    target = config.get("target", {})
    if not isinstance(target, Mapping):
        raise ValueError("internal_data.yaml target must be a mapping")
    preferred = _canonical_name(target.get("preferred", "pos_qty"))
    alternatives = tuple(_canonical_name(value) for value in target.get("alternatives", ()))
    configured = (preferred, *alternatives)
    invalid = sorted(set(configured) - set(DEMAND_TARGETS))
    if invalid:
        raise ValueError(f"Unsupported internal target columns: {invalid}")
    return config


def discover_internal_file(
    config_path: str | Path = "config/internal_data.yaml",
    *,
    root_dir: str | Path = ".",
) -> Path | None:
    """Return the first supported file under the configured landing pattern."""

    config = load_internal_config(config_path)
    source = config["source"]
    pattern = str(source.get("path", "data/internal/*"))
    candidate_root = Path(root_dir)
    matches = sorted(
        path
        for path in candidate_root.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    return matches[0] if matches else None


def read_internal_file(path: str | Path) -> pd.DataFrame:
    """Read one supported source file without retaining source-only columns."""

    source_path = Path(path)
    if source_path.suffix.lower() == ".csv":
        return pd.read_csv(source_path)
    if source_path.suffix.lower() == ".parquet":
        return pd.read_parquet(source_path)
    raise ValueError(f"Unsupported internal file format: {source_path.suffix or '<none>'}")


def map_internal_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Map configured source columns into the canonical internal schema.

    Unmapped and source-only columns are intentionally dropped. This prevents
    identifiers or other sensitive fields from flowing into Silver/Gold by
    accident.
    """

    mappings: dict[str, Any] = {}
    date_column = config.get("date_column")
    week_column = config.get("week_column")
    if week_column:
        mappings["week"] = week_column
    elif date_column:
        mappings["week"] = date_column
    for section in ("dimensions", "measures"):
        values = config.get(section, {})
        if isinstance(values, Mapping):
            mappings.update({str(key): value for key, value in values.items() if value})

    result = pd.DataFrame(index=frame.index)
    for canonical in CANONICAL_COLUMNS:
        source_column = mappings.get(canonical)
        if source_column and source_column in frame.columns and not _is_sensitive_column(source_column):
            result[canonical] = frame[source_column]

    if "week" not in result.columns:
        raise ValueError("Internal demand mapping requires week_column or date_column")
    parsed_week = pd.to_datetime(result["week"], errors="coerce")
    if isinstance(parsed_week.dtype, pd.DatetimeTZDtype):
        parsed_week = parsed_week.dt.tz_localize(None)
    elif pd.api.types.is_object_dtype(parsed_week):
        parsed_week = pd.to_datetime(result["week"], errors="coerce", utc=True).dt.tz_localize(None)
    result["week"] = parsed_week
    return result


def resolve_target(config: Mapping[str, Any], columns: Any) -> str:
    """Resolve the configured target without hardcoding a modeling target."""

    target = config.get("target", {})
    if not isinstance(target, Mapping):
        raise ValueError("Internal target configuration must be a mapping")
    preferred = _canonical_name(target.get("preferred"))
    alternatives = tuple(_canonical_name(value) for value in target.get("alternatives", ()))
    available = {str(column).lower() for column in columns}
    for candidate in (preferred, *alternatives):
        if candidate and candidate in available:
            return candidate
    configured = ", ".join(filter(None, (preferred, *alternatives)))
    raise ValueError(f"No configured internal target is present; expected one of: {configured}")


def build_internal_provenance(
    path: str | Path,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    extract_created_at: str | datetime | None = None,
    source_as_of_timestamp: str | datetime | None = None,
    load_timestamp: str | datetime | None = None,
) -> dict[str, Any]:
    """Build provenance metadata without requiring a person-name owner."""

    source_path = Path(path)
    week = pd.to_datetime(frame.get("week"), errors="coerce") if "week" in frame else pd.Series(dtype="datetime64[ns]")
    source = config.get("source", {})
    target = config.get("target", {})
    grain = config.get("grain", {}).get("preferred", "WEEK x PRODUCT_GROUP") if isinstance(config.get("grain"), Mapping) else "WEEK x PRODUCT_GROUP"
    return {
        "source_id": str(source.get("source_id", "internal_demand")),
        "source_name": str(source.get("source_name", "Approved de-identified internal demand extract")),
        "filename": source_path.name,
        "file_hash": _sha256(source_path),
        "extract_created_at": _iso_timestamp(extract_created_at),
        "source_as_of_timestamp": _iso_timestamp(source_as_of_timestamp),
        "load_timestamp": _iso_timestamp(load_timestamp) or datetime.now(timezone.utc).isoformat(),
        "min_business_date": _date_value(week.min() if not week.empty else None),
        "max_business_date": _date_value(week.max() if not week.empty else None),
        "row_count": int(len(frame)),
        "grain": str(grain),
        "data_classification": str(source.get("data_classification", "CONFIDENTIAL_INTERNAL_DE_IDENTIFIED")),
        "configured_target": str(target.get("preferred", "")),
    }


def validate_internal_source(
    path: str | Path | None = None,
    *,
    config_path: str | Path = "config/internal_data.yaml",
    model_config_path: str | Path = "config/models.yaml",
    root_dir: str | Path = ".",
    now: date | datetime | None = None,
) -> InternalValidationResult:
    """Validate a future internal file, safely treating absence as optional."""

    config = load_internal_config(config_path)
    source_path = Path(path) if path else discover_internal_file(config_path, root_dir=root_dir)
    if source_path is None:
        return InternalValidationResult(
            status="MISSING_OPTIONAL",
            source_status="MISSING_OPTIONAL",
            checks=({"check": "file_exists", "status": "REVIEW", "observed": False, "rule": "optional internal source may be absent"},),
        )

    checks: list[dict[str, Any]] = []
    if not source_path.is_file():
        checks.append({"check": "file_exists", "status": "FAIL", "observed": str(source_path), "rule": "configured file exists"})
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))
    suffix = source_path.suffix.lower()
    checks.append({"check": "file_exists", "status": "PASS", "observed": True, "rule": "configured file exists"})
    if suffix not in SUPPORTED_SUFFIXES:
        checks.append({"check": "supported_format", "status": "FAIL", "observed": suffix, "rule": "file suffix is .csv or .parquet"})
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))
    checks.append({"check": "supported_format", "status": "PASS", "observed": suffix, "rule": "file suffix is .csv or .parquet"})

    try:
        raw = read_internal_file(source_path)
        mapped = map_internal_columns(raw, config)
    except (OSError, ValueError, TypeError) as exc:
        checks.append({"check": "required_columns", "status": "FAIL", "observed": str(exc), "rule": "week, product_group, and one configured target"})
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))
    if mapped.empty:
        checks.append({"check": "non_empty", "status": "FAIL", "observed": 0, "rule": "approved internal extract contains at least one mapped row"})
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))

    required = {"week", "product_group"}
    try:
        model_config = load_yaml(model_config_path)
        target = resolve_target(config, mapped.columns)
    except (OSError, ValueError, TypeError) as exc:
        checks.append({"check": "required_columns", "status": "FAIL", "observed": str(exc), "rule": "week, product_group, and one configured target"})
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))
    required.add(target)
    missing = sorted(required - set(mapped.columns))
    checks.append({"check": "required_columns", "status": "PASS" if not missing else "FAIL", "observed": missing or sorted(required), "rule": "week, product_group, and one configured target"})
    if missing:
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))

    valid_dates = mapped["week"].notna()
    checks.append({"check": "valid_dates", "status": "PASS" if bool(valid_dates.all()) else "FAIL", "observed": int((~valid_dates).sum()), "rule": "week parses as a date"})
    if not valid_dates.all():
        return InternalValidationResult("FAIL", "INVALID", tuple(checks))
    non_saturday = int((mapped["week"].dt.weekday != 5).sum())
    checks.append({"check": "weekly_dates", "status": "REVIEW" if non_saturday else "PASS", "observed": non_saturday, "rule": "all canonical week values are completed Saturday weeks"})

    nullable_required = mapped[list(required)].isna().sum()
    null_required = {key: int(value) for key, value in nullable_required.items() if int(value) > 0}
    checks.append({"check": "unexpected_nulls", "status": "FAIL" if null_required else "PASS", "observed": null_required or 0, "rule": "required fields are non-null"})

    grain_columns = _grain_columns(mapped, config)
    duplicate_count = int(mapped.duplicated(subset=grain_columns, keep=False).sum()) if grain_columns else 0
    checks.append({"check": "duplicate_grain_keys", "status": "FAIL" if duplicate_count else "PASS", "observed": duplicate_count, "rule": "grain keys are unique"})

    numeric = pd.to_numeric(mapped[target], errors="coerce")
    target_invalid = int(numeric.isna().sum())
    checks.append({"check": "target_numeric", "status": "FAIL" if target_invalid else "PASS", "observed": target_invalid, "rule": "configured target is numeric"})
    negative_target = int((numeric < 0).sum()) if target_invalid < len(numeric) else 0
    checks.append({"check": "negative_quantities", "status": "FAIL" if negative_target else "PASS", "observed": negative_target, "rule": "configured demand quantity is non-negative"})

    all_negative: dict[str, int] = {}
    for measure in CANONICAL_MEASURES:
        if measure in mapped.columns:
            values = pd.to_numeric(mapped[measure], errors="coerce")
            count = int((values < 0).sum())
            if count:
                all_negative[measure] = count
    if all_negative and target not in all_negative:
        negative_status = "REVIEW"
    elif all_negative:
        negative_status = "FAIL"
    else:
        negative_status = "PASS"
    checks[-1]["observed"] = {"target": negative_target, "all_measures": all_negative or 0}
    checks[-1]["status"] = "FAIL" if negative_target else negative_status
    sorted_weeks = mapped["week"].drop_duplicates().sort_values()
    missing_weeks = int(max(0, ((sorted_weeks.iloc[-1] - sorted_weeks.iloc[0]).days // 7 + 1) - len(sorted_weeks))) if len(sorted_weeks) > 1 else 0
    checks.append({"check": "missing_weeks", "status": "REVIEW" if missing_weeks else "PASS", "observed": missing_weeks, "rule": "weekly date sequence has no gaps"})

    max_week = mapped["week"].max()
    current = pd.Timestamp(now or datetime.now(timezone.utc).date())
    if current.tzinfo is not None:
        current = current.tz_convert("UTC").tz_localize(None)
    current = current.normalize()
    days_since_completed = 7 if current.dayofweek == 5 else (current.dayofweek - 5) % 7
    last_completed_saturday = current - pd.Timedelta(days=days_since_completed)
    partial = max_week.weekday() != 5 or max_week.normalize() > last_completed_saturday
    checks.append({"check": "partial_latest_week", "status": "REVIEW" if partial else "PASS", "observed": str(max_week.date()), "rule": "latest week is a completed Saturday week"})

    sensitive = sorted(column for column in raw.columns if _is_sensitive_column(column))
    unknown = sorted(column for column in raw.columns if column not in _mapped_source_columns(config))
    checks.append({"check": "schema_drift", "status": "REVIEW" if unknown else "PASS", "observed": unknown, "rule": "source-only columns are reviewed and never mapped implicitly"})
    checks.append({"check": "deidentification_guardrail", "status": "REVIEW" if sensitive else "PASS", "observed": sensitive, "rule": "customer/person/order identifiers are not needed and are dropped"})

    optional_nulls = {column: int(mapped[column].isna().sum()) for column in CANONICAL_COLUMNS if column in mapped.columns and column not in required and mapped[column].isna().any()}
    checks.append({"check": "optional_nulls", "status": "REVIEW" if optional_nulls else "PASS", "observed": optional_nulls or 0, "rule": "optional fields may be null but are reported"})

    allowed_groups = config.get("allowed_product_groups", ())
    unexpected_groups = []
    if allowed_groups:
        expected = {str(value) for value in allowed_groups}
        unexpected_groups = sorted(set(mapped["product_group"].dropna().astype(str)) - expected)
    checks.append({"check": "product_group_values", "status": "REVIEW" if unexpected_groups else "PASS", "observed": unexpected_groups or 0, "rule": "product-group values match configured vocabulary when provided"})

    provenance = build_internal_provenance(source_path, mapped, config)
    statuses = [str(check["status"]) for check in checks]
    overall = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
    configured_model_target = model_config.get("target", {}).get("column") if isinstance(model_config.get("target"), Mapping) else None
    if configured_model_target and str(configured_model_target).lower() != target:
        checks.append({"check": "model_target_alignment", "status": "REVIEW", "observed": {"internal": target, "model": configured_model_target}, "rule": "model target is explicitly aligned before execution"})
        overall = "REVIEW" if overall == "PASS" else overall
    return InternalValidationResult(overall, "AVAILABLE", tuple(checks), provenance)


def _canonical_name(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def _mapped_source_columns(config: Mapping[str, Any]) -> set[str]:
    result = set()
    for value in (config.get("date_column"), config.get("week_column")):
        if value:
            result.add(str(value))
    for section in ("dimensions", "measures"):
        values = config.get(section, {})
        if isinstance(values, Mapping):
            result.update(str(value) for value in values.values() if value)
    return result


def _grain_columns(frame: pd.DataFrame, config: Mapping[str, Any]) -> list[str]:
    configured = config.get("grain", {}).get("keys") if isinstance(config.get("grain"), Mapping) else None
    if configured:
        keys = [str(value) for value in configured if str(value) in frame.columns]
        keys.extend(
            column
            for column in ("business_segment", "product_family", "region", "channel")
            if column in frame.columns and column not in keys
        )
    else:
        keys = ["week", "product_group"]
        keys.extend(column for column in ("business_segment", "product_family", "region", "channel") if column in frame.columns)
    return list(dict.fromkeys(keys))


def _is_sensitive_column(column: Any) -> bool:
    normalized = str(column).strip().lower().replace(" ", "_")
    return any(token in normalized for token in SENSITIVE_COLUMN_TOKENS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso_timestamp(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return stamp.isoformat()
    return str(value)


def _date_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()
