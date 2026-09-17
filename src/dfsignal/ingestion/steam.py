"""Optional, provenance-preserving Steam Hardware Survey ingestion.

Valve's survey is published as HTML and is intentionally not scraped here. This
adapter accepts one locally supplied CSV or Parquet historical extract, supports
long and common wide layouts, and normalizes them to a canonical monthly long
contract for the rest of DFSignal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import hashlib
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import load_yaml

DEFAULT_STEAM_HARDWARE_DIR = Path("data") / "external" / "steam_hardware"
DEFAULT_SOURCE_ID = "COMMUNITY_EXPORT"
SCHEMA_VERSION = "steam_hardware_canonical_v2"
SUPPORTED_SUFFIXES = frozenset({".csv", ".parquet"})
CANONICAL_COLUMNS = (
    "survey_month", "category", "metric_name", "metric_value", "unit",
    "vendor", "generation", "source_id", "source_name", "source_url",
    "original_filename", "file_hash", "load_timestamp", "row_count",
    "min_date", "max_date", "schema_version", "status", "source_change",
)
# Legacy names remain accepted and are emitted for callers of the original API.
LEGACY_COLUMNS = ("observation_date", "metric_name", "metric_value", "source_id", "source_url")


@dataclass(frozen=True, slots=True)
class SteamSourceMetadata:
    source_id: str | None
    source_name: str | None = None
    source_url: str | None = None
    original_filename: str | None = None
    file_hash: str | None = None
    load_timestamp: str | None = None
    row_count: int = 0
    min_date: date | None = None
    max_date: date | None = None
    schema_version: str = SCHEMA_VERSION
    status: str = "PASS"
    source_format: str | None = None
    enabled: bool = True
    metric_units: Mapping[str, str] = field(default_factory=dict)
    @property
    def observation_start(self) -> date | None:
        return self.min_date

    @property
    def observation_end(self) -> date | None:
        return self.max_date

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in ("min_date", "max_date"):
            if isinstance(result[key], date):
                result[key] = result[key].isoformat()
        result["date_coverage"] = {"start": result["min_date"], "end": result["max_date"]}
        result["metric_units"] = dict(self.metric_units)
        return result


def build_steam_source_metadata(
    frame: pd.DataFrame,
    *,
    source_id: str | None = DEFAULT_SOURCE_ID,
    source_name: str | None = None,
    source_url: str | None = None,
    source_path: str | Path | None = None,
    file_hash: str | None = None,
    load_timestamp: str | None = None,
    status: str = "PASS",
    enabled: bool = True,
    metric_units: Mapping[str, str] | None = None,
) -> SteamSourceMetadata:
    """Build metadata for a normalized frame without discarding provenance."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Steam source metadata requires a DataFrame")
    dates = pd.to_datetime(frame.get("survey_month", frame.get("observation_date")), errors="coerce")
    if dates.isna().any():
        raise ValueError("Steam source metadata contains invalid observation_date values")
    units: dict[str, str] = dict(metric_units or {})
    if "unit" in frame.columns:
        units.update({str(k): str(v) for k, v in frame[["metric_name", "unit"]].drop_duplicates().itertuples(index=False, name=None)})
    path = Path(source_path) if source_path is not None else None
    return SteamSourceMetadata(
        source_id=source_id or _single_value(frame, "source_id"),
        source_name=source_name or _single_value(frame, "source_name"),
        source_url=source_url or _single_value(frame, "source_url"),
        original_filename=path.name if path else _single_value(frame, "original_filename"),
        file_hash=file_hash or _single_value(frame, "file_hash"),
        load_timestamp=load_timestamp or _single_value(frame, "load_timestamp"),
        row_count=len(frame),
        min_date=dates.min().date() if not dates.empty else None,
        max_date=dates.max().date() if not dates.empty else None,
        schema_version=_single_value(frame, "schema_version") or SCHEMA_VERSION,
        status=status,
        source_format=path.suffix.removeprefix(".") if path else None,
        enabled=enabled,
        metric_units=units,
    )


class SteamHardwareSource:
    """Optional local source boundary; never scrapes Steam web pages."""

    def __init__(self, path: str | Path | None = None, *, enabled: bool = False,
                 source_id: str | None = DEFAULT_SOURCE_ID, source_name: str | None = None,
                 source_url: str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_STEAM_HARDWARE_DIR
        self.enabled = enabled
        self.source_id = source_id
        self.source_name = source_name
        self.source_url = source_url

    def read(self) -> pd.DataFrame:
        if not self.enabled:
            return _empty_steam_frame()
        return read_steam_hardware(self.path, source_id=self.source_id, source_name=self.source_name, source_url=self.source_url)


def read_steam_hardware(
    path: str | Path | None = None,
    *,
    source_id: str | None = DEFAULT_SOURCE_ID,
    source_name: str | None = "Steam Hardware and Software Survey extract",
    source_url: str | None = None,
    enabled: bool = True,
    mapping_path: str | Path | None = None,
) -> pd.DataFrame:
    """Read one CSV/Parquet extract and return canonical monthly long rows."""
    if not enabled:
        return _empty_steam_frame()
    source = _resolve_source_path(path)
    try:
        raw = pd.read_parquet(source) if source.suffix.lower() == ".parquet" else pd.read_csv(source)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Unable to read Steam dataset at {source}: {exc}") from exc
    if raw.empty:
        raise ValueError(f"Steam dataset is empty: {source}")
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    load_timestamp = datetime.now(timezone.utc).isoformat()
    result = _canonicalize(raw)
    result["source_id"] = _fill_provenance(result, "source_id", source_id)
    result["source_name"] = _fill_provenance(result, "source_name", source_name)
    result["source_url"] = _fill_provenance(result, "source_url", source_url)
    if result[["source_id", "source_url"]].isna().all(axis=1).any():
        raise ValueError("Steam dataset requires source_id or source_url provenance for every row")
    result["original_filename"] = source.name
    result["file_hash"] = file_hash
    result["load_timestamp"] = load_timestamp
    result["row_count"] = len(result)
    result["min_date"] = str(result["survey_month"].min().date())
    result["max_date"] = str(result["survey_month"].max().date())
    result["schema_version"] = SCHEMA_VERSION
    result["status"] = "PASS"
    result["generation"] = result.apply(
        lambda row: map_gpu_generation(row.get("metric_name"), row.get("category"), mapping_path=mapping_path), axis=1
    )
    return result[list(CANONICAL_COLUMNS) + ["observation_date", "metric_unit"]]

def map_gpu_generation(name: object, category: object = None, *, mapping_path: str | Path | None = None) -> str:
    """Map a GPU label using YAML rules; unknown labels remain ``UNKNOWN``."""
    text = f"{category or ''} {name or ''}".casefold()
    config_path = Path(mapping_path or Path("config") / "steam_hardware.yaml")
    config = _load_mapping_config(str(config_path)) if config_path.exists() else {}
    rules = config.get("gpu_generation_mapping", {}) if isinstance(config, Mapping) else {}
    if isinstance(rules, Mapping):
        for generation, patterns in rules.items():
            if any(re.search(str(pattern), text, flags=re.IGNORECASE) for pattern in (patterns if isinstance(patterns, list) else [patterns])):
                return str(generation)
    return "UNKNOWN"

@lru_cache(maxsize=16)
def _load_mapping_config(path: str) -> dict[str, Any]:
    payload = load_yaml(Path(path))
    return payload if isinstance(payload, dict) else {}


def _canonicalize(raw: pd.DataFrame) -> pd.DataFrame:
    normalized = {_slug(column): column for column in raw.columns}
    date_col = _first(normalized, "surveymonth", "month", "date", "observationdate", "surveydate", "period")
    if date_col is None:
        raise ValueError("Steam dataset requires a survey month/date column")
    category_col = _first(normalized, "category", "type", "section", "hardwarecategory")
    metric_col = _first(normalized, "metricname", "metric", "name", "item", "model", "gpu", "component")
    value_col = _first(normalized, "metricvalue", "value", "share", "percentage", "percent", "marketshare")
    dates = pd.to_datetime(raw[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    if dates.isna().any():
        raise ValueError(f"Steam {date_col} contains invalid values")
    if metric_col is not None and value_col is not None:
        result = pd.DataFrame({
            "survey_month": dates.dt.to_period("M").dt.to_timestamp(),
            "category": raw[category_col].astype("string").str.strip() if category_col else "UNKNOWN",
            "metric_name": raw[metric_col].astype("string").str.strip(),
            "metric_value": _numeric_percent(raw[value_col]),
            "unit": "percent",
            "vendor": raw[_first(normalized, "vendor", "manufacturer", "brand")].astype("string").str.strip() if _first(normalized, "vendor", "manufacturer", "brand") else pd.NA,
        })
    else:
        excluded = {date_col}
        if category_col:
            excluded.add(category_col)
        rows: list[dict[str, Any]] = []
        for column in raw.columns:
            if column in excluded:
                continue
            numeric = _numeric_percent(raw[column])
            if numeric.notna().sum() == 0:
                continue
            category = _infer_category(str(column))
            rows.extend({"survey_month": dates.iloc[i].to_period("M").to_timestamp(), "category": category,
                         "metric_name": str(column).strip(), "metric_value": numeric.iloc[i], "unit": "percent", "vendor": pd.NA}
                        for i in range(len(raw)) if pd.notna(numeric.iloc[i]))
        result = pd.DataFrame(rows)
        if result.empty:
            raise ValueError("Steam wide dataset contains no numeric metric columns")
    change_col = _first(normalized, "change", "delta")
    if change_col is not None and metric_col is not None and len(result) == len(raw):
        result["source_change"] = pd.to_numeric(raw[change_col], errors="coerce").to_numpy()
    else:
        result["source_change"] = pd.NA
    result["metric_name"] = result["metric_name"].astype("string").str.strip()
    # The community extract contains an unnamed aggregate row in
    # ``Video Card Description``. Preserve it explicitly instead of dropping
    # the raw observation or making the whole real extract unloadable.
    result["metric_name"] = result["metric_name"].fillna("UNKNOWN").replace("", "UNKNOWN")
    if result["metric_value"].isna().any() or not np.isfinite(result["metric_value"].to_numpy(float)).all():
        raise ValueError("Steam metric_value values must be finite numeric values")
    if ((result["metric_value"] < 0) | (result["metric_value"] > 100)).any():
        raise ValueError("Steam percentage values must be between 0 and 100")
    result["category"] = result["category"].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    result["vendor"] = result["vendor"].astype("string").str.strip()
    result["generation"] = "UNKNOWN"
    source_id_col = _first(normalized, "sourceid")
    source_url_col = _first(normalized, "sourceurl", "url")
    metric_unit_col = _first(normalized, "metricunit", "unit")
    result["source_id"] = raw[source_id_col].astype("string") if source_id_col else pd.NA
    result["source_name"] = pd.NA
    result["source_url"] = raw[source_url_col].astype("string") if source_url_col else pd.NA
    result["metric_unit"] = raw[metric_unit_col].astype("string") if metric_unit_col else result["unit"]
    duplicate = result.duplicated(["survey_month", "category", "metric_name"], keep=False)
    if duplicate.any():
        raise ValueError("Steam dataset contains duplicate keys for survey month/category/metric")
    result["observation_date"] = result["survey_month"].dt.date
    return result


def _numeric_percent(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(",", "", regex=False).str.strip()
    numeric = pd.to_numeric(text.str.replace("%", "", regex=False), errors="coerce")
    fraction = numeric.between(0, 1, inclusive="both")
    return numeric.where(~fraction, numeric * 100.0)


def _infer_category(name: str) -> str:
    lowered = name.casefold()
    for key, label in (("gpu", "GPU"), ("vram", "VRAM"), ("ram", "System RAM"), ("cpu", "CPU"), ("core", "CPU cores"), ("operating", "Operating System"), ("resolution", "Resolution")):
        if key in lowered:
            return label
    return "UNKNOWN"


def _fill_provenance(frame: pd.DataFrame, column: str, fallback: str | None) -> pd.Series:
    if column in frame.columns:
        values = frame[column].astype("string").str.strip().replace("", pd.NA)
        return values.fillna(fallback)
    return pd.Series(fallback, index=frame.index, dtype="string")


def _single_value(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype(str).str.strip().unique()
    if len(values) > 1:
        raise ValueError(f"Steam metadata has multiple {column} values")
    return values[0] if len(values) else None


def _slug(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _first(columns: Mapping[str, str], *names: str) -> str | None:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _resolve_source_path(path: str | Path | None) -> Path:
    source = Path(path) if path is not None else DEFAULT_STEAM_HARDWARE_DIR
    if not source.exists():
        raise FileNotFoundError(f"Steam dataset not found at {source}; disable source or add CSV/Parquet extract")
    if source.is_dir():
        candidates = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)
        if not candidates:
            raise FileNotFoundError(f"No CSV or Parquet Steam extract found under {source}")
        if len(candidates) > 1:
            raise ValueError(f"Multiple Steam extracts found under {source}; provide one explicit path")
        source = candidates[0]
    if source.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Steam dataset must be CSV or Parquet, not {source.suffix or 'unknown format'}")
    return source


def _empty_steam_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "observation_date": pd.Series(dtype="object"),
        "metric_name": pd.Series(dtype="string"),
        "metric_value": pd.Series(dtype="float64"),
        "source_id": pd.Series(dtype="string"),
        "source_url": pd.Series(dtype="string"),
    })
