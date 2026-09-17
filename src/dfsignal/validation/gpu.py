"""Validation diagnostics for the public GPU launch signal.

This module is deliberately independent of storage and orchestration.  Callers
pass raw or normalized DataFrames, receive dictionaries/DataFrames, and may
explicitly request persistence under an output directory.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from dfsignal.ingestion.gpu import derive_gpu_launch_events, derive_gpu_weekly_signals, filter_gpu_products, normalize_gpu_specs


PASS = "PASS"
REVIEW = "REVIEW"
FAIL = "FAIL"
_STATUS_RANK = {PASS: 0, REVIEW: 1, FAIL: 2}


def _status(*values: object) -> str:
    return max((str(value) for value in values), key=lambda value: _STATUS_RANK.get(value, 1), default=PASS)


def _jsonable(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    return value


def _as_products(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("GPU validation requires a pandas DataFrame")
    if {"product_name", "release_date"}.issubset(frame.columns):
        return frame.copy()
    return normalize_gpu_specs(frame)


def _iso_date(value: object) -> str | None:
    if value is None or value is pd.NA:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def calculate_gpu_coverage(
    products: pd.DataFrame,
    *,
    date_column: str = "release_date",
    expected_start: str | pd.Timestamp | None = None,
    expected_end: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Calculate coverage from parseable observed dates only.

    ``observed_start`` and ``observed_end`` are bounds of valid values in the
    supplied frame.  They are not claims that the upstream catalog is complete.
    Optional expected bounds are reported separately and never used to extend
    the observed range.  Year-only and month-only source dates are counted as
    valid parsed dates but their precision is retained in the result.
    """

    frame = _as_products(products)
    total_rows = int(len(frame))
    if date_column not in frame.columns:
        return {
            "status": FAIL,
            "reason": f"missing date column '{date_column}'",
            "total_rows": total_rows,
            "valid_date_rows": 0,
            "invalid_or_missing_date_rows": total_rows,
            "valid_date_ratio": 0.0 if total_rows else None,
            "observed_start": None,
            "observed_end": None,
            "observed_span_days": 0,
            "observed_span_weeks": 0.0,
            "coverage_basis": "observed valid dates only",
        }
    parsed = pd.to_datetime(frame[date_column], errors="coerce")
    valid = parsed.notna()
    valid_dates = parsed.loc[valid]
    valid_count = int(valid.sum())
    invalid_count = total_rows - valid_count
    if valid_count:
        observed_start = _iso_date(valid_dates.min())
        observed_end = _iso_date(valid_dates.max())
        span_days = int((valid_dates.max() - valid_dates.min()).days) + 1
        span_weeks = span_days / 7.0
        observed_years = sorted({int(value) for value in valid_dates.dt.year})
    else:
        observed_start = None
        observed_end = None
        span_days = 0
        span_weeks = 0.0
        observed_years = []
    precision_counts: dict[str, int] = {}
    if "release_date_precision" in frame.columns:
        precision = frame.loc[valid, "release_date_precision"].astype("string").fillna("unknown")
        precision_counts = {str(key): int(value) for key, value in precision.value_counts().sort_index().items()}
    expected: dict[str, Any] = {"start": _iso_date(expected_start), "end": _iso_date(expected_end)}
    expected_status = PASS
    expected_overlap_days: int | None = None
    if expected_start is not None or expected_end is not None:
        lower = pd.to_datetime(expected_start, errors="coerce") if expected_start is not None else valid_dates.min() if valid_count else pd.NaT
        upper = pd.to_datetime(expected_end, errors="coerce") if expected_end is not None else valid_dates.max() if valid_count else pd.NaT
        if pd.isna(lower) or pd.isna(upper) or lower > upper:
            expected_status = FAIL
            expected["reason"] = "expected bounds are not a valid ordered date range"
        elif valid_count:
            overlap_start = max(valid_dates.min(), lower)
            overlap_end = min(valid_dates.max(), upper)
            expected_overlap_days = max(0, int((overlap_end - overlap_start).days) + 1)
            if observed_start != _iso_date(lower) or observed_end != _iso_date(upper):
                expected_status = REVIEW
                expected["reason"] = "observed bounds do not equal the optional expected bounds"
    overall_status = FAIL if total_rows == 0 or valid_count == 0 else REVIEW if invalid_count or expected_status == REVIEW else PASS
    return {
        "status": overall_status,
        "total_rows": total_rows,
        "valid_date_rows": valid_count,
        "invalid_or_missing_date_rows": invalid_count,
        "valid_date_ratio": valid_count / total_rows if total_rows else None,
        "observed_start": observed_start,
        "observed_end": observed_end,
        "observed_span_days": span_days,
        "observed_span_weeks": span_weeks,
        "observed_years": observed_years,
        "observed_year_count": len(observed_years),
        "date_precision_counts": precision_counts,
        "expected_bounds": expected,
        "expected_overlap_days": expected_overlap_days,
        "coverage_basis": "observed valid dates only",
        "caveat": "A valid parsed date proves only that a row has a date; it does not prove catalog completeness or continuous product coverage.",
    }

def calculate_gpu_event_coverage(events: pd.DataFrame) -> dict[str, Any]:
    """Summarize generated event coverage without implying catalog completeness."""
    if events.empty:
        return {"status": REVIEW, "event_rows": 0, "observed_start": None, "observed_end": None, "vendors": [], "generations": [], "market_segments": []}
    dates = pd.to_datetime(events.get("event_date"), errors="coerce")
    valid = dates.notna()
    status = events.get("event_status", pd.Series("REVIEW", index=events.index)).astype("string").str.upper()
    return {
        "status": PASS if valid.all() and status.isin({"REVIEWED", "AUTO_DERIVED"}).all() else REVIEW,
        "event_rows": int(len(events)),
        "valid_event_dates": int(valid.sum()),
        "observed_start": _iso_date(dates[valid].min()) if valid.any() else None,
        "observed_end": _iso_date(dates[valid].max()) if valid.any() else None,
        "vendors": sorted(events.get("vendor", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()),
        "generations": sorted(events.get("generation", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()),
        "market_segments": sorted(events.get("market_segment", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()),
        "status_counts": {str(key): int(value) for key, value in status.value_counts().sort_index().items()},
        "coverage_basis": "observed generated event dates only",
    }


def _numeric_summary(values: pd.Series, *, metric: str, upper_bound: float) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    positive = numeric.loc[valid]
    impossible = positive.lt(0) | positive.gt(upper_bound)
    result: dict[str, Any] = {
        "metric": metric,
        "valid_count": int(valid.sum()),
        "missing_or_unparseable_count": int((~valid).sum()),
        "non_positive_count": int((positive <= 0).sum()),
        "impossible_count": int(impossible.sum()),
        "min": None,
        "p05": None,
        "median": None,
        "p95": None,
        "max": None,
        "upper_bound_for_sanity": upper_bound,
    }
    if not positive.empty:
        result.update({"min": float(positive.min()), "p05": float(positive.quantile(0.05)), "median": float(positive.median()), "p95": float(positive.quantile(0.95)), "max": float(positive.max())})
    return result


def _generation_conflicts(products: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    frame = products
    if "generation_conflict_flag" in frame.columns:
        flags = frame["generation_conflict_flag"].astype("boolean").fillna(False)
    else:
        raw_columns = [column for column in frame.columns if str(column).casefold() in {"graphics_generation_raw", "mobile_generation_raw", "integrated_generation_raw", "generation_raw"}]
        flags = pd.Series(False, index=frame.index, dtype="boolean")
        if raw_columns:
            def different(row: pd.Series) -> bool:
                values = [str(value).strip().casefold() for value in row.tolist() if value is not None and value is not pd.NA and not pd.isna(value) and str(value).strip() and str(value).strip().casefold() not in {"unknown", "nan"}]
                return len(set(values)) > 1
            flags = frame[raw_columns].apply(different, axis=1).astype("boolean")
    conflict_rows = frame.loc[flags].copy()
    preferred = [column for column in ["source_row_number", "gpu_id", "product_name", "generation_candidates", "generation_reason"] if column in conflict_rows.columns]
    raw_columns = [column for column in conflict_rows.columns if str(column).casefold() in {"graphics_generation_raw", "mobile_generation_raw", "integrated_generation_raw", "generation_raw"}]
    selected = list(dict.fromkeys(preferred + raw_columns))
    return flags, conflict_rows[selected].reset_index(drop=True) if selected else pd.DataFrame(index=range(len(conflict_rows)))


def diagnose_gpu_weekly_signals(products: pd.DataFrame, weekly_signals: pd.DataFrame | None = None) -> dict[str, Any]:
    """Run deterministic sanity checks over product and weekly launch signals.

    Spikes are flagged when a weekly unique-launch count exceeds
    ``max(3, Q3 + 1.5*IQR)``.  The threshold is a review aid, not a deletion rule.
    Duplicate rows are reported separately from unique launch counts, and any
    generation disagreement is surfaced as ``REVIEW``.
    """

    frame = _as_products(products)
    signal_frame = filter_gpu_products(frame)
    dates = pd.to_datetime(signal_frame.get("release_date", pd.Series(pd.NaT, index=signal_frame.index)), errors="coerce")
    names = signal_frame.get("product_name", pd.Series(pd.NA, index=signal_frame.index, dtype="string")).astype("string").str.strip()
    valid = dates.notna() & names.notna() & names.ne("")
    valid_frame = signal_frame.loc[valid].copy()
    valid_dates = dates.loc[valid]
    vendor = valid_frame.get("vendor", pd.Series("Unknown", index=valid_frame.index)).astype("string").fillna("Unknown")
    key = vendor.str.casefold() + "|" + names.loc[valid].str.casefold() + "|" + valid_dates.dt.strftime("%Y-%m-%d")
    duplicate_groups = key.value_counts()
    duplicate_group_count = int((duplicate_groups > 1).sum())
    duplicate_rows = int(duplicate_groups[duplicate_groups > 1].sum() - duplicate_group_count)
    exact_duplicate_rows = int(frame.duplicated(keep=False).sum())
    unique_launch_count = int(key.nunique())
    vendor_counts = {str(key_): int(value) for key_, value in vendor.value_counts().sort_index().items()}
    unknown_vendor_count = int(vendor.str.casefold().eq("unknown").sum())

    tdp_series = valid_frame["tdp_w"] if "tdp_w" in valid_frame.columns else pd.Series(dtype="float64")
    vram_column = "memory_size_mb" if "memory_size_mb" in valid_frame.columns else "memory_size"
    vram_series = valid_frame[vram_column] if vram_column in valid_frame.columns else pd.Series(dtype="float64")
    tdp_summary = _numeric_summary(tdp_series, metric="tdp_w", upper_bound=1000.0)
    vram_summary = _numeric_summary(vram_series, metric="memory_size_mb", upper_bound=1024 * 1024.0)
    generation_flags, generation_conflict_rows = _generation_conflicts(frame)
    generation_conflict_count = int(generation_flags.sum())

    weekly = derive_gpu_weekly_signals(frame) if weekly_signals is None else weekly_signals.copy()
    weekly_diagnostics = pd.DataFrame()
    spike_threshold: float | None = None
    spike_count = 0
    missing_week_count = 0
    weekly_count_sum = 0
    weekly_status = PASS
    if weekly.empty:
        weekly_status = FAIL if unique_launch_count else REVIEW
    elif "week" not in weekly.columns or "gpu_launch_count" not in weekly.columns:
        weekly_status = FAIL
    else:
        weekly["week"] = pd.to_datetime(weekly["week"], errors="coerce")
        weekly["gpu_launch_count"] = pd.to_numeric(weekly["gpu_launch_count"], errors="coerce")
        counts = weekly["gpu_launch_count"].dropna()
        weekly_count_sum = int(counts.sum())
        if not counts.empty:
            q1, q3 = float(counts.quantile(0.25)), float(counts.quantile(0.75))
            iqr = q3 - q1
            spike_threshold = max(3.0, q3 + 1.5 * iqr)
            spike_flags = weekly["gpu_launch_count"] > spike_threshold
            spike_count = int(spike_flags.sum())
            weekly_diagnostics = weekly[["week", "gpu_launch_count"]].copy()
            weekly_diagnostics["spike_threshold"] = spike_threshold
            weekly_diagnostics["spike_flag"] = spike_flags.astype("boolean")
            weekly_diagnostics["diagnostic_status"] = spike_flags.map(lambda value: REVIEW if value else PASS).astype("string")
            valid_weeks = weekly["week"].dropna()
            if not valid_weeks.empty:
                expected_weeks = pd.date_range(valid_weeks.min(), valid_weeks.max(), freq="7D")
                missing_week_count = int(len(expected_weeks.difference(valid_weeks)))
        if weekly_count_sum != unique_launch_count:
            weekly_status = FAIL
        elif spike_count or missing_week_count:
            weekly_status = REVIEW

    classification_counts: dict[str, int] = {}
    if "classification_status" in frame.columns:
        classification_counts = {str(key_): int(value) for key_, value in frame["classification_status"].fillna(REVIEW).value_counts().sort_index().items()}
    review_columns = {
        "vendor": int(frame.get("vendor_classification_status", pd.Series(REVIEW, index=frame.index)).astype("string").eq(REVIEW).sum()),
        "generation": int(frame.get("generation_status", pd.Series(REVIEW, index=frame.index)).astype("string").eq(REVIEW).sum()),
        "market_segment": int(frame.get("market_segment_status", pd.Series(REVIEW, index=frame.index)).astype("string").eq(REVIEW).sum()),
        "performance_tier": int(frame.get("performance_tier_status", pd.Series(REVIEW, index=frame.index)).astype("string").eq(REVIEW).sum()),
    }
    checks = [
        {"check": "non_empty_product_frame", "status": PASS if len(frame) else FAIL, "observed": len(frame), "rule": "at least one product row"},
        {"check": "valid_release_dates", "status": FAIL if not valid.any() else REVIEW if not valid.all() else PASS, "observed": int(valid.sum()), "rule": "weekly signals use valid parsed dates only"},
        {"check": "weekly_launch_count_reconciles", "status": PASS if weekly_count_sum == unique_launch_count and unique_launch_count else FAIL if unique_launch_count else REVIEW, "observed": {"weekly_sum": weekly_count_sum, "unique_products": unique_launch_count}, "rule": "weekly unique launch sum equals unique product/date keys"},
        {"check": "duplicate_product_keys", "status": REVIEW if duplicate_rows else PASS, "observed": {"duplicate_groups": duplicate_group_count, "duplicate_extra_rows": duplicate_rows, "exact_duplicate_rows": exact_duplicate_rows}, "rule": "duplicates are retained and require review"},
        {"check": "weekly_launch_spikes", "status": REVIEW if spike_count else PASS, "observed": {"spike_count": spike_count, "threshold": spike_threshold}, "rule": "count > max(3, Q3 + 1.5*IQR)"},
        {"check": "generation_conflicts", "status": REVIEW if generation_conflict_count else PASS, "observed": generation_conflict_count, "rule": "two or more distinct source generation labels on a row"},
        {"check": "tdp_range", "status": FAIL if tdp_summary["impossible_count"] else REVIEW if tdp_summary["missing_or_unparseable_count"] else PASS, "observed": tdp_summary, "rule": "0 <= TDP <= 1000 W sanity envelope"},
        {"check": "vram_range", "status": FAIL if vram_summary["impossible_count"] else REVIEW if vram_summary["missing_or_unparseable_count"] else PASS, "observed": vram_summary, "rule": "0 <= VRAM <= 1 TiB sanity envelope; canonical unit is MB"},
        {"check": "vendor_classification", "status": REVIEW if unknown_vendor_count or review_columns["vendor"] else PASS, "observed": {"unknown_vendor_rows": unknown_vendor_count, "review_rows": review_columns["vendor"]}, "rule": "unmapped brands remain Unknown/REVIEW"},
    ]
    summary_status = _status(*(check["status"] for check in checks))
    summary = {
        "status": summary_status,
        "product_rows": int(len(frame)),
        "valid_dated_rows": int(valid.sum()),
        "unique_launch_count": unique_launch_count,
        "weekly_rows": int(len(weekly)),
        "weekly_count_sum": weekly_count_sum,
        "vendor_counts": vendor_counts,
        "classification_status_counts": classification_counts,
        "classification_review_counts": review_columns,
        "tdp": tdp_summary,
        "vram": vram_summary,
        "duplicate_groups": duplicate_group_count,
        "duplicate_extra_rows": duplicate_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "generation_conflict_count": generation_conflict_count,
        "spike_count": spike_count,
        "spike_threshold": spike_threshold,
        "missing_week_count": missing_week_count,
        "status_rule": "FAIL means a structural/reconciliation check failed; REVIEW means a human must assess uncertainty or unusual but parseable data.",
    }
    return {"status": summary_status, "summary": summary, "checks": pd.DataFrame(checks), "weekly_diagnostics": weekly_diagnostics, "generation_conflict_rows": generation_conflict_rows}


def build_gpu_validation_sample(products: pd.DataFrame, sample_size: int = 24) -> pd.DataFrame:
    """Build a deterministic 20–30 row review sample across available strata."""

    frame = _as_products(products).copy()
    if frame.empty:
        return pd.DataFrame(columns=["sample_rank", "sample_stratum"])
    try:
        requested = int(sample_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("sample_size must be an integer") from exc
    target = min(len(frame), max(20, min(30, requested)))
    dates = pd.to_datetime(frame.get("release_date", pd.Series(pd.NaT, index=frame.index)), errors="coerce")
    frame["__sample_vendor"] = frame.get("vendor", pd.Series("Unknown", index=frame.index)).astype("string").fillna("Unknown").replace("", "Unknown")
    frame["__sample_year"] = dates.dt.year.astype("Int64").astype("string").fillna("UNKNOWN_YEAR")
    frame["__sample_segment"] = frame.get("market_segment", pd.Series("Unknown", index=frame.index)).astype("string").fillna("Unknown").replace("", "Unknown")
    frame["__sample_stratum"] = frame["__sample_vendor"] + "|" + frame["__sample_year"] + "|" + frame["__sample_segment"]
    stable_columns = ["__sample_vendor", "__sample_year", "__sample_segment", "product_name", "release_date", "source_row_number"]
    for column in stable_columns:
        if column not in frame.columns:
            frame[column] = ""
    frame["__stable_key"] = frame[stable_columns].astype("string").fillna("").agg("|".join, axis=1)
    ordered = frame.sort_values(["__stable_key"], kind="mergesort")
    remaining = list(ordered.index)
    selected: list[Any] = []
    seen_vendor: set[str] = set()
    seen_year: set[str] = set()
    seen_segment: set[str] = set()
    while remaining and len(selected) < target:
        best_position = 0
        best_score = -1
        best_key = ""
        for position, row_index in enumerate(remaining):
            row = frame.loc[row_index]
            dimensions = (str(row["__sample_vendor"]), str(row["__sample_year"]), str(row["__sample_segment"]))
            score = int(dimensions[0] not in seen_vendor) + int(dimensions[1] not in seen_year) + int(dimensions[2] not in seen_segment)
            if score > best_score or (score == best_score and str(row["__stable_key"]) < best_key):
                best_position, best_score, best_key = position, score, str(row["__stable_key"])
        row_index = remaining.pop(best_position)
        row = frame.loc[row_index]
        selected.append(row_index)
        seen_vendor.add(str(row["__sample_vendor"]))
        seen_year.add(str(row["__sample_year"]))
        seen_segment.add(str(row["__sample_segment"]))
    result = frame.loc[selected].copy()
    result.insert(0, "sample_rank", range(1, len(result) + 1))
    result.insert(1, "sample_stratum", result["__sample_stratum"])
    drop_columns = ["__sample_vendor", "__sample_year", "__sample_segment", "__sample_stratum", "__stable_key"]
    result = result.drop(columns=[column for column in drop_columns if column in result.columns])
    preferred = ["sample_rank", "sample_stratum", "source_row_number", "gpu_id", "vendor_raw", "vendor", "product_name", "release_date_raw", "release_date", "release_date_precision", "market_segment_raw", "market_segment", "market_segment_candidates", "market_segment_status", "generation_raw", "generation", "generation_source", "generation_candidates", "generation_status", "generation_conflict_flag", "performance_tier", "performance_tier_status", "performance_tier_reason", "launch_importance", "launch_importance_status", "high_power_flag", "high_power_status", "high_vram_flag", "high_vram_status", "duplicate_product_flag", "classification_status", "classification_reasons"]
    columns = [column for column in preferred if column in result.columns] + [column for column in result.columns if column not in preferred]
    return result[columns].reset_index(drop=True)


def _report_json_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("diagnostics", {})
    return {
        "status": report.get("status"),
        "event_coverage": report.get("event_coverage", {}),
        "coverage": report.get("coverage", {}),
        "diagnostics": {key: value for key, value in diagnostics.items() if not isinstance(value, pd.DataFrame)},
        "checks": report.get("checks", pd.DataFrame()).to_dict("records") if isinstance(report.get("checks"), pd.DataFrame) else report.get("checks", []),
        "sample_row_count": int(len(report.get("sample", pd.DataFrame()))) if isinstance(report.get("sample"), pd.DataFrame) else 0,
    }


def _parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Serialize mixed diagnostic cells while preserving scalar-only columns."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype != "object":
            continue
        if not result[column].map(lambda value: isinstance(value, (Mapping, list, tuple))).any():
            continue
        result[column] = result[column].map(
            lambda value: (
                json.dumps(_jsonable(value), sort_keys=True)
                if isinstance(value, (Mapping, list, tuple))
                else None
                if value is None or value is pd.NA
                else str(value)
            )
        )
    return result

def persist_gpu_validation_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Persist report JSON and review tables under an explicit output directory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(_report_json_payload(report))
    report_path = destination / "gpu_validation_report.json"
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifacts: dict[str, str] = {"report_json": str(report_path)}
    for key, filename in (("weekly_diagnostics", "gpu_weekly_diagnostics.parquet"), ("sample", "gpu_validation_sample.parquet"), ("checks", "gpu_validation_checks.parquet"), ("generation_conflict_rows", "gpu_generation_conflicts.parquet")):
        value = report.get(key)
        if isinstance(value, pd.DataFrame):
            path = destination / filename
            _parquet_safe_frame(value).to_parquet(path, index=False)
            artifacts[key] = str(path)
            metadata = {"row_count": int(len(value)), "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return artifacts


def generate_gpu_validation_report(
    products: pd.DataFrame | None = None,
    *,
    raw: pd.DataFrame | None = None,
    weekly_signals: pd.DataFrame | None = None,
    sample_size: int = 24,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generate a reviewable GPU validation report, optionally persisting it."""

    if products is None:
        if raw is None:
            raise ValueError("provide normalized products or raw GPU data")
        products = normalize_gpu_specs(raw)
    frame = _as_products(products)
    coverage = calculate_gpu_coverage(frame)
    events = derive_gpu_launch_events(frame)
    event_coverage = calculate_gpu_event_coverage(events)
    diagnostics = diagnose_gpu_weekly_signals(frame, weekly_signals)
    sample = build_gpu_validation_sample(frame, sample_size)
    sample_status = FAIL if sample.empty else REVIEW if "classification_status" in sample.columns and (sample["classification_status"].fillna(REVIEW) != PASS).any() else PASS
    report_status = _status(coverage["status"], diagnostics["status"], sample_status, event_coverage["status"])
    checks = diagnostics["checks"].copy()
    checks = pd.concat([checks, pd.DataFrame([{"check": "stratified_review_sample", "status": sample_status, "observed": int(len(sample)), "rule": "20–30 deterministic rows when at least 20 products are available"}, {"check": "gpu_event_coverage", "status": event_coverage["status"], "observed": event_coverage["event_rows"], "rule": "generated events use configured scope and observed dates only"}])], ignore_index=True)
    report: dict[str, Any] = {"status": report_status, "summary": {"status": report_status, "product_rows": int(len(frame)), "event_rows": int(len(events)), "sample_rows": int(len(sample)), "coverage_status": coverage["status"], "event_coverage_status": event_coverage["status"], "diagnostics_status": diagnostics["status"], "sample_status": sample_status}, "coverage": coverage, "event_coverage": event_coverage, "diagnostics": diagnostics["summary"], "checks": checks, "weekly_diagnostics": diagnostics["weekly_diagnostics"], "generation_conflict_rows": diagnostics["generation_conflict_rows"], "sample": sample}
    if output_dir is not None:
        report["artifacts"] = persist_gpu_validation_report(report, output_dir)
    return report


# A concise alias for callers that prefer a validation verb.
validate_gpu_signals = generate_gpu_validation_report
