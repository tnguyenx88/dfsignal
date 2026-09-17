"""Leakage-safe Steam Hardware Survey normalization and adoption signals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..ingestion.events import usable_launch_events
from ..ingestion.steam import SCHEMA_VERSION, map_gpu_generation
from .weekly import to_week_saturday


def normalize_steam_hardware(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical or legacy Steam rows to monthly long form."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Steam normalization requires a DataFrame")
    if frame.empty:
        return pd.DataFrame(columns=["survey_month", "category", "metric_name", "metric_value", "unit", "vendor", "generation", "source_id"])
    result = frame.copy()
    if "survey_month" not in result.columns:
        if "observation_date" not in result.columns:
            raise ValueError("Steam data requires survey_month or observation_date")
        result["survey_month"] = result["observation_date"]
    for required in ("category", "metric_name", "metric_value"):
        if required not in result.columns:
            raise ValueError(f"Steam data missing required column: {required}")
    result["survey_month"] = pd.to_datetime(result["survey_month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    if result["survey_month"].isna().any():
        raise ValueError("Steam survey_month contains invalid dates")
    result["category"] = result["category"].astype("string").fillna("UNKNOWN").str.strip().replace("", "UNKNOWN")
    result["metric_name"] = result["metric_name"].astype("string").str.strip()
    result["metric_value"] = pd.to_numeric(result["metric_value"], errors="coerce")
    if result["metric_name"].isna().any() or result["metric_value"].isna().any():
        raise ValueError("Steam metric names and values must be non-null")
    if ((result["metric_value"] < 0) | (result["metric_value"] > 100)).any():
        raise ValueError("Steam percentage values must be between 0 and 100")
    for column, default in (("unit", "percent"), ("vendor", pd.NA), ("generation", "UNKNOWN"), ("source_id", "STEAM_HARDWARE")):
        if column not in result.columns:
            result[column] = default
    result["generation"] = result["generation"].astype("string").fillna("UNKNOWN")
    result["schema_version"] = result.get("schema_version", SCHEMA_VERSION)
    return result.sort_values(["survey_month", "category", "metric_name"]).reset_index(drop=True)

def apply_steam_quality_flags(
    frame: pd.DataFrame,
    *,
    anomaly_path: str | Path | None = None,
) -> pd.DataFrame:
    """Apply configuration-driven quality flags without changing source values."""
    normalized = normalize_steam_hardware(frame)
    if normalized.empty:
        return normalized
    result = normalized.copy()
    result["quality_flag"] = "NORMAL"
    result["quality_reason"] = pd.Series("", index=result.index, dtype="string")
    config = load_yaml(anomaly_path or Path("config") / "steam_anomalies.yaml")
    rules = config.get("rules", []) if isinstance(config, Mapping) else []
    for rule in rules if isinstance(rules, list) else []:
        if not isinstance(rule, Mapping):
            continue
        start = pd.to_datetime(rule.get("start_month"), errors="coerce")
        end = pd.to_datetime(rule.get("end_month"), errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        categories = rule.get("category", "*")
        categories = categories if isinstance(categories, list) else [categories]
        category_values = {str(value).strip().casefold() for value in categories}
        category_mask = (
            pd.Series(True, index=result.index)
            if category_values & {"*", "all"}
            else result["category"].astype(str).str.casefold().isin(category_values)
        )
        mask = result["survey_month"].between(start.to_period("M").to_timestamp(), end.to_period("M").to_timestamp()) & category_mask
        if not mask.any():
            continue
        flag = str(rule.get("quality_flag", "KNOWN_ANOMALY")).upper()
        reason = str(rule.get("reason", "configured Steam source quality review"))
        previous = result.loc[mask, "quality_reason"].astype(str)
        result.loc[mask, "quality_flag"] = flag
        result.loc[mask, "quality_reason"] = previous.where(previous.eq(""), previous + "; ") + reason
    return result

def monthly_to_weekly(frame: pd.DataFrame, *, publication_lag_months: int = 0) -> pd.DataFrame:
    """Carry monthly values to Saturdays after a conservative publication lag."""
    if int(publication_lag_months) < 0:
        raise ValueError("publication_lag_months must be non-negative")
    monthly = normalize_steam_hardware(frame)
    if monthly.empty:
        return pd.DataFrame(columns=["week", "category", "metric_name", "metric_value", "unit", "vendor", "generation", "source_id"])
    rows: list[pd.DataFrame] = []
    key = ["category", "metric_name"]
    for _, group in monthly.groupby(key, dropna=False, sort=False):
        group = group.sort_values("survey_month").drop_duplicates(["survey_month"], keep="last").copy()
        if publication_lag_months:
            group["effective_month"] = group["survey_month"] + pd.DateOffset(months=int(publication_lag_months))
        else:
            group["effective_month"] = group["survey_month"]
        group["effective_week"] = pd.to_datetime(to_week_saturday(group["effective_month"]))
        weeks = pd.date_range(group["effective_week"].min(), group["effective_week"].max(), freq="W-SAT")
        source = group.set_index("effective_week").reindex(weeks, method="ffill").reset_index(names="week")
        out = pd.DataFrame({"week": weeks})
        for column in group.columns:
            if column not in {"survey_month", "effective_month", "effective_week"}:
                out[column] = source[column].to_numpy()
        rows.append(out)
    return pd.concat(rows, ignore_index=True).sort_values(["week", "category", "metric_name"]).reset_index(drop=True)


def derive_steam_signals(
    weekly: pd.DataFrame,
    *,
    mapping_path: str | Path | None = None,
    publication_lag_months: int = 0,
) -> pd.DataFrame:
    """Derive only adoption signals supported by observed Steam metrics."""
    if weekly.empty:
        return pd.DataFrame(columns=["week", "source_id"])
    frame = (
        monthly_to_weekly(weekly, publication_lag_months=publication_lag_months)
        if "survey_month" in weekly.columns
        else weekly.copy()
    )
    frame["week"] = pd.to_datetime(frame["week"], errors="raise")
    frame["metric_name"] = frame["metric_name"].astype(str)
    frame["category"] = frame["category"].astype(str)
    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["generation"] = frame.apply(
        lambda r: r.get("generation", "UNKNOWN")
        if str(r.get("generation", "UNKNOWN")).upper() not in {"", "NAN", "NONE", "UNKNOWN"}
        else map_gpu_generation(r["metric_name"], r["category"], mapping_path=mapping_path),
        axis=1,
    )
    configured_order = load_yaml(mapping_path or Path("config") / "steam_hardware.yaml").get(
        "gpu_generation_order", []
    )
    generation_order = (
        [str(value) for value in configured_order]
        if isinstance(configured_order, list) and configured_order
        else ["RTX_50", "RTX_40", "RTX_30", "RX_9000", "RX_7000"]
    )
    out_rows: list[dict[str, Any]] = []
    for week, group in frame.groupby("week", sort=True):
        row: dict[str, Any] = {
            "week": week,
            "source_id": str(group["source_id"].dropna().iloc[0])
            if "source_id" in group and group["source_id"].notna().any()
            else "STEAM_HARDWARE",
        }
        text = (group["category"] + " " + group["metric_name"]).str.casefold()
        gpu_mask = text.str.contains(r"gpu|graphics|video\s+card\s+description", regex=True)
        gpu = group.loc[gpu_mask]
        for label, pattern in (("nvidia_share", "nvidia"), ("amd_share", "amd|ati")):
            selected = gpu.loc[text.loc[gpu.index].str.contains(pattern, regex=True)]
            if not selected.empty:
                row[label] = float(selected["metric_value"].sum())
        generation_values: dict[str, float] = {}
        for generation in generation_order:
            selected = gpu[gpu["generation"].astype(str).eq(generation)]
            if not selected.empty:
                generation_values[generation] = float(selected["metric_value"].sum())
                row[generation.lower().replace("_", "") + "_share"] = generation_values[generation]
        if generation_values:
            ordered = [value for value in generation_order if value in generation_values]
            row["latest_generation_share"] = generation_values[ordered[0]]
            if len(ordered) > 1:
                row["previous_generation_share"] = generation_values[ordered[1]]
                row["latest_two_generation_share"] = float(sum(generation_values[g] for g in ordered[:2]))
            for vendor, prefix in (("nvidia", ("RTX", "GTX")), ("amd", ("RX",))):
                vendor_generations = [g for g in ordered if g.upper().startswith(prefix)]
                if vendor_generations:
                    row[f"latest_{vendor}_generation_share"] = generation_values[vendor_generations[0]]
                    if len(vendor_generations) > 1:
                        row[f"previous_{vendor}_generation_share"] = generation_values[vendor_generations[1]]
        ram = group.loc[text.str.contains("ram|memory", regex=True)]
        for label, pattern in (("ram_16gb_share", r"16\s*gb"), ("ram_32gb_plus_share", r"(?:32|64|128)\s*gb|32gb\+")):
            selected = ram.loc[text.loc[ram.index].str.contains(pattern, regex=True)]
            if not selected.empty:
                row[label] = float(selected["metric_value"].sum())
        vram = group.loc[text.str.contains("vram|video memory|gpu memory", regex=True)]
        for label, pattern in (("vram_8gb_plus_share", r"(?:8|12|16|24|32)\s*gb"), ("vram_12gb_plus_share", r"(?:12|16|24|32)\s*gb")):
            selected = vram.loc[text.loc[vram.index].str.contains(pattern, regex=True)]
            if not selected.empty:
                row[label] = float(selected["metric_value"].sum())
        out_rows.append(row)
    result = pd.DataFrame(out_rows).sort_values("week").reset_index(drop=True)
    return add_steam_momentum(result)
def validate_steam_source(
    frame: pd.DataFrame,
    *,
    anomaly_path: str | Path | None = None,
    mapping_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create PASS/REVIEW/FAIL validation output for an optional source."""
    if frame.empty:
        checks = [{"check": "source_available", "status": "REVIEW", "detail": "optional Steam source is empty"}]
        return {"status": "REVIEW", "checks": pd.DataFrame(checks), "summary": {"rows": 0}}
    raw = frame.copy()
    values = pd.to_numeric(raw.get("metric_value"), errors="coerce")
    invalid_percentages = int(values.isna().sum() + ((values < 0) | (values > 100)).fillna(False).sum())
    duplicate_months = int(raw.duplicated(["survey_month", "category", "metric_name"]).sum())
    normalized = apply_steam_quality_flags(raw, anomaly_path=anomaly_path)
    normalized["generation"] = normalized.apply(
        lambda row: map_gpu_generation(row["metric_name"], row["category"], mapping_path=mapping_path),
        axis=1,
    )
    dates = pd.DatetimeIndex(normalized["survey_month"].dropna().unique()).sort_values()
    expected = pd.date_range(dates.min(), dates.max(), freq="MS") if len(dates) else pd.DatetimeIndex([])
    missing_months = expected.difference(dates)
    gpu_text = normalized["category"].astype(str) + " " + normalized["metric_name"].astype(str)
    gpu_rows = normalized.loc[gpu_text.str.contains(r"gpu|graphics|video\s+card", case=False, regex=True)]
    unknown_gpu_mappings = int(gpu_rows["generation"].astype(str).str.upper().eq("UNKNOWN").sum())
    unknown_categories = int(normalized["category"].astype(str).str.upper().eq("UNKNOWN").sum())
    jump = normalized.sort_values("survey_month").groupby(["category", "metric_name"])["metric_value"].diff().abs()
    large_jumps = int((jump > 20).sum())
    known_anomaly_rows = normalized[normalized["quality_flag"].astype(str).str.upper() != "NORMAL"]
    checks: list[dict[str, Any]] = [
        {"check": "schema", "status": "PASS", "detail": str(list(normalized.columns))},
        {"check": "date_coverage", "status": "PASS", "detail": f"{dates.min().date()} to {dates.max().date()}"},
        {"check": "duplicate_metrics", "status": "FAIL" if duplicate_months else "PASS", "detail": duplicate_months},
        {"check": "invalid_percentages", "status": "FAIL" if invalid_percentages else "PASS", "detail": invalid_percentages},
        {"check": "missing_months", "status": "REVIEW" if len(missing_months) else "PASS", "detail": ",".join(str(value.date()) for value in missing_months)},
        {"check": "unknown_categories", "status": "REVIEW" if unknown_categories else "PASS", "detail": unknown_categories},
        {"check": "unknown_gpu_mappings", "status": "REVIEW" if unknown_gpu_mappings else "PASS", "detail": unknown_gpu_mappings},
        {"check": "known_anomalies", "status": "REVIEW" if not known_anomaly_rows.empty else "PASS", "detail": int(len(known_anomaly_rows))},
        {"check": "large_monthly_jumps", "status": "REVIEW" if large_jumps else "PASS", "detail": large_jumps},
    ]
    statuses = [str(item["status"]) for item in checks]
    overall = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "PASS"
    summary = {
        "rows": len(normalized),
        "min_date": str(normalized["survey_month"].min().date()),
        "max_date": str(normalized["survey_month"].max().date()),
        "months_represented": len(dates),
        "missing_months": [str(value.date()) for value in missing_months],
        "categories": sorted(normalized["category"].astype(str).unique().tolist()),
        "known_anomaly_periods": sorted(normalized.loc[normalized["quality_flag"] != "NORMAL", "survey_month"].dt.strftime("%Y-%m").unique().tolist()),
        "unknown_gpu_mappings": unknown_gpu_mappings,
        "invalid_percentages": invalid_percentages,
    }
    return {"status": overall, "checks": pd.DataFrame(checks), "summary": summary, "normalized": normalized}
def add_steam_momentum(signals: pd.DataFrame) -> pd.DataFrame:
    """Add historical-only changes using prior observed weekly values."""
    result = signals.sort_values("week").copy()
    for source, target in (("latest_generation_share", "latest_generation_share_change_1m"), ("latest_generation_share", "latest_generation_share_change_3m"), ("ram_32gb_plus_share", "ram_32gb_share_change"), ("vram_8gb_plus_share", "high_vram_share_change")):
        if source not in result.columns:
            continue
        periods = 4 if "1m" in target or target in {"ram_32gb_share_change", "high_vram_share_change"} else 13
        result[target] = result[source] - result[source].shift(periods)
    if "latest_generation_share_change_1m" in result:
        result["gpu_adoption_momentum"] = result["latest_generation_share_change_1m"]
    return result


def join_launch_adoption(steam_signals: pd.DataFrame, launch_events: pd.DataFrame) -> pd.DataFrame:
    """Attach descriptive weeks-since-launch and adoption interaction fields."""
    launch_events = usable_launch_events(launch_events)
    if steam_signals.empty or launch_events.empty:
        return steam_signals.copy()
    result = steam_signals.copy().sort_values("week")
    launches = pd.to_datetime(launch_events["event_date"], errors="coerce")
    gpu_launches = launches[
        launch_events["event_type"].astype(str).str.contains("GPU", case=False, na=False)
    ].dropna()
    if gpu_launches.empty:
        return result
    latest_launch = gpu_launches.max()
    result["weeks_since_major_gpu_launch"] = ((result["week"] - latest_launch).dt.days // 7).clip(lower=0)
    if "latest_generation_share" in result:
        prior = result.loc[result["week"] < latest_launch, "latest_generation_share"].dropna()
        baseline = float(prior.iloc[-1]) if not prior.empty else float(result["latest_generation_share"].dropna().iloc[0])
        result["adoption_since_launch"] = result["latest_generation_share"] - baseline
        result["adoption_velocity_after_launch"] = result["latest_generation_share"].diff().where(result["week"] >= latest_launch)
        result["major_launch_x_adoption"] = result["latest_generation_share"] * (result["week"] >= latest_launch).astype(int)
    return result


def diagnose_steam_launch_adoption(steam_signals: pd.DataFrame, launch_events: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive launch/adoption milestones; not a causal analysis."""
    launch_events = usable_launch_events(launch_events if launch_events is not None else pd.DataFrame())
    if steam_signals.empty or launch_events.empty or "latest_generation_share" not in steam_signals:
        return pd.DataFrame(columns=["launch_date", "weeks_to_1pct", "weeks_to_5pct", "peak_velocity_week"])
    frame = steam_signals.sort_values("week")
    rows = []
    gpu_events = launch_events.loc[launch_events["event_type"].astype(str).str.contains("GPU", case=False, na=False)]
    for launch in pd.to_datetime(gpu_events["event_date"], errors="coerce").dropna():
        after = frame[frame["week"] >= launch]
        base = float(frame.loc[frame["week"] < launch, "latest_generation_share"].iloc[-1]) if (frame["week"] < launch).any() else 0.0
        adoption = after["latest_generation_share"] - base
        rows.append({"launch_date": launch.date(), "weeks_to_1pct": next((int((w - launch).days // 7) for w, v in zip(after["week"], adoption) if v >= 1), None), "weeks_to_5pct": next((int((w - launch).days // 7) for w, v in zip(after["week"], adoption) if v >= 5), None), "peak_velocity_week": after.loc[after["latest_generation_share"].diff().idxmax(), "week"] if len(after) > 1 else None})
    return pd.DataFrame(rows)




# Readable aliases for orchestration callers.
normalize_steam = normalize_steam_hardware
build_steam_weekly_signals = derive_steam_signals
add_adoption_momentum = add_steam_momentum
