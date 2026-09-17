"""Evaluate observed Steam Hardware Survey signals against a fixed baseline.

This module uses a locally supplied, provenance-preserving Steam extract. It does
not scrape Steam and it does not manufacture future Steam values. Forecast inputs
remain governed by the configured observed-only policy.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRegressor

from ..config import load_yaml
from ..ingestion.events import read_launch_events, usable_launch_events
from ..ingestion.steam import read_steam_hardware
from ..transformation.steam import (
    apply_steam_quality_flags,
    derive_steam_signals,
    diagnose_steam_launch_adoption,
    join_launch_adoption,
    validate_steam_source,
)


STEAM_SIGNAL_COLUMNS = (
    "nvidia_share",
    "amd_share",
    "latest_nvidia_generation_share",
    "previous_nvidia_generation_share",
    "latest_amd_generation_share",
    "previous_amd_generation_share",
    "ram_16gb_share",
    "ram_32gb_plus_share",
    "vram_8gb_plus_share",
    "vram_12gb_plus_share",
    "latest_generation_share",
    "previous_generation_share",
    "latest_two_generation_share",
    "latest_generation_share_change_1m",
    "latest_generation_share_change_3m",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _latest_gpu_mapping_review(frame: pd.DataFrame) -> pd.DataFrame:
    category_text = frame["category"].astype("string").str.casefold()
    name_text = frame["metric_name"].astype("string").str.casefold()
    gpu_mask = category_text.str.contains("gpu|graphics|video card", regex=True, na=False) | name_text.str.contains(
        "gpu|graphics|video card", regex=True, na=False
    )
    gpu = frame[gpu_mask].copy()
    if gpu.empty:
        return pd.DataFrame(
            columns=["metric_name", "observations", "first_month", "last_month", "latest_observed_share", "mean_observed_share"]
        )
    gpu["metric_value"] = pd.to_numeric(gpu["metric_value"], errors="coerce")
    latest_month = gpu["survey_month"].max()
    latest = gpu[gpu["survey_month"].eq(latest_month)].set_index("metric_name")["metric_value"]
    unknown = gpu[gpu["generation"].astype("string").str.upper().eq("UNKNOWN")]
    review = (
        unknown.groupby("metric_name", dropna=False)
        .agg(
            observations=("metric_value", "size"),
            first_month=("survey_month", "min"),
            last_month=("survey_month", "max"),
            mean_observed_share=("metric_value", "mean"),
        )
        .reset_index()
    )
    review["latest_observed_share"] = review["metric_name"].map(latest)
    return review.sort_values(["latest_observed_share", "mean_observed_share"], ascending=False, na_position="last")

def _launch_diagnostics(steam: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    events = usable_launch_events(events)
    if steam.empty or events.empty:
        return pd.DataFrame(
            columns=[
                "event_date", "event_type", "product_name", "generation",
                "matched_observations", "weeks_to_1pct", "weeks_to_5pct",
                "peak_velocity_month", "peak_velocity_percentage_points",
                "latest_observed_share", "diagnostic_status",
            ]
        )
    category_text = steam["category"].astype("string").str.casefold()
    name_text = steam["metric_name"].astype("string").str.casefold()
    gpu_mask = category_text.str.contains("gpu|graphics|video card", regex=True, na=False) | name_text.str.contains(
        "gpu|graphics|video card", regex=True, na=False
    )
    gpu = steam[gpu_mask].copy()
    gpu["survey_month"] = pd.to_datetime(gpu["survey_month"])
    gpu["metric_value"] = pd.to_numeric(gpu["metric_value"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        event_date = pd.Timestamp(event.event_date)
        generation = str(event.generation)
        matched = gpu[gpu["generation"].astype(str).eq(generation)]
        if matched.empty and generation.upper() not in {"GENERIC", "UNKNOWN", "NAN"}:
            matched = gpu[gpu["metric_name"].astype(str).str.contains(generation, case=False, regex=False, na=False)]
        monthly = matched.groupby("survey_month", as_index=False)["metric_value"].sum().sort_values("survey_month")
        before = monthly[monthly["survey_month"] < event_date]
        after = monthly[monthly["survey_month"] >= event_date].copy()
        baseline = float(before["metric_value"].iloc[-1]) if not before.empty else 0.0
        after["adoption_from_baseline"] = after["metric_value"] - baseline
        first_1 = after[after["adoption_from_baseline"] >= 1]
        first_5 = after[after["adoption_from_baseline"] >= 5]
        velocity = monthly.assign(monthly_change=monthly["metric_value"].diff())
        velocity = velocity[velocity["survey_month"] >= event_date]
        peak = velocity.loc[velocity["monthly_change"].idxmax()] if not velocity.empty and velocity["monthly_change"].notna().any() else None
        rows.append(
            {
                "event_date": event_date,
                "event_type": event.event_type,
                "product_name": event.product_name,
                "generation": generation,
                "matched_observations": int(len(matched)),
                "weeks_to_1pct": int((first_1.iloc[0]["survey_month"] - event_date).days // 7) if not first_1.empty else None,
                "weeks_to_5pct": int((first_5.iloc[0]["survey_month"] - event_date).days // 7) if not first_5.empty else None,
                "peak_velocity_month": peak["survey_month"] if peak is not None else None,
                "peak_velocity_percentage_points": float(peak["monthly_change"]) if peak is not None else None,
                "latest_observed_share": float(monthly["metric_value"].iloc[-1]) if not monthly.empty else None,
                "diagnostic_status": "OBSERVED" if not monthly.empty else "NO_MATCHING_GENERATION",
            }
        )
    return pd.DataFrame(rows)


def _future_policy(config_dir: Path, steam: pd.DataFrame) -> dict[str, Any]:
    model_config = load_yaml(config_dir / "models.yaml")
    steam_config = load_yaml(config_dir / "steam_hardware.yaml")
    return {
        "forecast_feature_set": model_config.get("forecast_feature_set"),
        "steam_future_strategy": steam_config.get("future_strategy", "OBSERVED_ONLY"),
        "publication_lag_months": steam_config.get("publication_lag_months", 0),
        "last_observed_survey_month": steam["survey_month"].max() if not steam.empty else None,
        "future_rows_fabricated": False,
        "interpretation": "Steam is used only through its last observed survey month; no future Steam values are generated.",
    }


def _summarize_backtest(backtest: pd.DataFrame) -> pd.DataFrame:
    metrics = ["wape", "mae", "rmse", "bias", "fva"]
    summary = (
        backtest.groupby(["feature_set", "model", "horizon_bucket"], dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part) != "").rstrip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _ablation(backtest: pd.DataFrame) -> pd.DataFrame:
    keys = ["cutoff", "model", "horizon_bucket"]
    metrics = ["wape", "mae", "rmse", "bias", "fva"]
    without_steam = backtest[backtest["feature_set"].eq("gpu_signal")].set_index(keys)[metrics].add_suffix("_without_steam")
    with_steam = backtest[backtest["feature_set"].eq("all_external")].set_index(keys)[metrics].add_suffix("_with_steam")
    result = with_steam.join(without_steam, how="inner").reset_index()
    for metric in metrics:
        result[f"{metric}_delta_with_steam_minus_without"] = result[f"{metric}_with_steam"] - result[f"{metric}_without_steam"]
    return result
def _shap_stability(features: pd.DataFrame, steam_signal_columns: list[str], feature_set: str, cutoffs: list[Any]) -> pd.DataFrame:
    excluded = set(steam_signal_columns) if feature_set == "gpu_signal" else set()
    model_excluded = {
        "week", "target", "pos_qty", "demand_qty", "demand_revenue",
        "business_family", "region", "channel", "source_id",
    }
    candidate = [
        column
        for column in features.columns
        if column not in model_excluded and column not in excluded
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    rows: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        clean = features.loc[features["week"] <= cutoff, ["target", *candidate]].dropna()
        if len(clean) < 30 or not candidate:
            continue
        clean = clean.tail(min(len(clean), 256))
        model = LGBMRegressor(
            n_estimators=40,
            learning_rate=0.05,
            max_depth=3,
            random_state=7,
            n_jobs=1,
            verbosity=-1,
        )
        try:
            model.fit(clean[candidate], clean["target"])
            values = shap.TreeExplainer(model)(clean[candidate]).values
        except (ValueError, TypeError):
            continue
        shap_frame = pd.DataFrame(
            {
                "feature_name": candidate,
                "shap_value": np.asarray(values).mean(axis=0),
                "abs_shap": np.asarray(values).mean(axis=0).__abs__(),
                "cutoff": cutoff,
                "feature_set": feature_set,
            }
        )
        total = float(shap_frame["abs_shap"].sum())
        shap_frame["abs_shap_share"] = shap_frame["abs_shap"] / total if total else 0.0
        shap_frame["top_10"] = shap_frame["abs_shap"].rank(method="first", ascending=False) <= 10
        rows.append(shap_frame)
        del model, values, clean
        gc.collect()
    if not rows:
        return pd.DataFrame(columns=["feature_set", "feature_name", "mean_abs_shap_share", "std_abs_shap_share", "top_10_fold_frequency"])
    combined = pd.concat(rows, ignore_index=True)
    return (
        combined.groupby(["feature_set", "feature_name"], as_index=False)
        .agg(
            mean_abs_shap_share=("abs_shap_share", "mean"),
            std_abs_shap_share=("abs_shap_share", "std"),
            top_10_fold_frequency=("top_10", "mean"),
            folds=("cutoff", "nunique"),
        )
        .sort_values(["mean_abs_shap_share", "top_10_fold_frequency"], ascending=False)
    )


def run_real_experiment(
    *,
    source_path: str | Path = "data/external/steam_hardware/shs.csv",
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    config_dir: str | Path = "config",
    fetch_external: bool = True,
    start_date: str = "2016-01-01",
    end_date: str = "2025-12-31",
) -> dict[str, Any]:
    """Run validation, observed Steam diagnostics, backtest comparison, and ablation."""
    print("[steam-real] Bắt đầu: đọc Steam extract và kiểm tra chất lượng...", flush=True)
    source_path = Path(source_path)
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    config_dir = Path(config_dir)
    print(f"[steam-real] Đang đọc nguồn: {source_path}", flush=True)
    steam = read_steam_hardware(source_path, source_id="STEAM_HARDWARE", source_name="jdegene steamHWsurvey shs.csv", source_url="https://github.com/jdegene/steamHWsurvey/blob/master/shs.csv", mapping_path=config_dir / "steam_hardware.yaml")
    steam = apply_steam_quality_flags(steam, anomaly_path=config_dir / "steam_anomalies.yaml")
    validation = validate_steam_source(steam, anomaly_path=config_dir / "steam_anomalies.yaml", mapping_path=config_dir / "steam_hardware.yaml")
    print(f"[steam-real] Đã đọc {len(steam):,} dòng; đang ghi validation artifacts...", flush=True)
    normalized = validation.get("normalized", steam)
    _write_json(output_dir / "validation" / "steam_real_validation.json", {"status": validation["status"], "summary": validation.get("summary", {}), "checks": validation["checks"].to_dict(orient="records")})
    _write_frame(output_dir / "validation" / "steam_real_validation_checks.csv", validation["checks"])
    _write_frame(output_dir / "validation" / "steam_real_normalized.csv", normalized)
    normalized.to_parquet(output_dir / "validation" / "steam_real_normalized.parquet", index=False)
    _write_frame(output_dir / "validation" / "steam_gpu_mapping_review.csv", _latest_gpu_mapping_review(normalized))

    events_path = config_dir / "launch_events.csv"
    events = read_launch_events(events_path) if events_path.exists() else pd.DataFrame()
    weekly = derive_steam_signals(normalized, mapping_path=config_dir / "steam_hardware.yaml", publication_lag_months=int(load_yaml(config_dir / "steam_hardware.yaml").get("publication_lag_months", 0)))
    launch_diagnostics = _launch_diagnostics(normalized, events) if not events.empty else diagnose_steam_launch_adoption(weekly, events)
    _write_frame(output_dir / "validation" / "steam_release_adoption_diagnostics.csv", launch_diagnostics)
    _write_json(output_dir / "validation" / "steam_future_policy.json", _future_policy(config_dir, normalized))
    print("[steam-real] Validation xong; bắt đầu rolling model comparison (có thể mất vài phút)...", flush=True)

    pipeline_result = run_all(root_dir=root_dir, output_dir=output_dir, fetch_external=fetch_external, start_date=start_date, end_date=end_date, config_dir=config_dir)
    backtest = pipeline_result["backtest"]
    print(f"[steam-real] Backtest xong: {len(backtest):,} rows; đang tổng hợp ablation...", flush=True)
    features = pipeline_result["features"].copy()
    _write_frame(output_dir / "validation" / "steam_model_comparison.csv", _summarize_backtest(backtest))
    _write_frame(output_dir / "validation" / "steam_ablation.csv", _ablation(backtest))
    fold = backtest[backtest["feature_set"].isin(["gpu_signal", "all_external"])].copy()
    fold_summary = fold.groupby(["feature_set", "cutoff", "model", "horizon_bucket"], as_index=False)[["wape", "mae", "rmse", "bias", "fva"]].mean()
    _write_frame(output_dir / "validation" / "steam_fold_stability.csv", fold_summary)
    steam_signal_columns = [column for column in STEAM_SIGNAL_COLUMNS if column in features.columns]
    cutoffs = sorted(backtest["cutoff"].dropna().unique().tolist())
    shap_parts = []
    for feature_set in ("gpu_signal", "all_external"):
        print(f"[steam-real] SHAP stability: {feature_set} ({len(cutoffs)} cutoffs)...", flush=True)
        shap_parts.append(_shap_stability(features, steam_signal_columns, feature_set, cutoffs))
    shap = pd.concat(shap_parts, ignore_index=True)
    _write_frame(output_dir / "validation" / "steam_shap_stability.csv", shap)
    print("[steam-real] Đã ghi SHAP; hoàn tất và trả kết quả.", flush=True)
    _write_json(output_dir / "validation" / "steam_experiment_report.json", {"validation": validation.get("summary", {}), "pipeline_status": pipeline_result["status"], "artifacts": sorted(str(path.relative_to(output_dir)) for path in (output_dir / "validation").glob("steam_*.csv"))})
    return {"status": pipeline_result["status"], "validation": validation["status"], "rows": len(normalized), "backtest_rows": len(backtest), "output_dir": str(output_dir / "validation")}


if __name__ == "__main__":
    print(json.dumps(run_real_experiment(), indent=2, default=_json_default))
