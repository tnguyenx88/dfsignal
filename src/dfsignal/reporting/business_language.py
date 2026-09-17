"""Plain-English translations for DFSignal technical outputs.

The helpers deliberately describe predictive results without turning them into
causal claims. Technical values remain available to callers and artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MODEL_DESCRIPTIONS = {
    "seasonal_naive": "Assume demand follows a similar pattern to the previous seasonal period.",
    "ets": "Forecast based mainly on historical level, trend, and seasonality.",
    "lightgbm": "Forecast using historical demand plus optional market signals.",
}

METRIC_HELP = {
    "wape": "Measures how far predictions were from actual demand overall. Lower is better.",
    "bias": "Shows whether the forecast tends to predict too high or too low.",
    "fva": "Measures whether adding a signal improved accuracy compared with a simpler baseline.",
    "prediction_interval": "A range around the forecast that represents expected uncertainty; it is not a guarantee.",
}


def _number(value: Any, digits: int = 1) -> str:
    return "not available" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 1) -> str:
    return "not available" if value is None else f"{float(value) * 100:.{digits}f}%"


def explain_metric(metric: str, value: float | None, *, baseline: float | None = None) -> str:
    """Translate one metric while retaining its direction and uncertainty."""
    key = metric.lower()
    if value is None:
        return f"{metric} is not available for this result."
    if key == "wape":
        return f"Forecast error was approximately {_percent(value)}."
    if key == "mae":
        return f"The forecast missed actual demand by about {_number(value)} units on average."
    if key == "rmse":
        return f"The typical error, with larger misses weighted more heavily, was about {_number(value)} units."
    if key == "mase":
        return f"The forecast error was {_number(value, 2)} times the comparison method's typical error."
    if key == "bias":
        if value > 0:
            return "The forecast tended to predict too low."
        if value < 0:
            return "The forecast tended to predict too high."
        return "The forecast had no clear tendency to be too high or too low."
    if key in {"fva", "fva_pct", "wape_improvement"}:
        if key == "fva_pct":
            magnitude = _percent(value)
        else:
            magnitude = _percent(value, 2)
        if value > 0:
            return f"Adding this signal reduced historical forecast error by about {magnitude} compared with the baseline."
        if value < 0:
            return f"Adding this signal increased historical forecast error by about {_percent(abs(value), 2)} compared with the baseline."
        return "Adding this signal did not change historical forecast error."
    if key in {"interval_width", "prediction_interval_width"}:
        return "The forecast range is relatively wide, so there is more uncertainty around the estimate." if value > 0 else "The forecast range is relatively narrow."
    if key in {"mapping_coverage", "share_weighted_coverage"}:
        return f"About {_percent(value)} of the source is mapped well enough for this analysis."
    return f"{metric}: {_number(value)}."


def explain_model(model: str) -> str:
    return MODEL_DESCRIPTIONS.get(model.lower(), "A forecasting method evaluated against the same historical test periods.")


def explain_quality_status(status: str) -> str:
    return {
        "PASS": "No material data issue was detected.",
        "REVIEW": "The data can be used for testing, but some quality issues should be reviewed.",
        "FAIL": "The issue is significant enough that results should not be trusted yet.",
        "MISSING_OPTIONAL": "This optional dataset is not currently available, but the rest of the pipeline can still run.",
    }.get(str(status).upper(), "The data status is not yet classified.")


def explain_gate(status: str) -> str:
    return {
        "GO": "The system is ready for the next validation stage.",
        "CONDITIONAL_GO": "The system is ready to continue testing, but important questions remain.",
        "STOP_AND_FIX": "Results should not be used until the identified issue is corrected.",
    }.get(str(status).upper(), "The gate status is not yet classified.")


def confidence_language(label: str) -> str:
    return {
        "STRONG EVIDENCE": "The result was consistent across most historical test periods.",
        "MIXED EVIDENCE": "The signal helped in some periods but hurt in others.",
        "LIMITED EVIDENCE": "The result is based on limited data or unstable performance.",
        "NOT YET TESTED": "The pipeline supports this capability, but it has not yet been validated.",
    }.get(label.upper(), "The evidence strength is not yet classified.")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_gate(root: Path) -> dict[str, Any]:
    return _read_json(root / "phase1_exit_gate" / "phase1_gate.json")


def generate_latest_executive_summary(output_dir: str | Path = "outputs") -> Path:
    """Generate the compact business summary from current persisted evidence."""
    root = Path(output_dir)
    gate = _load_gate(root)
    metrics = gate.get("key_metrics", {})
    status = gate.get("gate_status", "NOT_YET_TESTED")
    best = metrics.get("best_feature_set_wape")
    text = [
        "# Executive Summary", "",
        "## What DFSignal currently sees", "",
        f"Historical testing currently shows about {_percent(best)} forecast error at the best observed model result. This is a synthetic-demand result, so it describes how the pipeline behaved in a controlled test rather than actual Corsair demand.",
        "", "## What is driving the forecast", "",
        "- Historical demand and recurring seasonal patterns are the strongest demonstrated inputs.",
        "- GPU release signals provide mixed additional value across forecast horizons.",
        "- Steam hardware adoption provides market context but is effectively neutral overall so far.",
        "- Macro indicators are not included as a clean observed comparison because the current FRED Silver output is empty.",
        "", "## What changed", "",
        "No comparable prior run is available for a defensible run-to-run comparison. The current artifacts cover demand through "
        f"{metrics.get('demand_period_end', 'not available')} and Steam observations through the latest available survey period.",
        "", "## What we are confident about", "",
        "The pipeline is technically functioning end to end. ETS currently performs best in the synthetic backtests, and historical demand dominates model-attributed importance.",
        "", "## What remains uncertain", "",
        "Synthetic demand may not represent real buying behavior. Steam classification is incomplete, GPU mapping requires review, and forecast-range historical coverage has not yet been proven.",
        "", "## Key Assumptions", "",
        "- Demand is modeled weekly, with weeks ending Saturday.",
        "- Future external values are observed-only; future Steam values are not guessed.",
        "- External signals and SHAP outputs indicate predictive/model-attributed relationships, not causation.",
        "- Real internal demand is not yet connected.",
        "", "## Data quality warnings", "",
        f"- Steam share-weighted mapping coverage: {_number(metrics.get('steam_share_weighted_mapping_coverage_pct'), 1)}% overall.",
        f"- Latest-month Steam share-weighted coverage: {_number(metrics.get('steam_latest_share_weighted_coverage_pct'), 1)}%.",
        "- GPU generation mappings and the missing FRED Silver data remain under review.",
        "", "## Recommended next step", "",
        "The system is ready for continued testing, but business value has not yet been proven with real demand data. Connect a small de-identified real internal demand extract and run a point-in-time backtest.",
        "", f"Gate: **{status}** — {explain_gate(status)}", "",
        "Exact metrics remain in `phase1_exit_gate/phase1_summary.md` and the detailed CSV/JSON artifacts.",
    ]
    path = root / "latest_executive_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path


def write_latest_recap(output_dir: str | Path = "outputs", *, previous: Mapping[str, Any] | None = None) -> Path:
    """Write a short, structured recap from the latest validation and gate artifacts."""
    root = Path(output_dir)
    validation = _read_json(root / "validation" / "steam_real_validation.json")
    gate = _load_gate(root)
    summary = validation.get("summary", {})
    metrics = gate.get("key_metrics", {})
    status = gate.get("gate_status", "NOT_YET_TESTED")
    lines = [
        "# DFSignal Run Recap", "",
        "## WHAT HAPPENED", "- Existing synthetic demand, GPU, and Steam artifacts were processed.", "- No confidential company data was used.", "",
        "## WHAT CHANGED", "- No comparable prior run is available; no unsupported comparison is reported.", "",
        "## WHAT MATTERS", f"- Best historical forecast error was {_percent(metrics.get('best_feature_set_wape'))} using {str(metrics.get('best_model', 'selected model')).upper()}.", "- Historical demand remained the strongest demonstrated input.", "- External-signal value was mixed or effectively neutral.", "",
        "## WHAT IS UNCERTAIN", "- Results are based on synthetic demand, not real Corsair demand.", "- Steam mapping and GPU generation classification remain incomplete or under review.", "- Historical forecast-range coverage has not yet been proven.", "",
        "## ASSUMPTIONS", "- Weekly demand grain; Saturday week ending.", "- Future external data is observed-only.", "- Signal usefulness is predictive evidence, not causal evidence.", "",
        "## DATA QUALITY", f"- Steam share-weighted mapping coverage: {_number(metrics.get('steam_share_weighted_mapping_coverage_pct'), 1)}% overall.", "- FRED Silver produced no rows in the current run.", "",
        "## NEXT STEP", "- Connect a small de-identified real internal demand extract and run a point-in-time backtest.", "", f"Gate: **{status}** — {explain_gate(status)}",
    ]
    path = root / "latest_recap.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def generate_phase1_summaries(output_dir: str | Path = "outputs") -> list[Path]:
    """Generate the three business-facing views from the existing Phase 1 gate."""
    root = Path(output_dir)
    gate = _read_json(root / "phase1_exit_gate" / "phase1_gate.json")
    metrics = gate.get("key_metrics", {})
    status = gate.get("gate_status", "NOT_YET_TESTED")
    target = root / "phase1_exit_gate"
    target.mkdir(parents=True, exist_ok=True)
    one_minute = [
        "# ONE-MINUTE SUMMARY", "",
        f"- DFSignal tested whether public market signals can help forecast weekly demand; the current gate is **{status}**.",
        f"- The test used {metrics.get('demand_rows', 'not available')} weeks of deterministic synthetic demand from {metrics.get('demand_period_start', 'not available')} to {metrics.get('demand_period_end', 'not available')}.",
        f"- ETS had the lowest historical error at {_percent(metrics.get('best_feature_set_wape'))}; this is a backtest result.",
        f"- GPU signals showed mixed value: overall improvement was {_percent(metrics.get('lightgbm_gpu_vs_macro_wape_improvement'), 2)}.",
        f"- Steam was effectively neutral overall: improvement was {_percent(metrics.get('lightgbm_steam_vs_gpu_wape_improvement'), 2)}.",
        "- Source coverage and the empty macro extract limit confidence.",
        "- The next important test is a small de-identified real internal demand extract.",
    ]
    business = [
        "# EXECUTIVE SUMMARY", "",
        "## WHAT WE TESTED", "",
        "We tested whether public GPU release information and Steam Hardware Survey adoption information could improve a weekly demand forecast. The experiment compared simple historical methods with models that could use these signals.", "",
        "## WHAT WE FOUND", "",
        f"Across 16 historical test cutoffs and four forecast horizons, the best average result came from ETS, a method based mainly on historical level, trend, and seasonality. Its historical forecast error was about {_percent(metrics.get('best_feature_set_wape'))}. Seasonal Naive, the simpler comparison, had about {_percent(metrics.get('seasonal_naive_wape'))} error.", "",
        "## WHAT HELPED", "",
        "GPU signals helped at longer horizons in this test, but the overall improvement was very small. This is mixed evidence, not a stable business uplift.", "",
        "## WHAT DID NOT HELP", "",
        "Steam was effectively neutral overall and helped in some horizons while hurting in others. These are predictive comparisons, not proof that either signal causes demand changes.", "",
        "## WHAT WE ARE NOT SURE ABOUT", "",
        "The demand series is synthetic, the macro source produced no rows, GPU generation coverage needs review, and Steam mapping is incomplete. Forecast ranges exist, but historical interval coverage has not yet been proven.", "",
        "## DATA QUALITY ISSUES", "",
        f"Steam mapping covered {_number(metrics.get('steam_share_weighted_mapping_coverage_pct'), 1)}% of observed share overall and {_number(metrics.get('steam_latest_share_weighted_coverage_pct'), 1)}% in the latest month. The configured macro source produced no rows in this run.", "",
        "## WHAT THIS MEANS", "",
        f"{explain_gate(status)} DFSignal currently demonstrates a repeatable research pipeline, not a production demand forecast for Corsair. Synthetic demand may not reproduce real buying behavior, promotions, inventory constraints, channel mix, or product availability.", "",
        "## WHAT WE SHOULD TEST NEXT", "",
        f"{gate.get('recommendation', 'Use a small de-identified real internal demand extract with documented grain, as-of timestamp, and owner.')}", "",
        "Technical evidence remains in `phase1_summary.md`, `phase1_gate.json`, and the CSV artifacts in this folder.",
    ]
    technical = [
        "# TECHNICAL DETAILS", "", "The technical outputs are unchanged and remain the source of truth for exact values.", "",
        "- [Phase 1 technical evidence report](phase1_summary.md)", "- [Phase 1 gate JSON](phase1_gate.json)", "- [Model leaderboard](model_leaderboard.csv)", "- [Horizon performance](horizon_performance.csv)", "- [GPU incremental value](gpu_incremental_value.csv)", "- [Steam incremental value](steam_incremental_value.csv)", "- [Fold stability](fold_stability.csv)", "- [Data quality summary](data_quality_summary.csv)", "- [Signal importance](signal_importance.csv)", "", "Business interpretation is a translation layer; it does not replace these artifacts or add unsupported conclusions.",
    ]
    paths = []
    for name, content in (("one_minute_summary.md", one_minute), ("executive_summary.md", business), ("technical_details.md", technical)):
        path = target / name
        path.write_text("\n".join(content) + "\n", encoding="utf-8")
        paths.append(path)
    return paths
def cli_recap(*, command: str, result: Mapping[str, Any]) -> str:
    """Return a concise terminal recap without hiding the technical JSON result."""
    status = result.get("status", {})
    if isinstance(status, Mapping):
        health = status.get("pipeline", {}).get("health", "completed") if isinstance(status.get("pipeline"), Mapping) else "completed"
    else:
        health = "completed"
    lines = [
        "",
        "DFSignal completed successfully.",
        "What happened:",
        "* Existing demand and public market-signal artifacts were processed.",
        "* Historical demand remained the strongest demonstrated source of forecast information.",
        "* External-signal value was mixed or effectively neutral in the current synthetic experiment.",
        f"* Pipeline health: {health}.",
        "Next recommended step: test a small de-identified real internal demand extract.",
    ]
    if command == "steam-real":
        lines[2] = "What happened:"
        lines[3] = "* Steam Hardware Survey was evaluated as a market signal, not direct customer demand."
    return "\n".join(lines)
