"""Unified visual dashboard for DFSignal technical and business users."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from dfsignal.reporting.business_language import explain_model, generate_latest_executive_summary

st.set_page_config(page_title="DFSignal | Demand Intelligence", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation"
HORIZON_ORDER = ["1-4", "5-8", "9-13", "14-26"]
MODEL_LABELS = {
    "seasonal_naive": "Seasonal Baseline",
    "ets": "Trend & Seasonal Forecast",
    "lightgbm": "Market-Signal Forecast",
}
FEATURE_SET_LABELS = {
    "history_only": "Demand history",
    "calendar": "Calendar effects",
    "macro": "Economic conditions",
    "gpu_signal": "Hardware releases",
    "all_external": "All available signals",
}


@st.cache_data(ttl=30)
def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def _read_json(path: str) -> dict[str, Any]:
    file_path = Path(path)
    return json.loads(file_path.read_text(encoding="utf-8")) if file_path.exists() else {}


@st.cache_data(ttl=30)
def _read_parquet(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return pd.read_parquet(file_path) if file_path.exists() else pd.DataFrame()


def _pct(value: Any, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}%}"


def _pp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.2f} pp"


def _number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def _load_results() -> dict[str, Any]:
    summary_path = OUTPUT_DIR / "latest_executive_summary.md"
    assumptions_path = ROOT_DIR / "config" / "assumptions.yaml"
    assumptions = yaml.safe_load(assumptions_path.read_text(encoding="utf-8")) if assumptions_path.exists() else {}
    return {
        "validation": _read_json(str(VALIDATION_DIR / "steam_real_validation.json")),
        "checks": _read_csv(str(VALIDATION_DIR / "steam_real_validation_checks.csv")),
        "comparison": _read_csv(str(VALIDATION_DIR / "steam_model_comparison.csv")),
        "ablation": _read_csv(str(VALIDATION_DIR / "steam_ablation.csv")),
        "fold": _read_csv(str(VALIDATION_DIR / "steam_fold_stability.csv")),
        "shap": _read_csv(str(VALIDATION_DIR / "steam_shap_stability.csv")),
        "mapping": _read_csv(str(VALIDATION_DIR / "steam_gpu_mapping_review.csv")),
        "policy": _read_json(str(VALIDATION_DIR / "steam_future_policy.json")),
        "gate": _read_json(str(OUTPUT_DIR / "phase1_exit_gate" / "phase1_gate.json")),
        "forecast": _read_parquet(str(OUTPUT_DIR / "forecast.parquet")),
        "backtest": _read_parquet(str(OUTPUT_DIR / "backtest.parquet")),
        "demand": _read_parquet(str(OUTPUT_DIR / "feature_dataset.parquet")),
        "executive_summary": summary_path.read_text(encoding="utf-8") if summary_path.exists() else "",
        "assumptions": assumptions.get("assumptions", []) if isinstance(assumptions, dict) else [],
    }


def _polish(fig: Any) -> Any:
    fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20), legend_title_text="", hovermode="x unified")
    return fig


def _style_page() -> None:
    st.markdown("""<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stMetricValue"] { font-size: 1.45rem; }
    .hero { padding: 1.25rem 1.5rem; border-radius: 14px; background: linear-gradient(135deg, #172554 0%, #1d4ed8 62%, #0f766e 100%); color: white; margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; color: #dbeafe; }
    .badge { display: inline-block; padding: .25rem .6rem; margin: .8rem .35rem 0 0; border-radius: 999px; font-size: .75rem; font-weight: 700; background: rgba(255,255,255,.16); }
    .section-kicker { text-transform: uppercase; letter-spacing: .12em; color: #64748b; font-size: .7rem; font-weight: 800; margin-top: .8rem; }
    </style>""", unsafe_allow_html=True)


def _header(data: dict[str, Any]) -> None:
    gate = data["gate"]
    validation = data["validation"]
    demand = data["demand"]
    demand_end = pd.to_datetime(demand["week"], errors="coerce").max().strftime("%Y-%m-%d") if not demand.empty and "week" in demand else "not available"
    st.markdown(
        f'<div class="hero"><h1>DFSignal</h1><p>Demand direction, forecast range, signal context, and data health in one view.</p>'
        f'<span class="badge">Latest data: {demand_end}</span><span class="badge">Data: synthetic test demand</span>'
        f'<span class="badge">Project mode: public-data testing</span><span class="badge">Ready for continued testing</span></div>',
        unsafe_allow_html=True,
    )
    if gate.get("gate_status") == "CONDITIONAL_GO":
        st.caption("Current status: ready for continued testing, but business value has not yet been proven with real company demand.")


def _forecast_frame(data: dict[str, Any], horizon: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand = data["demand"].copy()
    forecast = data["forecast"].copy()
    if not demand.empty and "week" in demand:
        demand["week"] = pd.to_datetime(demand["week"], errors="coerce")
        target = "pos_qty" if "pos_qty" in demand else next((c for c in demand.columns if c.endswith("qty")), None)
        demand = demand.groupby("week", as_index=False)[target].sum().rename(columns={target: "value"}) if target else pd.DataFrame()
    if not forecast.empty:
        forecast["week"] = pd.to_datetime(forecast.get("week"), errors="coerce")
        if "model_name" in forecast:
            forecast = forecast[forecast["model_name"].eq("ets")].copy()
        forecast = forecast[forecast["horizon_weeks"].eq(horizon)].copy() if "horizon_weeks" in forecast else forecast.head(0)
        if "point_forecast" in forecast and "value" not in forecast:
            forecast = forecast.rename(columns={"point_forecast": "value"})
        forecast = forecast.groupby("week", as_index=False).agg({c: "mean" for c in ["value", "lower_bound", "upper_bound"] if c in forecast})
    return demand, forecast


def _direction(demand: pd.DataFrame) -> str:
    if demand.empty or len(demand) < 12:
        return "Not available"
    recent = demand["value"].tail(12).mean()
    prior = demand["value"].tail(24).head(12).mean()
    change = (recent - prior) / max(abs(prior), 1e-9)
    return "Increasing" if change > .05 else "Decreasing" if change < -.05 else "Stable"


def _overview(data: dict[str, Any]) -> None:
    _header(data)
    demand, forecast = _forecast_frame(data, 4)
    gate_metrics = data["gate"].get("key_metrics", {})
    backtest = data["backtest"]
    bias = backtest["bias"].mean() if not backtest.empty and "bias" in backtest else None
    latest = pd.to_datetime(demand["week"], errors="coerce").max().strftime("%Y-%m-%d") if not demand.empty else "—"
    error = gate_metrics.get("best_feature_set_wape")
    steam_value = gate_metrics.get("lightgbm_steam_vs_gpu_wape_improvement")
    cards = st.columns(6)
    cards[0].metric("Expected Direction", _direction(demand))
    cards[1].metric("Next 4 Weeks", _pct((forecast["value"].mean() / demand["value"].tail(4).mean() - 1) if not forecast.empty and not demand.empty else None))
    cards[2].metric("Historical Forecast Error", _pct(error), help="On average, past forecasts were about this far from actual values.")
    cards[3].metric("Forecast Tendency", "Tends high" if bias is not None and bias < 0 else "Tends low" if bias is not None and bias > 0 else "Balanced")
    cards[4].metric("Data Status", "Review needed")
    cards[5].metric("Extra Signal Value", "Mixed" if steam_value is not None else "Not tested", help="Whether extra market information improved historical accuracy; not a causal result.")
    st.markdown('<div class="section-kicker">Main forecast</div>', unsafe_allow_html=True)
    st.subheader("Where demand is heading")
    if demand.empty or forecast.empty:
        st.info("A forecast chart will appear when demand and forecast artifacts are available.")
    else:
        context = demand.tail(104).rename(columns={"value": "Actual"})
        future = forecast[["week", "value"]].rename(columns={"value": "Forecast"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=context["week"], y=context["Actual"], name="Actual", mode="lines", line=dict(color="#2563eb", width=2)))
        if {"lower_bound", "upper_bound"}.issubset(forecast.columns):
            fig.add_trace(go.Scatter(x=forecast["week"], y=forecast["upper_bound"], name="Expected Range", mode="lines", line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast["week"], y=forecast["lower_bound"], mode="lines", fill="tonexty", fillcolor="rgba(14,116,144,.18)", line=dict(width=0), name="Expected Range"))
        fig.add_trace(go.Scatter(x=future["week"], y=future["Forecast"], name="Forecast", mode="lines+markers", line=dict(color="#0f766e", width=3)))
        fig.add_vline(x=context["week"].max().timestamp() * 1000, line_dash="dot", line_color="#64748b", annotation_text="Forecast starts")
        fig.update_layout(xaxis_title="Week", yaxis_title="Demand", hovermode="x unified")
        st.plotly_chart(_polish(fig), use_container_width=True)
        st.caption("Expected Range shows a reasonable range around the forecast. It is not a guarantee. Technical name: prediction interval.")
    st.markdown("### What This Means")
    st.info("Demand is expected to remain relatively stable over the next several weeks. Historical demand and recurring seasonal patterns remain the strongest demonstrated signals. Hardware-release and gaming-adoption data currently add limited forecast improvement. These results use synthetic demand, so business value is not yet validated with real company data.")
    left, right = st.columns(2)
    with left:
        st.markdown("### What's Changing")
        change_rows = [
            {"Area": "Recent Demand", "Direction": _direction(demand)},
            {"Area": "Hardware Releases", "Direction": "Mixed"},
            {"Area": "Gaming Hardware Adoption", "Direction": "Stable"},
            {"Area": "Economic Conditions", "Direction": "Unavailable"},
        ]
        st.dataframe(pd.DataFrame(change_rows), hide_index=True, use_container_width=True)
    with right:
        st.markdown("### What the Forecast Relied On")
        importance = data["shap"]
        if importance.empty:
            st.info("Model driver details are not available.")
        else:
            st.dataframe(pd.DataFrame([
                {"Driver": "Recent Demand", "Relative use": "Highest"},
                {"Driver": "Typical Seasonal Pattern", "Relative use": "High"},
                {"Driver": "Hardware Releases", "Relative use": "Limited"},
                {"Driver": "Gaming Hardware Adoption", "Relative use": "Some use"},
            ]), hide_index=True, use_container_width=True)
    st.markdown("### Next Step")
    st.success("Connect a small de-identified real internal demand extract and run a point-in-time historical test.")


def _forecast_tab(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Forecast</div>', unsafe_allow_html=True)
    horizon = st.select_slider("Forecast horizon", options=[4, 8, 13, 26], value=4, format_func=lambda x: f"Next {x} weeks")
    demand, forecast = _forecast_frame(data, horizon)
    if demand.empty or forecast.empty:
        st.info("Forecast data is not available for this horizon.")
        return
    context = demand.tail(104)
    fig = px.line(context, x="week", y="value", labels={"value": "Actual demand", "week": "Week"}, title=f"Actual demand and forecast · next {horizon} weeks")
    fig.update_traces(name="Actual", showlegend=True)
    fig.add_scatter(x=forecast["week"], y=forecast["value"], mode="lines+markers", name="Forecast", line=dict(color="#0f766e", width=3))
    if {"lower_bound", "upper_bound"}.issubset(forecast.columns):
        fig.add_scatter(x=forecast["week"], y=forecast["upper_bound"], mode="lines", line=dict(width=0), showlegend=False)
        fig.add_scatter(x=forecast["week"], y=forecast["lower_bound"], mode="lines", fill="tonexty", fillcolor="rgba(14,116,144,.18)", line=dict(width=0), name="Expected Range")
    st.plotly_chart(_polish(fig), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader("Typical Demand Pattern")
        seasonal = demand.assign(week_of_year=demand["week"].dt.isocalendar().week.astype(int)).groupby("week_of_year", as_index=False)["value"].mean()
        st.plotly_chart(_polish(px.line(seasonal, x="week_of_year", y="value", labels={"week_of_year": "Week of year", "value": "Typical demand"}, title="Typical Pattern Through the Year")), use_container_width=True)
    with right:
        st.subheader("How This Year Compares")
        yoy = demand.assign(year=demand["week"].dt.year, week_of_year=demand["week"].dt.isocalendar().week.astype(int))
        recent_year = int(yoy["year"].max())
        view = yoy[yoy["year"].isin([recent_year, recent_year - 1])]
        st.plotly_chart(_polish(px.line(view, x="week_of_year", y="value", color="year", labels={"week_of_year": "Week of year", "value": "Demand", "year": "Year"}, title="This Year vs Previous Year")), use_container_width=True)
    st.subheader("How Accuracy Changes Further Into the Future")
    bt = data["backtest"]
    if not bt.empty:
        horizon_view = bt.groupby("horizon_bucket", as_index=False)["wape"].mean()
        horizon_view["horizon_bucket"] = pd.Categorical(horizon_view["horizon_bucket"], HORIZON_ORDER, ordered=True)
        st.plotly_chart(_polish(px.bar(horizon_view.sort_values("horizon_bucket"), x="horizon_bucket", y="wape", labels={"horizon_bucket": "Forecast distance", "wape": "Historical Forecast Error"}, title="Historical Accuracy by Forecast Distance")), use_container_width=True)
    with st.expander("Advanced Details"):
        st.dataframe(forecast, hide_index=True, use_container_width=True)
        st.caption("Technical fields include model name, WAPE, interval width, and forecast origin. The expected range uses residual-based interval calibration.")


def _signals_tab(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Signals</div>', unsafe_allow_html=True)
    st.header("What may be influencing the forecast")
    rows = [
        {"Signal": "Recent Demand", "What it represents": "Recent demand behavior", "Status": "STRONG", "Historical value": "Dominant demonstrated input"},
        {"Signal": "Typical Seasonal Pattern", "What it represents": "Recurring pattern through the year", "Status": "STRONG", "Historical value": "Used by the best current method"},
        {"Signal": "Hardware Releases", "What it represents": "PC hardware upgrade cycle", "Status": "MIXED", "Historical value": "Helped some periods; hurt others"},
        {"Signal": "Gaming Hardware Adoption", "What it represents": "Hardware used by surveyed PC gamers", "Status": "NEUTRAL", "Historical value": "Little added value overall so far"},
        {"Signal": "Economic Conditions", "What it represents": "Consumer/economic environment", "Status": "NOT_AVAILABLE", "Historical value": "Not enough valid data"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("These are signals the model used or tested. They do not prove that one factor caused another.")
    st.subheader("Did Extra Signals Improve the Forecast?")
    comparison = data["comparison"]
    if comparison.empty:
        st.info("Signal comparison is not available.")
    else:
        view = comparison[comparison["model"].eq("lightgbm")].copy()
        view["Information used"] = view["feature_set"].map(FEATURE_SET_LABELS).fillna(view["feature_set"])
        view = view.groupby("Information used", as_index=False)["wape_mean"].mean()
        st.plotly_chart(_polish(px.bar(view.sort_values("wape_mean"), x="wape_mean", y="Information used", orientation="h", labels={"wape_mean": "Historical Forecast Error", "Information used": "Information added"}, title="Lower historical error is better")), use_container_width=True)
    with st.expander("Advanced signal details"):
        st.dataframe(data["ablation"], hide_index=True, use_container_width=True)
        st.caption("Technical comparison: FVA and fold-level results. SHAP is model-attributed influence, not causation.")


def _data_tab(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Data</div>', unsafe_allow_html=True)
    st.header("Data available")
    metrics = data["gate"].get("key_metrics", {})
    validation = data["validation"].get("summary", {})
    rows = [
        {"Data source": "Historical Demand", "Available through": metrics.get("demand_period_end", "—"), "Status": "Available", "Notes": "Synthetic test demand"},
        {"Data source": "GPU Releases", "Available through": "2026-01-01", "Status": "Needs Review", "Notes": "Some products and generations need review"},
        {"Data source": "Gaming Hardware Adoption", "Available through": validation.get("max_date", "—"), "Status": "Needs Review", "Notes": f"{metrics.get('steam_share_weighted_mapping_coverage_pct', '—')}% share-weighted classification"},
        {"Data source": "Economic Data", "Available through": "—", "Status": "Unavailable", "Notes": "No valid FRED Silver rows in this run"},
        {"Data source": "Internal Demand", "Available through": "—", "Status": "Not Connected", "Notes": "Real internal demand has not been connected"},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.info("Current project state: synthetic test demand is active; public market data is available; real internal demand is not connected; Fabric is planned, not implemented.")
    warnings = ["Economic data is currently unavailable.", "Some GPU products are not fully classified.", "Internal demand has not yet been connected."]
    st.subheader("Important warnings")
    for warning in warnings:
        st.warning(warning)
    with st.expander("Technical source details"):
        st.json(data["validation"])


def _about_tab(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">About</div>', unsafe_allow_html=True)
    st.header("About DFSignal")
    st.write("DFSignal combines historical demand with external market signals to test whether those signals improve demand forecasting.")
    st.write("It currently uses synthetic demand for testing, public hardware-release information, Steam Hardware Survey adoption information, calendar patterns, and configured economic indicators when valid data is available.")
    st.warning("External signals are used for prediction and context. They do not prove causation.")
    with st.expander("Model explanations"):
        for model, description in MODEL_LABELS.items():
            st.markdown(f"**{description}** — {explain_model(model)}")
    with st.expander("Important notes"):
        for assumption in data["assumptions"][:5]:
            st.markdown(f"- {assumption.get('plain_english', assumption.get('description', ''))}")
    with st.expander("Glossary and limitations"):
        st.markdown("See `docs/business_glossary.md` for definitions. Current limitations include synthetic demand, incomplete Steam classification, GPU mapping review, unavailable macro comparison, and unproven historical coverage of forecast ranges.")


def main() -> None:
    _style_page()
    data = _load_results()
    if not data["executive_summary"]:
        generate_latest_executive_summary(OUTPUT_DIR)
        data["executive_summary"] = (OUTPUT_DIR / "latest_executive_summary.md").read_text(encoding="utf-8")
    if data["demand"].empty and data["forecast"].empty:
        st.title("DFSignal")
        st.warning("No demand or forecast data is available yet. Run the pipeline, then refresh this page.")
        return
    with st.sidebar:
        st.header("Filters")
        date_window = st.selectbox("Date window", ["52 weeks", "104 weeks", "All available"], index=1)
        st.caption("The date window changes the historical charts. Use the Forecast tab to change the forecast horizon.")
        if date_window != "All available" and not data["demand"].empty:
            data["demand"] = data["demand"].tail(int(date_window.split()[0])).copy()
        if st.button("Refresh data"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.caption("Start with Overview. Technical metrics are inside Advanced Details sections.")
    tabs = st.tabs(["Overview", "Forecast", "Signals", "Data", "About"])
    with tabs[0]:
        _overview(data)
    with tabs[1]:
        _forecast_tab(data)
    with tabs[2]:
        _signals_tab(data)
    with tabs[3]:
        _data_tab(data)
    with tabs[4]:
        _about_tab(data)


if __name__ == "__main__":
    main()
