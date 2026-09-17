"""English Streamlit dashboard for DFSignal forecast and Steam evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="DFSignal | Demand Intelligence", layout="wide")

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "validation"
HORIZON_ORDER = ["1-4", "5-8", "9-13", "14-26"]
FEATURE_SET_LABELS = {
    "history_only": "A · History only",
    "calendar": "B · History + calendar",
    "macro": "C · History + macro",
    "gpu_signal": "D · GPU signal, no Steam",
    "all_external": "E · All external signals + Steam",
}
MODEL_LABELS = {"seasonal_naive": "Seasonal Naive", "ets": "ETS", "lightgbm": "LightGBM"}


@st.cache_data(ttl=30)
def _read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()


@st.cache_data(ttl=30)
def _read_json(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


@st.cache_data(ttl=30)
def _read_parquet(path: str) -> pd.DataFrame:
    file_path = Path(path)
    return pd.read_parquet(file_path) if file_path.exists() else pd.DataFrame()


def _pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def _pp(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.2f} pp"


def _number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def _source_status(validation: dict[str, Any]) -> tuple[str, str]:
    status = str(validation.get("status", "MISSING")).upper()
    if status == "PASS":
        return "PASS", "good"
    if status == "REVIEW":
        return "REVIEW", "review"
    return "MISSING", "bad"


def _polish(fig: Any) -> Any:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=65, b=20),
        legend_title_text="",
        hovermode="x unified",
    )
    return fig


def _style_page() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
        [data-testid="stMetricValue"] { font-size: 1.65rem; }
        .hero { padding: 1.5rem 1.7rem; border-radius: 16px; background: linear-gradient(135deg, #172554 0%, #1d4ed8 62%, #0f766e 100%); color: white; margin-bottom: 1.2rem; box-shadow: 0 10px 30px rgba(15, 23, 42, .25); }
        .hero h1 { margin: 0; font-size: 2.2rem; letter-spacing: -.03em; }
        .hero p { margin: .45rem 0 0; color: #dbeafe; font-size: 1.02rem; }
        .hero .meta { margin-top: 1rem; display: flex; gap: .55rem; flex-wrap: wrap; }
        .badge { display: inline-block; padding: .28rem .65rem; border-radius: 999px; font-size: .78rem; font-weight: 700; background: rgba(255,255,255,.16); color: #f8fafc; }
        .good { color: #34d399; font-weight: 700; }
        .review { color: #fbbf24; font-weight: 700; }
        .bad { color: #f87171; font-weight: 700; }
        .insight { border-left: 5px solid #38bdf8; background: #e0f2fe; color: #0f172a; padding: .9rem 1rem; border-radius: 8px; margin: .7rem 0 1rem; }
        .risk { border-left: 5px solid #f59e0b; background: #fffbeb; color: #0f172a; padding: .9rem 1rem; border-radius: 8px; margin: .7rem 0 1rem; }
        .section-kicker { text-transform: uppercase; letter-spacing: .12em; color: #64748b; font-size: .72rem; font-weight: 800; margin-bottom: .2rem; }
        .small-note { color: #94a3b8; font-size: .82rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _empty_state() -> None:
    st.warning("No result artifacts are available yet. Run the experiment first:")
    st.code("python -m dfsignal steam-real --no-fetch-external", language="bash")


def _load_results() -> dict[str, Any]:
    return {
        "validation": _read_json(str(VALIDATION_DIR / "steam_real_validation.json")),
        "checks": _read_csv(str(VALIDATION_DIR / "steam_real_validation_checks.csv")),
        "comparison": _read_csv(str(VALIDATION_DIR / "steam_model_comparison.csv")),
        "ablation": _read_csv(str(VALIDATION_DIR / "steam_ablation.csv")),
        "fold": _read_csv(str(VALIDATION_DIR / "steam_fold_stability.csv")),
        "shap": _read_csv(str(VALIDATION_DIR / "steam_shap_stability.csv")),
        "mapping": _read_csv(str(VALIDATION_DIR / "steam_gpu_mapping_review.csv")),
        "release": _read_csv(str(VALIDATION_DIR / "steam_release_adoption_diagnostics.csv")),
        "policy": _read_json(str(VALIDATION_DIR / "steam_future_policy.json")),
        "forecast": _read_parquet(str(OUTPUT_DIR / "forecast.parquet")),
        "backtest": _read_parquet(str(OUTPUT_DIR / "backtest.parquet")),
    }


def _render_header(validation: dict[str, Any], policy: dict[str, Any]) -> None:
    status, _ = _source_status(validation)
    strategy = policy.get("steam_future_strategy", "UNKNOWN")
    st.markdown(
        f'<div class="hero"><h1>DFSignal · External Demand Intelligence</h1>'
        '<p>Decision-oriented view of source quality, forecast performance, and the incremental value of Steam Hardware Survey signals.</p>'
        f'<div class="meta"><span class="badge">Source: {status}</span><span class="badge">Demand: synthetic / seeded</span><span class="badge">Steam future policy: {strategy}</span></div></div>',
        unsafe_allow_html=True,
    )


def _horizon_summary(ablation: pd.DataFrame, model: str) -> pd.DataFrame:
    if ablation.empty:
        return pd.DataFrame()
    view = ablation[ablation["model"].eq(model)].copy()
    if view.empty:
        return pd.DataFrame()
    result = view.groupby("horizon_bucket", as_index=False).agg(
        wape_delta=("wape_delta_with_steam_minus_without", "mean"),
        mae_delta=("mae_delta_with_steam_minus_without", "mean"),
        fva_delta=("fva_delta_with_steam_minus_without", "mean"),
        folds=("cutoff", "nunique"),
    )
    result["horizon_bucket"] = pd.Categorical(result["horizon_bucket"], HORIZON_ORDER, ordered=True)
    result = result.sort_values("horizon_bucket")
    result["readout"] = result["fva_delta"].map(lambda x: "Positive contribution" if x > 0.002 else "Negative contribution" if x < -0.002 else "Neutral")
    return result


def _overview(data: dict[str, Any], selected_model: str, selected_horizons: list[str], selected_feature_sets: list[str]) -> None:
    validation = data["validation"]
    summary = validation.get("summary", {})
    comparison = data["comparison"]
    ablation = data["ablation"]
    policy = data["policy"]
    _render_header(validation, policy)

    status, status_class = _source_status(validation)
    horizon = _horizon_summary(ablation, selected_model)
    best_uplift = horizon.loc[horizon["fva_delta"].idxmax(), "fva_delta"] if not horizon.empty else None
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Source status", status)
    k2.metric("Survey observations", _number(summary.get("rows")))
    k3.metric("Months represented", _number(summary.get("months_represented")))
    k4.metric("Best Steam FVA", _pp(best_uplift))
    k5.metric("Evaluated backtest rows", _number(len(data["backtest"])))
    st.markdown(f'<p class="{status_class}">Quality gate: {status} · coverage {summary.get("min_date", "—")} to {summary.get("max_date", "—")}</p>', unsafe_allow_html=True)

    if status == "REVIEW":
        st.markdown(
            f'<div class="risk"><b>Read this before the model results:</b> {summary.get("unknown_gpu_mappings", 0):,} GPU observations remain unmapped, '
            f'{len(summary.get("missing_months", []))} survey months are missing, and {len(summary.get("known_anomaly_periods", []))} anomaly periods are registered. '
            "The experiment completed, but source quality limits interpretation.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-kicker">Executive readout</div>', unsafe_allow_html=True)
    if not horizon.empty:
        positive = horizon[horizon["fva_delta"] > 0.002]["horizon_bucket"].astype(str).tolist()
        negative = horizon[horizon["fva_delta"] < -0.002]["horizon_bucket"].astype(str).tolist()
        neutral = horizon[horizon["fva_delta"].abs() <= 0.002]["horizon_bucket"].astype(str).tolist()
        positive_text = ", ".join(positive) if positive else "none"
        negative_text = ", ".join(negative) if negative else "none"
        neutral_text = ", ".join(neutral) if neutral else "none"
        st.markdown(
            f'<div class="insight"><b>Decision signal:</b> For {MODEL_LABELS.get(selected_model, selected_model)}, Steam shows a positive contribution at <b>{positive_text}</b>, '
            f'negative contribution at <b>{negative_text}</b>, and a neutral result at <b>{neutral_text}</b>. '
            "Treat this as a synthetic-data research result, not causal evidence or a production approval.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("No ablation summary is available for the selected model.")

    left, right = st.columns([1.55, 1])
    with left:
        if not comparison.empty:
            view = comparison[comparison["model"].eq(selected_model)].copy()
            if selected_horizons:
                view = view[view["horizon_bucket"].isin(selected_horizons)]
            if selected_feature_sets:
                view = view[view["feature_set"].isin(selected_feature_sets)]
            view["feature_set_label"] = view["feature_set"].map(FEATURE_SET_LABELS).fillna(view["feature_set"])
            fig = px.bar(
                view,
                x="horizon_bucket",
                y="wape_mean",
                color="feature_set_label",
                barmode="group",
                category_orders={"horizon_bucket": HORIZON_ORDER},
                labels={"horizon_bucket": "Forecast horizon", "wape_mean": "Mean WAPE", "feature_set_label": "Feature set"},
                title=f"Forecast error by horizon · {MODEL_LABELS.get(selected_model, selected_model)}",
                hover_data={"wape_mean": ":.2%", "wape_std": ":.2%"},
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(_polish(fig), width="stretch")
    with right:
        if not horizon.empty:
            readout = horizon.copy()
            readout["wape_delta"] = readout["wape_delta"].map(_pp)
            readout["mae_delta"] = readout["mae_delta"].map(lambda x: f"{x:+.2f}")
            readout["fva_delta"] = readout["fva_delta"].map(_pp)
            readout = readout[["horizon_bucket", "readout", "wape_delta", "mae_delta", "fva_delta", "folds"]]
            readout.columns = ["Horizon", "Readout", "Δ WAPE", "Δ MAE", "Δ FVA", "Folds"]
            st.dataframe(readout, hide_index=True, width="stretch")
            st.caption("Readout threshold: ±0.2 percentage points of FVA. Lower WAPE is better.")

    st.markdown('<div class="section-kicker">Model landscape</div>', unsafe_allow_html=True)
    if not comparison.empty:
        landscape = comparison.copy()
        landscape["feature_set_label"] = landscape["feature_set"].map(FEATURE_SET_LABELS).fillna(landscape["feature_set"])
        landscape = landscape.groupby(["feature_set_label", "model"], as_index=False)["wape_mean"].mean()
        landscape["model_label"] = landscape["model"].map(MODEL_LABELS).fillna(landscape["model"])
        fig = px.bar(
            landscape,
            x="wape_mean",
            y="feature_set_label",
            color="model_label",
            barmode="group",
            orientation="h",
            labels={"wape_mean": "Average WAPE across horizons", "feature_set_label": "Feature set", "model_label": "Model"},
            title="Average model error across all evaluated horizons",
        )
        fig.update_xaxes(tickformat=".1%")
        st.plotly_chart(_polish(fig), width="stretch")


def _ablation_view(data: dict[str, Any], selected_model: str) -> None:
    st.markdown('<div class="section-kicker">Incremental signal value</div>', unsafe_allow_html=True)
    st.subheader("Does Steam improve the forecast?")
    ablation = data["ablation"]
    if ablation.empty:
        st.info("No ablation artifact is available.")
        return
    selected = ablation[ablation["model"].eq(selected_model)].copy()
    summary = _horizon_summary(ablation, selected_model)
    if selected.empty or summary.empty:
        st.info("No ablation rows are available for the selected model.")
        return

    best = summary.loc[summary["fva_delta"].idxmax()]
    worst = summary.loc[summary["fva_delta"].idxmin()]
    a1, a2, a3 = st.columns(3)
    a1.metric("Best horizon", str(best["horizon_bucket"]), _pp(best["fva_delta"]))
    a2.metric("Weakest horizon", str(worst["horizon_bucket"]), _pp(worst["fva_delta"]))
    a3.metric("Folds per horizon", _number(summary["folds"].max()))

    left, right = st.columns([1.2, 1])
    with left:
        fig = px.bar(
            summary,
            x="horizon_bucket",
            y="fva_delta",
            color="fva_delta",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            category_orders={"horizon_bucket": HORIZON_ORDER},
            labels={"horizon_bucket": "Forecast horizon", "fva_delta": "Δ FVA: Steam − no Steam"},
            title="Incremental forecast value · positive is better",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(_polish(fig), width="stretch")
    with right:
        detail = selected.groupby("horizon_bucket", as_index=False).agg(
            steam_wape=("wape_with_steam", "mean"),
            baseline_wape=("wape_without_steam", "mean"),
            fva=("fva_delta_with_steam_minus_without", "mean"),
            fva_std=("fva_delta_with_steam_minus_without", "std"),
        )
        detail["horizon_bucket"] = pd.Categorical(detail["horizon_bucket"], HORIZON_ORDER, ordered=True)
        detail = detail.sort_values("horizon_bucket")
        for column in ["steam_wape", "baseline_wape", "fva", "fva_std"]:
            detail[column] = detail[column].map(_pct)
        detail.columns = ["Horizon", "Steam WAPE", "No-Steam WAPE", "Δ FVA", "FVA std"]
        st.dataframe(detail, hide_index=True, width="stretch")
    st.caption("The ablation compares the same rolling-origin folds. Results are based on synthetic demand and should not be read as causal impact.")

    fold = data["fold"]
    if not fold.empty:
        fold = fold[fold["feature_set"].isin(["gpu_signal", "all_external"]) & fold["model"].eq(selected_model)].copy()
        if not fold.empty:
            fig = px.box(
                fold,
                x="horizon_bucket",
                y="fva",
                color="feature_set",
                category_orders={"horizon_bucket": HORIZON_ORDER},
                labels={"horizon_bucket": "Forecast horizon", "fva": "FVA", "feature_set": "Feature set"},
                title="Fold-level FVA distribution",
            )
            fig.update_yaxes(tickformat=".1%")
            st.plotly_chart(_polish(fig), width="stretch")


def _shap_view(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Model explanation</div>', unsafe_allow_html=True)
    st.subheader("Which features drive LightGBM predictions?")
    shap_frame = data["shap"]
    if shap_frame.empty:
        st.info("No SHAP stability artifact is available.")
        return
    options = [x for x in ["all_external", "gpu_signal"] if x in shap_frame["feature_set"].unique()]
    if not options:
        st.info("No supported feature set is available in the SHAP artifact.")
        return
    feature_set = st.selectbox("Feature set", options=options, format_func=lambda x: FEATURE_SET_LABELS.get(x, x))
    top_n = st.slider("Number of features", min_value=5, max_value=20, value=12, step=1)
    view = shap_frame[shap_frame["feature_set"].eq(feature_set)].head(top_n).copy().sort_values("mean_abs_shap_share")
    fig = px.bar(
        view,
        x="mean_abs_shap_share",
        y="feature_name",
        orientation="h",
        error_x="std_abs_shap_share",
        labels={"mean_abs_shap_share": "Mean absolute SHAP share", "feature_name": "Feature"},
        title="Top model-attributed feature influence",
        hover_data={"top_10_fold_frequency": ":.0%", "folds": True},
    )
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(_polish(fig), width="stretch")
    detail = view.sort_values("mean_abs_shap_share", ascending=False).copy()
    detail["mean_abs_shap_share"] = detail["mean_abs_shap_share"].map(_pct)
    detail["std_abs_shap_share"] = detail["std_abs_shap_share"].map(_pct)
    detail["top_10_fold_frequency"] = detail["top_10_fold_frequency"].map(_pct)
    st.dataframe(detail, hide_index=True, width="stretch")
    st.caption("SHAP reports model-attributed influence across rolling cutoffs. It does not establish causality.")


def _quality_view(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Trust and coverage</div>', unsafe_allow_html=True)
    st.subheader("Can the source be trusted for this experiment?")
    validation = data["validation"]
    checks = data["checks"]
    summary = validation.get("summary", {})
    check_counts = checks["status"].astype(str).str.upper().value_counts().to_dict() if not checks.empty else {}
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Checks passed", _number(check_counts.get("PASS", 0)))
    q2.metric("Checks requiring review", _number(check_counts.get("REVIEW", 0)))
    q3.metric("Missing months", _number(len(summary.get("missing_months", []))))
    q4.metric("Unknown GPU mappings", _number(summary.get("unknown_gpu_mappings")))
    if not checks.empty:
        checks_display = checks.copy()
        checks_display.columns = ["Check", "Status", "Detail"]
        st.dataframe(checks_display, hide_index=True, width="stretch")
        st.download_button("Download validation checks", checks.to_csv(index=False), "steam_validation_checks.csv", "text/csv")

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("GPU mapping review")
        mapping = data["mapping"]
        if mapping.empty:
            st.info("No GPU mapping review rows are available.")
        else:
            st.dataframe(mapping.head(20), hide_index=True, width="stretch")
            st.caption(f"Showing 20 of {len(mapping):,} unresolved GPU names, ordered by latest observed share.")
    with right:
        st.subheader("Release-to-adoption diagnostics")
        release = data["release"]
        if release.empty:
            st.info("No release diagnostic is available.")
        else:
            st.dataframe(release, hide_index=True, width="stretch")
            if release["diagnostic_status"].astype(str).str.contains("NO_MATCHING", na=False).any():
                st.warning("The configured launch event is still a GENERIC example. Replace it with reviewed public release dates before interpreting adoption milestones.")


def _forecast_view(data: dict[str, Any]) -> None:
    st.markdown('<div class="section-kicker">Forward-looking output</div>', unsafe_allow_html=True)
    st.subheader("Forecast and uncertainty")
    forecast = data["forecast"]
    if forecast.empty:
        st.info("No forecast.parquet artifact is available.")
        return
    forecast = forecast.copy()
    forecast["week"] = pd.to_datetime(forecast["week"], errors="coerce")
    models = [str(x) for x in forecast["model_name"].dropna().unique()]
    default_model = "lightgbm" if "lightgbm" in models else models[0]
    selected = st.selectbox("Forecast model", models, index=models.index(default_model), format_func=lambda x: MODEL_LABELS.get(x, x))
    horizons = sorted(forecast.loc[forecast["model_name"].eq(selected), "horizon_weeks"].dropna().astype(int).unique())
    horizon = st.selectbox("Horizon", horizons, index=len(horizons) - 1 if horizons else 0)
    view = forecast[forecast["model_name"].eq(selected) & forecast["horizon_weeks"].eq(horizon)].sort_values("week")
    if not view.empty:
        fig = px.line(view, x="week", y="point_forecast", markers=True, labels={"week": "Week", "point_forecast": "Point forecast"}, title=f"{MODEL_LABELS.get(selected, selected)} · {horizon}-week horizon")
        if {"lower_bound", "upper_bound"}.issubset(view.columns):
            fig.add_scatter(x=view["week"], y=view["upper_bound"], mode="lines", line=dict(width=0), showlegend=False)
            fig.add_scatter(x=view["week"], y=view["lower_bound"], mode="lines", fill="tonexty", fillcolor="rgba(56,189,248,.16)", line=dict(width=0), name="Prediction interval")
        st.plotly_chart(_polish(fig), width="stretch")
        st.dataframe(view, hide_index=True, width="stretch")
    policy = data["policy"]
    if policy:
        st.info(
            f"Future Steam policy: **{policy.get('steam_future_strategy', '—')}** · "
            f"Last observed survey month: **{str(policy.get('last_observed_survey_month', '—'))[:10]}** · "
            f"Fabricated future rows: **{policy.get('future_rows_fabricated', '—')}**"
        )


def main() -> None:
    _style_page()
    data = _load_results()
    if not data["validation"] and data["comparison"].empty and data["forecast"].empty:
        st.title("DFSignal")
        _empty_state()
        return

    comparison = data["comparison"]
    available_models = list(comparison["model"].dropna().astype(str).unique()) if not comparison.empty else ["lightgbm"]
    default_model = "lightgbm" if "lightgbm" in available_models else available_models[0]
    feature_options = [x for x in FEATURE_SET_LABELS if comparison.empty or x in comparison["feature_set"].unique()]
    with st.sidebar:
        st.header("Dashboard controls")
        selected_model = st.selectbox("Model", available_models, index=available_models.index(default_model), format_func=lambda x: MODEL_LABELS.get(x, x))
        selected_horizons = st.multiselect("Forecast horizons", HORIZON_ORDER, default=HORIZON_ORDER)
        selected_feature_sets = st.multiselect("Feature sets", feature_options, default=feature_options, format_func=lambda x: FEATURE_SET_LABELS.get(x, x))
        st.divider()
        st.caption("Local, read-only artifact view. No company data, credentials, or causal claims are used.")
        if st.button("Refresh artifacts"):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        st.markdown("**How to use**")
        st.caption("Start with Overview. Use Steam vs no Steam for incremental value, then Data quality before trusting any model result.")

    tabs = st.tabs(["Overview", "Steam vs no Steam", "SHAP drivers", "Data quality", "Forecast"])
    with tabs[0]:
        _overview(data, selected_model, selected_horizons, selected_feature_sets)
    with tabs[1]:
        _ablation_view(data, selected_model)
    with tabs[2]:
        _shap_view(data)
    with tabs[3]:
        _quality_view(data)
    with tabs[4]:
        _forecast_view(data)


if __name__ == "__main__":
    main()
