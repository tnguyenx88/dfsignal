# DFSignal Business Interpretation Guide

## Purpose

DFSignal produces technical forecasting and source-quality artifacts. This guide is the plain-English layer for people who need to understand the decision meaning without reading model code. Exact values remain in the technical artifacts; translations must not add claims.

## Read every result in four parts

1. **Fact** — what the source directly shows. Example: Steam GPU adoption increased.
2. **Model result** — what the backtest or model comparison measured. Example: adding Steam changed historical forecast error by a very small amount.
3. **Interpretation** — a cautious explanation of what that result may mean. Example: Steam may contain limited mid-term information in this test.
4. **Hypothesis** — a possible explanation that still needs testing. Example: gaming hardware adoption may precede some component demand changes.

Never turn an interpretation or hypothesis into a causal fact.

## Model descriptions

- **Seasonal Naive:** Assume demand follows a similar pattern to the previous seasonal period.
- **ETS:** Forecast based mainly on historical level, trend, and seasonality.
- **LightGBM:** Forecast using historical demand plus optional market signals.

For business users, the description should appear before the model name. Model names are implementation details, not conclusions.

## Metric translations and tooltip text

| Dashboard label | Technical term | Help text |
|---|---|---|
| Forecast Error (WAPE) | WAPE | Measures how far predictions were from actual demand overall. Lower is better. |
| Forecast Error (MAE) | MAE | Shows the average miss in demand units. |
| Forecast Tendency | Bias | Shows whether the forecast tends to predict too high or too low. |
| Value Added by External Signals | FVA | Measures whether adding a signal improved accuracy compared with a simpler baseline. |
| Forecast Uncertainty | Prediction Interval | A range around the forecast that represents expected uncertainty; it is not a guarantee. |
| Model-attributed influence | SHAP | Shows which inputs the model relied on most. It does not establish causality. |

## Explaining this Phase 1 experiment

### Steam Hardware Survey

Steam Hardware Survey is a monthly view of the hardware used by a large population of PC gamers. It can help estimate GPU generation adoption, memory adoption, and hardware upgrade cycles. Steam users are not the same as all Corsair customers, so Steam is a market signal, not direct Corsair demand.

### External signal families

- **GPU launches** — represents new GPU product releases and hardware upgrade cycles. It might matter because new GPU generations may encourage consumers to upgrade PCs and related components. Current evidence is mixed: GPU signals helped at longer horizons and hurt at shorter horizons in this synthetic experiment.
- **GPU adoption** — represents how common GPU generations are in the Steam survey. It might matter as a broad indicator of the installed hardware base. Current evidence is model-attributed and predictive only, not causal.
- **Steam adoption signals** — represents observed changes in surveyed PC-gamer hardware. It might contain mid-term market context. Current evidence was effectively neutral overall and varied by horizon.
- **Macro signals** — represents configured public economic indicators. The current run produced no macro Silver rows, so a clean observed macro comparison has not been proven.

## Phase 1 evidence in plain English

DFSignal currently demonstrates a repeatable pipeline that ingests public sources, creates synthetic weekly demand, builds features, compares forecasting methods over historical rehearsals, creates model-attribution outputs, and produces forecast ranges. It does not yet demonstrate that GPU or Steam signals improve real Corsair demand forecasts.

The current gate is **CONDITIONAL_GO**: the system is ready to continue testing, but important questions remain. Synthetic demand is the central limitation. It is useful for checking that the pipeline works, but it may not reproduce actual buying behavior, promotions, inventory constraints, channel mix, product availability, or customer segments.

The next important test is a small de-identified real internal demand extract with documented grain, source-as-of timestamp, and owner. It should compare history-only, calendar, GPU, and Steam feature sets using only information available at each historical prediction point.

## Confidence language

Use **strong evidence** only when the result is consistent across most historical test periods. Use **mixed evidence** when the signal helps in some periods but hurts in others. Use **limited evidence** when the data or performance is unstable. Use **not yet tested** when the pipeline supports a capability but the capability has not been validated.

## Data-quality language

- **PASS:** No material data issue was detected.
- **REVIEW:** The data can be used for testing, but some quality issues should be reviewed.
- **FAIL:** The issue is significant enough that results should not be trusted yet.
- **MISSING_OPTIONAL:** This optional dataset is not currently available, but the rest of the pipeline can still run.

## Technical references

- [`outputs/phase1_exit_gate/phase1_summary.md`](../outputs/phase1_exit_gate/phase1_summary.md) — exact technical narrative.
- [`outputs/phase1_exit_gate/phase1_gate.json`](../outputs/phase1_exit_gate/phase1_gate.json) — machine-readable gate and metrics.
- [`outputs/phase1_exit_gate/`](../outputs/phase1_exit_gate/) — detailed model, signal, fold, and quality artifacts.
