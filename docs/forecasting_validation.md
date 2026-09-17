# Forecasting and validation contract

DFSignal runs in `LOCAL_DEV` only. The pipeline keeps source-native data in Bronze,
weekly canonical tables in Silver, and model-ready features and outputs in Gold.
Every stage remains independently callable through the storage backend.

## Validation

`validate` writes the regular contract checks to the cache table
`validation_report`. When GPU rows are available it also calls
`generate_gpu_validation_report` and writes a human-reviewable package below
`outputs/validation/`:

- `gpu_validation_report.json` — status, observed date coverage, diagnostics, and
  sample count;
- `gpu_validation_checks.parquet` and metadata sidecar;
- `gpu_weekly_diagnostics.parquet` and metadata sidecar;
- `gpu_validation_sample.parquet` and metadata sidecar;
- `gpu_generation_conflicts.parquet` and metadata sidecar when applicable.

Coverage is based only on parseable observed release dates. A missing optional
Steam extract is recorded as unavailable and does not fail validation.

## Feature provenance and future availability

Silver tables carry `source_id`, `provenance_source_id`,
`provenance_table`, `provenance_layer`, and `source_as_of` where a weekly date is
available. Gold features carry source IDs and the feature table provenance.
External features are merged only after they have been normalized to weekly
observations.

The configured feature sets are nested:

1. `history_only` — shifted lags, rolling windows, WoW, and YoY;
2. `calendar` — history plus known calendar fields;
3. `macro` — calendar plus configured FRED fields;
4. `gpu_signal` — macro plus GPU launch, event, and optional Steam fields.

Future external values use the YAML `observed_only` strategy. The forecast
builder consults rows no later than the forecast origin and carries the latest
observed numeric value forward; it never reads holdout labels or future source
rows. Calendar values are generated from the future week. LightGBM target lags
and rolling features are generated recursively from prior predictions.

## Backtest and FVA

`backtest` uses expanding rolling origins, minimum training history, all
configured 4/8/13/26-week horizons, and no random split. It persists:

- `model_performance` — one row per cutoff, horizon, feature set, and model,
  including WAPE, MASE, RMSE, signed Bias, horizon bucket, and FVA against the
  same-cutoff Seasonal Naive baseline;
- `backtest_residuals` — point-level actual, prediction, and residual rows used
  for interval calibration;
- `feature_value_add` — horizon-bucket FVA summaries.

FVA is `baseline WAPE - model WAPE`; positive values indicate lower error than
Seasonal Naive. It is a predictive comparison, not a causal claim.

## Training and forecasting

`train` persists `model_training` rows with model-run IDs, training bounds,
parameters, convergence, warnings, fallback use, and split sizes. ETS warnings
and non-convergence are retained rather than hidden.

`forecast` creates a new `forecast_run_id`. Each output includes both the
canonical fields (`forecast_run_id`, `origin`, `horizon_week`, `model`,
`target`, `value`, `lower_bound`, `upper_bound`) and backward-compatible aliases.
The origin is the latest observed demand week, and every horizon week is a
strictly later weekly date. Residual/conformal bounds are populated for enabled
models when backtest residuals are available; calibration coverage and width are
persisted with each row. The output contains no holdout-labelled future rows.
