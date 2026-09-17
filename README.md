# DFSignal

## Explainable AI demand forecasting for manufacturing and commerce

DFSignal is a reproducible proof of concept for helping manufacturing and
commercial businesses anticipate demand, compare forecasting strategies, and
turn model output into planning decisions.

The project combines historical demand with external signals such as
macroeconomic indicators, product events, and public market or hardware data.
It is designed for environments where demand is volatile, planning decisions
must be made before the next sales cycle, and users need to understand why a
forecast moved.

> DFSignal is an AI application competition prototype. It demonstrates an
> end-to-end decision-support workflow; it is not a production planning system
> and does not contain confidential enterprise data.

## Why this matters

Demand planning teams commonly face three connected problems:

1. **Forecast uncertainty** — demand changes with seasonality, holidays,
   economic conditions, product launches, and market adoption.
2. **Signal fragmentation** — useful leading indicators live in different
   sources, cadences, formats, and levels of quality.
3. **Limited explainability** — a forecast is difficult to trust when users
   cannot see the evidence, validation results, and drivers behind it.

DFSignal addresses these problems with a transparent pipeline that benchmarks
simple and machine-learning models, validates them using realistic
rolling-origin backtests, estimates prediction intervals, and exposes
model-attributed drivers through a local dashboard.

## What the prototype demonstrates

- Configurable ingestion of public sources and clearly labelled synthetic
  demand
- Provenance metadata at the source and pipeline boundaries
- Weekly normalization across sources with different cadences
- Leakage-safe lag, rolling, event, and market-signal features
- Benchmarking across Seasonal Naive, ETS, and LightGBM
- Rolling-origin evaluation using WAPE, MAE, MASE, RMSE, and Bias
- Prediction intervals based on historical forecast residuals
- SHAP-based model attribution for feature-level explanations
- A Streamlit dashboard for business-facing review of coverage, accuracy,
  signal value, data quality, drivers, and future forecasts
- An architecture that keeps storage, ingestion, modeling, evaluation, and
  presentation boundaries explicit

## Phase 1 vertical slice

The current working example represents a generic **Memory / PC Components**
business family. It uses:

- configurable FRED macroeconomic series
- public GPU specifications
- manually maintained public hardware launch events
- a reviewed local extract of public Steam Hardware Survey observations
- deterministic synthetic weekly POS demand with trend, annual seasonality,
  holiday uplift, hardware-cycle influence, and seeded variation
- weekly normalization to Saturday
- Seasonal Naive, ETS, and LightGBM forecasting
- rolling-origin backtesting and prediction intervals
- SHAP driver analysis
- a local Streamlit decision-support view

All demand observations in the current example are synthetic or public. Results
are suitable for demonstrating methodology and reproducibility only. They are
not findings about any specific business and must not be interpreted as causal
analysis.

## AI application value

DFSignal is more than a single model:

- **Signal engineering:** external indicators are aligned to the demand
  planning grain and transformed into features available at forecast time.
- **Model selection:** machine-learning output is compared with strong,
  interpretable baselines instead of being accepted automatically.
- **Decision transparency:** backtest metrics, interval coverage, data-quality
  checks, and SHAP drivers are presented together.
- **Operational readiness:** ingestion, storage, orchestration, metadata, and
  dashboard layers are separated so the prototype can evolve without
  rewriting the forecasting logic.

## Architecture

```text
Public sources + synthetic demand
              ↓
Bronze source-native data + provenance metadata
              ↓
Silver normalized weekly tables
              ↓
Gold features / forecasts / evaluation / SHAP
              ↓
DuckDB query layer + Streamlit decision dashboard
```

`StorageBackend` and `LocalStorageBackend` isolate persistence. Core adapters,
feature builders, models, backtesting, and explainability consume DataFrames
and do not embed absolute filesystem paths. A future warehouse or lakehouse
adapter can implement the same backend contract.

## Repository structure

```text
config/                         # source, model, feature, and event configuration
data/
  bronze/                       # raw/source-native Parquet
  silver/                       # normalized Parquet
  gold/                         # model-ready Parquet
  internal/                     # ignored landing zone for approved extracts
  cache/ external/ synthetic/   # local cache and source-specific files
src/dfsignal/
  ingestion/                    # public, synthetic, event, and optional internal adapters
  transformation/               # weekly normalization
  features/                     # leakage-safe feature assembly
  forecasting/                  # Seasonal Naive, ETS, and LightGBM
  evaluation/                   # rolling metrics, intervals, and signal analysis
  explainability/               # SHAP driver output
  storage.py                    # backend abstraction and local implementation
  pipeline.py                   # local orchestration
  app.py                        # Streamlit decision dashboard
notebooks/                      # exploration only
tests/                          # focused behavioral tests
docs/                           # design, validation, migration, and data contracts
outputs/                        # ignored local dashboard and model artifacts
```

`data/internal/` and `outputs/internal_pilot/` are fully gitignored. Do not add
placeholder files to either path.

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No credentials or populated `.env` values are required for the current
prototype. Never commit credentials, API keys, confidential company data, or
proprietary code.

## Run the project

```bash
python -m dfsignal status
python -m dfsignal internal-check
python -m dfsignal run-all
python -m dfsignal steam-real --no-fetch-external
python -m pytest
python -m streamlit run src/dfsignal/app.py
```

If the dashboard has no data, run the `steam-real` command first and then
restart Streamlit.

`steam-real` is synchronous. Rolling backtesting and SHAP analysis can take
several minutes, with progress printed by stage. To keep PowerShell available:

```powershell
Start-Process python -WorkingDirectory (Get-Location) `
  -ArgumentList @("-m", "dfsignal", "steam-real", "--no-fetch-external") `
  -RedirectStandardOutput outputs\steam-real.log `
  -RedirectStandardError outputs\steam-real-error.log
Get-Content outputs\steam-real.log -Wait
```

## Dashboard

The dashboard is organized around questions a planning user needs to answer:

- **Overview:** What is the current source status, coverage, and model
  comparison?
- **Steam vs no Steam:** Does the observed public hardware signal add
  incremental value, and is that value stable across folds?
- **SHAP drivers:** Which features are associated with the model's forecast?
- **Data quality:** Are there validation failures, unresolved mappings, or
  release diagnostics to investigate?
- **Forecast:** What are the future points, prediction intervals, and observed
  versus unavailable external signals?

The dashboard is read-only. It presents model evidence and does not claim that
feature attribution proves causation.

## Data and reproducibility principles

- Public datasets and synthetic demand are used for the current demonstration.
- Synthetic demand is deterministic and seeded so runs are reproducible.
- Source metadata records provenance, refresh state, cadence, and coverage.
- Future values for the observed-only hardware signal are never fabricated.
- External signals are evaluated by ablation rather than assumed to help.
- Validation is performed with rolling-origin splits to reduce temporal leakage.
- Local generated data, model artifacts, caches, and databases remain ignored
  by Git.
- Optional internal extracts must be approved, de-identified, mapped, and
  validated before they can be enabled.

## Configuration

- `config/sources.yaml`: source registry, mode, storage root, URLs, cadence,
  license notes, and enabled flags
- `config/fred.yaml`: enabled FRED series, feature names, categories, and
  frequencies
- `config/models.yaml`: target column, horizons, grain, and model switches
- `config/features.yaml`: lag and rolling windows
- `config/product_signal_mapping.yaml`: business-family signal mapping
- `config/launch_events.csv`: manually maintained public event records
- `config/gpu_generations.yaml`: canonical GPU generation taxonomy and rules
- `config/gpu_mapping_overrides.yaml`: reviewed and auditable product
  corrections
- `config/gpu_signal.yaml`: market-segment scope, launch threshold, and
  transparent scoring weights
- `config/internal_data.yaml`: disabled optional source, supported formats,
  source-column mappings, grain keys, and target preferences

## Adding a source or business family

Implement a source adapter under `src/dfsignal/ingestion/`, register its
metadata in `config/sources.yaml`, write Bronze provenance, normalize to the
weekly Silver contract, and add a focused behavioral test. Add business
families through configuration with explicit target grain and signal mappings.
The modeling and backtesting layers should not need to be rewritten for a new
source or storage backend.

## Optional enterprise-data workflow

The repository does not contain enterprise demand data. When an approved
de-identified extract becomes available:

1. Confirm its classification and approved use.
2. Place the CSV or Parquet file under `data/internal/`.
3. Confirm `git status --short` does not show the file.
4. Configure source-column mappings in `config/internal_data.yaml`.
5. Keep `internal_demand` disabled until the file and mappings are reviewed.
6. Run `python -m dfsignal internal-check`.
7. Stop at the validation report until an isolated pilot is explicitly
   approved.

The internal adapter drops unmapped source-only columns, including identifiers,
and keeps optional enterprise objects separate from synthetic tables and
outputs. See `docs/internal_data_contract.md` for the full contract.

## Documentation

- `docs/ARCHITECTURE.md`: system boundaries and data flow
- `docs/PHASE1_DESIGN.md`: schema, feature, model, and backtest contracts
- `docs/forecasting_validation.md`: validation approach and interpretation
- `docs/gpu_signal_validation.md`: public hardware signal validation
- `docs/internal_data_contract.md`: safeguards for optional enterprise extracts
- `docs/fabric_migration.md`: future warehouse/lakehouse migration seams

## Future direction

The next steps are to validate the approach on additional public signals,
expand beyond the initial business family, compare forecast value across
planning horizons, and—only with proper approval—evaluate de-identified
enterprise demand data. The core principle remains unchanged: forecast early,
measure honestly, explain clearly, and protect sensitive data.
