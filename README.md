# DFSignal

Personal proof of concept for an external demand signal and forecasting platform.
Current execution mode is `LOCAL_DEV`: Python, local filesystem, Parquet, DuckDB,
public datasets, and synthetic/sample demand only.

## Phase 1 vertical slice

The working example models `Memory / PC Components` using:

- configurable FRED macroeconomic series
- public Kaggle GPU specifications (`ellimaaac/gpus-specs-from-1986-to-2026`)
- manually maintained hardware launch events
- deterministic synthetic POS demand with trend, annual seasonality, holiday
  uplift, hardware-cycle influence, and seeded variation
- weekly normalization to Saturday
- Seasonal Naive, ETS, and LightGBM
- rolling-origin WAPE, MAE, MASE, RMSE, and Bias evaluation
- SHAP model-attributed feature influence
- local Streamlit output view

Synthetic results are not Corsair business findings and must not be interpreted as
causal analysis.

## Architecture

```text
Public sources + synthetic demand
              ↓
Bronze Parquet + metadata
              ↓
Silver normalized weekly tables
              ↓
Gold features / forecasts / evaluation / SHAP
              ↓
DuckDB query layer + Streamlit
```

`StorageBackend` and `LocalStorageBackend` isolate persistence. Core adapters,
features, models, backtesting, and explainability consume DataFrames and do not
embed absolute filesystem paths. Future Fabric storage can implement the same
backend contract.

## Repository structure

```text
config/                         # YAML registry, model, feature, and event config
data/
  bronze/                       # raw/source-native Parquet
  silver/                       # normalized Parquet
  gold/                         # model-ready Parquet
  internal/                     # ignored landing zone for approved internal files
  cache/ external/ synthetic/   # cache and source-specific local files
src/dfsignal/
  ingestion/                    # FRED, GPU, Steam, events, internal, synthetic
  transformation/               # weekly normalization
  features/                     # leakage-safe feature assembly
  forecasting/                  # Seasonal Naive, ETS, LightGBM
  evaluation/                   # rolling metrics
  explainability/               # SHAP driver output
  storage.py                    # backend abstraction and local implementation
  pipeline.py                   # local orchestration
  app.py                        # Streamlit view
notebooks/                      # exploration only
tests/                          # focused tests
docs/                           # design, migration, and integration contracts
outputs/                        # ignored synthetic/public dashboard copies
outputs/internal_pilot/         # ignored future internal-only outputs
```

`data/internal/` and `outputs/internal_pilot/` are fully gitignored. Do not
add placeholder files to either path.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

No credentials or `.env` values are required. Never commit credentials, API keys,
company data, or proprietary code.

## Commands

```bash
python -m dfsignal status
python -m dfsignal internal-check
python -m dfsignal run-all
python -m dfsignal steam-real --no-fetch-external
python -m pytest
python -m streamlit run src/dfsignal/app.py
```

Dashboard tabs:

- `Overview`: source status, coverage, WAPE comparison, and a plain-language decision readout
- `Steam vs no Steam`: ablation result, incremental value, and fold-level FVA stability
- `SHAP drivers`: top model-attributed features and cross-fold variation
- `Data quality`: validation checks, unresolved GPU mappings, and release diagnostics
- `Forecast`: forecast points, prediction intervals, and the observed-only Steam policy

If the dashboard has no data, run `python -m dfsignal steam-real --no-fetch-external`
first, then restart Streamlit.

`steam-real` is synchronous. The rolling backtest and SHAP analysis can take several
minutes; the terminal remains occupied while they run. Progress is printed by stage.
To keep the terminal free in PowerShell:

```powershell
Start-Process python -WorkingDirectory (Get-Location) `
  -ArgumentList @("-m", "dfsignal", "steam-real", "--no-fetch-external") `
  -RedirectStandardOutput outputs\steam-real.log `
  -RedirectStandardError outputs\steam-real-error.log
Get-Content outputs\steam-real.log -Wait
```

`steam-real` reads the reviewed local Steam Hardware Survey extract, validates
coverage/anomalies/mappings, computes descriptive launch/adoption diagnostics,
and compares `gpu_signal` against `all_external` using the existing rolling-origin
evaluation. It writes evidence under `outputs/validation/`. Steam future values
are never fabricated; the configured observed-only policy is recorded in
`steam_future_policy.json`.

`run-all` fetches enabled FRED series, downloads the public GPU archive only when
not already cached, writes Bronze/Silver/Gold Parquet plus metadata and
`data/dfsignal.duckdb`, then writes dashboard copies under `outputs/`.

## Configuration

- `config/sources.yaml`: source registry, mode, storage root, URLs, cadence,
  license notes, and enabled flags
- `config/fred.yaml`: arbitrary enabled FRED series (`series_id`, feature name,
  category, frequency)
- `config/models.yaml`: target column, horizons, grain, and model switches
- `config/features.yaml`: lag and rolling windows
- `config/product_signal_mapping.yaml`: business-family signal mapping
- `config/launch_events.csv`: manual public event records; example rows are
  retained but excluded from derived diagnostics
- `config/gpu_generations.yaml`: canonical GPU generation taxonomy and rules
- `config/gpu_mapping_overrides.yaml`: reviewed, auditable product corrections
- `config/gpu_signal.yaml`: market-segment scope, major-launch threshold, and
  transparent scoring weights

- `config/internal_data.yaml`: disabled internal source, supported formats,
  source-column mappings, grain keys, and target preferences

## Future internal-data workflow

Real internal data is not present and must not be synthesized for this PoC.
When an approved de-identified extract becomes available:

1. Obtain the approved extract and confirm its classification.
2. Copy the CSV or Parquet file into `data/internal/`.
3. Run `git status --short`; the file must not appear.
4. Configure source column mappings in `config/internal_data.yaml`.
5. Keep `internal_demand` disabled until the file and mappings are reviewed.
6. Run `python -m dfsignal internal-check`.
7. Stop at the validation report. Enabling the source and running an internal
   pilot require a future isolated runner and explicit approval; no such
   execution route exists in this preparation phase.

The internal adapter drops unmapped source-only columns, including identifiers,
and future internal objects must remain separate from synthetic tables and
outputs. See `docs/internal_data_contract.md` for the full contract.

## Adding sources and business families

Implement a source adapter under `src/dfsignal/ingestion/`, register metadata in
`config/sources.yaml`, write Bronze provenance, normalize to the weekly Silver
contract, and add a focused test. Add business families through configuration and
retain explicit target grain/signal mappings. Do not rewrite modeling or
backtesting code for a new storage backend or source adapter.

## Future compatibility

- `docs/fabric_migration.md` maps local Parquet/DuckDB/Python to future Lakehouse,
  Warehouse, Spark, Pipeline, and Key Vault seams.
- `docs/internal_data_integration.md` documents replacement of synthetic demand
  with local CSV/Parquet or future Fabric adapters.
- `docs/PHASE1_DESIGN.md` contains schema, feature, model, and backtest contracts.
