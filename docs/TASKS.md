# Phase 1 Local Implementation Checklist

## Design

- [x] Inspect current repository and deployment constraints.
- [x] Define local Bronze/Silver/Gold architecture.
- [x] Define `StorageBackend` compatibility boundary.
- [x] Define source registry and dataset contracts.
- [x] Document future Fabric mapping.
- [x] Document future internal data adapter replacement.

## Implementation

- [x] Add `LocalStorageBackend` with Parquet persistence and DuckDB views.
- [x] Add source metadata sidecars with hashes, date bounds, counts, duplicates,
  and missing-rate statistics.
- [x] Add configurable FRED ingestion with HTTP retries.
- [x] Add public Kaggle GPU archive download/cache and schema-normalizing adapter.
- [x] Add GPU product and weekly launch-signal derivation.
- [x] Add manual launch event ingestion.
- [x] Add `SyntheticDemandSource` with trend, seasonality, holiday, hardware-cycle,
  and seeded variation.
- [x] Add local CSV/Parquet internal-demand adapter boundary with configurable
  mappings, target selection, validation, provenance, and de-identification warnings.
- [x] Add weekly feature engineering with shifted target-derived windows.
- [x] Add Seasonal Naive, ETS, LightGBM, rolling backtest, and SHAP outputs.
- [x] Add local CLI: `status`, `ingest`, `transform`, `features`, `train`,
  `forecast`, `run-all`.
- [x] Add minimal Streamlit dashboard.

## Verification
- [x] `python -m pytest` — `26 passed`.
- [x] Import verification for storage, adapters, models, DuckDB, Parquet,
  Streamlit, and configured dependencies.
- [x] Full `python -m dfsignal run-all` succeeded.
- [x] FRED `UMCSENT` and `RRSFS` returned HTTP 200.
- [x] Kaggle GPU archive downloaded and normalized locally.
- [x] Bronze, Silver, and Gold Parquet artifacts generated.
- [x] DuckDB query-layer views verified.
- [x] `python -m dfsignal status` emitted structured startup output.
- [x] `python -m dfsignal internal-check` — `MISSING_OPTIONAL` with no landing file.
- [x] Internal CSV mapping, de-identification, empty-file, date, duplicate, and
  target validation checks passed.
- [x] `data/internal/` and `outputs/internal_pilot/` are gitignored; both
  internal source configurations remain disabled.
- [x] Streamlit started and rendered in a browser.

## Internal-data preparation boundary

- The internal source is disabled in both `config/sources.yaml` and
  `config/internal_data.yaml`.
- `python -m dfsignal internal-check` reports `MISSING_OPTIONAL` when no
  approved file is present.
- Approved files must be de-identified CSV or Parquet under `data/internal/`;
  internal pilot outputs belong under `outputs/internal_pilot/`.
- Internal landing and pilot output paths are gitignored and must never be
  force-added.
- See `docs/internal_data_contract.md` for the canonical schema, grain,
  target, date, null, duplicate, provenance, and safety contract.

## Known limitations

- The Steam adapter is implemented but disabled until a public historical
  CSV/Parquet extract is selected and placed under local external data.
- GPU dataset coverage is measured from the downloaded file; the Kaggle title is
  not treated as authoritative metadata.
- CLI stage names currently execute the bounded local vertical pipeline rather
  than exposing independent persisted stage orchestration.
- Forecasts are a Phase 1 holdout demonstration without prediction intervals or
  learned decay models.
- SHAP values are model-attributed influence, not causal inference.
