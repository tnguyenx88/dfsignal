# Future Internal Data Integration

The current forecasting pipeline uses `SyntheticDemandSource` only. No real
internal demand data is available or executed in this preparation phase.
Synthetic rows remain labelled `SYNTHETIC` and stay in the existing synthetic
pipeline and output locations.

## Safe replacement path

```text
approved de-identified file
  → data/internal/
  → LocalCSVInternalDemandSource / LocalParquetInternalDemandSource
  → canonical internal schema
  → silver_internal_demand_weekly   (future execution boundary)
  → gold_internal_demand_features   (future execution boundary)
  → outputs/internal_pilot/         (future isolated outputs)
```

The adapter currently supports CSV and Parquet, configurable source column
mapping, target resolution, SHA-256 provenance, and PASS/REVIEW/FAIL
validation. It does not write internal Silver/Gold objects and does not run a
forecast experiment.

## Configuration

- `config/sources.yaml`: registers `internal_demand` as `local_internal` with
  `enabled: false`.
- `config/internal_data.yaml`: landing path, supported formats, canonical
  mappings, preferred target, alternatives, and grain keys.
- `config/models.yaml`: keeps the modeling target explicitly configured; it is
  not hardcoded in modeling logic.

## Dry run

```text
python -m dfsignal internal-check
```

With no file present, the command returns `MISSING_OPTIONAL` and does not fail
the existing pipeline. With an approved file, it reports validation checks and
provenance metadata without exposing data rows.

## De-identification boundary

Customer/person/order identifiers are neither required nor mapped. If present,
they produce a warning and are dropped from the canonical adapter frame. They
must not enter Silver, Gold, logs, reports, or fixtures.

See [`docs/internal_data_contract.md`](internal_data_contract.md) for required
columns, grain, target selection, null handling, duplicate handling, date
rules, data classification, and Git safety requirements.
