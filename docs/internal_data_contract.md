# Internal Demand Data Contract

## Purpose and boundary

This contract prepares DFSignal for a future approved, de-identified internal
extract. It does not authorize access to internal data and does not run an
internal forecasting experiment. The source is disabled by default in
`config/sources.yaml` and `config/internal_data.yaml`.

## Supported formats and landing path

- Supported formats: **CSV** and **Parquet**.
- Landing path: `data/internal/` (configured pattern: `data/internal/*`).
- Pilot outputs: `outputs/internal_pilot/`.
- Both paths are fully gitignored. Do not add a `.gitkeep`, sample file, or
  confidential-looking placeholder.

## Canonical schema

The adapter maps source-specific names to these canonical names and drops
unmapped source-only columns before any Silver/Gold handoff.

| Canonical column | Role | Required initially |
|---|---|---:|
| `week` | weekly business date | Yes |
| `business_segment` | business dimension | No |
| `product_group` | product grouping | Yes |
| `product_family` | product dimension | No |
| `region` | geography dimension | No |
| `channel` | route-to-market dimension | No |
| `pos_qty` | POS quantity | One configured target |
| `pos_revenue` | POS revenue | No |
| `shipment_qty` | shipment quantity | One configured target alternative |
| `shipment_revenue` | shipment revenue | No |
| `booking_qty` | booking quantity | One configured target alternative |
| `booking_revenue` | booking revenue | No |
| `asp` | average selling price | No |
| `inventory_qty` | inventory quantity | No |

Only `week`, `product_group`, and one configured demand target are mandatory
for the initial validation boundary. Optional columns must not be required for
file acceptance.

## Preferred grain and keys

Preferred initial grain:

```text
WEEK × BUSINESS_SEGMENT × PRODUCT_GROUP
```

`REGION` and `CHANNEL` are optional dimensions. If present and configured,
they participate in the duplicate-key check. The minimum validation key is:

```text
WEEK × PRODUCT_GROUP
```

The source must not contain multiple rows for the configured grain. Duplicate
rows are reported as `FAIL`; the adapter does not silently aggregate them.

## Column mapping and target selection

Source column names are configured in `config/internal_data.yaml`:

- `week_column` or `date_column` maps to canonical `week`.
- `dimensions` maps dimension fields.
- `measures` maps numeric measures.
- `target.preferred` defaults to `pos_qty`.
- `target.alternatives` permits `shipment_qty` and `booking_qty`.

Change mappings and target selection in YAML when an approved extract uses
different names. Do not change Python modeling logic for source naming
variations. The current model target remains explicitly configured in
`config/models.yaml`; no internal source is enabled by that setting.

## Date requirements

- Dates must parse successfully.
- Canonical `week` values should be completed Saturday weeks.
- The validator reports non-weekly dates and a latest partial week as `REVIEW`.
- Missing weeks between the minimum and maximum date are `REVIEW`.
- Source timezone and source-as-of semantics must be documented before the
  internal pilot runs.

## Target and null handling

- Default preferred target: `POS_QTY` (`pos_qty` canonically).
- Allowed alternatives: `SHIPMENT_QTY` and `BOOKING_QTY`.
- The selected target must be numeric, non-null, and non-negative.
- Nulls in required fields are `FAIL`.
- Nulls in optional fields are retained as `REVIEW` diagnostics.
- Negative optional measures are reported as `REVIEW`; a negative selected
  target is `FAIL` unless an explicitly approved signed-adjustment contract is
  added later.

## Schema drift and product groups

- Unmapped source columns are reported as `REVIEW`.
- Product-group vocabulary is optional. If configured, unexpected values are
  reported as `REVIEW` rather than silently recoded.
- Canonical output contains only the approved canonical schema; source-only
  columns are not copied into Gold features.

## Partial weeks and duplicates

The initial validator does not repair partial weeks or duplicates. It reports
both conditions so the data owner can correct the extract or approve an
explicit transformation rule. No rows are automatically dropped or summed.

## Data classification and de-identification

The expected classification is `CONFIDENTIAL_INTERNAL_DE_IDENTIFIED`.
The initial pilot does not require and must not use:

- customer name or customer ID
- email or address
- employee information
- order number
- serial number
- transaction ID

If such columns arrive, validation emits a warning. They are not mapped by
default and are dropped from the canonical adapter output. Do not place them in
Silver/Gold features, logs, reports, or test fixtures.

## Provenance metadata

When a file is available, the adapter prepares:

- `source_id`
- `source_name`
- `filename`
- `file_hash` (SHA-256)
- `extract_created_at`
- `source_as_of_timestamp`
- `load_timestamp`
- `min_business_date`
- `max_business_date`
- `row_count`
- `grain`
- `data_classification`

An individual person's name is not required as an owner field. Source ownership
and approval should be managed outside the row-level data contract.

## Logical object separation

Future internal objects are intentionally separate from the existing synthetic
objects:

```text
silver_internal_demand_weekly
  → gold_internal_demand_features
  → outputs/internal_pilot/
```

They must not overwrite `internal_demand_weekly`, `demand_features`, synthetic
backtests, or current public/synthetic outputs. This phase only prepares the
boundary; it does not create these objects or run an experiment.

## Git safety requirements

Before and after placing an approved file:

1. Confirm the file is under `data/internal/`.
2. Run `git status --short` and confirm the file is absent.
3. Keep internal outputs under `outputs/internal_pilot/`.
4. Never force-add files from either ignored path.
5. Never commit credentials, identifiers, raw extracts, or proprietary code.
