# GPU signal validation

This note defines the validation contract for the public Kaggle GPU catalog used
by DFSignal. The source is a static public catalog, not an authoritative launch
registry. A successful download or parse is therefore not evidence that every
GPU, launch date, vendor, or product segment is present.

## Inputs and outputs

`dfsignal.ingestion.gpu.normalize_gpu_specs(raw)` accepts the source-native
Kaggle DataFrame (or a small DataFrame with the same logical fields). It keeps
all source rows and returns a canonical product table. The existing contract
columns remain available (`gpu_id`, `vendor`, `product_name`, `release_date`,
`architecture`, `memory_size`, `memory_type`, `memory_bus`, `tdp_w`,
`process_nm`, `market_segment`, `generation`, and `source`). Additional fields
preserve raw values, classification reasons/statuses, duplicate keys, and
review flags.

`memory_size` and `memory_size_mb` are canonical MB. `tdp_w` is canonical watts.
Raw values (`memory_size_raw`, `tdp_raw`, and all release-date fields) remain in
the output. `is_high_power`/`high_power_flag` and
`is_high_vram`/`high_vram_flag` are nullable booleans; missing inputs are
`<NA>`, never false.

`dfsignal.ingestion.gpu.derive_gpu_weekly_signals(products)` uses valid release
dates and non-empty product names only. It counts one unique launch per
case-insensitive `vendor + product_name + release_date` key. Duplicate source
rows are retained in the product table and exposed through
`gpu_record_count` and `gpu_duplicate_record_count`, rather than silently
inflating launch counts.

`dfsignal.validation.gpu.generate_gpu_validation_report(products=..., raw=...,
weekly_signals=..., output_dir=...)` returns dictionaries and DataFrames and
has no filesystem requirement. Supplying `output_dir` calls
`persist_gpu_validation_report` and writes JSON, Parquet review tables, and
Parquet metadata sidecars beneath that explicitly supplied directory (for
example `outputs/validation/`). The orchestrator can pass its in-memory raw,
normalized, and weekly frames without changing the validation logic.

## Release-date handling and coverage

The observed Kaggle schema has three date columns:

1. `Graphics Card__Release Date`
2. `Mobile Graphics__Release Date`
3. `Integrated Graphics__Release Date`

Dates are coalesced **per row** in that order, with a generic `release_date`
column used last when present. The selected value and source are recorded as
`release_date_raw` and `release_date_source`; all three source-native raw
fields are also retained. Ordinal suffixes (`1st`, `2nd`, `3rd`, `4th`) are
removed only for parsing. `release_date_precision` records `day`, `month`,
`year`, `unknown`, or `missing`. A year-only value parses to January 1 and a
month-only value to the first day of that month; this imputation is not a claim
of day-level launch precision.

`calculate_gpu_coverage` reports bounds and ratios from parseable observed
`release_date` values only:

- `valid_date_rows / total_rows` is the observed valid-date ratio;
- `observed_start` and `observed_end` are the minimum and maximum valid dates;
- `observed_span_days`/`observed_span_weeks` describe the span between those
  bounds;
- precision counts expose year/month imputation;
- optional expected bounds are compared separately and never used to extend the
  observed bounds.

No date is invented for a row with a missing/invalid source value. Such rows
remain in the normalized table with `release_date_status=REVIEW` and are
excluded explicitly from weekly aggregation. An empty frame or a frame with no
valid dates is `FAIL`; a partially dated frame is `REVIEW`.

## Classification heuristics

Every classification has a value, a status, and a reason. `PASS` means a direct
recognized source field or an unambiguous cue was used. `REVIEW` means the
value is inferred, missing, ambiguous, or only supported by a numeric proxy.
`FAIL` is reserved for missing critical identity or structural input. The
row-level `classification_status` is the worst component status, and
`classification_reasons` identifies the component needing review.

### Vendor

`Brand` (then `vendor`/`manufacturer`) is matched case-insensitively to known
brands: NVIDIA, AMD, ATI, Intel, Matrox, 3dfx, XGI, SiS, Sony, Microsoft, S3,
and VIA. `vendor` preserves ATI as a historical label; `vendor_family` rolls ATI
into AMD for family-level weekly counts. An unmapped or missing brand becomes
`Unknown` with `REVIEW`; the original value is in `vendor_raw`.

### Generation

A non-sentinel generation is selected from the row's
`Graphics Card__Generation`, `Mobile Graphics__Generation`,
`Integrated Graphics__Generation`, or generic `generation` field, in that
order. The source and raw fields are retained. If multiple source generation
fields contain distinct values, the first deterministic value is retained but
`generation_conflict_flag` and `generation_status=REVIEW` are set and all
candidates are listed.

When no direct generation is present, architecture/name cues are checked for
Blackwell, Hopper, Ada Lovelace, Ampere, Turing, Volta, Pascal, Maxwell,
Kepler, Fermi, Tesla, RDNA, Vega, Navi, Polaris, GCN, TeraScale, Arc, or Xe.
NVIDIA/AMD series tokens are a secondary fallback. A product name fallback is
also `REVIEW`, not a confirmed generation. If no cue exists, generation is
`Unknown`/`REVIEW`.

### Market segment

A direct `market_segment`/`Market Segment` value wins when it is one of
consumer/desktop/discrete, mobile/laptop/notebook, workstation/professional,
server/datacenter, console, embedded, or integrated/IGP. It is normalized to
`consumer`, `mobile`, `workstation`, `datacenter`, `console`, `embedded`, or
`integrated`.

Absent a direct field, name/vendor/generation/architecture/memory cues are
checked in this order: console (PlayStation/Xbox/Sony/Microsoft), datacenter
(server/Tesla/GRID/Instinct and selected accelerator tokens), workstation
(Quadro/FirePro/FireGL/Radeon Pro/RTX Pro/Pro W), embedded, integrated (IGP,
UHD, Iris, HD Graphics, GMA, System Shared), mobile (Mobile/Mobility/Laptop/
Notebook/Max-Q/Go), then consumer (GeForce/Radeon/RX/GTX/Arc). Multiple cues
produce a deterministic priority selection but always receive `REVIEW` and
list `market_segment_candidates`. No cue gives `Unknown`/`REVIEW`.

### Performance tier

A direct `performance_tier` is normalized to `enthusiast`, `high`, `mid`,
`entry`, or `integrated`. Otherwise model-family cues are used: Titan/XTX and
RTX 4090/5090/3090 or RX 7900/6950 families are enthusiast; recognized 80/70
and RX 7800/6800/7700/6700 families are high; recognized 70/60 and RX
7600/6600/5600 families are mid; recognized lower 60/50/30 and RX 6500/6400/
5500/5400/5300 families are entry. Integrated cues produce integrated.

TDP >= 300 W or VRAM >= 16,384 MB can provide a high numeric proxy, and TDP >=
200 W or VRAM >= 8,192 MB can provide a mid numeric proxy, but a numeric-only
result is `REVIEW`. A product with no direct/model/numeric cue is
`Unknown`/`REVIEW`. These are relative signal tiers, not benchmark scores.

### Launch importance

This is an explicit prioritization heuristic, not a claim about commercial
impact. Enthusiast is `high` (score 3), high is `high` (score 2), mid is
`medium` (score 1), and entry/integrated is `low` (score 0). Values depending
on a heuristic generation or a numeric-only tier are `REVIEW`; an unknown tier
is `Unknown`/`REVIEW`. No sales, price, or causal effect is inferred.

### High-power and high-VRAM flags

`high_power_flag=True` when parsed TDP is at least 200 W; `high_vram_flag=True`
when parsed memory is at least 8,192 MB (8 GiB). Explicit units are converted
from KB/MB/GB/TB (and KiB/MiB/GiB/TiB); an `xN` memory suffix is multiplied.
Unitless numbers are treated as MB or W respectively but marked `REVIEW`.
Sentinels such as `unknown` and `System Shared` become missing and the flag is
`<NA>`/`REVIEW`, not false. The thresholds are operational review thresholds,
not hardware standards.

## Weekly sanity diagnostics

`diagnose_gpu_weekly_signals` returns a summary dictionary plus reviewable
`checks`, `weekly_diagnostics`, and `generation_conflict_rows` DataFrames.
Checks cover:

- **launch counts:** valid rows, unique launch keys, weekly row count, and
  reconciliation of weekly unique-launch sum to the product key count;
- **vendor counts:** deterministic product counts by normalized vendor, with
  unknown vendors explicitly reported;
- **TDP/VRAM ranges:** valid/missing counts, non-positive counts, min/P05/
  median/P95/max, and impossible-value envelopes (TDP 0–1000 W; VRAM 0–1 TiB);
- **duplicates:** exact source duplicates and extra rows in repeated
  vendor/product/date keys;
- **spikes:** weekly counts above `max(3, Q3 + 1.5*IQR)`, returned with the
  threshold and week; spikes are `REVIEW`, never removed;
- **generation conflicts:** distinct non-sentinel generation labels on the
  same row, returned with source values for manual review;
- **classification uncertainty:** counts of vendor, generation, segment, and
  performance rows still requiring review.

`FAIL` means a structural check or launch-count reconciliation failed.
`REVIEW` means a human should inspect an unusual but parseable condition. A
`PASS` report means these checks found no flagged condition; it does not certify
source completeness or causal validity.

## Deterministic manual sample

`build_gpu_validation_sample` returns 24 rows by default (clamped to 20–30 when
at least 20 products are available; otherwise it returns all available rows).
It sorts by stable vendor/year/segment/product/date/source-row keys and greedily
selects rows that add previously unseen vendor, year, and segment values before
filling the remaining slots. This is deterministic and has no random seed.
Unknown years/segments remain explicit strata. The sample includes raw values,
classification candidates/reasons/statuses, duplicate flags, and high-power/
high-VRAM flags. Any sampled uncertainty makes the sample check `REVIEW`; an
empty sample is `FAIL`.

## Real GPU launch events

`normalize_gpu_specs` preserves source-native release fields and adds a stable
`source_record_id`, canonical `memory_size_gb`, `memory_bus_bits`,
`market_segment_category`, and `event_status`. Dates are never created for rows
without an observed source date; year/month precision remains `REVIEW`.

`derive_gpu_launch_events` applies `config/gpu_signal.yaml` after classification.
The default experiment scope is `DESKTOP_CONSUMER`; mobile, workstation,
datacenter, integrated, console, legacy, and unknown rows remain in the product
table but do not become modeling events unless configuration opts them in.
Each event contains `event_id`, `event_date`, `week`, vendor/product identity,
generation, architecture, market segment, performance tier, memory, TDP,
`GPU_LAUNCH`, status, `source_id`, and `source_record_id`.

Generation taxonomy is configured in `config/gpu_generations.yaml`. Reviewed
corrections belong in `config/gpu_mapping_overrides.yaml` and are applied after
automatic rules with an auditable reason. Unmapped values remain `UNKNOWN` and
`REVIEW`; they are not silently assigned to a generation.

Manual rows in `config/launch_events.csv` remain supported. Placeholder rows are
retained as Bronze data with `is_example_event=true,event_status=EXAMPLE`, but
are excluded from event features and Steam release/adoption diagnostics.

`gpu_validation_report.json` records observed product and generated-event coverage
separately; `REVIEW` reflects date precision, unknown mappings, or configured
scope uncertainty and is not a completeness claim. Parquet review tables
serialize nested check observations as JSON strings so DuckDB/Arrow readers and
LightGBM receive compatible scalar dtypes at downstream boundaries.
