# Steam Hardware source boundary

The Steam Hardware Survey source is an **optional local input**. DFSignal does not
scrape or fetch Valve's HTML survey page. A user-selected, community-maintained
extract must be placed under `data/external/steam_hardware/` (or supplied as an
explicit path) before enabling the source.

## Enable or disable

The pipeline registry entry in `config/sources.yaml` and the companion
`config/steam_hardware.yaml` both default to `enabled: false`. A disabled or
missing optional extract records source status without blocking the local
pipeline:

```python
from dfsignal.ingestion.steam import SteamHardwareSource

steam = SteamHardwareSource(enabled=True)  # scans the relative default directory
frame = steam.read()
```

For an end-to-end run, set `enabled: true` and optionally provide `path` in the
`steam_hardware` source entry. Use an explicit path when more than one extract is
present:

```python
from dfsignal.ingestion.steam import read_steam_hardware

frame = read_steam_hardware(
    "data/external/steam_hardware/steam_hardware_2025.parquet",
    source_id="STEAM_HARDWARE_COMMUNITY_2025",
    source_url="https://example.invalid/source-notes",
)
```

`SteamHardwareSource(enabled=False).read()` and
`read_steam_hardware(enabled=False)` return the legacy empty contract. Calling
the enabled adapter without a file fails with an actionable `FileNotFoundError`;
an enabled directory must contain exactly one supported extract.

## Accepted format and schema

Exactly one `.csv` or `.parquet` file is accepted. Long-form extracts require:

| Column | Type/meaning |
| --- | --- |
| `observation_date` or `survey_month` | Parseable calendar date. |
| `metric_name` | Non-empty metric identifier, for example `ram_32gb_plus_share`. |
| `metric_value` | Finite numeric percentage in the inclusive range 0-100. |
| `source_id` or `source_url` | At least one non-empty provenance value for every row. |

Common wide exports are also accepted when their date column is accompanied by
numeric metric columns. The adapter normalizes both forms to a monthly long
contract with `survey_month`, `category`, `metric_name`, `metric_value`, `unit`,
`vendor`, `generation`, and source provenance. Legacy aliases including
`observation_date` and `metric_unit` remain available to existing callers.

The uniqueness key is `(survey_month, category, metric_name)`. Duplicate keys fail
loudly rather than being silently dropped. This prevents two observations for a
metric and date from being mistaken for a single value.

## Coverage, units, and provenance

The adapter validates dates but does not invent historical coverage or coerce an
extract into the configured 2016 start date. Coverage is the observed minimum and
maximum date. The configured source registry describes the expected cadence as
monthly; callers must inspect actual extract coverage before modeling.

Metric units are source-defined. Keep a `metric_unit` column in the extract when
it is supplied, or pass a mapping to `build_steam_source_metadata`:

```python
from dfsignal.ingestion.steam import build_steam_source_metadata

metadata = build_steam_source_metadata(
    frame,
    source_id="STEAM_HARDWARE_COMMUNITY_2025",
    source_path="data/external/steam_hardware/steam.csv",
    metric_units={"ram_32gb_plus_share": "percent_of_survey_respondents"},
)
metadata.as_dict()
```

The metadata contract records the source identifier, optional source URL and
format, observed coverage, declared metric units, and whether the source was
enabled. The pipeline persists row-level source metadata in the cache layer.

## Weekly signals and future policy

`src/dfsignal/transformation/steam.py` performs the optional signal transformation:

1. Monthly observations are aligned to Saturday week-ending dates.
2. Each latest observed monthly value is carried forward only through the next
   observed month; no values are generated beyond the extract's last observation.
3. GPU generation mappings are configuration-driven by `config/steam_hardware.yaml`.
4. Supported signals include vendor share, generation share, high-memory share,
   high-core CPU share, one-/three-month adoption changes, and descriptive launch
   interactions.
5. `future_strategy: OBSERVED_ONLY` is the default policy: Steam-derived future
   feature rows are not fabricated for forecasting. Missing Steam values remain
   missing until the existing sparse external-feature handling applies.

Launch/adoption outputs are descriptive diagnostics only. They do not establish
causal impact and should not be presented as business demand findings.

## Real-source evaluation

After placing the reviewed extract at
`data/external/steam_hardware/shs.csv`, run:

```text
python -m dfsignal steam-real --no-fetch-external
```

The command writes validation, mapping-review, launch/adoption, model comparison,
ablation, fold-stability, SHAP-stability, and future-policy artifacts under
`outputs/validation/`. With external FRED/GPU inputs enabled, omit
`--no-fetch-external` to run the full configured backtest.

The model comparison uses the existing rolling-origin evaluation and compares
the configured `gpu_signal` feature set (without Steam features) against
`all_external` (with observed Steam features). It reports WAPE, MAE, RMSE, bias,
and forecast value add (FVA) by model and horizon. The ablation is descriptive:
it does not prove that Steam caused a demand change.

## Provenance and operating rules

- Use only a licensed or permissioned CSV/Parquet historical extract and record its source URL or source notes.
- Do not place credentials, company data, or scraped Valve HTML in this folder.
- Keep the source disabled until coverage, cadence, units, and provenance have been reviewed.
- Feed the validated DataFrame into the normal Bronze/Silver boundary; model and evaluation code remains backend-agnostic and does not read this path directly.
