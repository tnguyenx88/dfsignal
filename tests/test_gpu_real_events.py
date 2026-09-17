import pandas as pd

from dfsignal.ingestion.events import read_launch_events, usable_launch_events
from dfsignal.ingestion.gpu import (
    derive_gpu_launch_events,
    derive_gpu_weekly_signals,
    normalize_gpu_specs,
)
from dfsignal.pipeline import _event_weekly
from dfsignal.validation.gpu import generate_gpu_validation_report


def _gpu_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Brand": ["NVIDIA", "NVIDIA", "NVIDIA"],
            "Name": ["GeForce RTX 4090", "GeForce RTX 4090 Laptop GPU", "H100 SXM"],
            "Graphics Processor__Architecture": ["Ada Lovelace", "Ada Lovelace", "Hopper"],
            "Graphics Card__Release Date": ["Oct 12th, 2022", "", "Apr 1st, 2023"],
            "Mobile Graphics__Release Date": ["", "Oct 12th, 2022", ""],
            "Graphics Card__Generation": ["Ada Lovelace", "Ada Lovelace", "Hopper"],
            "Memory__Memory Size": ["24 GB", "16 GB", "80 GB"],
            "Memory__Memory Type": ["GDDR6X", "GDDR6", "HBM3"],
            "Memory__Memory Bus": ["384 bit", "256 bit", "512 bit"],
            "Board Design__TDP": ["450 W", "150 W", "700 W"],
        }
    )


def test_placeholder_is_retained_but_not_usable() -> None:
    events = read_launch_events("config/launch_events.csv")
    assert bool(events.loc[0, "is_example_event"])
    assert events.loc[0, "event_status"] == "EXAMPLE"
    assert usable_launch_events(events).empty
    assert _event_weekly(events).empty


def test_gpu_mapping_filter_events_and_weekly_deduplication() -> None:
    products = normalize_gpu_specs(_gpu_rows())
    assert products["source_record_id"].is_unique
    assert products.loc[0, "generation"] == "RTX_40"
    assert products.loc[1, "market_segment_category"] == "MOBILE"

    events = derive_gpu_launch_events(products)
    assert len(events) == 1
    assert events.loc[0, "product_name"] == "GeForce RTX 4090"
    assert events.loc[0, "event_status"] == "AUTO_DERIVED"
    assert events.loc[0, "event_type"] == "GPU_LAUNCH"

    duplicated = pd.concat([products.iloc[[0]], products], ignore_index=True)
    weekly = derive_gpu_weekly_signals(duplicated)
    assert int(weekly["gpu_launch_count"].sum()) == 1
    assert int(weekly["gpu_duplicate_record_count"].sum()) == 1


def test_gpu_validation_persists_mixed_diagnostic_cells(tmp_path) -> None:
    products = normalize_gpu_specs(_gpu_rows())
    report = generate_gpu_validation_report(products=products, output_dir=tmp_path)

    checks_path = tmp_path / "gpu_validation_checks.parquet"
    assert checks_path.exists()
    checks = pd.read_parquet(checks_path)
    observed = checks.loc[checks["check"] == "weekly_launch_count_reconciles", "observed"].iloc[0]
    assert '"weekly_sum"' in observed
    assert report["artifacts"]["checks"] == str(checks_path)
