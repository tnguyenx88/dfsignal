import pandas as pd

from dfsignal.transformation.steam import (
    derive_steam_signals,
    join_launch_adoption,
    monthly_to_weekly,
    validate_steam_source,
)


def test_monthly_values_align_to_saturdays_without_future_rows() -> None:
    source = pd.DataFrame(
        {
            "survey_month": ["2025-01-01", "2025-02-01"],
            "category": ["GPU", "GPU"],
            "metric_name": ["RTX 50 Series", "RTX 50 Series"],
            "metric_value": [1.0, 2.5],
        }
    )

    weekly = monthly_to_weekly(source)

    assert weekly["week"].dt.dayofweek.eq(5).all()
    assert weekly["week"].max() == pd.Timestamp("2025-02-01")
    assert weekly.loc[weekly["week"] == "2025-01-25", "metric_value"].item() == 1.0
    assert weekly.loc[weekly["week"] == "2025-02-01", "metric_value"].item() == 2.5


def test_generation_mapping_and_momentum_are_observed_only() -> None:
    source = pd.DataFrame(
        {
            "survey_month": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "category": ["GPU"] * 3,
            "metric_name": ["RTX 50 Series"] * 3,
            "metric_value": [1.0, 2.0, 4.0],
        }
    )

    signals = derive_steam_signals(source, mapping_path="config/steam_hardware.yaml")

    assert signals["latest_generation_share"].iloc[-1] == 4.0
    assert signals["latest_generation_share_change_1m"].notna().any()
    assert signals.loc[signals["week"] == "2025-03-01", "latest_generation_share_change_1m"].item() == 2.0
    assert signals["week"].max() == pd.Timestamp("2025-03-01")


def test_launch_join_and_validation_are_descriptive_and_safe() -> None:
    signals = pd.DataFrame(
        {
            "week": pd.to_datetime(["2025-01-04", "2025-01-11"]),
            "latest_generation_share": [1.0, 2.0],
            "source_id": ["STEAM_HARDWARE"] * 2,
        }
    )
    launches = pd.DataFrame({"event_date": ["2025-02-01"], "event_type": ["GPU_LAUNCH"]})

    joined = join_launch_adoption(signals, launches)
    report = validate_steam_source(pd.DataFrame())

    assert joined["major_launch_x_adoption"].eq(0).all()
    assert report["status"] == "REVIEW"
