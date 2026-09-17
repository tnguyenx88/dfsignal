from datetime import date

import pandas as pd
import pytest

from dfsignal.ingestion.steam import (
    SteamHardwareSource,
    build_steam_source_metadata,
    read_steam_hardware,
)


@pytest.fixture
def steam_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "observation_date": ["2025-01-01", "2025-02-01"],
            "metric_name": ["ram_32gb_plus_share", "gpu_adoption"],
            "metric_value": [42.5, 61],
            "source_id": ["COMMUNITY_EXPORT", "COMMUNITY_EXPORT"],
            "source_url": ["https://example.invalid/export", "https://example.invalid/export"],
            "metric_unit": ["percent", "percent"],
        }
    )


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_reads_supported_formats_and_preserves_provenance(tmp_path, steam_rows, suffix) -> None:
    path = tmp_path / f"steam{suffix}"
    if suffix == ".csv":
        steam_rows.to_csv(path, index=False)
    else:
        steam_rows.to_parquet(path, index=False)

    result = read_steam_hardware(path)

    assert result["observation_date"].tolist() == [date(2025, 1, 1), date(2025, 2, 1)]
    assert result["metric_value"].tolist() == [42.5, 61.0]
    assert result["source_id"].tolist() == ["COMMUNITY_EXPORT", "COMMUNITY_EXPORT"]
    assert result["source_url"].tolist() == [
        "https://example.invalid/export",
        "https://example.invalid/export",
    ]
    assert result["metric_unit"].tolist() == ["percent", "percent"]


def test_disabled_source_returns_empty_contract_without_reading_path(tmp_path) -> None:
    result = SteamHardwareSource(tmp_path / "missing.csv").read()

    assert result.empty
    assert list(result.columns) == [
        "observation_date",
        "metric_name",
        "metric_value",
        "source_id",
        "source_url",
    ]


def test_default_relative_directory_is_discoverable(tmp_path, monkeypatch, steam_rows) -> None:
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "data" / "external" / "steam_hardware"
    source_dir.mkdir(parents=True)
    steam_rows.iloc[:1].to_csv(source_dir / "steam.csv", index=False)

    result = read_steam_hardware()

    assert result.loc[0, "source_id"] == "COMMUNITY_EXPORT"
    assert result.loc[0, "metric_name"] == "ram_32gb_plus_share"


def test_missing_required_provenance_fails_loudly(tmp_path) -> None:
    path = tmp_path / "steam.csv"
    pd.DataFrame(
        {
            "observation_date": ["2025-01-01"],
            "metric_name": ["gpu_adoption"],
            "metric_value": [12.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="provenance"):
        read_steam_hardware(path, source_id=None)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("observation_date", "not-a-date", "observation_date"),
        ("metric_value", "not-a-number", "metric_value"),
    ],
)
def test_malformed_required_values_fail_loudly(tmp_path, column, value, message) -> None:
    path = tmp_path / "steam.csv"
    rows = {
        "observation_date": ["2025-01-01"],
        "metric_name": ["gpu_adoption"],
        "metric_value": [12.0],
    }
    rows[column] = [value]
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        read_steam_hardware(path)


def test_duplicate_date_metric_key_fails_loudly(tmp_path) -> None:
    path = tmp_path / "steam.csv"
    pd.DataFrame(
        {
            "observation_date": ["2025-01-01", "2025-01-01"],
            "metric_name": ["gpu_adoption", "gpu_adoption"],
            "metric_value": [12.0, 13.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate keys"):
        read_steam_hardware(path)


def test_metadata_helper_records_coverage_units_and_format(steam_rows) -> None:
    metadata = build_steam_source_metadata(
        steam_rows.assign(observation_date=pd.to_datetime(steam_rows["observation_date"]).dt.date),
        source_path="data/external/steam_hardware/steam.parquet",
        metric_units={"ram_32gb_plus_share": "percent"},
    )

    assert metadata.source_id == "COMMUNITY_EXPORT"
    assert metadata.source_format == "parquet"
    assert metadata.observation_start == date(2025, 1, 1)
    assert metadata.observation_end == date(2025, 2, 1)
    assert metadata.as_dict()["date_coverage"] == {
        "start": "2025-01-01",
        "end": "2025-02-01",
    }
    assert metadata.as_dict()["metric_units"] == {"ram_32gb_plus_share": "percent"}
