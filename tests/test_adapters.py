import pandas as pd

from dfsignal.ingestion.steam import read_steam_hardware
from dfsignal.transformation.weekly import to_week_saturday


def test_steam_adapter_reads_long_form_csv(tmp_path) -> None:
    path = tmp_path / "steam.csv"
    pd.DataFrame(
        {
            "observation_date": ["2025-01-01"],
            "metric_name": ["ram_32gb_plus_share"],
            "metric_value": [42.5],
        }
    ).to_csv(path, index=False)

    result = read_steam_hardware(path)

    assert result.loc[0, "metric_name"] == "ram_32gb_plus_share"
    assert float(result.loc[0, "metric_value"]) == 42.5
    assert str(to_week_saturday(pd.Series(["2025-01-01"])).iloc[0]) == "2025-01-04"
