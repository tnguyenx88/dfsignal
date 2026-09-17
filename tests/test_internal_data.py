from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from dfsignal.ingestion.internal import (
    CANONICAL_COLUMNS,
    LocalCSVInternalDemandSource,
    map_internal_columns,
    validate_internal_source,
)


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "internal_data.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": {
                    "source_id": "internal_demand",
                    "source_name": "Test de-identified extract",
                    "type": "local_internal",
                    "enabled": False,
                    "path": "data/internal/*",
                    "supported_formats": ["csv", "parquet"],
                },
                "grain": {
                    "preferred": "WEEK x PRODUCT_GROUP",
                    "keys": ["week", "product_group"],
                    "minimum_keys": ["week", "product_group"],
                },
                "week_column": "period",
                "date_column": None,
                "dimensions": {"product_group": "group_label"},
                "measures": {"pos_qty": "quantity"},
                "target": {"preferred": "pos_qty", "alternatives": ["shipment_qty", "booking_qty"]},
                "allowed_product_groups": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mapping_drops_unmapped_and_sensitive_source_columns(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw = pd.DataFrame(
        {
            "period": ["2025-01-04"],
            "group_label": ["group_a"],
            "quantity": [10.0],
            "customer_id": ["not-needed"],
            "unmapped_note": ["drop-me"],
        }
    )

    result = map_internal_columns(raw, config)

    assert set(result.columns) == {"week", "product_group", "pos_qty"}
    assert set(result.columns).issubset(CANONICAL_COLUMNS)
    assert result.loc[0, "week"] == pd.Timestamp("2025-01-04")
    assert "customer_id" not in result.columns


def test_sensitive_source_column_cannot_be_explicitly_mapped(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dimensions"]["product_group"] = "customer_id"
    raw = pd.DataFrame(
        {
            "period": ["2025-01-04"],
            "customer_id": ["not-needed"],
            "quantity": [10.0],
        }
    )

    result = map_internal_columns(raw, config)

    assert set(result.columns) == {"week", "pos_qty"}
    assert "customer_id" not in result.columns


def test_missing_internal_file_is_optional(tmp_path: Path) -> None:
    result = validate_internal_source(
        config_path=_config(tmp_path),
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
    )

    payload = result.to_dict()
    assert payload["status"] == "MISSING_OPTIONAL"
    assert payload["source_status"] == "MISSING_OPTIONAL"
    assert payload["checks"][0]["check"] == "file_exists"
    assert payload["checks"][0]["status"] == "REVIEW"


def test_available_internal_file_reports_quality_findings_without_copying_identifiers(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    source_path = tmp_path / "approved_extract.csv"
    pd.DataFrame(
        {
            "period": ["2025-01-04", "2025-01-04"],
            "group_label": ["group_a", "group_a"],
            "quantity": [10.0, -2.0],
            "customer_id": ["not-needed", "not-needed"],
        }
    ).to_csv(source_path, index=False)

    adapter = LocalCSVInternalDemandSource(source_path, config_path)
    mapped = adapter.read()
    result = validate_internal_source(
        source_path,
        config_path=config_path,
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
        now="2025-01-08",
    )
    checks = {check["check"]: check for check in result.to_dict()["checks"]}

    assert set(mapped.columns) == {"week", "product_group", "pos_qty"}
    assert result.source_status == "AVAILABLE"
    assert result.status == "FAIL"
    assert checks["duplicate_grain_keys"]["status"] == "FAIL"
    assert checks["negative_quantities"]["status"] == "FAIL"
    assert checks["deidentification_guardrail"]["status"] == "REVIEW"
    assert result.to_dict()["provenance"]["row_count"] == 2
    assert result.to_dict()["provenance"]["data_classification"] == "CONFIDENTIAL_INTERNAL_DE_IDENTIFIED"


def test_explicit_missing_internal_file_is_a_failure(tmp_path: Path) -> None:
    result = validate_internal_source(
        tmp_path / "missing.csv",
        config_path=_config(tmp_path),
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
    )

    assert result.status == "FAIL"
    assert result.source_status == "INVALID"
    assert result.to_dict()["checks"][0]["status"] == "FAIL"


def test_timezone_dates_are_normalized_and_non_saturday_weeks_are_reviewed(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    source_path = tmp_path / "timezone_extract.csv"
    pd.DataFrame(
        {
            "period": ["2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z"],
            "group_label": ["group_a", "group_b"],
            "quantity": [10.0, 11.0],
        }
    ).to_csv(source_path, index=False)

    result = validate_internal_source(
        source_path,
        config_path=config_path,
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
        now=datetime(2025, 1, 8, tzinfo=timezone.utc),
    )
    checks = {check["check"]: check for check in result.to_dict()["checks"]}

    assert result.status == "REVIEW"
    assert checks["weekly_dates"]["status"] == "REVIEW"
    assert checks["partial_latest_week"]["status"] == "PASS"


def test_current_saturday_is_not_treated_as_completed(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    source_path = tmp_path / "current_saturday.csv"
    pd.DataFrame(
        {
            "period": ["2025-01-11"],
            "group_label": ["group_a"],
            "quantity": [10.0],
        }
    ).to_csv(source_path, index=False)

    result = validate_internal_source(
        source_path,
        config_path=config_path,
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
        now="2025-01-11",
    )
    checks = {check["check"]: check for check in result.to_dict()["checks"]}

    assert checks["partial_latest_week"]["status"] == "REVIEW"


def test_empty_internal_file_fails_validation(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    source_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["period", "group_label", "quantity"]).to_csv(source_path, index=False)

    result = validate_internal_source(
        source_path,
        config_path=config_path,
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
    )

    checks = {check["check"]: check for check in result.to_dict()["checks"]}
    assert result.status == "FAIL"
    assert checks["non_empty"]["status"] == "FAIL"


def test_configured_optional_dimensions_extend_duplicate_grain(tmp_path: Path) -> None:
    config_path = _config(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["dimensions"]["region"] = "region_name"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    source_path = tmp_path / "regional_extract.csv"
    pd.DataFrame(
        {
            "period": ["2025-01-04", "2025-01-04"],
            "group_label": ["group_a", "group_a"],
            "region_name": ["north", "south"],
            "quantity": [10.0, 11.0],
        }
    ).to_csv(source_path, index=False)

    result = validate_internal_source(
        source_path,
        config_path=config_path,
        model_config_path="config/models.yaml",
        root_dir=tmp_path,
        now="2025-01-08",
    )
    checks = {check["check"]: check for check in result.to_dict()["checks"]}

    assert checks["duplicate_grain_keys"]["status"] == "PASS"
