"""Storage abstraction for local Parquet and DuckDB.

Parquet files are the system of record for the local proof of concept.  DuckDB
is deliberately kept as a query layer: every view points back to one of the
Parquet files managed by :class:`LocalStorageBackend`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
from typing import Any, Protocol

import duckdb
import pandas as pd


SUPPORTED_LAYERS = frozenset({"bronze", "silver", "gold", "cache", "external", "synthetic"})
WATERMARK_TABLE = "watermarks"
SOURCE_REFRESH_TABLE = "source_refresh"
PIPELINE_RUN_TABLE = "pipeline_runs"
WATERMARK_COLUMNS = ["source_id", "watermark_type", "watermark_value", "updated_at"]
SOURCE_REFRESH_COLUMNS = [
    "source_id",
    "status",
    "refreshed_at",
    "updated_at",
    "row_count",
    "min_date",
    "max_date",
    "latest_data_week",
    "table_name",
    "layer",
    "error",
]
PIPELINE_RUN_COLUMNS = ["run_id", "status", "started_at", "finished_at", "stages", "error"]


class StorageBackend(Protocol):
    """Backend contract shared by local and future Fabric implementations."""

    def read_table(self, table_name: str, layer: str) -> pd.DataFrame: ...

    def write_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        layer: str,
        source_id: str,
        *,
        mode: str = "replace",
        key_columns: Sequence[str] | None = None,
    ) -> Path: ...

    def table_exists(self, table_name: str, layer: str) -> bool: ...

    def read_metadata(self, table_name: str, layer: str) -> dict[str, Any]: ...

    def read_watermarks(self) -> pd.DataFrame: ...

    def get_watermark(self, source_id: str, watermark_type: str) -> str | None: ...

    def set_watermark(
        self,
        source_id: str,
        watermark_type: str,
        watermark_value: Any,
        *,
        updated_at: datetime | str | None = None,
    ) -> Path: ...

    def read_source_refresh(self, source_id: str | None = None) -> pd.DataFrame: ...


class LocalStorageBackend:
    """Persist medallion tables as Parquet with DuckDB query-layer views.

    ``replace`` is the default write mode.  It makes a stage rerun a clean
    replacement of the table rather than an append.  ``upsert`` is available
    for incremental callers and replaces rows matching ``key_columns`` (or
    exact duplicate rows when no keys are supplied).
    """

    def __init__(self, root_dir: str | Path = "data", database_path: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir)
        self.database_path = Path(database_path or self.root_dir / "dfsignal.duckdb")
        self.connection: duckdb.DuckDBPyConnection | None = None

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open and cache the local DuckDB query-layer connection."""

        if self.connection is None:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = duckdb.connect(str(self.database_path))
        return self.connection

    def resolve_local_path(self, filename: str | Path, area: str = "external") -> Path:
        """Resolve a managed local file path without exposing the root layout.

        Adapters occasionally need a local source archive (for example, the
        GPU ZIP).  They receive this path from storage rather than constructing
        a ``data/...`` path themselves.
        """

        if area not in SUPPORTED_LAYERS:
            raise ValueError(f"Unsupported storage area: {area}")
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Managed local paths must be relative to their storage area")
        return self.root_dir / area / relative

    def _path(self, table_name: str, layer: str) -> Path:
        if layer not in SUPPORTED_LAYERS:
            raise ValueError(f"Unsupported storage layer: {layer}")
        table = Path(table_name)
        if not table_name or table.name != table_name or table.suffix:
            raise ValueError("Table names must be simple names without path or suffix components")
        return self.root_dir / layer / f"{table_name}.parquet"

    @staticmethod
    def _metadata_path(path: Path) -> Path:
        return path.with_suffix(".metadata.json")

    def table_exists(self, table_name: str, layer: str) -> bool:
        return self._path(table_name, layer).exists()

    def read_table(self, table_name: str, layer: str) -> pd.DataFrame:
        path = self._path(table_name, layer)
        if not path.exists():
            raise FileNotFoundError(path)
        return pd.read_parquet(path)

    def read_metadata(self, table_name: str, layer: str) -> dict[str, Any]:
        """Read a table sidecar, returning an empty mapping when not present."""

        path = self._metadata_path(self._path(table_name, layer))
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata sidecar must contain an object: {path}")
        return payload

    # ``metadata`` was documented by the original architecture even though
    # the first implementation only wrote sidecars.  Keep it as a stable alias.
    metadata = read_metadata

    def write_table(
        self,
        table_name: str,
        frame: pd.DataFrame,
        layer: str,
        source_id: str,
        *,
        mode: str = "replace",
        key_columns: Sequence[str] | None = None,
        keys: Sequence[str] | None = None,
    ) -> Path:
        """Write a table idempotently and refresh its DuckDB view.

        ``keys`` is accepted as a convenience alias for ``key_columns`` for
        callers using incremental-load terminology.  A caller may choose
        ``mode="upsert"`` to retain rows not present in the current batch.
        """

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        normalized_mode = str(mode).lower()
        if normalized_mode == "overwrite":
            normalized_mode = "replace"
        if normalized_mode not in {"replace", "upsert"}:
            raise ValueError("mode must be 'replace' or 'upsert'")
        if key_columns is not None and keys is not None and list(key_columns) != list(keys):
            raise ValueError("Provide only one of key_columns and keys")
        selected_keys = list(key_columns if key_columns is not None else keys or [])
        if len(selected_keys) != len(set(selected_keys)):
            raise ValueError("key_columns must not contain duplicates")
        if selected_keys and not set(selected_keys).issubset(frame.columns):
            missing = sorted(set(selected_keys) - set(frame.columns))
            raise ValueError(f"Upsert key columns missing from frame: {missing}")

        persisted = frame.copy()
        if normalized_mode == "upsert" and self.table_exists(table_name, layer):
            previous = self.read_table(table_name, layer)
            if selected_keys:
                if not set(selected_keys).issubset(previous.columns):
                    missing = sorted(set(selected_keys) - set(previous.columns))
                    raise ValueError(f"Upsert key columns missing from existing table: {missing}")
                previous = previous.drop_duplicates(subset=selected_keys, keep="last")
                persisted = persisted.drop_duplicates(subset=selected_keys, keep="last")
                incoming_keys = persisted[selected_keys].drop_duplicates()
                # A merge indicator avoids tuple hashing surprises with
                # nullable/object key values and preserves incoming dtypes.
                previous_index = pd.MultiIndex.from_frame(previous[selected_keys])
                incoming_index = pd.MultiIndex.from_frame(incoming_keys[selected_keys])
                persisted = pd.concat(
                    [previous.loc[~previous_index.isin(incoming_index)], persisted],
                    ignore_index=True,
                    sort=False,
                )
            else:
                persisted = pd.concat([previous, persisted], ignore_index=True, sort=False).drop_duplicates(
                    keep="last"
                )
        elif normalized_mode == "upsert":
            persisted = persisted.drop_duplicates(subset=selected_keys or None, keep="last")

        path = self._path(table_name, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        persisted.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)

        metadata = {
            "source_id": str(source_id),
            "load_timestamp": timestamp,
            "updated_at": timestamp,
            "row_count": int(len(persisted)),
            "min_date": _date_bound(persisted, "week", "date", "observation_date", "event_date"),
            "max_date": _date_bound(persisted, "week", "date", "observation_date", "event_date", maximum=True),
            "latest_data_week": _date_bound(
                persisted, "week", "date", "observation_date", "event_date", maximum=True
            ),
            "duplicate_count": int(persisted.duplicated().sum()),
            "missing_rate": float(persisted.isna().mean().mean()) if not persisted.empty else 0.0,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "write_mode": normalized_mode,
            "key_columns": selected_keys,
        }
        metadata_path = self._metadata_path(path)
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata, metadata_path)
        self._refresh_view(table_name, path)

        # Keep a reusable source-level record in addition to each table's
        # sidecar.  Metadata writes use the private path to avoid recursion.
        self.record_source_refresh(
            str(source_id),
            status="success",
            refreshed_at=timestamp,
            row_count=len(persisted),
            min_date=metadata["min_date"],
            max_date=metadata["max_date"],
            latest_data_week=metadata["latest_data_week"],
            table_name=table_name,
            layer=layer,
        )
        return path

    def _refresh_view(self, table_name: str, path: Path) -> None:
        connection = self.connect()
        escaped = str(path.resolve()).replace("\\", "/").replace("'", "''")
        connection.execute(f'CREATE OR REPLACE VIEW "{table_name}" AS SELECT * FROM read_parquet(\'{escaped}\')')

    def query(self, sql: str, parameters: Sequence[Any] | Mapping[str, Any] | None = None) -> pd.DataFrame:
        """Execute a read/query-layer SQL statement and return a DataFrame."""

        connection = self.connect()
        if parameters is None:
            return connection.execute(sql).df()
        return connection.execute(sql, parameters).df()

    def _persist_internal(
        self,
        table_name: str,
        frame: pd.DataFrame,
        layer: str,
        *,
        source_id: str,
        mode: str = "replace",
        key_columns: Sequence[str] | None = None,
    ) -> Path:
        """Persist bookkeeping tables without recursively recording refreshes."""

        normalized_mode = str(mode).lower()
        if normalized_mode not in {"replace", "upsert"}:
            raise ValueError("mode must be 'replace' or 'upsert'")
        persisted = frame.copy()
        if normalized_mode == "upsert" and self.table_exists(table_name, layer):
            previous = self.read_table(table_name, layer)
            if key_columns:
                previous = previous.drop_duplicates(subset=list(key_columns), keep="last")
                persisted = persisted.drop_duplicates(subset=list(key_columns), keep="last")
                incoming_keys = pd.MultiIndex.from_frame(persisted[list(key_columns)])
                previous_keys = pd.MultiIndex.from_frame(previous[list(key_columns)])
                persisted = pd.concat(
                    [previous.loc[~previous_keys.isin(incoming_keys)], persisted],
                    ignore_index=True,
                    sort=False,
                )
            else:
                persisted = pd.concat([previous, persisted], ignore_index=True, sort=False).drop_duplicates(
                    keep="last"
                )
        path = self._path(table_name, layer)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        persisted.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, path)
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "source_id": source_id,
            "load_timestamp": timestamp,
            "updated_at": timestamp,
            "row_count": int(len(persisted)),
            "min_date": _date_bound(persisted, "week", "date", "observation_date", "event_date"),
            "max_date": _date_bound(persisted, "week", "date", "observation_date", "event_date", maximum=True),
            "duplicate_count": int(persisted.duplicated().sum()),
            "missing_rate": float(persisted.isna().mean().mean()) if not persisted.empty else 0.0,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "write_mode": normalized_mode,
            "key_columns": list(key_columns or []),
        }
        metadata_path = self._metadata_path(path)
        temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
        temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(temporary_metadata, metadata_path)
        self._refresh_view(table_name, path)
        return path

    def read_watermarks(self) -> pd.DataFrame:
        """Return all persisted watermarks using a stable four-column schema."""

        if not self.table_exists(WATERMARK_TABLE, "cache"):
            return pd.DataFrame(columns=WATERMARK_COLUMNS)
        frame = self.read_table(WATERMARK_TABLE, "cache")
        for column in WATERMARK_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return frame[WATERMARK_COLUMNS]

    def get_watermark(self, source_id: str, watermark_type: str) -> str | None:
        """Return one watermark value, or ``None`` when it has not been set."""

        frame = self.read_watermarks()
        if frame.empty:
            return None
        matches = frame[
            frame["source_id"].astype(str).eq(str(source_id))
            & frame["watermark_type"].astype(str).eq(str(watermark_type))
        ]
        if matches.empty:
            return None
        value = matches.iloc[-1]["watermark_value"]
        return None if pd.isna(value) else str(value)

    def get_watermark_record(self, source_id: str, watermark_type: str) -> dict[str, Any] | None:
        """Return the complete watermark record for callers needing timestamps."""

        frame = self.read_watermarks()
        if frame.empty:
            return None
        matches = frame[
            frame["source_id"].astype(str).eq(str(source_id))
            & frame["watermark_type"].astype(str).eq(str(watermark_type))
        ]
        if matches.empty:
            return None
        return _record_to_dict(matches.iloc[-1])

    def set_watermark(
        self,
        source_id: str,
        watermark_type: str,
        watermark_value: Any,
        *,
        updated_at: datetime | str | None = None,
    ) -> Path:
        """Upsert a source watermark keyed by source and watermark type."""

        timestamp = _as_timestamp(updated_at)
        row = pd.DataFrame(
            [
                {
                    "source_id": str(source_id),
                    "watermark_type": str(watermark_type),
                    "watermark_value": _as_string(watermark_value),
                    "updated_at": timestamp,
                }
            ],
            columns=WATERMARK_COLUMNS,
        )
        return self._persist_internal(
            WATERMARK_TABLE,
            row,
            "cache",
            source_id="DFSIGNAL_METADATA",
            mode="upsert",
            key_columns=("source_id", "watermark_type"),
        )

    # Explicit aliases make the bookkeeping API discoverable without forcing
    # callers to remember whether the operation is a read or a setter.
    write_watermark = set_watermark
    update_watermark = set_watermark

    def record_source_refresh(
        self,
        source_id: str,
        *,
        status: str = "success",
        refreshed_at: datetime | str | None = None,
        row_count: int | None = None,
        min_date: Any = None,
        max_date: Any = None,
        latest_data_week: Any = None,
        table_name: str | None = None,
        layer: str | None = None,
        error: str | None = None,
    ) -> Path:
        """Persist source availability and refresh information idempotently."""

        timestamp = _as_timestamp(refreshed_at)
        row = pd.DataFrame(
            [
                {
                    "source_id": str(source_id),
                    "status": str(status),
                    "refreshed_at": timestamp,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "row_count": row_count,
                    "min_date": _as_string(min_date),
                    "max_date": _as_string(max_date),
                    "latest_data_week": _as_string(latest_data_week),
                    "table_name": table_name,
                    "layer": layer,
                    "error": error,
                }
            ],
            columns=SOURCE_REFRESH_COLUMNS,
        )
        return self._persist_internal(
            SOURCE_REFRESH_TABLE,
            row,
            "cache",
            source_id="DFSIGNAL_METADATA",
            mode="upsert",
            key_columns=("source_id",),
        )

    write_source_refresh = record_source_refresh
    write_source_refresh_metadata = record_source_refresh

    def read_source_refresh(self, source_id: str | None = None) -> pd.DataFrame:
        """Read source refresh records, optionally filtered to one source."""

        if not self.table_exists(SOURCE_REFRESH_TABLE, "cache"):
            return pd.DataFrame(columns=SOURCE_REFRESH_COLUMNS)
        frame = self.read_table(SOURCE_REFRESH_TABLE, "cache")
        for column in SOURCE_REFRESH_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[SOURCE_REFRESH_COLUMNS]
        if source_id is not None:
            frame = frame[frame["source_id"].astype(str).eq(str(source_id))]
        return frame.reset_index(drop=True)

    source_refresh_metadata = read_source_refresh

    def get_source_refresh(self, source_id: str) -> dict[str, Any] | None:
        frame = self.read_source_refresh(source_id)
        return None if frame.empty else _record_to_dict(frame.iloc[-1])

    def record_pipeline_run(
        self,
        run_id: str,
        *,
        status: str,
        started_at: datetime | str,
        finished_at: datetime | str | None = None,
        stages: Mapping[str, Any] | str | None = None,
        error: str | None = None,
    ) -> Path:
        """Persist one pipeline run summary for status and operational review."""

        stages_value = json.dumps(stages, default=str, sort_keys=True) if not isinstance(stages, str) else stages
        row = pd.DataFrame(
            [
                {
                    "run_id": str(run_id),
                    "status": str(status),
                    "started_at": _as_timestamp(started_at),
                    "finished_at": _as_timestamp(finished_at),
                    "stages": stages_value,
                    "error": error,
                }
            ],
            columns=PIPELINE_RUN_COLUMNS,
        )
        return self._persist_internal(
            PIPELINE_RUN_TABLE,
            row,
            "cache",
            source_id="DFSIGNAL_METADATA",
            mode="upsert",
            key_columns=("run_id",),
        )

    def read_pipeline_runs(self) -> pd.DataFrame:
        if not self.table_exists(PIPELINE_RUN_TABLE, "cache"):
            return pd.DataFrame(columns=PIPELINE_RUN_COLUMNS)
        frame = self.read_table(PIPELINE_RUN_TABLE, "cache")
        for column in PIPELINE_RUN_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return frame[PIPELINE_RUN_COLUMNS].sort_values("started_at", na_position="first").reset_index(drop=True)

    def list_tables(self, layer: str | None = None) -> list[dict[str, Any]]:
        """List managed Parquet tables and their sidecar summaries."""

        layers = [layer] if layer is not None else sorted(SUPPORTED_LAYERS - {"external", "synthetic"})
        records: list[dict[str, Any]] = []
        for selected_layer in layers:
            if selected_layer not in SUPPORTED_LAYERS:
                raise ValueError(f"Unsupported storage layer: {selected_layer}")
            directory = self.root_dir / selected_layer
            if not directory.exists():
                continue
            for path in sorted(directory.glob("*.parquet")):
                table = path.stem
                record = {"table_name": table, "layer": selected_layer, "path": str(path)}
                record.update(self.read_metadata(table, selected_layer))
                records.append(record)
        return records

    def export_table(
        self,
        filename: str,
        frame: pd.DataFrame,
        output_dir: str | Path = "outputs",
    ) -> Path:
        """Write a dashboard copy without exposing storage layout to stages."""

        output_path = Path(output_dir) / f"{filename}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        frame.to_parquet(temporary_path, index=False)
        os.replace(temporary_path, output_path)
        return output_path

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None


def _as_timestamp(value: datetime | str | date | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat() if value.tzinfo else value.replace(tzinfo=timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _as_string(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def _record_to_dict(record: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in record.to_dict().items():
        result[str(key)] = None if pd.isna(value) else value
    return result


def _date_bound(frame: pd.DataFrame, *columns: str, maximum: bool = False) -> str | None:
    for column in columns:
        if column in frame.columns and not frame[column].dropna().empty:
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not dates.empty:
                return str((dates.max() if maximum else dates.min()).date())
    return None
