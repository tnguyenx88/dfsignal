# Future Fabric Migration

DFSignal currently runs entirely in `LOCAL_DEV`. This document records the
replacement seams; it is not a Fabric implementation.

| Local component | Future Fabric mapping | Preserved contract |
|---|---|---|
| `LocalStorageBackend` | Lakehouse/OneLake backend or Warehouse backend | `StorageBackend` table methods |
| Bronze Parquet | Lakehouse Bronze Delta/Parquet | source-native rows + provenance metadata |
| Silver Parquet | Spark Notebook / Lakehouse Silver | weekly normalized schema |
| Gold Parquet | Lakehouse Gold or Warehouse tables | feature and forecast schemas |
| DuckDB | Lakehouse SQL endpoint or Fabric Warehouse SQL | analytical query semantics |
| Python transforms | Fabric Notebook/PySpark implementation | function-level data contracts |
| Local pipeline CLI | Fabric Pipeline orchestration | stage ordering and run metadata |
| Local env vars | Fabric connections / Key Vault later | settings names, no secret values |
| Local model execution | Fabric Data Science / Notebook | model input/output schemas |

The migration should add a backend implementation and deployment adapter, not
rewrite the domain modules. Any engine-specific SQL must remain isolated in
`storage.py` or the future backend.
