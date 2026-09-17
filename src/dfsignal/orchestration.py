"""Reusable stage orchestration for the local DFSignal pipeline.

The orchestration layer owns stage ordering, dependency handling, run records,
and failure propagation.  Stage implementations remain in ``pipeline.py`` and
operate on DataFrames plus a storage backend.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from pathlib import Path
from typing import Any


STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "validate": ("ingest",),
    "transform": ("validate",),
    "features": ("transform",),
    "backtest": ("features",),
    "train": ("features",),
    "forecast": ("features", "train"),
    "explain": ("features", "train"),
    "status": (),
}
STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "validate",
    "transform",
    "features",
    "backtest",
    "train",
    "forecast",
    "explain",
)
REQUIRED_STAGES = frozenset(STAGE_DEPENDENCIES) - {"status"}
OPTIONAL_SOURCES = frozenset({"STEAM_HARDWARE"})


@dataclass
class PipelineContext:
    """Shared inputs and in-memory artifacts for one pipeline invocation."""

    storage: Any
    output_dir: str | Path = "outputs"
    start_date: str = "2016-01-01"
    end_date: str = "2025-12-31"
    fetch_external: bool = True
    config_dir: str | Path = "config"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    artifacts: dict[str, Any] = field(default_factory=dict)
    stage_records: dict[str, "StageRecord"] = field(default_factory=dict)
    source_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.config_dir = Path(self.config_dir)


@dataclass
class StageRecord:
    """Operational result for one stage."""

    stage: str
    status: str
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    result_summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_summary": self.result_summary,
        }


@dataclass
class PipelineRun:
    """Summary returned by the dependency-aware runner."""

    run_id: str
    status: str
    stages: dict[str, StageRecord]
    started_at: str
    finished_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stages": {name: record.as_dict() for name, record in self.stages.items()},
        }


class PipelineRunError(RuntimeError):
    """Raised after a run has recorded failures and skipped dependents."""

    def __init__(self, run: PipelineRun) -> None:
        failed = [name for name, record in run.stages.items() if record.status == "failed"]
        super().__init__(f"Pipeline run {run.run_id} failed at stage(s): {', '.join(failed)}")
        self.run = run


def make_context(
    root_dir: str | Path = "data",
    output_dir: str | Path = "outputs",
    *,
    storage: Any | None = None,
    start_date: str = "2016-01-01",
    end_date: str = "2025-12-31",
    fetch_external: bool = True,
    config_dir: str | Path = "config",
) -> PipelineContext:
    """Create a context while keeping local filesystem construction in storage."""

    if storage is None:
        from .storage import LocalStorageBackend

        storage = LocalStorageBackend(root_dir)
    return PipelineContext(
        storage=storage,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        fetch_external=fetch_external,
        config_dir=config_dir,
    )


def stage_functions() -> dict[str, Callable[[PipelineContext], Any]]:
    """Load stage callables lazily to avoid a pipeline/orchestration cycle."""

    from . import pipeline

    return {
        name: getattr(pipeline, name)
        for name in STAGE_DEPENDENCIES
        if name != "status"
    }


def _dependency_closure(stage_names: Iterable[str]) -> list[str]:
    requested = list(stage_names)
    unknown = sorted(set(requested) - set(STAGE_DEPENDENCIES))
    if unknown:
        raise ValueError(f"Unknown pipeline stage(s): {unknown}")
    required: set[str] = set()

    def visit(stage: str) -> None:
        if stage in required:
            return
        required.add(stage)
        for dependency in STAGE_DEPENDENCIES[stage]:
            visit(dependency)

    for stage in requested:
        visit(stage)
    return [stage for stage in STAGE_ORDER if stage in required] + (["status"] if "status" in required else [])


def _summary(value: Any) -> dict[str, Any]:
    """Keep run records compact while retaining useful observable facts."""

    if value is None:
        return {}
    if hasattr(value, "shape"):
        shape = getattr(value, "shape")
        return {"kind": type(value).__name__, "rows": int(shape[0]), "columns": int(shape[1])}
    if isinstance(value, Mapping):
        summary: dict[str, Any] = {"kind": type(value).__name__, "keys": sorted(map(str, value.keys()))}
        for key, item in value.items():
            if hasattr(item, "shape"):
                summary[f"{key}_rows"] = int(item.shape[0])
        return summary
    return {"kind": type(value).__name__}


def run_pipeline(
    context: PipelineContext,
    stages: Iterable[str] | None = None,
    *,
    functions: Mapping[str, Callable[[PipelineContext], Any]] | None = None,
    raise_on_failure: bool = True,
) -> PipelineRun:
    """Execute stages in dependency order and stop failed dependents.

    A failed stage is recorded and all stages that depend on it are marked
    ``skipped``.  Independent siblings still run, allowing one run record to
    show the complete blast radius.  ``raise_on_failure`` controls whether the
    completed run is then raised as :class:`PipelineRunError`.
    """

    selected = list(STAGE_ORDER if stages is None else stages)
    ordered = _dependency_closure(selected)
    if "status" in ordered:
        ordered.remove("status")
    available = dict(functions or stage_functions())
    started_at = datetime.now(timezone.utc).isoformat()
    context.stage_records.clear()
    run = PipelineRun(context.run_id, "running", context.stage_records, started_at)

    for stage in ordered:
        started = datetime.now(timezone.utc).isoformat()
        dependencies = STAGE_DEPENDENCIES[stage]
        blocked = [dependency for dependency in dependencies if context.stage_records.get(dependency, None) and context.stage_records[dependency].status != "success"]
        if blocked:
            record = StageRecord(
                stage=stage,
                status="skipped",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"Blocked by failed or skipped stage(s): {', '.join(blocked)}",
            )
            context.stage_records[stage] = record
            continue
        function = available.get(stage)
        if function is None:
            raise ValueError(f"No callable registered for stage: {stage}")
        try:
            result = function(context)
        except Exception as exc:
            record = StageRecord(
                stage=stage,
                status="failed",
                started_at=started,
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{type(exc).__name__}: {exc}",
            )
            context.stage_records[stage] = record
            continue
        context.artifacts[stage] = result
        context.stage_records[stage] = StageRecord(
            stage=stage,
            status="success",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            result_summary=_summary(result),
        )

    failed = any(record.status == "failed" for record in context.stage_records.values())
    run.status = "failed" if failed else "success"
    run.finished_at = datetime.now(timezone.utc).isoformat()
    try:
        context.storage.record_pipeline_run(
            context.run_id,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            stages={name: record.as_dict() for name, record in context.stage_records.items()},
            error=None if not failed else "One or more required stages failed",
        )
    except AttributeError:
        # A future backend may implement only the core table contract.  Run
        # execution remains valid; LocalStorageBackend records run history.
        pass
    if failed and raise_on_failure:
        raise PipelineRunError(run)
    return run


def run_stage(
    stage: str,
    context: PipelineContext,
    *,
    functions: Mapping[str, Callable[[PipelineContext], Any]] | None = None,
) -> Any:
    """Invoke one stage directly without automatically running dependencies."""

    if stage not in STAGE_DEPENDENCIES or stage == "status":
        raise ValueError(f"Stage is not independently executable: {stage}")
    function = (functions or stage_functions()).get(stage)
    if function is None:
        raise ValueError(f"No callable registered for stage: {stage}")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = function(context)
    except Exception as exc:
        context.stage_records[stage] = StageRecord(
            stage=stage,
            status="failed",
            started_at=started,
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise
    context.artifacts[stage] = result
    context.stage_records[stage] = StageRecord(
        stage=stage,
        status="success",
        started_at=started,
        finished_at=datetime.now(timezone.utc).isoformat(),
        result_summary=_summary(result),
    )
    return result
