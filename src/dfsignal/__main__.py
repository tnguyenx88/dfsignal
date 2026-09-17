"""DFSignal local command-line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from typing import Any

from .config import get_settings
from .logging import configure_logging


COMMANDS = (
    "status",
    "ingest",
    "validate",
    "transform",
    "features",
    "backtest",
    "train",
    "forecast",
    "explain",
    "run-all",
    "steam-real",
    "internal-check",
    "demo",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dfsignal", description="Run DFSignal local pipeline stages")
    parser.add_argument("command", nargs="?", choices=COMMANDS, default="status")
    parser.add_argument("--root-dir", default="data", help="Managed Parquet storage root")
    parser.add_argument("--output-dir", default="outputs", help="Dashboard export directory")
    parser.add_argument("--config-dir", default="config", help="Source and model configuration directory")
    parser.add_argument("--steam-source", default="data/external/steam_hardware/shs.csv", help="Local Steam CSV/Parquet extract for steam-real")
    parser.add_argument("--internal-source", default=None, help="Optional approved internal CSV/Parquet path for internal-check")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--no-fetch-external", action="store_true", help="Use synthetic/manual inputs only")
    return parser


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _stage_summary(value: Any) -> dict[str, Any]:
    if hasattr(value, "shape"):
        return {"kind": type(value).__name__, "rows": int(value.shape[0]), "columns": int(value.shape[1])}
    if isinstance(value, Mapping):
        return {"kind": type(value).__name__, "keys": sorted(map(str, value.keys()))}
    return {"kind": type(value).__name__}


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    settings = get_settings()
    configure_logging(settings.log_level)
    from .orchestration import make_context, run_stage
    from .pipeline import run_all, status
    fetch_external = not args.no_fetch_external
    if args.command == "internal-check":
        try:
            from pathlib import Path

            from .ingestion.internal import validate_internal_source

            result = validate_internal_source(
                path=args.internal_source,
                config_path=Path(args.config_dir) / "internal_data.yaml",
                model_config_path=Path(args.config_dir) / "models.yaml",
                root_dir=".",
            )
        except Exception as exc:
            raise SystemExit(f"dfsignal internal-check failed: {type(exc).__name__}: {exc}") from exc
        print(json.dumps(result.to_dict(), indent=2, default=_json_default))
        return
    if args.command == "steam-real":
        try:
            from .evaluation.steam_real import run_real_experiment

            result = run_real_experiment(
                source_path=args.steam_source,
                root_dir=args.root_dir,
                output_dir=args.output_dir,
                config_dir=args.config_dir,
                fetch_external=fetch_external,
                start_date=args.start_date,
                end_date=args.end_date,
            )
        except Exception as exc:
            raise SystemExit(f"dfsignal steam-real failed: {type(exc).__name__}: {exc}") from exc
        print(json.dumps(result, indent=2, default=_json_default))
        return
    if args.command in {"run-all", "demo"}:
        try:
            result = run_all(
                root_dir=args.root_dir,
                output_dir=args.output_dir,
                fetch_external=fetch_external,
                start_date=args.start_date,
                end_date=args.end_date,
                config_dir=args.config_dir,
            )
        except Exception as exc:
            raise SystemExit(f"dfsignal {args.command} failed: {type(exc).__name__}: {exc}") from exc
        print(json.dumps(result["status"], indent=2, default=_json_default))
        return

    context = make_context(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        fetch_external=fetch_external,
        start_date=args.start_date,
        end_date=args.end_date,
        config_dir=args.config_dir,
    )
    try:
        if args.command == "status":
            result = status(context)
            print(json.dumps(result, indent=2, default=_json_default))
            return
        result = run_stage(args.command, context)
        print(
            json.dumps(
                {
                    "stage": args.command,
                    "status": "success",
                    "summary": _stage_summary(result),
                    "health": status(context)["pipeline"]["health"],
                },
                indent=2,
                default=_json_default,
            )
        )
    except Exception as exc:
        raise SystemExit(f"dfsignal {args.command} failed: {type(exc).__name__}: {exc}") from exc
    finally:
        context.storage.close()


if __name__ == "__main__":
    main()
