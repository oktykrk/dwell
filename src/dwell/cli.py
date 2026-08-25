from __future__ import annotations

import asyncio
import functools
import json
import os
import time
import urllib.error
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, ParamSpec, TypeVar

import typer
from pydantic import BaseModel, ValidationError

from dwell import __version__
from dwell.config import DwellConfig
from dwell.doctor import has_errors, run_doctor
from dwell.domain import JobRecord, JobStatus
from dwell.errors import DwellError
from dwell.process import (
    api_request,
    restart_server,
    server_status,
    start_server,
    stop_server,
)

P = ParamSpec("P")
R = TypeVar("R")
_verbose = False


app = typer.Typer(
    name="dwell",
    help="Personal local AI inference gateway for Apple Silicon.",
    no_args_is_help=True,
    invoke_without_command=True,
    pretty_exceptions_show_locals=False,
)
models_app = typer.Typer(
    help="Inspect, install, and control registered models.", no_args_is_help=True
)
jobs_app = typer.Typer(help="Inspect and manage inference jobs.", no_args_is_help=True)
outputs_app = typer.Typer(help="Inspect persistent inference outputs.", no_args_is_help=True)
config_app = typer.Typer(help="Show resolved local configuration.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(jobs_app, name="jobs")
app.add_typer(outputs_app, name="outputs")
app.add_typer(config_app, name="config")


def _handled(function: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (typer.Exit, typer.Abort):
            raise
        except DwellError as exc:
            typer.echo(f"✗ {exc.message}", err=True)
            if _verbose and exc.details is not None:
                typer.echo(json.dumps(exc.details, indent=2, default=str), err=True)
            raise typer.Exit(code=1) from exc
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            location = ".".join(str(item) for item in first.get("loc", ()))
            prefix = f"{location}: " if location else ""
            typer.echo(f"✗ {prefix}{first['msg']}", err=True)
            raise typer.Exit(code=2) from exc
        except (OSError, ValueError, urllib.error.URLError) as exc:
            typer.echo(f"✗ {exc}", err=True)
            raise typer.Exit(code=1) from exc
        except Exception as exc:
            if _verbose:
                raise
            typer.echo(f"✗ Unexpected error: {exc}", err=True)
            typer.echo("Run again with --verbose for a traceback.", err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


@app.callback()
def main(
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show detailed errors."),
    ] = False,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the Dwell version and exit.", is_eager=True),
    ] = False,
) -> None:
    global _verbose
    _verbose = verbose
    if version:
        typer.echo(f"Dwell {__version__}")
        raise typer.Exit()


def _config() -> DwellConfig:
    return DwellConfig.from_env()


def _model_manager(config: DwellConfig):  # type: ignore[no-untyped-def]
    from dwell.model_manager import ModelManager
    from dwell.registry import ModelRegistry

    return ModelManager(config, registry=ModelRegistry.load(config))


def _job_store(config: DwellConfig):  # type: ignore[no-untyped-def]
    from dwell.jobs import JobStore

    config.ensure_layout()
    return JobStore(config.jobs_db)


def _enable_local_logging(config: DwellConfig) -> None:
    from dwell.logging_config import configure_logging

    config.ensure_layout()
    configure_logging(config)


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    total = max(0, int(seconds))
    days, remainder = divmod(total, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def _table(headers: list[str], rows: list[list[Any]]) -> None:
    values = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in values:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    typer.echo("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    for row in values:
        typer.echo("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _model_disk_impact(cache_path: str | None) -> str:
    if not cache_path:
        return "unknown"
    path = Path(cache_path)
    root = path.parent.parent if path.parent.name == "snapshots" else path
    total = 0
    try:
        for directory, _subdirectories, names in os.walk(root, followlinks=False):
            for name in names:
                file_path = Path(directory) / name
                if not file_path.is_symlink():
                    total += file_path.stat().st_size
    except OSError:
        return "unknown (shared-cache accounting unavailable)"
    return f"up to {total / (1024**3):.2f} GB in the shared repository cache"


@app.command()
@_handled
def start() -> None:
    """Start the localhost API in the background."""

    config = _config()
    state = start_server(config)
    status = api_request(config, "/v1/status", timeout=2)
    loaded = status.get("loaded_models", []) if isinstance(status, dict) else []
    typer.echo("Dwell started")
    typer.echo(f"API: {config.api_url}")
    typer.echo(f"PID: {state.pid}")
    typer.echo(f"Models loaded: {', '.join(loaded) if loaded else 'none'}")


@app.command()
@_handled
def stop() -> None:
    """Gracefully stop the API and active inference child process."""

    config = _config()
    if stop_server(config):
        typer.echo("Dwell stopped")
    else:
        typer.echo("Dwell is already stopped")


@app.command()
@_handled
def restart() -> None:
    """Cleanly stop and start Dwell."""

    config = _config()
    state = restart_server(config)
    typer.echo("Dwell restarted")
    typer.echo(f"API: {config.api_url}")
    typer.echo(f"PID: {state.pid}")


@app.command()
@_handled
def status() -> None:
    """Show service, model, queue, and memory state."""

    config = _config()
    status_data = server_status(config)
    typer.echo(f"Dwell: {'running' if status_data['running'] else 'stopped'}")
    typer.echo(f"API: {status_data['api']}")
    if not status_data["running"]:
        typer.echo("PID: -")
        typer.echo("Active models: none")
        typer.echo("Active job: none")
        queued = _job_store(config).count(JobStatus.QUEUED)
        typer.echo(f"Queued jobs: {queued}")
        return

    typer.echo(f"PID: {status_data['pid']}")
    typer.echo(f"Uptime: {_format_duration(status_data['uptime_seconds'])}")
    if status_data["rss_mb"] is not None:
        typer.echo(f"RSS: {status_data['rss_mb']:.1f} MB")
    remote = status_data.get("api_status") or {}
    if status_data.get("api_status") is None:
        typer.echo("Active models: unknown (API unreachable)")
        typer.echo("Active job: unknown (API unreachable)")
        typer.echo("Queued jobs: unknown (API unreachable)")
        return
    loaded = remote.get("loaded_models", [])
    typer.echo(f"Active models: {', '.join(loaded) if loaded else 'none'}")
    typer.echo(f"Active job: {remote.get('active_job') or 'none'}")
    typer.echo(f"Queued jobs: {remote.get('queued_jobs', 'unknown')}")


@app.command()
@_handled
def logs(
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="Continue printing appended log lines."),
    ] = False,
    lines: Annotated[
        int,
        typer.Option("--lines", "-n", min=1, help="Number of recent lines to show."),
    ] = 100,
) -> None:
    """Show recent server logs."""

    path = _config().log_file
    if not path.exists():
        typer.echo(f"No logs yet ({path})")
        return
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        typer.echo(line)
    if not follow:
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        stream.seek(0, os.SEEK_END)
        try:
            while True:
                line = stream.readline()
                if line:
                    typer.echo(line, nl=False)
                else:
                    time.sleep(0.25)
        except KeyboardInterrupt:
            return


@app.command()
@_handled
def doctor() -> None:
    """Diagnose the local environment without downloading models."""

    checks = run_doctor(_config())
    for check in checks:
        typer.echo(f"{check.marker} {check.name}: {check.detail}")
    if has_errors(checks):
        raise typer.Exit(code=1)


def _models_list() -> None:
    config = _config()
    manager = _model_manager(config)
    rows = []
    for model in manager.list_models():
        rows.append(
            [
                model.id,
                model.modality.value,
                model.runtime,
                "yes" if model.installed else "no",
                "yes" if model.available else "no",
                "yes" if model.loaded else "no",
                "partial" if model.partial else model.state.value,
            ]
        )
    _table(
        ["MODEL", "MODALITY", "RUNTIME", "INSTALLED", "AVAILABLE", "LOADED", "STATE"],
        rows,
    )


@models_app.command("list")
@_handled
def models_list() -> None:
    """List all registered models and their distinct lifecycle states."""

    _models_list()


@models_app.command("ls")
@_handled
def models_ls() -> None:
    """Alias for `models list`."""

    _models_list()


@models_app.command("info")
@_handled
def models_info(model_id: str) -> None:
    """Show registry metadata and local state without contacting the network."""

    config = _config()
    manager = _model_manager(config)
    definition = manager.registry.get(model_id)
    view = manager.get_model(model_id)
    values = {
        "name": definition.id,
        "family": definition.family,
        "version": definition.version,
        "modality": definition.modality.value,
        "runtime": definition.runtime,
        "quantization": definition.profile.quantization or "none",
        "source": definition.weights.repository or "not configured",
        "revision": definition.weights.revision or "not configured",
        "install_state": "partial" if view.partial else view.state.value,
        "cache_path": view.cache_path or "not present",
        "estimated_disk_size": (
            f"{definition.weights.estimated_size_gb:g} GB"
            if definition.weights.estimated_size_gb is not None
            else "unknown"
        ),
        "memory_profile": definition.profile.memory or "unknown",
        "runtime_requirements": ", ".join(definition.runtime_requirements) or "none",
        "persistent_loading": str(definition.capabilities.persistent_loading).lower(),
        "progress_reporting": str(definition.capabilities.progress_reporting).lower(),
        "cancellation": str(definition.capabilities.cancellation).lower(),
        "streaming": str(definition.capabilities.streaming).lower(),
        "structured_output": str(definition.capabilities.structured_output).lower(),
    }
    for name, value in values.items():
        typer.echo(f"{name}: {value}")
    if definition.weights.notes:
        typer.echo(f"notes: {definition.weights.notes}")


@models_app.command("install")
@_handled
def models_install(
    model_id: str,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show the plan without network or file changes."),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Confirm a potentially large download."),
    ] = False,
) -> None:
    """Explicitly install model weights into the shared cache."""

    config = _config()
    manager = _model_manager(config)
    plan = manager.install_plan(model_id)
    payload = _jsonable(plan)
    if not isinstance(payload, dict):
        payload = vars(plan)
    typer.echo(f"Model: {model_id}")
    typer.echo(f"Source: {payload.get('repository') or payload.get('source') or 'not configured'}")
    typer.echo(f"Revision: {payload.get('revision') or 'not configured'}")
    size = payload.get("estimated_size_gb")
    typer.echo(
        f"Approximate size: {f'{size:g} GB' if isinstance(size, int | float) else 'unknown'}"
    )
    notes = payload.get("notes")
    if notes:
        typer.echo(f"Note: {notes}")
    typer.echo(f"Downloadable: {'yes' if payload.get('downloadable') else 'no'}")
    required = payload.get("required_files") or ()
    if required:
        typer.echo("Required files:")
        for name in required:
            typer.echo(f"  {name}")
    if dry_run:
        typer.echo("Dry run only; no download was performed.")
        return
    if not yes and not typer.confirm("Proceed with this model download?"):
        typer.echo("Installation cancelled")
        raise typer.Exit()
    _enable_local_logging(config)
    result = manager.install(model_id)
    typer.echo(f"✓ Model installed: {model_id}")
    cache_path = getattr(result, "cache_path", None)
    if cache_path:
        typer.echo(f"Cache: {cache_path}")


@models_app.command("remove")
@_handled
def models_remove(
    model_id: str,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Confirm removal.")] = False,
) -> None:
    """Conservatively remove model data when shared-cache safety is provable."""

    config = _config()
    manager = _model_manager(config)
    view = manager.get_model(model_id)
    definition = manager.registry.get(model_id)
    if not view.installed and not view.partial:
        typer.echo(f"Model '{model_id}' is not installed.")
        return
    typer.echo(f"Model: {model_id}")
    typer.echo(f"Cache path: {view.cache_path or 'unknown'}")
    typer.echo(f"Disk impact: {_model_disk_impact(view.cache_path)}")
    required = definition.weights.required_files
    typer.echo("Expected registered files:")
    if required:
        for name in required:
            typer.echo(f"  {name}")
    else:
        typer.echo("  not configured")
    if not yes and not typer.confirm("Remove this model's local data?"):
        typer.echo("Removal cancelled")
        raise typer.Exit()
    _enable_local_logging(config)
    manager.remove(model_id)
    typer.echo(f"✓ Removed local data for {model_id}")


@models_app.command("load")
@_handled
def models_load(model_id: str) -> None:
    """Load a model only when its runtime supports persistent residency."""

    config = _config()
    _enable_local_logging(config)
    manager = _model_manager(config)
    view = asyncio.run(manager.load(model_id))
    definition = manager.registry.get(model_id)
    if view.loaded:
        typer.echo(f"✓ Model loaded: {model_id}")
    elif not definition.capabilities.persistent_loading:
        typer.echo("This runtime does not support persistent model residency.")
        typer.echo(f"Model '{model_id}' is installed and ready for on-demand inference.")
    else:
        typer.echo(f"Model '{model_id}' is ready.")


@models_app.command("unload")
@_handled
def models_unload(
    model_id: Annotated[str | None, typer.Argument(help="Registered model ID.")] = None,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Unload every persistently loaded model."),
    ] = False,
) -> None:
    """Release persistent model residency where supported."""

    if all_models == (model_id is not None):
        raise DwellError("invalid_request", "Provide one model ID or --all.")
    config = _config()
    _enable_local_logging(config)
    manager = _model_manager(config)
    if all_models:
        asyncio.run(manager.unload_all())
        typer.echo("No persistent models remain loaded.")
    else:
        asyncio.run(manager.unload(model_id))
        definition = manager.registry.get(model_id)
        if definition.capabilities.persistent_loading:
            typer.echo(f"Model '{model_id}' is not loaded.")
        else:
            typer.echo("This runtime has no persistent model residency to unload.")


def _jobs_list() -> None:
    jobs = _job_store(_config()).list(limit=100)
    if not jobs:
        typer.echo("No jobs")
        return
    rows = [
        [
            job.id,
            job.type,
            job.model,
            job.status.value,
            job.created_at.isoformat(timespec="seconds"),
        ]
        for job in jobs
    ]
    _table(["JOB", "TYPE", "MODEL", "STATUS", "CREATED"], rows)


@jobs_app.command("list")
@_handled
def jobs_list() -> None:
    """List recent persisted jobs."""

    _jobs_list()


@jobs_app.command("ls")
@_handled
def jobs_ls() -> None:
    """Alias for `jobs list`."""

    _jobs_list()


@jobs_app.command("show")
@_handled
def jobs_show(job_id: str) -> None:
    """Show a persisted job and its request/result."""

    job: JobRecord = _job_store(_config()).get(job_id)
    payload = job.model_dump(mode="json")
    payload["request"] = job.request
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))


@jobs_app.command("cancel")
@_handled
def jobs_cancel(job_id: str) -> None:
    """Cancel a queued or running job."""

    config = _config()
    current = server_status(config)
    if current["running"] and current.get("api_status") is not None:
        result = api_request(config, f"/v1/jobs/{job_id}", method="DELETE", timeout=12)
        status_value = (
            result.get("status", "cancelled") if isinstance(result, dict) else "cancelled"
        )
    else:
        store = _job_store(config)
        job = store.get(job_id)
        if job.status == JobStatus.QUEUED:
            job = store.cancel_queued(job_id)
            if job.status == JobStatus.RUNNING:
                raise DwellError(
                    "runtime_not_available",
                    (
                        "The job started while cancellation was requested; "
                        "start the API to cancel it safely."
                    ),
                    details={"job_id": job_id},
                    status_code=503,
                )
        elif job.status == JobStatus.RUNNING:
            raise DwellError(
                "runtime_not_available",
                "Cannot safely cancel a running job while the API is unreachable.",
                details={"job_id": job_id},
                status_code=503,
            )
        status_value = job.status.value
    typer.echo(f"Job {job_id}: {status_value}")


@jobs_app.command("clear")
@_handled
def jobs_clear() -> None:
    """Remove completed, failed, and cancelled history only."""

    removed = _job_store(_config()).clear_terminal()
    typer.echo(f"Cleared {removed} terminal job(s). Active work was not changed.")


@outputs_app.command("list")
@_handled
def outputs_list() -> None:
    """List generated media and text without deleting anything."""

    root = _config().outputs_dir
    files = sorted(
        (path for path in root.glob("*/*") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        typer.echo("No outputs")
        return
    rows = []
    for path in files:
        stat = path.stat()
        rows.append(
            [
                path.parent.name,
                path.stem,
                f"{stat.st_size / (1024 * 1024):.1f} MB",
                datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                str(path),
            ]
        )
    _table(["TYPE", "JOB", "SIZE", "MODIFIED", "PATH"], rows)


@config_app.command("show")
@_handled
def config_show() -> None:
    """Show fully resolved configuration."""

    for name, value in _config().display().items():
        typer.echo(f"{name}: {value}")


if __name__ == "__main__":
    app()
