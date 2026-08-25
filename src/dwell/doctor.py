from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from dwell.config import DwellConfig
from dwell.process import api_is_healthy, port_is_available


class CheckLevel(StrEnum):
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    level: CheckLevel
    detail: str

    @property
    def marker(self) -> str:
        return {
            CheckLevel.OK: "✓",
            CheckLevel.WARNING: "!",
            CheckLevel.ERROR: "✗",
        }[self.level]


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 10,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _first_line(value: str) -> str:
    lines = value.strip().splitlines()
    return lines[0] if lines else ""


def _tool_check(name: str, version_args: list[str]) -> DoctorCheck:
    executable = shutil.which(name)
    if executable is None:
        return DoctorCheck(name, CheckLevel.ERROR, "not found")
    result = _run([executable, *version_args])
    version = _first_line(result.stdout or result.stderr)
    if result.returncode:
        return DoctorCheck(name, CheckLevel.ERROR, version or f"exit {result.returncode}")
    return DoctorCheck(name, CheckLevel.OK, version or executable)


def _python_for_mlx(config: DwellConfig, *, runtime_trusted: bool) -> Path:
    runtime_python = config.runtimes_dir / "ltx-2-mlx" / ".venv" / "bin" / "python"
    return runtime_python if runtime_trusted and runtime_python.is_file() else Path(sys.executable)


def _mlx_checks(config: DwellConfig, *, runtime_trusted: bool = False) -> list[DoctorCheck]:
    python = _python_for_mlx(config, runtime_trusted=runtime_trusted)
    script = (
        "import importlib.metadata; import mlx.core as mx; "
        "print(importlib.metadata.version('mlx')); "
        "info=mx.metal.device_info(); print(info.get('device_name', info))"
    )
    result = _run([str(python), "-c", script], env=config.subprocess_env(), timeout=15)
    if result.returncode:
        detail = _first_line(result.stderr) or "not importable"
        return [
            DoctorCheck("MLX", CheckLevel.WARNING, detail),
            DoctorCheck("Metal", CheckLevel.WARNING, "not verified because MLX is unavailable"),
        ]
    lines = result.stdout.strip().splitlines()
    version = lines[0] if lines else "available"
    device = lines[1] if len(lines) > 1 else "Apple Metal device available"
    return [
        DoctorCheck("MLX", CheckLevel.OK, f"{version} via {python}"),
        DoctorCheck("Metal", CheckLevel.OK, device),
    ]


def _mlx_lm_checks(config: DwellConfig) -> list[DoctorCheck]:
    script = "import importlib.metadata; print(importlib.metadata.version('mlx-lm'))"
    result = _run([sys.executable, "-c", script], env=config.subprocess_env(), timeout=15)
    if result.returncode:
        detail = _first_line(result.stderr) or "not importable"
        return [DoctorCheck("MLX-LM runtime", CheckLevel.ERROR, detail)]
    version = _first_line(result.stdout)
    level = CheckLevel.OK if version == "0.31.3" else CheckLevel.ERROR
    detail = f"{version} via {sys.executable}"
    if level == CheckLevel.ERROR:
        detail += "; Dwell expects 0.31.3"
    return [DoctorCheck("MLX-LM runtime", level, detail)]


def _directory_check(config: DwellConfig) -> DoctorCheck:
    try:
        config.ensure_layout()
        with tempfile.NamedTemporaryFile(dir=config.state_dir, prefix="doctor-", delete=True):
            pass
    except OSError as exc:
        return DoctorCheck("Dwell directories", CheckLevel.ERROR, str(exc))
    return DoctorCheck("Dwell directories", CheckLevel.OK, f"writable: {config.home}")


def _port_check(config: DwellConfig) -> DoctorCheck:
    if port_is_available(config):
        return DoctorCheck("API port", CheckLevel.OK, f"{config.host}:{config.port} available")
    if api_is_healthy(config):
        return DoctorCheck("API port", CheckLevel.OK, f"{config.host}:{config.port} used by Dwell")
    return DoctorCheck(
        "API port",
        CheckLevel.ERROR,
        f"{config.host}:{config.port} is occupied by another process",
    )


def _runtime_audit(config: DwellConfig) -> tuple[list[DoctorCheck], bool]:
    runtime = config.runtimes_dir / "ltx-2-mlx"
    if runtime.is_symlink():
        return [DoctorCheck("LTX runtime", CheckLevel.ERROR, "runtime path is a symlink")], False
    if not runtime.exists():
        return [DoctorCheck("LTX runtime", CheckLevel.WARNING, f"missing: {runtime}")], False

    from dwell.setup import (
        SetupManager,
        _venv_provenance_is_valid,
        load_runtime_manifest,
    )

    try:
        spec = next(item for item in load_runtime_manifest() if item.id == "ltx-2-mlx")
    except Exception as exc:
        return [DoctorCheck("LTX runtime manifest", CheckLevel.ERROR, str(exc))], False
    inspection = SetupManager(config, runtime=spec, runner=_run).inspect(probe_venv=False)
    results = [
        DoctorCheck("LTX runtime", CheckLevel.OK, str(runtime)),
    ]
    if spec.uses_archive:
        archive_ok = inspection.remote == spec.archive_url
        checksum_ok = inspection.source_hash == spec.archive_sha256
        results.extend(
            (
                DoctorCheck(
                    "LTX archive",
                    CheckLevel.OK if archive_ok else CheckLevel.ERROR,
                    inspection.remote or "unreadable",
                ),
                DoctorCheck(
                    "LTX commit",
                    CheckLevel.OK if inspection.commit == spec.commit else CheckLevel.ERROR,
                    inspection.commit or "unreadable",
                ),
                DoctorCheck(
                    "LTX archive checksum",
                    CheckLevel.OK if checksum_ok else CheckLevel.ERROR,
                    inspection.source_hash or "unreadable",
                ),
                DoctorCheck(
                    "LTX source tree",
                    CheckLevel.WARNING if inspection.dirty else CheckLevel.OK,
                    "local or unsafe changes present" if inspection.dirty else "clean",
                ),
            )
        )
    else:
        repository = spec.repository or ""
        remote_ok = inspection.remote is not None and inspection.remote.removesuffix(
            "/"
        ) == repository.removesuffix("/")
        results.extend(
            (
                DoctorCheck(
                    "LTX remote",
                    CheckLevel.OK if remote_ok else CheckLevel.ERROR,
                    inspection.remote or "unreadable",
                ),
                DoctorCheck(
                    "LTX commit",
                    CheckLevel.OK if inspection.commit == spec.commit else CheckLevel.ERROR,
                    inspection.commit or "unreadable",
                ),
                DoctorCheck(
                    "LTX working tree",
                    CheckLevel.WARNING if inspection.dirty else CheckLevel.OK,
                    "local changes present" if inspection.dirty else "clean",
                ),
            )
        )

    source_trusted = inspection.source_valid and not inspection.dirty
    if not source_trusted:
        detail = "; ".join(inspection.problems) or "source verification failed"
        results.append(
            DoctorCheck(
                "LTX CLI",
                CheckLevel.ERROR,
                "not executed because the runtime source is not trusted: " + detail,
            )
        )
        return results, False

    if not _venv_provenance_is_valid(runtime, spec):
        results.append(
            DoctorCheck(
                "LTX CLI",
                CheckLevel.ERROR,
                "runtime venv provenance is missing or stale; run dwell setup --repair",
            )
        )
        return results, False

    cli = runtime / ".venv" / "bin" / "ltx-2-mlx"
    help_result = _run(
        [
            str(cli),
            "generate",
            "--help",
        ],
        env=config.subprocess_env(),
        cwd=runtime,
        timeout=30,
    )
    if help_result.returncode:
        detail = _first_line(help_result.stderr) or f"exit {help_result.returncode}"
        results.append(DoctorCheck("LTX CLI", CheckLevel.ERROR, detail))
    else:
        results.append(DoctorCheck("LTX CLI", CheckLevel.OK, "offline help check passed"))
    return results, True


def _runtime_checks(config: DwellConfig) -> list[DoctorCheck]:
    checks, _trusted = _runtime_audit(config)
    return checks


def _environment_checks(config: DwellConfig) -> list[DoctorCheck]:
    results: list[DoctorCheck] = []
    expected = {
        "HF_HOME": str(config.hf_home),
        "HF_HUB_CACHE": str(config.hf_hub_cache),
        "HF_XET_CACHE": str(config.hf_xet_cache),
    }
    for name, desired in expected.items():
        actual = os.environ.get(name)
        if actual == desired:
            results.append(DoctorCheck(name, CheckLevel.OK, actual))
        elif actual:
            results.append(
                DoctorCheck(name, CheckLevel.WARNING, f"current={actual}; expected={desired}")
            )
        else:
            results.append(
                DoctorCheck(
                    name,
                    CheckLevel.OK,
                    f"not set in this shell; Dwell injects {desired} into child processes",
                )
            )
    return results


def _registry_checks(config: DwellConfig) -> list[DoctorCheck]:
    try:
        from dwell.model_manager import ModelManager
        from dwell.registry import ModelRegistry

        registry = ModelRegistry.load(config)
        models = registry.list()
        manager = ModelManager(config, registry=registry)
        views = manager.list_models()
    except Exception as exc:  # registry parse errors should become a doctor result
        return [DoctorCheck("Model registry", CheckLevel.ERROR, str(exc))]

    results = [DoctorCheck("Model registry", CheckLevel.OK, f"{len(models)} registered model(s)")]
    for view in views:
        if view.installed:
            level = CheckLevel.OK
            detail = "installed"
        elif view.partial:
            level = CheckLevel.WARNING
            detail = "partial download; not installed"
        else:
            level = CheckLevel.WARNING
            detail = "not installed"
        results.append(DoctorCheck(view.id, level, detail))
    return results


def run_doctor(config: DwellConfig | None = None) -> list[DoctorCheck]:
    config = config or DwellConfig.from_env()
    checks: list[DoctorCheck] = []
    runtime_checks, runtime_trusted = _runtime_audit(config)

    system = platform.system()
    machine = platform.machine()
    if system == "Darwin" and machine == "arm64":
        macos_version = platform.mac_ver()[0]
        try:
            supported_macos = int(macos_version.split(".", 1)[0]) >= 14
        except (TypeError, ValueError):
            supported_macos = False
        checks.append(
            DoctorCheck(
                "Platform",
                CheckLevel.OK if supported_macos else CheckLevel.ERROR,
                f"macOS {macos_version} arm64"
                if supported_macos
                else (
                    f"macOS {macos_version or 'unknown'} arm64; requires macOS 14 Sonoma or newer"
                ),
            )
        )
    else:
        checks.append(DoctorCheck("Platform", CheckLevel.ERROR, f"{system} {machine}"))

    python_version = platform.python_version()
    python_level = CheckLevel.OK if sys.version_info >= (3, 11) else CheckLevel.ERROR
    checks.append(DoctorCheck("Python", python_level, f"{python_version} ({sys.executable})"))
    checks.extend(
        (
            _tool_check("uv", ["--version"]),
            _tool_check("ffmpeg", ["-version"]),
            _tool_check("ffprobe", ["-version"]),
        )
    )
    checks.extend(_mlx_checks(config, runtime_trusted=runtime_trusted))
    checks.extend(_mlx_lm_checks(config))
    checks.extend(_environment_checks(config))
    checks.append(_directory_check(config))
    checks.append(_port_check(config))
    checks.extend(runtime_checks)
    checks.extend(_registry_checks(config))
    return checks


def has_errors(checks: list[DoctorCheck]) -> bool:
    return any(check.level == CheckLevel.ERROR for check in checks)
