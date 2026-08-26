from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from typer.testing import CliRunner

from dwell import cli as cli_module
from dwell.cli import app
from dwell.setup import SetupMode, SetupReport

runner = CliRunner()


def _env(home: Path) -> dict[str, str]:
    return {
        "DWELL_HOME": str(home),
        "DWELL_HOST": "127.0.0.1",
        "DWELL_PORT": "18189",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def test_help_and_command_tree() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("start", "stop", "setup", "doctor", "models", "jobs", "outputs", "config"):
        assert command in result.stdout

    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout == "Dwell 0.1.0\n"


def test_config_status_jobs_and_outputs_without_server(tmp_path: Path) -> None:
    env = _env(tmp_path / "dwell")

    result = runner.invoke(app, ["config", "show"], env=env)
    assert result.exit_code == 0
    assert f"home: {tmp_path / 'dwell'}" in result.stdout
    assert "api_host: 127.0.0.1" in result.stdout

    result = runner.invoke(app, ["status"], env=env)
    assert result.exit_code == 0
    assert "Dwell: stopped" in result.stdout
    assert "Queued jobs: 0" in result.stdout

    result = runner.invoke(app, ["jobs", "list"], env=env)
    assert result.exit_code == 0
    assert "No jobs" in result.stdout

    result = runner.invoke(app, ["outputs", "list"], env=env)
    assert result.exit_code == 0
    assert "No outputs" in result.stdout


def test_model_list_info_and_install_dry_runs_are_offline(tmp_path: Path) -> None:
    env = _env(tmp_path / "dwell")

    result = runner.invoke(app, ["models", "list"], env=env)
    assert result.exit_code == 0, result.output
    assert "ltx-2.5-bf16" in result.stdout
    assert "ltx-2.5-q8" in result.stdout
    assert "qwen3-coder-30b-a3b-4bit" in result.stdout
    assert "no" in result.stdout

    result = runner.invoke(app, ["models", "info", "ltx-2.5-q8"], env=env)
    assert result.exit_code == 0, result.output
    assert "source: not configured" in result.stdout
    assert "install_state: not_installed" in result.stdout

    result = runner.invoke(
        app,
        ["models", "install", "ltx-2.5-q8", "--dry-run"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "Source: not configured" in result.stdout
    assert "no download was performed" in result.stdout

    result = runner.invoke(
        app,
        ["models", "install", "qwen3-coder-30b-a3b-4bit", "--dry-run"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    assert "Source: mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit" in result.stdout
    assert "Revision: 6e302ea604ad9ab206367e2c501d1571023e7b6d" in result.stdout
    assert "Approximate size: 17.2 GB" in result.stdout
    assert "Downloadable: yes" in result.stdout
    assert "no download was performed" in result.stdout
    assert not (tmp_path / "dwell").exists()


def test_bf16_install_plan_shows_capacity_and_legal_notices_without_downloading(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        ["models", "install", "ltx-2.5-bf16", "--dry-run"],
        env=_env(tmp_path / "dwell"),
    )

    assert result.exit_code == 0, result.output
    assert "Approximate size: 71 GB" in result.stdout
    assert "License: https://" in result.stdout
    assert "Acceptable use: https://" in result.stdout
    assert "Disk:" in result.stdout or "Disk warning:" in result.stdout
    assert "Memory:" in result.stdout or "Memory warning:" in result.stdout
    assert "no download was performed" in result.stdout
    assert not (tmp_path / "dwell").exists()


def test_setup_cli_selects_read_only_check_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[SetupMode] = []

    class FakeSetupManager:
        def __init__(self, _config) -> None:
            pass

        def run(self, mode: SetupMode, *, force: bool = False) -> SetupReport:
            assert force is False
            calls.append(mode)
            return SetupReport(mode, False, True, ("Runtime: healthy",))

    monkeypatch.setattr("dwell.setup.SetupManager", FakeSetupManager)
    result = runner.invoke(app, ["setup", "--check"], env=_env(tmp_path / "dwell"))

    assert result.exit_code == 0, result.output
    assert calls == [SetupMode.CHECK]
    assert "Runtime: healthy" in result.stdout
    assert "Setup is healthy" in result.stdout


def test_setup_cli_rejects_conflicting_modes_without_creating_home(tmp_path: Path) -> None:
    home = tmp_path / "dwell"
    result = runner.invoke(app, ["setup", "--check", "--repair"], env=_env(home))

    assert result.exit_code == 1
    assert "Use only one" in result.output
    assert not home.exists()


def test_setup_cli_rejects_force_without_upgrade(tmp_path: Path) -> None:
    home = tmp_path / "dwell"
    result = runner.invoke(app, ["setup", "--force"], env=_env(home))

    assert result.exit_code == 1
    assert "--force may only be used with --upgrade" in result.output
    assert not home.exists()


def test_missing_model_error_is_concise(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["models", "info", "does-not-exist"],
        env=_env(tmp_path / "dwell"),
    )

    assert result.exit_code == 1
    assert "not registered" in result.output
    assert "Traceback" not in result.output


def test_persistent_model_load_and_unload_are_owned_by_running_daemon(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_id = "qwen3-coder-30b-a3b-4bit"
    definition = SimpleNamespace(
        capabilities=SimpleNamespace(persistent_loading=True),
    )
    manager = SimpleNamespace(
        registry=SimpleNamespace(get=lambda requested: definition),
    )
    requests: list[tuple[str, str]] = []

    def fake_api_request(
        _config,
        path: str,
        *,
        method: str = "GET",
        timeout: float = 10,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        del timeout
        requests.append((method, path))
        return {
            "id": model_id,
            "family": "qwen3-coder",
            "version": "test",
            "modality": "text",
            "runtime": "mlx-lm",
            "quantization": "4bit",
            "registered": True,
            "installed": True,
            "available": True,
            "loaded": True,
            "state": "loaded",
            "partial": False,
            "cache_path": str(tmp_path / "snapshot"),
        }

    monkeypatch.setattr(cli_module, "_model_manager", lambda _config: manager)
    monkeypatch.setattr(
        cli_module,
        "server_status",
        lambda _config: {"running": True, "api_status": {"server": "running"}},
    )
    monkeypatch.setattr(cli_module, "api_request", fake_api_request)
    env = _env(tmp_path / "dwell")

    loaded = runner.invoke(app, ["models", "load", model_id], env=env)
    unloaded = runner.invoke(app, ["models", "unload", model_id], env=env)

    assert loaded.exit_code == 0, loaded.output
    assert f"Model loaded: {model_id}" in loaded.stdout
    assert unloaded.exit_code == 0, unloaded.output
    assert requests == [
        ("POST", f"/v1/models/{model_id}/load"),
        ("DELETE", f"/v1/models/{model_id}/load"),
    ]
