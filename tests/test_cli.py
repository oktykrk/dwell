from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from dwell.cli import app

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
    for command in ("start", "stop", "doctor", "models", "jobs", "outputs", "config"):
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


def test_model_list_info_and_unconfigured_q8_dry_run_are_offline(tmp_path: Path) -> None:
    env = _env(tmp_path / "dwell")

    result = runner.invoke(app, ["models", "list"], env=env)
    assert result.exit_code == 0, result.output
    assert "ltx-2.5-bf16" in result.stdout
    assert "ltx-2.5-q8" in result.stdout
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
    assert not (tmp_path / "dwell").exists()


def test_missing_model_error_is_concise(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["models", "info", "does-not-exist"],
        env=_env(tmp_path / "dwell"),
    )

    assert result.exit_code == 1
    assert "not registered" in result.output
    assert "Traceback" not in result.output
