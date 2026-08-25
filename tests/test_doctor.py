from __future__ import annotations

import subprocess
from pathlib import Path

from dwell import doctor as doctor_module
from dwell.config import DwellConfig
from dwell.doctor import CheckLevel, _environment_checks, _runtime_checks


def test_ltx_doctor_help_is_strictly_offline_and_never_syncs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted-python"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "untrusted-venv"))
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "active-venv"))
    runtime = config.runtimes_dir / "ltx-2-mlx"
    (runtime / ".git").mkdir(parents=True)
    (runtime / ".git" / "HEAD").write_text(
        "8ebae0a7cb08312fbf884790b91b4d155e714cdc\n",
        encoding="utf-8",
    )
    (runtime / ".git" / "config").write_text(
        '[remote "origin"]\nurl = https://github.com/xocialize/ltx-2-mlx.git\n',
        encoding="utf-8",
    )
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run(
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 10,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, timeout
        calls.append((command, env))
        if Path(command[0]).name == "git" and command[-3:] == [
            "config",
            "--get",
            "remote.origin.url",
        ]:
            return subprocess.CompletedProcess(
                command,
                0,
                "https://github.com/xocialize/ltx-2-mlx.git\n",
                "",
            )
        if Path(command[0]).name == "git" and command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "8ebae0a7cb08312fbf884790b91b4d155e714cdc\n",
                "",
            )
        if Path(command[0]).name == "git":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "usage: ltx-2-mlx generate\n", "")

    monkeypatch.setattr(doctor_module, "_run", fake_run)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr("dwell.setup._venv_provenance_is_valid", lambda *_args: True)

    checks = _runtime_checks(config)

    assert all(check.level == CheckLevel.OK for check in checks)
    command, environment = calls[-1]
    assert command == [
        str(runtime / ".venv/bin/ltx-2-mlx"),
        "generate",
        "--help",
    ]
    assert environment is not None
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert "PYTHONPATH" not in environment
    assert "UV_PROJECT_ENVIRONMENT" not in environment
    assert "VIRTUAL_ENV" not in environment


def test_untrusted_runtime_executable_is_never_run(tmp_path: Path, monkeypatch) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    runtime = config.runtimes_dir / "ltx-2-mlx"
    (runtime / ".git").mkdir(parents=True)
    (runtime / ".git" / "HEAD").write_text("f" * 40 + "\n", encoding="utf-8")
    (runtime / ".git" / "config").write_text(
        '[remote "origin"]\nurl = https://example.invalid/evil.git\n',
        encoding="utf-8",
    )
    executed_runtime = False

    def fake_run(
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 10,
    ) -> subprocess.CompletedProcess[str]:
        del env, cwd, timeout
        nonlocal executed_runtime
        if Path(command[0]).name == "git" and command[-3:] == [
            "config",
            "--get",
            "remote.origin.url",
        ]:
            return subprocess.CompletedProcess(command, 0, "https://example.invalid/evil.git\n", "")
        if Path(command[0]).name == "git" and command[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, "f" * 40 + "\n", "")
        if Path(command[0]).name == "git":
            return subprocess.CompletedProcess(command, 0, "", "")
        executed_runtime = True
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(doctor_module, "_run", fake_run)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda name: f"/tools/{name}")

    checks = _runtime_checks(config)

    assert executed_runtime is False
    assert any(check.name == "LTX CLI" and "not executed" in check.detail for check in checks)


def test_runtime_without_valid_venv_provenance_is_never_run(tmp_path: Path, monkeypatch) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    runtime = config.runtimes_dir / "ltx-2-mlx"
    (runtime / ".git").mkdir(parents=True)
    (runtime / ".git" / "HEAD").write_text(
        "8ebae0a7cb08312fbf884790b91b4d155e714cdc\n",
        encoding="utf-8",
    )
    (runtime / ".git" / "config").write_text(
        '[remote "origin"]\nurl = https://github.com/xocialize/ltx-2-mlx.git\n',
        encoding="utf-8",
    )
    executed_runtime = False

    def fake_run(
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float = 10,
    ) -> subprocess.CompletedProcess[str]:
        del env, cwd, timeout
        nonlocal executed_runtime
        if Path(command[0]).name == "git":
            return subprocess.CompletedProcess(command, 0, "", "")
        executed_runtime = True
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(doctor_module, "_run", fake_run)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda name: f"/tools/{name}")
    monkeypatch.setattr("dwell.setup._venv_provenance_is_valid", lambda *_args: False)

    checks = _runtime_checks(config)

    assert executed_runtime is False
    assert any(check.name == "LTX CLI" and "provenance" in check.detail for check in checks)


def test_unset_hugging_face_shell_variables_are_healthy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)

    checks = _environment_checks(config)

    assert all(check.level == CheckLevel.OK for check in checks)
    assert all("Dwell injects" in check.detail for check in checks)


def test_explicit_wrong_hugging_face_shell_variables_warn(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    monkeypatch.setenv("HF_HOME", "/tmp/wrong-hf-home")
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/wrong-hf-cache")
    monkeypatch.setenv("HF_XET_CACHE", "/tmp/wrong-xet-cache")

    checks = _environment_checks(config)

    assert all(check.level == CheckLevel.WARNING for check in checks)
