from __future__ import annotations

import subprocess
from pathlib import Path

from dwell import doctor as doctor_module
from dwell.config import DwellConfig
from dwell.doctor import CheckLevel, _runtime_checks


def test_ltx_doctor_help_is_strictly_offline_and_never_syncs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell")
    runtime = config.runtimes_dir / "ltx-2-mlx"
    (runtime / ".git").mkdir(parents=True)
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
        if command[:3] == ["git", "-C", str(runtime)]:
            return subprocess.CompletedProcess(command, 0, "ltx-2.5\n", "")
        return subprocess.CompletedProcess(command, 0, "usage: ltx-2-mlx generate\n", "")

    monkeypatch.setattr(doctor_module, "_run", fake_run)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda name: f"/tools/{name}")

    checks = _runtime_checks(config)

    assert all(check.level == CheckLevel.OK for check in checks)
    command, environment = calls[-1]
    assert command == [
        "/tools/uv",
        "run",
        "--project",
        str(runtime),
        "--offline",
        "--no-sync",
        "--no-python-downloads",
        "ltx-2-mlx",
        "generate",
        "--help",
    ]
    assert environment is not None
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
