from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from dwell import setup as dwell_setup_module
from dwell.config import DwellConfig
from dwell.doctor import CheckLevel, DoctorCheck
from dwell.errors import DwellError
from dwell.setup import RuntimeSpec, SetupManager, SetupMode, load_runtime_manifest

_REAL_GIT = shutil.which("git") or "git"


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        [_REAL_GIT, "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _commit(repository: Path, name: str, contents: str) -> str:
    (repository / name).write_text(contents, encoding="utf-8")
    _git(repository, "add", name)
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": "Dwell Test",
            "GIT_AUTHOR_EMAIL": "dwell@example.invalid",
            "GIT_COMMITTER_NAME": "Dwell Test",
            "GIT_COMMITTER_EMAIL": "dwell@example.invalid",
        }
    )
    result = subprocess.run(
        [_REAL_GIT, "-C", str(repository), "commit", "-m", name],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return _git(repository, "rev-parse", "HEAD")


def _source_repo(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "source"
    result = subprocess.run(
        [_REAL_GIT, "init", "-b", "ltx-2.5", str(repository)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (repository / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (repository / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repository, "add", ".gitignore", "uv.lock")
    commit = _commit(repository, "runtime.txt", "one\n")
    return repository, commit


def _spec(repository: Path, commit: str) -> RuntimeSpec:
    return RuntimeSpec(
        id="ltx-2-mlx",
        repository=str(repository),
        branch="ltx-2.5",
        commit=commit,
        python="3.11",
    )


def _doctor_ok(_config: DwellConfig) -> list[DoctorCheck]:
    return [DoctorCheck("test doctor", CheckLevel.OK, "ok")]


class FakeRunner:
    def __init__(
        self,
        *,
        fail_sync: bool = False,
        fail_tool: str | None = None,
        on_sync: Callable[[Path], None] | None = None,
    ) -> None:
        self.fail_sync = fail_sync
        self.fail_tool = fail_tool
        self.on_sync = on_sync
        self.commands: list[tuple[list[str], Path | None, dict[str, str] | None]] = []
        self.sync_directories: list[Path] = []

    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env=None,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        environment = dict(env) if env is not None else None
        self.commands.append((command, cwd, environment))
        tool = Path(command[0]).name
        if tool in {"uv", "git", "ffmpeg", "ffprobe"} and command[1:] in (
            ["--version"],
            ["-version"],
        ):
            if tool == self.fail_tool:
                return subprocess.CompletedProcess(command, 9, "", "simulated broken tool")
            return subprocess.CompletedProcess(command, 0, f"{tool} test-version\n", "")
        if Path(command[0]).name == "uv" and command[1:2] == ["sync"]:
            assert cwd is not None
            assert command == [
                command[0],
                "sync",
                "--project",
                str(cwd),
                "--locked",
                "--python",
                "3.11",
            ]
            assert environment is not None
            assert environment["UV_PROJECT"] == str(cwd)
            assert environment["UV_PROJECT_ENVIRONMENT"] == str(cwd / ".venv")
            assert environment["HF_XET_CACHE"] == str(cwd.parent.parent / "models/huggingface/xet")
            self.sync_directories.append(cwd)
            if self.on_sync is not None:
                self.on_sync(cwd)
            if self.fail_sync:
                return subprocess.CompletedProcess(command, 1, "", "simulated sync failure")
            bin_dir = cwd / ".venv" / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            python = bin_dir / "python"
            cli = bin_dir / "ltx-2-mlx"
            (cwd / ".venv" / "pyvenv.cfg").write_text(
                "implementation = CPython\nversion_info = 3.11.99\n",
                encoding="utf-8",
            )
            python.write_text("#!/bin/sh\necho 'Python 3.11.99'\n", encoding="utf-8")
            # Match distlib's safe launcher for paths that contain spaces or
            # exceed the kernel's shebang-length limit.
            launcher = f"#!/bin/sh\n'''exec' '{python}' \"$0\" \"$@\"\n' '''\n"
            cli.write_text(
                launcher
                + "# generated console script\n"
                + "from ltx_pipelines_mlx.cli import main\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            cli.chmod(0o755)
            return subprocess.CompletedProcess(command, 0, "synced\n", "")
        return dwell_setup_module._run(command, cwd=cwd, env=environment, timeout=timeout)


@pytest.fixture(autouse=True)
def supported_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dwell_setup_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(dwell_setup_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        dwell_setup_module.shutil,
        "which",
        lambda name: _REAL_GIT if name == "git" else f"/test-tools/{name}",
    )


def _manager(
    config: DwellConfig,
    spec: RuntimeSpec,
    runner: FakeRunner | None = None,
) -> tuple[SetupManager, FakeRunner]:
    command_runner = runner or FakeRunner()
    return (
        SetupManager(
            config,
            runtime=spec,
            runner=command_runner,
            doctor_runner=_doctor_ok,
        ),
        command_runner,
    )


def test_packaged_runtime_manifest_is_immutable() -> None:
    (runtime,) = load_runtime_manifest()

    assert runtime.id == "ltx-2-mlx"
    assert runtime.branch == "main"
    assert runtime.commit == "8ebae0a7cb08312fbf884790b91b4d155e714cdc"
    assert runtime.python == "3.11"


def test_clean_setup_installs_source_then_syncs_only_at_final_path(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))

    report = manager.run()

    assert report.changed is True
    assert report.healthy is True
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == commit
    assert runner.sync_directories == [manager.runtime_dir]
    assert (manager.runtime_dir / ".venv/bin/ltx-2-mlx").is_file()
    state = json.loads(config.setup_state_file.read_text(encoding="utf-8"))
    assert state["dwell_version"] == "0.1.0"
    assert state["runtime"]["commit"] == commit
    assert not any(config.models_dir.rglob("*.safetensors"))


def test_second_setup_is_runtime_and_state_noop(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    state_before = config.setup_state_file.read_bytes()

    report = manager.run()

    assert report.changed is False
    assert len(runner.sync_directories) == 1
    assert config.setup_state_file.read_bytes() == state_before

    upgraded = manager.run(SetupMode.UPGRADE)
    assert upgraded.changed is False
    assert upgraded.healthy is True
    assert len(runner.sync_directories) == 1


def test_healthy_setup_is_noop_while_server_is_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()

    import dwell.process as process_module

    monkeypatch.setattr(
        process_module,
        "read_server_state",
        lambda *_args, **_kwargs: SimpleNamespace(pid=123),
    )

    report = manager.run()

    assert report.changed is False
    assert report.healthy is True
    assert len(runner.sync_directories) == 1


def test_setup_supports_operational_home_with_spaces(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home with spaces")
    manager, _runner = _manager(config, _spec(source, commit))

    first = manager.run()
    checked = manager.run(SetupMode.CHECK)

    assert first.healthy is True
    assert checked.healthy is True


def test_check_is_read_only_when_setup_is_missing(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "missing-home")
    manager, runner = _manager(config, _spec(source, commit))

    report = manager.run(SetupMode.CHECK)

    assert report.changed is False
    assert report.healthy is False
    assert not config.home.exists()
    assert all(
        command[1:] in (["--version"], ["-version"])
        for command, _cwd, _environment in runner.commands
    )


def test_check_does_not_modify_an_installed_setup(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    before = {
        path.relative_to(config.home): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in config.home.rglob("*")
        if path.is_file()
    }
    sync_count = len(runner.sync_directories)
    command_count = len(runner.commands)

    report = manager.run(SetupMode.CHECK)

    after = {
        path.relative_to(config.home): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in config.home.rglob("*")
        if path.is_file()
    }
    assert report.changed is False
    assert report.healthy is True
    assert after == before
    assert len(runner.sync_directories) == sync_count
    check_commands = runner.commands[command_count:]
    assert not any(
        command[0].startswith(str(manager.runtime_dir / ".venv"))
        for command, _cwd, _environment in check_commands
    )


def test_check_never_executes_checkout_git_filters_or_fsmonitor(tmp_path: Path) -> None:
    source, _first_commit = _source_repo(tmp_path)
    commit = _commit(source, ".gitattributes", "runtime.txt filter=audit\n")
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    filter_marker = tmp_path / "filter-executed"
    fsmonitor_marker = tmp_path / "fsmonitor-executed"
    _git(
        manager.runtime_dir,
        "config",
        "filter.audit.clean",
        f"sh -c 'touch {filter_marker}; cat'",
    )
    _git(
        manager.runtime_dir,
        "config",
        "core.fsmonitor",
        f"sh -c 'touch {fsmonitor_marker}'",
    )
    command_count = len(runner.commands)

    report = manager.run(SetupMode.CHECK)

    assert report.healthy is True
    assert not filter_marker.exists()
    assert not fsmonitor_marker.exists()
    assert not any(
        "status" in command for command, _cwd, _environment in runner.commands[command_count:]
    )


def test_wrong_platform_and_missing_tool_fail_before_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, _runner = _manager(config, _spec(source, commit))
    monkeypatch.setattr(dwell_setup_module.platform, "machine", lambda: "x86_64")

    with pytest.raises(DwellError, match="requires macOS on Apple Silicon"):
        manager.run()
    assert not config.home.exists()

    monkeypatch.setattr(dwell_setup_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(dwell_setup_module.shutil, "which", lambda _name: None)
    with pytest.raises(DwellError, match="Required setup tools are missing"):
        manager.run()
    assert not config.home.exists()


def test_unusable_tool_fails_preflight_before_layout(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit), FakeRunner(fail_tool="ffprobe"))

    with pytest.raises(DwellError, match="missing or unusable: ffprobe"):
        manager.run()

    assert not config.home.exists()
    assert not runner.sync_directories


@pytest.mark.parametrize("mismatch", ["remote", "commit"])
def test_normal_setup_refuses_wrong_remote_or_commit(tmp_path: Path, mismatch: str) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    sentinel = first_manager.runtime_dir / "runtime.txt"

    if mismatch == "remote":
        _git(first_manager.runtime_dir, "remote", "set-url", "origin", str(tmp_path / "other"))
        target = first_commit
    else:
        target = _commit(source, "runtime.txt", "two\n")

    manager, _runner = _manager(config, _spec(source, target))
    with pytest.raises(DwellError, match="use dwell setup --upgrade"):
        manager.run()

    assert sentinel.read_text(encoding="utf-8") == "one\n"


def test_untrusted_checkout_is_rejected_without_executing_its_venv(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    _git(manager.runtime_dir, "remote", "set-url", "origin", str(tmp_path / "untrusted"))
    runner.commands.clear()

    with pytest.raises(DwellError, match="does not match the manifest"):
        manager.run()

    assert not any(
        command[0].startswith(str(manager.runtime_dir / ".venv"))
        for command, _cwd, _environment in runner.commands
    )


def test_raw_remote_identity_ignores_git_url_rewrites(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, _runner = _manager(config, _spec(source, commit))
    manager.run()
    _git(
        manager.runtime_dir,
        "config",
        "url.file:///rewritten/.insteadOf",
        str(source),
    )
    assert _git(manager.runtime_dir, "remote", "get-url", "origin") != str(source)

    report = manager.run()

    assert report.changed is False
    assert report.healthy is True


def test_dangling_venv_symlink_is_refused_without_uv_mutation(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    config.ensure_layout()
    clone = subprocess.run(
        [_REAL_GIT, "clone", str(source), str(config.runtimes_dir / "ltx-2-mlx")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert clone.returncode == 0, clone.stderr
    runtime = config.runtimes_dir / "ltx-2-mlx"
    _git(runtime, "checkout", "--detach", commit)
    (runtime / ".venv").symlink_to(tmp_path / "missing-venv", target_is_directory=True)
    manager, runner = _manager(config, _spec(source, commit))

    with pytest.raises(DwellError, match="symlinked runtime virtual environment"):
        manager.run()

    assert (runtime / ".venv").is_symlink()
    assert not runner.sync_directories


def test_uv_sync_pins_project_and_venv_despite_ambient_uv_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    monkeypatch.setenv("UV_PROJECT", str(tmp_path / "outside-project"))
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", str(tmp_path / "outside-venv"))
    monkeypatch.setenv("UV_CONFIG_FILE", str(tmp_path / "outside.toml"))
    monkeypatch.setenv("UV_NO_INSTALL_PROJECT", "1")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "active-venv"))

    manager.run()

    sync = next(item for item in runner.commands if item[0][1:2] == ["sync"])
    command, cwd, environment = sync
    assert cwd == manager.runtime_dir
    assert command[2:4] == ["--project", str(manager.runtime_dir)]
    assert environment is not None
    assert environment["UV_PROJECT"] == str(manager.runtime_dir)
    assert environment["UV_PROJECT_ENVIRONMENT"] == str(manager.runtime_dir / ".venv")
    assert "UV_CONFIG_FILE" not in environment
    assert "UV_NO_INSTALL_PROJECT" not in environment
    assert "VIRTUAL_ENV" not in environment


def test_dirty_runtime_is_never_overwritten(tmp_path: Path) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    user_file = first_manager.runtime_dir / "user-change.txt"
    user_file.write_text("keep me\n", encoding="utf-8")
    second_commit = _commit(source, "runtime.txt", "two\n")

    manager, _runner = _manager(config, _spec(source, second_commit))
    with pytest.raises(DwellError, match="refused to overwrite"):
        manager.run(SetupMode.UPGRADE)

    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit


def test_partial_install_completes_missing_venv(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    config.ensure_layout()
    _git_clone = subprocess.run(
        [_REAL_GIT, "clone", str(source), str(config.runtimes_dir / "ltx-2-mlx")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert _git_clone.returncode == 0, _git_clone.stderr
    _git(config.runtimes_dir / "ltx-2-mlx", "checkout", "--detach", commit)
    manager, runner = _manager(config, _spec(source, commit))

    report = manager.run()

    assert report.changed is True
    assert runner.sync_directories == [manager.runtime_dir]
    assert (manager.runtime_dir / ".venv/bin/python").is_file()


def test_repair_rebuilds_an_unhealthy_venv(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    cli = manager.runtime_dir / ".venv/bin/ltx-2-mlx"
    cli.write_text("#!/bin/sh\nexit 9\n", encoding="utf-8")
    cli.chmod(0o755)

    report = manager.run(SetupMode.REPAIR)

    assert report.changed is True
    assert len(runner.sync_directories) == 2
    assert subprocess.run([str(cli), "--help"], check=False).returncode == 0


def test_setup_never_executes_a_venv_after_its_provenance_changes(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, runner = _manager(config, _spec(source, commit))
    manager.run()
    injected = manager.runtime_dir / ".venv/lib/python3.11/site-packages/sitecustomize.pyc"
    injected.parent.mkdir(parents=True)
    injected.write_bytes(b"untrusted sourceless bytecode")
    command_count = len(runner.commands)

    with pytest.raises(DwellError, match="run dwell setup --repair"):
        manager.run()

    assert not any(
        command[0].startswith(str(manager.runtime_dir / ".venv"))
        for command, _cwd, _environment in runner.commands[command_count:]
    )

    repaired = manager.run(SetupMode.REPAIR)
    assert repaired.healthy is True
    assert not injected.exists()


def test_provenance_hash_detects_same_size_and_mtime_file_tampering(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, _runner = _manager(config, _spec(source, commit))
    manager.run()
    metadata = manager.runtime_dir / ".venv/pyvenv.cfg"
    original = metadata.read_text(encoding="utf-8")
    original_stat = metadata.stat()
    modified = original.replace("CPython", "XPython")
    assert len(modified) == len(original)
    metadata.write_text(modified, encoding="utf-8")
    os.utime(metadata, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(DwellError, match="run dwell setup --repair"):
        manager.run()


def test_provenance_hash_detects_python_symlink_target_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    manager, _runner = _manager(config, _spec(source, commit))
    external_python = tmp_path / "managed-python"

    def establish_python_symlink(runtime: Path) -> bool:
        python = runtime / ".venv/bin/python"
        external_python.write_bytes(python.read_bytes())
        external_python.chmod(0o755)
        python.unlink()
        python.symlink_to(external_python)
        return True

    monkeypatch.setattr(manager, "_venv_probe_passes", establish_python_symlink)
    manager.run()
    original = external_python.read_bytes()
    original_stat = external_python.stat()
    modified = original.replace(b"3.11.99", b"3.11.98")
    assert len(modified) == len(original)
    external_python.write_bytes(modified)
    external_python.chmod(0o755)
    os.utime(external_python, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    with pytest.raises(DwellError, match="run dwell setup --repair"):
        manager.run()


def test_partial_failed_sync_is_retried_by_normal_setup(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")

    def leave_partial_venv(runtime: Path) -> None:
        partial = runtime / ".venv" / "partial.txt"
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text("incomplete\n", encoding="utf-8")

    failed_manager, _failed_runner = _manager(
        config,
        _spec(source, commit),
        FakeRunner(fail_sync=True, on_sync=leave_partial_venv),
    )
    with pytest.raises(DwellError, match="uv sync --locked failed"):
        failed_manager.run()
    assert (failed_manager.runtime_dir / ".venv/partial.txt").is_file()
    assert not config.setup_state_file.exists()

    manager, runner = _manager(config, _spec(source, commit))
    report = manager.run()

    assert report.healthy is True
    assert runner.sync_directories == [manager.runtime_dir]
    assert not (manager.runtime_dir / ".venv/partial.txt").exists()


def test_upgrade_swaps_source_and_preserves_user_data(tmp_path: Path) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    model = config.models_dir / "keep.safetensors"
    output = config.outputs_dir / "video" / "keep.mp4"
    model.write_bytes(b"model")
    output.write_bytes(b"video")
    second_commit = _commit(source, "runtime.txt", "two\n")
    manager, runner = _manager(config, _spec(source, second_commit))

    report = manager.run(SetupMode.UPGRADE)

    assert report.changed is True
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == second_commit
    assert (manager.runtime_dir / "runtime.txt").read_text(encoding="utf-8") == "two\n"
    assert runner.sync_directories == [manager.runtime_dir]
    assert model.read_bytes() == b"model"
    assert output.read_bytes() == b"video"
    assert not list(config.runtimes_dir.glob(".ltx-2-mlx.backup-*"))


def test_failed_upgrade_restores_previous_runtime(tmp_path: Path) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    second_commit = _commit(source, "runtime.txt", "two\n")
    manager, _runner = _manager(config, _spec(source, second_commit), FakeRunner(fail_sync=True))

    with pytest.raises(DwellError, match="previous runtime was restored"):
        manager.run(SetupMode.UPGRADE)

    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert (manager.runtime_dir / ".venv/bin/ltx-2-mlx").is_file()
    assert not list(config.runtimes_dir.glob(".ltx-2-mlx.backup-*"))


def test_doctor_failure_rolls_back_upgrade_before_backup_cleanup(tmp_path: Path) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    state_before = config.setup_state_file.read_bytes()
    second_commit = _commit(source, "runtime.txt", "two\n")
    backup_seen = False

    def doctor_fails(_config: DwellConfig) -> list[DoctorCheck]:
        nonlocal backup_seen
        backup_seen = bool(list(config.runtimes_dir.glob(".ltx-2-mlx.backup-*")))
        return [DoctorCheck("runtime integration", CheckLevel.ERROR, "simulated doctor failure")]

    manager = SetupManager(
        config,
        runtime=_spec(source, second_commit),
        runner=FakeRunner(),
        doctor_runner=doctor_fails,
    )

    with pytest.raises(DwellError, match="previous runtime was restored"):
        manager.run(SetupMode.UPGRADE)

    assert backup_seen is True
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert config.setup_state_file.read_bytes() == state_before
    assert not list(config.runtimes_dir.glob(".ltx-2-mlx.backup-*"))
    assert not list(config.runtimes_dir.glob(".ltx-2-mlx.failed-*"))


def test_upgrade_preserves_edit_created_during_staging_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    second_commit = _commit(source, "runtime.txt", "two\n")
    manager, _runner = _manager(config, _spec(source, second_commit))
    original_clone = manager._clone_staging
    user_file = manager.runtime_dir / "created-during-clone.txt"

    def clone_then_edit() -> Path:
        staging = original_clone()
        user_file.write_text("keep me\n", encoding="utf-8")
        return staging

    monkeypatch.setattr(manager, "_clone_staging", clone_then_edit)

    with pytest.raises(DwellError, match="changed while upgrade"):
        manager.run(SetupMode.UPGRADE)

    assert user_file.read_text(encoding="utf-8") == "keep me\n"
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert not list(config.tmp_dir.glob("ltx-2-mlx-*"))


def test_upgrade_rolls_back_when_backup_is_edited_during_sync(tmp_path: Path) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    second_commit = _commit(source, "runtime.txt", "two\n")

    def edit_backup(_runtime: Path) -> None:
        (next(config.runtimes_dir.glob(".ltx-2-mlx.backup-*")) / "late-edit.txt").write_text(
            "keep me\n",
            encoding="utf-8",
        )

    manager, _runner = _manager(
        config,
        _spec(source, second_commit),
        FakeRunner(on_sync=edit_backup),
    )

    with pytest.raises(DwellError, match="previous runtime was preserved"):
        manager.run(SetupMode.UPGRADE)

    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert (manager.runtime_dir / "late-edit.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not list(config.runtimes_dir.glob(".ltx-2-mlx.backup-*"))


def test_upgrade_rechecks_active_work_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    second_commit = _commit(source, "runtime.txt", "two\n")
    manager, _runner = _manager(config, _spec(source, second_commit))
    calls = 0

    def refuse_after_staging() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise DwellError("server_busy", "server started during staging", status_code=409)

    monkeypatch.setattr(manager, "_refuse_active_work", refuse_after_staging)

    with pytest.raises(DwellError, match="server started during staging"):
        manager.run(SetupMode.UPGRADE)

    assert calls == 2
    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert not list(config.tmp_dir.glob("ltx-2-mlx-*"))


@pytest.mark.parametrize("active_kind", ["server", "job"])
def test_upgrade_refuses_active_server_or_ltx_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_kind: str,
) -> None:
    source, first_commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    first_manager, _runner = _manager(config, _spec(source, first_commit))
    first_manager.run()
    second_commit = _commit(source, "runtime.txt", "two\n")
    manager, runner = _manager(config, _spec(source, second_commit))

    import dwell.process as process_module

    if active_kind == "server":
        monkeypatch.setattr(
            process_module,
            "read_server_state",
            lambda *_args, **_kwargs: SimpleNamespace(pid=123),
        )
    else:
        monkeypatch.setattr(process_module, "read_server_state", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            process_module,
            "_read_ltx_child_state",
            lambda _config: {"pid": 456, "job_id": "job-1"},
        )
        monkeypatch.setattr(process_module, "_recorded_ltx_group_may_be_active", lambda _p: True)

    with pytest.raises(DwellError, match="Stop Dwell|may still be active"):
        manager.run(SetupMode.UPGRADE)

    assert _git(manager.runtime_dir, "rev-parse", "HEAD") == first_commit
    assert not any("clone" in command for command, _cwd, _env in runner.commands)


@pytest.mark.parametrize("lock_name", ["start_lock_file", "ltx_lock_file"])
def test_runtime_mutation_lock_collision_is_reported(tmp_path: Path, lock_name: str) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    config.ensure_layout()
    manager, runner = _manager(config, _spec(source, commit))
    path = getattr(config, lock_name)

    with path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DwellError, match="lifecycle operation|LTX job"):
            manager.run()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    assert not manager.runtime_dir.exists()
    assert not any("clone" in command for command, _cwd, _env in runner.commands)


def test_setup_lock_collision_is_reported(tmp_path: Path) -> None:
    source, commit = _source_repo(tmp_path)
    config = DwellConfig(home=tmp_path / "home")
    config.state_dir.mkdir(parents=True)
    manager, _runner = _manager(config, _spec(source, commit))

    with config.setup_lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(DwellError, match="already in progress"):
            manager.run()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    assert not manager.runtime_dir.exists()
