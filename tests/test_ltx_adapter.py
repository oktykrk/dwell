from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import dwell.runtimes.ltx as ltx_module
from dwell.config import DwellConfig
from dwell.domain import VideoGenerationRequest
from dwell.errors import DwellError
from dwell.runtimes.ltx import LTXSubprocessRuntime

_DISTILLED_FILES = (
    "config.json",
    "embedded_config.json",
    "transformer-distilled.safetensors",
    "connector.safetensors",
    "vae_encoder.safetensors",
    "vae_decoder.safetensors",
    "audio_vae.safetensors",
    "vocoder.safetensors",
    "spatial_upscaler_x2_v1_1.safetensors",
    "spatial_upscaler_x2_v1_1_config.json",
)


@dataclass
class AdapterFixture:
    adapter: LTXSubprocessRuntime
    config: DwellConfig
    model: Path
    gemma: Path
    uv: Path
    ffmpeg: Path
    ffprobe: Path
    resolved_models: list[str]


@dataclass
class SpawnCall:
    argv: tuple[str, ...]
    kwargs: dict[str, Any]
    process: Any


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_fake_output(path: Path) -> None:
    path.write_bytes(b"not-a-real-mp4-but-never-decoded")


@pytest.fixture(autouse=True)
def _fake_isolated_sessions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injected subprocess PIDs model start_new_session=True without real children."""

    monkeypatch.setattr(ltx_module.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(ltx_module.os, "getsid", lambda pid: pid)


def _write_active_state(
    path: Path,
    *,
    job_id: str = "persisted-job",
    pid: int = 73_001,
    ownership_token: str = "a" * 32,
) -> dict[str, Any]:
    payload = {
        "version": 1,
        "runtime_id": "ltx-2-mlx",
        "job_id": job_id,
        "ownership_token": ownership_token,
        "owner_pid": os.getpid(),
        "pid": pid,
        "pgid": pid,
        "sid": pid,
        "started_at": "2026-08-24T12:00:00+00:00",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def _adapter(tmp_path: Path, **kwargs: Any) -> AdapterFixture:
    config = DwellConfig(home=tmp_path / "home")
    runtime = config.runtimes_dir / "ltx-2-mlx"
    runtime.mkdir(parents=True)
    (runtime / "pyproject.toml").write_text("[project]\nname='fake-ltx'\n", encoding="utf-8")
    _executable(runtime / ".venv" / "bin" / "ltx-2-mlx")

    tools = tmp_path / "tools"
    uv = _executable(tools / "uv")
    ffmpeg = _executable(tools / "ffmpeg")
    ffprobe = _executable(tools / "ffprobe")

    model = tmp_path / "models" / "ltx-local"
    model.mkdir(parents=True)
    for name in _DISTILLED_FILES:
        (model / name).write_bytes(b"local")
    gemma = model / "gemma4-12b-ltx-v1"
    gemma.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (gemma / name).write_bytes(b"local")
    (gemma / "model.safetensors").write_bytes(b"local weights")

    resolved_models: list[str] = []

    def resolve(model_id: str) -> Path:
        resolved_models.append(model_id)
        return model

    adapter = LTXSubprocessRuntime(
        config,
        resolve,
        uv_binary=uv,
        ffmpeg_binary=ffmpeg,
        ffprobe_binary=ffprobe,
        **kwargs,
    )
    return AdapterFixture(adapter, config, model, gemma, uv, ffmpeg, ffprobe, resolved_models)


def _request(**overrides: Any) -> VideoGenerationRequest:
    values: dict[str, Any] = {
        "model": "local-ltx",
        "prompt": "A secret prompt over Istanbul",
        "width": 576,
        "height": 1024,
        "frames": 121,
        "fps": 23.5,
        "seed": 42,
    }
    values.update(overrides)
    return VideoGenerationRequest(**values)


class CompletedProcess:
    _next_pid = 40_000

    def __init__(
        self,
        argv: tuple[str, ...],
        state_file: Path,
        observed_states: list[tuple[dict[str, Any], int]],
        *,
        ltx_exit_code: int = 0,
        probe_dimensions: tuple[int, int] = (576, 1024),
        write_output: bool = True,
    ) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None
        self.argv = argv
        self.state_file = state_file
        self.observed_states = observed_states
        self.ltx_exit_code = ltx_exit_code
        self.probe_dimensions = probe_dimensions
        self.write_output = write_output

    async def communicate(self) -> tuple[bytes, bytes]:
        if "generate" in self.argv:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
            mode = stat.S_IMODE(self.state_file.stat().st_mode)
            self.observed_states.append((state, mode))
            if self.ltx_exit_code == 0 and self.write_output:
                output = Path(self.argv[self.argv.index("--output") + 1])
                _write_fake_output(output)
            self.returncode = self.ltx_exit_code
            return b"runtime stdout\n", b"runtime stderr\n"

        self.returncode = 0
        width, height = self.probe_dimensions
        payload = {"streams": [{"codec_type": "video", "width": width, "height": height}]}
        return json.dumps(payload).encode(), b""


def _completed_factory(
    fixture: AdapterFixture,
    *,
    ltx_exit_code: int = 0,
    probe_dimensions: tuple[int, int] = (576, 1024),
    write_output: bool = True,
) -> tuple[list[SpawnCall], list[tuple[dict[str, Any], int]], Any]:
    calls: list[SpawnCall] = []
    observed_states: list[tuple[dict[str, Any], int]] = []

    async def create_subprocess(*argv: str, **kwargs: Any) -> CompletedProcess:
        process = CompletedProcess(
            argv,
            fixture.adapter.active_state_file,
            observed_states,
            ltx_exit_code=ltx_exit_code,
            probe_dimensions=probe_dimensions,
            write_output=write_output,
        )
        calls.append(SpawnCall(argv, kwargs, process))
        return process

    return calls, observed_states, create_subprocess


@pytest.mark.asyncio
async def test_generate_uses_exact_offline_distilled_argv_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fixture = _adapter(tmp_path)
    calls, observed_states, create_subprocess = _completed_factory(fixture)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with caplog.at_level(logging.INFO, logger="dwell.runtimes.ltx"):
        result = await fixture.adapter.generate(_request(), "job-argv")

    temporary_output = fixture.config.tmp_dir.absolute() / "ltx-jobs/job-argv/output.mp4"
    expected = (
        str(fixture.uv.resolve()),
        "run",
        "--project",
        str(fixture.adapter.runtime_dir.resolve()),
        "--offline",
        "--no-sync",
        "--no-python-downloads",
        "ltx-2-mlx",
        "generate",
        "--model",
        str(fixture.model.resolve()),
        "--gemma",
        str(fixture.gemma.resolve()),
        "--distilled",
        "--low-ram",
        "--quiet",
        "--prompt",
        "A secret prompt over Istanbul",
        "--output",
        str(temporary_output),
        "--height",
        "1024",
        "--width",
        "576",
        "--frames",
        "121",
        "--frame-rate",
        "23.5",
        "--seed",
        "42",
    )
    assert calls[0].argv == expected
    assert calls[0].kwargs["start_new_session"] is True
    assert calls[0].kwargs["stdin"] == subprocess.DEVNULL
    assert calls[0].kwargs["cwd"] == str(fixture.adapter.runtime_dir.resolve())
    environment = calls[0].kwargs["env"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["UV_NO_SYNC"] == "1"
    assert environment["UV_PYTHON_DOWNLOADS"] == "never"
    assert environment["TMPDIR"] == str(temporary_output.parent)
    assert environment["DWELL_LTX_JOB_ID"] == "job-argv"
    assert len(environment["DWELL_LTX_OWNERSHIP_TOKEN"]) == 32
    int(environment["DWELL_LTX_OWNERSHIP_TOKEN"], 16)

    assert Path(calls[1].argv[0]) == fixture.ffprobe.resolve()
    assert calls[1].argv[-1] == str(temporary_output)
    whitelist = calls[1].argv.index("-protocol_whitelist")
    assert calls[1].argv[whitelist + 1] == "file"
    assert calls[1].kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert "DWELL_LTX_OWNERSHIP_TOKEN" not in calls[1].kwargs["env"]

    final_output = fixture.config.outputs_dir.absolute() / "video/job-argv.mp4"
    assert result.output_path == final_output
    assert final_output.read_bytes() == b"not-a-real-mp4-but-never-decoded"
    assert result.exit_code == 0
    assert result.stdout == "runtime stdout\n"
    assert result.stderr == "runtime stderr\n"
    assert result.duration_seconds >= 0
    assert not temporary_output.parent.exists()
    assert not fixture.adapter.active_state_file.exists()
    assert fixture.resolved_models == ["local-ltx"]

    state, mode = observed_states[0]
    assert state == {
        "job_id": "job-argv",
        "ownership_token": environment["DWELL_LTX_OWNERSHIP_TOKEN"],
        "owner_pid": os.getpid(),
        "pgid": calls[0].process.pid,
        "sid": calls[0].process.pid,
        "pid": calls[0].process.pid,
        "runtime_id": "ltx-2-mlx",
        "started_at": state["started_at"],
        "version": 1,
    }
    assert mode == 0o600
    assert "A secret prompt" not in json.dumps(state)
    assert state["ownership_token"] == environment["DWELL_LTX_OWNERSHIP_TOKEN"]
    assert "A secret prompt" not in caplog.text
    assert "<redacted>" in caplog.text


@pytest.mark.asyncio
async def test_existing_output_is_never_started_or_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    final_output = fixture.config.outputs_dir.absolute() / "video/existing.mp4"
    final_output.parent.mkdir(parents=True)
    final_output.write_bytes(b"keep me")

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("generation must not start for an existing output")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "existing")

    assert caught.value.code == "output_exists"
    assert final_output.read_bytes() == b"keep me"


@pytest.mark.asyncio
async def test_live_persisted_process_group_is_reported_and_refuses_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    payload = _write_active_state(fixture.adapter.active_state_file)
    monkeypatch.setattr(fixture.adapter, "_process_group_alive", lambda _pgid: True)

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a recorded live group must prevent subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    status = await fixture.adapter.status()
    assert status.available is True
    assert status.active_jobs == ("persisted-job",)

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "blocked-by-persisted")

    assert caught.value.code == "server_busy"
    assert caught.value.details == {
        "job_id": "persisted-job",
        "owner_pid": os.getpid(),
        "pid": payload["pid"],
        "pgid": payload["pgid"],
        "sid": payload["sid"],
    }
    assert fixture.adapter.active_state_file.exists()
    assert fixture.resolved_models == []


@pytest.mark.asyncio
async def test_dead_owned_process_group_state_is_compare_cleared_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    _write_active_state(fixture.adapter.active_state_file)
    monkeypatch.setattr(fixture.adapter, "_process_group_alive", lambda _pgid: False)
    calls, _states, create_subprocess = _completed_factory(fixture)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    result = await fixture.adapter.generate(_request(), "after-stale")

    assert result.exit_code == 0
    assert len(calls) == 2
    assert not fixture.adapter.active_state_file.exists()


@pytest.mark.asyncio
async def test_legacy_state_without_ownership_token_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    payload = _write_active_state(fixture.adapter.active_state_file)
    payload.pop("ownership_token")
    original = json.dumps(payload) + "\n"
    fixture.adapter.active_state_file.write_text(original, encoding="utf-8")

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("unverifiable legacy state must prevent subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    status = await fixture.adapter.validate_installation()
    assert status.available is False
    assert "legacy ownership record" in (status.message or "")

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "blocked-by-legacy")

    assert caught.value.code == "runtime_not_available"
    assert fixture.adapter.active_state_file.read_text(encoding="utf-8") == original
    assert fixture.resolved_models == []


@pytest.mark.asyncio
async def test_atomic_publish_loses_race_without_clobbering_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    calls, _states, create_subprocess = _completed_factory(fixture)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    final_output = fixture.config.outputs_dir.absolute() / "video/racing.mp4"

    def competing_publish(_source: Path, destination: Path, **_kwargs: Any) -> None:
        destination.write_bytes(b"winner")
        raise FileExistsError(destination)

    monkeypatch.setattr(ltx_module.os, "link", competing_publish)
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "racing")

    assert caught.value.code == "output_exists"
    assert final_output.read_bytes() == b"winner"
    assert len(calls) == 2
    assert not (fixture.config.tmp_dir / "ltx-jobs/racing").exists()


@pytest.mark.asyncio
async def test_distilled_adapter_rejects_dimensions_not_aligned_to_64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("invalid dimensions must fail before subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    # Exercise the adapter's defensive boundary even though the public model
    # normally rejects this value before runtime dispatch.
    invalid_request = _request().model_copy(update={"width": 608})
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(invalid_request, "bad-dimensions")

    assert caught.value.code == "invalid_request"
    assert caught.value.status_code == 422
    assert fixture.resolved_models == []


@pytest.mark.asyncio
async def test_missing_runtime_cli_fails_before_model_resolution_or_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    (fixture.adapter.runtime_dir / ".venv/bin/ltx-2-mlx").unlink()

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an unavailable runtime must fail before subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "missing-runtime")

    assert caught.value.code == "runtime_not_available"
    assert fixture.resolved_models == []


@pytest.mark.asyncio
async def test_zero_length_distilled_component_fails_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    (fixture.model / "connector.safetensors").write_bytes(b"")

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("an incomplete model must fail before subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "zero-component")

    assert caught.value.code == "model_not_installed"
    assert "connector.safetensors" in caught.value.details["missing_files"]


@pytest.mark.asyncio
async def test_ffprobe_must_report_requested_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    _calls, _states, create_subprocess = _completed_factory(
        fixture,
        probe_dimensions=(640, 1024),
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "wrong-size")

    assert caught.value.code == "generation_failed"
    assert caught.value.details["expected"] == {"width": 576, "height": 1024}
    assert caught.value.details["actual"] == {"width": 640, "height": 1024}
    assert not (fixture.config.outputs_dir / "video/wrong-size.mp4").exists()


@pytest.mark.asyncio
async def test_cancellation_during_ffprobe_spawn_terminates_probe_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(
        tmp_path,
        termination_grace_seconds=0.05,
        kill_wait_seconds=0.05,
    )
    observed_states: list[tuple[dict[str, Any], int]] = []
    probe_spawned = asyncio.Event()
    permit_probe = asyncio.Event()
    probe = HangingProcess(fixture.adapter.active_state_file, [])

    async def spawn(*argv: str, **_kwargs: Any) -> Any:
        if "generate" in argv:
            return CompletedProcess(argv, fixture.adapter.active_state_file, observed_states)
        probe_spawned.set()
        await permit_probe.wait()
        return probe

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, requested_signal: signal.Signals) -> None:
        signals.append((pgid, requested_signal))
        probe.finish(requested_signal)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ltx_module.os, "killpg", killpg)

    generation = asyncio.create_task(fixture.adapter.generate(_request(), "probe-spawn-cancel"))
    await asyncio.wait_for(probe_spawned.wait(), timeout=1)
    generation.cancel()
    await asyncio.sleep(0)
    permit_probe.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(generation, timeout=1)

    assert signals == [(probe.pid, signal.SIGTERM)]
    assert probe.returncode == -int(signal.SIGTERM)
    assert not fixture.adapter.active_state_file.exists()
    assert not (fixture.config.outputs_dir / "video/probe-spawn-cancel.mp4").exists()
    assert not (fixture.config.tmp_dir / "ltx-jobs/probe-spawn-cancel").exists()


@pytest.mark.asyncio
async def test_exit_zero_without_nonempty_mp4_is_generation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    calls, _states, create_subprocess = _completed_factory(fixture, write_output=False)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "missing-output")

    assert caught.value.code == "generation_failed"
    assert "produced no MP4" in caught.value.message
    assert len(calls) == 1  # ffprobe is never invoked for a missing file.
    assert not (fixture.config.outputs_dir / "video/missing-output.mp4").exists()
    assert not fixture.adapter.active_state_file.exists()


@pytest.mark.asyncio
async def test_exited_launcher_with_live_descendant_group_is_terminated_and_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    calls, _states, create_subprocess = _completed_factory(fixture)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_subprocess)
    group_live = True
    monkeypatch.setattr(fixture.adapter, "_process_group_alive", lambda _pgid: group_live)
    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, requested_signal: signal.Signals) -> None:
        nonlocal group_live
        signals.append((pgid, requested_signal))
        group_live = False

    monkeypatch.setattr(ltx_module.os, "killpg", killpg)

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "orphan-descendant")

    assert caught.value.code == "generation_failed"
    assert "child process remained active" in caught.value.message
    assert signals == [(calls[0].process.pid, signal.SIGTERM)]
    assert len(calls) == 1
    assert not fixture.adapter.active_state_file.exists()
    assert not (fixture.config.outputs_dir / "video/orphan-descendant.mp4").exists()


class HangingProcess:
    _next_pid = 50_000

    def __init__(self, state_file: Path, observed_states: list[dict[str, Any]]) -> None:
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode: int | None = None
        self.state_file = state_file
        self.observed_states = observed_states
        self.finished = asyncio.Event()
        self.exit_signal: signal.Signals | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.state_file.exists():
            self.observed_states.append(json.loads(self.state_file.read_text(encoding="utf-8")))
        await self.finished.wait()
        assert self.exit_signal is not None
        self.returncode = -int(self.exit_signal)
        return b"partial stdout", b"terminated"

    def finish(self, requested_signal: signal.Signals) -> None:
        self.exit_signal = requested_signal
        self.finished.set()


@pytest.mark.asyncio
async def test_cancellation_during_spawn_still_terminates_new_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(
        tmp_path,
        termination_grace_seconds=0.05,
        kill_wait_seconds=0.05,
    )
    spawn_entered = asyncio.Event()
    permit_spawn = asyncio.Event()
    observed_states: list[dict[str, Any]] = []
    process = HangingProcess(fixture.adapter.active_state_file, observed_states)

    async def delayed_spawn(*_argv: str, **_kwargs: Any) -> HangingProcess:
        spawn_entered.set()
        await permit_spawn.wait()
        return process

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, requested_signal: signal.Signals) -> None:
        signals.append((pgid, requested_signal))
        process.finish(requested_signal)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    monkeypatch.setattr(ltx_module.os, "killpg", killpg)

    generation = asyncio.create_task(fixture.adapter.generate(_request(), "spawn-cancel"))
    await asyncio.wait_for(spawn_entered.wait(), timeout=1)
    generation.cancel()
    await asyncio.sleep(0)
    permit_spawn.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(generation, timeout=1)

    assert signals == [(process.pid, signal.SIGTERM)]
    assert observed_states[0]["pid"] == process.pid
    assert observed_states[0]["pgid"] == process.pid
    assert not fixture.adapter.active_state_file.exists()
    assert fixture.adapter.active_pid is None
    assert not (fixture.config.tmp_dir / "ltx-jobs/spawn-cancel").exists()


@pytest.mark.asyncio
async def test_child_session_mismatch_refuses_state_and_terminates_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(
        tmp_path,
        termination_grace_seconds=0.05,
        kill_wait_seconds=0.05,
    )
    process = HangingProcess(fixture.adapter.active_state_file, [])

    async def spawn(*_argv: str, **_kwargs: Any) -> HangingProcess:
        return process

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, requested_signal: signal.Signals) -> None:
        signals.append((pgid, requested_signal))
        process.finish(requested_signal)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ltx_module.os, "getsid", lambda pid: pid + 1)
    monkeypatch.setattr(ltx_module.os, "killpg", killpg)

    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "wrong-session")

    assert caught.value.code == "runtime_not_available"
    assert "persist" in caught.value.message
    assert signals == [(process.pid, signal.SIGTERM)]
    assert not fixture.adapter.active_state_file.exists()
    assert fixture.adapter.active_pid is None


@pytest.mark.asyncio
async def test_explicit_cancel_escalates_to_sigkill_and_maps_to_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(
        tmp_path,
        termination_grace_seconds=0.01,
        kill_wait_seconds=0.05,
    )
    observed_states: list[dict[str, Any]] = []
    process = HangingProcess(fixture.adapter.active_state_file, observed_states)
    spawned = asyncio.Event()

    async def spawn(*_argv: str, **_kwargs: Any) -> HangingProcess:
        spawned.set()
        return process

    signals: list[tuple[int, signal.Signals]] = []

    def killpg(pgid: int, requested_signal: signal.Signals) -> None:
        signals.append((pgid, requested_signal))
        if requested_signal == signal.SIGKILL:
            process.finish(requested_signal)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(ltx_module.os, "killpg", killpg)

    generation = asyncio.create_task(fixture.adapter.generate(_request(), "explicit-cancel"))
    await asyncio.wait_for(spawned.wait(), timeout=1)
    for _ in range(100):
        if fixture.adapter.active_state_file.exists():
            break
        await asyncio.sleep(0)
    assert fixture.adapter.active_state_file.exists()

    assert await fixture.adapter.cancel("explicit-cancel") is True
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(generation, timeout=1)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert not fixture.adapter.active_state_file.exists()
    assert fixture.adapter.active_pid is None
    assert not (fixture.config.outputs_dir / "video/explicit-cancel.mp4").exists()


@pytest.mark.asyncio
async def test_unconfirmed_sigkill_is_generation_failure_and_retains_recovery_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(
        tmp_path,
        termination_grace_seconds=0.01,
        kill_wait_seconds=0.01,
    )
    observed_states: list[dict[str, Any]] = []
    process = HangingProcess(fixture.adapter.active_state_file, observed_states)
    spawned = asyncio.Event()

    async def spawn(*_argv: str, **_kwargs: Any) -> HangingProcess:
        spawned.set()
        return process

    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(
        ltx_module.os,
        "killpg",
        lambda pgid, requested_signal: signals.append((pgid, requested_signal)),
    )

    generation = asyncio.create_task(fixture.adapter.generate(_request(), "stubborn-child"))
    await asyncio.wait_for(spawned.wait(), timeout=1)
    for _ in range(100):
        if fixture.adapter.active_state_file.exists():
            break
        await asyncio.sleep(0)
    generation.cancel()

    with pytest.raises(DwellError) as caught:
        await asyncio.wait_for(generation, timeout=1)

    assert caught.value.code == "generation_failed"
    assert "did not exit" in caught.value.message
    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert fixture.adapter.active_state_file.exists()
    assert fixture.adapter.active_pid == process.pid
    assert (fixture.config.tmp_dir / "ltx-jobs/stubborn-child").exists()

    # Finish the injected process so the test leaves no pending asyncio task.
    active = fixture.adapter._active
    assert active is not None
    process.finish(signal.SIGKILL)
    await active.communicate_task
    fixture.adapter._clear_active_state(active)
    fixture.adapter._active = None


@pytest.mark.asyncio
async def test_incomplete_gemma_shard_set_fails_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _adapter(tmp_path)
    (fixture.gemma / "model.safetensors").unlink()
    (fixture.gemma / "model-00001-of-00002.safetensors").write_bytes(b"only one shard")

    async def must_not_spawn(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("incomplete weights must fail before subprocess creation")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", must_not_spawn)
    with pytest.raises(DwellError) as caught:
        await fixture.adapter.generate(_request(), "missing-shard")

    assert caught.value.code == "model_not_installed"
    assert "<complete safetensors set>" in caught.value.details["missing_files"][0]
