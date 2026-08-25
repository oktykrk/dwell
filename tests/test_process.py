from __future__ import annotations

import json
import signal
import socket
from pathlib import Path

import pytest

from dwell import process as process_module
from dwell.config import DwellConfig
from dwell.errors import DwellError
from dwell.process import (
    _read_ltx_child_state,
    _terminate_recorded_ltx_child,
    api_is_healthy,
    read_server_state,
    server_status,
    start_server,
    stop_server,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def test_background_server_start_duplicate_refusal_status_and_stop(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    assert server_status(config)["running"] is False

    try:
        state = start_server(config, timeout=15)
        assert state.pid > 1
        assert config.pid_file.read_text(encoding="utf-8").strip() == str(state.pid)
        assert read_server_state(config) is not None
        assert api_is_healthy(config)
        assert server_status(config)["api_status"]["server"] == "running"

        with pytest.raises(DwellError, match="already running") as duplicate:
            start_server(config)
        assert duplicate.value.code == "server_busy"
    finally:
        stop_server(config)

    assert read_server_state(config) is None
    assert not config.pid_file.exists()
    assert not config.server_state_file.exists()
    assert "starting Dwell" in config.log_file.read_text(encoding="utf-8")


def test_corrupt_active_child_state_blocks_unsafe_cleanup(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    config.ensure_layout()
    config.ltx_active_process_file.write_text("not-json", encoding="utf-8")

    with pytest.raises(DwellError) as error:
        _read_ltx_child_state(config)

    assert error.value.code == "runtime_not_available"
    assert config.ltx_active_process_file.read_text(encoding="utf-8") == "not-json"


def test_incomplete_active_child_state_is_not_silently_ignored(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    config.ensure_layout()
    config.ltx_active_process_file.write_text(json.dumps({"version": 1}), encoding="utf-8")

    with pytest.raises(DwellError) as error:
        stop_server(config)

    assert error.value.code == "runtime_not_available"
    assert config.ltx_active_process_file.exists()


def _active_state(pid: int, *, token: str = "a" * 32) -> dict[str, object]:
    return {
        "version": 1,
        "runtime_id": "ltx-2-mlx",
        "job_id": "job-recovery",
        "ownership_token": token,
        "owner_pid": 9876,
        "pid": pid,
        "pgid": pid,
        "sid": pid,
        "started_at": "2026-08-25T00:00:00+00:00",
    }


def test_recovery_refuses_live_leader_with_mismatched_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    config.ensure_layout()
    config.ltx_active_process_file.write_text(
        json.dumps(_active_state(4321)),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "process_alive", lambda pid: pid == 4321)
    monkeypatch.setattr(process_module.os, "getpgid", lambda _pid: 4321)
    monkeypatch.setattr(process_module, "_process_command", lambda _pid: "uv ltx-2-mlx")
    monkeypatch.setattr(
        process_module,
        "_active_process_group_members",
        lambda _pgid: [(4321, 4321)],
    )
    monkeypatch.setattr(process_module, "_process_has_ltx_token", lambda _pid, _token: False)

    def must_not_signal(_pgid: int, _signal: signal.Signals) -> None:
        raise AssertionError("an ownership mismatch must never be signalled")

    monkeypatch.setattr(process_module.os, "killpg", must_not_signal)

    with pytest.raises(DwellError) as error:
        _terminate_recorded_ltx_child(config, expected_owner_pid=9876)

    assert error.value.code == "runtime_not_available"
    assert config.ltx_active_process_file.exists()


def test_recovery_terminates_owned_descendants_after_leader_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    config.ensure_layout()
    config.ltx_active_process_file.write_text(
        json.dumps(_active_state(5432)),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "process_alive", lambda _pid: False)
    monkeypatch.setattr(process_module, "_process_has_ltx_token", lambda _pid, _token: True)
    groups = iter([[(6000, 5432)], [], [], []])
    monkeypatch.setattr(
        process_module,
        "_active_process_group_members",
        lambda _pgid: next(groups),
    )
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        process_module.os,
        "killpg",
        lambda pgid, requested: signals.append((pgid, requested)),
    )

    assert _terminate_recorded_ltx_child(config, expected_owner_pid=9876, timeout=0.01)
    assert signals == [(5432, signal.SIGTERM)]
    assert not config.ltx_active_process_file.exists()


def test_recovery_refuses_descendants_without_matching_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DwellConfig(home=tmp_path / "dwell", port=_free_port())
    config.ensure_layout()
    config.ltx_active_process_file.write_text(
        json.dumps(_active_state(6543)),
        encoding="utf-8",
    )
    monkeypatch.setattr(process_module, "process_alive", lambda _pid: False)
    monkeypatch.setattr(
        process_module,
        "_active_process_group_members",
        lambda _pgid: [(7000, 6543)],
    )
    monkeypatch.setattr(process_module, "_process_has_ltx_token", lambda _pid, _token: False)

    def must_not_signal(_pgid: int, _signal: signal.Signals) -> None:
        raise AssertionError("an unverifiable descendant must never be signalled")

    monkeypatch.setattr(process_module.os, "killpg", must_not_signal)

    with pytest.raises(DwellError) as error:
        _terminate_recorded_ltx_child(config, expected_owner_pid=9876)

    assert error.value.code == "runtime_not_available"
    assert config.ltx_active_process_file.exists()
