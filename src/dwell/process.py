from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dwell.config import DwellConfig
from dwell.errors import DwellError


@dataclass(frozen=True)
class ServerState:
    pid: int
    started_at: str
    api_url: str
    command: list[str]

    @property
    def uptime_seconds(self) -> float:
        try:
            started = datetime.fromisoformat(self.started_at)
        except ValueError:
            return 0.0
        return max(0.0, (datetime.now(UTC) - started).total_seconds())


def process_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    state = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if state.startswith("Z"):
        return False
    return True


def _process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _active_process_group_members(pgid: int) -> list[tuple[int, int]] | None:
    """Return non-zombie members, or ``None`` when ownership cannot be inspected."""

    if pgid <= 1:
        return []
    result = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,stat="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    members: list[tuple[int, int]] = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3:
            continue
        try:
            pid = int(fields[0])
            member_pgid = int(fields[1])
        except ValueError:
            continue
        if member_pgid != pgid or pid <= 1 or fields[2].startswith("Z"):
            continue
        try:
            session_id = os.getsid(pid)
        except ProcessLookupError:
            continue
        except OSError:
            return None
        members.append((pid, session_id))
    return members


def _process_has_ltx_token(pid: int, token: str) -> bool:
    result = subprocess.run(
        ["ps", "-Eww", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    marker = f"DWELL_LTX_OWNERSHIP_TOKEN={token}"
    return result.returncode == 0 and marker in result.stdout


def _looks_like_dwell(pid: int) -> bool:
    command = _process_command(pid)
    return "uvicorn" in command and "dwell.api" in command


def _atomic_write(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents, encoding="utf-8")
    os.replace(temporary, path)


def write_server_state(config: DwellConfig, state: ServerState) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(config.pid_file, f"{state.pid}\n")
    _atomic_write(config.server_state_file, json.dumps(asdict(state), indent=2) + "\n")


def clear_server_state(config: DwellConfig, *, expected_pid: int | None = None) -> None:
    if expected_pid is not None:
        try:
            current_pid = int(config.pid_file.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            current_pid = None
        if current_pid not in {None, expected_pid}:
            return
    for path in (config.pid_file, config.server_state_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_server_state(config: DwellConfig, *, clean_stale: bool = True) -> ServerState | None:
    try:
        pid = int(config.pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        if clean_stale and config.pid_file.exists():
            clear_server_state(config)
        return None

    metadata: dict[str, Any] = {}
    try:
        metadata = json.loads(config.server_state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    if not process_alive(pid) or not _looks_like_dwell(pid):
        if clean_stale:
            clear_server_state(config, expected_pid=pid)
        return None

    started_at = metadata.get("started_at")
    if not isinstance(started_at, str):
        started_at = datetime.fromtimestamp(config.pid_file.stat().st_mtime, UTC).isoformat()
    command = metadata.get("command")
    if not isinstance(command, list):
        command = _process_command(pid).split()
    api_url = metadata.get("api_url")
    if not isinstance(api_url, str):
        api_url = config.api_url
    return ServerState(pid=pid, started_at=started_at, api_url=api_url, command=command)


@contextmanager
def _lifecycle_lock(config: DwellConfig) -> Iterator[None]:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    with config.start_lock_file.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DwellError(
                "server_busy",
                "Another Dwell lifecycle operation is already in progress.",
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def api_request(
    config: DwellConfig,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 1.0,
) -> Any:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"content-type": "application/json"} if body is not None else {}
    request = urllib.request.Request(
        f"{config.api_url}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def api_is_healthy(config: DwellConfig, *, timeout: float = 0.5) -> bool:
    try:
        payload = api_request(config, "/health", timeout=timeout)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def port_is_available(config: DwellConfig) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((config.host, config.port))
        except OSError:
            return False
    return True


def _last_log_lines(path: Path, count: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(lines[-count:])


def _read_ltx_child_state(config: DwellConfig) -> dict[str, Any] | None:
    try:
        payload = json.loads(config.ltx_active_process_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process state is unreadable; refusing an unsafe operation.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        ) from exc
    if not isinstance(payload, dict):
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process state has an unexpected format.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    required = ("version", "runtime_id", "owner_pid", "pid", "pgid", "job_id")
    if any(name not in payload for name in required):
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process state is incomplete; refusing an unsafe operation.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    if payload.get("version") != 1 or payload.get("runtime_id") != "ltx-2-mlx":
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process state has an unexpected format.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    if not all(type(payload.get(name)) is int for name in ("owner_pid", "pid", "pgid")):
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process identifiers are invalid.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    if not isinstance(payload.get("job_id"), str) or not payload["job_id"]:
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX job identifier is invalid.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    if payload["pid"] <= 1 or payload["pgid"] != payload["pid"]:
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process group is unsafe.",
            details={"path": str(config.ltx_active_process_file)},
            status_code=503,
        )
    return payload


def _clear_ltx_child_state(config: DwellConfig, *, expected_pid: int) -> None:
    payload = _read_ltx_child_state(config)
    if payload is not None and payload.get("pid") != expected_pid:
        return
    try:
        config.ltx_active_process_file.unlink()
    except FileNotFoundError:
        pass


def _terminate_recorded_ltx_child(
    config: DwellConfig,
    *,
    expected_owner_pid: int | None,
    timeout: float = 3.0,
) -> bool:
    payload = _read_ltx_child_state(config)
    if payload is None:
        return False
    owner_pid = int(payload["owner_pid"])
    child_pid = int(payload["pid"])
    pgid = int(payload["pgid"])
    ownership_token = payload.get("ownership_token")
    session_id = payload.get("sid")

    if expected_owner_pid is not None and owner_pid != expected_owner_pid:
        return False
    if expected_owner_pid is None and process_alive(owner_pid) and _looks_like_dwell(owner_pid):
        raise DwellError(
            "server_busy",
            "A live Dwell server still owns the recorded LTX process.",
            details={"owner_pid": owner_pid, "child_pid": child_pid},
            status_code=409,
        )
    leader_alive = process_alive(child_pid)
    if leader_alive:
        try:
            actual_pgid = os.getpgid(child_pid)
        except ProcessLookupError:
            actual_pgid = None
        if actual_pgid != pgid:
            raise DwellError(
                "runtime_not_available",
                "Refusing to signal an LTX PID whose process group no longer matches state.",
                details={"pid": child_pid, "recorded_pgid": pgid, "actual_pgid": actual_pgid},
                status_code=503,
            )
    if leader_alive and "ltx-2-mlx" not in _process_command(child_pid):
        raise DwellError(
            "runtime_not_available",
            "Refusing to signal an active-process PID that no longer belongs to LTX.",
            details={"pid": child_pid},
            status_code=503,
        )

    members = _active_process_group_members(pgid)
    if members is None:
        raise DwellError(
            "runtime_not_available",
            "The recorded LTX process group could not be inspected safely.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )
    token_is_valid = (
        isinstance(ownership_token, str)
        and len(ownership_token) == 32
        and all(character in "0123456789abcdef" for character in ownership_token)
    )
    session_is_valid = type(session_id) is int and session_id == pgid
    if members and (
        not token_is_valid
        or not session_is_valid
        or any(member_session != session_id for _member, member_session in members)
    ):
        raise DwellError(
            "runtime_not_available",
            "A surviving LTX process group could not be ownership-validated.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )
    if leader_alive and members and not _process_has_ltx_token(child_pid, ownership_token):
        raise DwellError(
            "runtime_not_available",
            "The live LTX process does not match its recorded ownership token.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )
    if (
        not leader_alive
        and members
        and not any(_process_has_ltx_token(member, ownership_token) for member, _session in members)
    ):
        raise DwellError(
            "runtime_not_available",
            "No surviving LTX process exposes the recorded ownership token.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )
    if leader_alive and not any(member == child_pid for member, _session in members):
        raise DwellError(
            "runtime_not_available",
            "The live LTX process could not be found in its recorded process group.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )

    if members:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            members = _active_process_group_members(pgid)
            if members == []:
                break
            if members is None:
                break
            time.sleep(0.05)
    members = _active_process_group_members(pgid)
    if members:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            members = _active_process_group_members(pgid)
            if members == []:
                break
            if members is None:
                break
            time.sleep(0.05)
    members = _active_process_group_members(pgid)
    if members is None or members:
        raise DwellError(
            "runtime_not_available",
            "The active LTX process could not be terminated safely.",
            details={"pid": child_pid, "pgid": pgid},
            status_code=503,
        )
    _clear_ltx_child_state(config, expected_pid=child_pid)
    return True


def _recorded_ltx_group_may_be_active(payload: dict[str, Any]) -> bool:
    """Conservatively report whether a persisted LTX group may still own weights."""

    if process_alive(int(payload["pid"])):
        return True
    members = _active_process_group_members(int(payload["pgid"]))
    return members is None or bool(members)


def start_server(config: DwellConfig, *, timeout: float = 20.0) -> ServerState:
    config.ensure_layout()
    with _lifecycle_lock(config):
        return _start_server_locked(config, timeout=timeout)


def _start_server_locked(config: DwellConfig, *, timeout: float) -> ServerState:
    running = read_server_state(config)
    if running is not None:
        raise DwellError("server_busy", f"Dwell is already running (PID {running.pid}).")

    # A prior server may have crashed after starting an isolated inference
    # session. Reap only the ownership-validated recorded child before restart.
    _terminate_recorded_ltx_child(config, expected_owner_pid=None)
    if not port_is_available(config):
        raise DwellError(
            "server_busy",
            f"Port {config.port} on {config.host} is already in use.",
        )

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "dwell.api.app:create_app",
        "--factory",
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    with config.log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(UTC).isoformat()}] starting Dwell\n")
        log.flush()
        process = subprocess.Popen(  # noqa: S603
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**config.subprocess_env(), "PYTHONUNBUFFERED": "1"},
            start_new_session=True,
            close_fds=True,
        )

    state = ServerState(
        pid=process.pid,
        started_at=datetime.now(UTC).isoformat(),
        api_url=config.api_url,
        command=command,
    )
    write_server_state(config, state)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _terminate_recorded_ltx_child(config, expected_owner_pid=process.pid)
            clear_server_state(config, expected_pid=process.pid)
            detail = _last_log_lines(config.log_file)
            raise DwellError(
                "runtime_not_available",
                "Dwell API exited during startup.",
                details=detail or None,
            )
        if api_is_healthy(config):
            return state
        time.sleep(0.1)

    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired as exc:
        raise DwellError(
            "runtime_not_available",
            "The unhealthy Dwell process could not be terminated.",
            details={"pid": process.pid},
            status_code=503,
        ) from exc
    _terminate_recorded_ltx_child(config, expected_owner_pid=process.pid)
    clear_server_state(config, expected_pid=process.pid)
    raise DwellError(
        "runtime_not_available",
        f"Dwell did not become healthy within {timeout:g} seconds.",
        details=_last_log_lines(config.log_file) or None,
    )


def stop_server(config: DwellConfig, *, timeout: float = 10.0) -> bool:
    config.ensure_layout()
    with _lifecycle_lock(config):
        return _stop_server_locked(config, timeout=timeout)


def _stop_server_locked(config: DwellConfig, *, timeout: float) -> bool:
    state = read_server_state(config)
    if state is None:
        clear_server_state(config)
        _terminate_recorded_ltx_child(config, expected_owner_pid=None)
        return False

    try:
        os.kill(state.pid, signal.SIGTERM)
    except ProcessLookupError:
        _terminate_recorded_ltx_child(config, expected_owner_pid=state.pid)
        clear_server_state(config, expected_pid=state.pid)
        return False

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_alive(state.pid):
            _terminate_recorded_ltx_child(config, expected_owner_pid=state.pid)
            clear_server_state(config, expected_pid=state.pid)
            return True
        time.sleep(0.1)

    # The child is a separate process group by design. Terminate it explicitly
    # before force-killing a server that could no longer run lifespan cleanup.
    _terminate_recorded_ltx_child(config, expected_owner_pid=state.pid)
    if _looks_like_dwell(state.pid):
        try:
            os.killpg(state.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        force_deadline = time.monotonic() + 2
        while time.monotonic() < force_deadline and process_alive(state.pid):
            time.sleep(0.05)
    if process_alive(state.pid):
        raise DwellError(
            "runtime_not_available",
            "Dwell could not be stopped; the server process is still alive.",
            details={"pid": state.pid},
            status_code=503,
        )
    clear_server_state(config, expected_pid=state.pid)
    return True


def restart_server(config: DwellConfig, *, timeout: float = 20.0) -> ServerState:
    config.ensure_layout()
    with _lifecycle_lock(config):
        _stop_server_locked(config, timeout=10.0)
        return _start_server_locked(config, timeout=timeout)


def process_rss_mb(pid: int) -> float | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip()) / 1024
    except ValueError:
        return None


def server_status(config: DwellConfig) -> dict[str, Any]:
    state = read_server_state(config)
    if state is None:
        return {
            "running": False,
            "api": config.api_url,
            "pid": None,
            "uptime_seconds": None,
            "rss_mb": None,
            "api_status": None,
        }

    try:
        remote_status = api_request(config, "/v1/status", timeout=1.0)
    except (OSError, ValueError, urllib.error.URLError):
        remote_status = None
    return {
        "running": True,
        "api": config.api_url,
        "pid": state.pid,
        "uptime_seconds": state.uptime_seconds,
        "rss_mb": process_rss_mb(state.pid),
        "api_status": remote_status,
    }
