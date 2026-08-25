from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dwell.config import DwellConfig
from dwell.domain import (
    GenerationResult,
    ModelDefinition,
    RuntimeCapabilities,
    VideoGenerationRequest,
)
from dwell.errors import DwellError
from dwell.runtimes.base import ModelRuntime, RuntimeStatus, VideoEngine

logger = logging.getLogger(__name__)

_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_GEMMA_SHARD = re.compile(
    r"(?P<prefix>.+)-(?P<number>[0-9]{5})-of-(?P<total>[0-9]{5})\.safetensors\Z"
)
_OWNERSHIP_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_FAILURE_OUTPUT_LIMIT = 16_384
_ACTIVE_STATE_LIMIT = 64 * 1024
_GEMMA_DIRECTORY = "gemma4-12b-ltx-v1"
_OWNERSHIP_ENV = "DWELL_LTX_OWNERSHIP_TOKEN"
_JOB_ID_ENV = "DWELL_LTX_JOB_ID"
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
_GEMMA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class LTXGenerationResult(GenerationResult):
    """Successful LTX subprocess result, including its verified exit status."""

    exit_code: int = 0


@dataclass(slots=True)
class _ActiveGeneration:
    job_id: str
    ownership_token: str
    process: asyncio.subprocess.Process
    pgid: int
    sid: int
    communicate_task: asyncio.Task[tuple[bytes, bytes]]
    started_at: str
    terminate_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancellation_requested: bool = False


@dataclass(frozen=True, slots=True)
class _PersistedGeneration:
    job_id: str
    ownership_token: str
    owner_pid: int
    pid: int
    pgid: int
    sid: int
    started_at: str | None


class LTXSubprocessRuntime(ModelRuntime, VideoEngine):
    """Network-isolated adapter for the local LTX-2.5 MLX command line runtime."""

    runtime_id = "ltx-2-mlx"
    capabilities = RuntimeCapabilities(
        persistent_loading=False,
        progress_reporting=False,
        cancellation=True,
        streaming=False,
        structured_output=False,
    )

    def __init__(
        self,
        config: DwellConfig,
        model_resolver: Callable[[str], Path],
        *,
        uv_binary: str | Path = "uv",
        ffmpeg_binary: str | Path = "ffmpeg",
        ffprobe_binary: str | Path = "ffprobe",
        termination_grace_seconds: float = 3.0,
        kill_wait_seconds: float = 2.0,
        ffprobe_timeout_seconds: float = 30.0,
    ) -> None:
        if termination_grace_seconds <= 0 or kill_wait_seconds <= 0:
            raise ValueError("process termination timeouts must be positive")
        if ffprobe_timeout_seconds <= 0:
            raise ValueError("ffprobe timeout must be positive")

        self.config = config
        self.model_resolver = model_resolver
        self.runtime_dir = config.runtimes_dir / self.runtime_id
        self.uv_binary = str(uv_binary)
        self.ffmpeg_binary = str(ffmpeg_binary)
        self.ffprobe_binary = str(ffprobe_binary)
        self.termination_grace_seconds = termination_grace_seconds
        self.kill_wait_seconds = kill_wait_seconds
        self.ffprobe_timeout_seconds = ffprobe_timeout_seconds
        self.active_state_file = config.ltx_active_process_file
        self.process_lock_file = config.ltx_lock_file
        self._generation_lock = asyncio.Lock()
        self._active: _ActiveGeneration | None = None

    @property
    def active_job_id(self) -> str | None:
        active = self._active
        if active is None or not self._active_is_live(active):
            return None
        return active.job_id

    @property
    def active_pid(self) -> int | None:
        active = self._active
        if active is None or not self._active_is_live(active):
            return None
        return active.process.pid

    async def validate_installation(self) -> RuntimeStatus:
        problems = self._runtime_problems()
        return RuntimeStatus(
            runtime_id=self.runtime_id,
            available=not problems,
            active_jobs=self._active_jobs(),
            message="; ".join(problems) if problems else "local subprocess runtime is available",
        )

    async def prepare(
        self,
        model: ModelDefinition,
        model_path: Path,
    ) -> RuntimeStatus:
        self._ensure_runtime_available()
        if model.runtime != self.runtime_id:
            raise DwellError(
                "runtime_not_available",
                f"Model '{model.id}' requires runtime '{model.runtime}', not '{self.runtime_id}'.",
                status_code=409,
            )
        self._validated_model_paths(model.id, model_path)
        return RuntimeStatus(
            runtime_id=self.runtime_id,
            available=True,
            active_jobs=self._active_jobs(),
            message="ready; models are loaded only inside generation subprocesses",
        )

    async def release(self, model: ModelDefinition | None = None) -> RuntimeStatus:
        # LTX subprocesses do not keep a model resident. Release means stopping
        # the one child that may currently own Metal resources.
        del model
        active = self._active
        if active is not None and self._active_is_live(active):
            active.cancellation_requested = True
            await self._terminate(active)
        return await self.status()

    async def status(self) -> RuntimeStatus:
        problems = self._runtime_problems()
        active_jobs = self._active_jobs()
        if active_jobs:
            message = f"generation active for job {active_jobs[0]}"
        else:
            message = "; ".join(problems) if problems else "idle; no model is resident"
        return RuntimeStatus(
            runtime_id=self.runtime_id,
            available=not problems,
            loaded_models=(),
            active_jobs=active_jobs,
            message=message,
        )

    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> LTXGenerationResult:
        """Generate, verify, and publish one job without any implicit installation."""

        self._validate_request(request, job_id)
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError

        async with self._generation_lock:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            if self._active is not None and self._active_is_live(self._active):
                raise self._busy_error()
            with self._exclusive_process_lock():
                return await self._generate_locked(request, job_id, cancel_event)

    async def generate_video(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> LTXGenerationResult:
        """Job-manager-compatible spelling for :meth:`generate`."""

        return await self.generate(request, job_id, cancel_event)

    async def cancel(self, job_id: str) -> bool:
        active = self._active
        if active is None or active.job_id != job_id or not self._active_is_live(active):
            return False
        active.cancellation_requested = True
        return await self._terminate(active)

    async def _generate_locked(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: asyncio.Event | None,
    ) -> LTXGenerationResult:
        self._ensure_runtime_available()
        model_path, gemma_path = self._validated_model_paths(
            request.model,
            self.model_resolver(request.model),
        )
        job_directory, temporary_output, final_output = self._job_paths(job_id)

        if self._path_entry_exists(final_output):
            raise self._output_exists(job_id)

        job_directory.parent.mkdir(parents=True, exist_ok=True)
        final_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            job_directory.mkdir()
        except FileExistsError as exc:
            raise DwellError(
                "output_exists",
                f"Temporary artifacts for job '{job_id}' already exist; refusing to overwrite.",
                details={"path": str(job_directory)},
                status_code=409,
            ) from exc

        process_completed = False
        try:
            command = self._command(request, model_path, gemma_path, temporary_output)
            environment = self.config.subprocess_env()
            environment.update(
                {
                    "PYTHONUNBUFFERED": "1",
                    "TMPDIR": str(job_directory),
                }
            )
            logger.info("Starting LTX job %s: %s", job_id, self._redacted_command(command))
            exit_code, stdout, stderr, duration = await self._run_ltx(
                command,
                environment,
                job_id,
                cancel_event,
            )
            process_completed = True
            logger.info(
                "LTX child for job %s exited %d after %.3fs",
                job_id,
                exit_code,
                duration,
            )
            if exit_code != 0:
                raise DwellError(
                    "generation_failed",
                    "The LTX runtime exited without generating a video.",
                    details={
                        "exit_code": exit_code,
                        "stdout": self._failure_output(stdout),
                        "stderr": self._failure_output(stderr),
                        "duration_seconds": duration,
                    },
                    status_code=500,
                )

            self._validate_regular_output(temporary_output)
            await self._validate_mp4(
                temporary_output,
                expected_width=request.width,
                expected_height=request.height,
                environment=environment,
            )
            self._publish_no_clobber(temporary_output, final_output, job_id)
            logger.info(
                "Completed LTX job %s with exit %d in %.3fs: %s",
                job_id,
                exit_code,
                duration,
                final_output,
            )
            return LTXGenerationResult(
                output_path=final_output,
                duration_seconds=duration,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
            )
        finally:
            # If SIGKILL could not be confirmed, retain the exact directory for
            # the still-live child and for force-stop recovery. Otherwise every
            # runtime-created artifact is safely confined to this directory.
            active = self._active
            child_still_active = (
                active is not None and active.job_id == job_id and self._active_is_live(active)
            )
            if process_completed or not child_still_active:
                shutil.rmtree(job_directory, ignore_errors=True)

    async def _run_ltx(
        self,
        command: list[str],
        environment: dict[str, str],
        job_id: str,
        cancel_event: asyncio.Event | None,
    ) -> tuple[int, str, str, float]:
        started = time.monotonic()
        ownership_token = uuid4().hex
        child_environment = {
            **environment,
            _OWNERSHIP_ENV: ownership_token,
            _JOB_ID_ENV: job_id,
        }
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.runtime_dir.resolve()),
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            ),
            name=f"dwell-ltx-spawn-{job_id}",
        )
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError:
            try:
                process = await self._finish_spawn_while_cancelled(spawn_task)
            except OSError:
                raise asyncio.CancelledError from None
            active = self._track_process(process, job_id, ownership_token)
            try:
                self._write_active_state(active)
            except OSError:
                logger.exception("Could not persist cancelled LTX child for job %s", job_id)
            active.cancellation_requested = True
            terminated = await self._terminate_while_cancelled(active)
            if self._termination_is_confirmed(active):
                self._clear_active_state(active)
                if self._active is active:
                    self._active = None
            if not terminated:
                raise self._termination_failed(active) from None
            raise
        except OSError as exc:
            raise DwellError(
                "runtime_not_available",
                "The local LTX runtime could not be started.",
                details={"reason": str(exc)},
                status_code=503,
            ) from exc

        active = self._track_process(process, job_id, ownership_token)

        try:
            try:
                self._write_active_state(active)
            except OSError as exc:
                raise DwellError(
                    "runtime_not_available",
                    "Dwell could not persist the active LTX child process.",
                    details={"reason": str(exc)},
                    status_code=500,
                ) from exc

            stdout_bytes, stderr_bytes = await self._communicate(active, cancel_event)
            if active.cancellation_requested:
                raise asyncio.CancelledError
            if self._process_group_alive(active.pgid):
                if not await self._terminate(active):
                    raise self._termination_failed(active)
                raise DwellError(
                    "generation_failed",
                    "The LTX launcher exited while a child process remained active.",
                    details={
                        "job_id": active.job_id,
                        "pid": active.process.pid,
                        "pgid": active.pgid,
                        "sid": active.sid,
                    },
                    status_code=500,
                )
        except asyncio.CancelledError:
            active.cancellation_requested = True
            if not await self._terminate_while_cancelled(active):
                raise self._termination_failed(active) from None
            raise
        except BaseException as exc:
            if self._active_is_live(active) and not await self._terminate(active):
                raise self._termination_failed(active) from exc
            raise
        finally:
            if self._termination_is_confirmed(active):
                self._clear_active_state(active)
                if self._active is active:
                    self._active = None

        exit_code = process.returncode
        if exit_code is None:
            # communicate() only returns after process exit. Keep this explicit
            # so a non-conforming injected process cannot make a result truthful.
            raise DwellError(
                "generation_failed",
                "The LTX child did not report an exit status.",
                status_code=500,
            )
        return (
            exit_code,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
            time.monotonic() - started,
        )

    def _track_process(
        self,
        process: asyncio.subprocess.Process,
        job_id: str,
        ownership_token: str,
    ) -> _ActiveGeneration:
        communicate_task = asyncio.create_task(
            process.communicate(),
            name=f"dwell-ltx-communicate-{job_id}",
        )
        active = _ActiveGeneration(
            job_id=job_id,
            ownership_token=ownership_token,
            process=process,
            pgid=process.pid,
            sid=process.pid,
            communicate_task=communicate_task,
            started_at=datetime.now(UTC).isoformat(),
        )
        self._active = active
        return active

    @staticmethod
    async def _finish_spawn_while_cancelled(
        spawn_task: asyncio.Task[asyncio.subprocess.Process],
    ) -> asyncio.subprocess.Process:
        while not spawn_task.done():
            try:
                await asyncio.shield(spawn_task)
            except asyncio.CancelledError:
                continue
        return spawn_task.result()

    async def _communicate(
        self,
        active: _ActiveGeneration,
        cancel_event: asyncio.Event | None,
    ) -> tuple[bytes, bytes]:
        if cancel_event is None:
            return await asyncio.shield(active.communicate_task)

        cancel_task = asyncio.create_task(
            cancel_event.wait(),
            name=f"dwell-ltx-cancel-event-{active.job_id}",
        )
        try:
            done, _pending = await asyncio.wait(
                (active.communicate_task, cancel_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_task in done and cancel_task.result():
                active.cancellation_requested = True
                if not await self._terminate(active):
                    raise self._termination_failed(active)
                await asyncio.shield(active.communicate_task)
                raise asyncio.CancelledError
            return active.communicate_task.result()
        finally:
            if not cancel_task.done():
                cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)

    async def _terminate_while_cancelled(self, active: _ActiveGeneration) -> bool:
        termination = asyncio.create_task(
            self._terminate(active),
            name=f"dwell-ltx-terminate-{active.job_id}",
        )
        while not termination.done():
            try:
                await asyncio.shield(termination)
            except asyncio.CancelledError:
                # A repeated task cancellation must not abandon the process
                # group. _terminate itself has strict TERM/KILL time bounds.
                continue
        return termination.result()

    async def _terminate(self, active: _ActiveGeneration) -> bool:
        async with active.terminate_lock:
            if self._termination_is_confirmed(active):
                return True

            self._signal_group(active.pgid, signal.SIGTERM)
            if await self._wait_for_exit(active, self.termination_grace_seconds):
                return True

            self._signal_group(active.pgid, signal.SIGKILL)
            return await self._wait_for_exit(active, self.kill_wait_seconds)

    @staticmethod
    def _signal_group(pgid: int, requested_signal: signal.Signals) -> bool:
        if pgid <= 1:
            return False
        try:
            os.killpg(pgid, requested_signal)
        except ProcessLookupError:
            return True
        except OSError as exc:
            logger.error(
                "Could not signal process group %d with %s: %s",
                pgid,
                requested_signal.name,
                exc,
            )
            return False
        return True

    async def _wait_for_exit(self, active: _ActiveGeneration, wait_seconds: float) -> bool:
        deadline = time.monotonic() + wait_seconds
        while True:
            if self._termination_is_confirmed(active):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if active.communicate_task.done():
                await asyncio.sleep(min(0.05, remaining))
                continue
            try:
                await asyncio.wait_for(
                    asyncio.shield(active.communicate_task),
                    timeout=min(0.05, remaining),
                )
            except TimeoutError:
                continue
            except (BrokenPipeError, ConnectionResetError):
                await asyncio.sleep(min(0.05, remaining))

    async def _validate_mp4(
        self,
        output: Path,
        *,
        expected_width: int,
        expected_height: int,
        environment: dict[str, str],
    ) -> None:
        command = [
            self._resolved_tool(self.ffprobe_binary) or self.ffprobe_binary,
            "-v",
            "error",
            "-nostdin",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,width,height",
            "-of",
            "json",
            str(output),
        ]
        spawn_task = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                cwd=str(output.parent),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                close_fds=True,
            ),
            name="dwell-ffprobe-spawn",
        )
        try:
            process = await asyncio.shield(spawn_task)
        except asyncio.CancelledError:
            try:
                process = await self._finish_spawn_while_cancelled(spawn_task)
            except OSError:
                raise asyncio.CancelledError from None
            communicate_task = asyncio.create_task(process.communicate())
            if not await self._terminate_untracked_while_cancelled(process, communicate_task):
                raise self._untracked_termination_failed("ffprobe", process.pid) from None
            raise
        except OSError as exc:
            raise DwellError(
                "runtime_not_available",
                "ffprobe could not be started for local output validation.",
                details={"reason": str(exc)},
                status_code=503,
            ) from exc

        communicate_task = asyncio.create_task(process.communicate())
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=self.ffprobe_timeout_seconds,
            )
        except TimeoutError as exc:
            if not await self._terminate_untracked(process, communicate_task):
                raise self._untracked_termination_failed("ffprobe", process.pid) from exc
            raise DwellError(
                "generation_failed",
                "ffprobe timed out while validating the generated MP4.",
                status_code=500,
            ) from exc
        except asyncio.CancelledError:
            if not await self._terminate_untracked_while_cancelled(process, communicate_task):
                raise self._untracked_termination_failed("ffprobe", process.pid) from None
            raise

        if self._process_group_alive(process.pid):
            if not await self._terminate_untracked(process, communicate_task):
                raise self._untracked_termination_failed("ffprobe", process.pid)
            raise DwellError(
                "generation_failed",
                "ffprobe exited while a descendant process remained active.",
                details={"pid": process.pid, "pgid": process.pid},
                status_code=500,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise DwellError(
                "generation_failed",
                "The LTX runtime did not produce a valid MP4.",
                details={
                    "ffprobe_exit_code": process.returncode,
                    "stderr": self._failure_output(stderr),
                },
                status_code=500,
            )

        try:
            probe = json.loads(stdout)
            streams = probe["streams"]
            stream = streams[0]
            width = int(stream["width"])
            height = int(stream["height"])
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DwellError(
                "generation_failed",
                "ffprobe found no readable video stream in the generated MP4.",
                details={
                    "ffprobe_stdout": self._failure_output(stdout),
                    "ffprobe_stderr": self._failure_output(stderr),
                },
                status_code=500,
            ) from exc

        if width != expected_width or height != expected_height:
            raise DwellError(
                "generation_failed",
                "The generated MP4 dimensions do not match the request.",
                details={
                    "expected": {"width": expected_width, "height": expected_height},
                    "actual": {"width": width, "height": height},
                },
                status_code=500,
            )

    async def _terminate_untracked(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[tuple[bytes, bytes]],
    ) -> bool:
        if self._untracked_termination_is_confirmed(process):
            return True
        pgid = process.pid
        self._signal_group(pgid, signal.SIGTERM)
        if await self._wait_for_untracked_exit(
            process,
            communicate_task,
            self.termination_grace_seconds,
        ):
            return True
        self._signal_group(pgid, signal.SIGKILL)
        return await self._wait_for_untracked_exit(
            process,
            communicate_task,
            self.kill_wait_seconds,
        )

    async def _terminate_untracked_while_cancelled(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[tuple[bytes, bytes]],
    ) -> bool:
        termination = asyncio.create_task(
            self._terminate_untracked(process, communicate_task),
            name="dwell-ffprobe-terminate",
        )
        while not termination.done():
            try:
                await asyncio.shield(termination)
            except asyncio.CancelledError:
                continue
        return termination.result()

    async def _wait_for_untracked_exit(
        self,
        process: asyncio.subprocess.Process,
        communicate_task: asyncio.Task[tuple[bytes, bytes]],
        wait_seconds: float,
    ) -> bool:
        deadline = time.monotonic() + wait_seconds
        while True:
            if self._untracked_termination_is_confirmed(process):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if communicate_task.done():
                await asyncio.sleep(min(0.05, remaining))
                continue
            try:
                await asyncio.wait_for(
                    asyncio.shield(communicate_task),
                    timeout=min(0.05, remaining),
                )
            except TimeoutError:
                continue
            except (BrokenPipeError, ConnectionResetError):
                await asyncio.sleep(min(0.05, remaining))

    def _untracked_termination_is_confirmed(
        self,
        process: asyncio.subprocess.Process,
    ) -> bool:
        return process.returncode is not None and not self._process_group_alive(process.pid)

    def _command(
        self,
        request: VideoGenerationRequest,
        model_path: Path,
        gemma_path: Path,
        output_path: Path,
    ) -> list[str]:
        command = [
            self._resolved_tool(self.uv_binary) or self.uv_binary,
            "run",
            "--project",
            str(self.runtime_dir.resolve()),
            "--offline",
            "--no-sync",
            "--no-python-downloads",
            "ltx-2-mlx",
            "generate",
            "--model",
            str(model_path),
            "--gemma",
            str(gemma_path),
            "--distilled",
            "--low-ram",
            "--quiet",
            "--prompt",
            request.prompt,
            "--output",
            str(output_path),
            "--height",
            str(request.height),
            "--width",
            str(request.width),
            "--frames",
            str(request.frames),
            "--frame-rate",
            format(request.fps, "g"),
        ]
        if request.seed is not None:
            command.extend(("--seed", str(request.seed)))
        return command

    @staticmethod
    def _redacted_command(command: list[str]) -> str:
        redacted = list(command)
        try:
            prompt_index = redacted.index("--prompt") + 1
        except ValueError:
            pass
        else:
            if prompt_index < len(redacted):
                redacted[prompt_index] = "<redacted>"
        return shlex.join(redacted)

    @staticmethod
    def _failure_output(value: str) -> str:
        if len(value) <= _FAILURE_OUTPUT_LIMIT:
            return value
        omitted = len(value) - _FAILURE_OUTPUT_LIMIT
        return f"{value[:_FAILURE_OUTPUT_LIMIT]}\n... <{omitted} characters omitted>"

    def _validated_model_paths(self, model_id: str, raw_path: Path) -> tuple[Path, Path]:
        try:
            model_path = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise self._incomplete_model(model_id, (str(raw_path),)) from exc
        if not model_path.is_dir():
            raise self._incomplete_model(model_id, (str(model_path),))

        missing = [name for name in _DISTILLED_FILES if not self._nonempty_file(model_path / name)]
        gemma_candidate = model_path / _GEMMA_DIRECTORY
        try:
            gemma_path = gemma_candidate.resolve(strict=True)
        except (OSError, RuntimeError):
            gemma_path = gemma_candidate
        if not gemma_path.is_dir():
            missing.append(_GEMMA_DIRECTORY)
        else:
            missing.extend(
                f"{_GEMMA_DIRECTORY}/{name}"
                for name in _GEMMA_FILES
                if not self._nonempty_file(gemma_path / name)
            )
            if not self._gemma_weights_complete(gemma_path):
                missing.append(f"{_GEMMA_DIRECTORY}/<complete safetensors set>")

        if missing:
            raise self._incomplete_model(model_id, tuple(missing))
        return model_path, gemma_path

    @staticmethod
    def _nonempty_file(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @classmethod
    def _gemma_weights_complete(cls, gemma_path: Path) -> bool:
        weights = sorted(
            path for path in gemma_path.glob("*.safetensors") if cls._nonempty_file(path)
        )
        if not weights:
            return False

        indexes = sorted(gemma_path.glob("*.safetensors.index.json"))
        if indexes:
            for index in indexes:
                try:
                    payload = json.loads(index.read_text(encoding="utf-8"))
                    referenced = set(payload["weight_map"].values())
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if referenced and all(
                    isinstance(name, str)
                    and Path(name).name == name
                    and name.endswith(".safetensors")
                    and cls._nonempty_file(gemma_path / name)
                    for name in referenced
                ):
                    return True
            return False

        shard_groups: dict[tuple[str, int], set[int]] = {}
        unsharded = 0
        for weight in weights:
            match = _GEMMA_SHARD.fullmatch(weight.name)
            if match is None:
                unsharded += 1
                continue
            number = int(match.group("number"))
            total = int(match.group("total"))
            shard_groups.setdefault((match.group("prefix"), total), set()).add(number)

        if shard_groups:
            if unsharded or len(shard_groups) != 1:
                return False
            (_prefix, total), numbers = next(iter(shard_groups.items()))
            return total > 0 and numbers == set(range(1, total + 1))
        return len(weights) == 1

    @staticmethod
    def _incomplete_model(model_id: str, missing: tuple[str, ...]) -> DwellError:
        return DwellError(
            "model_not_installed",
            f"Model '{model_id}' is not installed locally.",
            details={"missing_files": list(missing)},
            status_code=409,
        )

    def _job_paths(self, job_id: str) -> tuple[Path, Path, Path]:
        job_directory = self.config.tmp_dir.absolute() / "ltx-jobs" / job_id
        temporary_output = job_directory / "output.mp4"
        final_output = self.config.outputs_dir.absolute() / "video" / f"{job_id}.mp4"
        return job_directory, temporary_output, final_output

    @staticmethod
    def _path_entry_exists(path: Path) -> bool:
        return os.path.lexists(path)

    @staticmethod
    def _validate_request(request: VideoGenerationRequest, job_id: str) -> None:
        if not _JOB_ID.fullmatch(job_id):
            raise DwellError(
                "invalid_request",
                "The job ID is not safe for local output paths.",
                status_code=422,
            )
        if request.width % 64 or request.height % 64:
            raise DwellError(
                "invalid_request",
                "The distilled LTX runtime requires width and height divisible by 64.",
                details={"width": request.width, "height": request.height},
                status_code=422,
            )

    @staticmethod
    def _validate_regular_output(output: Path) -> None:
        try:
            metadata = output.lstat()
        except FileNotFoundError as exc:
            raise DwellError(
                "generation_failed",
                "The LTX runtime exited successfully but produced no MP4.",
                status_code=500,
            ) from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
            raise DwellError(
                "generation_failed",
                "The LTX runtime produced an empty or unsafe MP4 output.",
                status_code=500,
            )

    @staticmethod
    def _publish_no_clobber(source: Path, destination: Path, job_id: str) -> None:
        try:
            os.link(source, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise LTXSubprocessRuntime._output_exists(job_id) from exc
        except OSError as exc:
            raise DwellError(
                "generation_failed",
                "The verified MP4 could not be published atomically.",
                details={"reason": str(exc)},
                status_code=500,
            ) from exc

    @staticmethod
    def _output_exists(job_id: str) -> DwellError:
        return DwellError(
            "output_exists",
            f"Output for job '{job_id}' already exists; refusing to overwrite it.",
            status_code=409,
        )

    def _runtime_problems(self) -> list[str]:
        problems: list[str] = []
        if not self.runtime_dir.is_dir():
            problems.append(f"runtime directory is missing: {self.runtime_dir}")
        elif not (self.runtime_dir / "pyproject.toml").is_file():
            problems.append(f"runtime pyproject is missing: {self.runtime_dir / 'pyproject.toml'}")
        runtime_cli = self.runtime_dir / ".venv" / "bin" / "ltx-2-mlx"
        if not runtime_cli.is_file() or not os.access(runtime_cli, os.X_OK):
            problems.append(f"runtime CLI is unavailable: {runtime_cli}")
        if self._resolved_tool(self.uv_binary) is None:
            problems.append(f"uv executable is unavailable: {self.uv_binary}")
        if self._resolved_tool(self.ffmpeg_binary) is None:
            problems.append(f"ffmpeg executable is unavailable: {self.ffmpeg_binary}")
        if self._resolved_tool(self.ffprobe_binary) is None:
            problems.append(f"ffprobe executable is unavailable: {self.ffprobe_binary}")
        _persisted, state_error = self._persisted_generation_status()
        if state_error is not None:
            problems.append(state_error.message)
        return problems

    def _ensure_runtime_available(self) -> None:
        problems = self._runtime_problems()
        if problems:
            raise DwellError(
                "runtime_not_available",
                "The local LTX runtime is unavailable.",
                details={"problems": problems},
                status_code=503,
            )

    @staticmethod
    def _resolved_tool(binary: str) -> str | None:
        if os.sep in binary or (os.altsep is not None and os.altsep in binary):
            path = Path(binary).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
            return None
        return shutil.which(binary)

    def _active_jobs(self) -> tuple[str, ...]:
        active = self._active
        if active is not None and self._active_is_live(active):
            return (active.job_id,)
        persisted, _state_error = self._persisted_generation_status()
        return (persisted.job_id,) if persisted is not None else ()

    def _active_is_live(self, active: _ActiveGeneration) -> bool:
        return active.process.returncode is None or self._process_group_alive(active.pgid)

    def _termination_is_confirmed(self, active: _ActiveGeneration) -> bool:
        return active.process.returncode is not None and not self._process_group_alive(active.pgid)

    @staticmethod
    def _process_group_alive(pgid: int) -> bool:
        if pgid <= 1:
            return False
        try:
            result = subprocess.run(
                ["ps", "-axo", "pgid=,stat="],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None and result.returncode == 0:
            for line in result.stdout.splitlines():
                fields = line.split(None, 1)
                if len(fields) != 2:
                    continue
                try:
                    member_pgid = int(fields[0])
                except ValueError:
                    continue
                if member_pgid == pgid and not fields[1].startswith("Z"):
                    return True
            return False
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _persisted_generation_status(
        self,
        *,
        clean_stale: bool = False,
    ) -> tuple[_PersistedGeneration | None, DwellError | None]:
        try:
            persisted = self._read_persisted_generation()
        except DwellError as exc:
            return None, exc
        if persisted is None:
            return None, None
        if self._process_group_alive(persisted.pgid):
            return persisted, None
        if clean_stale:
            self._clear_persisted_generation(persisted)
        return None, None

    def _read_persisted_generation(self) -> _PersistedGeneration | None:
        try:
            metadata = self.active_state_file.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._unsafe_state("could not be inspected") from exc
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _ACTIVE_STATE_LIMIT:
            raise self._unsafe_state("is not a bounded regular file")
        try:
            payload = json.loads(self.active_state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise self._unsafe_state("is unreadable") from exc
        if not isinstance(payload, dict):
            raise self._unsafe_state("does not contain an object")

        version = payload.get("version")
        runtime_id = payload.get("runtime_id")
        job_id = payload.get("job_id")
        ownership_token = payload.get("ownership_token")
        owner_pid = payload.get("owner_pid")
        pid = payload.get("pid")
        pgid = payload.get("pgid")
        sid = payload.get("sid")
        started_at = payload.get("started_at")
        identifiers = (owner_pid, pid, pgid, sid)
        valid = (
            type(version) is int
            and version == 1
            and runtime_id == self.runtime_id
            and isinstance(job_id, str)
            and _JOB_ID.fullmatch(job_id) is not None
            and isinstance(ownership_token, str)
            and _OWNERSHIP_TOKEN.fullmatch(ownership_token) is not None
            and all(type(value) is int for value in identifiers)
            and owner_pid > 1
            and pid > 1
            and pgid == pid
            and sid == pid
            and (started_at is None or isinstance(started_at, str))
        )
        if not valid:
            raise self._unsafe_state("has an invalid or legacy ownership record")
        return _PersistedGeneration(
            job_id=job_id,
            ownership_token=ownership_token,
            owner_pid=owner_pid,
            pid=pid,
            pgid=pgid,
            sid=sid,
            started_at=started_at,
        )

    def _clear_persisted_generation(self, expected: _PersistedGeneration) -> None:
        try:
            current = self._read_persisted_generation()
        except DwellError:
            return
        if current != expected:
            return
        try:
            self.active_state_file.unlink()
        except FileNotFoundError:
            pass

    def _assert_no_persisted_generation(self) -> None:
        persisted, state_error = self._persisted_generation_status(clean_stale=True)
        if state_error is not None:
            raise state_error
        if persisted is not None:
            raise self._busy_error(persisted)

    @contextmanager
    def _exclusive_process_lock(self) -> Iterator[None]:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        with self.process_lock_file.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise self._busy_error() from exc
            try:
                self._assert_no_persisted_generation()
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _busy_error(persisted: _PersistedGeneration | None = None) -> DwellError:
        details = None
        if persisted is not None:
            details = {
                "job_id": persisted.job_id,
                "owner_pid": persisted.owner_pid,
                "pid": persisted.pid,
                "pgid": persisted.pgid,
                "sid": persisted.sid,
            }
        return DwellError(
            "server_busy",
            "The LTX runtime already has an active generation process.",
            details=details,
            status_code=409,
        )

    def _unsafe_state(self, reason: str) -> DwellError:
        return DwellError(
            "runtime_not_available",
            f"The recorded LTX process state {reason}; refusing unsafe generation.",
            details={"path": str(self.active_state_file)},
            status_code=503,
        )

    @staticmethod
    def _termination_failed(active: _ActiveGeneration) -> DwellError:
        return DwellError(
            "generation_failed",
            "The LTX process group did not exit after bounded TERM/KILL cancellation.",
            details={
                "job_id": active.job_id,
                "pid": active.process.pid,
                "pgid": active.pgid,
                "sid": active.sid,
            },
            status_code=500,
        )

    @staticmethod
    def _untracked_termination_failed(process_name: str, pgid: int) -> DwellError:
        return DwellError(
            "generation_failed",
            f"The {process_name} process group did not exit after bounded TERM/KILL.",
            details={"pid": pgid, "pgid": pgid},
            status_code=500,
        )

    def _write_active_state(self, active: _ActiveGeneration) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        if self._path_entry_exists(self.active_state_file):
            raise FileExistsError(
                f"refusing to replace existing active process state: {self.active_state_file}"
            )
        actual_pgid = os.getpgid(active.process.pid)
        actual_sid = os.getsid(active.process.pid)
        if actual_pgid != active.process.pid or actual_sid != active.process.pid:
            raise OSError(
                "LTX child does not own the expected isolated process session "
                f"(pid={active.process.pid}, pgid={actual_pgid}, sid={actual_sid})"
            )
        active.pgid = actual_pgid
        active.sid = actual_sid
        contents = (
            json.dumps(
                {
                    "version": 1,
                    "runtime_id": self.runtime_id,
                    "job_id": active.job_id,
                    "ownership_token": active.ownership_token,
                    "owner_pid": os.getpid(),
                    "pid": active.process.pid,
                    "pgid": active.pgid,
                    "sid": active.sid,
                    "started_at": active.started_at,
                },
                sort_keys=True,
            )
            + "\n"
        )
        temporary = self.active_state_file.with_name(
            f".{self.active_state_file.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.active_state_file)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def _clear_active_state(self, active: _ActiveGeneration) -> None:
        try:
            payload = json.loads(self.active_state_file.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        if (
            payload.get("pid") != active.process.pid
            or payload.get("job_id") != active.job_id
            or payload.get("ownership_token") != active.ownership_token
        ):
            return
        try:
            self.active_state_file.unlink()
        except FileNotFoundError:
            pass


# Compatibility names for callers that distinguish runtime and engine roles.
LTXRuntime = LTXSubprocessRuntime
LTXVideoEngine = LTXSubprocessRuntime


__all__ = [
    "LTXGenerationResult",
    "LTXRuntime",
    "LTXSubprocessRuntime",
    "LTXVideoEngine",
]
