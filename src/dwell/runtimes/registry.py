from __future__ import annotations

import asyncio
import os
import platform
import shutil
from importlib import metadata, util

from dwell.config import DwellConfig
from dwell.domain import Modality, ModelDefinition
from dwell.errors import DwellError
from dwell.registry import ModelRegistry
from dwell.runtimes.base import ModelRuntime, TextEngine, VideoEngine

_MLX_LM_VERSION = "0.31.3"
_MAX_INTERACTIVE_WAITERS = 4
_INTERACTIVE_WAIT_SECONDS = 120.0


class RuntimeRegistry:
    """Software runtimes are registered independently from model metadata."""

    def __init__(
        self,
        config: DwellConfig | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.config = config or DwellConfig.from_env()
        self.model_registry = registry or ModelRegistry.load(self.config)
        self._instances: dict[str, ModelRuntime] = {}
        self._inference_lock = asyncio.Lock()
        self._interactive_waiters = 0

    def is_available(self, runtime_id: str) -> bool:
        if runtime_id == "mlx-lm":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                return False
            try:
                return (
                    metadata.version("mlx-lm") == _MLX_LM_VERSION
                    and util.find_spec("mlx_lm.server") is not None
                )
            except (ImportError, ValueError, metadata.PackageNotFoundError):
                return False
        if runtime_id != "ltx-2-mlx":
            return False
        root = self.config.runtimes_dir / runtime_id
        cli = root / ".venv" / "bin" / "ltx-2-mlx"
        return (
            (root / "pyproject.toml").is_file()
            and cli.is_file()
            and os.access(cli, os.X_OK)
            and shutil.which("uv") is not None
            and shutil.which("ffmpeg") is not None
            and shutil.which("ffprobe") is not None
        )

    def get(self, runtime_id: str) -> ModelRuntime:
        if runtime_id not in {"ltx-2-mlx", "mlx-lm"}:
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{runtime_id}' is not registered.",
                status_code=503,
            )
        if runtime_id not in self._instances:
            if runtime_id == "mlx-lm":
                from dwell.runtimes.mlx_lm import MLXLMRuntime

                self._instances[runtime_id] = MLXLMRuntime(
                    self.config,
                    model_resolver=self.model_registry.resolve_local_path,
                )
            else:
                from dwell.runtimes.ltx import LTXSubprocessRuntime

                self._instances[runtime_id] = LTXSubprocessRuntime(
                    self.config,
                    model_resolver=self.model_registry.resolve_local_path,
                )
        return self._instances[runtime_id]

    def text_engine(self, model: ModelDefinition | str) -> TextEngine:
        definition = self.model_registry.get(model) if isinstance(model, str) else model
        if definition.modality != Modality.TEXT:
            raise DwellError(
                "invalid_request",
                f"Model '{definition.id}' is not a text model.",
                status_code=422,
            )
        runtime = self.get(definition.runtime)
        if not isinstance(runtime, TextEngine):
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{definition.runtime}' has no text engine.",
                status_code=503,
            )
        return runtime

    def video_engine(self, model: ModelDefinition | str) -> VideoEngine:
        definition = self.model_registry.get(model) if isinstance(model, str) else model
        if definition.modality != Modality.VIDEO:
            raise DwellError(
                "invalid_request",
                f"Model '{definition.id}' is not a video model.",
                status_code=422,
            )
        runtime = self.get(definition.runtime)
        if not isinstance(runtime, VideoEngine):
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{definition.runtime}' has no video engine.",
                status_code=503,
            )
        return runtime

    async def acquire_inference(self, *, wait: bool) -> None:
        if wait or not self._inference_lock.locked():
            await self._inference_lock.acquire()
            return
        if self._interactive_waiters >= _MAX_INTERACTIVE_WAITERS:
            raise DwellError(
                "server_busy",
                "The local text inference queue is full.",
                status_code=429,
            )
        self._interactive_waiters += 1
        try:
            await asyncio.wait_for(
                self._inference_lock.acquire(),
                timeout=_INTERACTIVE_WAIT_SECONDS,
            )
        except TimeoutError as exc:
            raise DwellError(
                "server_busy",
                "Timed out waiting for the Apple GPU inference lease.",
                status_code=429,
            ) from exc
        finally:
            self._interactive_waiters -= 1

    async def release_inference(self) -> None:
        if self._inference_lock.locked():
            self._inference_lock.release()

    async def release_resident_except(self, runtime_id: str) -> tuple[str, ...]:
        """Free other resident models while holding the shared inference lease."""

        released: list[str] = []
        for other_id, runtime in list(self._instances.items()):
            if other_id == runtime_id:
                continue
            current = await runtime.status()
            if current.active_jobs:
                raise DwellError(
                    "server_busy",
                    f"Runtime '{other_id}' still owns an active inference request.",
                    status_code=429,
                )
            if current.loaded_models:
                await runtime.release()
                released.extend(current.loaded_models)
        return tuple(released)

    async def release_all(self) -> None:
        for runtime in list(self._instances.values()):
            await runtime.release()
