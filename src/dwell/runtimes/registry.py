from __future__ import annotations

import os
import shutil

from dwell.config import DwellConfig
from dwell.domain import Modality, ModelDefinition
from dwell.errors import DwellError
from dwell.registry import ModelRegistry
from dwell.runtimes.base import ModelRuntime, VideoEngine


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

    def is_available(self, runtime_id: str) -> bool:
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
        if runtime_id != "ltx-2-mlx":
            raise DwellError(
                "runtime_not_available",
                f"Runtime '{runtime_id}' is not registered.",
                status_code=503,
            )
        if runtime_id not in self._instances:
            from dwell.runtimes.ltx import LTXSubprocessRuntime

            self._instances[runtime_id] = LTXSubprocessRuntime(
                self.config,
                model_resolver=self.model_registry.resolve_local_path,
            )
        return self._instances[runtime_id]

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

    async def release_all(self) -> None:
        for runtime in list(self._instances.values()):
            await runtime.release()
