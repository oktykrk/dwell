from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from dwell.domain import (
    GenerationResult,
    ModelDefinition,
    RuntimeCapabilities,
    VideoGenerationRequest,
)


class RuntimeStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    runtime_id: str
    available: bool
    loaded_models: tuple[str, ...] = ()
    active_jobs: tuple[str, ...] = ()
    message: str | None = None


class ModelRuntime(ABC):
    """Lifecycle contract kept separate from modality-specific inference."""

    runtime_id: str
    capabilities: RuntimeCapabilities

    @abstractmethod
    async def validate_installation(self) -> RuntimeStatus:
        """Inspect runtime software locally; implementations must not download."""

    @abstractmethod
    async def prepare(
        self,
        model: ModelDefinition,
        model_path: Path,
    ) -> RuntimeStatus:
        """Prepare a model, loading it only when persistent residency is real."""

    @abstractmethod
    async def release(self, model: ModelDefinition | None = None) -> RuntimeStatus:
        """Release a model, or all runtime resources when model is ``None``."""

    @abstractmethod
    async def status(self) -> RuntimeStatus:
        """Return truthful local runtime and residency state."""


class Engine(ABC):
    runtime_id: str
    capabilities: RuntimeCapabilities


class VideoEngine(Engine):
    @abstractmethod
    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> GenerationResult:
        """Generate one video for a previously validated, installed model."""
