from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

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


class TextStream:
    """One upstream text stream plus an idempotent asynchronous closer."""

    def __init__(
        self,
        iterator: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]],
    ) -> None:
        self._iterator = iterator
        self._close = close
        self._close_task: asyncio.Task[None] | None = None

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._iterator

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._cleanup(),
                name="dwell-text-stream-close",
            )
        await asyncio.shield(self._close_task)

    async def _cleanup(self) -> None:
        try:
            iterator_close = getattr(self._iterator, "aclose", None)
            if callable(iterator_close):
                await iterator_close()
        finally:
            await self._close()

    async def __aenter__(self) -> TextStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()


class TextEngine(Engine):
    """OpenAI-compatible text generation backed by one installed model."""

    @abstractmethod
    async def complete(
        self,
        model: ModelDefinition,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Return one complete chat-completion response."""

    @abstractmethod
    async def open_stream(
        self,
        model: ModelDefinition,
        payload: dict[str, Any],
    ) -> TextStream:
        """Open a validated SSE stream that the caller must close."""


class VideoEngine(Engine):
    @abstractmethod
    async def generate(
        self,
        request: VideoGenerationRequest,
        job_id: str,
        cancel_event: asyncio.Event | None = None,
    ) -> GenerationResult:
        """Generate one video for a previously validated, installed model."""
