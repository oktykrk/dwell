from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dwell.config import DwellConfig
from dwell.domain import ModelDefinition, RuntimeCapabilities
from dwell.errors import DwellError
from dwell.registry import ModelRegistry
from dwell.runtimes import registry as registry_module
from dwell.runtimes.base import ModelRuntime, RuntimeStatus, TextStream
from dwell.runtimes.registry import RuntimeRegistry


class FakeResidentRuntime(ModelRuntime):
    runtime_id = "resident"
    capabilities = RuntimeCapabilities(persistent_loading=True)

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.release_calls = 0

    async def validate_installation(self) -> RuntimeStatus:
        return await self.status()

    async def prepare(
        self,
        model: ModelDefinition,
        model_path: Path,
    ) -> RuntimeStatus:
        del model, model_path
        return await self.status()

    async def release(self, model: ModelDefinition | None = None) -> RuntimeStatus:
        del model
        self.release_calls += 1
        self.model_id = ""
        return await self.status()

    async def status(self) -> RuntimeStatus:
        loaded = (self.model_id,) if self.model_id else ()
        return RuntimeStatus(runtime_id=self.runtime_id, available=True, loaded_models=loaded)


def test_mlx_lm_availability_uses_import_free_package_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looked_up: list[str] = []

    monkeypatch.setattr(registry_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(registry_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(registry_module.metadata, "version", lambda _name: "0.31.3")

    def find_spec(name: str) -> object:
        looked_up.append(name)
        if name == "mlx_lm.server":
            raise AssertionError("submodule lookup would import mlx_lm")
        return object()

    monkeypatch.setattr(registry_module.util, "find_spec", find_spec)
    config = DwellConfig(home=tmp_path)
    registry = RuntimeRegistry(config, ModelRegistry(config, definitions=[]))

    assert registry.is_available("mlx-lm") is True
    assert looked_up == ["mlx_lm"]


@pytest.mark.asyncio
async def test_shared_gpu_lease_serializes_parallel_interactive_inference(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path)
    registry = RuntimeRegistry(config, ModelRegistry(config, definitions=[]))
    await registry.acquire_inference(wait=False)

    waiter = asyncio.create_task(registry.acquire_inference(wait=False))
    await asyncio.sleep(0)
    assert waiter.done() is False

    await registry.release_inference()
    await waiter
    await registry.release_inference()


@pytest.mark.asyncio
async def test_shared_gpu_lease_rejects_an_overflowing_interactive_queue(
    tmp_path: Path,
) -> None:
    config = DwellConfig(home=tmp_path)
    registry = RuntimeRegistry(config, ModelRegistry(config, definitions=[]))
    registry._interactive_waiters = 4
    await registry.acquire_inference(wait=True)

    with pytest.raises(DwellError) as caught:
        await registry.acquire_inference(wait=False)

    assert caught.value.code == "server_busy"
    assert caught.value.status_code == 429
    await registry.release_inference()


@pytest.mark.asyncio
async def test_switching_modalities_releases_other_resident_models(tmp_path: Path) -> None:
    config = DwellConfig(home=tmp_path)
    registry = RuntimeRegistry(config, ModelRegistry(config, definitions=[]))
    resident = FakeResidentRuntime("local-coder")
    registry._instances[resident.runtime_id] = resident

    released = await registry.release_resident_except("ltx-2-mlx")

    assert released == ("local-coder",)
    assert resident.release_calls == 1


@pytest.mark.asyncio
async def test_text_stream_cleanup_survives_caller_cancellation() -> None:
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    lease_released = asyncio.Event()

    class BlockingIterator:
        def __aiter__(self):  # type: ignore[no-untyped-def]
            return self

        async def __anext__(self) -> bytes:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()

    async def release_lease() -> None:
        lease_released.set()

    stream = TextStream(BlockingIterator(), release_lease)  # type: ignore[arg-type]
    closing = asyncio.create_task(stream.aclose())
    await cleanup_started.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    allow_cleanup.set()
    await asyncio.wait_for(lease_released.wait(), timeout=0.2)
    await stream.aclose()
