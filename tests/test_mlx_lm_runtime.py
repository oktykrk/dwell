from __future__ import annotations

import asyncio
import io
import os
import threading
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from dwell.config import DwellConfig
from dwell.domain import (
    Modality,
    ModelDefinition,
    ModelProfile,
    RuntimeCapabilities,
    WeightSource,
)
from dwell.errors import DwellError
from dwell.runtimes.mlx_lm import MLXLMRuntime
from dwell.runtimes.mlx_server import (
    _authenticated_handler,
    _GenerationController,
    _track_response_generator,
)


def _model() -> ModelDefinition:
    return ModelDefinition(
        id="local-coder",
        family="qwen3-coder",
        version="test",
        modality=Modality.TEXT,
        runtime="mlx-lm",
        weights=WeightSource(provider="local"),
        profile=ModelProfile(quantization="4bit"),
        capabilities=RuntimeCapabilities(
            persistent_loading=True,
            cancellation=True,
            streaming=True,
        ),
    )


def _read_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"filePath": {"type": "string"}},
                "required": ["filePath"],
            },
        },
    }


class FakeProcess:
    def __init__(self, *, stubborn: bool = False) -> None:
        self.pid = 4242
        self._returncode: int | None = None
        self.stubborn = stubborn
        self.terminate_calls = 0
        self.kill_calls = 0

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.stubborn:
            self._returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self._returncode = -9

    async def wait(self) -> int:
        if self._returncode is None:
            await __import__("asyncio").Future()
        assert self._returncode is not None
        return self._returncode


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        *,
        payload: Any = None,
        body: bytes = b"",
        chunks: tuple[bytes, ...] = (),
    ) -> None:
        self.status_code = status_code
        self.payload = payload
        self.body = body
        self.chunks = chunks
        self.closed = False

    def json(self) -> Any:
        return self.payload

    async def aread(self) -> bytes:
        return self.body

    async def aclose(self) -> None:
        self.closed = True

    def aiter_raw(self) -> AsyncIterator[bytes]:
        async def iterate() -> AsyncIterator[bytes]:
            for chunk in self.chunks:
                yield chunk

        return iterate()


class FakeClient:
    def __init__(self) -> None:
        self.get_responses: list[FakeResponse] = [FakeResponse()]
        self.post_response = FakeResponse(
            payload={
                "id": "chatcmpl-test",
                "model": "default_model",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            }
        )
        self.cancel_response = FakeResponse(payload={"status": "stopped", "requests": 1})
        self.stream_response = FakeResponse(
            chunks=(b'data: {"choices":[]}\n\n', b"data: [DONE]\n\n")
        )
        self.post_payloads: list[dict[str, Any]] = []
        self.post_urls: list[str] = []
        self.stream_payloads: list[dict[str, Any]] = []
        self.factory_kwargs: dict[str, Any] = {}
        self.closed = False

    async def get(self, _url: str) -> FakeResponse:
        if len(self.get_responses) > 1:
            return self.get_responses.pop(0)
        return self.get_responses[0]

    async def post(self, url: str, *, json: Mapping[str, Any]) -> FakeResponse:
        self.post_urls.append(url)
        self.post_payloads.append(dict(json))
        return self.cancel_response if url == "/dwell/cancel" else self.post_response

    def build_request(self, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
        request = {"method": method, "url": url, **kwargs}
        self.stream_payloads.append(dict(kwargs["json"]))
        return request

    async def send(self, _request: Any, *, stream: bool = False) -> FakeResponse:
        assert stream is True
        return self.stream_response

    async def aclose(self) -> None:
        self.closed = True


class RuntimeFixture:
    def __init__(
        self,
        tmp_path: Path,
        *,
        process: FakeProcess | None = None,
        termination_grace_seconds: float = 0.1,
        cancel_ack_timeout_seconds: float = 0.1,
        transport_close_timeout_seconds: float = 0.1,
    ) -> None:
        self.config = DwellConfig(home=tmp_path / "dwell", mlx_lm_port=19189)
        self.model_path = tmp_path / "snapshot"
        self.model_path.mkdir()
        self.process = process or FakeProcess()
        self.client = FakeClient()
        self.spawn_command: tuple[str, ...] | None = None
        self.spawn_kwargs: dict[str, Any] = {}

        async def process_factory(*command: str, **kwargs: Any) -> FakeProcess:
            self.spawn_command = command
            self.spawn_kwargs = kwargs
            return self.process

        def client_factory(**kwargs: Any) -> FakeClient:
            self.client.factory_kwargs = kwargs
            return self.client

        self.runtime = MLXLMRuntime(
            self.config,
            model_resolver=lambda _model_id: self.model_path,
            python_binary="/test/python",
            client_factory=client_factory,
            process_factory=process_factory,
            installation_probe=lambda: True,
            port_probe=lambda _host, _port: True,
            readiness_timeout_seconds=0.5,
            readiness_poll_seconds=0.001,
            termination_grace_seconds=termination_grace_seconds,
            kill_wait_seconds=0.1,
            cancel_ack_timeout_seconds=cancel_ack_timeout_seconds,
            transport_close_timeout_seconds=transport_close_timeout_seconds,
        )


def test_capabilities_and_offline_installation_probe(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)

    assert fixture.runtime.runtime_id == "mlx-lm"
    assert fixture.runtime.capabilities.persistent_loading is True
    assert fixture.runtime.capabilities.streaming is True
    assert fixture.runtime.capabilities.cancellation is True
    assert fixture.runtime.capabilities.tool_calling is True


@pytest.mark.asyncio
async def test_validate_installation_is_truthful_without_starting_a_server(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    status = await fixture.runtime.validate_installation()

    assert status.available is True
    assert status.loaded_models == ()
    assert fixture.spawn_command is None

    unavailable = MLXLMRuntime(
        fixture.config,
        model_resolver=lambda _model_id: fixture.model_path,
        installation_probe=lambda: "pinned package missing",
    )
    status = await unavailable.validate_installation()
    assert status.available is False
    assert status.message == "pinned package missing"


@pytest.mark.asyncio
async def test_prepare_uses_exact_local_path_private_token_and_single_concurrency(
    tmp_path: Path,
) -> None:
    fixture = RuntimeFixture(tmp_path)
    status = await fixture.runtime.prepare(_model(), fixture.model_path)

    assert status.loaded_models == ("local-coder",)
    assert fixture.spawn_command == (
        "/test/python",
        "-m",
        "dwell.runtimes.mlx_server",
        "--model",
        str(fixture.model_path.resolve()),
        "--host",
        "127.0.0.1",
        "--port",
        "19189",
        "--max-tokens",
        "8192",
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--prompt-cache-size",
        "1",
    )
    environment = fixture.spawn_kwargs["env"]
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["DWELL_MLX_PARENT_PID"] == str(os.getpid())
    assert len(environment["DWELL_MLX_SERVER_TOKEN"]) == 64
    assert fixture.client.factory_kwargs["base_url"] == "http://127.0.0.1:19189"
    assert fixture.client.factory_kwargs["headers"] == {
        "Authorization": f"Bearer {environment['DWELL_MLX_SERVER_TOKEN']}"
    }

    released = await fixture.runtime.release(_model())
    assert released.loaded_models == ()
    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True


@pytest.mark.asyncio
async def test_completion_rewrites_only_the_runtime_model_and_rejects_loading_controls(
    tmp_path: Path,
) -> None:
    fixture = RuntimeFixture(tmp_path)
    payload = {
        "model": "local-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 128,
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }

    result = await fixture.runtime.complete(_model(), payload)

    assert result["model"] == "local-coder"
    assert fixture.client.post_payloads == [
        {
            **payload,
            "model": "default_model",
            "stream": False,
        }
    ]
    assert payload["model"] == "local-coder", "the caller's payload must not be mutated"

    for field in ("adapters", "draft_model", "trust_remote_code"):
        with pytest.raises(DwellError) as caught:
            await fixture.runtime.complete(_model(), {**payload, field: None})
        assert caught.value.code == "invalid_request"
        assert caught.value.status_code == 422

    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_stream_holds_the_single_request_lease_until_closed(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    payload = {
        "model": "local-coder",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    stream = await fixture.runtime.open_stream(_model(), payload)

    assert fixture.client.stream_payloads[0]["model"] == "default_model"
    assert fixture.client.stream_payloads[0]["stream"] is True
    with pytest.raises(DwellError) as caught:
        await fixture.runtime.complete(_model(), payload)
    assert caught.value.code == "server_busy"
    assert caught.value.status_code == 429

    chunks = [chunk async for chunk in stream]
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert fixture.client.stream_response.closed is True
    assert "/dwell/cancel" not in fixture.client.post_urls

    result = await fixture.runtime.complete(_model(), payload)
    assert result["choices"]
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_stream_finishes_at_done_without_waiting_for_transport_eof(tmp_path: Path) -> None:
    class KeepAliveResponse(FakeResponse):
        def aiter_raw(self) -> AsyncIterator[bytes]:
            async def iterate() -> AsyncIterator[bytes]:
                yield b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
                await asyncio.Future()

            return iterate()

    fixture = RuntimeFixture(tmp_path)
    fixture.client.stream_response = KeepAliveResponse()
    stream = await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "hello"}]},
    )

    async def collect() -> list[bytes]:
        return [chunk async for chunk in stream]

    chunks = await asyncio.wait_for(collect(), timeout=0.1)

    assert chunks[-1] == b"data: [DONE]\n\n"
    assert fixture.client.stream_response.closed is True
    assert "/dwell/cancel" not in fixture.client.post_urls
    assert fixture.runtime._request_lock.locked() is False
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_unacknowledged_stream_cancel_terminates_resident_server(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopping", "requests": 1})
    stream = await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long task"}]},
    )

    await stream.aclose()

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True
    assert fixture.client.stream_response.closed is True
    status = await fixture.runtime.status()
    assert status.loaded_models == ()
    assert status.active_jobs == ()


@pytest.mark.asyncio
async def test_shutdown_with_unacknowledged_stream_cancel_does_not_deadlock(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopping", "requests": 1})
    await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long task"}]},
    )

    await asyncio.wait_for(fixture.runtime.release(), timeout=0.2)

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True


@pytest.mark.asyncio
async def test_shutdown_continues_when_stream_transport_close_fails(tmp_path: Path) -> None:
    class FailingCloseResponse(FakeResponse):
        async def aclose(self) -> None:
            self.closed = True
            raise RuntimeError("socket close failed")

    fixture = RuntimeFixture(tmp_path)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopping", "requests": 1})
    fixture.client.stream_response = FailingCloseResponse(chunks=())
    await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long"}]},
    )

    status = await fixture.runtime.release()

    assert status.loaded_models == ()
    assert status.active_jobs == ()
    assert fixture.client.stream_response.closed is True
    assert fixture.client.closed is True
    assert fixture.process.terminate_calls == 1


@pytest.mark.asyncio
async def test_shutdown_bounds_a_hung_stream_transport_close(tmp_path: Path) -> None:
    class HangingCloseResponse(FakeResponse):
        async def aclose(self) -> None:
            await asyncio.Future()

    fixture = RuntimeFixture(tmp_path, transport_close_timeout_seconds=0.001)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopping", "requests": 1})
    fixture.client.stream_response = HangingCloseResponse(chunks=())
    await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long"}]},
    )

    status = await asyncio.wait_for(fixture.runtime.release(), timeout=0.2)

    assert status.loaded_models == ()
    assert status.active_jobs == ()
    assert fixture.client.closed is True
    assert fixture.process.terminate_calls == 1


@pytest.mark.asyncio
async def test_cancelled_shutdown_finishes_owned_teardown_before_unlocking(tmp_path: Path) -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class GatedCloseResponse(FakeResponse):
        async def aclose(self) -> None:
            close_started.set()
            await allow_close.wait()
            self.closed = True

    fixture = RuntimeFixture(tmp_path, transport_close_timeout_seconds=1.0)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopping", "requests": 1})
    fixture.client.stream_response = GatedCloseResponse(chunks=())
    await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long"}]},
    )

    release_task = asyncio.create_task(fixture.runtime.release())
    await asyncio.wait_for(close_started.wait(), timeout=0.1)
    release_task.cancel()
    await asyncio.sleep(0)

    assert fixture.process.returncode is None
    assert fixture.runtime._request_lock.locked() is True
    assert release_task.done() is False

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True
    assert fixture.runtime._request_lock.locked() is False


@pytest.mark.asyncio
async def test_zero_request_cancel_ack_restarts_after_stream_send_race(tmp_path: Path) -> None:
    sent = asyncio.Event()
    fixture = RuntimeFixture(tmp_path)
    fixture.client.cancel_response = FakeResponse(payload={"status": "stopped", "requests": 0})

    async def hanging_send(_request: Any, *, stream: bool = False) -> FakeResponse:
        assert stream is True
        sent.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    fixture.client.send = hanging_send  # type: ignore[method-assign]
    request_task = asyncio.create_task(
        fixture.runtime.open_stream(
            _model(),
            {"model": "local-coder", "messages": [{"role": "user", "content": "long"}]},
        )
    )
    await asyncio.wait_for(sent.wait(), timeout=0.1)
    request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True
    assert fixture.runtime._request_lock.locked() is False


@pytest.mark.asyncio
async def test_zero_request_cancel_ack_restarts_after_completion_send_race(
    tmp_path: Path,
) -> None:
    sent = asyncio.Event()
    fixture = RuntimeFixture(tmp_path)
    cancel_response = FakeResponse(payload={"status": "stopped", "requests": 0})

    async def hanging_post(url: str, *, json: Mapping[str, Any]) -> FakeResponse:
        fixture.client.post_urls.append(url)
        fixture.client.post_payloads.append(dict(json))
        if url == "/dwell/cancel":
            return cancel_response
        sent.set()
        await asyncio.Future()
        raise AssertionError("unreachable")

    fixture.client.post = hanging_post  # type: ignore[method-assign]
    request_task = asyncio.create_task(
        fixture.runtime.complete(
            _model(),
            {"model": "local-coder", "messages": [{"role": "user", "content": "long"}]},
        )
    )
    await asyncio.wait_for(sent.wait(), timeout=0.1)
    request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True
    assert fixture.runtime._request_lock.locked() is False


@pytest.mark.asyncio
async def test_cancel_ack_timeout_is_bounded_and_terminates_server(tmp_path: Path) -> None:
    class HangingCancelResponse(FakeResponse):
        async def aread(self) -> bytes:
            await asyncio.Future()
            raise AssertionError("unreachable")

    fixture = RuntimeFixture(tmp_path, cancel_ack_timeout_seconds=0.001)
    cancel_response = HangingCancelResponse(payload={"status": "stopped", "requests": 1})
    fixture.client.cancel_response = cancel_response
    stream = await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long task"}]},
    )

    await asyncio.wait_for(stream.aclose(), timeout=0.2)

    assert cancel_response.closed is True
    assert fixture.process.terminate_calls == 1


@pytest.mark.asyncio
async def test_cancel_ack_timeout_bounds_a_hung_control_response_close(tmp_path: Path) -> None:
    class HangingCancelResponse(FakeResponse):
        async def aread(self) -> bytes:
            await asyncio.Future()
            raise AssertionError("unreachable")

        async def aclose(self) -> None:
            await asyncio.Future()

    fixture = RuntimeFixture(
        tmp_path,
        cancel_ack_timeout_seconds=0.001,
        transport_close_timeout_seconds=0.001,
    )
    fixture.client.cancel_response = HangingCancelResponse(
        payload={"status": "stopped", "requests": 1}
    )
    stream = await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "long task"}]},
    )

    await asyncio.wait_for(stream.aclose(), timeout=0.2)

    assert fixture.process.terminate_calls == 1
    assert fixture.client.closed is True
    assert fixture.runtime._request_lock.locked() is False


@pytest.mark.asyncio
async def test_stream_rewrites_public_model_across_split_sse_chunks(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.stream_response = FakeResponse(
        chunks=(
            b'data: {"id":"chatcmpl-test","model":"def',
            b'ault_model","choices":[]}\n\ndata: [DONE]\n\n',
        )
    )
    stream = await fixture.runtime.open_stream(
        _model(),
        {"model": "local-coder", "messages": [{"role": "user", "content": "hi"}]},
    )

    body = b"".join([chunk async for chunk in stream])

    assert b'"model":"local-coder"' in body
    assert b"default_model" not in body
    assert body.endswith(b"data: [DONE]\n\n")
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_stream_with_tools_yields_normal_text_before_upstream_done(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)

    class GatedResponse(FakeResponse):
        def __init__(self) -> None:
            super().__init__()
            self.release_tail = asyncio.Event()

        def aiter_raw(self) -> AsyncIterator[bytes]:
            async def iterate() -> AsyncIterator[bytes]:
                yield (
                    b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                    b'[{"index":0,"finish_reason":null,"delta":{"role":"assistant",'
                    b'"content":"live text"}}]}\n\n'
                )
                await self.release_tail.wait()
                yield (
                    b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                    b'[{"index":0,"finish_reason":"stop","delta":{}}]}'
                    b"\n\ndata: [DONE]\n\n"
                )

            return iterate()

    response = GatedResponse()
    fixture.client.stream_response = response
    stream = await fixture.runtime.open_stream(
        _model(),
        {
            "model": "local-coder",
            "messages": [{"role": "user", "content": "answer normally"}],
            "tools": [_read_tool()],
        },
    )

    first = await asyncio.wait_for(anext(stream), timeout=0.2)
    assert b'"content":"live text"' in first
    response.release_tail.set()
    tail = b"".join([chunk async for chunk in stream])
    assert tail.endswith(b"data: [DONE]\n\n")
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_nonstream_recovers_qwen_tool_markup_when_upstream_parser_misses_it(
    tmp_path: Path,
) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.post_response = FakeResponse(
        payload={
            "id": "chatcmpl-test",
            "model": "default_model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<function=read>\n<parameter=filePath>\n"
                            "pyproject.toml\n</parameter>\n</function>\n</tool_call>"
                        ),
                    },
                }
            ],
        }
    )

    result = await fixture.runtime.complete(
        _model(),
        {
            "model": "local-coder",
            "messages": [{"role": "user", "content": "read the project"}],
            "tools": [_read_tool()],
        },
    )

    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "read"
    assert call["function"]["arguments"] == '{"filePath":"pyproject.toml"}'
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_stream_recovers_qwen_tool_markup_for_opencode(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.stream_response = FakeResponse(
        chunks=(
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":null,"delta":{"role":"assistant",'
                b'"content":"<fun"}}]}\n\n'
            ),
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":null,"delta":{"role":"assistant",'
                b'"content":"ction=read>\\n<parameter=filePath>\\npyproject.toml'
                b'\\n</parameter>\\n</function>\\n</tool_call>"}}]}'
                b"\n\n"
            ),
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":"stop","delta":{"role":"assistant"}}]}'
                b"\n\ndata: [DONE]\n\n"
            ),
        )
    )
    stream = await fixture.runtime.open_stream(
        _model(),
        {
            "model": "local-coder",
            "messages": [{"role": "user", "content": "read the project"}],
            "tools": [_read_tool()],
        },
    )

    body = b"".join([chunk async for chunk in stream])

    assert b'"finish_reason":"tool_calls"' in body
    assert b'"tool_calls":[{"index":0' in body
    assert b'"name":"read"' in body
    assert b'\\"filePath\\":\\"pyproject.toml\\"' in body
    assert b"<function=read>" not in body
    assert b'"model":"local-coder"' in body
    assert body.endswith(b"data: [DONE]\n\n")
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_stream_recovers_raw_call_alongside_native_tool_call(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.client.stream_response = FakeResponse(
        chunks=(
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":null,"delta":{"role":"assistant",'
                b'"content":"<function=read>\\n<parameter=filePath>\\npyproject.toml'
                b'\\n</parameter>\\n</function>\\n</tool_call>"}}]}\n\n'
            ),
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":null,"delta":{"tool_calls":'
                b'[{"index":0,"id":"call_native","type":"function","function":'
                b'{"name":"read","arguments":"{\\"filePath\\":\\"pyproject.toml\\"}"}},'
                b'{"index":1,"id":"call_other","type":"function","function":'
                b'{"name":"read","arguments":"{\\"filePath\\":\\"README.md\\"}"}}]}}]}'
                b"\n\n"
            ),
            (
                b'data: {"id":"chatcmpl-test","model":"default_model","choices":'
                b'[{"index":0,"finish_reason":"tool_calls","delta":{}}]}'
                b"\n\ndata: [DONE]\n\n"
            ),
        )
    )
    stream = await fixture.runtime.open_stream(
        _model(),
        {
            "model": "local-coder",
            "messages": [{"role": "user", "content": "read both files"}],
            "tools": [_read_tool()],
        },
    )

    body = b"".join([chunk async for chunk in stream])

    assert body.count(b'"name":"read"') == 2
    assert b'"tool_calls":[{"index":0' in body
    assert b'},{"index":1' in body
    assert b"pyproject.toml" in body
    assert b"README.md" in body
    assert b"<function=read>" not in body
    await fixture.runtime.release()


@pytest.mark.asyncio
async def test_prepare_refuses_an_occupied_private_port(tmp_path: Path) -> None:
    fixture = RuntimeFixture(tmp_path)
    fixture.runtime._port_probe = lambda _host, _port: False

    with pytest.raises(DwellError) as caught:
        await fixture.runtime.prepare(_model(), fixture.model_path)

    assert caught.value.code == "server_busy"
    assert caught.value.status_code == 503
    assert fixture.spawn_command is None


@pytest.mark.asyncio
async def test_release_escalates_from_bounded_term_to_kill(tmp_path: Path) -> None:
    process = FakeProcess(stubborn=True)
    fixture = RuntimeFixture(tmp_path, process=process, termination_grace_seconds=0.001)
    await fixture.runtime.prepare(_model(), fixture.model_path)

    await fixture.runtime.release()

    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.asyncio
async def test_installation_probe_is_cached_and_timeout_kills_child(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class HangingProbe:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.kill_calls = 0
            self.wait_calls = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.Future()
            raise AssertionError("unreachable")

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9

        async def wait(self) -> int:
            self.wait_calls += 1
            assert self.returncode is not None
            return self.returncode

    probe = HangingProbe()
    spawn_calls = 0

    async def fake_create_subprocess_exec(*_args: Any, **_kwargs: Any) -> HangingProbe:
        nonlocal spawn_calls
        spawn_calls += 1
        return probe

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    runtime = MLXLMRuntime(
        DwellConfig(home=tmp_path / "dwell", mlx_lm_port=19189),
        model_resolver=lambda _model_id: tmp_path,
        installation_probe_timeout_seconds=0.001,
        probe_kill_wait_seconds=0.1,
    )

    first = await runtime.validate_installation()
    second = await runtime.validate_installation()

    assert first.available is False
    assert "timed out" in first.message
    assert second.message == first.message
    assert spawn_calls == 1
    assert probe.kill_calls == 1
    assert probe.wait_calls == 1


@pytest.mark.asyncio
async def test_client_factory_failure_terminates_spawned_server(tmp_path: Path) -> None:
    process = FakeProcess()

    async def process_factory(*_args: Any, **_kwargs: Any) -> FakeProcess:
        return process

    def failing_client_factory(**_kwargs: Any) -> FakeClient:
        raise RuntimeError("client setup failed")

    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    runtime = MLXLMRuntime(
        DwellConfig(home=tmp_path / "dwell", mlx_lm_port=19189),
        model_resolver=lambda _model_id: model_path,
        client_factory=failing_client_factory,
        process_factory=process_factory,
        installation_probe=lambda: True,
        port_probe=lambda _host, _port: True,
    )

    with pytest.raises(DwellError) as caught:
        await runtime.prepare(_model(), model_path)

    assert caught.value.code == "runtime_not_available"
    assert process.terminate_calls == 1


class _HandlerBase:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.path = "/health"
        self.statuses: list[int] = []
        self.response_headers: list[tuple[str, str]] = []
        self.wfile = io.BytesIO()
        self.rfile = io.BytesIO()
        self.get_calls = 0
        self.post_calls = 0

    def send_response(self, status: int) -> None:
        self.statuses.append(status)

    def send_header(self, name: str, value: str) -> None:
        self.response_headers.append((name, value))

    def end_headers(self) -> None:
        pass

    def do_GET(self) -> None:  # noqa: N802
        self.get_calls += 1

    def do_POST(self) -> None:  # noqa: N802
        self.post_calls += 1


def test_private_server_handler_requires_token_and_reports_model_readiness() -> None:
    ready = threading.Event()
    handler_type = _authenticated_handler(_HandlerBase, "s" * 64, ready)
    handler = handler_type()

    handler.do_GET()
    assert handler.statuses == [401]
    assert handler.get_calls == 0

    handler.headers["Authorization"] = f"Bearer {'s' * 64}"
    handler.path = "/dwell/ready"
    handler.do_GET()
    assert handler.statuses[-1] == 503

    ready.set()
    handler.do_GET()
    assert handler.statuses[-1] == 200

    handler.path = "/health"
    handler.do_GET()
    handler.do_POST()
    assert handler.get_calls == 1
    assert handler.post_calls == 1


def test_private_server_treats_client_disconnect_as_normal_cancellation() -> None:
    class DisconnectingHandler(_HandlerBase):
        def do_POST(self) -> None:  # noqa: N802
            raise BrokenPipeError

    handler_type = _authenticated_handler(DisconnectingHandler, "s" * 64, threading.Event())
    handler = handler_type()
    handler.headers["Authorization"] = f"Bearer {'s' * 64}"

    handler.do_POST()


def test_private_server_stops_context_when_headers_fail_before_iteration() -> None:
    controller = _GenerationController()

    class Context:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    context = Context()

    class ResponseGenerator:
        def generate(self) -> tuple[Context, Any]:
            return context, iter((b"token",))

    response_generator = ResponseGenerator()
    _track_response_generator(response_generator, controller)

    class HeaderFailureHandler(_HandlerBase):
        iterator_started = False

        def do_POST(self) -> None:  # noqa: N802
            _context, response = response_generator.generate()
            self.end_headers()
            self.iterator_started = True
            list(response)

        def end_headers(self) -> None:
            raise BrokenPipeError

    handler_type = _authenticated_handler(
        HeaderFailureHandler,
        "s" * 64,
        threading.Event(),
        controller,
    )
    handler = handler_type()
    handler.headers["Authorization"] = f"Bearer {'s' * 64}"

    handler.do_POST()

    assert handler.iterator_started is False
    assert context.stopped is False
    assert controller.cancel_and_wait(timeout=0) == (1, False)
    assert context.stopped is True

    controller.clear(context)
    assert controller.cancel_and_wait(timeout=0) == (0, True)


def test_private_server_cancel_endpoint_stops_tracked_generation() -> None:
    controller = _GenerationController()

    class Context:
        stopped = False

        def stop(self) -> None:
            self.stopped = True
            controller.clear(self)

    context = Context()
    controller.track(context)
    handler_type = _authenticated_handler(
        _HandlerBase,
        "s" * 64,
        threading.Event(),
        controller,
    )
    handler = handler_type()
    handler.headers["Authorization"] = f"Bearer {'s' * 64}"
    handler.headers["Content-Length"] = "2"
    handler.rfile = io.BytesIO(b"{}")
    handler.path = "/dwell/cancel"

    handler.do_POST()

    assert context.stopped is True
    assert handler.statuses == [200]
    assert b'"status":"stopped"' in handler.wfile.getvalue()


def test_pending_request_is_stopped_if_context_arrives_after_cancel() -> None:
    controller = _GenerationController()
    token = controller.begin_request()
    requested, stopped = controller.cancel_and_wait(timeout=0)

    class Context:
        stopped = False

        def stop(self) -> None:
            self.stopped = True
            controller.clear(self)

    context = Context()
    controller.track(context)
    controller.end_request(token)

    assert requested == 1
    assert stopped is False
    assert context.stopped is True
    assert controller.cancel_and_wait(timeout=0) == (0, True)
