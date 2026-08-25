from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dwell.api.openai import (
    _ClosableStreamingResponse,
    _stream_body,
    _while_client_connected,
    create_openai_router,
)
from dwell.domain import Modality
from dwell.errors import DwellError, model_not_installed

MODEL_ID = "qwen3-coder-30b-a3b-4bit"


def model_view(
    model_id: str = MODEL_ID,
    *,
    modality: Modality = Modality.TEXT,
    installed: bool = True,
    available: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=model_id,
        modality=modality,
        runtime="mlx-lm",
        installed=installed,
        available=available,
    )


class FakeTextStream:
    def __init__(self, chunks: list[bytes | str]) -> None:
        self.chunks = chunks
        self.closed = False

    async def _iterate(self) -> AsyncIterator[bytes | str]:
        for chunk in self.chunks:
            yield chunk

    def __aiter__(self) -> AsyncIterator[bytes | str]:
        return self._iterate()

    async def aclose(self) -> None:
        self.closed = True


class FakeManager:
    def __init__(self, models: list[SimpleNamespace] | None = None) -> None:
        self.models = models or [model_view()]
        self.complete_calls: list[tuple[str, dict[str, Any]]] = []
        self.stream_calls: list[tuple[str, dict[str, Any]]] = []
        self.stream = FakeTextStream(
            [
                (
                    'data: {"id":"chatcmpl-local","object":"chat.completion.chunk",'
                    '"model":"qwen3-coder-30b-a3b-4bit","choices":[{"index":0,'
                    '"delta":{"role":"assistant","content":"hello"},'
                    '"finish_reason":null}]}\n\n'
                ),
                (
                    'data: {"id":"chatcmpl-local","object":"chat.completion.chunk",'
                    '"model":"qwen3-coder-30b-a3b-4bit","choices":[{"index":0,'
                    '"delta":{},"finish_reason":"stop"}]}\n\n'
                ),
                "data: [DONE]\n\n",
            ]
        )
        self.complete_error: DwellError | None = None
        self.stream_error: DwellError | None = None

    def list_models(self) -> list[SimpleNamespace]:
        return self.models

    def ensure_installed(self, model_id: str) -> SimpleNamespace:
        model = next((item for item in self.models if item.id == model_id), None)
        if model is None:
            raise DwellError(
                "model_not_found",
                f"Model '{model_id}' is not registered.",
                status_code=404,
            )
        if not model.installed:
            raise model_not_installed(model_id)
        return model

    async def complete_text(
        self,
        model_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self.complete_error is not None:
            raise self.complete_error
        self.complete_calls.append((model_id, payload))
        return {
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def open_text_stream(
        self,
        model_id: str,
        payload: dict[str, Any],
    ) -> FakeTextStream:
        if self.stream_error is not None:
            raise self.stream_error
        self.stream_calls.append((model_id, payload))
        return self.stream


def client_for(manager: FakeManager) -> TestClient:
    app = FastAPI()
    app.include_router(create_openai_router(manager))
    return TestClient(app)


def chat_request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    payload.update(overrides)
    return payload


def test_models_lists_only_installed_available_text_models() -> None:
    manager = FakeManager(
        [
            model_view(),
            model_view("missing-text", installed=False, available=False),
            model_view("unavailable-text", available=False),
            model_view("available-video", modality=Modality.VIDEO),
        ]
    )

    with client_for(manager) as client:
        response = client.get("/openai/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "dwell",
            }
        ],
    }


def test_nonstream_completion_and_openai_tools_are_passed_to_manager() -> None:
    manager = FakeManager()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a local file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Read pyproject.toml"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"pyproject.toml"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "[project]"},
    ]

    with client_for(manager) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(
                messages=messages,
                max_tokens=512,
                temperature=0,
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
            ),
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hello"
    assert manager.complete_calls[0][0] == MODEL_ID
    forwarded = manager.complete_calls[0][1]
    assert forwarded["model"] == MODEL_ID
    assert forwarded["messages"][0] == messages[0]
    assert forwarded["messages"][1]["tool_calls"] == messages[1]["tool_calls"]
    assert forwarded["messages"][2] == messages[2]
    assert forwarded["tools"] == tools
    assert forwarded["tool_choice"] == "auto"
    assert forwarded["parallel_tool_calls"] is True
    assert forwarded["stream"] is False


def test_stream_is_raw_sse_and_always_closed() -> None:
    manager = FakeManager()

    with client_for(manager) as client:
        with client.stream(
            "POST",
            "/openai/v1/chat/completions",
            json=chat_request(stream=True, stream_options={"include_usage": True}),
        ) as response:
            body = b"".join(response.iter_bytes()).decode()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "data: [DONE]\n\n" in body
    assert '"content":"hello"' in body
    assert manager.stream_calls[0][0] == MODEL_ID
    assert manager.stream_calls[0][1]["model"] == MODEL_ID
    assert manager.stream_calls[0][1]["stream"] is True
    assert manager.stream.closed is True


def test_validation_errors_use_openai_envelope_without_echoing_prompts() -> None:
    secret = "private source code must not be echoed"
    manager = FakeManager()

    with client_for(manager) as client:
        empty_messages = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(messages=[]),
        )
        too_many_tokens = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(
                messages=[{"role": "user", "content": secret}],
                max_tokens=8193,
            ),
        )
        unsupported_format = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(response_format={"type": "json_object"}),
        )
        unsupported_tool_choice = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(tool_choice="required"),
        )
        unsupported_serial_tools = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(parallel_tool_calls=False),
        )

    for response in (
        empty_messages,
        too_many_tokens,
        unsupported_format,
        unsupported_tool_choice,
        unsupported_serial_tools,
    ):
        assert response.status_code == 422
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert response.json()["error"]["code"] == "invalid_request"
        assert secret not in response.text
    assert manager.complete_calls == []
    assert manager.stream_calls == []


def test_unavailable_or_non_text_model_fails_before_runtime() -> None:
    unavailable = FakeManager([model_view(available=False)])
    video = FakeManager([model_view(modality=Modality.VIDEO)])

    with client_for(unavailable) as client:
        unavailable_response = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(),
        )
    with client_for(video) as client:
        video_response = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(),
        )

    assert unavailable_response.status_code == 503
    assert unavailable_response.json()["error"]["code"] == "runtime_not_available"
    assert video_response.status_code == 422
    assert video_response.json()["error"]["code"] == "invalid_request"
    assert unavailable.complete_calls == []
    assert video.complete_calls == []


def test_manager_errors_before_response_use_openai_envelope() -> None:
    manager = FakeManager()
    manager.complete_error = DwellError(
        "server_busy",
        "The text runtime is busy.",
        status_code=429,
    )

    with client_for(manager) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(),
        )

    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "message": "The text runtime is busy.",
            "type": "invalid_request_error",
            "param": None,
            "code": "server_busy",
        }
    }


def test_stream_open_failure_does_not_commit_sse_headers() -> None:
    manager = FakeManager()
    manager.stream_error = DwellError(
        "runtime_start_failed",
        "The text runtime could not start.",
        status_code=503,
    )

    with client_for(manager) as client:
        response = client.post(
            "/openai/v1/chat/completions",
            json=chat_request(stream=True),
        )

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "runtime_start_failed"
    assert manager.stream.closed is False


@pytest.mark.asyncio
async def test_asgi_disconnect_closes_upstream_before_response_returns() -> None:
    class HangingStream(FakeTextStream):
        def __init__(self) -> None:
            super().__init__([])
            self.keep_open = asyncio.Event()

        async def _iterate(self) -> AsyncIterator[bytes | str]:
            yield b"data: first\n\n"
            await self.keep_open.wait()

    stream = HangingStream()
    response = _ClosableStreamingResponse(
        _stream_body(stream),
        stream,
        media_type="text/event-stream",
    )
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, str]:
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await response(
        {"type": "http", "asgi": {"spec_version": "2.3"}},  # type: ignore[arg-type]
        receive,  # type: ignore[arg-type]
        send,  # type: ignore[arg-type]
    )

    assert sent
    assert stream.closed is True


@pytest.mark.asyncio
async def test_disconnect_while_operation_is_starting_cancels_it() -> None:
    operation_started = asyncio.Event()
    operation_cancelled = asyncio.Event()

    class DisconnectedRequest:
        async def is_disconnected(self) -> bool:
            await operation_started.wait()
            return True

    async def slow_operation() -> str:
        operation_started.set()
        try:
            await asyncio.Future()
        finally:
            operation_cancelled.set()
        raise AssertionError("unreachable")

    with pytest.raises(DwellError) as caught:
        await _while_client_connected(  # type: ignore[arg-type]
            DisconnectedRequest(),
            slow_operation(),
        )

    assert caught.value.code == "client_disconnected"
    assert caught.value.status_code == 499
    assert operation_cancelled.is_set()
