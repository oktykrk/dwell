from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable
from typing import Annotated, Any, Literal, Protocol, TypeVar

import anyio
from fastapi import APIRouter, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import Receive, Scope, Send

from dwell.domain import Modality
from dwell.errors import DwellError

_T = TypeVar("_T")


class TextStream(Protocol):
    """The streaming contract supplied by a text runtime."""

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def aclose(self) -> None: ...


class OpenAIManager(Protocol):
    """The small model-manager surface required by the compatibility router."""

    def list_models(self) -> list[Any]: ...

    def ensure_installed(self, model_id: str) -> Any: ...

    async def complete_text(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def open_text_stream(
        self,
        model_id: str,
        payload: dict[str, Any],
    ) -> TextStream: ...


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextContentPart(_StrictModel):
    type: Literal["text"]
    text: str


class SystemMessage(_StrictModel):
    role: Literal["system"]
    content: str


class UserMessage(_StrictModel):
    role: Literal["user"]
    content: str | list[TextContentPart]


class FunctionCall(_StrictModel):
    name: str = Field(min_length=1)
    arguments: str


class AssistantToolCall(_StrictModel):
    id: str = Field(min_length=1)
    type: Literal["function"]
    function: FunctionCall


class AssistantMessage(_StrictModel):
    role: Literal["assistant"]
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[AssistantToolCall] | None = None


class ToolMessage(_StrictModel):
    role: Literal["tool"]
    content: str
    tool_call_id: str = Field(min_length=1)


ChatMessage = Annotated[
    SystemMessage | UserMessage | AssistantMessage | ToolMessage,
    Field(discriminator="role"),
]


class FunctionDefinition(_StrictModel):
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any]
    strict: bool | None = None


class FunctionTool(_StrictModel):
    type: Literal["function"]
    function: FunctionDefinition


class NamedToolChoiceFunction(_StrictModel):
    name: str = Field(min_length=1)


class NamedToolChoice(_StrictModel):
    type: Literal["function"]
    function: NamedToolChoiceFunction


class StreamOptions(_StrictModel):
    include_usage: bool = False


class ChatCompletionRequest(_StrictModel):
    """OpenCode's OpenAI-compatible Chat Completions request subset."""

    model: str = Field(min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    stop: str | list[str] | None = None
    seed: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    parallel_tool_calls: bool | None = None
    response_format: dict[str, Any] | None = None

    @model_validator(mode="after")
    def supported_features(self) -> ChatCompletionRequest:
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Provide only one of max_tokens or max_completion_tokens.")
        if self.tool_choice not in (None, "auto"):
            raise ValueError("Only tool_choice='auto' is supported by this runtime.")
        if self.parallel_tool_calls is False:
            raise ValueError("parallel_tool_calls=false is not supported by this runtime.")
        if self.response_format is not None:
            raise ValueError("response_format is not supported by this runtime.")
        return self


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for raw_error in exc.errors():
        error = dict(raw_error)
        # Inputs may contain source code or prompts. They do not belong in an
        # error response or the logs of a calling harness.
        error.pop("input", None)
        error.pop("url", None)
        context = error.get("ctx")
        if isinstance(context, dict):
            error["ctx"] = {key: str(value) for key, value in context.items()}
        details.append(jsonable_encoder(error))
    return details


def _error_response(
    *,
    status_code: int,
    message: str,
    code: str,
    param: str | None = None,
    details: Any = None,
) -> JSONResponse:
    error: dict[str, Any] = {
        "message": message,
        "type": "invalid_request_error" if status_code < 500 else "server_error",
        "param": param,
        "code": code,
    }
    if details is not None:
        error["details"] = jsonable_encoder(details)
    return JSONResponse(status_code=status_code, content={"error": error})


class _OpenAIAPIRoute(APIRoute):
    """Keep request and operational errors parseable by OpenAI clients."""

    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def route_handler(request: Request):
            try:
                return await original_handler(request)
            except RequestValidationError as exc:
                return _error_response(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    message="The request is invalid.",
                    code="invalid_request",
                    details=_validation_details(exc),
                )
            except DwellError as exc:
                return _error_response(
                    status_code=exc.status_code,
                    message=exc.message,
                    code=exc.code,
                    details=exc.details,
                )

        return route_handler


def _modality_value(model: Any) -> Any:
    modality = getattr(model, "modality", None)
    return getattr(modality, "value", modality)


def _validate_text_model(model: Any, model_id: str) -> None:
    if _modality_value(model) != Modality.TEXT.value:
        raise DwellError(
            "invalid_request",
            f"Model '{model_id}' does not support text generation.",
            details={"param": "model", "modality": _modality_value(model)},
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        )
    if not bool(getattr(model, "available", False)):
        runtime_id = getattr(model, "runtime", None)
        raise DwellError(
            "runtime_not_available",
            f"Runtime '{runtime_id}' is not available locally.",
            details={"param": "model", "runtime": runtime_id},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )


async def _stream_body(stream: TextStream) -> AsyncIterator[str | bytes]:
    try:
        async for chunk in stream:
            if not isinstance(chunk, (str, bytes)):
                raise TypeError("Text runtime stream chunks must be str or bytes.")
            yield chunk
    finally:
        # Starlette disconnects by cancelling the response task. A shielded
        # close lets the runtime stop upstream generation and release its GPU
        # lease before this generator disappears.
        with anyio.CancelScope(shield=True):
            await stream.aclose()


class _ClosableStreamingResponse(StreamingResponse):
    """Close the upstream even when an ASGI disconnect abandons its iterator."""

    def __init__(self, content: AsyncIterator[str | bytes], stream: TextStream, **kwargs: Any):
        super().__init__(content, **kwargs)
        self._dwell_stream = stream

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            # ASGI 2.3 may cancel StreamingResponse between iterator yield and
            # socket send without finalizing the async generator immediately.
            with anyio.CancelScope(shield=True):
                await self._dwell_stream.aclose()


async def _while_client_connected(request: Request, operation: Awaitable[_T]) -> _T:
    """Cancel startup/generation if the caller leaves before a response exists."""

    async def wait_for_disconnect() -> None:
        # Give fast operations a chance to finish without touching the ASGI
        # receive channel. Some test/embedded transports do not implement a
        # cancellation-safe empty receive, while real model startup is slow.
        await asyncio.sleep(0.05)
        while not await request.is_disconnected():  # noqa: ASYNC110 - ASGI polling API
            await asyncio.sleep(0.05)

    operation_task = asyncio.create_task(operation, name="dwell-openai-operation")
    disconnect_task = asyncio.create_task(
        wait_for_disconnect(),
        name="dwell-openai-disconnect",
    )
    try:
        done, _pending = await asyncio.wait(
            {operation_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except BaseException:
        operation_task.cancel()
        disconnect_task.cancel()
        await asyncio.gather(operation_task, disconnect_task, return_exceptions=True)
        raise
    if operation_task in done:
        disconnect_task.cancel()
        await asyncio.gather(disconnect_task, return_exceptions=True)
        return operation_task.result()

    operation_task.cancel()
    await asyncio.gather(operation_task, return_exceptions=True)
    raise DwellError(
        "client_disconnected",
        "The client disconnected before generation could start.",
        status_code=499,
    )


def create_openai_router(manager: OpenAIManager) -> APIRouter:
    """Create an OpenAI-compatible router backed by a Dwell model manager."""

    router = APIRouter(prefix="/openai/v1", route_class=_OpenAIAPIRoute)

    @router.get("/models")
    async def list_models() -> dict[str, Any]:
        models = [
            {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": "dwell",
            }
            for model in manager.list_models()
            if bool(getattr(model, "installed", False))
            and bool(getattr(model, "available", False))
            and _modality_value(model) == Modality.TEXT.value
        ]
        return {"object": "list", "data": models}

    @router.post("/chat/completions")
    async def create_chat_completion(completion: ChatCompletionRequest, request: Request):
        model = manager.ensure_installed(completion.model)
        _validate_text_model(model, completion.model)
        payload = completion.model_dump(mode="json", exclude_none=True)

        if not completion.stream:
            result = await _while_client_connected(
                request,
                manager.complete_text(completion.model, payload),
            )
            return JSONResponse(content=jsonable_encoder(result))

        # Open the runtime stream before response headers are committed so
        # startup/model-load failures can still use a truthful HTTP status.
        stream = await _while_client_connected(
            request,
            manager.open_text_stream(completion.model, payload),
        )
        return _ClosableStreamingResponse(
            _stream_body(stream),
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router
