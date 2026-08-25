from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import httpx

from dwell.config import DwellConfig
from dwell.domain import ModelDefinition, RuntimeCapabilities
from dwell.errors import DwellError
from dwell.runtimes.base import ModelRuntime, RuntimeStatus, TextEngine, TextStream

_PINNED_MLX_LM_VERSION = "0.31.3"
_SERVER_TOKEN_ENV = "DWELL_MLX_SERVER_TOKEN"
_PARENT_PID_ENV = "DWELL_MLX_PARENT_PID"
_PINNED_VERSION_ENV = "DWELL_MLX_LM_VERSION"
_MAX_OUTPUT_TOKENS = 8192
_TOOL_CALL_PATTERN = re.compile(
    r"(?:<tool_call>\s*)?<function=([^>\s]+)>(.*?)</function>\s*(?:</tool_call>)?",
    re.DOTALL,
)
_TOOL_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_RAW_TOOL_MARKERS = ("<tool_call>", "<function=")
_FORBIDDEN_REQUEST_FIELDS = frozenset(
    {
        "adapter",
        "adapter_path",
        "adapters",
        "draft_model",
        "num_draft_tokens",
        "trust_remote_code",
    }
)


class _Process(Protocol):
    pid: int

    @property
    def returncode(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


class _Response(Protocol):
    status_code: int

    def json(self) -> Any: ...

    async def aread(self) -> bytes: ...

    async def aclose(self) -> None: ...

    def aiter_raw(self) -> AsyncIterator[bytes]: ...


class _Client(Protocol):
    async def get(self, url: str) -> _Response: ...

    async def post(self, url: str, *, json: Mapping[str, Any]) -> _Response: ...

    def build_request(self, method: str, url: str, **kwargs: Any) -> Any: ...

    async def send(self, request: Any, *, stream: bool = False) -> _Response: ...

    async def aclose(self) -> None: ...


ProcessFactory = Callable[..., _Process | Awaitable[_Process]]
ClientFactory = Callable[..., _Client]
InstallationProbe = Callable[[], bool | str | Awaitable[bool | str]]
PortProbe = Callable[[str, int], bool]


async def _resolve_awaitable(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _finish_cleanup(awaitable: Awaitable[None], *, name: str) -> None:
    """Finish safety-critical cleanup before propagating caller cancellation."""

    task = asyncio.create_task(awaitable, name=name)
    interrupted = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            interrupted = True
            if task.done():
                break
    task.result()
    if interrupted:
        raise asyncio.CancelledError


def _loopback_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _rewrite_sse_event(event: bytes, model_id: str) -> bytes:
    lines: list[bytes] = []
    for line in event.splitlines():
        if line.startswith(b"data: ") and line[6:] != b"[DONE]":
            try:
                payload = json.loads(line[6:])
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict) and "model" in payload:
                payload["model"] = model_id
                line = b"data: " + json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
        lines.append(line)
    return b"\n".join(lines) + b"\n\n"


def _convert_tool_argument(value: str, schema: Mapping[str, Any]) -> Any:
    value = value.strip("\n")
    if value.lower() == "null":
        return None
    value_type = schema.get("type")
    if value_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if value_type == "number":
        try:
            return float(value)
        except ValueError:
            return value
    if value_type == "boolean":
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        return value
    if value_type in {"array", "object"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _fallback_tool_calls(
    text: str,
    tools: object,
) -> tuple[str | None, list[dict[str, Any]]] | None:
    if not isinstance(tools, list) or not tools:
        return None
    definitions = {
        function["name"]: function
        for tool in tools
        if isinstance(tool, Mapping)
        and isinstance((function := tool.get("function")), Mapping)
        and isinstance(function.get("name"), str)
    }
    matches = list(_TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return None
    prefix = text[: matches[0].start()].strip() or None
    suffix = text[matches[-1].end() :].strip()
    if suffix and suffix != "</tool_call>":
        return None

    calls: list[dict[str, Any]] = []
    for match in matches:
        name = match.group(1)
        definition = definitions.get(name)
        if definition is None:
            return None
        parameters = definition.get("parameters")
        properties = parameters.get("properties", {}) if isinstance(parameters, Mapping) else {}
        arguments: dict[str, Any] = {}
        for parameter in _TOOL_PARAMETER_PATTERN.finditer(match.group(2)):
            parameter_name = parameter.group(1)
            schema = properties.get(parameter_name, {}) if isinstance(properties, Mapping) else {}
            arguments[parameter_name] = _convert_tool_argument(
                parameter.group(2),
                schema if isinstance(schema, Mapping) else {},
            )
        calls.append(
            {
                "id": f"call_{uuid4().hex}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
    return prefix, calls


def _rewrite_nonstream_tool_markup(
    response: dict[str, Any],
    tools: object,
) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return response
    message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("tool_calls"):
        return response
    content = message.get("content")
    if not isinstance(content, str):
        return response
    parsed = _fallback_tool_calls(content, tools)
    if parsed is None:
        return response
    prefix, calls = parsed
    message["content"] = prefix
    message["tool_calls"] = calls
    choices[0]["finish_reason"] = "tool_calls"
    return response


def _decode_sse_json(event: bytes) -> dict[str, Any] | None:
    for line in event.splitlines():
        if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
            continue
        try:
            payload = json.loads(line[6:])
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _encode_sse_json(payload: Mapping[str, Any]) -> bytes:
    return (
        b"data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n\n"
    )


def _stream_content(payload: Mapping[str, Any]) -> str | None:
    parts: list[str] = []
    found = False
    for choice in payload.get("choices", []):
        if not isinstance(choice, Mapping):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, Mapping) or not isinstance(delta.get("content"), str):
            continue
        found = True
        parts.append(delta["content"])
    return "".join(parts) if found else None


def _content_sse_event(template: Mapping[str, Any], content: str) -> bytes:
    payload = deepcopy(template)
    choices = payload.get("choices")
    choice_template = choices[0] if isinstance(choices, list) and choices else {}
    delta_template = choice_template.get("delta") if isinstance(choice_template, Mapping) else {}
    delta: dict[str, Any] = {"content": content}
    if isinstance(delta_template, Mapping) and isinstance(delta_template.get("role"), str):
        delta["role"] = delta_template["role"]
    payload["choices"] = [
        {
            "index": choice_template.get("index", 0) if isinstance(choice_template, Mapping) else 0,
            "finish_reason": None,
            "delta": delta,
        }
    ]
    payload.pop("usage", None)
    return _encode_sse_json(payload)


def _without_stream_content(payload: Mapping[str, Any]) -> bytes | None:
    rewritten = deepcopy(payload)
    choices = rewritten.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    retained: list[dict[str, Any]] = []
    for raw_choice in choices:
        if not isinstance(raw_choice, Mapping):
            continue
        choice = dict(raw_choice)
        raw_delta = choice.get("delta")
        delta = dict(raw_delta) if isinstance(raw_delta, Mapping) else {}
        delta.pop("content", None)
        choice["delta"] = delta
        if delta or choice.get("finish_reason") is not None:
            retained.append(choice)
    if not retained:
        return None
    rewritten["choices"] = retained
    return _encode_sse_json(rewritten)


def _stream_event_is_terminal(payload: Mapping[str, Any]) -> bool:
    choices = payload.get("choices")
    if choices == []:
        return True
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, Mapping)
        and (
            choice.get("finish_reason") is not None
            or (
                isinstance((delta := choice.get("delta")), Mapping)
                and bool(delta.get("tool_calls"))
            )
        )
        for choice in choices
    )


def _raw_tool_marker_index(text: str) -> int | None:
    positions = [position for marker in _RAW_TOOL_MARKERS if (position := text.find(marker)) >= 0]
    return min(positions) if positions else None


def _raw_tool_marker_suffix_length(text: str) -> int:
    maximum = min(len(text), max(map(len, _RAW_TOOL_MARKERS)) - 1)
    for length in range(maximum, 0, -1):
        suffix = text[-length:]
        if any(marker.startswith(suffix) for marker in _RAW_TOOL_MARKERS):
            return length
    return 0


def _collect_stream_tool_calls(payloads: list[dict[str, Any] | None]) -> list[dict[str, Any]]:
    collected: dict[int, dict[str, Any]] = {}
    for payload in payloads:
        if payload is None:
            continue
        for choice in payload.get("choices", []):
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, Mapping):
                continue
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for fallback_index, fragment in enumerate(tool_calls):
                if not isinstance(fragment, Mapping):
                    continue
                raw_index = fragment.get("index", fallback_index)
                index = raw_index if isinstance(raw_index, int) else fallback_index
                call = collected.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                call_id = fragment.get("id")
                if isinstance(call_id, str):
                    call["id"] += call_id
                call_type = fragment.get("type")
                if isinstance(call_type, str):
                    call["type"] = call_type
                function = fragment.get("function")
                if isinstance(function, Mapping):
                    for field in ("name", "arguments"):
                        value = function.get(field)
                        if isinstance(value, str):
                            call["function"][field] += value
    return [collected[index] for index in sorted(collected)]


def _tool_call_signature(call: Mapping[str, Any]) -> tuple[Any, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None, None
    return function.get("name"), function.get("arguments")


def _rewrite_buffered_tool_events(
    events: list[bytes],
    tools: object,
) -> list[bytes]:
    payloads = [_decode_sse_json(event) for event in events]
    content = "".join(
        delta.get("content", "")
        for payload in payloads
        if payload is not None
        for choice in payload.get("choices", [])
        if isinstance(choice, Mapping)
        and isinstance((delta := choice.get("delta")), Mapping)
        and isinstance(delta.get("content"), str)
    )
    parsed = _fallback_tool_calls(content, tools)
    if parsed is None:
        return events
    prefix, fallback_calls = parsed
    fallback_signatures = {_tool_call_signature(call) for call in fallback_calls}
    native_calls = [
        call
        for call in _collect_stream_tool_calls(payloads)
        if _tool_call_signature(call) not in fallback_signatures
    ]
    calls = fallback_calls + native_calls

    template = next(
        (
            payload
            for payload in payloads
            if payload is not None
            and isinstance(payload.get("choices"), list)
            and payload["choices"]
        ),
        None,
    )
    if template is None:
        return events
    rewritten: list[bytes] = []
    if prefix:
        prefix_payload = deepcopy(template)
        prefix_payload["choices"] = [
            {
                "index": 0,
                "finish_reason": None,
                "delta": {"role": "assistant", "content": prefix},
            }
        ]
        rewritten.append(_encode_sse_json(prefix_payload))
    tool_payload = deepcopy(template)
    streaming_calls = [{"index": index, **call} for index, call in enumerate(calls)]
    tool_payload["choices"] = [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "delta": {"role": "assistant", "tool_calls": streaming_calls},
        }
    ]
    rewritten.append(_encode_sse_json(tool_payload))
    rewritten.extend(
        event
        for event, payload in zip(events, payloads, strict=True)
        if payload is not None and payload.get("choices") == []
    )
    return rewritten


class _HTTPTextStream(TextStream):
    """Own one upstream SSE response and its runtime concurrency lease."""

    def __init__(
        self,
        response: _Response,
        release: Callable[[], Awaitable[None]],
        cancel: Callable[[], Awaitable[None]],
        model_id: str,
        tools: object = None,
        close_timeout_seconds: float = 2.0,
    ) -> None:
        self._response = response
        self._release = release
        self._cancel = cancel
        self._source = response.aiter_raw().__aiter__()
        self._iterator = self._rewritten_events(model_id, tools).__aiter__()
        self._close_task: asyncio.Task[None] | None = None
        self._close_timeout_seconds = close_timeout_seconds
        self._finished = False

    async def _rewritten_events(self, model_id: str, tools: object) -> AsyncIterator[bytes]:
        buffer = b""
        tool_events: list[bytes] = []
        pending_text = ""
        pending_template: dict[str, Any] | None = None
        buffering_raw_tool = False

        def flush_pending() -> list[bytes]:
            nonlocal pending_text, pending_template
            if not pending_text or pending_template is None:
                return []
            result = [_content_sse_event(pending_template, pending_text)]
            pending_text = ""
            pending_template = None
            return result

        def handle_event(raw_event: bytes) -> list[bytes]:
            nonlocal buffering_raw_tool, pending_text, pending_template, tool_events
            rewritten = _rewrite_sse_event(raw_event, model_id)
            if not tools or raw_event.startswith(b":"):
                if raw_event.startswith(b"data: [DONE]"):
                    self._finished = True
                return [rewritten]
            if raw_event.startswith(b"data: [DONE]"):
                self._finished = True
                if buffering_raw_tool:
                    result = _rewrite_buffered_tool_events(tool_events, tools)
                    tool_events = []
                    buffering_raw_tool = False
                else:
                    result = flush_pending()
                result.append(rewritten)
                return result

            payload = _decode_sse_json(rewritten)
            if payload is None:
                return [*flush_pending(), rewritten]
            if buffering_raw_tool:
                tool_events.append(rewritten)
                return []

            content = _stream_content(payload)
            terminal = _stream_event_is_terminal(payload)
            if content is None:
                prefix = flush_pending() if terminal else []
                return [*prefix, rewritten]

            combined = pending_text + content
            marker_index = _raw_tool_marker_index(combined)
            if marker_index is not None:
                result: list[bytes] = []
                safe_prefix = combined[:marker_index]
                if safe_prefix:
                    result.append(_content_sse_event(payload, safe_prefix))
                tool_events = [_content_sse_event(payload, combined[marker_index:])]
                remainder = _without_stream_content(payload)
                if remainder is not None:
                    tool_events.append(remainder)
                pending_text = ""
                pending_template = None
                buffering_raw_tool = True
                return result

            if terminal:
                result = [_content_sse_event(payload, combined)] if combined else []
                remainder = _without_stream_content(payload)
                if remainder is not None:
                    result.append(remainder)
                pending_text = ""
                pending_template = None
                return result

            suffix_length = _raw_tool_marker_suffix_length(combined)
            safe_content = combined[:-suffix_length] if suffix_length else combined
            pending_text = combined[-suffix_length:] if suffix_length else ""
            pending_template = payload if pending_text else None
            return [_content_sse_event(payload, safe_content)] if safe_content else []

        async for chunk in self._source:
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for rewritten in handle_event(event):
                    yield rewritten
                if self._finished:
                    return
        if buffer:
            for rewritten in handle_event(buffer):
                yield rewritten
            if self._finished:
                return
        if buffering_raw_tool and tool_events:
            for buffered in _rewrite_buffered_tool_events(tool_events, tools):
                yield buffered
        else:
            for pending in flush_pending():
                yield pending

    def __aiter__(self) -> _HTTPTextStream:
        return self

    async def __anext__(self) -> bytes:
        if self._close_task is not None:
            raise StopAsyncIteration
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise
        except BaseException:
            await self.aclose()
            raise

    async def __aenter__(self) -> _HTTPTextStream:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._cleanup(),
                name="dwell-mlx-stream-close",
            )
        await asyncio.shield(self._close_task)

    async def _cleanup(self) -> None:
        try:
            if not self._finished:
                try:
                    await self._cancel()
                except Exception:
                    # Runtime teardown is the final safety net when the private
                    # cancellation control request cannot be acknowledged.
                    pass
            try:
                await asyncio.wait_for(
                    self._response.aclose(),
                    timeout=self._close_timeout_seconds,
                )
            except Exception:
                # Socket close is best-effort after an acknowledged cancel, and
                # lifecycle release will terminate the child after an unacked one.
                pass
        finally:
            await self._release()


class MLXLMRuntime(ModelRuntime, TextEngine):
    """Supervise one offline MLX-LM server and expose its chat endpoint safely."""

    runtime_id = "mlx-lm"
    capabilities = RuntimeCapabilities(
        persistent_loading=True,
        progress_reporting=False,
        cancellation=True,
        streaming=True,
        structured_output=False,
        tool_calling=True,
    )

    def __init__(
        self,
        config: DwellConfig,
        model_resolver: Callable[[str], Path],
        *,
        python_binary: str | Path | None = None,
        client_factory: ClientFactory | None = None,
        process_factory: ProcessFactory | None = None,
        installation_probe: InstallationProbe | None = None,
        port_probe: PortProbe | None = None,
        installation_probe_timeout_seconds: float = 15.0,
        probe_kill_wait_seconds: float = 2.0,
        cancel_ack_timeout_seconds: float = 3.0,
        transport_close_timeout_seconds: float = 2.0,
        readiness_timeout_seconds: float = 180.0,
        readiness_poll_seconds: float = 0.1,
        termination_grace_seconds: float = 5.0,
        kill_wait_seconds: float = 2.0,
    ) -> None:
        if (
            installation_probe_timeout_seconds <= 0
            or probe_kill_wait_seconds <= 0
            or cancel_ack_timeout_seconds <= 0
            or transport_close_timeout_seconds <= 0
            or readiness_timeout_seconds <= 0
            or readiness_poll_seconds <= 0
        ):
            raise ValueError("readiness timeouts must be positive")
        if termination_grace_seconds <= 0 or kill_wait_seconds <= 0:
            raise ValueError("process termination timeouts must be positive")

        self.config = config
        self.model_resolver = model_resolver
        self.python_binary = str(python_binary or sys.executable)
        self._client_factory = client_factory or cast(ClientFactory, httpx.AsyncClient)
        self._process_factory = process_factory or cast(
            ProcessFactory, asyncio.create_subprocess_exec
        )
        self._installation_probe = installation_probe
        self._port_probe = port_probe or _loopback_port_available
        self.installation_probe_timeout_seconds = installation_probe_timeout_seconds
        self.probe_kill_wait_seconds = probe_kill_wait_seconds
        self.cancel_ack_timeout_seconds = cancel_ack_timeout_seconds
        self.transport_close_timeout_seconds = transport_close_timeout_seconds
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.readiness_poll_seconds = readiness_poll_seconds
        self.termination_grace_seconds = termination_grace_seconds
        self.kill_wait_seconds = kill_wait_seconds

        self._lifecycle_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._process: _Process | None = None
        self._client: _Client | None = None
        self._active_stream: _HTTPTextStream | None = None
        self._loaded_model: str | None = None
        self._loaded_path: Path | None = None
        self._server_token: str | None = None
        self._log_stream: Any = None
        self._installation_checked = False
        self._installation_problem_cache: str | None = None
        self._release_in_progress = False

    @property
    def server_url(self) -> str:
        return f"http://127.0.0.1:{self.config.mlx_lm_port}"

    async def validate_installation(self) -> RuntimeStatus:
        problem = await self._installation_problem()
        return RuntimeStatus(
            runtime_id=self.runtime_id,
            available=problem is None,
            loaded_models=self._loaded_models(),
            active_jobs=self._active_jobs(),
            message=problem or "pinned local mlx-lm runtime is available",
        )

    async def prepare(
        self,
        model: ModelDefinition,
        model_path: Path,
    ) -> RuntimeStatus:
        self._validate_model(model)
        exact_path = self._validate_model_path(model, model_path)

        async with self._lifecycle_lock:
            if self._process_is_live() and (
                self._loaded_model == model.id and self._loaded_path == exact_path
            ):
                return await self._status_locked()

            problem = await self._installation_problem()
            if problem is not None:
                raise DwellError(
                    "runtime_not_available",
                    "The pinned local mlx-lm runtime is unavailable.",
                    details={"reason": problem},
                    status_code=503,
                )

            await self._release_locked()
            await self._start_locked(model, exact_path)
            return await self._status_locked()

    async def release(self, model: ModelDefinition | None = None) -> RuntimeStatus:
        async with self._lifecycle_lock:
            if model is not None and self._loaded_model not in {None, model.id}:
                return await self._status_locked()
            await self._release_locked()
            return await self._status_locked()

    async def status(self) -> RuntimeStatus:
        async with self._lifecycle_lock:
            if self._process is not None and not self._process_is_live():
                await self._release_locked(terminate=False)
            return await self._status_locked()

    async def complete(self, model: ModelDefinition, payload: Mapping[str, Any]) -> dict[str, Any]:
        await self._ensure_prepared(model)
        request_payload = self._safe_payload(payload, stream=False)
        await self._acquire_request()
        release_request = True
        try:
            client = self._require_client()
            try:
                response = await client.post("/v1/chat/completions", json=request_payload)
            except asyncio.CancelledError:
                try:
                    await _finish_cleanup(
                        self._cancel_active_generation(),
                        name="dwell-mlx-completion-cancel",
                    )
                except Exception:
                    release_request = False
                raise
            except Exception as exc:
                try:
                    await self._cancel_active_generation()
                except Exception:
                    release_request = False
                    raise
                raise self._upstream_unavailable(exc) from exc
            await self._raise_for_upstream_error(response)
            try:
                result = response.json()
            except (TypeError, ValueError) as exc:
                raise DwellError(
                    "generation_failed",
                    "The mlx-lm runtime returned invalid JSON.",
                    status_code=502,
                ) from exc
            if not isinstance(result, dict):
                raise DwellError(
                    "generation_failed",
                    "The mlx-lm runtime returned an invalid completion response.",
                    status_code=502,
                )
            result = dict(result)
            result = _rewrite_nonstream_tool_markup(result, payload.get("tools"))
            result["model"] = model.id
            return result
        finally:
            if release_request:
                self._release_request()

    async def open_stream(
        self,
        model: ModelDefinition,
        payload: Mapping[str, Any],
    ) -> TextStream:
        await self._ensure_prepared(model)
        request_payload = self._safe_payload(payload, stream=True)
        await self._acquire_request()
        release_request = True
        try:
            client = self._require_client()
            request = client.build_request(
                "POST",
                "/v1/chat/completions",
                json=request_payload,
            )
            try:
                response = await client.send(request, stream=True)
            except asyncio.CancelledError:
                try:
                    await _finish_cleanup(
                        self._cancel_active_generation(),
                        name="dwell-mlx-stream-start-cancel",
                    )
                except Exception:
                    release_request = False
                raise
            except Exception as exc:
                try:
                    await self._cancel_active_generation()
                except Exception:
                    release_request = False
                    raise
                raise self._upstream_unavailable(exc) from exc
            await self._raise_for_upstream_error(response)
            stream = _HTTPTextStream(
                response,
                self._stream_released,
                self._cancel_active_generation,
                model.id,
                payload.get("tools"),
                self.transport_close_timeout_seconds,
            )
            self._active_stream = stream
            return stream
        except BaseException:
            if release_request:
                self._release_request()
            raise

    async def _ensure_prepared(self, model: ModelDefinition) -> None:
        self._validate_model(model)
        resolved = self.model_resolver(model.id)
        await self.prepare(model, resolved)

    def _safe_payload(self, payload: Mapping[str, Any], *, stream: bool) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise DwellError(
                "invalid_request",
                "Chat completion payload must be a JSON object.",
                status_code=422,
            )
        forbidden = sorted(_FORBIDDEN_REQUEST_FIELDS.intersection(payload))
        if forbidden:
            raise DwellError(
                "invalid_request",
                "Runtime-controlled model loading options are not allowed.",
                details={"fields": forbidden},
                status_code=422,
            )
        if "messages" not in payload or not isinstance(payload["messages"], list):
            raise DwellError(
                "invalid_request",
                "Chat completion payload must contain a messages array.",
                status_code=422,
            )
        for field in ("max_tokens", "max_completion_tokens"):
            value = payload.get(field)
            if value is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > _MAX_OUTPUT_TOKENS
            ):
                raise DwellError(
                    "invalid_request",
                    f"'{field}' must be an integer between 0 and {_MAX_OUTPUT_TOKENS}.",
                    status_code=422,
                )

        result = dict(payload)
        result["model"] = "default_model"
        result["stream"] = stream
        return result

    async def _start_locked(self, model: ModelDefinition, model_path: Path) -> None:
        self.config.ensure_layout()
        if not self._port_probe("127.0.0.1", self.config.mlx_lm_port):
            raise DwellError(
                "server_busy",
                f"The private mlx-lm port {self.config.mlx_lm_port} is already in use.",
                details={"host": "127.0.0.1", "port": self.config.mlx_lm_port},
                status_code=503,
            )
        token = secrets.token_hex(32)
        environment = self.config.subprocess_env()
        environment.update(
            {
                _SERVER_TOKEN_ENV: token,
                _PARENT_PID_ENV: str(os.getpid()),
                _PINNED_VERSION_ENV: _PINNED_MLX_LM_VERSION,
                "PYTHONUNBUFFERED": "1",
            }
        )
        command = self._server_command(model_path)
        self._log_stream = self.config.log_file.open("a", encoding="utf-8")
        try:
            process = await _resolve_awaitable(
                self._process_factory(
                    *command,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        except Exception as exc:
            self._close_log_stream()
            raise DwellError(
                "runtime_not_available",
                "The local mlx-lm server could not be started.",
                details={"reason": str(exc)},
                status_code=503,
            ) from exc

        self._process = cast(_Process, process)
        self._loaded_model = model.id
        self._loaded_path = model_path
        self._server_token = token
        try:
            self._client = self._new_client(token)
            await self._wait_until_ready()
        except BaseException as exc:
            await self._release_locked()
            if isinstance(exc, (DwellError, asyncio.CancelledError)):
                raise
            raise DwellError(
                "runtime_not_available",
                "The local mlx-lm client could not be initialized.",
                details={"reason": str(exc)},
                status_code=503,
            ) from exc

    def _server_command(self, model_path: Path) -> list[str]:
        return [
            self.python_binary,
            "-m",
            "dwell.runtimes.mlx_server",
            "--model",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config.mlx_lm_port),
            "--max-tokens",
            str(_MAX_OUTPUT_TOKENS),
            "--decode-concurrency",
            "1",
            "--prompt-concurrency",
            "1",
            "--prompt-cache-size",
            "1",
        ]

    def _new_client(self, token: str) -> _Client:
        return self._client_factory(
            base_url=self.server_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=httpx.Timeout(None, connect=5.0),
        )

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.readiness_timeout_seconds
        last_problem = "server did not answer"
        while time.monotonic() < deadline:
            if not self._process_is_live():
                code = self._process.returncode if self._process is not None else None
                raise DwellError(
                    "runtime_not_available",
                    "The local mlx-lm server exited during model loading.",
                    details={"exit_code": code},
                    status_code=503,
                )
            try:
                response = await self._require_client().get("/dwell/ready")
                if response.status_code == 200:
                    return
                last_problem = f"readiness returned HTTP {response.status_code}"
            except Exception as exc:
                last_problem = str(exc)
            await asyncio.sleep(self.readiness_poll_seconds)
        raise DwellError(
            "runtime_not_available",
            "The local mlx-lm model did not become ready in time.",
            details={"reason": last_problem},
            status_code=503,
        )

    async def _release_locked(
        self,
        *,
        terminate: bool = True,
        close_active_stream: bool = True,
    ) -> None:
        self._release_in_progress = True
        try:
            await _finish_cleanup(
                self._release_resources(
                    terminate=terminate,
                    close_active_stream=close_active_stream,
                ),
                name="dwell-mlx-runtime-release",
            )
        finally:
            self._release_in_progress = False

    async def _release_resources(
        self,
        *,
        terminate: bool,
        close_active_stream: bool,
    ) -> None:
        stream = self._active_stream
        self._active_stream = None
        if stream is not None and close_active_stream:
            try:
                await stream.aclose()
            except BaseException:
                # Process teardown below is authoritative. A broken or wedged
                # response transport must never prevent it from running.
                pass

        client = self._client
        self._client = None
        if client is not None:
            try:
                await asyncio.wait_for(
                    client.aclose(),
                    timeout=self.transport_close_timeout_seconds,
                )
            except BaseException:
                pass

        process = self._process
        if terminate and process is not None and process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), self.termination_grace_seconds)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), self.kill_wait_seconds)
                except TimeoutError as exc:
                    raise DwellError(
                        "generation_failed",
                        "The mlx-lm server did not exit after bounded TERM/KILL shutdown.",
                        details={"pid": process.pid},
                        status_code=500,
                    ) from exc

        self._process = None
        self._loaded_model = None
        self._loaded_path = None
        self._server_token = None
        self._close_log_stream()

    async def _status_locked(self) -> RuntimeStatus:
        problem = await self._installation_problem()
        live = self._process_is_live()
        if live:
            message = f"model '{self._loaded_model}' is resident in the local mlx-lm server"
        else:
            message = problem or "available; no model is resident"
        return RuntimeStatus(
            runtime_id=self.runtime_id,
            available=problem is None,
            loaded_models=self._loaded_models() if live else (),
            active_jobs=self._active_jobs(),
            message=message,
        )

    async def _installation_problem(self) -> str | None:
        if self._installation_checked:
            return self._installation_problem_cache

        if self._installation_probe is not None:
            result = await _resolve_awaitable(self._installation_probe())
            if result is True:
                problem = None
            elif result is False:
                problem = f"mlx-lm {_PINNED_MLX_LM_VERSION} is not available"
            else:
                problem = str(result) if result else None
            self._installation_checked = True
            self._installation_problem_cache = problem
            return problem

        script = (
            "import importlib.metadata as m; import mlx_lm.server; "
            f"v=m.version('mlx-lm'); assert v == '{_PINNED_MLX_LM_VERSION}', v"
        )
        try:
            probe = await asyncio.create_subprocess_exec(
                self.python_binary,
                "-c",
                script,
                env=self.config.subprocess_env(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            problem = f"mlx-lm probe failed: {exc}"
            self._installation_checked = True
            self._installation_problem_cache = problem
            return problem
        try:
            _stdout, stderr = await asyncio.wait_for(
                probe.communicate(),
                timeout=self.installation_probe_timeout_seconds,
            )
        except TimeoutError:
            await self._stop_probe(probe)
            problem = (
                "mlx-lm probe failed: timed out after "
                f"{self.installation_probe_timeout_seconds:g} seconds"
            )
            self._installation_checked = True
            self._installation_problem_cache = problem
            return problem
        except asyncio.CancelledError:
            await asyncio.shield(self._stop_probe(probe))
            raise
        if probe.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip().splitlines()
            suffix = detail[-1] if detail else f"exit {probe.returncode}"
            problem = f"mlx-lm {_PINNED_MLX_LM_VERSION} is unavailable: {suffix}"
        else:
            problem = None
        self._installation_checked = True
        self._installation_problem_cache = problem
        return problem

    async def _stop_probe(self, probe: asyncio.subprocess.Process) -> None:
        if probe.returncode is None:
            try:
                probe.kill()
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(probe.wait(), timeout=self.probe_kill_wait_seconds)
        except TimeoutError:
            # asyncio's process transport will reap a killed child when the
            # loop receives SIGCHLD; never block daemon shutdown indefinitely.
            pass

    async def _raise_for_upstream_error(self, response: _Response) -> None:
        if response.status_code < 400:
            return
        raw = await response.aread()
        await response.aclose()
        message = raw.decode("utf-8", errors="replace")[:4096]
        status_code = response.status_code if response.status_code < 500 else 502
        code = "invalid_request" if response.status_code < 500 else "generation_failed"
        raise DwellError(
            code,
            "The mlx-lm runtime rejected the completion request."
            if response.status_code < 500
            else "The mlx-lm runtime failed while generating a completion.",
            details={"upstream_status": response.status_code, "upstream_body": message},
            status_code=status_code,
        )

    async def _acquire_request(self) -> None:
        if self._request_lock.locked():
            raise DwellError(
                "server_busy",
                "The mlx-lm runtime already has an active generation request.",
                status_code=429,
            )
        await self._request_lock.acquire()

    def _release_request(self) -> None:
        if self._request_lock.locked():
            self._request_lock.release()

    async def _stream_released(self) -> None:
        self._active_stream = None
        self._release_request()

    async def _cancel_active_generation(self) -> None:
        client = self._client
        if client is None:
            return

        async def request_ack() -> bool:
            response = await client.post("/dwell/cancel", json={})
            try:
                await response.aread()
                payload = response.json()
                requests = payload.get("requests") if isinstance(payload, Mapping) else None
                return (
                    response.status_code == 200
                    and isinstance(payload, Mapping)
                    and payload.get("status") == "stopped"
                    and isinstance(requests, int)
                    and not isinstance(requests, bool)
                    and requests > 0
                )
            finally:
                try:
                    await asyncio.wait_for(
                        response.aclose(),
                        timeout=self.transport_close_timeout_seconds,
                    )
                except BaseException:
                    pass

        try:
            stopped = await asyncio.wait_for(
                request_ack(),
                timeout=self.cancel_ack_timeout_seconds,
            )
        except Exception:
            stopped = False
        if stopped:
            return
        if self._release_in_progress:
            # The lifecycle owner that invoked stream.close() will terminate
            # the process immediately after the close callback returns.
            return

        # The private server could not acknowledge that decoding stopped.
        # Terminate it before releasing the shared inference lease; the next
        # request will lazily start a clean resident process.
        async with self._lifecycle_lock:
            await self._release_locked(close_active_stream=False)

    def _require_client(self) -> _Client:
        if self._client is None:
            raise DwellError(
                "runtime_not_available",
                "The local mlx-lm client is unavailable.",
                status_code=503,
            )
        return self._client

    def _process_is_live(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _loaded_models(self) -> tuple[str, ...]:
        return (self._loaded_model,) if self._loaded_model and self._process_is_live() else ()

    def _active_jobs(self) -> tuple[str, ...]:
        return ("chat-completion",) if self._request_lock.locked() else ()

    def _close_log_stream(self) -> None:
        stream = self._log_stream
        self._log_stream = None
        if stream is not None:
            stream.close()

    def _validate_model(self, model: ModelDefinition) -> None:
        if model.runtime != self.runtime_id:
            raise DwellError(
                "runtime_not_available",
                f"Model '{model.id}' requires runtime '{model.runtime}', not '{self.runtime_id}'.",
                status_code=409,
            )

    @staticmethod
    def _validate_model_path(model: ModelDefinition, model_path: Path) -> Path:
        try:
            resolved = Path(model_path).resolve(strict=True)
        except OSError as exc:
            raise DwellError(
                "model_not_installed",
                f"Model '{model.id}' has no complete local snapshot.",
                details={"path": str(model_path)},
                status_code=409,
            ) from exc
        if not resolved.is_dir():
            raise DwellError(
                "model_not_installed",
                f"Model '{model.id}' local snapshot is not a directory.",
                details={"path": str(resolved)},
                status_code=409,
            )
        return resolved

    @staticmethod
    def _upstream_unavailable(exc: Exception) -> DwellError:
        return DwellError(
            "runtime_not_available",
            "The local mlx-lm server became unavailable.",
            details={"reason": str(exc)},
            status_code=503,
        )


# Compatibility spelling for callers that distinguish adapter and runtime roles.
MLXLMTextEngine = MLXLMRuntime
