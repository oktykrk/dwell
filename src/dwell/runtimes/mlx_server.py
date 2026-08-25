from __future__ import annotations

import argparse
import hmac
import importlib.metadata
import json
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any

_SERVER_TOKEN_ENV = "DWELL_MLX_SERVER_TOKEN"
_PARENT_PID_ENV = "DWELL_MLX_PARENT_PID"
_PINNED_VERSION_ENV = "DWELL_MLX_LM_VERSION"
_DEFAULT_PINNED_VERSION = "0.31.3"


class _RequestToken:
    def __init__(self) -> None:
        self.cancelled = False


class _GenerationController:
    """Track private generation contexts and provide bounded stop acknowledgement."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._contexts: dict[int, Any] = {}
        self._requests: dict[int, Any] = {}
        self._local = threading.local()

    def begin_request(self) -> _RequestToken:
        token = _RequestToken()
        with self._condition:
            self._requests[id(token)] = token
        self._local.request_token = token
        return token

    def end_request(self, token: _RequestToken) -> None:
        with self._condition:
            self._requests.pop(id(token), None)
            self._condition.notify_all()
        if getattr(self._local, "request_token", None) is token:
            del self._local.request_token

    def track(self, context: Any) -> None:
        with self._condition:
            self._contexts[id(context)] = context
            token = getattr(self._local, "request_token", None)
            if token is not None:
                if token.cancelled:
                    context.stop()

    def clear(self, context: Any) -> None:
        with self._condition:
            self._contexts.pop(id(context), None)
            self._condition.notify_all()

    def cancel_and_wait(self, timeout: float) -> tuple[int, bool]:
        with self._condition:
            requests = tuple(self._requests.values())
            contexts = tuple(self._contexts.values())
            for request in requests:
                request.cancelled = True
            for context in contexts:
                context.stop()
            stopped = self._condition.wait_for(
                lambda: not self._requests and not self._contexts,
                timeout=timeout,
            )
            return max(len(requests), len(contexts)), stopped


def _track_response_generator(response_generator: Any, controller: _GenerationController) -> None:
    original_generate = response_generator.generate

    def generate_and_track(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        context, response = original_generate(*args, **kwargs)
        controller.track(context)

        def tracked_response():  # type: ignore[no-untyped-def]
            try:
                yield from response
            finally:
                controller.clear(context)

        return context, tracked_response()

    response_generator.generate = generate_and_track


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dwell's private MLX-LM server wrapper.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--decode-concurrency", type=int, default=1)
    parser.add_argument("--prompt-concurrency", type=int, default=1)
    parser.add_argument("--prompt-cache-size", type=int, default=1)
    return parser


def _parent_is_alive(parent_pid: int) -> bool:
    if parent_pid <= 1 or os.getppid() != parent_pid:
        return False
    try:
        os.kill(parent_pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _watch_parent(parent_pid: int, stop: threading.Event) -> None:
    while not stop.wait(0.5):
        if not _parent_is_alive(parent_pid):
            os.kill(os.getpid(), signal.SIGTERM)
            return


def _send_json(handler: Any, status_code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)
    handler.wfile.flush()


def _authenticated_handler(
    base: type,
    token: str,
    ready: threading.Event,
    controller: _GenerationController | None = None,
    *,
    cancel_wait_seconds: float = 2.0,
) -> type:
    generation_controller = controller or _GenerationController()

    class AuthenticatedAPIHandler(base):  # type: ignore[misc, valid-type]
        def _dwell_authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {token}"
            if hmac.compare_digest(supplied, expected):
                return True
            _send_json(self, 401, {"error": "unauthorized"})
            return False

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._dwell_authorized():
                return
            if self.path == "/dwell/ready":
                if ready.is_set():
                    _send_json(self, 200, {"status": "ready"})
                else:
                    _send_json(self, 503, {"status": "loading"})
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if not self._dwell_authorized():
                return
            if self.path == "/dwell/cancel":
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length:
                    self.rfile.read(content_length)
                requested, stopped = generation_controller.cancel_and_wait(cancel_wait_seconds)
                _send_json(
                    self,
                    200,
                    {"status": "stopped" if stopped else "stopping", "requests": requested},
                )
                return
            request_token = generation_controller.begin_request()
            try:
                try:
                    super().do_POST()
                except (BrokenPipeError, ConnectionResetError):
                    # Client cancellation is routine for interactive agent streams.
                    # Any context that did not reach its iterator finalizer stays
                    # tracked, so cancellation cannot falsely acknowledge it and
                    # the parent will replace this child before accepting work.
                    return
            finally:
                generation_controller.end_request(request_token)

    return AuthenticatedAPIHandler


def _validated_environment() -> tuple[str, int]:
    token = os.environ.get(_SERVER_TOKEN_ENV, "")
    if len(token) < 32:
        raise RuntimeError(f"{_SERVER_TOKEN_ENV} must contain a private random token")
    raw_parent = os.environ.get(_PARENT_PID_ENV, "")
    try:
        parent_pid = int(raw_parent)
    except ValueError as exc:
        raise RuntimeError(f"{_PARENT_PID_ENV} must contain the Dwell parent PID") from exc
    if not _parent_is_alive(parent_pid):
        raise RuntimeError("the Dwell parent process is not alive")
    expected_version = os.environ.get(_PINNED_VERSION_ENV, _DEFAULT_PINNED_VERSION)
    installed_version = importlib.metadata.version("mlx-lm")
    if installed_version != expected_version:
        raise RuntimeError(f"mlx-lm {expected_version} is required; found {installed_version}")
    return token, parent_pid


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.host != "127.0.0.1":
        raise RuntimeError("the private mlx-lm server may only bind to 127.0.0.1")
    if not 1 <= args.port <= 65535:
        raise RuntimeError("port must be between 1 and 65535")
    if args.max_tokens != 8192:
        raise RuntimeError("the Dwell mlx-lm output ceiling must be 8192 tokens")
    if args.decode_concurrency != 1 or args.prompt_concurrency != 1 or args.prompt_cache_size != 1:
        raise RuntimeError("the Dwell mlx-lm server requires single-request concurrency")
    try:
        model_path = args.model.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"model path is unavailable: {args.model}") from exc
    if not model_path.is_dir():
        raise RuntimeError(f"model path is not a directory: {model_path}")

    token, parent_pid = _validated_environment()

    # Importing mlx-lm initializes MLX/Metal, so it happens only in the private
    # child after all cheap configuration and ownership checks pass.
    from mlx_lm import server as mlx_server

    ready = threading.Event()
    generation_controller = _GenerationController()
    stop_watchdog = threading.Event()
    original_load_default = mlx_server.ModelProvider.load_default
    original_run_http_server = mlx_server._run_http_server

    def load_default_and_signal(provider: Any) -> None:
        original_load_default(provider)
        ready.set()

    handler = _authenticated_handler(
        mlx_server.APIHandler,
        token,
        ready,
        generation_controller,
    )

    def run_private_http_server(host: str, port: int, response_generator: Any) -> None:
        _track_response_generator(response_generator, generation_controller)
        original_run_http_server(
            host,
            port,
            response_generator,
            handler_class=handler,
        )

    mlx_server.ModelProvider.load_default = load_default_and_signal
    mlx_server._run_http_server = run_private_http_server
    watchdog = threading.Thread(
        target=_watch_parent,
        args=(parent_pid, stop_watchdog),
        name="dwell-mlx-parent-watchdog",
        daemon=True,
    )
    watchdog.start()

    forwarded = [
        "mlx_lm.server",
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-tokens",
        "8192",
        "--decode-concurrency",
        "1",
        "--prompt-concurrency",
        "1",
        "--prompt-cache-size",
        "1",
        "--allowed-origins",
        "http://127.0.0.1",
    ]
    previous_argv = sys.argv
    try:
        sys.argv = forwarded
        mlx_server.main()
    finally:
        stop_watchdog.set()
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
