import functools
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import httpx2


@dataclass
class Trace:
    tool_calls: list[dict] = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)
    llm_calls: int = 0
    embedding_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0


_active: Trace | None = None


def recorded(fn: Callable) -> Callable:
    """Marks a tool so every framework's call to it lands in the Trace."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _active is not None:
            _active.tool_calls.append({"name": fn.__name__, "args": list(args), "kwargs": kwargs})
        return fn(*args, **kwargs)

    return wrapper


def _body(request: Any) -> Any:
    try:
        return json.loads(request.content)
    except (ValueError, httpx.RequestNotRead, httpx2.RequestNotRead):
        return None


def _observe(request: Any, response: Any, streamed: bool) -> None:
    if _active is None:
        return
    path = request.url.path
    is_embedding = path.endswith("/embeddings")
    if not is_embedding and not path.endswith(("/chat/completions", "/responses")):
        return
    _active.requests.append({"path": path, "body": _body(request)})
    usage = {}
    if not streamed and "json" in response.headers.get("content-type", ""):
        usage = response.json().get("usage") or {}
    if is_embedding:
        _active.embedding_calls += 1
        _active.embedding_tokens += usage.get("prompt_tokens", 0)
        return
    _active.llm_calls += 1
    _active.input_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0))
    _active.output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0))


def _patch_sync(client: Any) -> None:
    original = client.send

    def send(self: Any, request: Any, **kwargs: Any) -> Any:
        response = original(self, request, **kwargs)
        streamed = bool(kwargs.get("stream"))
        if not streamed:
            response.read()
        _observe(request, response, streamed)
        return response

    client.send = send


def _patch_async(client: Any) -> None:
    original = client.send

    async def send(self: Any, request: Any, **kwargs: Any) -> Any:
        response = await original(self, request, **kwargs)
        streamed = bool(kwargs.get("stream"))
        if not streamed:
            await response.aread()
        _observe(request, response, streamed)
        return response

    client.send = send


for _lib in (httpx, httpx2):
    _patch_sync(_lib.Client)
    _patch_async(_lib.AsyncClient)


@contextmanager
def capture() -> Iterator[Trace]:
    global _active
    _active = Trace()
    try:
        yield _active
    finally:
        _active = None
