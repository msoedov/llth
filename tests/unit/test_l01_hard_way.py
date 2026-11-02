import json

import httpx
import pytest
from inline_snapshot import snapshot

from lessons.l01_tool_agent import hard_way
from lessons.l01_tool_agent.hard_way import MAX_STEPS, Agent, tool_schema
from lessons.l01_tool_agent.problem import get_weather


def add(a: int, b: float = 1.5, *, verbose: bool = False) -> float:
    """Add two numbers.

    Second line of the description.
    """
    return a + b


def no_docstring() -> str:
    return "ok"


def untyped(city):
    return city


def typed_generic(cities: list[str]) -> str:
    return ",".join(cities)


@pytest.mark.parametrize(
    ("fn", "expected"),
    [
        (
            get_weather,
            snapshot(
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a given city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ),
        ),
        (
            add,
            snapshot(
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.\n\nSecond line of the description.",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}, "b": {"type": "number"}, "verbose": {"type": "boolean"}},
                            "required": ["a"],
                        },
                    },
                }
            ),
        ),
        (
            no_docstring,
            snapshot(
                {
                    "type": "function",
                    "function": {
                        "name": "no_docstring",
                        "description": None,
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    },
                }
            ),
        ),
    ],
)
def test_tool_schema(fn, expected):
    assert tool_schema(fn) == expected


@pytest.mark.parametrize("fn", [untyped, typed_generic])
def test_tool_schema_rejects_unmapped_annotations(fn):
    with pytest.raises(KeyError):
        tool_schema(fn)


def tool_call(name: str, arguments: str, id: str = "call_1") -> dict:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(**message) -> dict:
    return {"choices": [{"message": {"role": "assistant", **message}}]}


@pytest.fixture
def server(monkeypatch):
    """Scripted OpenRouter: answers each POST with the next body and keeps the requests."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    bodies: list[dict] = []
    requests: list[dict] = []

    def post(url, headers, json, timeout):
        requests.append({"url": url, "headers": headers, "json": json})
        request = httpx.Request("POST", url)
        status = 200 if bodies else 500
        return httpx.Response(status, json=bodies.pop(0) if bodies else {}, request=request)

    monkeypatch.setattr(hard_way.httpx, "post", post)
    return bodies, requests


def test_run_answers_directly_without_tools(server):
    bodies, requests = server
    bodies.append(reply(content="4"))
    assert Agent([get_weather], "be nice").run("2+2?") == "4"
    assert len(requests) == 1
    assert requests[0]["url"] == snapshot("https://openrouter.ai/api/v1/chat/completions")
    assert requests[0]["headers"] == snapshot({"Authorization": "Bearer test-key"})
    assert requests[0]["json"]["model"] == snapshot("openai/gpt-5.4-mini")
    assert requests[0]["json"]["messages"] == snapshot(
        [{"role": "system", "content": "be nice"}, {"role": "user", "content": "2+2?"}]
    )


def test_run_executes_tool_calls_and_sends_results_back(server):
    bodies, requests = server
    calls = [tool_call("get_weather", '{"city": "Paris"}', "a"), tool_call("get_weather", '{"city": "Tokyo"}', "b")]
    bodies += [reply(content=None, tool_calls=calls), reply(content="both sunny")]
    assert Agent([get_weather], "p").run("q") == "both sunny"
    assert requests[1]["json"]["messages"][2:] == snapshot(
        [
            {"role": "assistant", "content": None, "tool_calls": calls},
            {"role": "tool", "tool_call_id": "a", "content": "It's always sunny in Paris!"},
            {"role": "tool", "tool_call_id": "b", "content": "It's always sunny in Tokyo!"},
        ]
    )


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (tool_call("unknown", "{}"), KeyError),
        (tool_call("get_weather", "not json"), json.JSONDecodeError),
        (tool_call("get_weather", '{"town": "Paris"}'), TypeError),
    ],
)
def test_run_raises_on_bad_tool_call(server, call, error):
    bodies, _ = server
    bodies.append(reply(content=None, tool_calls=[call]))
    with pytest.raises(error):
        Agent([get_weather], "p").run("q")


def test_run_stops_after_max_steps(server):
    bodies, requests = server
    looping = reply(content=None, tool_calls=[tool_call("get_weather", '{"city": "Paris"}')])
    bodies += [looping] * (MAX_STEPS + 1)
    with pytest.raises(RuntimeError, match="no final answer"):
        Agent([get_weather], "p").run("q")
    assert len(requests) == MAX_STEPS


def test_run_raises_on_http_error(server):
    with pytest.raises(httpx.HTTPStatusError):
        Agent([get_weather], "p").run("q")


def test_run_requires_api_key(server, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(KeyError):
        Agent([get_weather], "p").run("q")
