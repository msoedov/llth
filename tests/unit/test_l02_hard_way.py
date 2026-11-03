import json
from dataclasses import dataclass
from typing import Optional

import httpx
import pytest
from inline_snapshot import snapshot

from lessons.l02_structured_output import hard_way
from lessons.l02_structured_output.hard_way import ContactInfo, json_type, response_format


@dataclass
class Empty:
    pass


@dataclass
class Mixed:
    count: int
    score: float | None
    active: bool


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (str, snapshot("string")),
        (int, snapshot("integer")),
        (float, snapshot("number")),
        (bool, snapshot("boolean")),
        (str | None, snapshot(["string", "null"])),
        (Optional[str], snapshot(["string", "null"])),
        (int | str | None, snapshot(["integer", "string", "null"])),
    ],
)
def test_json_type(annotation, expected):
    assert json_type(annotation) == expected


@pytest.mark.parametrize("annotation", [list[str], dict, bytes])
def test_json_type_rejects_unmapped(annotation):
    with pytest.raises(KeyError):
        json_type(annotation)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (
            Mixed,
            snapshot({'type': 'json_schema', 'json_schema': {'name': 'Mixed', 'strict': True, 'schema': {'type': 'object', 'properties': {'count': {'type': 'integer'}, 'score': {'type': ['number', 'null']}, 'active': {'type': 'boolean'}}, 'required': ['count', 'score', 'active'], 'additionalProperties': False}}}),
        ),
        (
            Empty,
            snapshot({'type': 'json_schema', 'json_schema': {'name': 'Empty', 'strict': True, 'schema': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}),
        ),
    ],
)
def test_response_format(cls, expected):
    assert response_format(cls) == expected


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


def test_run_sends_strict_schema(server):
    bodies, requests = server
    bodies.append(reply(content='{"name": "Ana Lee", "email": null, "phone": null}'))
    hard_way.run("My name is Ana Lee.")
    assert len(requests) == 1
    assert requests[0]["url"] == snapshot("https://openrouter.ai/api/v1/chat/completions")
    assert requests[0]["headers"] == snapshot({"Authorization": "Bearer test-key"})
    assert requests[0]["json"] == snapshot(
        {
            "model": "openai/gpt-5.4-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "Extract contact info from the text. Use null for anything the text does not state.",
                },
                {"role": "user", "content": "My name is Ana Lee."},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ContactInfo",
                    "strict": True,
                    "schema": {"type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": ["string", "null"]},
                            "phone": {"type": ["string", "null"]},
                        },
                        "required": ["name", "email", "phone"],
                        "additionalProperties": False,
                    },
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            '{"name": "John Doe", "email": "john@example.com", "phone": "(555) 123-4567"}',
            ContactInfo(name="John Doe", email="john@example.com", phone="(555) 123-4567"),
        ),
        ('{"name": "Ana Lee", "email": null, "phone": null}', ContactInfo(name="Ana Lee", email=None, phone=None)),
    ],
)
def test_run_parses_content(server, content, expected):
    bodies, _ = server
    bodies.append(reply(content=content))
    assert hard_way.run("text") == expected


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"content": None, "refusal": "I can't help with that."}, ValueError),
        ({"content": '{"name": "Ana'}, json.JSONDecodeError),
        ({"content": None}, TypeError),
        ({"content": '{"name": "Ana Lee", "email": null}'}, TypeError),
        ({"content": '{"name": "Ana Lee", "email": null, "phone": null, "age": 3}'}, TypeError),
    ],
)
def test_run_raises_on_bad_reply(server, message, error):
    bodies, _ = server
    bodies.append(reply(**message))
    with pytest.raises(error):
        hard_way.run("text")


def test_run_raises_on_http_error(server):
    with pytest.raises(httpx.HTTPStatusError):
        hard_way.run("text")


def test_run_requires_api_key(server, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(KeyError):
        hard_way.run("text")
