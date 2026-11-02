import json

import httpx
import httpx2
import pytest
from inline_snapshot import snapshot

from evals.capture import capture, recorded


def backend(body: dict, content_type: str = "application/json", lib=httpx2):
    return lib.Client(
        transport=lib.MockTransport(
            lambda _: lib.Response(200, content=json.dumps(body), headers={"content-type": content_type})
        )
    )


@pytest.mark.parametrize(
    ("path", "usage", "expected"),
    [
        ("/v1/chat/completions", {"prompt_tokens": 3, "completion_tokens": 2}, (1, 0, 3, 2, 0)),
        ("/v1/responses", {"input_tokens": 5, "output_tokens": 4}, (1, 0, 5, 4, 0)),
        ("/v1/embeddings", {"prompt_tokens": 7}, (0, 1, 0, 0, 7)),
        ("/v1/models", {"prompt_tokens": 9}, (0, 0, 0, 0, 0)),
    ],
)
@pytest.mark.parametrize("lib", [httpx, httpx2])
def test_counts_openai_traffic(lib, path, usage, expected):
    client = backend({"usage": usage}, lib=lib)
    with capture() as trace:
        client.post(f"https://api.test{path}", json={"model": "m"})
    assert (trace.llm_calls, trace.embedding_calls, trace.input_tokens, trace.output_tokens, trace.embedding_tokens) == expected


def test_stores_request_body_without_headers():
    with capture() as trace:
        backend({}).post("https://api.test/v1/responses", json={"model": "m"}, headers={"authorization": "Bearer secret"})
    assert trace.requests == snapshot([{"path": "/v1/responses", "body": {"model": "m"}}])


def test_non_json_response_counts_call_without_tokens():
    with capture() as trace:
        backend({}, content_type="text/event-stream").post("https://api.test/v1/responses", json={})
    assert (trace.llm_calls, trace.input_tokens) == (1, 0)


def test_nothing_recorded_outside_capture():
    tool = recorded(lambda: "ok")
    backend({}).post("https://api.test/v1/responses", json={})
    assert tool() == "ok"
    with capture() as trace:
        pass
    assert trace.llm_calls == 0
    assert trace.tool_calls == []


def test_recorded_tool_keeps_identity_and_logs_args():
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    tool = recorded(add)
    with capture() as trace:
        assert tool(1, b=2) == 3
    assert tool.__name__ == "add"
    assert tool.__doc__ == "Add."
    assert trace.tool_calls == snapshot([{"name": "add", "args": [1], "kwargs": {"b": 2}}])
