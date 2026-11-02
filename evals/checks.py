from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evals.capture import Trace


@dataclass
class Check:
    name: str
    fn: Callable[[Any, Trace], bool]


def _text(output: Any, at: int | None) -> str:
    value = output if at is None else output[at]
    return str(value).lower()


def contains(*needles: str, at: int | None = None) -> Check:
    return Check(
        f"contains {needles}",
        lambda output, _: all(n.lower() in _text(output, at) for n in needles),
    )


def excludes(*needles: str, at: int | None = None) -> Check:
    return Check(
        f"excludes {needles}",
        lambda output, _: not any(n.lower() in _text(output, at) for n in needles),
    )


def field_is(name: str, expected: Any) -> Check:
    return Check(f"{name} == {expected!r}", lambda output, _: output.get(name) == expected)


def _names(trace: Trace) -> list[str]:
    return [call["name"] for call in trace.tool_calls]


def called(tool: str, times: int | None = None) -> Check:
    def check(_: Any, trace: Trace) -> bool:
        count = _names(trace).count(tool)
        return count >= 1 if times is None else count == times

    return Check(f"called {tool}" + ("" if times is None else f" x{times}"), check)


def not_called(tool: str | None = None) -> Check:
    if tool is None:
        return Check("no tool calls", lambda _, trace: not trace.tool_calls)
    return Check(f"did not call {tool}", lambda _, trace: tool not in _names(trace))


def called_with(tool: str, **expected: Any) -> Check:
    def check(_: Any, trace: Trace) -> bool:
        return any(
            call["name"] == tool and all(expected[k].lower() in str(call["kwargs"].get(k, "")).lower() for k in expected)
            for call in trace.tool_calls
        )

    return Check(f"called {tool} with {expected}", check)


def max_llm_calls(limit: int) -> Check:
    return Check(f"llm calls <= {limit}", lambda _, trace: trace.llm_calls <= limit)


def used_embeddings() -> Check:
    return Check("used embeddings", lambda _, trace: trace.embedding_calls > 0)
