"""
Tracing.
Source: https://docs.langchain.com/langsmith/trace-with-langchain

See what a run did without rewriting it. Instrument a small two-step app and
produce a span tree recording nesting, kind, failures, and LLM token usage. A
failing step must still leave a complete tree behind.

The app: answer(question) calls retrieve(question), then generate(question,
context), which calls the model once. Output is the root span:
{"name": str, "kind": "chain"|"tool"|"llm", "error": bool, "tokens": int, "children": [spans]}

This lesson has no Agno reference: Agno's tracing runs through AgentOS and a
database, which is a different problem from a local span tree.
"""

from evals.checks import Check
from evals.harness import Case

CONTEXT = "The Halvorsen Lab muster point is the east bicycle shed."


def retrieve_context(question: str) -> str:
    if "secret" in question:
        raise PermissionError("restricted document")
    return CONTEXT


def shape(span: dict) -> dict:
    return {"kind": span["kind"], "error": span["error"], "children": [shape(child) for child in span["children"]]}


comparable = shape


def llm_span(span: dict) -> dict:
    return span["children"][1]["children"][0]


def has_shape(expected: dict) -> Check:
    return Check("span tree shape", lambda output, _: shape(output) == expected)


def leaf(kind: str, error: bool = False) -> dict:
    return {"kind": kind, "error": error, "children": []}


CASES = [
    Case(
        "success",
        "Where is the muster point?",
        [
            has_shape(
                {
                    "kind": "chain",
                    "error": False,
                    "children": [leaf("tool"), {"kind": "chain", "error": False, "children": [leaf("llm")]}],
                }
            ),
            Check("llm tokens recorded", lambda output, _: llm_span(output)["tokens"] > 0),
        ],
    ),
    Case(
        "failing_step",
        "Show me the secret report.",
        [has_shape({"kind": "chain", "error": True, "children": [leaf("tool", error=True)]})],
    ),
]
