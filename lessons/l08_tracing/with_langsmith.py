from langsmith import traceable, tracing_context
from langsmith.run_trees import RunTree
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from lessons import MODEL
from lessons.l08_tracing.problem import retrieve_context

client = wrap_openai(OpenAI())


@traceable(run_type="tool")
def retrieve(question: str) -> str:
    return retrieve_context(question)


@traceable(run_type="chain")
def generate(question: str, context: str) -> str:
    messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]
    return client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content


@traceable(run_type="chain")
def answer(question: str) -> str:
    return generate(question, retrieve(question))


def to_span(run: RunTree) -> dict:
    usage = (run.outputs or {}).get("usage_metadata") or {}
    return {
        "name": run.name,
        "kind": run.run_type,
        "error": run.error is not None,
        "tokens": usage.get("total_tokens", 0),
        "children": [to_span(child) for child in run.child_runs],
    }


def run(question: str) -> dict:
    finished: list[RunTree] = []
    with tracing_context(enabled="local"):
        try:
            answer(question, langsmith_extra={"on_end": finished.append})
        except PermissionError:
            pass
    return to_span(finished[0])
