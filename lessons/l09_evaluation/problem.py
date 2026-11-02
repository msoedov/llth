"""
Evaluation.
Source: https://docs.langchain.com/langsmith/evaluation-quickstart

Run a target over a dataset, score every row, and aggregate. A row whose target
crashes must not stop the run, and it scores None rather than 0.

The target needs no model, so the expected scores are exact. Output is
{"scores": [one per row], "mean": average of the non-None scores}.

The reference runs LangSmith offline (upload_results=False), so it needs no
LangSmith key. This lesson has no Agno reference. The evals/ package in this
repo is itself a hard-way answer to this lesson.
"""

from evals.checks import field_is
from evals.harness import Case

CAPITALS = {"France": "Paris", "Japan": "Kyoto"}


def target(inputs: dict) -> dict:
    return {"answer": CAPITALS[inputs["country"]]}


def correct(outputs: dict, reference_outputs: dict) -> dict:
    return {"key": "correct", "score": float(outputs["answer"] == reference_outputs["answer"])}


def row(country: str, answer: str) -> dict:
    return {"inputs": {"country": country}, "outputs": {"answer": answer}}


CASES = [
    Case("all_correct", [row("France", "Paris")], [field_is("scores", [1.0]), field_is("mean", 1.0)]),
    Case(
        "mixed",
        [row("France", "Paris"), row("Japan", "Tokyo")],
        [field_is("scores", [1.0, 0.0]), field_is("mean", 0.5)],
    ),
    Case(
        "target_crashes",
        [row("France", "Paris"), row("Atlantis", "Poseidonis"), row("Japan", "Tokyo")],
        [field_is("scores", [1.0, None, 0.0]), field_is("mean", 0.5)],
    ),
]
