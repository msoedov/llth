import pytest
from inline_snapshot import snapshot

from evals.capture import recorded
from evals.checks import called, contains
from evals.harness import (
    Case,
    agreement,
    evaluate,
    implementations,
    lesson_names,
    lines_of_code,
    report,
    resolve,
    run_case,
    summarize,
    table,
    third_party_imports,
)

SOURCE = '''"""Module doc."""
import json
import numpy as np
from lessons import MODEL
from openai import OpenAI

# comment

def f() -> str:
    """Doc."""
    return "x"
'''


def test_lines_of_code_skips_comments_docstrings_and_blanks():
    assert lines_of_code(SOURCE) == 6


def test_third_party_imports_skip_stdlib_and_project():
    assert third_party_imports(SOURCE) == ["numpy", "openai"]


def tool():
    return recorded(lambda: "done")


def test_run_case_records_passes_and_failures():
    result = run_case(lambda _: "hello", Case("c", "in", [contains("hello"), contains("bye")]))
    assert result["output"] == "hello"
    assert result["checks"] == snapshot({"contains ('hello',)": True, "contains ('bye',)": False})


def test_run_case_turns_exception_into_failed_checks():
    def boom(_):
        raise ValueError("bad")

    result = run_case(boom, Case("c", "in", [contains("x")]))
    assert result["error"] == "ValueError: bad"
    assert result["checks"] == snapshot({"contains ('x',)": False})


def test_run_case_counts_a_crashing_check_as_failed():
    result = run_case(lambda _: {"not": "a list"}, Case("c", "in", [contains("x", at=5)]))
    assert result["error"] is None
    assert list(result["checks"].values()) == [False]


def test_run_case_propagates_stub():
    def stub(_):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        run_case(stub, Case("c", "in", [called("x")]))


def never_judge(*_):
    raise AssertionError("judge must not run")


@pytest.mark.parametrize(
    ("reference", "candidate", "expected"),
    [
        ("same", "same", (True, "identical")),
        ({"a": 1, "b": 2}, {"a": 1, "b": 3}, (False, "1/2 fields match")),
        ({"a": 1}, {"a": 1, "extra": 2}, (True, "1/1 fields match")),
        ("text", None, (False, "no output")),
    ],
)
def test_agreement_without_judge(reference, candidate, expected):
    assert agreement("q", reference, candidate, never_judge) == expected


def test_agreement_defers_text_to_judge():
    assert agreement("q", "a", "b", lambda *_: (True, "same facts")) == (True, "same facts")


def test_table_aligns_columns():
    assert table([{"a": 1, "long": "x"}, {"a": 200, "long": "y"}]) == snapshot("a    long\n1    x   \n200  y   ")


def test_every_lesson_has_cases_and_entry_points():
    for lesson in lesson_names():
        assert implementations(lesson)[-1] == "hard_way"
        assert len(implementations(lesson)) >= 2, lesson


def test_resolve_rejects_unknown_prefix():
    assert resolve("l01") == "l01_tool_agent"
    with pytest.raises(SystemExit):
        resolve("l99")


RESULT = {
    "cases": {
        "uses_tool": {
            "output": "sunny",
            "error": None,
            "seconds": 1.0,
            "checks": {"a": True, "b": False},
            "tool_calls": [],
            "llm_calls": 2,
            "input_tokens": 10,
            "output_tokens": 5,
            "embedding_tokens": 0,
        }
    },
    "complexity": {"loc": 6, "deps": ["agno"], "import_seconds": 0.5, "modules": 100},
}


def test_summarize():
    assert summarize(RESULT) == snapshot(
        {
            "checks": "1/2",
            "llm": 2,
            "in_tok": 10,
            "out_tok": 5,
            "embed_tok": 0,
            "sec": "1.0",
            "loc": 6,
            "deps": 1,
            "import_s": "0.50",
            "modules": 100,
        }
    )


def test_report_without_hard_way_lists_failures_and_todo():
    text = report("l01_tool_agent", {"with_agno": RESULT, "hard_way": None})
    assert "with_agno / uses_tool: ['b']" in text
    assert text.endswith("hard_way: not implemented yet")


def test_evaluate_returns_none_for_stub():
    assert evaluate("l09_evaluation", "hard_way") is None
