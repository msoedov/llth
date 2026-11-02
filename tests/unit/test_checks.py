import pytest

from evals.capture import Trace
from evals.checks import called, called_with, contains, excludes, field_is, max_llm_calls, not_called, used_embeddings

TRACE = Trace(
    tool_calls=[{"name": "add", "args": [], "kwargs": {"city": "San Francisco"}}, {"name": "add", "args": [], "kwargs": {}}],
    llm_calls=2,
)


@pytest.mark.parametrize(
    ("check", "output", "expected"),
    [
        (contains("sunny"), "It is SUNNY", True),
        (contains("a", "b"), "a only", False),
        (contains("bob", at=-1), ["carol", "Bob"], True),
        (contains("bob", at=0), ["carol", "Bob"], False),
        (excludes("bob"), "Carol", True),
        (excludes("bob", at=-1), ["x", "bob"], False),
        (field_is("phone", None), {"phone": None}, True),
        (field_is("phone", None), {"phone": "1"}, False),
        (called("add"), "", True),
        (called("add", times=1), "", False),
        (called("add", times=2), "", True),
        (called("divide"), "", False),
        (not_called(), "", False),
        (not_called("add"), "", False),
        (not_called("divide"), "", True),
        (called_with("add", city="francisco"), "", True),
        (called_with("add", city="paris"), "", False),
        (called_with("divide", city="francisco"), "", False),
        (max_llm_calls(2), "", True),
        (max_llm_calls(1), "", False),
        (used_embeddings(), "", False),
    ],
)
def test_checks(check, output, expected):
    assert check.fn(output, TRACE) is expected
