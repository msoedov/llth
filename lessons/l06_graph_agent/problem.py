"""
Agent as a graph.
Source: https://docs.langchain.com/oss/python/langgraph/quickstart

Model the agent loop explicitly: call the model, run requested tools, go back to
the model, stop when it asks for no tools. The state tracks the messages. Tool
failures go back to the model as text so it can explain them.
"""

from evals.capture import recorded
from evals.checks import called, contains, max_llm_calls
from evals.harness import Case

SYSTEM_PROMPT = "You are a helpful assistant tasked with performing arithmetic. Use the tools for every operation."


@recorded
def add(a: float, b: float) -> float:
    """Add a and b."""
    return a + b


@recorded
def multiply(a: float, b: float) -> float:
    """Multiply a and b."""
    return a * b


@recorded
def divide(a: float, b: float) -> float | str:
    """Divide a by b."""
    return a / b if b else "error: division by zero"


CASES = [
    Case("one_step", "Add 3 and 4.", [called("add", times=1), contains("7"), max_llm_calls(2)]),
    Case(
        "chained_steps",
        "Add 3 and 4, multiply the result by 5, then divide that by 7.",
        [called("add", times=1), called("multiply", times=1), called("divide", times=1), contains("5"), max_llm_calls(4)],
    ),
    Case("tool_reports_error", "Divide 10 by 0.", [called("divide", times=1), contains("zero"), max_llm_calls(2)]),
]
