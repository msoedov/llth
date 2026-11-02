"""
Tool-calling agent.
Source: https://docs.langchain.com/oss/python/langchain/quickstart

Let a model decide when to call a Python function, run the function, send the
result back, and finish with a natural-language answer. Questions that need no
tool must be answered directly.
"""

from evals.capture import recorded
from evals.checks import called, called_with, contains, max_llm_calls, not_called
from evals.harness import Case

SYSTEM_PROMPT = "You are a helpful assistant"


@recorded
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


CASES = [
    Case(
        "uses_tool",
        "What's the weather in San Francisco?",
        [called("get_weather", times=1), called_with("get_weather", city="San Francisco"), contains("sunny"), max_llm_calls(2)],
    ),
    Case(
        "two_cities",
        "Compare the weather in Paris and Tokyo.",
        [called("get_weather", times=2), contains("Paris", "Tokyo"), max_llm_calls(3)],
    ),
    Case("no_tool_needed", "What is 2 + 2? Reply with the number only.", [not_called(), contains("4"), max_llm_calls(1)]),
]
