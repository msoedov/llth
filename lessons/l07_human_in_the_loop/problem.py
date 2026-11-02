"""
Human in the loop.
Source: https://docs.langchain.com/oss/python/langgraph/interrupts

Pause before a side effect, wait for a person, then resume where the run left
off. A rejected action must never execute. An approved action executes once.

Input is {"request": str, "approve": bool}. The decision stands in for the
person answering. Output is {"paused": bool, "result": str}: whether the run
stopped for approval, and what happened after the decision.
"""

from evals.capture import recorded
from evals.checks import called, field_is, not_called
from evals.harness import Case

SYSTEM_PROMPT = "You are an email assistant. Use send_email for every request."
REQUEST = "Email finance@acme.dev that the Q3 report is ready."


@recorded
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


def comparable(output: dict) -> dict:
    return {"paused": output["paused"]}


CASES = [
    Case(
        "approved",
        {"request": REQUEST, "approve": True},
        [field_is("paused", True), called("send_email", times=1)],
    ),
    Case(
        "rejected",
        {"request": REQUEST, "approve": False},
        [field_is("paused", True), not_called("send_email")],
    ),
]
