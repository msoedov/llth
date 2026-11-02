"""
Short-term memory.
Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Keep a conversation going across calls. Each thread sees only its own history,
and a thread stays usable after many turns.

Input is a list of (thread_id, message) turns. Output is one reply per turn.
"""

from evals.checks import contains, excludes
from evals.harness import Case

SYSTEM_PROMPT = "You are a helpful assistant. Keep replies to one short sentence."
FORGOT = "If you do not know, say you do not know."
FILLER = [("a", f"Name one fact about the number {n}.") for n in range(1, 7)]

CASES = [
    Case(
        "recalls",
        [("a", "My name is Bob."), ("a", "What's my name?")],
        [contains("Bob", at=-1)],
    ),
    Case(
        "threads_are_isolated",
        [("a", "My name is Bob."), ("b", f"What's my name? {FORGOT}")],
        [excludes("Bob", at=-1)],
    ),
    Case(
        "survives_interleaving",
        [("a", "My name is Bob."), ("b", "My name is Carol."), ("a", "What's my name?"), ("b", "What's my name?")],
        [contains("Bob", at=2), contains("Carol", at=3)],
    ),
    Case(
        "long_thread",
        [("a", "My name is Bob."), *FILLER, ("a", "What's my name?")],
        [contains("Bob", at=-1)],
    ),
]
