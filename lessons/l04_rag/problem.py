"""
RAG agent.
Source: https://docs.langchain.com/oss/python/langchain/rag

Answer questions from a private document set: split, embed, search, and answer
from what was retrieved. The docs tutorial loads a public website through a deep
agent. This lesson keeps the retrieval core and uses an invented handbook, so a
correct answer can only come from retrieval.
"""

from evals.checks import contains, excludes, used_embeddings
from evals.harness import Case

SYSTEM_PROMPT = (
    "Answer questions about the Halvorsen Lab handbook. Always search the handbook first. "
    "Answer only from what the search returns. If it does not contain the answer, reply exactly: I don't know."
)

DOCS = {
    "equipment": (
        "The Halvorsen Lab keeps its calibration rig, nicknamed Old Marigold, in room B-117. "
        "Booking Old Marigold requires a green badge. Sessions on the rig last at most 90 minutes. "
        "The rig is recalibrated every second Thursday by the duty technician."
    ),
    "travel": (
        "Conference travel at Halvorsen Lab is approved by Ingrid Tveit. "
        "The daily meal allowance is 41 euros. Requests must be filed 18 days before departure."
    ),
    "safety": (
        "Spills of cryogenic fluid are reported on extension 4471. "
        "The muster point for a fire alarm is the east bicycle shed. "
        "Fire drills take place on the first Monday of each quarter."
    ),
    "onboarding": (
        "New staff receive a blue badge on day one. A green badge is granted after passing the Tier 2 safety course, "
        "which is offered every six weeks. Every new hire is paired with a mentor for 120 days."
    ),
}

CASES = [
    Case("single_fact", "How much is the daily meal allowance?", [contains("41"), used_embeddings()]),
    Case("exact_detail", "Where is Old Marigold kept?", [contains("B-117"), used_embeddings()]),
    Case("multi_document", "What must a new hire do before they can book Old Marigold?", [contains("Tier 2"), contains("green")]),
    Case("not_in_documents", "What is the parking fee at Halvorsen Lab?", [contains("don't know"), excludes("euro")]),
]
