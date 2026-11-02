"""
Structured output.
Source: https://docs.langchain.com/oss/python/langchain/structured-output

Turn free text into a typed record the program can trust. Fields missing from
the text must come back as null, never invented.
"""

from typing import TypedDict

from evals.checks import field_is, max_llm_calls, not_called
from evals.harness import Case

SYSTEM_PROMPT = "Extract contact info from the text. Use null for anything the text does not state."


class ContactInfo(TypedDict):
    """Contact information for a person."""

    name: str
    email: str | None
    phone: str | None


CONTACT_SCHEMA = {
    "title": "ContactInfo",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": ["string", "null"]},
        "phone": {"type": ["string", "null"]},
    },
    "required": ["name", "email", "phone"],
    "additionalProperties": False,
}

CASES = [
    Case(
        "complete",
        "John Doe, john@example.com, (555) 123-4567",
        [field_is("name", "John Doe"), field_is("email", "john@example.com"), field_is("phone", "(555) 123-4567"), max_llm_calls(1), not_called()],
    ),
    Case(
        "buried_in_prose",
        "Thanks for calling! If the invoice is late, ask for Priya Nair (priya.nair@acme.dev), she handles billing.",
        [field_is("name", "Priya Nair"), field_is("email", "priya.nair@acme.dev"), field_is("phone", None)],
    ),
    Case("missing_fields", "My name is Ana Lee.", [field_is("name", "Ana Lee"), field_is("email", None), field_is("phone", None)]),
]
