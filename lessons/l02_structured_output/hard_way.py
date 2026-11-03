import json
import os
from dataclasses import dataclass
from types import NoneType, UnionType
from typing import get_args, get_type_hints

import httpx

URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_PROMPT = "Extract contact info from the text. Use null for anything the text does not state."
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", NoneType: "null"}


@dataclass
class ContactInfo:
    name: str
    email: str | None
    phone: str | None


def json_type(annotation: type) -> str | list[str]:
    if isinstance(annotation, UnionType):
        return [JSON_TYPES[arg] for arg in get_args(annotation)]
    return JSON_TYPES[annotation]


def response_format(cls: type) -> dict:
    """Strict mode: every key required, nullable fields spelled as ["string", "null"]."""
    hints = get_type_hints(cls)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": cls.__name__,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: {"type": json_type(annotation)} for key, annotation in hints.items()},
                "required": list(hints),
                "additionalProperties": False,
            },
        },
    }


RESPONSE_FORMAT = response_format(ContactInfo)


def run(text: str) -> ContactInfo:
    response = httpx.post(
        URL,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={
            "model": "openai/gpt-5.4-mini",
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}],
            "response_format": RESPONSE_FORMAT,
        },
        timeout=60,
    )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    if message.get("refusal"):
        raise ValueError(f"model refused: {message['refusal']}")
    return ContactInfo(**json.loads(message["content"]))
