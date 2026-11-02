import inspect
import json
import os
from collections.abc import Callable

import httpx

from lessons import MODEL
from lessons.l01_tool_agent.problem import SYSTEM_PROMPT, get_weather

URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_STEPS = 5
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


def tool_schema(fn: Callable) -> dict:
    params = inspect.signature(fn).parameters.values()
    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": inspect.getdoc(fn),
            "parameters": {
                "type": "object",
                "properties": {p.name: {"type": JSON_TYPES[p.annotation]} for p in params},
                "required": [p.name for p in params if p.default is p.empty],
            },
        },
    }


class Agent:
    def __init__(self, tools: list[Callable], prompt: str):
        self.tools = {fn.__name__: fn for fn in tools}
        self.schemas = [tool_schema(fn) for fn in tools]
        self.prompt = prompt

    def _complete(self, messages: list[dict]) -> dict:
        response = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json={"model": f"openai/{MODEL}", "messages": messages, "tools": self.schemas},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    def _execute(self, call: dict) -> dict:
        fn = self.tools[call["function"]["name"]]
        args = json.loads(call["function"]["arguments"])
        return {"role": "tool", "tool_call_id": call["id"], "content": str(fn(**args))}

    def run(self, question: str) -> str:
        messages = [{"role": "system", "content": self.prompt}, {"role": "user", "content": question}]
        for _ in range(MAX_STEPS):
            message = self._complete(messages)
            if not message.get("tool_calls"):
                return message["content"]
            messages.append(message)
            messages += [self._execute(call) for call in message["tool_calls"]]
        raise RuntimeError(f"no final answer after {MAX_STEPS} steps")


agent = Agent([get_weather], SYSTEM_PROMPT)


def run(question: str) -> str:
    return agent.run(question)
