from langchain.agents import create_agent

from lessons import MODEL
from lessons.l02_structured_output.problem import SYSTEM_PROMPT, ContactInfo

agent = create_agent(model=f"openai:{MODEL}", tools=[], system_prompt=SYSTEM_PROMPT, response_format=ContactInfo)


def run(text: str) -> dict:
    return agent.invoke({"messages": [{"role": "user", "content": text}]})["structured_response"]
