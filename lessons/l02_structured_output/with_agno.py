from agno.agent import Agent

from lessons import MODEL
from lessons.l02_structured_output.problem import CONTACT_SCHEMA, SYSTEM_PROMPT

output_schema = {"type": "json_schema", "name": "ContactInfo", "strict": True, "schema": CONTACT_SCHEMA}
agent = Agent(model=f"openai:{MODEL}", instructions=SYSTEM_PROMPT, output_schema=output_schema, telemetry=False)


def run(text: str) -> dict:
    return agent.run(text).content
