from agno.agent import Agent

from lessons import MODEL
from lessons.l01_tool_agent.problem import SYSTEM_PROMPT, get_weather

agent = Agent(model=f"openai:{MODEL}", tools=[get_weather], instructions=SYSTEM_PROMPT, telemetry=False)


def run(question: str) -> str:
    return agent.run(question).content
