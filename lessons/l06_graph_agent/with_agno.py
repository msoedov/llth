from agno.agent import Agent

from lessons import MODEL
from lessons.l06_graph_agent.problem import SYSTEM_PROMPT, add, divide, multiply

agent = Agent(model=f"openai:{MODEL}", tools=[add, multiply, divide], instructions=SYSTEM_PROMPT, telemetry=False)


def run(question: str) -> str:
    return agent.run(question).content
