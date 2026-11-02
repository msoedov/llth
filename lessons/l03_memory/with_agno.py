from uuid import uuid4

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb

from lessons import MODEL
from lessons.l03_memory.problem import SYSTEM_PROMPT

agent = Agent(
    model=f"openai:{MODEL}",
    instructions=SYSTEM_PROMPT,
    db=InMemoryDb(),
    add_history_to_context=True,
    num_history_runs=20,
    telemetry=False,
)


def run(turns: list[tuple[str, str]]) -> list[str]:
    conversation = uuid4().hex
    return [agent.run(message, session_id=f"{conversation}-{thread}").content for thread, message in turns]
