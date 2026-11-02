from uuid import uuid4

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.tools import tool

from lessons import MODEL
from lessons.l07_human_in_the_loop.problem import SYSTEM_PROMPT, send_email

guarded_send_email = tool(requires_confirmation=True)(send_email)
agent = Agent(model=f"openai:{MODEL}", tools=[guarded_send_email], instructions=SYSTEM_PROMPT, db=InMemoryDb(), telemetry=False)


def run(case: dict) -> dict:
    response = agent.run(case["request"], session_id=uuid4().hex)
    paused = response.is_paused
    for requirement in response.active_requirements:
        if case["approve"]:
            requirement.confirm()
        else:
            requirement.reject()
    final = agent.continue_run(run_response=response, requirements=response.requirements)
    return {"paused": paused, "result": final.content}
