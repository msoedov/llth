from agno.agent import Agent
from agno.team import Team

from lessons import MODEL
from lessons.l05_supervisor.problem import (
    CALENDAR_PROMPT,
    EMAIL_PROMPT,
    SUPERVISOR_PROMPT,
    create_calendar_event,
    get_available_time_slots,
    send_email,
)

model = f"openai:{MODEL}"
calendar_agent = Agent(
    name="Calendar",
    role="Schedules calendar events",
    model=model,
    tools=[create_calendar_event, get_available_time_slots],
    instructions=CALENDAR_PROMPT,
    telemetry=False,
)
email_agent = Agent(
    name="Email",
    role="Sends emails",
    model=model,
    tools=[send_email],
    instructions=EMAIL_PROMPT,
    telemetry=False,
)
team = Team(members=[calendar_agent, email_agent], model=model, instructions=SUPERVISOR_PROMPT, telemetry=False)


def run(request: str) -> str:
    return team.run(request).content
