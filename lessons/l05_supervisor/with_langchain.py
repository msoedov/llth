from langchain.agents import create_agent
from langchain.tools import tool

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
calendar_agent = create_agent(model, tools=[create_calendar_event, get_available_time_slots], system_prompt=CALENDAR_PROMPT)
email_agent = create_agent(model, tools=[send_email], system_prompt=EMAIL_PROMPT)


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language. Include attendee emails and times."""
    return calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language. Include recipients and the message."""
    return email_agent.invoke({"messages": [{"role": "user", "content": request}]})["messages"][-1].text


supervisor = create_agent(model, tools=[schedule_event, manage_email], system_prompt=SUPERVISOR_PROMPT)


def run(request: str) -> str:
    return supervisor.invoke({"messages": [{"role": "user", "content": request}]})["messages"][-1].text
