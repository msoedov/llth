"""
Supervisor with subagents.
Source: https://docs.langchain.com/oss/python/langchain/supervisor

A personal assistant splits one request between a calendar specialist and an
email specialist. The supervisor never touches the low-level tools: only the
specialist that owns a tool may call it.
"""

from datetime import date

from evals.capture import recorded
from evals.checks import called, not_called
from evals.harness import Case

CALENDAR_PROMPT = (
    f"Today's date is {date.today().isoformat()}. "
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests into ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)
EMAIL_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests and delegate each part. "
    "Delegate only what the user asked for."
)


@recorded
def create_calendar_event(title: str, start_time: str, end_time: str, attendees: list[str], location: str = "") -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@recorded
def get_available_time_slots(attendees: list[str], date: str, duration_minutes: int) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


@recorded
def send_email(to: list[str], subject: str, body: str) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


CASES = [
    Case(
        "both_specialists",
        "Schedule a meeting with design@acme.dev next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups.",
        [called("create_calendar_event", times=1), called("send_email", times=1)],
    ),
    Case(
        "calendar_only",
        "Book a 30 minute sync with ops@acme.dev next Friday at 9am.",
        [called("create_calendar_event", times=1), not_called("send_email")],
    ),
    Case(
        "email_only",
        "Email finance@acme.dev that the Q3 report is ready.",
        [called("send_email", times=1), not_called("create_calendar_event")],
    ),
]
