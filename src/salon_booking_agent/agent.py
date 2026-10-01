"""The appointment assistant: `create_agent` + tools + human-in-the-loop approval."""

from __future__ import annotations

from datetime import timedelta

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    ModelRequest,
    dynamic_prompt,
)

from . import db
from .tools import TOOLS, SalonContext, current_time, get_settings

SYSTEM_PROMPT = """\
You are the booking assistant of {salon_name}, a small hair salon in Antalya, Turkey.
You chat with customers about services, free times and their appointments.

Today is {today:%A %Y-%m-%d}. The next days are: {upcoming}.
Opening hours: Monday–Friday 09:00–19:00, Saturday 10:00–17:00, closed on Sunday.
Appointments start on the hour or half hour.

Rules:
- Use the tools for every fact: services and prices (list_services), free times
  (check_availability), existing bookings (find_appointments). Never invent a time.
- To book you need the customer's full name, the service, the date and the time.
  Check availability first, then call book_appointment once.
- To cancel, look the booking up with find_appointments and cancel it by its number.
- Booking and cancelling are reviewed by the salon owner before anything is saved.
  Trust only the tool result:
  - "BOOKED ..." / "CANCELLED ..." means it is done; confirm the details in that result
    (the owner may have changed the time — use the time in the result).
  - "NOT BOOKED ..." / "NOT CANCELLED ..." means it did not happen; say why.
  - "User rejected the tool call ..." means the owner declined. Tell the customer it was
    not booked (or not cancelled) and pass on the owner's reason. Do not try again unless
    the customer asks for something different.
- Reply in the customer's language (English or Turkish), briefly and warmly.
"""


@dynamic_prompt
def salon_prompt(request: ModelRequest) -> str:
    settings = get_settings(request.runtime)
    today = current_time(settings).date()
    upcoming = ", ".join(f"{d:%a %Y-%m-%d}" for d in (today + timedelta(days=i) for i in range(1, 8)))
    return SYSTEM_PROMPT.format(salon_name=settings.salon_name, today=today, upcoming=upcoming)


# The two description builders run inside the middleware on the server's event loop,
# where `langgraph dev` rejects blocking I/O. So they read no database: service
# details come from the static catalogue, appointment details from the conversation.


def describe_booking(tool_call, state, runtime) -> str:
    """Markdown shown to the owner on the approval card."""
    args = tool_call["args"]
    lines = [
        "**New booking request**",
        f"- Customer: {args.get('customer_name')}",
        f"- Service: {args.get('service')}",
        f"- Start: {args.get('start_time')}",
    ]
    wanted = str(args.get("service", "")).strip().lower()
    catalogue = {name.lower(): (minutes, price) for name, minutes, price in db.SERVICES}
    try:
        start = db.parse_start(str(args.get("start_time", "")))
        if wanted in catalogue:
            minutes, price = catalogue[wanted]
            end = start + timedelta(minutes=minutes)
            lines.append(f"- Ends: {end:%H:%M} ({minutes} min, {price} TRY)")
    except (db.BookingError, OverflowError):
        pass  # the tool itself reports bad input after approval
    lines.append("\nApprove, edit (e.g. change `start_time`) or reject with a reason for the customer.")
    return "\n".join(lines)


def describe_cancellation(tool_call, state, runtime) -> str:
    appointment_id = tool_call["args"].get("appointment_id")
    detail = f"#{appointment_id}"
    # find_appointments lists bookings as "#<id>: <service> for <name>, <when>".
    for message in reversed(state.get("messages", [])):
        if getattr(message, "type", "") != "tool":
            continue
        line = next((row for row in str(message.content).splitlines() if row.startswith(f"#{appointment_id}:")), None)
        if line:
            detail = line
            break
    return f"**Cancellation request**\n- Appointment {detail}\n\nApprove or reject with a reason."


INTERRUPT_ON: dict[str, InterruptOnConfig] = {
    # Explicit lists, never `True`: `True` also allows "respond", which agent-chat-ui cannot render.
    "book_appointment": {"allowed_decisions": ["approve", "edit", "reject"], "description": describe_booking},
    "cancel_appointment": {"allowed_decisions": ["approve", "reject"], "description": describe_cancellation},
}

EDIT_NOTICE = (
    "Note: the salon owner changed this request before it ran. Tell the customer the "
    "details in the tool response below, not the ones you proposed."
)


def build_agent(model, checkpointer=None):
    """Build the agent. Scripts and tests pass a checkpointer; under `langgraph dev`
    the server provides persistence, so none is passed there."""
    return create_agent(
        model,
        tools=TOOLS,
        middleware=[
            salon_prompt,
            HumanInTheLoopMiddleware(interrupt_on=INTERRUPT_ON, edit_notice=EDIT_NOTICE),
        ],
        context_schema=SalonContext,
        checkpointer=checkpointer,
        name="appointment_assistant",
    )
