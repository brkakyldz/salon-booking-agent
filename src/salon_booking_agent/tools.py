"""The five tools the assistant can call.

Each tool is a thin wrapper over `db.py`. Business-rule failures (a taken slot, a
closed day) are *returned* as text that starts with NOT BOOKED / NOT CANCELLED: an
exception raised inside a tool would abort the whole agent run, while a returned
refusal reaches the model, which then explains it to the customer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time

from langchain.tools import ToolRuntime, tool

from . import db


@dataclass
class SalonContext:
    """Runtime settings for one run. Every field has a default because the dev
    server (and agent-chat-ui) invoke the graph without any context."""

    salon_name: str = "Salon Lara"
    today: str | None = None  # 'YYYY-MM-DD' to pin the date (tests, demos); None = real clock


def get_settings(runtime) -> SalonContext:
    ctx = getattr(runtime, "context", None)
    return ctx if isinstance(ctx, SalonContext) else SalonContext()


def current_time(settings: SalonContext) -> datetime:
    if settings.today:
        return datetime.combine(date.fromisoformat(settings.today), time(8, 0))
    return datetime.now().replace(second=0, microsecond=0)


def _format_appointment(a: db.Appointment) -> str:
    start = datetime.fromisoformat(a.start_time)
    return f"#{a.id}: {a.service} for {a.customer_name}, {start:%A %Y-%m-%d %H:%M}–{a.end_time[11:]}"


@tool
def list_services() -> str:
    """List the salon's services with their duration and price in Turkish lira."""
    with db.connect() as conn:
        services = db.list_services(conn)
    return "\n".join(f"- {s.name}: {s.duration_minutes} min, {s.price_try} TRY" for s in services)


@tool
def check_availability(date: str, service: str, runtime: ToolRuntime[SalonContext]) -> str:
    """Return the free start times for a service on a date.

    Args:
        date: the day, formatted YYYY-MM-DD.
        service: the service name, e.g. "Women's haircut".
    """
    try:
        day = datetime.fromisoformat(date.strip()).date()
    except ValueError:
        return f"Could not read the date '{date}'. Use YYYY-MM-DD."
    now = current_time(get_settings(runtime))
    if day < now.date():
        return f"{day:%A %Y-%m-%d} is in the past."
    hours = db.opening_hours_for(day)
    if hours is None:
        return f"The salon is closed on {day:%A}s ({day.isoformat()})."
    try:
        with db.connect() as conn:
            service_obj = db.get_service(conn, service)
            slots = db.available_slots(conn, day, service_obj.name, now=now)
    except db.BookingError as exc:
        return str(exc)
    header = (
        f"{service_obj.name} ({service_obj.duration_minutes} min) on {day:%A %Y-%m-%d}, "
        f"open {hours[0]:%H:%M}–{hours[1]:%H:%M}."
    )
    if not slots:
        return f"{header} No free start times left that day."
    return f"{header} Free start times: {', '.join(slots)}."


@tool
def find_appointments(customer_name: str, runtime: ToolRuntime[SalonContext]) -> str:
    """Find a customer's upcoming (not cancelled) appointments by their full name."""
    with db.connect() as conn:
        found = db.find_appointments(conn, customer_name, now=current_time(get_settings(runtime)))
    if not found:
        return f"No booked appointments found for '{customer_name}'."
    return "\n".join(_format_appointment(a) for a in found)


@tool
def book_appointment(
    customer_name: str, service: str, start_time: str, runtime: ToolRuntime[SalonContext]
) -> str:
    """Book an appointment. The salon owner must approve it before it is saved.

    Args:
        customer_name: the customer's full name.
        service: the service name, e.g. "Men's haircut".
        start_time: local start time, formatted YYYY-MM-DD HH:MM (on the hour or half hour).
    """
    now = current_time(get_settings(runtime))
    try:
        with db.connect() as conn:
            appointment = db.book_appointment(conn, customer_name, service, start_time, now=now)
    except db.BookingError as exc:
        return f"NOT BOOKED: {exc}"
    return f"BOOKED {_format_appointment(appointment)}."


@tool
def cancel_appointment(appointment_id: int, runtime: ToolRuntime[SalonContext]) -> str:
    """Cancel an appointment by its number (from find_appointments). The salon owner
    must approve it before it happens."""
    try:
        with db.connect() as conn:
            appointment = db.cancel_appointment(
                conn, int(appointment_id), now=current_time(get_settings(runtime))
            )
    except (db.BookingError, ValueError) as exc:
        return f"NOT CANCELLED: {exc}"
    return f"CANCELLED {_format_appointment(appointment)}."


TOOLS = [list_services, check_availability, find_appointments, book_appointment, cancel_appointment]
