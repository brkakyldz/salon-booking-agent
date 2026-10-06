"""SQLite data layer for the salon: services, opening hours and appointments.

Every rule that protects the calendar lives here, not in the prompt: a booking is
checked against opening hours, the 30-minute slot grid and the past, and then against
every existing appointment inside one write transaction, so neither the model nor an
approving owner can create a double booking.
"""

from __future__ import annotations

import os
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

SLOT_MINUTES = 30

# Weekday (Monday=0) -> (open, close). Sunday is closed.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(9, 0), time(19, 0)),
    1: (time(9, 0), time(19, 0)),
    2: (time(9, 0), time(19, 0)),
    3: (time(9, 0), time(19, 0)),
    4: (time(9, 0), time(19, 0)),
    5: (time(10, 0), time(17, 0)),
}

SERVICES: list[tuple[str, int, int]] = [
    # (name, duration in minutes, price in TRY)
    ("Women's haircut", 60, 750),
    ("Men's haircut", 30, 400),
    ("Blow-dry", 30, 350),
    ("Beard trim", 30, 250),
    ("Hair colouring", 120, 2200),
    ("Highlights", 90, 1800),
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS services (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    duration_minutes INTEGER NOT NULL CHECK (duration_minutes % 30 = 0),
    price_try INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    service_id INTEGER NOT NULL REFERENCES services(id),
    start_time TEXT NOT NULL,   -- local salon time, ISO 'YYYY-MM-DDTHH:MM'
    end_time TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'booked' CHECK (status IN ('booked', 'cancelled')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_appointments_start ON appointments(start_time);
"""

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "salon.db"


class BookingError(ValueError):
    """A booking or cancellation that the salon's rules do not allow."""


@dataclass(frozen=True)
class Service:
    id: int
    name: str
    duration_minutes: int
    price_try: int


@dataclass(frozen=True)
class Appointment:
    id: int
    customer_name: str
    service: str
    start_time: str
    end_time: str
    status: str


def db_path() -> Path:
    return Path(os.environ.get("SALON_DB_PATH", DEFAULT_DB_PATH))


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open an autocommit connection that always closes; callers open their own transactions."""
    target = Path(path) if path else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, isolation_level=None)  # explicit transactions
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


# --- reads -----------------------------------------------------------------


def list_services(conn: sqlite3.Connection) -> list[Service]:
    rows = conn.execute(
        "SELECT id, name, duration_minutes, price_try FROM services ORDER BY id"
    ).fetchall()
    return [Service(**dict(r)) for r in rows]


def get_service(conn: sqlite3.Connection, name: str) -> Service:
    """Find a service by name, case-insensitively; tolerate a partial name."""
    wanted = name.strip().lower()
    services = list_services(conn)
    exact = [s for s in services if s.name.lower() == wanted]
    if exact:
        return exact[0]
    partial = [s for s in services if wanted and wanted in s.name.lower()]
    if len(partial) == 1:
        return partial[0]
    known = ", ".join(s.name for s in services)
    raise BookingError(f"Unknown service '{name}'. Known services: {known}.")


def opening_hours_for(day: date) -> tuple[time, time] | None:
    return OPENING_HOURS.get(day.weekday())


def _booked_intervals(conn: sqlite3.Connection, day: date) -> list[tuple[datetime, datetime]]:
    rows = conn.execute(
        "SELECT start_time, end_time FROM appointments "
        "WHERE status = 'booked' AND substr(start_time, 1, 10) = ?",
        (day.isoformat(),),
    ).fetchall()
    return [(datetime.fromisoformat(r["start_time"]), datetime.fromisoformat(r["end_time"])) for r in rows]


def available_slots(
    conn: sqlite3.Connection, day: date, service_name: str, now: datetime | None = None
) -> list[str]:
    """Start times ('HH:MM') on `day` where `service_name` fits without overlap."""
    service = get_service(conn, service_name)
    hours = opening_hours_for(day)
    if hours is None:
        return []
    now = now or datetime.now()
    opens, closes = (datetime.combine(day, t) for t in hours)
    length = timedelta(minutes=service.duration_minutes)
    booked = _booked_intervals(conn, day)
    slots: list[str] = []
    start = opens
    while start + length <= closes:
        end = start + length
        free = all(end <= b_start or start >= b_end for b_start, b_end in booked)
        if free and start > now:
            slots.append(start.strftime("%H:%M"))
        start += timedelta(minutes=SLOT_MINUTES)
    return slots


def normalize_name(name: str) -> str:
    """Fold case and Turkish letters so 'AYŞE YILMAZ', 'ayşe yılmaz' and 'Ayse Yilmaz'
    match. casefold() alone maps 'I' to 'i' (not 'ı') and 'İ' to 'i' + a combining dot."""
    decomposed = unicodedata.normalize("NFKD", name.replace("ı", "i").replace("I", "i"))
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(plain.casefold().split())


def _same_name(a: str, b: str) -> bool:
    return normalize_name(a) == normalize_name(b)


def find_appointments(
    conn: sqlite3.Connection, customer_name: str, now: datetime | None = None
) -> list[Appointment]:
    """A customer's upcoming bookings (from `now` on)."""
    now = now or datetime.now()
    rows = conn.execute(
        "SELECT a.id, a.customer_name, s.name AS service, a.start_time, a.end_time, a.status "
        "FROM appointments a JOIN services s ON s.id = a.service_id "
        "WHERE a.status = 'booked' AND a.start_time >= ? ORDER BY a.start_time",
        (now.isoformat(timespec="minutes"),),
    ).fetchall()
    return [Appointment(**dict(r)) for r in rows if _same_name(r["customer_name"], customer_name)]


def get_appointment(conn: sqlite3.Connection, appointment_id: int) -> Appointment | None:
    row = conn.execute(
        "SELECT a.id, a.customer_name, s.name AS service, a.start_time, a.end_time, a.status "
        "FROM appointments a JOIN services s ON s.id = a.service_id WHERE a.id = ?",
        (appointment_id,),
    ).fetchone()
    return Appointment(**dict(row)) if row else None


# --- writes ----------------------------------------------------------------


def parse_start(value: str) -> datetime:
    """Accept 'YYYY-MM-DD HH:MM' or 'YYYY-MM-DDTHH:MM' (seconds tolerated)."""
    try:
        parsed = datetime.fromisoformat(value.strip().replace(" ", "T"))
    except ValueError as exc:
        raise BookingError(
            f"Could not read start time '{value}'. Use the format YYYY-MM-DD HH:MM."
        ) from exc
    return parsed.replace(second=0, microsecond=0, tzinfo=None)


def book_appointment(
    conn: sqlite3.Connection,
    customer_name: str,
    service_name: str,
    start_time: str,
    now: datetime | None = None,
) -> Appointment:
    """Create a booking, or raise BookingError explaining why the salon refuses it."""
    name = customer_name.strip()
    if not name:
        raise BookingError("A customer name is required to book.")
    service = get_service(conn, service_name)
    start = parse_start(start_time)
    try:
        end = start + timedelta(minutes=service.duration_minutes)
    except OverflowError as exc:
        raise BookingError(f"{start_time} is not a usable date.") from exc
    now = now or datetime.now()

    if start <= now:
        raise BookingError(f"{start:%Y-%m-%d %H:%M} is in the past.")
    if start.minute % SLOT_MINUTES:
        raise BookingError("Appointments start on the hour or half hour (e.g. 10:00, 10:30).")
    hours = opening_hours_for(start.date())
    if hours is None:
        raise BookingError(f"The salon is closed on {start:%A}s.")
    opens, closes = (datetime.combine(start.date(), t) for t in hours)
    if start < opens or end > closes:
        raise BookingError(
            f"{service.name} takes {service.duration_minutes} minutes and must fit between "
            f"{opens:%H:%M} and {closes:%H:%M} on {start:%A}."
        )

    # BEGIN IMMEDIATE takes the write lock before the overlap check, so two
    # concurrent bookings cannot both pass it.
    conn.execute("BEGIN IMMEDIATE")
    try:
        clash = conn.execute(
            "SELECT a.id, a.start_time, a.end_time FROM appointments a "
            "WHERE a.status = 'booked' AND a.start_time < ? AND a.end_time > ? LIMIT 1",
            (end.isoformat(timespec="minutes"), start.isoformat(timespec="minutes")),
        ).fetchone()
        if clash:
            raise BookingError(
                f"{start:%Y-%m-%d %H:%M}–{end:%H:%M} overlaps an existing appointment "
                f"({clash['start_time'][11:]}–{clash['end_time'][11:]})."
            )
        cur = conn.execute(
            "INSERT INTO appointments (customer_name, service_id, start_time, end_time) "
            "VALUES (?, ?, ?, ?)",
            (name, service.id, start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes")),
        )
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    created = get_appointment(conn, cur.lastrowid)
    assert created is not None
    return created


def cancel_appointment(
    conn: sqlite3.Connection, appointment_id: int, now: datetime | None = None
) -> Appointment:
    try:
        appointment = get_appointment(conn, appointment_id)
    except OverflowError:  # an id beyond SQLite's INTEGER range
        appointment = None
    if appointment is None:
        raise BookingError(f"There is no appointment #{appointment_id}.")
    if appointment.status != "booked":
        raise BookingError(f"Appointment #{appointment_id} is already cancelled.")
    if datetime.fromisoformat(appointment.start_time) <= (now or datetime.now()):
        raise BookingError(f"Appointment #{appointment_id} has already started or taken place.")
    cur = conn.execute(
        "UPDATE appointments SET status = 'cancelled' WHERE id = ? AND status = 'booked'", (appointment_id,)
    )
    if cur.rowcount != 1:  # cancelled by a concurrent call between the check and the update
        raise BookingError(f"Appointment #{appointment_id} is already cancelled.")
    cancelled = get_appointment(conn, appointment_id)
    assert cancelled is not None
    return cancelled


def count_appointments(conn: sqlite3.Connection, status: str = "booked") -> int:
    return conn.execute("SELECT count(*) FROM appointments WHERE status = ?", (status,)).fetchone()[0]
