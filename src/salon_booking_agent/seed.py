"""Reset the salon database: services plus a few booked appointments.

Bookings are placed on the next three open days after `today`, so the demo works on
any date. Names are fictional.

    uv run salon-seed                      # relative to today
    uv run salon-seed --today 2026-10-05   # relative to a fixed date (tests use this)
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, time, timedelta
from pathlib import Path

from . import db

# (open-day index, customer, service, start 'HH:MM'). Times fit every open day,
# including Saturday's shorter 10:00-17:00.
SEED_BOOKINGS: list[tuple[int, str, str, str]] = [
    (0, "Ayşe Yılmaz", "Women's haircut", "10:00"),
    (0, "Mehmet Demir", "Men's haircut", "11:30"),
    (0, "Zeynep Kaya", "Hair colouring", "14:00"),
    (1, "Elif Şahin", "Highlights", "10:00"),
    (1, "Can Öztürk", "Beard trim", "13:00"),
    (2, "Ayşe Yılmaz", "Blow-dry", "16:00"),
]


def next_open_days(today: date, count: int) -> list[date]:
    """The first `count` days after `today` on which the salon is open."""
    days: list[date] = []
    day = today
    while len(days) < count:
        day += timedelta(days=1)
        if db.opening_hours_for(day) is not None:
            days.append(day)
    return days


def seed(conn: sqlite3.Connection, today: date) -> list[db.Appointment]:
    conn.executescript("DROP TABLE IF EXISTS appointments; DROP TABLE IF EXISTS services;")
    db.init_db(conn)
    conn.executemany(
        "INSERT INTO services (name, duration_minutes, price_try) VALUES (?, ?, ?)", db.SERVICES
    )
    days = next_open_days(today, 3)
    booked = []
    for day_index, customer, service, start in SEED_BOOKINGS:
        booked.append(
            db.book_appointment(
                conn, customer, service, f"{days[day_index].isoformat()} {start}",
                now=datetime.combine(today, time(0, 0)),
            )
        )
    return booked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--db", type=Path, default=None, help="database file (default: data/salon.db)")
    args = parser.parse_args()
    with db.connect(args.db) as conn:
        booked = seed(conn, args.today)
    print(f"Seeded {len(db.SERVICES)} services and {len(booked)} appointments into {args.db or db.db_path()}")
    for a in booked:
        print(f"  #{a.id:<2} {a.start_time.replace('T', ' ')}–{a.end_time[11:]}  {a.service:<16} {a.customer_name}")


if __name__ == "__main__":
    main()
