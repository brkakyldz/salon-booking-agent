"""The calendar rules, tested without a model: the tools lean on these."""

from datetime import date, datetime

import pytest

from salon_booking_agent import db
from salon_booking_agent.seed import next_open_days, seed

TODAY = date(2026, 10, 5)  # a Monday
NOW = datetime(2026, 10, 5, 8, 0)
TUE, WED, THU = date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8)


@pytest.fixture
def conn(tmp_path):
    with db.connect(tmp_path / "salon.db") as conn:
        seed(conn, TODAY)
        yield conn


def test_seed_uses_next_open_days():
    assert next_open_days(TODAY, 3) == [TUE, WED, THU]
    # Saturday -> skips Sunday (closed) and lands on Monday.
    assert next_open_days(date(2026, 10, 10), 1) == [date(2026, 10, 12)]


@pytest.mark.parametrize("offset", range(7))
def test_seed_works_whatever_weekday_today_is(tmp_path, offset):
    today = date(2026, 10, 5 + offset)
    with db.connect(tmp_path / "any.db") as c:
        assert len(seed(c, today)) == 6


def test_availability_matches_seed(conn):
    # Tuesday has 10:00-11:00, 11:30-12:00 and 14:00-16:00 booked.
    assert db.available_slots(conn, TUE, "Women's haircut", now=NOW) == [
        "09:00", "12:00", "12:30", "13:00", "16:00", "16:30", "17:00", "17:30", "18:00",
    ]
    # A 30-minute service also fits the half-hour gaps.
    assert "11:00" in db.available_slots(conn, TUE, "Men's haircut", now=NOW)
    assert "11:30" not in db.available_slots(conn, TUE, "Men's haircut", now=NOW)


def test_closed_day_and_short_saturday(conn):
    assert db.available_slots(conn, date(2026, 10, 11), "Blow-dry", now=NOW) == []  # Sunday
    saturday = db.available_slots(conn, date(2026, 10, 10), "Blow-dry", now=NOW)
    assert saturday[0] == "10:00" and saturday[-1] == "16:30"


def test_past_slots_are_hidden(conn):
    assert db.available_slots(conn, TUE, "Blow-dry", now=datetime(2026, 10, 6, 17, 0)) == [
        "17:30", "18:00", "18:30",
    ]


def test_booking_creates_row_and_removes_slot(conn):
    before = db.count_appointments(conn)
    appt = db.book_appointment(conn, "Deniz Aydın", "Women's haircut", "2026-10-06 12:00", now=NOW)
    assert (appt.start_time, appt.end_time, appt.status) == ("2026-10-06T12:00", "2026-10-06T13:00", "booked")
    assert db.count_appointments(conn) == before + 1
    assert "12:00" not in db.available_slots(conn, TUE, "Women's haircut", now=NOW)


@pytest.mark.parametrize(
    "start, message",
    [
        ("2026-10-06 10:30", "overlaps"),          # clashes with 10:00-11:00
        ("2026-10-06 13:30", "overlaps"),          # 13:30-14:30 runs into 14:00
        ("2026-10-06 18:30", "must fit between"),  # would end 19:30
        ("2026-10-11 10:00", "closed"),            # Sunday
        ("2026-10-06 12:15", "half hour"),
        ("2026-10-05 07:00", "in the past"),
        ("next tuesday", "Could not read"),
    ],
)
def test_booking_rules_are_enforced_in_code(conn, start, message):
    before = db.count_appointments(conn)
    with pytest.raises(db.BookingError, match=message):
        db.book_appointment(conn, "Deniz Aydın", "Women's haircut", start, now=NOW)
    assert db.count_appointments(conn) == before


def test_cancel_frees_the_slot(conn):
    [haircut] = [a for a in db.find_appointments(conn, "ayşe yılmaz", now=NOW) if a.service == "Women's haircut"]
    db.cancel_appointment(conn, haircut.id, now=NOW)
    assert db.get_appointment(conn, haircut.id).status == "cancelled"
    assert "10:00" in db.available_slots(conn, TUE, "Women's haircut", now=NOW)
    with pytest.raises(db.BookingError, match="already cancelled"):
        db.cancel_appointment(conn, haircut.id, now=NOW)


def test_find_and_service_lookup(conn):
    assert [a.service for a in db.find_appointments(conn, "Ayşe Yılmaz", now=NOW)] == ["Women's haircut", "Blow-dry"]
    assert db.find_appointments(conn, "Nobody", now=NOW) == []
    assert db.get_service(conn, "colouring").name == "Hair colouring"
    with pytest.raises(db.BookingError, match="Unknown service"):
        db.get_service(conn, "manicure")


@pytest.mark.parametrize("typed", ["AYŞE YILMAZ", "ayşe yılmaz", "Ayse Yilmaz", "  ayşe   YILMAZ "])
def test_name_matching_folds_turkish_letters(conn, typed):
    assert len(db.find_appointments(conn, typed, now=NOW)) == 2
    assert len(db.find_appointments(conn, "ELİF ŞAHİN", now=NOW)) == 1


def test_past_appointments_are_not_listed_or_cancellable(conn):
    later = datetime(2026, 10, 7, 12, 0)  # after Tuesday's bookings, before Thursday's
    assert [a.service for a in db.find_appointments(conn, "Ayşe Yılmaz", now=later)] == ["Blow-dry"]
    [tuesday] = [a for a in db.find_appointments(conn, "Ayşe Yılmaz", now=NOW) if a.start_time.startswith("2026-10-06")]
    with pytest.raises(db.BookingError, match="already started or taken place"):
        db.cancel_appointment(conn, tuesday.id, now=later)


def test_absurd_inputs_are_refused_not_crashed(conn):
    with pytest.raises(db.BookingError, match="not a usable date"):
        db.book_appointment(conn, "Deniz Aydın", "Hair colouring", "9999-12-31 23:00", now=NOW)
    with pytest.raises(db.BookingError, match="no appointment"):
        db.cancel_appointment(conn, 2**70, now=NOW)
