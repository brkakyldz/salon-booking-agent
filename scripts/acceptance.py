"""End-to-end acceptance checks against the real model and LangSmith.

    uv run python scripts/acceptance.py

Uses its own database (data/acceptance.db, reseeded relative to today) so the
demo database is left alone. Check 5 needs LANGSMITH_API_KEY in .env.
"""

from __future__ import annotations

import os
import re
import sys
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parents[1]
os.environ["SALON_DB_PATH"] = str(PROJECT_DIR / "data" / "acceptance.db")

from salon_booking_agent.config import LANGSMITH_PROJECT, load_env, make_model  # noqa: E402

load_env()

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from salon_booking_agent import db  # noqa: E402
from salon_booking_agent.agent import build_agent  # noqa: E402
from salon_booking_agent.seed import next_open_days, seed  # noqa: E402
from salon_booking_agent.tools import SalonContext, book_appointment  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []
THREADS: dict[str, str] = {}


def check(name: str, ok: bool, evidence: str) -> None:
    RESULTS.append((name, ok, evidence))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {evidence}\n")


class Conversation:
    def __init__(self, agent, label: str):
        self.agent, self.config = agent, {"configurable": {"thread_id": str(uuid.uuid4())}}
        THREADS[label] = self.config["configurable"]["thread_id"]
        self.interrupts: list[dict] = []
        self.tool_results: list[str] = []

    def say(self, text: str, decide=None, follow_ups=("Yes, please go ahead and book it.",)) -> str:
        """Send a message; `decide(action)` answers each approval request. If a write was
        expected but the model only asked for confirmation, confirm once."""
        result = self._run({"messages": [{"role": "user", "content": text}]}, decide)
        for extra in follow_ups if decide and not self._decided else ():
            result = self._run({"messages": [{"role": "user", "content": extra}]}, decide)
        return result.value["messages"][-1].text

    def _run(self, payload, decide):
        self._decided = False
        result = self.agent.invoke(payload, self.config, context=SalonContext(), version="v2")
        while result.interrupts:
            request = result.interrupts[0].value
            self.interrupts.append(request)
            decisions = [decide(a) for a in request["action_requests"]]
            self._decided = True
            result = self.agent.invoke(Command(resume={"decisions": decisions}), self.config,
                                       context=SalonContext(), version="v2")
        self.tool_results = [m.content for m in result.value["messages"] if m.type == "tool"]
        return result


def booked_count() -> int:
    with db.connect() as conn:
        return db.count_appointments(conn)


def rows_for(name: str) -> list[db.Appointment]:
    with db.connect() as conn:
        return db.find_appointments(conn, name)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    today = date.today()
    with db.connect() as conn:
        seed(conn, today)
    d1, d2, _ = (d.isoformat() for d in next_open_days(today, 3))
    agent = build_agent(make_model(), checkpointer=InMemorySaver())
    print(f"Model {make_model().model_name}, database {db.db_path()}, first open days {d1} / {d2}\n")

    # 1. Availability matches the seed.
    with db.connect() as conn:
        expected = db.available_slots(conn, date.fromisoformat(d1), "Women's haircut")
    chat = Conversation(agent, "availability")
    answer = chat.say(f"Which times are free for a women's haircut on {d1}?")
    opens, closes = (t.strftime("%H:%M") for t in db.opening_hours_for(date.fromisoformat(d1)))
    listed = {h.replace(".", ":") for h in re.findall(r"\b(?:[01]\d|2[0-3])[:.][0-5]\d\b", answer)}
    listed -= {opens, closes} - set(expected)  # the answer may mention the opening hours
    check("1. availability matches seed data", listed == set(expected) and not chat.interrupts,
          f"expected {expected}; times in the answer {sorted(listed)}; answer: {answer!r}")

    # 2a. Approve -> row.
    chat = Conversation(agent, "approve")
    answer = chat.say(f"Hi, I'm Deniz Aydın. Please book a women's haircut on {d1} at 12:00.",
                      decide=lambda a: {"type": "approve"})
    rows = rows_for("Deniz Aydın")
    check("2a. booking interrupts; approve writes the row",
          len(chat.interrupts) == 1 and [r.start_time for r in rows] == [f"{d1}T12:00"],
          f"interrupt: {chat.interrupts[0]['action_requests'][0]['args'] if chat.interrupts else None}; "
          f"rows: {[(r.id, r.start_time) for r in rows]}; answer: {answer!r}")
    approve_chat = chat

    # 2b. Reject -> no row, customer is told.
    chat = Conversation(agent, "reject")
    before = booked_count()
    answer = chat.say(f"I'm Emre Koç. Book me a beard trim on {d2} at 15:00, please.",
                      decide=lambda a: {"type": "reject", "message": "The barber is off that afternoon."})
    after = booked_count()
    plain = answer.lower().replace("’", "'")
    negated = re.search(r"\b(not|wasn't|couldn't|can't|cannot|unable|isn't|declined|rejected)\b", plain)
    told = bool(negated) and "barber" in plain  # says it did not happen, and passes on the reason
    check("2b. reject writes nothing and the customer is told",
          len(chat.interrupts) == 1 and before == after and not rows_for("Emre Koç") and told,
          f"appointments before/after: {before}/{after}; answer: {answer!r}")

    # 2c. Edit -> row at the owner's time.
    chat = Conversation(agent, "edit")

    def move_to_1300(action):  # times chosen to fit every open day, Saturday 10-17 included
        return {"type": "edit", "edited_action": {"name": action["name"],
                                                  "args": {**action["args"], "start_time": f"{d1} 13:00"}}}

    answer = chat.say(f"I'm Selin Arslan. Please book a blow-dry on {d1} at 11:00.", decide=move_to_1300)
    rows = rows_for("Selin Arslan")
    check("2c. edit books the edited time",
          [r.start_time for r in rows] == [f"{d1}T13:00"] and "13:00" in answer,
          f"rows: {[(r.id, r.start_time) for r in rows]}; answer: {answer!r}")

    # 3. A conflicting time is refused inside the tool, even when the owner approves it.
    chat = Conversation(agent, "conflict")

    def move_onto_ayse(action):  # the owner (by mistake) moves it onto Ayşe's 10:00-11:00 haircut
        return {"type": "edit", "edited_action": {"name": action["name"],
                                                  "args": {**action["args"], "start_time": f"{d1} 10:30"}}}

    before = booked_count()
    answer = chat.say(f"I'm Can Er. Please book a men's haircut on {d1} at 16:00.", decide=move_onto_ayse)
    after = booked_count()
    refusal = next((t for t in chat.tool_results if "NOT BOOKED" in t), "")
    direct = book_appointment.func("Can Er", "Men's haircut", f"{d1} 10:00",
                                   runtime=SimpleNamespace(context=SalonContext()))
    check("3. conflicting slot refused by the tool",
          before == after and "overlaps" in refusal and direct.startswith("NOT BOOKED"),
          f"tool result after approved clash: {refusal!r}; direct tool call: {direct!r}")

    # 4. Same thread remembers; a new thread does not.
    answer = approve_chat.say("Sorry, what name and time did you book me under?")
    fresh = Conversation(agent, "fresh").say("Sorry, what name and time did you book me under?")
    check("4. same thread remembers earlier messages",
          "Deniz" in answer and "12:00" in answer and "Deniz" not in fresh,
          f"same thread: {answer!r}; new thread: {fresh!r}")

    # 5. Traces in LangSmith, before and after the interrupt, grouped by thread.
    if not os.environ.get("LANGSMITH_API_KEY"):
        check("5. LangSmith traces", False, "NOT RUN: LANGSMITH_API_KEY is not set")
    else:
        from langchain_core.tracers.langchain import wait_for_all_tracers
        from langsmith import Client

        wait_for_all_tracers()
        client, project = Client(), LANGSMITH_PROJECT  # the project set in config.py, not whatever the env says
        wanted, roots = THREADS["approve"], []
        for _ in range(36):  # ingestion is asynchronous and can take a couple of minutes
            roots = sorted(
                (r for r in client.list_runs(project_name=project, is_root=True, limit=100,
                                             start_time=datetime.now(UTC) - timedelta(minutes=30))
                 if ((r.extra or {}).get("metadata") or {}).get("thread_id") == wanted),
                key=lambda r: r.start_time,
            )
            if len(roots) >= 3:
                break
            time.sleep(5)
        names = [{r.name for r in client.list_runs(project_name=project, trace_id=root.id)} for root in roots[:2]]
        before = bool(names) and "HumanInTheLoopMiddleware.after_model" in names[0] and "book_appointment" not in names[0]
        after = len(names) > 1 and "book_appointment" in names[1]
        check("5. traces before and after the interrupt land in LangSmith, in one thread",
              len(roots) >= 3 and before and after,
              f"project {project}, thread {wanted}: {len(roots)} root runs; "
              f"run 1 stops at HumanInTheLoopMiddleware.after_model: {before}; "
              f"run 2 (after approval) contains the book_appointment tool run: {after}"
              + (f"; {client.get_run_url(run=roots[0], project_name=project)}" if roots else ""))

    passed = sum(ok for _, ok, _ in RESULTS)
    print(f"{passed}/{len(RESULTS)} checks passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
