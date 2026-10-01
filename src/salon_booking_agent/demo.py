"""Terminal demo: chat as a customer, approve or reject as the salon owner.

    uv run salon-chat              # interactive: you play both the customer and the owner
    uv run salon-chat --scripted   # three canned conversations: approve, reject, edit

Every conversation uses its own thread id, so the agent remembers earlier turns
within it and nothing across them.
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import date

from .config import load_env, make_model

load_env()

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from . import db  # noqa: E402
from .agent import build_agent  # noqa: E402
from .seed import next_open_days, seed  # noqa: E402
from .tools import SalonContext  # noqa: E402

BOLD, DIM, GREEN, RED, YELLOW, CYAN, RESET = (
    "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[0m",
)


def say(role: str, text: str, colour: str) -> None:
    print(f"{colour}{BOLD}{role}:{RESET} {text}\n")


def show_request(action: dict) -> None:
    print(f"{YELLOW}{BOLD}── approval needed ─────────────────────────────{RESET}")
    print(action.get("description") or action)
    print(f"{YELLOW}{BOLD}────────────────────────────────────────────────{RESET}")


def ask_owner(action: dict, allowed: list[str]) -> dict:
    """Interactive owner decision for one action request."""
    show_request(action)
    options = "/".join(f"[{d[0]}]{d[1:]}" for d in allowed)
    while True:
        choice = input(f"{YELLOW}Owner{RESET} {options}: ").strip().lower()[:1]
        if choice == "a" and "approve" in allowed:
            return {"type": "approve"}
        if choice == "r" and "reject" in allowed:
            reason = input("  reason for the customer: ").strip() or "The salon cannot take this booking."
            return {"type": "reject", "message": reason}
        if choice == "e" and "edit" in allowed:
            args = dict(action["args"])
            new_time = input(f"  new start_time [{args.get('start_time')}]: ").strip()
            if new_time:
                args["start_time"] = new_time
            return {"type": "edit", "edited_action": {"name": action["name"], "args": args}}


def turn(agent, config: dict, context: SalonContext, text: str, decide) -> None:
    """One customer message, including any owner review it triggers."""
    say("Customer", text, CYAN)
    result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=context, version="v2")
    while result.interrupts:
        request = result.interrupts[0].value
        decisions = [
            decide(action, review["allowed_decisions"])
            for action, review in zip(request["action_requests"], request["review_configs"], strict=True)
        ]
        for d in decisions:
            colour = {"approve": GREEN, "reject": RED}.get(d["type"], YELLOW)
            detail = d.get("message") or d.get("edited_action", {}).get("args", {}).get("start_time", "")
            print(f"{colour}{BOLD}Owner → {d['type']}{RESET} {detail}\n")
        result = agent.invoke(Command(resume={"decisions": decisions}), config, context=context, version="v2")
    say("Assistant", result.value["messages"][-1].text, GREEN)


def scripted(agent, context: SalonContext) -> None:
    day1, day2, _ = next_open_days(date.today(), 3)
    d1, d2 = day1.isoformat(), day2.isoformat()

    def owner(*decisions):
        queue = list(decisions)

        def decide(action, allowed):
            show_request(action)
            decision = queue.pop(0)
            if decision["type"] == "edit":
                decision["edited_action"]["args"] = {**action["args"], **decision["edited_action"]["args"]}
            return decision

        return decide

    print(f"{DIM}=== 1. Availability + booking, approved ==={RESET}\n")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    turn(agent, config, context, f"Hi! Which times are free for a women's haircut on {d1}?", owner())
    turn(agent, config, context, "12:00 please. My name is Deniz Aydın.", owner({"type": "approve"}))

    print(f"{DIM}=== 2. Booking, rejected by the owner ==={RESET}\n")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    turn(agent, config, context, f"Can I book a beard trim on {d2} at 15:00? I'm Emre Koç.",
         owner({"type": "reject", "message": "Our barber is off that afternoon, please pick the morning."}))

    print(f"{DIM}=== 3. Booking, time edited by the owner ==={RESET}\n")
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    turn(agent, config, context, f"I'm Selin Arslan. Please book a blow-dry for {d1} at 11:00.",
         owner({"type": "edit", "edited_action": {"name": "book_appointment", "args": {"start_time": f"{d1} 13:00"}}}))
    turn(agent, config, context, "Great, and what's my appointment again?", owner())

    with db.connect() as conn:
        rows = conn.execute(
            "SELECT a.id, a.customer_name, s.name, a.start_time FROM appointments a "
            "JOIN services s ON s.id = a.service_id WHERE a.customer_name IN "
            "('Deniz Aydın', 'Emre Koç', 'Selin Arslan') ORDER BY a.id"
        ).fetchall()
    print(f"{DIM}=== Database rows written in this demo ==={RESET}")
    for r in rows:
        print(f"  #{r[0]} {r[1]:<13} {r[2]:<16} {r[3].replace('T', ' ')}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scripted", action="store_true", help="run three canned conversations")
    parser.add_argument("--no-reseed", action="store_true", help="keep the current database")
    args = parser.parse_args()

    context = SalonContext()  # real clock
    if not args.no_reseed:
        with db.connect() as conn:
            seed(conn, date.today())
    agent = build_agent(make_model(), checkpointer=InMemorySaver())

    if args.scripted:
        scripted(agent, context)
        return
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"{DIM}Salon Lara booking assistant. Type 'quit' to exit.{RESET}\n")
    while (text := input(f"{CYAN}You{RESET}: ").strip()).lower() not in {"quit", "exit"}:
        if text:
            print("\033[1A\033[2K", end="")  # replace the raw input line with the formatted one
            turn(agent, config, context, text, ask_owner)


if __name__ == "__main__":
    main()
