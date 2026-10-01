"""The HITL flow end to end with a scripted model: no network, deterministic.

The scripted model plays the LLM's part (which tool to call); everything else is
the real agent: HumanInTheLoopMiddleware, the tools, the SQLite database.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from salon_booking_agent import db
from salon_booking_agent.agent import build_agent
from salon_booking_agent.seed import seed
from salon_booking_agent.tools import SalonContext

CTX = SalonContext(today="2026-10-05")  # Monday; seed puts bookings on Tue/Wed/Thu
BOOK = {"customer_name": "Deniz Aydın", "service": "Women's haircut", "start_time": "2026-10-06 12:00"}


class ScriptedModel(BaseChatModel):
    """Returns queued AIMessages in order and records what it was sent."""

    replies: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


@pytest.fixture(autouse=True)
def salon_db(tmp_path, monkeypatch):
    path = tmp_path / "salon.db"
    monkeypatch.setenv("SALON_DB_PATH", str(path))
    with db.connect(path) as conn:
        seed(conn, db.date(2026, 10, 5))
    return path


def booked_count() -> int:
    with db.connect() as conn:
        return db.count_appointments(conn)


def run(agent, payload, thread="t1"):
    return agent.invoke(payload, {"configurable": {"thread_id": thread}}, context=CTX, version="v2")


def tool_messages(result) -> list[ToolMessage]:
    return [m for m in result.value["messages"] if isinstance(m, ToolMessage)]


def test_booking_pauses_for_the_owner_then_approve_writes_the_row():
    model = ScriptedModel(replies=[call("book_appointment", BOOK), AIMessage("You're booked!")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    before = booked_count()

    paused = run(agent, {"messages": [{"role": "user", "content": "Book me a haircut Tue 12:00"}]})
    [interrupt] = paused.interrupts
    request = interrupt.value["action_requests"][0]
    assert request["name"] == "book_appointment" and request["args"] == BOOK
    assert "Deniz Aydın" in request["description"] and "13:00" in request["description"]
    assert interrupt.value["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject"]
    assert booked_count() == before  # nothing written while waiting

    done = run(agent, Command(resume={"decisions": [{"type": "approve"}]}))
    assert done.interrupts == ()
    assert booked_count() == before + 1
    assert tool_messages(done)[-1].content.startswith("BOOKED #")


def test_reject_writes_nothing_and_the_model_hears_the_reason():
    model = ScriptedModel(replies=[call("book_appointment", BOOK), AIMessage("Sorry, not booked.")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    before = booked_count()
    run(agent, {"messages": [{"role": "user", "content": "Book me"}]})

    done = run(agent, Command(resume={"decisions": [{"type": "reject", "message": "Stylist is on leave."}]}))
    assert booked_count() == before
    rejection = tool_messages(done)[-1]
    assert rejection.status == "error" and "Stylist is on leave." in rejection.content
    assert "Stylist is on leave." in model.seen[-1][-1].content  # the model was told


def test_edit_books_the_owner_s_time_instead():
    model = ScriptedModel(replies=[call("book_appointment", BOOK), AIMessage("Booked at 16:00.")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    run(agent, {"messages": [{"role": "user", "content": "Book me"}]})

    edited = {**BOOK, "start_time": "2026-10-06 16:00"}
    done = run(agent, Command(resume={"decisions": [
        {"type": "edit", "edited_action": {"name": "book_appointment", "args": edited}},
    ]}))
    with db.connect() as conn:
        [row] = [a for a in db.find_appointments(conn, "Deniz Aydın")]
    assert row.start_time == "2026-10-06T16:00"
    assert "salon owner changed this request" in tool_messages(done)[-1].content


def test_conflicting_slot_is_refused_by_the_tool_even_after_approval():
    clash = {**BOOK, "start_time": "2026-10-06 10:30"}  # overlaps Ayşe's 10:00-11:00
    model = ScriptedModel(replies=[call("book_appointment", clash), AIMessage("That time is taken.")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    before = booked_count()
    run(agent, {"messages": [{"role": "user", "content": "Book me 10:30"}]})

    done = run(agent, Command(resume={"decisions": [{"type": "approve"}]}))
    assert booked_count() == before
    assert tool_messages(done)[-1].content.startswith("NOT BOOKED:")
    assert "overlaps" in tool_messages(done)[-1].content


def test_cancel_allows_only_approve_or_reject():
    model = ScriptedModel(replies=[call("cancel_appointment", {"appointment_id": 1}), AIMessage("ok")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    paused = run(agent, {"messages": [{"role": "user", "content": "Cancel #1"}]})
    assert paused.interrupts[0].value["review_configs"][0]["allowed_decisions"] == ["approve", "reject"]
    with pytest.raises(ValueError, match="not allowed"):
        run(agent, Command(resume={"decisions": [
            {"type": "edit", "edited_action": {"name": "cancel_appointment", "args": {"appointment_id": 2}}},
        ]}))


def test_same_thread_remembers_earlier_messages():
    model = ScriptedModel(replies=[AIMessage("Hi Deniz!"), AIMessage("Your name is Deniz."), AIMessage("I don't know yet.")])
    agent = build_agent(model, checkpointer=InMemorySaver())
    run(agent, {"messages": [{"role": "user", "content": "Hi, I'm Deniz."}]}, thread="memory")
    run(agent, {"messages": [{"role": "user", "content": "What's my name?"}]}, thread="memory")
    second_call = [m.content for m in model.seen[-1]]
    assert "Hi, I'm Deniz." in second_call and "Hi Deniz!" in second_call

    run(agent, {"messages": [{"role": "user", "content": "What's my name?"}]}, thread="other")
    assert "Hi, I'm Deniz." not in [m.content for m in model.seen[-1]]
