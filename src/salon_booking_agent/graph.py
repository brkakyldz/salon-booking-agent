"""Entry point for `langgraph dev` (see langgraph.json). No checkpointer here: the
server persists threads itself."""

from salon_booking_agent.config import load_env, make_model

load_env()

from salon_booking_agent.agent import build_agent  # noqa: E402  (env must be loaded before LangChain traces)

agent = build_agent(make_model())
