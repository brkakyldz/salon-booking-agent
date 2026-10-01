"""Environment and model setup, shared by the dev server, the demo and the tests.

Keys live in `.env` at the repository root (see `.env.example`). This module loads
it and pins the LangSmith project *before* anything traced runs:
langsmith caches its environment lookups, so setting them later has no effect.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

LANGSMITH_PROJECT = "salon-booking-agent"
DEFAULT_MODEL = "gpt-6-luna"
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")


def make_model():
    """gpt-6 models only call tools on Chat Completions at reasoning effort `none`,
    so the Responses API is requested explicitly."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        use_responses_api=True,
        reasoning={"effort": "low"},
    )
