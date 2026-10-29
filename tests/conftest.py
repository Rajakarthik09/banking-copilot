"""Tests run against an isolated `<db>_test` database with MockLLM + MockEmbedder.

DB-backed tests use the `db` fixture: it seeds the test DB once per session, and skips
(or fails, if REQUIRE_DB=1 as in CI) when Postgres isn't reachable.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from setup_test_db import setup_test_db, use_test_env  # noqa: E402

use_test_env()  # must run before any app module reads settings

from app.llm import ChatResult, ToolCall  # noqa: E402

_state: dict = {}


@pytest.fixture(scope="session")
def db():
    if "ok" not in _state:
        try:
            setup_test_db()
            _state["ok"] = True
        except Exception as e:  # noqa: BLE001
            _state["ok"], _state["err"] = False, e
    if not _state["ok"]:
        if os.environ.get("REQUIRE_DB") == "1":
            pytest.fail(f"database required but unavailable: {_state['err']}")
        pytest.skip(f"database unavailable: {_state['err']}")


class ScriptedLLM:
    """Plays back a fixed list of responses; records every message list it was sent."""

    def __init__(self, responses: list[ChatResult]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, tool_choice=None) -> ChatResult:
        self.calls.append(messages)
        if not self.responses:
            return ChatResult(content="done")
        return self.responses.pop(0)


def tool_call(name: str, args: str) -> ChatResult:
    return ChatResult(tool_calls=[ToolCall(f"c{abs(hash((name, args))) % 10_000}", name, args)])


@pytest.fixture
def scripted():
    return ScriptedLLM
