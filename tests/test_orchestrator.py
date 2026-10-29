import json

import pytest

from app.config import get_settings
from app.llm import ChatResult, parse_text_tool_calls
from app.orchestrator import NO_SOURCE, answer_question, detect_refusal
from tests.conftest import tool_call

NAMES = {"get_transaction", "list_past_due_loans"}


def test_parse_tool_call_from_tagged_text():
    calls = parse_text_tool_calls('<tool_call>\n{"name": "get_transaction", "arguments": {"txn_id": "T000001"}}\n'
                                  "</tool_call>", NAMES)
    assert len(calls) == 1 and calls[0].name == "get_transaction"
    assert json.loads(calls[0].arguments) == {"txn_id": "T000001"}


def test_parse_tool_call_from_bare_json_and_code_fence():
    for text in ['{"name": "list_past_due_loans", "arguments": {"min_dpd": 60}}',
                 '```json\n{"name": "list_past_due_loans", "parameters": {"min_dpd": 60}}\n```']:
        calls = parse_text_tool_calls(text, NAMES)
        assert calls and json.loads(calls[0].arguments) == {"min_dpd": 60}


def test_unknown_or_malformed_text_calls_ignored():
    assert parse_text_tool_calls('{"name": "drop_tables", "arguments": {}}', NAMES) == []
    assert parse_text_tool_calls("<tool_call>{not json}</tool_call>", NAMES) == []
    assert parse_text_tool_calls("The answer is 42.", NAMES) == []


@pytest.mark.parametrize("text,reason", [
    ("I don't have a policy source for that.", "no_source"),
    ("I can only help with bank policies and your portfolio.", "out_of_scope"),
    ("Access denied: customer C0002 not found or not in your portfolio.", "access_denied"),
    ("I don't have access to that information.", "no_data_access"),
    ("You have 8 loans past due.", None),
])
def test_detect_refusal(text, reason):
    assert detect_refusal(text) == reason


def test_tool_loop_is_capped(db, scripted):
    cap = get_settings().max_tool_calls
    llm = scripted([tool_call("list_past_due_loans", json.dumps({"min_dpd": d})) for d in range(10)]
                   + [ChatResult(content="final")])
    r = answer_question("loans past due", "RM-1", "tools", verify=False, llm=llm, audit=False)
    assert len(r.tools_called) == cap
    assert "Tool call limit reached" in llm.calls[-1][-1]["content"]


def test_bad_arguments_are_fed_back_so_the_model_can_retry(db, scripted):
    llm = scripted([tool_call("list_past_due_loans", '{"min_dpd": "lots"}'),
                    tool_call("list_past_due_loans", '{"min_dpd": 60}'),
                    ChatResult(content="done")])
    r = answer_question("loans more than 60 days past due", "RM-1", "tools", verify=False, llm=llm, audit=False)
    assert [t["ok"] for t in r.tools_called] == [False, True]
    feedback = [m for m in llm.calls[1] if m["role"] == "tool"][0]["content"]
    assert "invalid_arguments" in feedback


def test_weak_retrieval_refuses_without_calling_llm(db, scripted):
    llm = scripted([])
    r = answer_question("What's the weather in Leipzig?", "RM-1", "rag", llm=llm, audit=False)
    assert r.answer == NO_SOURCE and r.refusal_reason == "no_source" and llm.calls == []


def test_rag_answer_without_citation_is_not_trusted(db, scripted):
    llm = scripted([ChatResult(content="Customers can get 2 waivers.")])
    r = answer_question("How many fee waivers can a customer get per year?", "RM-1", "rag", llm=llm, audit=False)
    assert r.answer == NO_SOURCE and r.verified is False


def test_verify_off_keeps_raw_answer_but_still_measures(db, scripted):
    llm = scripted([tool_call("list_past_due_loans", "{}"), ChatResult(content="You have 999 loans past due.")])
    r = answer_question("loans past due", "RM-1", "tools", verify=False, llm=llm, audit=False)
    assert r.answer == "You have 999 loans past due." and r.verified is False


def test_mock_llm_end_to_end_modes(db):
    for mode in ["prompt_only", "rag", "tools", "tools_rag"]:
        r = answer_question("How many fee waivers can a customer get per year?", "RM-1", mode, audit=False)
        assert r.answer
