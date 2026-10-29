"""MockLLM: a deterministic, rule-based stand-in for the LLM (tests and CI, no model needed).

It is not a fake that returns canned strings: it picks tools by keyword, really executes
them through the orchestrator, and writes its answer from the tool results / excerpts.
So a mock eval run exercises the real pipeline: RBAC, masking, verifier, audit, metrics.
"""

import json
import re
import uuid

from app.llm import ChatResult, ToolCall

_TXN = re.compile(r"\bT\d{6}\b")
_CUST = re.compile(r"\bC\d{4}\b")
_DAYS = re.compile(r"(\d+)\s*(?:\+\s*)?days?")
_MONTHS = re.compile(r"(\d+)\s*months?")
_TIER = re.compile(r"\btier\s*([ABCD])\b", re.I)
_TAG = re.compile(r"\[[a-z_]+ §\d+\]")
_WORD = re.compile(r"[a-z]{4,}")


def _call(name: str, args: dict) -> ToolCall:
    return ToolCall(f"call_{uuid.uuid4().hex[:8]}", name, json.dumps(args))


def _segment(q: str) -> str | None:
    return "sme" if "sme" in q else "retail" if "retail" in q else None


def _best_line(question: str, excerpt: str) -> str:
    """Best-overlapping rule line of a tagged excerpt, with its source tag moved to the end once."""
    words = set(_WORD.findall(question.lower()))
    lines = [ln.strip() for ln in excerpt.split("\n")[1:] if len(ln.strip()) > 20]
    best = max(lines, key=lambda s: len(words & set(_WORD.findall(s.lower()))), default="")
    tag = _TAG.search(excerpt)
    body = re.sub(r"^\d+\.\s*", "", _TAG.sub("", best)).strip()
    return f"{body} {tag.group(0)}" if tag else body


class MockLLM:
    def chat(self, messages, tools=None, tool_choice=None) -> ChatResult:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        question = next(m["content"] for m in messages if m["role"] == "user")
        q = question.lower()
        tool_msgs = [m for m in messages if m["role"] == "tool"]

        if tools and not tool_msgs:
            names = {t["function"]["name"] for t in tools}
            calls = self._pick_tools(question, q, names)
            if calls:
                return ChatResult(tool_calls=calls, tokens=50)
            return ChatResult(content="I can only help with bank policies and your portfolio.", tokens=20)
        if tool_msgs:
            return ChatResult(content=self._answer_from_tools(question, tool_msgs), tokens=80)
        if "Policy excerpts:" in system:
            return ChatResult(content=self._answer_from_excerpts(question, system), tokens=60)
        return ChatResult(content="I don't have access to that information.", tokens=10)

    def _pick_tools(self, question: str, q: str, names: set[str]) -> list[ToolCall]:
        calls = []
        if m := _TXN.search(question):
            calls.append(_call("get_transaction", {"txn_id": m.group(0)}))
            if "search_policies" in names:
                calls.append(_call("search_policies", {"query": "AML transaction monitoring flag rules"}))
        elif m := _CUST.search(question):
            calls.append(_call("get_customer_profile", {"customer_id": m.group(0)}))
        elif "kyc" in q:
            m = _MONTHS.search(q)
            calls.append(_call("customers_due_for_kyc", {"months": int(m.group(1)) if m else 24}))
        elif any(w in q for w in ("past due", "overdue", "arrears", "dpd")):
            m = _DAYS.search(q)
            args = {"min_dpd": int(m.group(1)) if m else 30}
            if seg := _segment(q):
                args["segment"] = seg
            calls.append(_call("list_past_due_loans", args))
        elif any(w in q for w in ("exposure", "outstanding", "portfolio")):
            args = {}
            if seg := _segment(q):
                args["segment"] = seg
            if m := _TIER.search(question):
                args["risk_tier"] = m.group(1).upper()
            calls.append(_call("get_portfolio_exposure", args))
        elif "search_policies" in names:
            calls.append(_call("search_policies", {"query": question}))
        return calls

    def _answer_from_tools(self, question: str, tool_msgs: list[dict]) -> str:
        parts = []
        for m in tool_msgs:
            data = json.loads(m["content"])
            if "error" in data:
                parts.append(f"Access denied: {data['error']['message']}."
                             if data["error"]["code"] == "access_denied"
                             else f"The tool returned an error: {data['error']['message']}.")
            elif "results" in data:
                if data.get("found") and data["results"]:
                    top = data["results"][0]
                    parts.append(_best_line(question, top["content"]))
                else:
                    parts.append("I don't have a policy source for that.")
            else:
                facts = [f"{k.replace('_', ' ')}: {v}" for k, v in data.items()
                         if isinstance(v, (int, float, str)) and not isinstance(v, bool) and v is not None]
                parts.append("; ".join(facts) + ".")
        return " ".join(parts)

    def _answer_from_excerpts(self, question: str, system: str) -> str:
        excerpts = system.split("Policy excerpts:", 1)[1].strip()
        if not _TAG.search(excerpts):
            return "I don't have a policy source for that."
        first = excerpts.split("\n\n", 1)[0]
        return _best_line(question, first)
