"""Tool registry: OpenAI-style JSON schemas for function calling, and one guarded executor.

Schemas are generated from the Pydantic argument models, so they never contain `rm_id`.
`execute_tool` is the single entry point used by the orchestrator (and the MCP server):
validate args -> call the tool with the server-side rm_id -> mask PII -> JSON-safe dict.
Errors come back as data (never exceptions), so the LLM can read them and recover.
"""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from app.guardrails.pii import mask_pii
from app.rag.retrieve import format_excerpt, retrieve, source_tag
from app.tools import banking_tools as bt

log = logging.getLogger(__name__)


class SearchPoliciesArgs(bt._Args):
    query: str = Field(min_length=3, max_length=500, description="What to look up in the bank policies")


def search_policies(rm_id: str, query: str) -> dict:
    """Hybrid search over the policy documents. rm_id is unused (policies are not customer data)."""
    bt._check_rm(rm_id)
    r = retrieve(query)
    if r.weak:
        return {"found": False, "message": "No relevant policy found.", "results": []}
    return {
        "found": True,
        "results": [{"source": source_tag(c), "content": format_excerpt(c)} for c in r.chunks],
    }


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable
    args: type[BaseModel]
    description: str


TOOLS: dict[str, ToolSpec] = {
    "list_past_due_loans": ToolSpec(
        bt.list_past_due_loans, bt.ListPastDueLoansArgs,
        "List the RM's loans that are more than min_dpd days past due, with count and total outstanding.",
    ),
    "get_portfolio_exposure": ToolSpec(
        bt.get_portfolio_exposure, bt.PortfolioExposureArgs,
        "Total exposure (sum of outstanding balances) of the RM's loans, optionally by segment / risk tier.",
    ),
    "get_customer_profile": ToolSpec(
        bt.get_customer_profile, bt.CustomerProfileArgs,
        "Profile of one customer in the RM's portfolio: segment, region, KYC date, loans, flagged txns.",
    ),
    "get_transaction": ToolSpec(
        bt.get_transaction, bt.TransactionArgs,
        "One transaction of the RM's customers, including whether it was flagged and the AML flag_rule.",
    ),
    "customers_due_for_kyc": ToolSpec(
        bt.customers_due_for_kyc, bt.KycDueArgs,
        "The RM's customers whose last KYC review is at least `months` months old.",
    ),
}
POLICY_TOOL = "search_policies"
TOOLS_WITH_POLICIES = {
    **TOOLS,
    POLICY_TOOL: ToolSpec(search_policies, SearchPoliciesArgs,
                          "Search the bank's policy documents. Results carry [doc §N] source tags to cite."),
}


def _clean_schema(schema: dict) -> dict:
    """Pydantic JSON schema -> compact OpenAI parameters schema (no titles, no anyOf-null)."""
    props = {}
    for name, p in schema.get("properties", {}).items():
        p = {k: v for k, v in p.items() if k != "title"}
        if "anyOf" in p:  # Optional[X] -> X, not required
            non_null = [s for s in p.pop("anyOf") if s.get("type") != "null"]
            p = {**non_null[0], **p}
        props[name] = p
    return {"type": "object", "properties": props, "required": schema.get("required", []),
            "additionalProperties": False}


def tool_schemas(include_policies: bool = False) -> list[dict]:
    specs = TOOLS_WITH_POLICIES if include_policies else TOOLS
    return [
        {"type": "function",
         "function": {"name": name, "description": s.description,
                      "parameters": _clean_schema(s.args.model_json_schema())}}
        for name, s in specs.items()
    ]


def execute_tool(name: str, raw_args: str | dict | None, rm_id: str, include_policies: bool = True) -> dict:
    """Run a tool for `rm_id`. Returns {"ok": bool, "result"|"error": ..., "args": ...}."""
    specs = TOOLS_WITH_POLICIES if include_policies else TOOLS
    if name not in specs:
        return {"ok": False, "error": {"code": "unknown_tool", "message": f"No tool named {name!r}."}}
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        if not isinstance(args, dict):
            raise ValueError("arguments must be a JSON object")
    except (json.JSONDecodeError, ValueError) as e:
        return {"ok": False, "error": {"code": "invalid_arguments", "message": str(e)}}
    if "rm_id" in args:
        log.warning("LLM tried to pass rm_id=%r to %s; ignored and rejected", args["rm_id"], name)
        return {"ok": False, "args": args, "error": {
            "code": "invalid_arguments",
            "message": "rm_id is set by the server from the logged-in user and cannot be passed."}}
    spec = specs[name]
    try:
        parsed = spec.args.model_validate(args)
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
        return {"ok": False, "args": args, "error": {"code": "invalid_arguments", "message": msg}}
    clean_args = parsed.model_dump(exclude_none=True)
    try:
        out = spec.fn(rm_id, **parsed.model_dump())
    except bt.AccessDenied as e:
        return {"ok": False, "args": clean_args, "error": {"code": "access_denied", "message": str(e)}}
    payload = out.model_dump(mode="json") if isinstance(out, BaseModel) else out
    # Defence in depth: mask the serialized result too, not only the known PII fields.
    payload = json.loads(mask_pii(json.dumps(payload, ensure_ascii=False)))
    return {"ok": True, "args": clean_args, "result": payload}


def tool_message_content(outcome: dict) -> str:
    """What the LLM sees for a tool call."""
    body = outcome["result"] if outcome["ok"] else {"error": outcome["error"]}
    return json.dumps(body, ensure_ascii=False, default=str)


class ToolBackend(Protocol):
    """Where tool calls go: in-process (LocalToolBackend) or over MCP (eval/mcp_client.py)."""

    def schemas(self, include_policies: bool, rm_id: str) -> list[dict]: ...

    def execute(self, name: str, raw_args: str | dict | None, rm_id: str, include_policies: bool) -> dict: ...


class LocalToolBackend:
    def schemas(self, include_policies: bool, rm_id: str) -> list[dict]:
        return tool_schemas(include_policies)

    def execute(self, name, raw_args, rm_id, include_policies) -> dict:
        return execute_tool(name, raw_args, rm_id, include_policies)
