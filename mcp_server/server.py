"""MCP server exposing the same governed banking tools (plus search_policies) to any MCP client.

The RM is fixed by the RM_ID environment variable of this server process. An MCP client
(Claude Desktop, Cursor, an agent) can call the tools but cannot change who it is: no tool
takes an rm_id argument, and RBAC runs in the same tool code as the API path.

stdio transport: stdout carries the protocol, so nothing in here may print; logs go to stderr.

Run:  RM_ID=RM-1 python -m mcp_server.server
"""

import logging
import os
import re
import sys
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from app.tools.registry import execute_tool

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

RM_ID = os.environ.get("RM_ID", "")
if not re.fullmatch(r"RM-\d+", RM_ID):
    sys.stderr.write("RM_ID must be set, e.g. RM_ID=RM-1\n")
    sys.exit(2)

mcp = FastMCP("banking-copilot")


def _run(name: str, args: dict) -> dict:
    out = execute_tool(name, {k: v for k, v in args.items() if v is not None}, RM_ID)
    if not out["ok"]:
        raise ToolError(f"{out['error']['code']}: {out['error']['message']}")
    return out["result"]


@mcp.tool()
def list_past_due_loans(min_dpd: int = 30, segment: Literal["retail", "sme"] | None = None) -> dict:
    """List your loans that are MORE than min_dpd days past due, with count and total outstanding (EUR)."""
    return _run("list_past_due_loans", {"min_dpd": min_dpd, "segment": segment})


@mcp.tool()
def get_portfolio_exposure(segment: Literal["retail", "sme"] | None = None,
                           risk_tier: Literal["A", "B", "C", "D"] | None = None) -> dict:
    """Total exposure (sum of outstanding balances, EUR) of your loans, optionally by segment / risk tier."""
    return _run("get_portfolio_exposure", {"segment": segment, "risk_tier": risk_tier})


@mcp.tool()
def get_customer_profile(customer_id: str) -> dict:
    """Profile of one customer in your portfolio (e.g. C0042): segment, region, KYC date, loans, flags."""
    return _run("get_customer_profile", {"customer_id": customer_id})


@mcp.tool()
def get_transaction(txn_id: str) -> dict:
    """One transaction of your customers (e.g. T000123), including whether it was flagged and the AML rule."""
    return _run("get_transaction", {"txn_id": txn_id})


@mcp.tool()
def customers_due_for_kyc(months: int = 24) -> dict:
    """Your customers whose last KYC review is at least `months` months old."""
    return _run("customers_due_for_kyc", {"months": months})


@mcp.tool()
def search_policies(query: str) -> dict:
    """Search the bank's policy documents. Results carry [doc §N] source tags to cite."""
    return _run("search_policies", {"query": query})


if __name__ == "__main__":
    mcp.run()  # stdio
