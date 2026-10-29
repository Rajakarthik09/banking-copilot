"""MCP tool backend: the orchestrator's tool calls go over stdio to mcp_server/server.py.

Used by the eval's `mcp` config to measure MCP against in-process function calling on the
same questions. The LLM sees the tool schemas *as advertised by the MCP server*.

One server process per RM, because RM_ID is fixed per process. MCP's stdio client uses anyio
cancel scopes that must be exited by the task that entered them, so each session lives in
its own long-running "owner" task on a background event loop.
"""

import asyncio
import json
import os
import re
import sys
import threading
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
_ERR = re.compile(r"(access_denied|invalid_arguments|unknown_tool):\s*(.*)", re.S)


class McpToolBackend:
    def __init__(self, timeout_s: float = 300) -> None:
        self.timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._sessions: dict[str, ClientSession] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self._tools: list | None = None

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.timeout_s)

    async def _owner(self, rm_id: str, ready: asyncio.Future, stop: asyncio.Event) -> None:
        params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server"],
                                       env={**os.environ, "RM_ID": rm_id}, cwd=str(ROOT))
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                ready.set_result(session)
                await stop.wait()
        except Exception as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e)

    async def _open(self, rm_id: str) -> ClientSession:
        ready, stop = self._loop.create_future(), asyncio.Event()
        self._loop.create_task(self._owner(rm_id, ready, stop))
        session = await ready
        self._sessions[rm_id], self._stops[rm_id] = session, stop
        return session

    def _session(self, rm_id: str) -> ClientSession:
        return self._sessions.get(rm_id) or self._run(self._open(rm_id))

    def schemas(self, include_policies: bool, rm_id: str) -> list[dict]:
        if self._tools is None:
            self._tools = self._run(self._session(rm_id).list_tools()).tools
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description or "", "parameters": t.inputSchema}}
                for t in self._tools if include_policies or t.name != "search_policies"]

    def execute(self, name: str, raw_args, rm_id: str, include_policies: bool) -> dict:
        if name == "search_policies" and not include_policies:
            return {"ok": False, "error": {"code": "unknown_tool", "message": f"No tool named {name!r}."}}
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            return {"ok": False, "error": {"code": "invalid_arguments", "message": str(e)}}
        result = self._run(self._session(rm_id).call_tool(name, args))
        text = "".join(getattr(c, "text", "") for c in result.content)
        if result.isError:
            m = _ERR.search(text)
            code, msg = (m.group(1), m.group(2).strip()) if m else ("invalid_arguments", text[:300])
            return {"ok": False, "args": args, "error": {"code": code, "message": msg}}
        payload = result.structuredContent
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if payload is None:
            payload = json.loads(text)
        return {"ok": True, "args": args, "result": payload}

    def close(self) -> None:
        for stop in self._stops.values():
            self._loop.call_soon_threadsafe(stop.set)
