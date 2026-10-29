"""FastAPI: POST /chat, GET /me, GET /health.

Endpoints are sync `def`: FastAPI runs them in a thread pool, which suits the blocking
LLM and psycopg calls (see docs/decisions.md for what changes at scale).
"""

import logging
from typing import Literal

import httpx
import openai
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.auth import current_rm
from app.config import get_settings
from app.db import fetch_one
from app.orchestrator import answer_question

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Banking Operations Copilot", version="0.1.0")


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no rm_id (or anything else) in the body

    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["prompt_only", "rag", "tools", "tools_rag"] = "tools_rag"
    verify: bool = True


class ChatOut(BaseModel):
    answer: str
    sources: list[dict]
    tools_called: list[dict]
    verified: bool | None
    refused: bool
    refusal_reason: str | None
    removed: list[str]
    latency_ms: int
    tokens: int


@app.post("/chat", response_model=ChatOut)
def chat(req: ChatRequest, rm_id: str = Depends(current_rm)) -> ChatOut:
    try:
        r = answer_question(req.question, rm_id, req.mode, req.verify)
    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        log.error("LLM unavailable: %s", e)
        raise HTTPException(503, "The language model is unavailable. Is Ollama running?") from e
    except RuntimeError as e:  # e.g. embedding-model mismatch, empty policy index
        raise HTTPException(503, str(e)) from e
    return ChatOut(**{k: getattr(r, k) for k in ChatOut.model_fields})


@app.get("/me")
def me(rm_id: str = Depends(current_rm)) -> dict:
    row = fetch_one("SELECT rm_id, name, region FROM relationship_managers WHERE rm_id = %s", (rm_id,))
    return row or {"rm_id": rm_id}


@app.get("/health")
def health() -> JSONResponse:
    s = get_settings()
    out: dict = {"status": "ok"}
    try:
        row = fetch_one("SELECT count(*) AS chunks, min(embed_model) AS embed_model FROM policy_chunks")
        out["db"] = "ok"
        out.update(row)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"status": "down", "db": f"error: {e.__class__.__name__}"}, status_code=503)
    if s.llm_mock:
        out["llm"] = "mock"
    else:
        try:
            r = httpx.get(f"{s.llm_base_url.rstrip('/')}/models", timeout=3,
                          headers={"Authorization": f"Bearer {s.llm_api_key}"})
            models = {m["id"] for m in r.json().get("data", [])}
            out["llm"] = "ok" if any(m.split(":")[0] == s.llm_model.split(":")[0] for m in models) \
                else f"model {s.llm_model} not found"
        except Exception as e:  # noqa: BLE001
            out["llm"] = f"unreachable: {e.__class__.__name__}"
    if out["llm"] not in ("ok", "mock") or not out.get("chunks"):
        out["status"] = "degraded"
    return JSONResponse(out)
