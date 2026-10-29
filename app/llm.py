"""LLM + embedding clients.

One OpenAI-compatible code path covers Ollama, OpenAI and Azure OpenAI (switch with env vars).
MockLLM / MockEmbedder make tests and CI run without any model.
"""

import hashlib
import json
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

from openai import OpenAI

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, validated later by the tool layer


@dataclass
class ChatResult:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens: int = 0


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             tool_choice: str | None = None) -> ChatResult: ...


class Embedder(Protocol):
    name: str  # stored with each chunk so query and document vectors always come from the same model

    def embed(self, texts: list[str], kind: str) -> list[list[float]]: ...


# Small models sometimes write the tool call into the message text instead of using the
# structured tool_calls field, e.g. <tool_call>{"name": ..., "arguments": {...}}</tool_call>.
_TOOL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_text_tool_calls(content: str, tool_names: set[str]) -> list[ToolCall]:
    """Recover tool calls that were emitted as text. Only names of real tools are accepted."""
    candidates = _TOOL_TAG.findall(content or "")
    if not candidates:
        stripped = (content or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        if stripped.strip().startswith("{"):
            candidates = [stripped.strip()]
    calls = []
    for raw in candidates:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("name") not in tool_names:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append(ToolCall(f"call_{uuid.uuid4().hex[:8]}", obj["name"],
                              args if isinstance(args, str) else json.dumps(args)))
    return calls


class OpenAICompatLLM:
    def __init__(self) -> None:
        s = get_settings()
        self.model = s.llm_model
        self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout_s)

    def chat(self, messages, tools=None, tool_choice=None) -> ChatResult:
        kwargs = {"model": self.model, "messages": messages, "temperature": 0, "seed": 42}
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}") for tc in msg.tool_calls or []]
        content = msg.content or ""
        if tools and not calls and tool_choice != "none":
            calls = parse_text_tool_calls(content, {t["function"]["name"] for t in tools})
            if calls:
                log.info("recovered %d tool call(s) from message text", len(calls))
                content = ""
        tokens = resp.usage.total_tokens if resp.usage else 0
        return ChatResult(content=content, tool_calls=calls, tokens=tokens)


class OpenAICompatEmbedder:
    def __init__(self) -> None:
        s = get_settings()
        self.model, self.dim, self.prefixes = s.embed_model, s.embed_dim, s.embed_task_prefixes
        self.name = f"{self.model}/{self.dim}"
        self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout_s)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        """kind is "query" or "document" (nomic-embed-text needs a task prefix for each)."""
        if self.prefixes:
            texts = [f"search_{kind}: {t}" for t in texts]
        kwargs = {"model": self.model, "input": texts}
        if not self.model.startswith("nomic"):
            kwargs["dimensions"] = self.dim  # e.g. text-embedding-3-small -> 768 to fit the schema
        resp = self.client.embeddings.create(**kwargs)
        vectors = [d.embedding for d in resp.data]
        if any(len(v) != self.dim for v in vectors):
            raise ValueError(f"embedding size {len(vectors[0])} != EMBED_DIM {self.dim}")
        return vectors


_WORD = re.compile(r"[a-z0-9]+")


class MockEmbedder:
    """Deterministic hashed bag-of-words vectors: cosine similarity tracks word overlap."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.name = f"mock-hash/{dim}"

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in _WORD.findall(text.lower()):
                if len(word) < 3:
                    continue
                h = int(hashlib.md5(word[:6].encode()).hexdigest(), 16)  # crude stemming
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


@lru_cache
def get_llm() -> LLM:
    if get_settings().llm_mock:
        from app.mock_llm import MockLLM

        return MockLLM()
    return OpenAICompatLLM()


@lru_cache
def get_embedder() -> Embedder:
    if get_settings().llm_mock:
        return MockEmbedder(get_settings().embed_dim)
    return OpenAICompatEmbedder()
