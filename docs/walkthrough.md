# Walkthrough: one request, file by file

Question from RM-2 in the UI, mode `tools_rag`, verifier on:
*"Why was transaction T004015 flagged and which policy applies?"*

1. **`ui/streamlit_app.py`** posts `{"question", "mode": "tools_rag", "verify": true}` to `/chat` with RM-2's
   `X-API-Key`.
2. **`app/api.py`** validates the body with `ChatRequest` (`extra="forbid"`: an `rm_id` field would be a 422).
   `Depends(current_rm)` runs **`app/auth.py`**, which compares the key with `hmac.compare_digest` and
   returns `"RM-2"`. The endpoint is a sync `def`, so FastAPI runs it in a worker thread.
3. **`app/orchestrator.py` `answer_question`** masks the question (**`app/guardrails/pii.py`**), then
   `_run_tool_loop` builds the messages: system prompt = base rules + tool guide + tools_rag guide.
   Tool schemas come from **`app/tools/registry.py`** `tool_schemas(include_policies=True)`, generated
   from the Pydantic argument models; none has `rm_id`.
4. **`app/llm.py`** sends the messages to Ollama's OpenAI-compatible endpoint (temperature 0, seed 42).
   The model answers with `tool_calls=[get_transaction {"txn_id": "T004015"}]`. (If it had written the call
   as text, `parse_text_tool_calls` would recover it.)
5. **`registry.execute_tool`** parses the JSON, rejects an `rm_id` argument, validates with
   `TransactionArgs` (normalises `4015` → `T004015`) and calls
   **`app/tools/banking_tools.py` `get_transaction("RM-2", "T004015")`**: one parameterized query that joins
   `customers` and filters on `c.rm_id = 'RM-2'`. The result (`flag_rule: "AML-02 structuring < 10k"`) is
   PII-masked and appended as a `tool` message.
6. Second round: the model calls `search_policies {"query": "AML-02 structuring"}`. That runs
   **`app/rag/retrieve.py`**: embed the query (`search_query:` prefix), top-5 by cosine and top-5 by
   `ts_rank` (OR-query of stemmed lexemes), RRF merge, drop injection-looking chunks
   (**`app/guardrails/injection.py`**), apply the similarity+keyword gate, and return excerpts whose every
   line ends with its tag, e.g. `[aml_transaction_monitoring §2]`.
7. Third round: no tool calls, so the model's text is the raw answer. (After 5 tool calls the loop would stop
   and ask for a final answer without tools.)
8. **`app/verifier.py`** splits the answer into sentences, extracts numbers (with units and display precision),
   dates and citations, and checks each against the tool JSON, the cited chunk(s) and the question. Unsupported
   sentences are removed; `verified=False` if anything was removed.
9. **`app/audit.py`** writes one masked row (question, tools with arguments and outcome, sources, answer,
   verified, refused, latency, tokens). If that fails, the request fails.
10. The API returns `{answer, sources, tools_called, verified, refused, refusal_reason, removed, latency_ms,
    tokens}` and the UI shows the answer with an expander for tools, sources and removed sentences.

## Questions to be able to answer without looking

**How does RRF merge vector and full-text results?** Each list is ranked separately; a chunk's score is
the sum over lists of `1 / (60 + rank)`. A chunk found by both lists beats one that is first in only one.
It needs no score normalisation between cosine and `ts_rank`, which live on different scales. It also
means the fused score says nothing about absolute relevance, which is why the refusal gate uses cosine +
keyword evidence instead.

**How does the verifier decide what is supported, and where does it fail?** A number is supported if the
same value appears in the evidence, or an evidence value rounds to it at the answer's displayed precision
and unit (954,342.98 → "954,343" / "€0.95M"). A citation must match a retrieved chunk, and a cited sentence
must be backed by that chunk. It fails on: numbers written as words ("twice"); values the model computes
itself (sums, percentages), unless a tool already returns them; and small integers that happen to occur
somewhere in a large tool result (a false "supported").

**Why can't "I am RM-2" bypass RBAC?** The RM comes from the API key on the server. The model can't pass
an `rm_id` (not in the schema, extra arguments rejected), and every SQL query filters by the server-side
value. Even a fully hostile model only gets `access_denied` (`test_compromised_llm_still_cannot_escalate`).

**What breaks at 1,000 concurrent users, and the fix?**
- The LLM is the bottleneck: one local 3B model serves requests one by one → a hosted/batched inference
  endpoint (vLLM, Azure OpenAI, AI Core) with streaming responses.
- Sync endpoints hold a worker thread per request for 5-30 s → async endpoints with an async OpenAI client and
  `psycopg` async pool, or a job queue.
- Connection pool `max_size=10` → a larger pool behind PgBouncer.
- Every request re-embeds the question and re-retrieves → cache embeddings/retrievals for frequent questions.
- Audit inserts per request → batched writes or a log pipeline, still fail-closed.
- Demo API keys → SSO/OIDC with per-user rate limits.
