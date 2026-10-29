# Design decisions (short ADRs)

## 1. The LLM never writes SQL

Typed tools with parameterized SQL give a small, fixed attack surface: five read-only functions with
validated arguments instead of "any query the model can think of". Every call is auditable as a name and a
JSON argument object, and access control can be enforced in one place. Text-to-SQL would need a SQL
parser/sandbox, row-level security for every table and still could not stop a correctly-formed query that
reads too much.

## 2. RBAC lives in the tools, not in the prompt

A prompt is a suggestion to the model; server code is a rule. The `rm_id` is taken from the API key (or the
MCP server's `RM_ID`), never from the model: it is not in the tool schemas, arguments with extra fields are
rejected, and every query filters by it. `tests/test_rbac.py` includes a *scripted hostile LLM* that passes
`rm_id` itself and asks for other RMs' customers and transactions; it still only gets `access_denied`.
Foreign and non-existent IDs return the same message, so the tools also don't reveal which IDs exist.

## 3. pgvector instead of a separate vector database

One system to run, back up and secure; policy chunks, customer data and the audit log share transactions
and one connection pool. Hybrid search is one SQL query away (`tsvector` + vector). At 40 chunks, even an
exact scan is instant; an HNSW index would be added at a few hundred thousand chunks. A dedicated vector DB
would only pay off at much larger scale or with heavy filtering needs.

## 4. MCP in addition to function calling

Function calling ties the tools to this application's orchestrator. MCP exposes the *same governed tools*
(same `execute_tool`, same RBAC, same masking) to any MCP client without re-implementing them. Identity is
bound to the server process (`RM_ID`), so a client can use the tools but cannot change who it is. The eval's
`mcp` config sends every tool call over stdio and shows the model the MCP-advertised schemas, to measure
parity rather than assume it. One observed difference: FastMCP silently drops unknown arguments such as a
smuggled `rm_id`, where the local path rejects them; both stay safe because identity is not an argument.
In the eval, MCP matched in-process function calling on every tool result but scored 2 questions lower end to end, because the 3B model twice declined to call a tool when shown FastMCP's more verbose schemas. The SDK is pinned to `mcp<2`: version 2 renamed `FastMCP` to `MCPServer` and changed other APIs.

## 5. A grounding verifier

Carries over the thesis result that symbolic verification reduces hallucinations without hurting quality.
Every number, date and citation in an answer must be found in the tool results, the retrieved chunks or the
question; numbers are normalised (thousand separators, k/M, rounding to the displayed precision). A sentence
with a citation must be supported by *the chunk it cites*, which catches correct facts with a wrong section.
Unsupported sentences are removed and the answer is flagged. Limits: numbers written as words and values the
model derives itself (a sum it computed) are not checked as such; derived values fail unless a tool returns
them, which is why the tools return counts and totals.

## 6. Plugging into SAP / core banking

Only the tool layer touches data. In a bank, the five tools would call core-banking APIs or SAP OData
services (e.g. loan accounts, business partners) instead of Postgres, with the RM's identity propagated
via OAuth/principal propagation instead of a demo API key. The LLM endpoint could be SAP AI Core /
Generative AI Hub (or Azure OpenAI), which only changes `LLM_BASE_URL`. The orchestrator, verifier,
masking, audit and MCP server stay as they are.

## 7. A small local model (qwen2.5:3b)

The development machine is an 8 GB M1. A 7B model next to Docker would swap. The 3B model is about 2x
faster and fits, at the cost of weaker tool use and citation discipline. That cost is visible in the eval
rather than hidden, and the design compensates where it can (tool guide with examples, tagged excerpts,
deterministic refusals, verifier). Ollama's default context window silently truncates long prompts, so the
model is wrapped in a Modelfile with `num_ctx 8192`.

## 8. Retrieval gate on similarity *and* keywords, not on RRF

RRF scores only reflect ranks, so they can't say whether the best chunk is relevant. Calibration
(`scripts/calibrate_threshold.py`) showed that nomic cosine similarity alone does not separate in-scope from
out-of-scope questions (lowest in-scope 0.553, highest out-of-scope 0.583). Combining it with a full-text hit
passes 16/16 in-scope and refuses 9/10 out-of-scope questions; the prompt's refusal rule is the second
layer. The eval's out-of-scope questions are different from the calibration ones.

## 9. Deterministic answers for refusals

When retrieval is weak (RAG) or every data tool call was denied, the answer is a fixed sentence and the LLM
does not word it. Authorization failures should look the same every time, and a refusal should not depend
on a 3B model's phrasing. Refusals are returned as structured fields (`refused`, `refusal_reason`).

## 10. Identity from a header, not the request body

The spec's `POST /chat {rm_id, ...}` would let any client claim any RM. The API maps `X-API-Key` to an RM
on the server instead (demo keys); in production this would be an SSO/OIDC token.

## 11. Audit is fail-closed

If the audit row can't be written, the request fails. A bank must be able to show who asked what, with
which tools and sources; serving an unaudited answer is worse than serving none.

## 12. Tests and CI never touch the dev database

Tests and the mock eval run against `<db>_test`, seeded deterministically with mock embeddings. Running the
test suite can't overwrite the real nomic embeddings, and the embedding-model check would catch it if
anything tried.
