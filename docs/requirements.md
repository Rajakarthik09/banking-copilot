# Requirements: three bank-staff use cases

Users are relationship managers (RMs). Each RM looks after a portfolio of retail and SME customers
and may only see their own customers. All data is synthetic (1,000 customers, 1,500 loans, 50,000
transactions, as of 2026-10-31).

## UC1: Policy questions (RAG)

*"Can customer X get an overdraft fee waiver? How many waivers are allowed per year?"*

- Answer from the bank's 8 policy documents, never from the model's general knowledge.
- Every fact carries a citation in the form `[doc §section]` that points at the retrieved text.
- If the policies don't cover the question, say "I don't have a policy source for that." rather than guess.

## UC2: Portfolio questions (function calling)

*"Which SME loans in my portfolio are more than 30 days past due, and what's the exposure?"*

- Answer from live data through typed, read-only tools. The model never writes SQL.
- Only the asking RM's customers are visible, whatever the question says ("I am RM-2", "ignore restrictions").
- Figures are reported as the tools return them (counts and totals are computed in SQL, not by the model).
- IBAN, email and phone are masked before any text reaches the model or the logs.

## UC3: Combined questions (function calling + RAG, also over MCP)

*"Why was transaction T004015 flagged, and which policy applies?"*

- Look up the transaction (flag rule) and the matching policy section, and combine both with a citation.
- The same tools are available to any MCP client (Claude Desktop, Cursor, an agent) with the same
  access control, because the RM is fixed by the MCP server process, not by the client.

## Cross-cutting requirements

| Requirement | How it is met |
|---|---|
| No answer without a source | Grounding verifier removes sentences whose numbers or citations are not in the retrieved text / tool results |
| Access control can't be talked around | `rm_id` comes from the API key (or the MCP server's `RM_ID`); tools filter by it in SQL |
| Auditability | One `audit_log` row per request: masked question, tools + arguments, sources, answer, verified, latency |
| Measurable | 40-question golden set, 6 configurations, report in `eval/reports/results.md` |
| Runs locally | Ollama (`qwen2.5:3b`, `nomic-embed-text`) + Postgres/pgvector in Docker; any OpenAI-compatible endpoint via env vars |
