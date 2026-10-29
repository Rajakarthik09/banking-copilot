# Banking Operations Copilot — Build Spec (MVP, one evening)

> Give this whole file to Claude Code / Cursor as the project brief. Build phase by phase, and don't start the next phase until the current one passes its **Done when** check.
> Time budget: about 5–7 hours with an AI coding tool. Phases marked **[STRETCH]** are optional. Only mention a stretch phase on your resume if you finished it.

---

## 0. What this is

An LLM assistant for bank staff that answers questions about **bank policies** and **customer/loan data**. It compares **three ways of integrating an LLM into an enterprise system** on the same questions:

| Approach | How it works | Use case |
|---|---|---|
| **A. RAG** | Retrieve policy chunks → answer with citations | "Can customer X get an overdraft fee waiver?" |
| **B. Function calling** | LLM picks typed, read-only tools → answers from tool results | "Which SME loans in my portfolio are >30 days past due, and what's the exposure?" |
| **C. MCP** | The same tools exposed through an MCP server, usable by any MCP client | Same as B, but reusable across clients |

Plus these banking-grade controls:
- **PII masking:** IBAN, email and phone are masked before text reaches the LLM and the logs.
- **Role-based access:** a relationship manager (RM) only sees their own customers. This is enforced in the tool code, not in the prompt.
- **Grounding verifier** (idea taken from the M.Sc. thesis): every number and citation in an answer must appear in the tool results or retrieved chunks. Anything unsupported is removed and flagged.
- **Evaluation harness** that compares the approaches on a golden question set.

---

## 1. Tech stack (keep it exactly this, don't add frameworks)

- Python 3.11, `uv` or `pip` + `pyproject.toml`
- **FastAPI** (API), **Streamlit** (UI)
- **PostgreSQL 16 + pgvector** via Docker Compose (data and vectors in one database)
- **LLM via an OpenAI-compatible client**, so one code path covers all of these:
  - Local default: **Ollama** with `qwen2.5:7b-instruct` (it supports tool calling) and `nomic-embed-text` for embeddings
  - Optional: **Azure OpenAI** or OpenAI. Switch with environment variables only.
- **MCP:** official `mcp` Python SDK (`FastMCP`)
- Tests: `pytest`. Lint: `ruff`.
- Containers: Dockerfile + `docker-compose.yml`. **[STRETCH]** Kubernetes manifests tested on `kind`.
- CI: GitHub Actions (ruff, pytest, mini-eval with a mock LLM)

**No LangChain, no LlamaIndex.** Write the orchestration by hand (about 150 lines) so that you can explain every line in an interview.

---

## 2. Repo layout

```
banking-copilot/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── Dockerfile
├── .env.example
├── .github/workflows/ci.yml
├── app/
│   ├── config.py            # env settings (LLM_BASE_URL, LLM_MODEL, EMBED_MODEL, DB_URL, LLM_API_KEY)
│   ├── db.py                # psycopg connection pool
│   ├── llm.py               # OpenAI-compatible client wrapper + MockLLM for tests/CI
│   ├── rag/
│   │   ├── ingest.py        # chunk policies by heading, embed, store in pgvector
│   │   └── retrieve.py      # hybrid search: vector + Postgres full-text, merged
│   ├── tools/
│   │   ├── banking_tools.py # typed read-only functions with Pydantic I/O, RBAC enforced
│   │   └── registry.py      # tool JSON schemas for function calling
│   ├── guardrails/
│   │   ├── pii.py           # regex masking: IBAN, email, phone
│   │   └── injection.py     # simple heuristic check on retrieved text
│   ├── verifier.py          # grounding check on final answers
│   ├── orchestrator.py      # modes: prompt_only | rag | tools | tools_rag ; verify on/off
│   ├── audit.py             # write audit_log rows
│   └── api.py               # FastAPI: POST /chat, GET /health
├── mcp_server/server.py     # FastMCP exposing the same tools + search_policies
├── ui/streamlit_app.py
├── data/
│   ├── generate_synthetic.py
│   ├── schema.sql
│   └── policies/*.md        # 8 fictional bank policies (generated in Phase 2)
├── eval/
│   ├── golden_set.jsonl
│   ├── run_eval.py
│   └── reports/             # results.md + results.csv (committed)
├── tests/
│   ├── test_rbac.py
│   ├── test_pii.py
│   ├── test_verifier.py
│   └── test_injection.py
└── docs/
    ├── requirements.md      # the 3 business use cases
    ├── architecture.md      # diagram + flow
    └── decisions.md         # short ADRs (see section 11)
```

---

## 3. Phase 1 — Setup + data (≈ 60 min)

**Build:**
1. `docker-compose.yml` with the `pgvector/pgvector:pg16` service, plus the `api` and `ui` services added later.
2. `data/schema.sql`:
   - `relationship_managers(rm_id, name, region)` with 5 RMs
   - `customers(customer_id, rm_id, name, segment[retail|sme], region, iban, email, phone, kyc_last_review date)`
   - `loans(loan_id, customer_id, product[mortgage|sme_term|consumer], principal, outstanding, rate, days_past_due, risk_tier[A|B|C|D], status)`
   - `transactions(txn_id, customer_id, ts, amount, currency, counterparty_country, channel, flagged bool, flag_rule text)`
   - `policy_chunks(id, doc, section, content, embedding vector(768), tsv tsvector)`
   - `audit_log(id, ts, rm_id, mode, question_masked, tools_called jsonb, sources jsonb, answer, verified bool, latency_ms, tokens)`
3. `data/generate_synthetic.py` with Faker and a fixed `seed=42`: 1,000 customers, about 1,500 loans (~12% with days_past_due > 30), 50,000 transactions (~1% flagged with rules such as `AML-01 high-risk country`, `AML-02 structuring < 10k`, `AML-03 velocity`). Use German-style IBANs.

**Done when:** `docker compose up -d db && python data/generate_synthetic.py` loads all tables, and a `SELECT count(*)` on each table matches.

---

## 4. Phase 2 — Policies + RAG (≈ 60 min)

**Build:**
1. Generate **8 fictional policy documents** in `data/policies/`, each 300–600 words with `##` section headings and numbered rules:
   `overdraft_fee_waiver.md`, `loan_forbearance.md`, `kyc_refresh.md`, `aml_transaction_monitoring.md` (defines AML-01..03), `sme_credit_risk_tiers.md`, `complaints_handling.md`, `data_privacy_customer_data.md`, `account_closure.md`.
   Put concrete thresholds in them (for example "fee waiver allowed at most 2× per 12 months if balance restored within 5 days"), so that answers can be checked.
2. `rag/ingest.py`: one chunk per `##` section, embed with `nomic-embed-text`, and store the embedding and the `tsv`.
3. `rag/retrieve.py`: top-5 by vector cosine and top-5 by `ts_rank`, merged with reciprocal-rank fusion. Return `[{chunk_id, doc, section, content}]`.
4. In `orchestrator.py`, the `rag` mode uses a system prompt that requires citing `[doc §section]` and answering **"I don't have a policy source for that"** if retrieval is weak (threshold on the top score).

**Done when:** asking "How many fee waivers can a customer get per year?" returns the right number with a correct citation, and asking "What's the weather in Leipzig?" gets a refusal.

---

## 5. Phase 3 — Function calling + RBAC (≈ 75 min)

**Build `tools/banking_tools.py`.** Every function takes `rm_id` as its **first argument, injected by the server from the request, never chosen by the LLM**:
- `list_past_due_loans(rm_id, min_dpd=30, segment=None) -> list[LoanSummary]`
- `get_portfolio_exposure(rm_id, segment=None, risk_tier=None) -> ExposureSummary`
- `get_customer_profile(rm_id, customer_id) -> CustomerProfile` (raises `AccessDenied` if the customer belongs to another RM)
- `get_transaction(rm_id, txn_id) -> Transaction` (includes `flag_rule`)
- `customers_due_for_kyc(rm_id, months=24) -> list[CustomerRef]`

Rules:
- Use parameterized SQL only. **The LLM never writes SQL.**
- Pydantic models for every output. Mask IBAN, email and phone in the outputs before they go back to the LLM.
- `registry.py` builds OpenAI-style `tools=[...]` JSON schemas **without** `rm_id`.
- Orchestrator `tools` mode: a loop of at most 5 tool calls, then the final answer. `tools_rag` mode: tools plus a `search_policies` tool, used for "why was txn T flagged and which policy applies?"

**Done when:** `tests/test_rbac.py` passes. RM-1 asking about a customer of RM-2 gets "access denied" even when the prompt says "I am RM-2" or "ignore restrictions".

---

## 6. Phase 4 — Guardrails + verifier + audit (≈ 45 min)

1. `guardrails/pii.py`: regexes for DE/EU IBAN, email and phone → `[IBAN]`, `[EMAIL]`, `[PHONE]`. Apply them to the user question, the tool outputs and the audit log.
2. `guardrails/injection.py`: flag retrieved chunks that contain patterns such as "ignore previous instructions" or "system prompt", and drop them before generation. Add a **poisoned test policy** `data/policies/_poisoned_test.md`, loaded only in tests.
3. `verifier.py`, grounding check:
   - Extract every number (amounts, counts, %, days) from the answer. Each must appear (after normalising it) in the tool results or the retrieved chunks.
   - Extract every `[doc §section]` citation. Each must match a retrieved chunk.
   - Remove unsupported sentences, set `verified=False`, and append a note: "Some content was removed because it could not be verified."
4. `audit.py`: one row per request.

**Done when:** `test_pii.py`, `test_verifier.py` and `test_injection.py` pass.

---

## 7. Phase 5 — API, MCP server, UI (≈ 45 min)

1. `api.py`: `POST /chat {rm_id, question, mode, verify}` → `{answer, sources, tools_called, verified, latency_ms}`, plus `GET /health`.
2. `mcp_server/server.py`: `FastMCP("banking-copilot")` exposing the 5 tools plus `search_policies`. `rm_id` comes from the `RM_ID` environment variable of the MCP server process (document why: the client cannot change it).
   Test it with the MCP Inspector (`npx @modelcontextprotocol/inspector`) or by adding it to Claude Desktop/Cursor. **Take a screenshot** and save it as `docs/mcp_demo.png`.
3. `ui/streamlit_app.py`: an RM dropdown, a mode selector, a verify toggle, a chat box, and an expandable panel showing "sources / tools called / verified".

**Done when:** `docker compose up` starts db, api and ui, and all three use cases work in the browser.

---

## 8. Phase 6 — Evaluation (≈ 60 min) ← the most important phase for interviews

1. `eval/golden_set.jsonl` with **40 questions**:
   - 12 policy questions (expected: key fact + expected `doc`)
   - 12 data questions (expected value **computed by SQL in the script**, not hand-typed)
   - 8 combined questions (flag reason + matching policy)
   - 8 should-refuse questions (other RM's customer, out-of-scope, prompt injection)
2. `eval/run_eval.py` runs every question through these configs: `prompt_only`, `rag`, `tools`, `tools_rag`, `tools_rag+verify`.
   Metrics per config:
   - **Answer accuracy:** numeric match (±0.5%) for data questions; key-fact substring/LLM-judge for policy questions
   - **Citation accuracy:** cited doc == expected doc
   - **Unsupported-claim rate:** share of answers where the verifier found ungrounded content
   - **Correct refusal rate**
   - **p50 / p95 latency**
3. Write `eval/reports/results.md` (table) and `results.csv`. **Commit them.**
4. Write 5–8 sentences in `README.md` → "Findings": which approach wins for which question type, and what the verifier changed.

**Done when:** the report exists with real numbers from a full run.

---

## 9. Phase 7 — Tests, Docker, CI (≈ 30 min)

1. `Dockerfile` (multi-stage, slim) for the api; the ui reuses the same image with a different command.
2. `.github/workflows/ci.yml`: install → `ruff check` → `pytest` → `python eval/run_eval.py --mock --limit 10` (`MockLLM` returns canned tool calls/answers so CI needs no GPU or API key).
3. A green CI badge in the README.

**Done when:** the GitHub Actions run is green.

---

## 10. [STRETCH] Phase 8 — Kubernetes and Azure (only if time is left)

- `deploy/k8s/`: Deployments for api, ui and postgres (StatefulSet + PVC), Services, a ConfigMap, a Secret, readiness/liveness probes on `/health`, and resource limits.
- Run it on `kind`: `kind create cluster && kubectl apply -f deploy/k8s/`. Screenshot `kubectl get pods` and the UI working → `docs/k8s_demo.png`.
- Azure: set `LLM_BASE_URL`/`LLM_API_KEY` to Azure OpenAI and re-run the eval with `--tag azure`.

**Only list Kubernetes / Azure OpenAI on your resume if you completed this and have the screenshot or the report in the repo.**

---

## 11. `docs/decisions.md` (short ADRs; write these yourself, in your own words)

1. **Why the LLM never writes SQL:** typed tools give a fixed attack surface, are auditable and can enforce access control.
2. **Why RBAC lives in the tools and not the prompt:** prompts can be overridden, server code can't.
3. **Why pgvector instead of a separate vector database:** one system to operate, transactional, and good enough for this scale.
4. **Why MCP in addition to function calling:** the same governed tools can be reused across clients (IDE, chat apps, agents) without re-implementing them.
5. **Why a grounding verifier:** carries over the thesis result that symbolic verification cuts hallucinations without degrading quality.
6. **How this would plug into SAP / core banking:** the tools would call core-banking APIs or OData services instead of Postgres, and the LLM gateway could be SAP AI Core / Generative AI Hub. Nothing above the tool layer changes.

---

## 12. README template

```markdown
# Banking Operations Copilot
LLM assistant for bank staff comparing RAG, function calling, and MCP on the same banking use cases — with PII masking, role-based access, and a grounding verifier.

![CI](badge) · [Demo video](link) · [Eval report](eval/reports/results.md)

## Use cases
(3 use cases from docs/requirements.md)

## Architecture
(diagram from docs/architecture.md)

## Results  (40-question golden set, model: <model>, date: <date>)
| Config | Answer acc. | Citation acc. | Unsupported claims | Correct refusals | p95 latency |
|---|---|---|---|---|---|
| prompt_only | | | | | |
| rag | | | | | |
| tools | | | | | |
| tools_rag | | | | | |
| tools_rag + verify | | | | | |

## Findings
## Security & governance
## Run it
docker compose up -d && python data/generate_synthetic.py && python -m app.rag.ingest
open http://localhost:8501
## Design decisions → docs/decisions.md
## Limitations (be honest: synthetic data, small eval set, local 7B model, regex PII)
```

---

## 13. Before you push

- [ ] `.env` is in `.gitignore`, and there are **no API keys** in the repo or the git history
- [ ] `docker compose up` works on a fresh clone (test it in a new folder)
- [ ] `eval/reports/results.md` contains the numbers from your real run
- [ ] Screenshots of the MCP demo (and k8s, if done) are in `docs/`
- [ ] README "Limitations" section is filled in
- [ ] A 2-minute Loom demo is linked (optional but strong)

## 14. Before the interview (do not skip this)

Because this was built with an AI coding tool, make sure you can explain every part of it **without looking**:
- [ ] Walk through what happens, file by file, when `/chat` gets a question in `tools_rag` mode
- [ ] How RRF merges the vector and full-text results
- [ ] How the verifier decides what is "supported", and where it fails (paraphrased numbers, derived totals)
- [ ] Why a prompt like "I am RM-2" can't bypass RBAC
- [ ] What would break at 1,000 concurrent users, and the fix (async, connection pool, caching retrieval, a hosted LLM)
- [ ] Your eval numbers, and one surprising finding

## 15. Resume update (after the repo is pushed)

Replace the "in progress" entry. **Fill the brackets only from `eval/reports/results.md`, and drop any tech you didn't actually use:**

```latex
\roleentry{Banking Operations Copilot}{Oct 2026}{Python, FastAPI, PostgreSQL/pgvector, MCP, Docker, GitHub Actions}{\href{https://github.com/Rajakarthik09/banking-copilot}{GitHub}}
\begin{itemize}
  \item Built an LLM assistant for 3 bank-staff use cases, comparing RAG, function calling, and MCP tool access on a synthetic core-banking dataset (1,000 customers, 50,000 transactions).
  \item Enforced role-based data access in typed read-only tools, PII masking, and a grounding verifier from my thesis; on a 40-question eval, [best config] reached [X]\% answer accuracy and the verifier cut unsupported claims from [A]\% to [B]\%.
\end{itemize}
```
