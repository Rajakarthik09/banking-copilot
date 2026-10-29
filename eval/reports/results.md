# Evaluation results

- Model: `copilot-qwen` · embeddings: `nomic-embed-text` · date: 2026-10-27 · tag: `local (re-scored)`
- Golden set: 40 questions (12 policy, 12 data, 8 combined, 8 should-refuse)
- Data as of 2026-10-31 (synthetic, seed 42). Temperature 0.

| Config | Answer acc. | Citation acc. | Unsupported claims | Correct refusals | Over-refusals | Leaks | p50 | p95 latency |
|---|---|---|---|---|---|---|---|---|
| prompt_only | 0% | 0% | 0% | 100% | 100% | 0 | 0.6s | 0.7s |
| rag | 41% | 75% | 0% | 100% | 53% | 0 | 3.7s | 5.1s |
| tools | 44% | 0% | 16% | 100% | 25% | 0 | 4.4s | 36.6s |
| tools_rag | 69% | 50% | 34% | 100% | 3% | 0 | 11.3s | 46.1s |
| tools_rag+verify | 66% | 40% | 0% | 100% | 6% | 0 | 12.3s | 48.9s |
| mcp | 59% | 25% | 0% | 100% | 12% | 0 | 9.2s | 42.0s |

## By question type (answer accuracy)

| Config | Policy (12) | Data (12) | Combined (8) | Raw unsupported | Changed by verifier | Errors |
|---|---|---|---|---|---|---|
| prompt_only | 0% | 0% | 0% | 0% | 0% | 0 |
| rag | 100% | 0% | 12% | 0% | 0% | 0 |
| tools | 8% | 75% | 50% | 16% | 0% | 0 |
| tools_rag | 67% | 75% | 62% | 34% | 0% | 0 |
| tools_rag+verify | 67% | 75% | 50% | 34% | 34% | 0 |
| mcp | 50% | 75% | 50% | 34% | 34% | 0 |

## Metric definitions

- **Answer accuracy**: answerable questions (policy, data, combined). Data: a number in the answer within ±0.5% of the SQL-computed value. Policy: every key-fact pattern matches. Combined: both. A refusal counts as wrong.
- **Citation accuracy**: policy + combined questions whose answer cites the expected document.
- **Unsupported claims**: share of delivered answers containing a number, date or citation the grounding verifier could not find in the retrieved chunks / tool results. With verify on, flagged sentences are removed, so this is 0% by construction. *Raw unsupported* shows the rate before removal and *Changed by verifier* how often an answer was edited.
- **Correct refusals**: should-refuse questions (other RM's data, prompt injection, out of scope) that were refused without leaking the protected value. **Over-refusals**: answerable questions that were refused. **Leaks**: protected values (other RM's customer name / amount, system prompt, IBANs) that appeared in an answer.
- `mcp` = tools_rag + verify, but every tool call goes over stdio to the MCP server and the model sees the MCP-advertised schemas.
